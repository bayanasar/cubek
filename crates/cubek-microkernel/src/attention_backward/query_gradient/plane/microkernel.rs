use super::base::PlaneQueryGradientMicrokernel;
use crate::attention::{D, G, Q, S};
use crate::attention_backward::contract_into_window;
use crate::matmul::dense::plane::plane_fragment_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use crate::staging::StageRing;
use cubecl::tile::{Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's queries of one head: `delta = Σ_dv out_gradient · out`, then
/// `query_gradient = scale · Σ_s probabilities ⊙ (out_gradient · valueᵀ − delta) · key` over the
/// first `steps` steps of the queries' walk through `S`, summed in `EA`, the probabilities read
/// back from each query's `logsumexp`.
///
/// The plane stages its query and the output's gradient once, as its own copies, and its keys and
/// values a stage at a time, holding as many stages as `ring` says. Its scores, its gradient
/// against the probabilities and that gradient cast to `ES` lie in windows of its own, met on
/// `sync_plane`. Every tile is windowed to `queries`; `planes` is how many planes along the
/// queries share the cube, which its own grids are placed under. The dot product is written
/// before the walk, for the keys' walk to read.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_query_gradient_microkernel<EA: Float, ES: Float>(
    queries: &Region,
    query_gradient: &Tile,
    delta: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
) {
    // A tile's clone is a handle to the same cells: the dot product lands in `delta`.
    let mut delta = delta.clone();
    delta.dot::<EA>(out_gradient, out);
    let walk = queries.walk().range(0, steps);
    let mut sum = query_gradient
        .accumulator::<EA>(
            bias,
            key,
            comptime!(microkernel.query_gradient.tile_instruction()),
            Semiring::SUM_PROD,
        )
        .with_scratch(comptime!(microkernel.settings.scratch));
    sum.zero();
    // The query and the output's gradient span no key: staged once, before the walk.
    let mut staged_query = query.stage_for(&walk, StageStorage::Strided, None);
    staged_query.copy_from(query);
    let mut staged_out_gradient = out_gradient.stage_for(&walk, StageStorage::Strided, None);
    staged_out_gradient.copy_from(out_gradient);
    let scores = Tile::scratch::<EA>(&walk, comptime!(vec![G, Q, S]), StageStorage::Strided);
    let gradients = Tile::scratch::<EA>(&walk, comptime!(vec![G, Q, S]), StageStorage::Strided);
    let mut cast = Tile::scratch::<ES>(&walk, comptime!(vec![G, Q, S]), StageStorage::Strided);
    // The query, the output's gradient and the dot products land before any is read.
    sync_plane();
    let mut stages = Stages::smem(
        &walk,
        key,
        value,
        StageStorage::Strided,
        comptime!(ring.stages_in_flight.get()),
        None,
    );
    let wide = ComptimeOption::<Tile>::new_None();
    stages.walk(walk, comptime!(ring.prefetch), |slot, stage| {
        let bias = bias.at(stage);
        slot.consume(|key, value| {
            contract_into_window::<EA>(
                &scores,
                &staged_query,
                key,
                comptime!(microkernel.score),
                comptime!(microkernel.score_levels(&[(Q, planes)])),
            );
            // The scores' cells were stored in the fragments' layout; the softmax reads them by
            // rows.
            sync_plane();
            scores.along(S).probabilities::<EA>(scale, &bias, logsumexp);
            contract_into_window::<EA>(
                &gradients,
                &staged_out_gradient,
                value,
                comptime!(microkernel.score_gradient),
                comptime!(microkernel.score_gradient_levels(&[(Q, planes)])),
            );
            // The probabilities and the gradient against them land before both are read.
            sync_plane();
            gradients
                .along(S)
                .softmax_gradient::<EA>(&scores.along(S), &delta);
            sync_plane();
            cast.copy_cast_from(&gradients);
            // The gradient lands before the fragments load it.
            sync_plane();
            // A tile's clone is a handle to the same cells: the shared microkernel reads the
            // gradient and adds into the sum through it.
            plane_fragment_microkernel(
                stage,
                &sum.clone(),
                &wide,
                &cast.clone(),
                key,
                comptime!(microkernel.query_gradient.tile_instruction()),
                comptime!(None),
                comptime!(MatmulOperands::all(MatmulRead::AsBound)),
            );
        });
    });
    let mut scales = Array::<EA>::new(comptime!(microkernel.seq_q()));
    #[unroll]
    for q in 0..comptime!(microkernel.seq_q()) {
        scales[q] = scale;
    }
    sum.along(D).mul(&scales);
    sum.drained_into(query_gradient);
}
