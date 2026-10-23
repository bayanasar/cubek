use super::base::PlaneKeyGradientMicrokernel;
use crate::attention::{D, Q, S};
use crate::attention_backward::{contract_into_window, contract_on_grid};
use crate::matmul::dense::plane::plane_fragment_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use crate::staging::StageRing;
use cubecl::tile::{Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's keys of one key head: `value_gradient = Σ_{g, q} probabilities · out_gradient` and
/// `key_gradient = scale · Σ_{g, q} probabilities ⊙ (value · out_gradientᵀ − delta) · query`, the
/// scores held keys first, summed in `EA`, the probabilities read back from each query's
/// `logsumexp`. Each head of the group walks its steps of queries from `first_step` on: the
/// queries before it see none of the plane's keys.
///
/// The plane stages its keys and values once, as its own copies, and each head's queries and
/// output gradient a stage at a time, holding as many stages as `ring` says. Its
/// scores, its gradient against the probabilities and either cast to `ES` lie in windows of its
/// own, met on `sync_plane`. Every tile is windowed to `keys`; `planes` is how many planes along
/// the keys share the cube, which its own grids are placed under.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_key_gradient_microkernel<EA: Float, ES: Float>(
    keys: &Region,
    key_gradient: &Tile,
    value_gradient: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    delta: &Tile,
    bias: &Tile,
    first_step: usize,
    scale: EA,
    #[comptime] microkernel: PlaneKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] planes: usize,
) {
    let heads = keys.walk();
    // Every head has the first's shape: the stages and windows are placed once, by its walk.
    let first_head = heads.region(0usize);
    let shape = first_head.walk();
    // The windows span the keys and one head's queries, the axes of every product out of them.
    let scores = Tile::scratch::<EA>(&shape, comptime!(vec![S, Q]), StageStorage::Strided);
    let gradients = Tile::scratch::<EA>(&shape, comptime!(vec![S, Q]), StageStorage::Strided);
    let mut cast = Tile::scratch::<ES>(&shape, comptime!(vec![S, Q]), StageStorage::Strided);
    let mut key_sum = key_gradient
        .accumulator::<EA>(
            &scores,
            &query.at(&first_head),
            comptime!(microkernel.key_gradient.tile_instruction()),
            Semiring::SUM_PROD,
        )
        .with_scratch(comptime!(microkernel.settings.scratch));
    key_sum.zero();
    // The values' gradient, cut to its own grid below the planes' level.
    let value_window = value_gradient.with_levels(
        1usize,
        comptime!(microkernel.value_gradient_levels(&[(S, planes)])),
    );
    let mut value_sum = value_window
        .accumulator::<EA>(
            &scores,
            &out_gradient.at(&first_head),
            comptime!(microkernel.value_gradient.tile_instruction()),
            Semiring::SUM_PROD,
        )
        .with_scratch(comptime!(microkernel.settings.scratch));
    value_sum.zero();
    // The keys and the values span no query: staged once, before the walk.
    let mut staged_key = key.stage_for(&shape, StageStorage::Strided, None);
    staged_key.copy_from(key);
    let mut staged_value = value.stage_for(&shape, StageStorage::Strided, None);
    staged_value.copy_from(value);
    sync_plane();
    let wide = ComptimeOption::<Tile>::new_None();
    // A runtime loop over the heads: one ring of stages, filled from each head's queries in turn.
    for g in 0..heads.total() {
        let head = heads.region(g);
        let queries = head.walk();
        let steps = queries.total();
        let mut stages = Stages::smem(
            &queries,
            &query.at(&head),
            &out_gradient.at(&head),
            StageStorage::Strided,
            comptime!(ring.stages_in_flight.get()),
            None,
        );
        stages.walk(
            queries.range(first_step, steps),
            comptime!(ring.prefetch),
            |slot, stage| {
                let (bias, logsumexp, delta) =
                    (bias.at(stage), logsumexp.at(stage), delta.at(stage));
                slot.consume(|query, out_gradient| {
                    contract_into_window::<EA>(
                        &scores,
                        &staged_key,
                        query,
                        comptime!(microkernel.score),
                        comptime!(microkernel.score_levels(&[(S, planes)])),
                    );
                    // The scores' cells were stored in the fragments' layout; the softmax reads
                    // them along the queries.
                    sync_plane();
                    scores
                        .along(Q)
                        .probabilities::<EA>(scale, &bias, &logsumexp);
                    sync_plane();
                    cast.copy_cast_from(&scores);
                    // The probabilities land before the fragments load them.
                    sync_plane();
                    contract_on_grid(
                        &value_sum,
                        &cast,
                        out_gradient,
                        comptime!(microkernel.value_gradient),
                    );
                    contract_into_window::<EA>(
                        &gradients,
                        &staged_value,
                        out_gradient,
                        comptime!(microkernel.score_gradient),
                        comptime!(microkernel.score_gradient_levels(&[(S, planes)])),
                    );
                    // The gradient against the probabilities lands, and the cast probabilities
                    // are read, before either window is written again.
                    sync_plane();
                    gradients
                        .along(Q)
                        .softmax_gradient::<EA>(&scores.along(Q), &delta);
                    sync_plane();
                    cast.copy_cast_from(&gradients);
                    // The gradient lands before the fragments load it.
                    sync_plane();
                    // A tile's clone is a handle to the same cells: the shared microkernel reads
                    // the gradient and adds into the sum through it.
                    plane_fragment_microkernel(
                        stage,
                        &key_sum.clone(),
                        &wide,
                        &cast.clone(),
                        query,
                        comptime!(microkernel.key_gradient.tile_instruction()),
                        comptime!(None),
                        comptime!(MatmulOperands::all(MatmulRead::AsBound)),
                    );
                });
            },
        );
    }
    let mut scales = Array::<EA>::new(comptime!(microkernel.seq_kv()));
    #[unroll]
    for s in 0..comptime!(microkernel.seq_kv()) {
        scales[s] = scale;
    }
    key_sum.along(D).mul(&scales);
    key_sum.drained_into(key_gradient);
    value_sum.drained_into(&value_window);
}
