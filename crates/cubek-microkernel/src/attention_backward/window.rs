use crate::matmul::MatmulAxes;
use crate::matmul::dense::plane::PlaneFragmentMicrokernel;
use cubecl::tile::{Level, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// `window = lhs · rhs`, summed in `EA` on `product`'s own grid of fragments, placed as `levels`
/// state it below the walker's planes, then cast into the plane's `window` of shared memory: where
/// a plane keeps a score, or the gradient against its probabilities, between two contractions
/// ([`contract_on_grid`]). The caller meets the plane before the window is read.
#[cube]
pub(crate) fn contract_into_window<EA: Float>(
    window: &Tile,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] product: PlaneFragmentMicrokernel,
    #[comptime] levels: Vec<Level>,
) {
    // The accumulator, cut to the product's own grid below the planes' level.
    let placed = window.with_levels(1usize, levels);
    let mut sum = placed.accumulator::<EA>(
        lhs,
        rhs,
        comptime!(product.tile_instruction()),
        Semiring::SUM_PROD,
    );
    sum.zero();
    contract_on_grid(&sum, lhs, rhs, product);
    let MatmulAxes { m, n, .. } = comptime!(product.axes);
    let [(_, rows), (_, columns), _] = comptime!(product.instruction.shape());
    let fragment = comptime!(Level::every(&[(m, rows), (n, columns)]));
    let (sum_fragments, window_fragments) = (sum.over(&fragment), window.over(&fragment));
    // The accumulator's fragments and the window are one grid at two depths: the accumulator is
    // placed on the product's own grid, the window sits under the plane's region, and a region
    // only windows a tile at its depth.
    #[unroll]
    for at in 0..comptime!(product.settings.along_m.get() * product.settings.along_n.get()) {
        let mut cells = window.at(&window_fragments.region(at));
        cells.copy_cast_from(&sum.at(&sum_fragments.region(at)));
    }
}

/// `sum += lhs · rhs` on `product`'s own grid of fragments, `sum` an accumulator placed on it: each
/// fragment of the grid pairs the operands' fragments by position, the lhs's rows by steps of the
/// contraction and the rhs's columns by the same steps.
#[cube]
pub(crate) fn contract_on_grid(
    sum: &Tile,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] product: PlaneFragmentMicrokernel,
) {
    let [(_, m), (_, n), (_, k)] = comptime!(product.instruction.shape());
    let MatmulAxes {
        m: rows_axis,
        n: columns_axis,
        k: depth_axis,
    } = comptime!(product.axes);
    let (rows, columns, depth) = comptime!((
        product.settings.along_m.get(),
        product.settings.along_n.get(),
        product.settings.depth.get(),
    ));
    let lhs_fragments = lhs.over(&comptime!(Level::every(&[(rows_axis, m), (depth_axis, k)])));
    let rhs_fragments = rhs.over(&comptime!(Level::every(&[
        (columns_axis, n),
        (depth_axis, k)
    ])));
    let sum_fragments = sum.over(&comptime!(Level::every(&[
        (rows_axis, m),
        (columns_axis, n)
    ])));
    #[unroll]
    for step in 0..depth {
        #[unroll]
        for row in 0..rows {
            #[unroll]
            for column in 0..columns {
                let mut fragment = sum.at(&sum_fragments.region(row * columns + column));
                fragment.mma(
                    &lhs.at(&lhs_fragments.region(row * depth + step)),
                    &rhs.at(&rhs_fragments.region(column * depth + step)),
                );
            }
        }
    }
}
