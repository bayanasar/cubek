//! The CpuGemm routine
//!
//! # Supported layouts
//!
//! Each operand carries its own physical layout (row/col-major rides in its strides, storage-tiling
//! depth in its `levels`), and the kernel reads it through a layout-agnostic view, so the three
//! operands may differ. The supported set (per operand, independently):
//!
//! - **Row-major** (`cols` contiguous): takes the vectorized N path (rhs *and* output must both
//!   be row-major to vectorize along N).
//! - **Col-major** (`rows` contiguous): a col-major rhs beside a row-major lhs takes the
//!   vectorized K path, both read in runs along `K`, each cell holding a run's partials until the
//!   store; any other col-major operand is correct, scalar.
//! - **Tiled** (nested contiguous blocks, single or recursive, rectangular): correct,
//!   scalar. Only reachable via the direct [`launch_ref`] (a tiled buffer isn't a plain
//!   strided binding); the [`Strategy`](crate::strategy::Strategy) entry deduces row/col.
//!
//! Lhs, rhs, and the accumulator each carry their own element type; the leaf reads every
//! operand in its native dtype and widens the inputs into the accumulate element, so a
//! mixed-precision GEMM (e.g. `f16` inputs, `f32` accumulate) runs through the same kernel.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - **Quantized inputs**: unsupported.
//! - **Non-contiguous strided bindings** on the [`Strategy`](crate::strategy::Strategy) path:
//!   a binding contiguous in neither matrix axis is not a plain row/col matrix and is
//!   rejected by the strided deduction
//! - **A plane grid larger than a cube**: a forced blueprint with more planes than the device
//!   launches in one cube.

use cubek_std::{MatrixLayout, selector::divisor_at_most};
use cubek_tile::space::CubeOrder;
use std::fmt::Display;

use cubecl::ir::HardwareProperties;

use crate::{
    definition::{MatmulProblem, MatmulSetupError},
    routine::{BlueprintStrategy, DeviceSettings, Routine},
};

/// L1 data-cache budget the blocking targets, in bytes. Conservative constant until
/// the runtime exposes per-core cache sizes.
const L1_BYTES: usize = 32 * 1024;

/// Byte budget for the leaf's `tile_m × tile_n` accumulator block, which must stay in vector
/// registers across the `K` loop; overflowing spills every accumulator to the stack (~2×).
/// ~24 of 32 NEON q-registers (~12 of 16 AVX2 ymm), leaving room for the A/B operands.
const ACC_REG_BYTES: usize = 384;

/// The widest strip a swizzled cube order is given.
const MAX_SWIZZLE_WIDTH: usize = 8;

/// The "default" strip width, used when the device does not report its last-level cache.
const UNKNOWN_CACHE_SWIZZLE_WIDTH: usize = 4;

/// The largest divisor of `k` not exceeding `cap` that is a whole number of `align`-wide vectors,
/// or [`divisor_at_most`] where `k` has none.
fn aligned_divisor_at_most(k: usize, cap: usize, align: usize) -> usize {
    let align = align.max(1);
    let cap = cap.clamp(1, k.max(1));
    let mut best = None;
    for d in (align..=cap).step_by(align) {
        if k.is_multiple_of(d) {
            best = Some(d);
        }
    }
    best.unwrap_or_else(|| divisor_at_most(k, cap))
}

/// The divisor of `g` nearest `target`, ties going to the larger
fn nearest_divisor(g: usize, target: usize) -> usize {
    let target = target.clamp(1, g.max(1));
    let mut best: usize = 1;
    for d in 1..=g {
        if g.is_multiple_of(d)
            && (d.abs_diff(target) < best.abs_diff(target)
                || (d.abs_diff(target) == best.abs_diff(target) && d > best))
        {
            best = d;
        }
    }
    best
}

/// The planes one cube holds: each is a plane of `plane_size_max` units.
fn max_planes(hardware: &HardwareProperties) -> usize {
    (hardware.max_units_per_cube / hardware.plane_size_max.max(1)).max(1) as usize
}

/// The aspect-ratio split of a `grid_m × grid_n` leaf grid over about `cores` planes, held to the
/// `max_planes` a cube launches with.
fn plane_grid(grid_m: usize, grid_n: usize, cores: usize, max_planes: usize) -> PlaneGrid {
    let target_m = (cores as f64 * grid_m as f64 / grid_n as f64)
        .sqrt()
        .round() as usize;
    let m = nearest_divisor(grid_m, target_m);
    let m = if m <= max_planes {
        m
    } else {
        divisor_at_most(grid_m, max_planes)
    };
    let n = nearest_divisor(grid_n, (cores / m).max(1));
    let n = if m * n <= max_planes {
        n
    } else {
        divisor_at_most(grid_n, max_planes / m)
    };
    PlaneGrid { m, n }
}

/// A split of a `grid_m × grid_n` leaf grid over about `cores` planes that puts every plane it can
/// along `m`, held to the `max_planes` a cube launches with. Planes side by side along `n` store
/// into the same rows, a few values each, so they share the output's cache lines.
fn rows_first_plane_grid(
    grid_m: usize,
    grid_n: usize,
    cores: usize,
    max_planes: usize,
) -> PlaneGrid {
    let planes = cores.min(max_planes).max(1);
    let m = divisor_at_most(grid_m, planes);
    let n = divisor_at_most(grid_n, (planes / m).max(1));
    PlaneGrid { m, n }
}

/// The order the cubes take the output's boxes in: strips of `width` boxes along `m`, each strip
/// walked through every box along `n` before the next, so the rows of A a strip reads are reused
/// from cache by every column block instead of streamed from memory once per column block.
///
/// Every plane of a cube reads its share of the strip at once, from the cache they share, so
/// `width` is the widest power of two up to [`MAX_SWIZZLE_WIDTH`] whose strip, `width ×
/// cube_m` rows of `k` values, fits half of `last_level_cache`; the other half is left to B
/// and C. A strip of one box is the unswizzled order. Where the cache is not reported, the
/// width is [`UNKNOWN_CACHE_SWIZZLE_WIDTH`].
fn cube_order(cube_m: usize, k: usize, elem: usize, last_level_cache: Option<usize>) -> CubeOrder {
    let Some(cache) = last_level_cache else {
        return CubeOrder::SwizzleRow(UNKNOWN_CACHE_SWIZZLE_WIDTH);
    };
    let box_rows = (cube_m * k * elem).max(1);
    let width = (0..=MAX_SWIZZLE_WIDTH.ilog2())
        .rev()
        .map(|p| 1usize << p)
        .find(|w| w * box_rows <= cache / 2)
        .unwrap_or(1);
    CubeOrder::SwizzleRow(width).canonicalize()
}

/// The `m × n × k` extent of the innermost instruction
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstructionShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// How many planes a cube's stage tile is divided into, along `m` and `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneGrid {
    pub m: usize,
    pub n: usize,
}

/// A fully-resolved CpuGemm plan: the leaf each plane computes ([`InstructionShape`]), how
/// finely a cube's stage tile is split across planes ([`PlaneGrid`]), and the order the cubes
/// take the output's boxes in.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CpuGemmBlueprint {
    pub instruction: InstructionShape,
    pub planes: PlaneGrid,
    /// The order the cubes take the output's boxes in. Every thread walks every cube in turn,
    /// `m` fastest, so unswizzled a column block reads all of A before the next one starts; a
    /// swizzle holds a strip of A in cache while every column block reads it.
    pub order: CubeOrder,
}

impl CpuGemmBlueprint {
    /// Reject a degenerate blueprint.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, _problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let (i, p) = (self.instruction, self.planes);
        if i.m == 0 || i.n == 0 || i.k == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm instruction must be non-zero, got {}x{}x{}",
                i.m, i.n, i.k
            ))));
        }
        if p.m == 0 || p.n == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm plane grid must be non-zero, got {}x{}",
                p.m, p.n
            ))));
        }
        // Any strip width serves any grid, but a strip holds a box: the kernel asserts it at
        // expansion, which is too late for a plan to be turned down cleanly.
        if matches!(
            self.order,
            CubeOrder::SwizzleRow(0) | CubeOrder::SwizzleCol(0)
        ) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm: a {:?} cube order distributes strips of no boxes",
                self.order
            ))));
        }
        Ok(())
    }
}

/// `alpha` sets the contraction depth `tile_k` (the leaf is fixed to the register-block
/// size), trading
/// - shallow K, lighter L1 footprint (→0)
/// - deeper K panels, more A/B reuse per accumulator load (→1).
#[derive(Clone, Debug)]
pub struct CpuGemmStrategy {
    pub alpha: f32,
}

impl Default for CpuGemmStrategy {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl Display for CpuGemmStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_a{}", self.alpha)
    }
}

/// Pairs the [`CpuGemmStrategy`] knob with the [`CpuGemmBlueprint`] plan.
pub struct CpuGemmRoutine;

impl cubek_std::legacy::Routine for CpuGemmRoutine {
    type Strategy = CpuGemmStrategy;
    type Blueprint = CpuGemmBlueprint;
}

impl Routine<()> for CpuGemmRoutine {}

impl CpuGemmRoutine {
    /// Resolve `strategy` into a validated cuboid for `problem` on this device.
    #[allow(clippy::result_large_err)]
    pub fn blueprint(
        strategy: &BlueprintStrategy<CpuGemmRoutine>,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
    ) -> Result<CpuGemmBlueprint, MatmulSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(strategy) => {
                Self::select(strategy, problem, device_settings)
            }
        };
        blueprint.validate(problem)?;
        let planes = blueprint.planes.m * blueprint.planes.n;
        let max_planes = max_planes(&device_settings.client.properties().hardware);
        if planes > max_planes {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm plane grid of {planes} planes exceeds the {max_planes} a cube holds"
            ))));
        }
        Ok(blueprint)
    }

    /// The tile-size heuristic. The leaf's accumulator block is sized to the register file
    /// ([`ACC_REG_BYTES`]), not L1; `alpha` sets `k` depth; `cores` becomes the [`PlaneGrid`].
    fn select(
        strategy: &CpuGemmStrategy,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
    ) -> CpuGemmBlueprint {
        let (m, n, k) = (problem.m, problem.n, problem.k);
        let elem = problem.global_dtypes.out.size().max(1);
        let vw = device_settings.vector_sizes.out.max(1); // SIMD width along N
        let hardware = &device_settings.client.properties().hardware;
        let cores = hardware
            .num_cpu_cores
            .map(|c| c as usize)
            .unwrap_or(4)
            .max(1);
        let alpha = strategy.alpha.clamp(0.0, 1.0);

        // Balance the register grid, not the element block: N is read in `vw`-wide lines, so
        // an element-square tile would collapse to one N-line (nr=1) for wide `vw`, starving
        // N-reuse. Aim `tile_n ≈ √(budget·vw)`, then fill the budget along M. Snap each tile to
        // a divisor of its axis: a ragged tile masks every leaf and de-unrolls the kernel (~2×),
        // so a smaller clean tile wins. A narrow axis collapses its tile to the whole axis.
        // `tile_n` is read in `vw`-wide lines, so it snaps to a divisor that is also a whole
        // number of lines where one exists — see `aligned_divisor_at_most`. A width that is not
        // a multiple of `align` leaves a partial vector on every row of every step of `K`.
        let budget_elems = (ACC_REG_BYTES / elem).max(1);
        // A col-major rhs is read in `vw`-wide runs along `K` (see `launch_ref`), and a cell keeps
        // a run's partials, so the block holds `budget / vw` cells. Each step loads a line of A a
        // row and of B a column for a multiply-add a cell: a square block reuses its loads most.
        let along_k = problem.rhs_layout == MatrixLayout::ColMajor;
        let (tile_m, tile_n) = match along_k {
            true => {
                let cells = (budget_elems / vw).max(1);
                let tile_n = divisor_at_most(n, ((cells as f64).sqrt() as usize).max(1));
                (divisor_at_most(m, (cells / tile_n).max(1)), tile_n)
            }
            false => {
                let tn_target = ((budget_elems * vw) as f64).sqrt() as usize;
                let tile_n = aligned_divisor_at_most(n, tn_target.max(1), vw);
                (divisor_at_most(m, (budget_elems / tile_n).max(1)), tile_n)
            }
        };

        // K depth: `alpha` lerps from a shallow `vw` to the deepest panel that keeps A
        // (tile_m×tile_k) and B (tile_k×tile_n) in L1 with the C tile (tile_m×tile_n)
        // resident, then snaps to a divisor of `k`. A ragged K tile bounds-checks every leaf
        // and disables the register unroll (~2×), so a clean shallower tile beats a deep
        // ragged one. Read along `K`, the tile is a whole number of runs where one divides `k`.
        let l1_tk = (L1_BYTES / elem).saturating_sub(tile_m * tile_n) / (tile_m + tile_n);
        let tk_cap = (vw + (alpha * l1_tk.saturating_sub(vw) as f32) as usize).clamp(1, k.max(1));
        let tile_k = match along_k {
            true => aligned_divisor_at_most(k, tk_cap, vw),
            false => divisor_at_most(k, tk_cap),
        };

        // Plane grid: split the leaf grid among ~`cores` worker threads by aspect ratio.
        // Each plane is a thread and the cube loop is *serial*, so the factors must divide
        // the grid: an indivisible split inflates the cube count (serial depth) and idles
        // planes on the overhang. Snap the aspect-ratio target to grid divisors.
        let grid_m = m.div_ceil(tile_m).max(1);
        let grid_n = n.div_ceil(tile_n).max(1);
        // Read along `K`, each plane stores a handful of single values a cell at a time, so the
        // split goes along `m` first (see `rows_first_plane_grid`).
        let planes = match along_k {
            true => rows_first_plane_grid(grid_m, grid_n, cores, max_planes(hardware)),
            false => plane_grid(grid_m, grid_n, cores, max_planes(hardware)),
        };

        // Tiles already divide their axes; the clamp is a defensive [1, axis] floor.
        let instruction = InstructionShape {
            m: tile_m.clamp(1, m.max(1)),
            n: tile_n.clamp(1, n.max(1)),
            k: tile_k.clamp(1, k.max(1)),
        };

        CpuGemmBlueprint {
            instruction,
            planes,
            order: cube_order(
                planes.m * instruction.m,
                k,
                elem,
                hardware.last_level_cache_size,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tall_matvec_grid_fits_the_cube() {
        let grid = plane_grid(64, 1, 16, 16);
        assert!(grid.m * grid.n <= 16, "{grid:?}");
    }

    #[test]
    fn a_square_grid_fits_the_cube_of_a_twelve_thread_host() {
        let grid = plane_grid(512, 32, 12, 12);
        assert!(grid.m * grid.n <= 12, "{grid:?}");
    }

    #[test]
    fn a_split_that_fits_is_the_aspect_ratio_split() {
        assert_eq!(plane_grid(512, 32, 16, 16), PlaneGrid { m: 16, n: 1 });
        assert_eq!(plane_grid(512, 32, 32, 32), PlaneGrid { m: 16, n: 2 });
    }

    #[test]
    fn a_rows_first_split_puts_its_planes_along_m() {
        assert_eq!(
            rows_first_plane_grid(768, 816, 16, 16),
            PlaneGrid { m: 16, n: 1 }
        );
        // Too few rows of tiles to hold every plane: the rest go along `n`.
        assert_eq!(
            rows_first_plane_grid(4, 816, 16, 16),
            PlaneGrid { m: 4, n: 4 }
        );
        assert_eq!(
            rows_first_plane_grid(768, 816, 32, 16),
            PlaneGrid { m: 16, n: 1 }
        );
    }

    #[test]
    fn a_strip_is_as_wide_as_half_the_last_level_cache_holds() {
        const MB: usize = 1024 * 1024;
        // b23 expand on a 16 MB cache: 48 rows of 272 f32 a box, 52 KB, so the cap decides.
        assert_eq!(
            cube_order(48, 272, 4, Some(16 * MB)),
            CubeOrder::SwizzleRow(8)
        );
        // 48 rows of 16384 f32 are 3 MB a box: two fit the 8 MB half, four do not.
        assert_eq!(
            cube_order(48, 16384, 4, Some(16 * MB)),
            CubeOrder::SwizzleRow(2)
        );
        // A box already past half the cache is walked unswizzled.
        assert_eq!(cube_order(48, 65536, 4, Some(16 * MB)), CubeOrder::RowMajor);
    }

    #[test]
    fn an_unreported_cache_takes_the_multi_level_routines_width() {
        assert_eq!(cube_order(48, 272, 4, None), CubeOrder::SwizzleRow(4));
    }

    #[test]
    fn a_tile_width_is_a_whole_number_of_lines() {
        assert_eq!(aligned_divisor_at_most(1632, 39, 16), 32);
        assert_eq!(aligned_divisor_at_most(272, 39, 16), 16);
        assert_eq!(aligned_divisor_at_most(336, 39, 16), 16);
        assert_eq!(aligned_divisor_at_most(144, 39, 16), 16);
        assert_eq!(aligned_divisor_at_most(112, 39, 16), 16);
    }

    #[test]
    fn an_aligned_width_is_left_where_it_was() {
        for n in [192, 672, 960, 160, 448, 2688, 32] {
            assert_eq!(
                aligned_divisor_at_most(n, 39, 16),
                divisor_at_most(n, 39),
                "n = {n}"
            );
        }
    }

    #[test]
    fn an_axis_no_line_divides_keeps_its_largest_divisor() {
        assert_eq!(aligned_divisor_at_most(56, 39, 16), 28);
        assert_eq!(aligned_divisor_at_most(24, 39, 16), 24);
        assert_eq!(aligned_divisor_at_most(8, 39, 16), 8);
    }
}
