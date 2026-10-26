//! A packed stack read through an indexed launch answers the product with its dequantized values:
//! each route, or each group, reads the slab its entry names, words and scales both.

use super::super::packed::{assert_distinct, block_scheme, packed_matrix, packed_stack};
use super::*;
use cubecl::quant::scheme::{QuantScheme, QuantValue};
use cubek_matmul::kernels::m_grouped::{MGroupedMatmulTensors, MGroupedSettings};
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::stack::SlabAxis;
use cubek_matmul::kernels::tensor::PACKED_STACK_UNREAD;
use cubek_microkernel::matmul::MatmulOperand;
use half::f16;

/// `f16` values in `[-1, 1)`, and what they read back as.
fn f16_values(count: usize, seed: usize) -> (Vec<f16>, Vec<f32>) {
    let halves: Vec<f16> = values(count, seed).into_iter().map(f16::from_f32).collect();
    let floats = halves.iter().map(|half| half.to_f32()).collect();
    (halves, floats)
}

/// The `[slabs, k, n]` stack a `[slabs · n, k]` matrix of `values` holds, read at one cell.
fn stack_cell(values: &[f32], (n, k): (usize, usize), slab: usize, kk: usize, j: usize) -> f32 {
    values[(slab * n + j) * k + kk]
}

/// A second shape, whose `n` is not a multiple of the column tile.
const NARROW: Routed = Routed {
    tokens: 5,
    picks: 3,
    slabs: 4,
    k: 128,
    n: 80,
};

/// `routed`'s routes against a packed stack of `scheme`: each route's product is the one its
/// slab's dequantized values make.
fn a_routed_packed_stack_reads_each_routes_slab(scheme: QuantScheme) {
    for routed in [ROUTED, NARROW] {
        let Routed {
            tokens,
            picks,
            slabs,
            k,
            n,
        } = routed;
        let client = client();
        let (lhs_half, lhs) = f16_values(tokens * k, 1);
        let (stack, dequantized) =
            packed_stack(&client, MatmulOperand::Rhs, scheme, slabs, n, k, 2);
        let expected = routed.expected_from(&lhs, |slab, kk, j| {
            stack_cell(&dequantized, (n, k), slab, kk, j)
        });
        check_settings(
            PlaneShuffleMatmulRoutine,
            |routine, _| routine.packed_settings(),
            || {
                let lhs = upload(&client, &[tokens, 1, 1, k], &lhs_half);
                let entries = upload(&client, &[tokens, picks], &routed.routes());
                let table = MatmulIndexTable {
                    index: MatmulIndex::Slab { slabs },
                    entries,
                };
                (lhs, stack.clone(), table)
            },
            &expected,
            1e-2,
        );
    }
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_q8_slab() {
    a_routed_packed_stack_reads_each_routes_slab(block_scheme(QuantValue::Q8S));
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_q4_slab() {
    a_routed_packed_stack_reads_each_routes_slab(block_scheme(QuantValue::Q4S));
}

#[test]
fn a_routed_product_on_the_plane_shuffle_reads_each_route_s_packed_nvfp4_slab() {
    a_routed_packed_stack_reads_each_routes_slab(nvfp4_scheme());
}

/// Rows grouped by `sizes`, one group a slab of a packed stack of `scheme`: each row's product is
/// the one its group's slab makes, the second group empty.
fn a_grouped_packed_stack_reads_each_groups_slab(scheme: QuantScheme) {
    let (k, n, sizes) = (128usize, 96usize, [5usize, 0, 11, 2]);
    let (groups, rows) = (sizes.len(), sizes.iter().sum::<usize>());
    let ends: Vec<u32> = sizes
        .iter()
        .scan(0, |end, size| {
            *end += size;
            Some(*end as u32)
        })
        .collect();
    let group_of = |row: usize| ends.iter().position(|&end| row < end as usize).unwrap();
    let client = client();
    let hw = Hardware::from_client(&client);
    let (lhs_half, lhs) = f16_values(rows * k, 3);
    let (stack, dequantized) = packed_matrix(&client, MatmulOperand::Rhs, scheme, groups * n, k, 4);
    assert_distinct(&dequantized, &scheme, (n, k));
    let (mut ran, mut cubes_past_device) = (0, true);
    for settings in MGroupedSettings::plane_shuffle(&hw) {
        let tensors = MGroupedMatmulTensors::new(
            &client,
            upload(&client, &[rows, k], &lhs_half),
            StoredTensor::Packed(stack.clone()),
            upload(&client, &[groups], &ends),
            SlabAxis::Output,
        )
        .expect("a packed stack of whole slabs derives a grouped launch");
        let out = match tensors.launch(&hw, &PlaneShuffleMatmulRoutine, &settings) {
            Ok(out) => out,
            Err(why) => {
                cubes_past_device &= matches!(why, Refusal::CubePastDevice { .. });
                continue;
            }
        };
        ran += 1;
        let got = HostData::from_tensor_handle(&client, out.unpad(), HostDataType::F32);
        for row in 0..rows {
            for j in 0..n {
                let want: f32 = (0..k)
                    .map(|kk| {
                        lhs[row * k + kk] * stack_cell(&dequantized, (n, k), group_of(row), kk, j)
                    })
                    .sum();
                let have = got.get_f32(&[row, j]);
                assert!(
                    (have - want).abs() <= 1e-2 * (1. + want.abs()),
                    "{settings}: row {row} column {j} is {have}, want {want}"
                );
            }
        }
    }
    settled(ran, cubes_past_device);
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q8_slab() {
    a_grouped_packed_stack_reads_each_groups_slab(block_scheme(QuantValue::Q8S));
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_q4_slab() {
    a_grouped_packed_stack_reads_each_groups_slab(block_scheme(QuantValue::Q4S));
}

#[test]
fn a_grouped_product_on_the_plane_shuffle_reads_each_group_s_packed_nvfp4_slab() {
    a_grouped_packed_stack_reads_each_groups_slab(nvfp4_scheme());
}

/// Slabs along `K` that split a scale block, or a word of block scales, are refused rather than
/// read across.
#[test]
fn a_packed_stack_whose_slabs_split_a_scale_block_is_refused() {
    let client = client();
    let along_k = |scheme, slabs| {
        let (matrix, _) = packed_matrix(&client, MatmulOperand::Rhs, scheme, 8, 128, 5);
        matrix.stacked(slabs, K)
    };
    // 32-value blocks: 4 slabs hold whole blocks, 8 hold half of one.
    assert!(along_k(block_scheme(QuantValue::Q8S), 4).is_ok());
    assert_eq!(
        along_k(block_scheme(QuantValue::Q8S), 8).err(),
        Some(PACKED_STACK_UNREAD)
    );
    // NVFP4: four 16-value blocks a word of scales, so 4 slabs split a word.
    assert!(along_k(nvfp4_scheme(), 2).is_ok());
    assert_eq!(along_k(nvfp4_scheme(), 4).err(), Some(PACKED_STACK_UNREAD));
}
