//! A packed stack read through an indexed launch answers the product with its dequantized values:
//! each route, or each group, reads the slab its entry names, words and scales both.

use super::*;
use cubecl::ir::ElemType;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::tile::Axis;
use cubecl_common::e4m3;
use cubek_matmul::kernels::m_grouped::{MGroupedMatmulTensors, MGroupedSettings};
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::stack::SlabAxis;
use cubek_matmul::kernels::tensor::{PACKED_STACK_UNREAD, PackedTensor, WordsLaid};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::stored::Operand;
use cubek_test_utils::{InputDataType, TestInput};
use half::f16;

/// Block-scaled `value`s along the contraction, 32 to a scale.
fn block_scheme(value: QuantValue) -> QuantScheme {
    QuantScheme::default()
        .with_value(value)
        .per_block([32], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}

/// The per-tensor scale a two-level scheme is built under, folded into the host values.
const GLOBAL_SCALE: f32 = 2.;

/// Block scales with one or two mantissa bits, which the quantizer and a `ue4m3` byte hold exactly.
const MANTISSAS: [f32; 4] = [1., 1.5, 2., 3.];

/// A well-mixed number from `i` and `seed`, so no pattern of the indices repeats at a slab, a row
/// or a word.
fn hashed(i: usize, seed: usize) -> usize {
    let mut x = (i as u64) ^ ((seed as u64) << 32);
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (x ^ (x >> 32)) as usize
}

/// The values of one block of `scheme`'s scales along a row.
fn block_of(scheme: &QuantScheme) -> usize {
    scheme
        .block_size()
        .expect("a block-scaled scheme")
        .to_dim_vec(1)[0] as usize
}

/// `rows` rows of `k` values in blocks, each holding its peak and codes of the scheme under its own
/// scale, so the quantizer rounds nothing and the host's values are the dequantized ones.
fn exact_values(scheme: &QuantScheme, rows: usize, k: usize, seed: usize) -> Vec<f32> {
    let block = block_of(scheme);
    let peak = scheme.value.range().1;
    let codes: Vec<f32> = match scheme.value {
        QuantValue::E2M1 => vec![0., 0.5, 1., 1.5, 2., 3., 4., 6.],
        _ => (0..=peak as usize).map(|code| code as f32).collect(),
    };
    let headroom = peak.log2().ceil() as i32 + 2;
    (0..rows * k)
        .map(|i| {
            let (block_at, within) = (i / block, i % block);
            let h = hashed(block_at, seed);
            let scale = MANTISSAS[h % 4] * 2f32.powi(-headroom - ((h / 4) % 3) as i32);
            let pick = hashed(i, seed) % (2 * codes.len());
            let code = match within {
                0 => peak,
                _ if pick < codes.len() => codes[pick],
                _ => -codes[pick - codes.len()],
            };
            code * scale
        })
        .collect()
}

/// Whether more than a quarter of the cells of `a` and `b` differ.
fn mostly_differ(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).filter(|(a, b)| a != b).count() * 4 > a.len()
}

/// Slabs, rows and the words of a block all hold different values, so a read of the wrong one
/// cannot match: a shape or a seed that made them repeat fails here, not silently in a product.
fn assert_distinct(values: &[f32], scheme: &QuantScheme, (slab_rows, k): (usize, usize)) {
    let slab = slab_rows * k;
    let slabs: Vec<&[f32]> = values.chunks(slab).collect();
    for (i, a) in slabs.iter().enumerate() {
        for b in &slabs[i + 1..] {
            assert!(mostly_differ(a, b), "two slabs hold the same values");
        }
    }
    assert!(
        mostly_differ(&values[..k], &values[k..2 * k]),
        "two rows hold the same values"
    );
    let words: Vec<&[f32]> = values[..block_of(scheme)]
        .chunks(scheme.num_quants())
        .collect();
    for (i, a) in words.iter().enumerate() {
        for b in &words[i + 1..] {
            assert_ne!(a, b, "two words of a block hold the same values");
        }
    }
}

/// The axis of `operand` beside `K`, which a packed matrix's rows run along.
fn across(operand: MatmulOperand) -> Axis {
    operand
        .axes()
        .into_iter()
        .find(|&axis| axis != K)
        .expect("an operand has an axis beside K")
}

/// `scheme` as the reference quantizer takes it: block scales in `f32`, no per-tensor level. The
/// words it writes are the scheme's own.
fn block_level(scheme: &QuantScheme) -> QuantScheme {
    QuantScheme::default()
        .with_value(scheme.value)
        .per_block([block_of(scheme) as u8], ScaleDtype::F32)
        .with_store(scheme.store)
}

/// `scales`, `f32` block scales of `scheme`'s grid, stored in the type it states: every scale has
/// few mantissa bits ([`MANTISSAS`]), so a `ue4m3` byte holds it exactly.
fn stored_scales(
    client: &Client,
    scheme: &QuantScheme,
    scales: TensorHandle,
    (rows, blocks): (usize, usize),
) -> TensorHandle {
    let stored = scheme.scale_dtype();
    if stored == ScaleDtype::F32 {
        return scales;
    }
    assert_eq!(
        stored,
        ScaleDtype::UE4M3,
        "only `f32` and `ue4m3` scales are built here"
    );
    let host = HostData::from_tensor_handle(client, scales, HostDataType::F32);
    let bytes: Vec<u8> = (0..rows * blocks)
        .map(|i| e4m3::from_f32(host.get_f32(&[i / blocks, i % blocks])).to_bits())
        .collect();
    TensorHandle::new_contiguous(
        Shape::from(vec![rows, blocks]),
        client.create_from_slice(&bytes),
        ElemType::from_scale_dtype(stored),
    )
}

/// `rows` rows of `k` values, packed along `k` under `scheme` as `operand` reads them: the rows
/// are the axis of `operand` other than `K`. Its words run along `k` in a row each, as a
/// checkpoint stores them whichever axis the rows are. Its host values are the dequantized ones.
fn packed_matrix(
    client: &Client,
    operand: MatmulOperand,
    scheme: QuantScheme,
    rows: usize,
    k: usize,
    seed: usize,
) -> (PackedTensor, Vec<f32>) {
    let across = across(operand);
    let values = exact_values(&scheme, rows, k, seed);
    let tensor = TestInput::builder(client.clone(), Shape::from(vec![rows, k]))
        .dtype(InputDataType::Quantized(block_level(&scheme)))
        .custom(values.clone())
        .generate_test_tensor();
    let block = block_of(&scheme);
    let block_scales = stored_scales(
        client,
        &scheme,
        tensor.quantization.expect("a quantized tensor").scale,
        (rows, k / block),
    );
    let global = (scheme.num_levels() > 1).then(|| upload(client, &[1, 1], &[GLOBAL_SCALE]));
    let values = match global {
        Some(_) => values.into_iter().map(|v| v * GLOBAL_SCALE).collect(),
        None => values,
    };
    let packed = PackedTensor::new(
        PlainTensor::of_handle(tensor.handle),
        WordsLaid::InRows,
        PlainTensor::of_handle(block_scales),
        global,
        scheme,
        Operand::of_scheme(&scheme, K, [across, K]).expect("one block along k"),
        [(across, rows), (K, k)],
    );
    (packed, values)
}

/// A packed stack of `slabs` matrices, each `rows` rows of `k`, laid one after the other along
/// the axis of `operand` beside `K`.
fn packed_stack(
    client: &Client,
    operand: MatmulOperand,
    scheme: QuantScheme,
    slabs: usize,
    rows: usize,
    k: usize,
    seed: usize,
) -> (StoredTensor, Vec<f32>) {
    let (packed, values) = packed_matrix(client, operand, scheme, slabs * rows, k, seed);
    let stack = packed
        .stacked(slabs, across(operand))
        .expect("whole slabs of whole blocks");
    assert_distinct(&values, &scheme, (rows, k));
    (StoredTensor::Packed(stack), values)
}

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
