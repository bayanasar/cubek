//! Packed matrices as the tiled tests build them: block-scaled values a quantizer reads back
//! exactly, bound as the words and scales a kernel reads.

use super::indexed::upload;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Axis;
use cubecl::zspace::Shape;
use cubecl_common::e4m3;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor, WordsLaid};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::stored::Operand;
use cubek_std::matmul_axes::K;
use cubek_test_utils::{HostData, HostDataType, InputDataType, TestInput};

/// Block-scaled `value`s along the contraction, 32 to a scale.
pub(super) fn block_scheme(value: QuantValue) -> QuantScheme {
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
pub(super) fn assert_distinct(
    values: &[f32],
    scheme: &QuantScheme,
    (slab_rows, k): (usize, usize),
) {
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
pub(super) fn packed_matrix(
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
pub(super) fn packed_stack(
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
