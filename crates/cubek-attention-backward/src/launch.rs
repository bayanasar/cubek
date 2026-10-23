//! The entry point a caller runs a backward pass through.

use crate::kernels::UnitAttentionBackwardRoutine;
use crate::{AttentionBackwardGradients, AttentionBackwardRoutine, AttentionBackwardTensors};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// The gradients of the attention `tensors` state, on their client's device.
///
/// # Errors
///
/// Why the device cannot run the backward pass.
pub fn attention_backward(
    tensors: AttentionBackwardTensors,
) -> Result<AttentionBackwardGradients, Refusal> {
    let hw = Hardware::from_client(tensors.client());
    let routine = UnitAttentionBackwardRoutine;
    let settings = routine.settings(&hw).swap_remove(0);
    routine.launch_on(&hw, tensors, &settings)
}
