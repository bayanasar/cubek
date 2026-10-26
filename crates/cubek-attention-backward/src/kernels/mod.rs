//! The routines of attention's backward pass, in `routines`, and what they share: the problem,
//! the tensors, the walks a routine launches, and the autotune that picks one.

pub mod tune;

mod blueprint;
mod launch;
mod problem;
mod routine;
mod routines;
mod tensors;
mod walk;

pub use blueprint::*;
pub use launch::*;
pub use problem::*;
pub use routine::*;
pub use routines::*;
pub use tensors::*;
pub use walk::*;
