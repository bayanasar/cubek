//! Where a plane shuffle's keys and values come from: the lines one step loads from an operand's
//! tile, decoded as they arrive where it is packed, and a packed cache's keys with their fix-up.

mod base;
mod packed;
mod tile;

pub use base::*;
pub use packed::*;
pub use tile::*;
