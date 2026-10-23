//! The routines of attention's backward pass, one folder per scope: each walks the tiles of
//! queries by keys once owned by the queries, then once owned by the keys.

mod unit;

pub use unit::*;
