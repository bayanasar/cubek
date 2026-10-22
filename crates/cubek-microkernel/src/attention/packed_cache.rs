//! The packed KV cache's format as the attention reads it: the widths its codes are stored at, and
//! whether its keys carry the QJL fix-up. Its codes are read through tiles, which decode them
//! ([`Tile::decoded`](cubecl::tile::Tile::decoded)).

/// The per-coordinate widths a packed cache's codes are stored at: `32 / bits` coordinates share
/// each `u32` word. Keys and values need not agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvBitWidths {
    /// Per-coordinate width of the packed keys.
    pub key: u32,
    /// Per-coordinate width of the packed values.
    pub value: u32,
}

/// Whether a cached vector spends one bit per coordinate on the QJL sign fix-up of its rounding
/// residual.
///
/// Two different algorithms rather than a flag: [`Qjl`](Self::Qjl) emits three extra `O(d²)`
/// passes, writes a second scalar buffer, and shifts every packed index up by one bit. Naming them
/// keeps the choice legible at a call site, where a bare `true` said nothing about which of those
/// it selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyFixup {
    /// Round-only (Q_mse): the packed field is entirely the centroid index.
    RoundOnly,
    /// Q_prod: the field's low bit is the sign of the residual's projection, and the residual's
    /// length is stored beside the vector's own.
    Qjl,
}

impl KeyFixup {
    /// Bits per coordinate the fix-up spends, `0` or `1`. The rounding index gets whatever the
    /// packed field has left.
    pub fn bits(self) -> u32 {
        match self {
            KeyFixup::RoundOnly => 0,
            KeyFixup::Qjl => 1,
        }
    }
}
