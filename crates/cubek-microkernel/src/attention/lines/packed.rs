use crate::attention::{S, StepLines, StepLinesExpand, StepWalk};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The keys of a packed cache, read as lines, decoding as they are read ([`Tile::decoded`]).
///
/// With the QJL fix-up, the lines run twice the head dim: the first half the keys' tile, the
/// second the fix-up's, the same words read through the sign codebook under each key's residual
/// length. A query whose second half is its projection then scores both terms in one dot. Both
/// tiles span the kernel's head axis, so a key's row of each is the step's `lines` long, its
/// first half the key's coordinates.
#[derive(CubeType, Clone)]
pub struct PackedKeyLines {
    pub keys: Tile,
    pub fixup: ComptimeOption<Tile>,
}

#[cube]
impl<N: Size> StepLines<N> for PackedKeyLines {
    type Element = f32;

    fn load(
        &self,
        step: &Region,
        walk: &StepWalk,
        #[comptime] lines: usize,
    ) -> Array<Vector<f32, N>> {
        let qjl = self.fixup.is_some();
        let keys = self.keys.at(step);
        // Without a fix-up, the keys stand in for it, and are never read through it.
        let fixup = if comptime!(qjl) {
            self.fixup.clone().unwrap().at(step)
        } else {
            keys.clone()
        };
        let (keys, fixup) = (keys.decoded::<f32, N>(), fixup.decoded::<f32, N>());
        let half = comptime!(if qjl { lines / 2 } else { lines });
        let first_key = step.origin(S);
        let (teams, keys_in_flight) = comptime!((walk.teams, walk.keys_in_flight));
        let units_per_key = comptime!(walk.units_per_key);
        let per_key = comptime!(lines.div_ceil(units_per_key));
        let mut loaded = Array::<Vector<f32, N>>::new(comptime!(keys_in_flight * per_key));
        #[unroll]
        for c in 0..keys_in_flight {
            let at = c * teams + walk.team;
            let seen = ((first_key + at) as u32) < walk.attended;
            #[unroll]
            for p in 0..per_key {
                let line = walk.lane + p * units_per_key;
                let mut value = Vector::<f32, N>::cast_from(0u32);
                if line < lines && seen {
                    if comptime!(qjl) {
                        if line < half {
                            value = keys.read(at * lines + line);
                        } else {
                            value = fixup.read(at * lines + line - half);
                        }
                    } else {
                        value = keys.read(at * lines + line);
                    }
                }
                loaded[c * per_key + p] = value;
            }
        }
        loaded
    }

    fn at(&self, region: &Region) -> PackedKeyLines {
        PackedKeyLines {
            keys: self.keys.at(region),
            fixup: self.fixup.clone().map(|fixup| fixup.at(region)),
        }
    }
}
