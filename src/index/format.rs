use crate::{DIM, SCALE};

pub const LABEL_LEGIT: u8 = 0;
pub const LABEL_FRAUD: u8 = 1;

pub const MAGIC: [u8; 8] = *b"DFKNN001";
pub const VERSION: u32 = 4;
pub const KD_PAIR_VERSION: u32 = 6;
pub const HEADER_SIZE: usize = 64;
pub const PART_SIZE: usize = 76;
pub const NODE_SIZE: usize = 80;
pub const LANES: usize = 8;
pub const BLOCK_BYTES: usize = DIM * LANES * 2;
pub const DIM_PAIRS: usize = DIM / 2;
pub const MCC_TABLE_SIZE: usize = 1024;
pub const DEFAULT_LEAF_SIZE: usize = 256;
pub const EARLY_DISTANCE_MILLI: i32 = 140;
pub const EARLY_DISTANCE_LIMIT: i64 = {
    let v = (SCALE as i32 * EARLY_DISTANCE_MILLI / 1000) as i64;
    v * v
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub scale: u32,
    pub dim: u32,
    pub store_dim: u32,
    pub n_points: u32,
    pub part_count: u32,
    pub node_count: u32,
    pub block_count: u32,
    pub mcc_table_offset: u32,
    pub _pad: [u8; 20],
}

const _: () = assert!(std::mem::size_of::<Header>() == HEADER_SIZE);

#[inline(always)]
pub(crate) fn pair_offset(d: usize, lane: usize) -> usize {
    (d / 2) * LANES * 2 + lane * 2 + (d & 1)
}
