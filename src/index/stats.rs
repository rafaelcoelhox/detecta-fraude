#[cfg(feature = "knn_stats")]
use std::cell::Cell;

#[cfg(feature = "knn_stats")]
thread_local! {
    static NODES: Cell<u32> = Cell::new(0);
    static LEAVES: Cell<u32> = Cell::new(0);
    static BLOCKS: Cell<u32> = Cell::new(0);
    static PARTS: Cell<u32> = Cell::new(0);
    static PRIMARY_HIT: Cell<bool> = Cell::new(false);
    static EARLY_HIT: Cell<bool> = Cell::new(false);
}

#[inline(always)]
pub fn inc_nodes() {
    #[cfg(feature = "knn_stats")]
    NODES.with(|c| c.set(c.get() + 1));
}
#[inline(always)]
pub fn inc_leaves() {
    #[cfg(feature = "knn_stats")]
    LEAVES.with(|c| c.set(c.get() + 1));
}
#[inline(always)]
pub fn inc_blocks() {
    #[cfg(feature = "knn_stats")]
    BLOCKS.with(|c| c.set(c.get() + 1));
}
#[inline(always)]
pub fn inc_parts() {
    #[cfg(feature = "knn_stats")]
    PARTS.with(|c| c.set(c.get() + 1));
}
#[inline(always)]
pub fn set_primary_hit() {
    #[cfg(feature = "knn_stats")]
    PRIMARY_HIT.with(|c| c.set(true));
}
#[inline(always)]
pub fn set_early_hit() {
    #[cfg(feature = "knn_stats")]
    EARLY_HIT.with(|c| c.set(true));
}

#[cfg(feature = "knn_stats")]
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryStats {
    pub nodes: u32,
    pub leaves: u32,
    pub blocks: u32,
    pub partitions: u32,
    pub primary_hit: bool,
    pub early_hit: bool,
}

#[cfg(feature = "knn_stats")]
pub fn reset() {
    NODES.with(|c| c.set(0));
    LEAVES.with(|c| c.set(0));
    BLOCKS.with(|c| c.set(0));
    PARTS.with(|c| c.set(0));
    PRIMARY_HIT.with(|c| c.set(false));
    EARLY_HIT.with(|c| c.set(false));
}

#[cfg(feature = "knn_stats")]
pub fn snapshot() -> QueryStats {
    QueryStats {
        nodes: NODES.with(|c| c.get()),
        leaves: LEAVES.with(|c| c.get()),
        blocks: BLOCKS.with(|c| c.get()),
        partitions: PARTS.with(|c| c.get()),
        primary_hit: PRIMARY_HIT.with(|c| c.get()),
        early_hit: EARLY_HIT.with(|c| c.get()),
    }
}
