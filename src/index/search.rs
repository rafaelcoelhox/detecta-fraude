use super::distance::*;
use super::format::*;
use super::*;
use crate::{DIM, K, STORE_DIM};

const PARK_LIMIT: usize = 256;

struct Hunt {
    dist: [i64; K],
    tag: [u8; K],
    parked: [(i32, i64); PARK_LIMIT],
    held: usize,
    seen: u8,
    tally: u8,
    anchor: u8,
    seek: u8,
    pinned: bool,
    draining: bool,
}

impl Hunt {
    #[inline(always)]
    fn fresh() -> Self {
        Hunt {
            dist: [i64::MAX; K],
            tag: [0u8; K],
            parked: [(0i32, 0i64); PARK_LIMIT],
            held: 0,
            seen: 0,
            tally: 0,
            anchor: 0,
            seek: 0,
            pinned: false,
            draining: false,
        }
    }

    #[inline(always)]
    fn ceil(&self) -> i64 {
        self.dist[K - 1]
    }

    #[inline(always)]
    fn admit(&mut self, d: i64, t: u8) {
        if d >= self.dist[K - 1] {
            return;
        }
        let dropped = self.tag[K - 1];
        let full = self.seen as usize == K;
        let mut p = K - 1;
        while p > 0 && d < self.dist[p - 1] {
            self.dist[p] = self.dist[p - 1];
            self.tag[p] = self.tag[p - 1];
            p -= 1;
        }
        self.dist[p] = d;
        self.tag[p] = t;
        if full {
            self.tally = self.tally + t - dropped;
        } else {
            self.seen += 1;
            self.tally += t;
        }
        if !self.pinned
            && !self.draining
            && self.seen as usize == K
            && (self.tally == 0 || self.tally as usize == K)
        {
            self.pinned = true;
            self.anchor = self.tally;
            self.seek = if self.tally == 0 {
                1 << LABEL_FRAUD
            } else {
                1 << LABEL_LEGIT
            };
        }
    }

    #[inline(always)]
    fn skippable(&self, kinds: u8) -> bool {
        self.pinned && !self.draining && self.tally == self.anchor && kinds & self.seek == 0
    }

    #[inline(always)]
    fn stash(&mut self, node: i32, bound: i64) -> bool {
        if self.held < PARK_LIMIT {
            self.parked[self.held] = (node, bound);
            self.held += 1;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn unsettled(&self) -> bool {
        self.pinned && self.tally != self.anchor
    }

    #[inline(always)]
    fn verdict(&self) -> u8 {
        let mut total = 0u8;
        for t in self.tag {
            total += t;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn fraud_count_pair_avx2(idx: &IndexReader, query: &[i16; STORE_DIM]) -> u8 {
    let qp = query_pairs_avx2(query);
    let mut hunt = Hunt::fresh();

    let key = partition_key(query);
    let home = idx.part_by_key(key);
    if home >= 0 {
        let (root, _len) = read_partition_meta(idx, home as usize);
        explore(idx, root, 0, &qp, query, &mut hunt);
        if hunt.unsettled() {
            flush(idx, &qp, query, &mut hunt);
        }
    }

    let mut order = [(0i32, 0i64); 256];
    let mut n = 0usize;
    for p in 0..idx.part_count() as i32 {
        if p == home {
            continue;
        }
        let lb = lower_bound_partition_avx2(idx, p as usize, query);
        if lb >= hunt.ceil() {
            continue;
        }
        order[n] = (p, lb);
        n += 1;
    }
    order[..n].sort_unstable_by_key(|&(_, lb)| lb);

    for &(p, lb) in &order[..n] {
        if lb >= hunt.ceil() {
            break;
        }
        let (root, _len) = read_partition_meta(idx, p as usize);
        explore(idx, root, lb, &qp, query, &mut hunt);
        if hunt.unsettled() {
            flush(idx, &qp, query, &mut hunt);
        }
    }

    if hunt.unsettled() {
        flush(idx, &qp, query, &mut hunt);
    }

    hunt.verdict()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn explore(
    idx: &IndexReader,
    root: i32,
    bound0: i64,
    qp: &[std::arch::x86_64::__m256i; DIM_PAIRS],
    q: &[i16; STORE_DIM],
    hunt: &mut Hunt,
) {
    if root < 0 || root as u32 >= idx.node_count() {
        return;
    }
    let kinds = idx.kinds_ptr();
    let mut stack = [(0i32, 0i64); 128];
    let mut sp = 0usize;
    let mut node = root;
    let mut bound = bound0;

    loop {
        if bound < hunt.ceil() {
            let parked = hunt.skippable(*kinds.add(node as usize)) && hunt.stash(node, bound);
            if !parked {
                let (left, right, start, len) = read_node_meta(idx, node as usize);
                if left < 0 {
                    sweep_leaf(idx, start, len, qp, hunt);
                } else {
                    let bl = lower_bound_node_avx2(idx, left as usize, q);
                    let br = lower_bound_node_avx2(idx, right as usize, q);
                    let (first, fb, second, sb) = if bl <= br {
                        (left, bl, right, br)
                    } else {
                        (right, br, left, bl)
                    };
                    if sb < hunt.ceil() && sp < stack.len() {
                        stack[sp] = (second, sb);
                        sp += 1;
                    }
                    node = first;
                    bound = fb;
                    continue;
                }
            }
        }
        if sp == 0 {
            break;
        }
        sp -= 1;
        let (nn, nb) = stack[sp];
        node = nn;
        bound = nb;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sweep_leaf(
    idx: &IndexReader,
    start_block: i32,
    len: i32,
    qp: &[std::arch::x86_64::__m256i; DIM_PAIRS],
    hunt: &mut Hunt,
) {
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    let blocks = (len as usize).div_ceil(LANES);
    let labels = idx.labels_ptr();
    let vectors = idx.vectors_ptr();
    let total = len as usize;

    for b in 0..blocks {
        let base = start_block as usize + b;
        let ahead = b + 2;
        if ahead < blocks {
            let pb = start_block as usize + ahead;
            _mm_prefetch::<_MM_HINT_T0>(vectors.add(pb * DIM * LANES) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(labels.add(pb * LANES) as *const i8);
        }
        let d = distance_pair_block8(vectors, base * DIM * LANES, qp);
        let active = (total - b * LANES).min(LANES);
        let off = base * LANES;
        for lane in 0..active {
            let dd = d[lane];
            if dd < hunt.ceil() {
                hunt.admit(dd, *labels.add(off + lane));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flush(
    idx: &IndexReader,
    qp: &[std::arch::x86_64::__m256i; DIM_PAIRS],
    q: &[i16; STORE_DIM],
    hunt: &mut Hunt,
) {
    let count = hunt.held;
    hunt.held = 0;
    hunt.pinned = false;
    if count == 0 {
        return;
    }
    hunt.draining = true;
    let mut i = 0;
    while i < count {
        let (node, bound) = hunt.parked[i];
        if bound < hunt.ceil() {
            explore(idx, node, bound, qp, q, hunt);
        }
        i += 1;
    }
    hunt.draining = false;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn fraud_count_exact_avx2(idx: &IndexReader, query: &[i16; STORE_DIM]) -> u8 {
    let mut best_dists = [i64::MAX; K];
    let mut best_labels = [0u8; K];

    let key = partition_key(query);
    let primary = idx.part_by_key(key);
    if primary >= 0 {
        let (root, _len) = read_partition_meta(idx, primary as usize);
        if search_node_avx2(idx, root, 0, query, &mut best_dists, &mut best_labels) {
            return sum_labels(&best_labels);
        }
    }

    let mut probes = [(0i32, 0i64); 256];
    let mut n = 0usize;
    for p in 0..idx.part_count() as i32 {
        if p == primary {
            continue;
        }
        let lb = lower_bound_partition_avx2(idx, p as usize, query);
        if lb >= best_dists[K - 1] {
            continue;
        }
        probes[n] = (p, lb);
        n += 1;
    }
    probes[..n].sort_unstable_by_key(|&(_, lb)| lb);

    for &(part_idx, lb) in &probes[..n] {
        if lb >= best_dists[K - 1] {
            break;
        }
        let (root, _len) = read_partition_meta(idx, part_idx as usize);
        if search_node_avx2(idx, root, lb, query, &mut best_dists, &mut best_labels) {
            break;
        }
    }

    sum_labels(&best_labels)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_node_avx2(
    idx: &IndexReader,
    root: i32,
    root_bound: i64,
    query: &[i16; STORE_DIM],
    best_dists: &mut [i64; K],
    best_labels: &mut [u8; K],
) -> bool {
    if root < 0 || root as u32 >= idx.node_count() {
        return false;
    }

    let mut stack_node = [0i32; 128];
    let mut stack_bound = [0i64; 128];
    let mut sp = 0usize;
    let mut current = root;
    let mut current_bound = root_bound;

    loop {
        if current_bound < best_dists[K - 1] {
            let (left, right, start, len) = read_node_meta(idx, current as usize);
            if left < 0 {
                if scan_leaf_avx2(idx, start, len, query, best_dists, best_labels) {
                    return true;
                }
            } else {
                let lb = lower_bound_node_avx2(idx, left as usize, query);
                let rb = lower_bound_node_avx2(idx, right as usize, query);
                let (near, near_b, far, far_b) = if lb <= rb {
                    (left, lb, right, rb)
                } else {
                    (right, rb, left, lb)
                };
                if far_b < best_dists[K - 1] && sp < stack_node.len() {
                    stack_node[sp] = far;
                    stack_bound[sp] = far_b;
                    sp += 1;
                }
                current = near;
                current_bound = near_b;
                continue;
            }
        }

        if sp == 0 {
            break;
        }
        sp -= 1;
        current = stack_node[sp];
        current_bound = stack_bound[sp];
    }
    early_done(best_dists)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_leaf_avx2(
    idx: &IndexReader,
    start_block: i32,
    len: i32,
    query: &[i16; STORE_DIM],
    best_dists: &mut [i64; K],
    best_labels: &mut [u8; K],
) -> bool {
    use std::arch::x86_64::*;

    let blocks = (len as usize).div_ceil(LANES);
    let labels_ptr = idx.labels_ptr();
    let vectors_ptr = idx.vectors_ptr();

    let pair_layout = idx.is_kd_pair();
    let mut q_broadcast = [_mm256_setzero_si256(); DIM];
    if !pair_layout {
        for d in 0..DIM {
            q_broadcast[d] = _mm256_set1_epi32(query[d] as i32);
        }
    }
    let mut q_pairs = [_mm256_setzero_si256(); DIM_PAIRS];
    if pair_layout {
        for p in 0..DIM_PAIRS {
            let lo = query[p * 2] as u16 as u32;
            let hi = query[p * 2 + 1] as u16 as u32;
            q_pairs[p] = _mm256_set1_epi32((lo | (hi << 16)) as i32);
        }
    }

    let total_len = len as usize;
    for b in 0..blocks {
        let block_idx = start_block as usize + b;
        let labels_base = block_idx * LANES;
        let block_off_i16 = block_idx * DIM * LANES;
        let dists = if pair_layout {
            distance_pair_block8(vectors_ptr, block_off_i16, &q_pairs)
        } else {
            distance_block8(vectors_ptr, block_off_i16, &q_broadcast)
        };
        let lane_count = (total_len - b * LANES).min(LANES);
        for (lane, &d) in dists.iter().enumerate().take(lane_count) {
            if d < best_dists[K - 1] {
                let label = *labels_ptr.add(labels_base + lane);
                insert_best(d, label, best_dists, best_labels);
            }
        }
        if early_done(best_dists) {
            return true;
        }
    }
    false
}

pub(crate) fn fraud_count_scalar(idx: &IndexReader, query: &[i16; STORE_DIM]) -> u8 {
    let mut best_dists = [i64::MAX; K];
    let mut best_labels = [0u8; K];

    let key = partition_key(query);
    let primary = idx.part_by_key(key);
    if primary >= 0 {
        let (root, _len) = read_partition_meta(idx, primary as usize);
        if search_node_scalar(idx, root, 0, query, &mut best_dists, &mut best_labels) {
            return sum_labels(&best_labels);
        }
    }

    let mut probes = [(0i32, 0i64); 256];
    let mut n = 0usize;
    for p in 0..idx.part_count() as i32 {
        if p == primary {
            continue;
        }
        let lb = lower_bound_partition_scalar(idx, p as usize, query);
        if lb >= best_dists[K - 1] {
            continue;
        }
        probes[n] = (p, lb);
        n += 1;
    }
    probes[..n].sort_unstable_by_key(|&(_, lb)| lb);

    for &(part_idx, lb) in &probes[..n] {
        if lb >= best_dists[K - 1] {
            break;
        }
        let (root, _len) = read_partition_meta(idx, part_idx as usize);
        if search_node_scalar(idx, root, lb, query, &mut best_dists, &mut best_labels) {
            break;
        }
    }
    sum_labels(&best_labels)
}

fn search_node_scalar(
    idx: &IndexReader,
    root: i32,
    root_bound: i64,
    query: &[i16; STORE_DIM],
    best_dists: &mut [i64; K],
    best_labels: &mut [u8; K],
) -> bool {
    if root < 0 || root as u32 >= idx.node_count() {
        return false;
    }

    let mut stack_node = [0i32; 128];
    let mut stack_bound = [0i64; 128];
    let mut sp = 0usize;
    let mut current = root;
    let mut current_bound = root_bound;

    loop {
        if current_bound < best_dists[K - 1] {
            let (left, right, start, len) = read_node_meta(idx, current as usize);
            if left < 0 {
                if scan_leaf_scalar(idx, start, len, query, best_dists, best_labels) {
                    return true;
                }
            } else {
                let lb = lower_bound_node_scalar(idx, left as usize, query);
                let rb = lower_bound_node_scalar(idx, right as usize, query);
                let (near, near_b, far, far_b) = if lb <= rb {
                    (left, lb, right, rb)
                } else {
                    (right, rb, left, lb)
                };
                if far_b < best_dists[K - 1] && sp < stack_node.len() {
                    stack_node[sp] = far;
                    stack_bound[sp] = far_b;
                    sp += 1;
                }
                current = near;
                current_bound = near_b;
                continue;
            }
        }

        if sp == 0 {
            break;
        }
        sp -= 1;
        current = stack_node[sp];
        current_bound = stack_bound[sp];
    }
    early_done(best_dists)
}

fn scan_leaf_scalar(
    idx: &IndexReader,
    start_block: i32,
    len: i32,
    query: &[i16; STORE_DIM],
    best_dists: &mut [i64; K],
    best_labels: &mut [u8; K],
) -> bool {
    let blocks = (len as usize).div_ceil(LANES);
    let labels_ptr = idx.labels_ptr();
    let vectors_ptr = idx.vectors_ptr();
    let total_len = len as usize;
    for b in 0..blocks {
        let block_idx = start_block as usize + b;
        let lane_count = (total_len - b * LANES).min(LANES);
        for lane in 0..lane_count {
            let mut dist = 0i64;
            let block_off = block_idx * DIM * LANES;
            for d in 0..DIM {
                let off = if idx.is_kd_pair() {
                    pair_offset(d, lane)
                } else {
                    d * LANES + lane
                };
                let v = unsafe { *vectors_ptr.add(block_off + off) };
                let diff = v as i64 - query[d] as i64;
                dist += diff * diff;
            }
            if dist < best_dists[K - 1] {
                let label = unsafe { *labels_ptr.add(block_idx * LANES + lane) };
                insert_best(dist, label, best_dists, best_labels);
            }
        }
        if early_done(best_dists) {
            return true;
        }
    }
    false
}
