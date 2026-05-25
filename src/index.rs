use crate::{DIM, K, SCALE, STORE_DIM};
use std::path::Path;

pub const LABEL_LEGIT: u8 = 0;
pub const LABEL_FRAUD: u8 = 1;

pub const MAGIC: [u8; 8] = *b"DFKNN001";
pub const VERSION: u32 = 4;
pub const HEADER_SIZE: usize = 64;
pub const PART_SIZE: usize = 76;
pub const NODE_SIZE: usize = 80;
pub const LANES: usize = 8;
pub const BLOCK_BYTES: usize = DIM * LANES * 2;
pub const MCC_TABLE_SIZE: usize = 1024;
pub const DEFAULT_LEAF_SIZE: usize = 128;
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

pub struct IndexReader {
    _map: memmap2::Mmap,
    base: *const u8,
    len: usize,
    partitions_off: usize,
    nodes_off: usize,
    vectors_off: usize,
    labels_off: usize,
    mcc_table_off: usize,
    header: Header,
    part_by_key: [i32; 256],
}

unsafe impl Send for IndexReader {}
unsafe impl Sync for IndexReader {}

impl IndexReader {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };
        let base = map.as_ptr();
        let len = map.len();
        if len < HEADER_SIZE {
            return Err(invalid("index too small"));
        }

        let header: Header = unsafe { std::ptr::read_unaligned(base as *const Header) };
        if header.magic != MAGIC || header.version != VERSION {
            return Err(invalid("bad magic/version"));
        }
        if header.scale != SCALE as u32
            || header.dim as usize != DIM
            || header.store_dim as usize != STORE_DIM
        {
            return Err(invalid("dim/scale mismatch"));
        }

        let partitions_off = HEADER_SIZE;
        let nodes_off = partitions_off + header.part_count as usize * PART_SIZE;
        let vectors_off = nodes_off + header.node_count as usize * NODE_SIZE;
        let labels_off = vectors_off + header.block_count as usize * BLOCK_BYTES;
        let mcc_table_off = labels_off + header.block_count as usize * LANES;
        let end = mcc_table_off + MCC_TABLE_SIZE * 2;
        if end != len || header.mcc_table_offset as usize != mcc_table_off {
            return Err(invalid("index size mismatch"));
        }

        let mut part_by_key = [-1i32; 256];
        for i in 0..header.part_count as usize {
            let off = partitions_off + i * PART_SIZE;
            let key = read_u32_at(base, off);
            if (key as usize) < part_by_key.len() {
                part_by_key[key as usize] = i as i32;
            }
        }

        let idx = IndexReader {
            _map: map,
            base,
            len,
            partitions_off,
            nodes_off,
            vectors_off,
            labels_off,
            mcc_table_off,
            header,
            part_by_key,
        };
        idx.advise();
        idx.prefetch();
        Ok(idx)
    }

    #[inline]
    pub fn n_points(&self) -> u32 {
        self.header.n_points
    }

    #[inline]
    pub fn part_count(&self) -> u32 {
        self.header.part_count
    }

    #[inline]
    pub fn node_count(&self) -> u32 {
        self.header.node_count
    }

    #[inline]
    pub fn block_count(&self) -> u32 {
        self.header.block_count
    }

    #[inline]
    pub fn mcc_risk(&self, mcc: u32) -> i16 {
        let idx = (mcc as usize) % MCC_TABLE_SIZE;
        let off = self.mcc_table_off + idx * 2;
        read_i16_at(self.base, off)
    }

    #[inline]
    fn partitions_ptr(&self) -> *const u8 {
        unsafe { self.base.add(self.partitions_off) }
    }

    #[inline]
    fn nodes_ptr(&self) -> *const u8 {
        unsafe { self.base.add(self.nodes_off) }
    }

    #[inline]
    fn vectors_ptr(&self) -> *const i16 {
        unsafe { self.base.add(self.vectors_off) as *const i16 }
    }

    #[inline]
    fn labels_ptr(&self) -> *const u8 {
        unsafe { self.base.add(self.labels_off) }
    }

    #[inline]
    fn part_by_key(&self, key: u32) -> i32 {
        self.part_by_key[(key & 0xff) as usize]
    }

    #[inline]
    pub fn fraud_count(&self, query: &[i16; STORE_DIM]) -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { fraud_count_avx2(self, query) };
            }
        }
        fraud_count_scalar(self, query)
    }

    #[cfg(target_os = "linux")]
    fn advise(&self) {
        const MADV_HUGEPAGE: libc::c_int = 14;
        const MADV_RANDOM: libc::c_int = 1;
        const MADV_WILLNEED: libc::c_int = 3;
        unsafe {
            libc::madvise(self.base as *mut _, self.len, MADV_HUGEPAGE);
            let hot_start = self.vectors_off;
            let hot_len = self.len - hot_start;
            libc::madvise(self.base.add(hot_start) as *mut _, hot_len, MADV_HUGEPAGE);
            libc::madvise(self.base.add(hot_start) as *mut _, hot_len, MADV_RANDOM);
            libc::madvise(self.base.add(hot_start) as *mut _, hot_len, MADV_WILLNEED);
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn advise(&self) {}

    fn prefetch(&self) {
        const PAGE: usize = 4096;
        let mut acc = 0u8;
        let mut i = 0usize;
        while i < self.len {
            acc ^= unsafe { std::ptr::read_volatile(self.base.add(i)) };
            i += PAGE;
        }
        if acc == 0xFE {
            eprintln!("[index] prefetch sentinel hit");
        }
    }
}

fn invalid(msg: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[inline]
fn read_i16_at(base: *const u8, off: usize) -> i16 {
    unsafe { std::ptr::read_unaligned(base.add(off) as *const i16) }
}

#[inline]
fn read_i32_at(base: *const u8, off: usize) -> i32 {
    unsafe { std::ptr::read_unaligned(base.add(off) as *const i32) }
}

#[inline]
fn read_u32_at(base: *const u8, off: usize) -> u32 {
    unsafe { std::ptr::read_unaligned(base.add(off) as *const u32) }
}

#[inline]
pub fn partition_key(v: &[i16; STORE_DIM]) -> u32 {
    let mut key = 0u32;
    if v[5] >= 0 {
        key |= 1 << 0;
    }
    if v[9] > 0 {
        key |= 1 << 1;
    }
    if v[10] > 0 {
        key |= 1 << 2;
    }
    if v[11] > 0 {
        key |= 1 << 3;
    }
    let mr = v[12];
    if mr <= 2047 {
        // bucket 0
    } else if mr <= 4095 {
        key |= 1 << 4;
    } else if mr <= 6143 {
        key |= 2 << 4;
    } else {
        key |= 3 << 4;
    }
    if v[2] > 4096 {
        key |= 1 << 6;
    }
    if v[8] > 2048 {
        key |= 1 << 7;
    }
    key
}

#[inline]
fn lower_bound_dim(q: i16, lo: i16, hi: i16) -> i64 {
    let diff = if q < lo {
        lo as i64 - q as i64
    } else if q > hi {
        q as i64 - hi as i64
    } else {
        0
    };
    diff * diff
}

#[inline]
pub fn lower_bound_vec(
    q: &[i16; STORE_DIM],
    min: &[i16; STORE_DIM],
    max: &[i16; STORE_DIM],
) -> i64 {
    let mut acc = 0i64;
    let mut d = 0usize;
    while d < DIM {
        acc += lower_bound_dim(q[d], min[d], max[d]);
        d += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lower_bound_vec_avx2(
    q: &[i16; STORE_DIM],
    min: &[i16; STORE_DIM],
    max: &[i16; STORE_DIM],
) -> i64 {
    use std::arch::x86_64::*;

    let qv = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let mn = _mm256_loadu_si256(min.as_ptr() as *const __m256i);
    let mx = _mm256_loadu_si256(max.as_ptr() as *const __m256i);
    let zero = _mm256_setzero_si256();
    let below = _mm256_max_epi16(_mm256_sub_epi16(mn, qv), zero);
    let above = _mm256_max_epi16(_mm256_sub_epi16(qv, mx), zero);
    let diff = _mm256_max_epi16(below, above);
    let real_dims = _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let diff = _mm256_and_si256(diff, real_dims);
    let sq_pairs = _mm256_madd_epi16(diff, diff);
    let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_pairs));
    let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_pairs, 1));
    let sum = _mm256_add_epi64(lo, hi);
    let sum_hi = _mm256_extracti128_si256(sum, 1);
    let sum_128 = _mm_add_epi64(_mm256_castsi256_si128(sum), sum_hi);
    _mm_extract_epi64(sum_128, 0) + _mm_extract_epi64(sum_128, 1)
}

#[inline]
fn read_partition(
    idx: &IndexReader,
    part_idx: usize,
) -> (i32, i32, [i16; STORE_DIM], [i16; STORE_DIM]) {
    let p = idx.partitions_ptr();
    let off = part_idx * PART_SIZE;
    let root = read_i32_at(p, off + 4);
    let len = read_i32_at(p, off + 8);
    let min = read_qv(p, off + 12);
    let max = read_qv(p, off + 44);
    (root, len, min, max)
}

#[inline]
fn read_node(
    idx: &IndexReader,
    node_idx: usize,
) -> (i32, i32, i32, i32, [i16; STORE_DIM], [i16; STORE_DIM]) {
    let p = idx.nodes_ptr();
    let off = node_idx * NODE_SIZE;
    let left = read_i32_at(p, off);
    let right = read_i32_at(p, off + 4);
    let start = read_i32_at(p, off + 8);
    let len = read_i32_at(p, off + 12);
    let min = read_qv(p, off + 16);
    let max = read_qv(p, off + 48);
    (left, right, start, len, min, max)
}

#[inline]
fn read_qv(base: *const u8, off: usize) -> [i16; STORE_DIM] {
    let mut v = [0i16; STORE_DIM];
    for i in 0..STORE_DIM {
        v[i] = read_i16_at(base, off + i * 2);
    }
    v
}

#[inline(always)]
fn insert_best(dist: i64, label: u8, dists: &mut [i64; K], labels: &mut [u8; K]) {
    if dist >= dists[K - 1] {
        return;
    }
    let mut pos = K - 1;
    while pos > 0 && dist < dists[pos - 1] {
        dists[pos] = dists[pos - 1];
        labels[pos] = labels[pos - 1];
        pos -= 1;
    }
    dists[pos] = dist;
    labels[pos] = label;
}

#[inline(always)]
fn sum_labels(labels: &[u8; K]) -> u8 {
    let mut n = 0u8;
    for &l in labels {
        n += l;
    }
    n
}

#[inline(always)]
fn early_done(best: &[i64; K]) -> bool {
    best[K - 1] <= EARLY_DISTANCE_LIMIT
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fraud_count_avx2(idx: &IndexReader, query: &[i16; STORE_DIM]) -> u8 {
    let mut best_dists = [i64::MAX; K];
    let mut best_labels = [0u8; K];

    let key = partition_key(query);
    let primary = idx.part_by_key(key);
    if primary >= 0 {
        let (root, _len, _min, _max) = read_partition(idx, primary as usize);
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
        let (_root, _len, min, max) = read_partition(idx, p as usize);
        let lb = lower_bound_vec_avx2(query, &min, &max);
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
        let (root, _len, _min, _max) = read_partition(idx, part_idx as usize);
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
            let (left, right, start, len, _lo, _hi) = read_node(idx, current as usize);
            if left < 0 {
                if scan_leaf_avx2(idx, start, len, query, best_dists, best_labels) {
                    return true;
                }
            } else {
                let (_, _, _, _, lmin, lmax) = read_node(idx, left as usize);
                let (_, _, _, _, rmin, rmax) = read_node(idx, right as usize);
                let lb = lower_bound_vec_avx2(query, &lmin, &lmax);
                let rb = lower_bound_vec_avx2(query, &rmin, &rmax);
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

    let mut q_broadcast = [_mm256_setzero_si256(); DIM];
    for d in 0..DIM {
        q_broadcast[d] = _mm256_set1_epi32(query[d] as i32);
    }

    let total_len = len as usize;
    for b in 0..blocks {
        let block_idx = start_block as usize + b;
        let labels_base = block_idx * LANES;
        let block_off_i16 = block_idx * DIM * LANES;
        let dists = distance_block8(vectors_ptr, block_off_i16, &q_broadcast);
        let lane_count = (total_len - b * LANES).min(LANES);
        for (lane, &d) in dists.iter().enumerate().take(lane_count) {
            let label = *labels_ptr.add(labels_base + lane);
            insert_best(d, label, best_dists, best_labels);
        }
        if early_done(best_dists) {
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_block8(
    vectors: *const i16,
    block_off_i16: usize,
    q: &[std::arch::x86_64::__m256i; DIM],
) -> [i64; LANES] {
    use std::arch::x86_64::*;

    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let base = vectors.add(block_off_i16);
    for (d, qd) in q.iter().enumerate().take(DIM) {
        let packed = _mm_loadu_si128(base.add(d * LANES) as *const __m128i);
        let values = _mm256_cvtepi16_epi32(packed);
        let diff = _mm256_sub_epi32(values, *qd);
        let sq = _mm256_mullo_epi32(diff, diff);
        let sq_lo = _mm256_castsi256_si128(sq);
        let sq_hi = _mm256_extracti128_si256(sq, 1);
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(sq_lo));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(sq_hi));
    }
    let mut out = [0i64; LANES];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc_lo);
    _mm256_storeu_si256(out.as_mut_ptr().add(4) as *mut __m256i, acc_hi);
    out
}

fn fraud_count_scalar(idx: &IndexReader, query: &[i16; STORE_DIM]) -> u8 {
    let mut best_dists = [i64::MAX; K];
    let mut best_labels = [0u8; K];

    let key = partition_key(query);
    let primary = idx.part_by_key(key);
    if primary >= 0 {
        let (root, _len, _min, _max) = read_partition(idx, primary as usize);
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
        let (_root, _len, min, max) = read_partition(idx, p as usize);
        let lb = lower_bound_vec(query, &min, &max);
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
        let (root, _len, _min, _max) = read_partition(idx, part_idx as usize);
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
            let (left, right, start, len, _lo, _hi) = read_node(idx, current as usize);
            if left < 0 {
                if scan_leaf_scalar(idx, start, len, query, best_dists, best_labels) {
                    return true;
                }
            } else {
                let (_, _, _, _, lmin, lmax) = read_node(idx, left as usize);
                let (_, _, _, _, rmin, rmax) = read_node(idx, right as usize);
                let lb = lower_bound_vec(query, &lmin, &lmax);
                let rb = lower_bound_vec(query, &rmin, &rmax);
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
                let v = unsafe { *vectors_ptr.add(block_off + d * LANES + lane) };
                let diff = v as i64 - query[d] as i64;
                dist += diff * diff;
            }
            let label = unsafe { *labels_ptr.add(block_idx * LANES + lane) };
            insert_best(dist, label, best_dists, best_labels);
        }
        if early_done(best_dists) {
            return true;
        }
    }
    false
}

// ------------------------------ Builder ------------------------------

#[cfg(feature = "builder")]
pub mod build {
    use super::*;
    use crate::consts::{DEFAULT_MCC_RISK, MCC_RISK};
    use crate::quantize;
    use std::io::Write;
    use std::path::Path;

    pub struct Builder {
        vectors: Vec<[i16; STORE_DIM]>,
        labels: Vec<u8>,
    }

    #[derive(Clone, Copy)]
    struct BuildNode {
        left: i32,
        right: i32,
        start: i32,
        len: i32,
        min: [i16; STORE_DIM],
        max: [i16; STORE_DIM],
    }

    #[derive(Clone, Copy)]
    struct PartitionRoot {
        key: u32,
        root: i32,
    }

    impl Builder {
        pub fn new() -> Self {
            Builder {
                vectors: Vec::new(),
                labels: Vec::new(),
            }
        }

        pub fn add(&mut self, v: [i16; STORE_DIM], label: u8) {
            self.vectors.push(v);
            self.labels.push(label);
        }

        pub fn write_to(&self, path: &Path) -> std::io::Result<()> {
            assert!(!self.vectors.is_empty());
            assert_eq!(self.vectors.len(), self.labels.len());

            let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 256];
            for (i, v) in self.vectors.iter().enumerate() {
                buckets[partition_key(v) as usize].push(i);
            }

            let mut nodes: Vec<BuildNode> = Vec::new();
            let mut blocks: Vec<([i16; STORE_DIM], u8)> =
                Vec::with_capacity(self.vectors.len() + LANES);
            let mut roots: Vec<PartitionRoot> = Vec::new();

            for (key, indices) in buckets.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let root = build_tree(
                    &self.vectors,
                    &self.labels,
                    indices,
                    DEFAULT_LEAF_SIZE,
                    &mut blocks,
                    &mut nodes,
                );
                roots.push(PartitionRoot {
                    key: key as u32,
                    root: root as i32,
                });
            }

            assert_eq!(blocks.len() % LANES, 0);
            let block_count = blocks.len() / LANES;
            let partitions_off = HEADER_SIZE;
            let nodes_off = partitions_off + roots.len() * PART_SIZE;
            let vectors_off = nodes_off + nodes.len() * NODE_SIZE;
            let labels_off = vectors_off + block_count * BLOCK_BYTES;
            let mcc_table_off = labels_off + block_count * LANES;
            let total = mcc_table_off + MCC_TABLE_SIZE * 2;
            let mut out = vec![0u8; total];

            let header = Header {
                magic: MAGIC,
                version: VERSION,
                scale: SCALE as u32,
                dim: DIM as u32,
                store_dim: STORE_DIM as u32,
                n_points: self.vectors.len() as u32,
                part_count: roots.len() as u32,
                node_count: nodes.len() as u32,
                block_count: block_count as u32,
                mcc_table_offset: mcc_table_off as u32,
                _pad: [0; 20],
            };
            let header_bytes = unsafe {
                std::slice::from_raw_parts(&header as *const Header as *const u8, HEADER_SIZE)
            };
            out[..HEADER_SIZE].copy_from_slice(header_bytes);

            for (i, r) in roots.iter().enumerate() {
                let off = partitions_off + i * PART_SIZE;
                let n = &nodes[r.root as usize];
                out[off..off + 4].copy_from_slice(&r.key.to_le_bytes());
                out[off + 4..off + 8].copy_from_slice(&r.root.to_le_bytes());
                out[off + 8..off + 12].copy_from_slice(&n.len.to_le_bytes());
                write_qv(&mut out[off + 12..off + 44], &n.min);
                write_qv(&mut out[off + 44..off + 76], &n.max);
            }

            for (i, n) in nodes.iter().enumerate() {
                let off = nodes_off + i * NODE_SIZE;
                out[off..off + 4].copy_from_slice(&n.left.to_le_bytes());
                out[off + 4..off + 8].copy_from_slice(&n.right.to_le_bytes());
                let start_block = if n.left < 0 {
                    n.start / LANES as i32
                } else {
                    n.start
                };
                out[off + 8..off + 12].copy_from_slice(&start_block.to_le_bytes());
                out[off + 12..off + 16].copy_from_slice(&n.len.to_le_bytes());
                write_qv(&mut out[off + 16..off + 48], &n.min);
                write_qv(&mut out[off + 48..off + 80], &n.max);
            }

            for b in 0..block_count {
                let block_off = vectors_off + b * BLOCK_BYTES;
                for d in 0..DIM {
                    let dim_off = block_off + d * LANES * 2;
                    for lane in 0..LANES {
                        let slot = b * LANES + lane;
                        let val = blocks[slot].0[d];
                        out[dim_off + lane * 2..dim_off + lane * 2 + 2]
                            .copy_from_slice(&val.to_le_bytes());
                    }
                }
            }

            for b in 0..block_count {
                let base = labels_off + b * LANES;
                for lane in 0..LANES {
                    out[base + lane] = blocks[b * LANES + lane].1;
                }
            }

            let mcc_table = build_mcc_table();
            for (i, &v) in mcc_table.iter().enumerate() {
                let off = mcc_table_off + i * 2;
                out[off..off + 2].copy_from_slice(&v.to_le_bytes());
            }

            let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path)?);
            f.write_all(&out)?;
            f.flush()?;
            Ok(())
        }
    }

    fn build_mcc_table() -> [i16; MCC_TABLE_SIZE] {
        let mut table = [quantize(DEFAULT_MCC_RISK); MCC_TABLE_SIZE];
        for (mcc, risk) in MCC_RISK {
            let mut code = 0u32;
            for &b in mcc.iter() {
                code = code * 10 + (b - b'0') as u32;
            }
            table[(code as usize) % MCC_TABLE_SIZE] = quantize(*risk);
        }
        table
    }

    fn write_qv(dst: &mut [u8], v: &[i16; STORE_DIM]) {
        debug_assert_eq!(dst.len(), STORE_DIM * 2);
        for i in 0..STORE_DIM {
            dst[i * 2..i * 2 + 2].copy_from_slice(&v[i].to_le_bytes());
        }
    }

    fn bounds(
        vectors: &[[i16; STORE_DIM]],
        indices: &[usize],
    ) -> ([i16; STORE_DIM], [i16; STORE_DIM]) {
        let mut lo = [i16::MAX; STORE_DIM];
        let mut hi = [i16::MIN; STORE_DIM];
        for &i in indices {
            let v = &vectors[i];
            for d in 0..STORE_DIM {
                if v[d] < lo[d] {
                    lo[d] = v[d];
                }
                if v[d] > hi[d] {
                    hi[d] = v[d];
                }
            }
        }
        (lo, hi)
    }

    fn widest_dim(lo: &[i16; STORE_DIM], hi: &[i16; STORE_DIM]) -> usize {
        let mut best = 0usize;
        let mut best_w = i32::MIN;
        for d in 0..DIM {
            let w = hi[d] as i32 - lo[d] as i32;
            if w > best_w {
                best_w = w;
                best = d;
            }
        }
        best
    }

    fn build_tree(
        vectors: &[[i16; STORE_DIM]],
        labels: &[u8],
        indices: &[usize],
        leaf_size: usize,
        blocks: &mut Vec<([i16; STORE_DIM], u8)>,
        nodes: &mut Vec<BuildNode>,
    ) -> usize {
        let (lo, hi) = bounds(vectors, indices);
        let node_idx = nodes.len();
        nodes.push(BuildNode {
            left: -1,
            right: -1,
            start: 0,
            len: indices.len() as i32,
            min: lo,
            max: hi,
        });

        if indices.len() <= leaf_size {
            let start_slot = blocks.len() as i32;
            for &i in indices {
                blocks.push((vectors[i], labels[i]));
            }
            while blocks.len() % LANES != 0 {
                blocks.push(([i16::MAX; STORE_DIM], LABEL_LEGIT));
            }
            let node = &mut nodes[node_idx];
            node.start = start_slot;
            node.len = indices.len() as i32;
            return node_idx;
        }

        let split_dim = widest_dim(&lo, &hi);
        let mut sorted = indices.to_vec();
        sorted.sort_unstable_by_key(|&i| vectors[i][split_dim]);
        let mid = sorted.len() / 2;
        let (left_idx, right_idx) = sorted.split_at(mid);

        let left = build_tree(vectors, labels, left_idx, leaf_size, blocks, nodes);
        let right = build_tree(vectors, labels, right_idx, leaf_size, blocks, nodes);

        let left_start = nodes[left].start;
        let total_len = nodes[left].len + nodes[right].len;
        let node = &mut nodes[node_idx];
        node.left = left as i32;
        node.right = right as i32;
        node.start = left_start;
        node.len = total_len;
        node_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_key_matches_expected_bits() {
        let mut v = [0i16; STORE_DIM];
        assert_eq!(partition_key(&v), 1);
        v[5] = -10000;
        assert_eq!(partition_key(&v), 0);
        v[9] = 10000;
        v[10] = 10000;
        v[11] = 10000;
        v[12] = 8000;
        v[2] = 5000;
        v[8] = 3000;
        assert_eq!(partition_key(&v), 0b1111_1110);
    }

    #[test]
    fn lower_bound_is_zero_inside_box() {
        let q = [100i16; STORE_DIM];
        let lo = [50i16; STORE_DIM];
        let hi = [200i16; STORE_DIM];
        assert_eq!(lower_bound_vec(&q, &lo, &hi), 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn lower_bound_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let q = [
            -10000, -5000, -1, 0, 1, 450, 900, 1200, 2049, 4096, 7000, 10000, 1234, -4321, 7777,
            -8888,
        ];
        let lo = [
            -9000, -6000, -10, 0, 3, 0, 1000, 1100, 1000, 4097, 6000, 8000, 1234, -5000, -1, -1,
        ];
        let hi = [
            -8000, -4000, 10, 100, 4, 100, 1100, 1150, 2000, 5000, 6500, 9000, 2000, -4000, 1, 1,
        ];
        let scalar = lower_bound_vec(&q, &lo, &hi);
        let avx2 = unsafe { lower_bound_vec_avx2(&q, &lo, &hi) };
        assert_eq!(avx2, scalar);
    }
}
