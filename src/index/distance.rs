use super::*;
use crate::{DIM, K, STORE_DIM};

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

#[inline(always)]
fn lower_bound_ptr_scalar(q: &[i16; STORE_DIM], min: *const i16, max: *const i16) -> i64 {
    let mut acc = 0i64;
    let mut d = 0usize;
    while d < DIM {
        let lo = unsafe { *min.add(d) };
        let hi = unsafe { *max.add(d) };
        acc += lower_bound_dim(q[d], lo, hi);
        d += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lower_bound_ptr_avx2(q: &[i16; STORE_DIM], min: *const i16, max: *const i16) -> i64 {
    use std::arch::x86_64::*;

    let qv = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let mn = _mm256_loadu_si256(min as *const __m256i);
    let mx = _mm256_loadu_si256(max as *const __m256i);
    let zero = _mm256_setzero_si256();
    let below = _mm256_max_epi16(_mm256_sub_epi16(mn, qv), zero);
    let above = _mm256_max_epi16(_mm256_sub_epi16(qv, mx), zero);
    let gap = _mm256_max_epi16(below, above);
    let sq_pairs = _mm256_madd_epi16(gap, gap);
    let mut vals = [0i32; LANES];
    _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, sq_pairs);
    vals[0] as i64
        + vals[1] as i64
        + vals[2] as i64
        + vals[3] as i64
        + vals[4] as i64
        + vals[5] as i64
        + vals[6] as i64
        + vals[7] as i64
}

#[inline]
pub(crate) fn read_partition_meta(idx: &IndexReader, part_idx: usize) -> (i32, i32) {
    let p = idx.partitions_ptr();
    let off = part_idx * PART_SIZE;
    let root = read_i32_at(p, off + 4);
    let len = read_i32_at(p, off + 8);
    (root, len)
}

#[inline]
pub(crate) fn read_node_meta(idx: &IndexReader, node_idx: usize) -> (i32, i32, i32, i32) {
    let p = idx.nodes_ptr();
    let off = node_idx * NODE_SIZE;
    let left = read_i32_at(p, off);
    let right = read_i32_at(p, off + 4);
    let start = read_i32_at(p, off + 8);
    let len = read_i32_at(p, off + 12);
    (left, right, start, len)
}

#[inline(always)]
fn partition_bounds_ptr(idx: &IndexReader, part_idx: usize) -> (*const i16, *const i16) {
    let p = idx.partitions_ptr();
    let off = part_idx * PART_SIZE;
    unsafe { (p.add(off + 12) as *const i16, p.add(off + 44) as *const i16) }
}

#[inline(always)]
fn node_bounds_ptr(idx: &IndexReader, node_idx: usize) -> (*const i16, *const i16) {
    let p = idx.nodes_ptr();
    let off = node_idx * NODE_SIZE;
    unsafe { (p.add(off + 16) as *const i16, p.add(off + 48) as *const i16) }
}

#[inline(always)]
pub(crate) fn lower_bound_partition_scalar(
    idx: &IndexReader,
    part_idx: usize,
    q: &[i16; STORE_DIM],
) -> i64 {
    let (min, max) = partition_bounds_ptr(idx, part_idx);
    lower_bound_ptr_scalar(q, min, max)
}

#[inline(always)]
pub(crate) fn lower_bound_node_scalar(idx: &IndexReader, node_idx: usize, q: &[i16; STORE_DIM]) -> i64 {
    let (min, max) = node_bounds_ptr(idx, node_idx);
    lower_bound_ptr_scalar(q, min, max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn lower_bound_partition_avx2(
    idx: &IndexReader,
    part_idx: usize,
    q: &[i16; STORE_DIM],
) -> i64 {
    let (min, max) = partition_bounds_ptr(idx, part_idx);
    lower_bound_ptr_avx2(q, min, max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn lower_bound_node_avx2(idx: &IndexReader, node_idx: usize, q: &[i16; STORE_DIM]) -> i64 {
    let (min, max) = node_bounds_ptr(idx, node_idx);
    lower_bound_ptr_avx2(q, min, max)
}

#[inline(always)]
pub(crate) fn insert_best(dist: i64, label: u8, dists: &mut [i64; K], labels: &mut [u8; K]) {
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
pub(crate) fn sum_labels(labels: &[u8; K]) -> u8 {
    let mut n = 0u8;
    for &l in labels {
        n += l;
    }
    n
}

#[inline(always)]
pub(crate) fn early_done(best: &[i64; K]) -> bool {
    best[K - 1] <= EARLY_DISTANCE_LIMIT
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn distance_pair_block8(
    vectors: *const i16,
    block_off_i16: usize,
    q_pairs: &[std::arch::x86_64::__m256i; DIM_PAIRS],
) -> [i64; LANES] {
    use std::arch::x86_64::*;

    let base = vectors.add(block_off_i16);
    let mut acc = _mm256_setzero_si256();
    for p in 0..DIM_PAIRS {
        let packed = _mm256_loadu_si256(base.add(p * LANES * 2) as *const __m256i);
        let diff = _mm256_sub_epi16(q_pairs[p], packed);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
    }

    let mut vals = [0i32; LANES];
    _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, acc);
    [
        vals[0] as i64,
        vals[1] as i64,
        vals[2] as i64,
        vals[3] as i64,
        vals[4] as i64,
        vals[5] as i64,
        vals[6] as i64,
        vals[7] as i64,
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn query_pairs_avx2(query: &[i16; STORE_DIM]) -> [std::arch::x86_64::__m256i; DIM_PAIRS] {
    use std::arch::x86_64::*;

    let mut q_pairs = [_mm256_setzero_si256(); DIM_PAIRS];
    for p in 0..DIM_PAIRS {
        let lo = query[p * 2] as u16 as u32;
        let hi = query[p * 2 + 1] as u16 as u32;
        q_pairs[p] = _mm256_set1_epi32((lo | (hi << 16)) as i32);
    }
    q_pairs
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn distance_block8(
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
}
