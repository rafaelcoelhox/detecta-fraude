use super::format::pair_offset;
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
        write_kd_pair_to(&self.vectors, &self.labels, path)
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

fn write_kd_pair_to(
    vectors: &[[i16; STORE_DIM]],
    labels: &[u8],
    path: &Path,
) -> std::io::Result<()> {
    let leaf_size = kd_leaf_size();
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for (i, v) in vectors.iter().enumerate() {
        buckets[partition_key(v) as usize].push(i);
    }

    let mut nodes: Vec<BuildNode> = Vec::new();
    let mut blocks: Vec<([i16; STORE_DIM], u8)> = Vec::with_capacity(vectors.len() + LANES);
    let mut roots: Vec<PartitionRoot> = Vec::new();

    for (key, indices) in buckets.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let root = build_tree(vectors, labels, indices, leaf_size, &mut blocks, &mut nodes);
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
        version: KD_PAIR_VERSION,
        scale: SCALE as u32,
        dim: DIM as u32,
        store_dim: STORE_DIM as u32,
        n_points: vectors.len() as u32,
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
            for lane in 0..LANES {
                let slot = b * LANES + lane;
                let dst = block_off + pair_offset(d, lane) * 2;
                out[dst..dst + 2].copy_from_slice(&blocks[slot].0[d].to_le_bytes());
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

fn kd_leaf_size() -> usize {
    std::env::var("KD_LEAF_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| (LANES..=1024).contains(&v))
        .unwrap_or(DEFAULT_LEAF_SIZE)
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
