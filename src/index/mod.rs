use crate::{DIM, SCALE, STORE_DIM};
use std::path::Path;

mod distance;
mod format;
mod search;
pub mod stats;

#[cfg(feature = "builder")]
pub mod build;

pub use distance::{lower_bound_vec, partition_key};
pub use format::*;

use distance::read_node_meta;

#[allow(dead_code)]
enum Backing {
    File(memmap2::Mmap),
    Anon(AnonMap),
}

struct AnonMap {
    ptr: *mut u8,
    len: usize,
}

impl Drop for AnonMap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

fn want_huge() -> bool {
    cfg!(target_os = "linux") && std::env::var("INDEX_HUGE").ok().as_deref() == Some("1")
}

fn want_collapse() -> bool {
    std::env::var("INDEX_COLLAPSE").ok().as_deref() == Some("1")
}

pub struct IndexReader {
    _backing: Backing,
    base: *const u8,
    len: usize,
    partitions_off: usize,
    nodes_off: usize,
    vectors_off: usize,
    labels_off: usize,
    mcc_table_off: usize,
    header: Header,
    part_by_key: [i32; 256],
    kinds: Vec<u8>,
}

unsafe impl Send for IndexReader {}
unsafe impl Sync for IndexReader {}

impl IndexReader {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let huge = want_huge();
        let mut opts = memmap2::MmapOptions::new();
        if !huge {
            opts.populate();
        }
        let map = unsafe { opts.map(&file)? };
        let file_base = map.as_ptr();
        let len = map.len();
        if len < HEADER_SIZE {
            return Err(invalid("index too small"));
        }

        let header: Header = unsafe { std::ptr::read_unaligned(file_base as *const Header) };
        if header.magic != MAGIC || (header.version != VERSION && header.version != KD_PAIR_VERSION)
        {
            return Err(invalid("bad magic/version"));
        }
        if header.scale != SCALE as u32
            || header.dim as usize != DIM
            || header.store_dim as usize != STORE_DIM
        {
            return Err(invalid("dim/scale mismatch"));
        }

        let mut part_by_key = [-1i32; 256];
        let partitions_off = HEADER_SIZE;
        let nodes_off = partitions_off + header.part_count as usize * PART_SIZE;
        let vectors_off = nodes_off + header.node_count as usize * NODE_SIZE;
        let labels_off = vectors_off + header.block_count as usize * BLOCK_BYTES;
        let mcc_table_off = labels_off + header.block_count as usize * LANES;
        let end = mcc_table_off + MCC_TABLE_SIZE * 2;
        if end != len || header.mcc_table_offset as usize != mcc_table_off {
            return Err(invalid("index size mismatch"));
        }
        for i in 0..header.part_count as usize {
            let off = partitions_off + i * PART_SIZE;
            let key = read_u32_at(file_base, off);
            if (key as usize) < part_by_key.len() {
                part_by_key[key as usize] = i as i32;
            }
        }

        let (backing, base) = if huge {
            match unsafe { build_huge_copy(file_base, len) } {
                Ok(anon) => {
                    let base = anon.ptr as *const u8;
                    drop(map);
                    (Backing::Anon(anon), base)
                }
                Err(_) => {
                    let base = map.as_ptr();
                    (Backing::File(map), base)
                }
            }
        } else {
            let base = map.as_ptr();
            (Backing::File(map), base)
        };

        let mut idx = IndexReader {
            _backing: backing,
            base,
            len,
            partitions_off,
            nodes_off,
            vectors_off,
            labels_off,
            mcc_table_off,
            header,
            part_by_key,
            kinds: Vec::new(),
        };
        if !matches!(idx._backing, Backing::Anon(_)) {
            idx.advise();
            idx.prefetch();
            idx.lock_if_requested();
        }
        idx.kinds = idx.scan_kinds();
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
    fn kinds_ptr(&self) -> *const u8 {
        self.kinds.as_ptr()
    }

    fn scan_kinds(&self) -> Vec<u8> {
        let count = self.node_count() as usize;
        let mut out = vec![0u8; count];
        let labels = self.labels_ptr();
        for i in (0..count).rev() {
            let (left, right, start, len) = read_node_meta(self, i);
            if left < 0 {
                let total = len as usize;
                let blocks = total.div_ceil(LANES);
                let mut mask = 0u8;
                for b in 0..blocks {
                    let active = (total - b * LANES).min(LANES);
                    let off = (start as usize + b) * LANES;
                    for lane in 0..active {
                        mask |= 1u8 << unsafe { *labels.add(off + lane) };
                    }
                }
                out[i] = mask;
            } else {
                out[i] = out[left as usize] | out[right as usize];
            }
        }
        out
    }

    #[inline]
    fn is_kd_pair(&self) -> bool {
        self.header.version == KD_PAIR_VERSION
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
                if self.is_kd_pair() {
                    return unsafe { search::fraud_count_pair_avx2(self, query) };
                }
                return unsafe { search::fraud_count_exact_avx2(self, query) };
            }
        }
        search::fraud_count_scalar(self, query)
    }

    #[inline]
    pub fn fraud_count_exact(&self, query: &[i16; STORE_DIM]) -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                if self.is_kd_pair() {
                    return unsafe { search::fraud_count_pair_avx2(self, query) };
                }
                return unsafe { search::fraud_count_exact_avx2(self, query) };
            }
        }
        search::fraud_count_scalar(self, query)
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
        std::hint::black_box(acc);
    }

    #[cfg(target_os = "linux")]
    fn lock_if_requested(&self) {
        if std::env::var("INDEX_MLOCK").ok().as_deref() != Some("1") {
            return;
        }
        let _ = unsafe { libc::mlock(self.base as *const _, self.len) };
    }

    #[cfg(not(target_os = "linux"))]
    fn lock_if_requested(&self) {}
}

fn invalid(msg: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(target_os = "linux")]
unsafe fn build_huge_copy(file_base: *const u8, len: usize) -> std::io::Result<AnonMap> {
    const HPAGE: usize = 2 * 1024 * 1024;
    const COPY_CHUNK: usize = 8 * 1024 * 1024;
    let alloc_len = (len + HPAGE - 1) & !(HPAGE - 1);

    let ptr = libc::mmap(
        std::ptr::null_mut(),
        alloc_len,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    );
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    let anon = AnonMap {
        ptr: ptr as *mut u8,
        len: alloc_len,
    };

    libc::madvise(ptr, alloc_len, libc::MADV_HUGEPAGE);

    let mut off = 0usize;
    while off < len {
        let n = (len - off).min(COPY_CHUNK);
        std::ptr::copy_nonoverlapping(file_base.add(off), anon.ptr.add(off), n);
        libc::madvise(file_base.add(off) as *mut _, n, libc::MADV_DONTNEED);
        off += n;
    }

    if want_collapse() {
        const MADV_COLLAPSE: libc::c_int = 25;
        let _ = libc::madvise(ptr, alloc_len, MADV_COLLAPSE);
    }

    let _ = libc::mlock(ptr, alloc_len);
    libc::mprotect(ptr, alloc_len, libc::PROT_READ);

    Ok(anon)
}

#[cfg(not(target_os = "linux"))]
unsafe fn build_huge_copy(_file_base: *const u8, _len: usize) -> std::io::Result<AnonMap> {
    Err(invalid("huge pages only supported on linux"))
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
