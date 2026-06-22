
use detecta_fraude::index::IndexReader;
use detecta_fraude::response::Responses;
use detecta_fraude::server::{accept_lb, create_listener, Server};
use detecta_fraude::{SCALE, STORE_DIM};
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::thread;

fn main() {
    let prefix = env::var("API_SOCKET_PREFIX").unwrap_or_else(|_| "/sockets/api1".to_string());
    let workers: usize = env::var("API_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let index_path = env::var("INDEX_PATH").unwrap_or_else(|_| "/index/index.bin".to_string());

    let index = match IndexReader::open(&PathBuf::from(&index_path)) {
        Ok(i) => Box::leak(Box::new(i)) as &'static IndexReader,
        Err(_) => process::exit(1),
    };
    let responses: &'static Responses = Box::leak(Box::new(Responses::new()));
    warm_up_index(index);

    let mut handles = Vec::with_capacity(workers);
    for w in 0..workers {
        let socket = format!("{}-w{}.sock", prefix, w);
        let prefix_clone = prefix.clone();
        let handle = thread::Builder::new()
            .name(format!("worker-{}", w))
            .spawn(move || run_worker(w, socket, prefix_clone, index, responses))
            .expect("spawn worker");
        handles.push(handle);
    }
    for h in handles {
        let _ = h.join();
    }
}

fn warm_up_index(index: &IndexReader) {
    let count = env::var("API_WARMUP_QUERIES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048);
    let mut sum = 0u8;
    for i in 0..count {
        let mut q = [0i16; STORE_DIM];
        for (d, slot) in q.iter_mut().enumerate().take(14) {
            let raw = ((i * 313 + d * 1009) % (SCALE as usize + 1)) as i16;
            *slot = raw;
        }
        if i & 3 == 0 {
            q[5] = -(SCALE as i16);
            q[6] = -(SCALE as i16);
        }
        if i & 1 != 0 {
            q[9] = SCALE as i16;
        }
        if i & 2 != 0 {
            q[10] = SCALE as i16;
        }
        if i & 4 != 0 {
            q[11] = SCALE as i16;
        }
        sum ^= index.fraud_count(&q);
    }
    std::hint::black_box(sum);
}

fn set_worker_nice() {
    let nice: i32 = match env::var("WORKER_NICE").ok().and_then(|v| v.parse().ok()) {
        Some(n) => n,
        None => return,
    };
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, nice);
    }
}

fn run_worker(
    _w: usize,
    socket: String,
    prefix: String,
    index: &'static IndexReader,
    responses: &'static Responses,
) {
    set_worker_nice();
    let listener = match create_listener(&PathBuf::from(&socket)) {
        Ok(fd) => fd,
        Err(_) => process::exit(1),
    };

    if let Some(parent) = Path::new(&socket).parent() {
        let stem = Path::new(&socket)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let _ = std::fs::write(parent.join(format!("{}.ready", stem)), b"1");
        let prefix_name = Path::new(&prefix)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let _ = std::fs::write(parent.join(format!("{}.ready", prefix_name)), b"1");
    }

    loop {
        let uds_fd = match accept_lb(listener) {
            Ok(fd) => fd,
            Err(_) => continue,
        };
        match Server::new(uds_fd, index, responses) {
            Ok(mut s) => {
                let _ = s.run();
            }
            Err(_) => detecta_fraude::server::close_fd(uds_fd),
        }
    }
}
