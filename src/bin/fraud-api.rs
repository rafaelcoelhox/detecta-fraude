
use detecta_fraude::index::IndexReader;
use detecta_fraude::response::Responses;
use detecta_fraude::server::{accept_lb, create_listener, Server};
use detecta_fraude::SCALE;
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
        Err(e) => {
            eprintln!("[api] erro abrindo índice {}: {}", index_path, e);
            process::exit(1);
        }
    };
    let responses: &'static Responses = Box::leak(Box::new(Responses::new()));
    warm_up_index(index);

    eprintln!(
        "[api] índice carregado: {} pontos. workers: {} prefix: {}",
        index.n_points(),
        workers,
        prefix
    );

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

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[inline]
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

#[inline]
fn apply_jitter(v: i16, state: &mut u64, span: u64, jitter: i16, smax: i16) -> i16 {
    let j = (lcg(state) % span) as i64 - jitter as i64;
    (v as i64 + j).clamp(-(smax as i64), smax as i64) as i16
}

// Aquece com pontos reais do índice (mesma distribuição que o teste consulta),
// não com vetores sintéticos. Cada ponto recebe jitter nas dimensões contínuas
// para a query cair perto de um ponto sem coincidir — assim a maioria das
// queries percorre a busca completa (caminho caro que domina a cauda) em vez de
// disparar early_hit, reproduzindo a mistura de caminhos do tráfego real.
// API_WARMUP_JITTER controla a amplitude (em unidades quantizadas; SCALE=10000).
fn warm_up_index(index: &IndexReader) {
    let count = env_usize("API_WARMUP_QUERIES", 2048);
    let jitter = env_usize("API_WARMUP_JITTER", 120) as i16;
    let n = index.n_points();
    if n == 0 || count == 0 {
        return;
    }
    let smax: i16 = SCALE as i16;
    // Dims contínuas; categóricas (9,10,11) ficam de fora p/ não gerar valores
    // impossíveis. As temporais (5,6) só recebem jitter quando não são a
    // sentinela -SCALE ("sem última transação").
    const CONT: [usize; 9] = [0, 1, 2, 3, 4, 7, 8, 12, 13];
    let span = (2 * jitter as i64 + 1) as u64;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut sum = 0u8;
    for _ in 0..count {
        let idx = (lcg(&mut state) >> 33) as u32 % n;
        let mut q = index.point(idx);
        if jitter > 0 {
            for &d in &CONT {
                q[d] = apply_jitter(q[d], &mut state, span, jitter, smax);
            }
            if q[5] != -smax {
                q[5] = apply_jitter(q[5], &mut state, span, jitter, smax);
            }
            if q[6] != -smax {
                q[6] = apply_jitter(q[6], &mut state, span, jitter, smax);
            }
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
    w: usize,
    socket: String,
    prefix: String,
    index: &'static IndexReader,
    responses: &'static Responses,
) {
    set_worker_nice();
    let listener = match create_listener(&PathBuf::from(&socket)) {
        Ok(fd) => fd,
        Err(e) => {
            eprintln!("[api-w{}] erro criando UDS {}: {}", w, socket, e);
            process::exit(1);
        }
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
            Err(e) => {
                eprintln!("[api-w{}] accept_lb erro: {}", w, e);
                continue;
            }
        };
        eprintln!("[api-w{}] LB conectado (fd={})", w, uds_fd);
        match Server::new(uds_fd, index, responses) {
            Ok(mut s) => {
                if let Err(e) = s.run() {
                    eprintln!("[api-w{}] server.run erro: {}", w, e);
                }
            }
            Err(e) => {
                eprintln!("[api-w{}] Server::new erro: {}", w, e);
                detecta_fraude::server::close_fd(uds_fd);
            }
        }
    }
}
