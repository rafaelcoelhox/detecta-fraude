// Variáveis de ambiente:
//   API_SOCKET_PREFIX  — prefixo do socket por worker. Final é "<prefix>-wN.sock".
//                         Default: "/sockets/api1"
//   API_WORKERS        — quantidade de workers (default 2)
//   INDEX_PATH         — caminho do índice (default /index/index.bin)

use detecta_fraude::index::IndexReader;
use detecta_fraude::response::Responses;
use detecta_fraude::server::{accept_lb, create_listener, Server};
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

fn run_worker(
    w: usize,
    socket: String,
    prefix: String,
    index: &'static IndexReader,
    responses: &'static Responses,
) {
    let listener = match create_listener(&PathBuf::from(&socket)) {
        Ok(fd) => fd,
        Err(e) => {
            eprintln!("[api-w{}] erro criando UDS {}: {}", w, socket, e);
            process::exit(1);
        }
    };

    // Marca o socket pronto para o LB conectar.
    if let Some(parent) = Path::new(&socket).parent() {
        let stem = Path::new(&socket)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let _ = std::fs::write(parent.join(format!("{}.ready", stem)), b"1");
        // Marker geral por prefix (último worker escreve).
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
