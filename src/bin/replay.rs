//! replay — classifica a cauda do `fraud_count` em ALGORITMICA vs AMBIENTAL.
//!
//! Roda a busca OFFLINE, isolada, num core quieto, repetida em varias passadas
//! (min por query = piso livre de ruido de scheduler/IRQ). Como o trabalho da
//! busca (nodes/leaves/blocks/parts, early_hit) e DETERMINISTICO por query, ele
//! e a explicacao da cauda algoritmica — e o `knn_stats` ja mede isso no caminho
//! kd_pair (o mesmo do index.bin de producao).
//!
//! Uso (precisa da feature knn_stats e de um indice kd_pair):
//!   cargo build --release --features knn_stats --bin replay
//!   INDEX_PATH=target/idx/index.bin \
//!     ./target/release/replay --vectors queries.txt --passes 200 --top 20
//!
//! Modos de entrada:
//!   --payloads <arq|->  : corpos JSON (mesmo caminho da API: parse + vectorize + busca)
//!   --vectors  <arq|->  : 14 floats por linha (sao quantizados como no index-builder)
//!
//! Classificador (liga com a captura eBPF do passo 1):
//!   --baseline <arq>    : distribuicao de referencia; cada query da entrada e
//!                         posicionada por percentil de trabalho/tempo contra ela.
//!   Rode a entrada = queries lentas capturadas, baseline = amostra representativa.
//!   Trabalho alto => ALGORITMICA (conserta no indice/busca).
//!   Trabalho normal => AMBIENTAL (ja explicada pelo eBPF: C-state/run-queue).

use detecta_fraude::index::{stats, IndexReader};
use detecta_fraude::parse::parse_payload;
use detecta_fraude::vectorize::vectorize_q;
use detecta_fraude::{quantize, QVec, DIM};
use std::hint::black_box;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Row {
    min_ns: u64,
    max_ns: u64,
    nodes: u32,
    leaves: u32,
    blocks: u32,
    parts: u32,
    primary: bool,
    early: bool,
    count: u8,
}

enum Mode {
    Payloads,
    Vectors,
}

fn main() {
    let mut index = std::env::var("INDEX_PATH").unwrap_or_else(|_| "target/idx/index.bin".into());
    let mut input: Option<String> = None;
    let mut baseline: Option<String> = None;
    let mut mode = Mode::Vectors;
    let mut passes: usize = 100;
    let mut top: usize = 20;
    let mut csv: Option<String> = None;

    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--index" => index = a.next().expect("--index <arq>"),
            "--payloads" => {
                mode = Mode::Payloads;
                input = Some(a.next().expect("--payloads <arq|->"));
            }
            "--vectors" => {
                mode = Mode::Vectors;
                input = Some(a.next().expect("--vectors <arq|->"));
            }
            "--baseline" => baseline = Some(a.next().expect("--baseline <arq>")),
            "--passes" => passes = a.next().and_then(|s| s.parse().ok()).expect("--passes <n>"),
            "--top" => top = a.next().and_then(|s| s.parse().ok()).expect("--top <n>"),
            "--csv" => csv = Some(a.next().expect("--csv <arq>")),
            other => {
                eprintln!("argumento desconhecido: {other}");
                std::process::exit(2);
            }
        }
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("faltou --payloads <arq|-> ou --vectors <arq|->");
        std::process::exit(2);
    });

    let idx = match IndexReader::open(&PathBuf::from(&index)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("erro abrindo indice {index}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "indice: {index}  ({} pontos, parts={}, nodes={}, blocks={})",
        idx.n_points(),
        idx.part_count(),
        idx.node_count(),
        idx.block_count()
    );

    let (queries, errs) = load(&input, &mode);
    if queries.is_empty() {
        eprintln!("nenhuma query valida (erros de parse: {errs})");
        std::process::exit(1);
    }
    eprintln!("queries: {} (erros de parse: {errs}), passadas: {passes}", queries.len());

    let rows = measure(&idx, &queries, passes);

    if let Some(bpath) = baseline {
        let (bq, _) = load(&bpath, &mode);
        let brows = measure(&idx, &bq, passes);
        report_classify(&rows, &brows);
    } else {
        report_characterize(&rows, top);
    }

    if let Some(path) = csv {
        dump_csv(&path, &rows);
        eprintln!("csv -> {path}");
    }
}

fn load(path: &str, mode: &Mode) -> (Vec<QVec>, usize) {
    let data: Box<dyn Read> = if path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("erro abrindo {path}: {e}");
            std::process::exit(1);
        }))
    };
    let mut out = Vec::new();
    let mut errs = 0usize;
    for line in BufReader::new(data).lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        match mode {
            Mode::Payloads => match parse_payload(t.as_bytes()) {
                Ok(p) => out.push(vectorize_q(&p)),
                Err(_) => errs += 1,
            },
            Mode::Vectors => match parse_vector(t) {
                Some(q) => out.push(q),
                None => errs += 1,
            },
        }
    }
    (out, errs)
}

/// 14 floats separados por virgula/espaco -> QVec, quantizado como no index-builder.
fn parse_vector(line: &str) -> Option<QVec> {
    let mut q: QVec = [0i16; detecta_fraude::STORE_DIM];
    let mut n = 0usize;
    for tok in line.split([',', ' ', '\t', ';']).filter(|s| !s.is_empty()) {
        if n >= DIM {
            break;
        }
        let v: f64 = tok.parse().ok()?;
        q[n] = quantize(v);
        n += 1;
    }
    if n == DIM {
        Some(q)
    } else {
        None
    }
}

fn measure(idx: &IndexReader, queries: &[QVec], passes: usize) -> Vec<Row> {
    let mut rows = vec![Row::default(); queries.len()];

    // 1) trabalho deterministico (uma passada com knn_stats)
    for (qi, q) in queries.iter().enumerate() {
        stats::reset();
        let c = idx.fraud_count(q);
        let s = stats::snapshot();
        let r = &mut rows[qi];
        r.count = c;
        r.nodes = s.nodes;
        r.leaves = s.leaves;
        r.blocks = s.blocks;
        r.parts = s.partitions;
        r.primary = s.primary_hit;
        r.early = s.early_hit;
        r.min_ns = u64::MAX;
        r.max_ns = 0;
    }

    // 2) tempo: varias passadas INTERCALADAS (cache realista entre repeticoes),
    //    min por query = piso algoritmico; max-min = jitter mesmo offline.
    for _ in 0..passes {
        for (qi, q) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let c = idx.fraud_count(q);
            let dt = t0.elapsed().as_nanos() as u64;
            black_box(c);
            let r = &mut rows[qi];
            if dt < r.min_ns {
                r.min_ns = dt;
            }
            if dt > r.max_ns {
                r.max_ns = dt;
            }
        }
    }
    rows
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn pct_rank(sorted: &[u64], v: u64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let below = sorted.partition_point(|&x| x < v);
    100.0 * below as f64 / sorted.len() as f64
}

fn report_characterize(rows: &[Row], top: usize) {
    let n = rows.len();
    let mut t: Vec<u64> = rows.iter().map(|r| r.min_ns).collect();
    let mut lv: Vec<u64> = rows.iter().map(|r| r.leaves as u64).collect();
    t.sort_unstable();
    lv.sort_unstable();
    let early_rate = 100.0 * rows.iter().filter(|r| r.early).count() as f64 / n as f64;
    let prim_rate = 100.0 * rows.iter().filter(|r| r.primary).count() as f64 / n as f64;

    println!("=== caracterizacao ({n} queries) ===");
    println!(
        "tempo (min por query): p50={:.2}us p90={:.2}us p99={:.2}us max={:.2}us",
        pct(&t, 50.0) as f64 / 1e3,
        pct(&t, 90.0) as f64 / 1e3,
        pct(&t, 99.0) as f64 / 1e3,
        pct(&t, 100.0) as f64 / 1e3,
    );
    println!(
        "trabalho (leaves): p50={} p90={} p99={} max={}",
        pct(&lv, 50.0),
        pct(&lv, 90.0),
        pct(&lv, 99.0),
        pct(&lv, 100.0),
    );
    println!("early_hit: {early_rate:.1}%   primary_hit: {prim_rate:.1}%");
    println!();

    // trabalho -> tempo: prova que a cauda de tempo segue a cauda de trabalho
    let (mut e_t, mut ne_t): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
    for r in rows {
        if r.early {
            e_t.push(r.min_ns)
        } else {
            ne_t.push(r.min_ns)
        }
    }
    e_t.sort_unstable();
    ne_t.sort_unstable();
    println!("tempo por early_hit (mostra trabalho -> tempo):");
    println!(
        "  early=SIM ({:>5}): p50={:.2}us p99={:.2}us",
        e_t.len(),
        pct(&e_t, 50.0) as f64 / 1e3,
        pct(&e_t, 99.0) as f64 / 1e3
    );
    println!(
        "  early=NAO ({:>5}): p50={:.2}us p99={:.2}us",
        ne_t.len(),
        pct(&ne_t, 50.0) as f64 / 1e3,
        pct(&ne_t, 99.0) as f64 / 1e3
    );
    println!();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| rows[y].min_ns.cmp(&rows[x].min_ns));
    println!("top {} mais lentas (cauda algoritmica):", top.min(n));
    println!("   min_us  jitter   leaves nodes blocks parts  early prim cnt");
    for &i in order.iter().take(top) {
        let r = &rows[i];
        println!(
            "  {:7.2} {:7.2}   {:6} {:5} {:6} {:5}   {:>3}  {:>3}  {:>2}",
            r.min_ns as f64 / 1e3,
            (r.max_ns - r.min_ns) as f64 / 1e3,
            r.leaves,
            r.nodes,
            r.blocks,
            r.parts,
            if r.early { "sim" } else { "nao" },
            if r.primary { "sim" } else { "nao" },
            r.count,
        );
    }
}

fn report_classify(suspects: &[Row], base: &[Row]) {
    let mut bt: Vec<u64> = base.iter().map(|r| r.min_ns).collect();
    let mut bl: Vec<u64> = base.iter().map(|r| r.leaves as u64).collect();
    bt.sort_unstable();
    bl.sort_unstable();

    println!(
        "=== classificacao ({} suspeitas vs baseline {} queries) ===",
        suspects.len(),
        base.len()
    );
    println!("baseline: leaves p99={}  tempo p99={:.2}us", pct(&bl, 99.0), pct(&bt, 99.0) as f64 / 1e3);
    println!();
    println!("  leaves_pctl  tempo_pctl  veredito   (leaves / min_us)");
    let mut algo = 0usize;
    for r in suspects {
        let lp = pct_rank(&bl, r.leaves as u64);
        let tp = pct_rank(&bt, r.min_ns);
        // trabalho no topo da baseline => intrinsecamente pesada
        let verdict = if lp >= 90.0 {
            algo += 1;
            "ALGORITMICA"
        } else {
            "AMBIENTAL"
        };
        println!(
            "  {:9.1}%  {:9.1}%  {:<11} ({} / {:.2})",
            lp,
            tp,
            verdict,
            r.leaves,
            r.min_ns as f64 / 1e3
        );
    }
    println!();
    let n = suspects.len();
    println!();
    println!(
        "resumo: {algo}/{n} ({:.1}%) ALGORITMICAS  (trabalho >= p90 da baseline -> conserta no indice/busca)",
        100.0 * algo as f64 / n as f64
    );
    println!(
        "        {}/{n} ({:.1}%) AMBIENTAIS    (trabalho normal -> causa no wakeup/fila; ver captura eBPF)",
        n - algo,
        100.0 * (n - algo) as f64 / n as f64
    );
}

fn dump_csv(path: &str, rows: &[Row]) {
    let mut f = std::fs::File::create(path).expect("criar csv");
    writeln!(f, "idx,min_ns,max_ns,nodes,leaves,blocks,parts,primary,early,count").unwrap();
    for (i, r) in rows.iter().enumerate() {
        writeln!(
            f,
            "{i},{},{},{},{},{},{},{},{},{}",
            r.min_ns, r.max_ns, r.nodes, r.leaves, r.blocks, r.parts,
            r.primary as u8, r.early as u8, r.count
        )
        .unwrap();
    }
}
