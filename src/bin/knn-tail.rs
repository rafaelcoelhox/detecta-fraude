
#[cfg(all(feature = "builder", feature = "knn_stats"))]
fn main() {
    use detecta_fraude::index::{stats, IndexReader};
    use detecta_fraude::parse::parse_payload;
    use detecta_fraude::vectorize::vectorize_q;
    use detecta_fraude::QVec;
    use serde_json::Value;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    fn summary(name: &str, unit: &str, scale: f64, xs: &[u64]) {
        if xs.is_empty() {
            println!("{name}: (vazio)");
            return;
        }
        let mut s = xs.to_vec();
        s.sort_unstable();
        let n = s.len();
        let mean = s.iter().sum::<u64>() as f64 / n as f64;
        let p = |q: f64| s[(((q / 100.0) * (n - 1) as f64).round() as usize).min(n - 1)] as f64;
        println!(
            "{:<12}{:<3} mean={:>10.2}  p50={:>9.2}  p90={:>9.2}  p99={:>9.2}  p99.9={:>9.2}  max={:>9.2}",
            name,
            unit,
            mean * scale,
            p(50.0) * scale,
            p(90.0) * scale,
            p(99.0) * scale,
            p(99.9) * scale,
            p(100.0) * scale,
        );
    }

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("uso: {} <index.bin> <test-data.json>", args[0]);
        std::process::exit(2);
    }
    let idx = IndexReader::open(&PathBuf::from(&args[1])).expect("abrir índice");
    eprintln!("índice: {} pontos", idx.n_points());

    let raw = fs::read(&args[2]).expect("ler test-data");
    let v: Value = serde_json::from_slice(&raw).expect("parse test-data");
    let entries = v["entries"].as_array().expect("entries");

    let mut queries: Vec<(QVec, bool)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let req = serde_json::to_vec(&entry["request"]).unwrap();
        let expected_approved = entry["expected_approved"].as_bool().unwrap();
        let p = parse_payload(&req).expect("parse payload");
        queries.push((vectorize_q(&p), expected_approved));
    }

    for (q, _) in &queries {
        std::hint::black_box(idx.fraud_count(q));
    }

    let n = queries.len();
    let mut times = Vec::with_capacity(n);
    let mut parts = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);
    let mut leaves = Vec::with_capacity(n);
    let mut blocks = Vec::with_capacity(n);
    let mut primary_hits = 0u64;
    let mut early_hits = 0u64;
    let mut full_scans = 0u64;
    let mut errors = 0u64;

    struct Rec {
        ns: u64,
        st: stats::QueryStats,
        expected_approved: bool,
        approved: bool,
    }
    let mut recs: Vec<Rec> = Vec::with_capacity(n);

    for (q, expected_approved) in &queries {
        stats::reset();
        let t0 = Instant::now();
        let fc = idx.fraud_count(q) as u32;
        let ns = t0.elapsed().as_nanos() as u64;
        let st = stats::snapshot();
        let approved = (fc as f32 / 5.0) < 0.6;

        times.push(ns);
        parts.push(st.partitions as u64);
        nodes.push(st.nodes as u64);
        leaves.push(st.leaves as u64);
        blocks.push(st.blocks as u64);
        if st.primary_hit {
            primary_hits += 1;
        } else if st.early_hit {
            early_hits += 1;
        } else {
            full_scans += 1;
        }
        if approved != *expected_approved {
            errors += 1;
        }
        recs.push(Rec {
            ns,
            st,
            expected_approved: *expected_approved,
            approved,
        });
    }

    println!("\n=== distribuição por query (N={n}) ===");
    summary("knn_latency", "us", 1.0 / 1000.0, &times);
    summary("partitions", "", 1.0, &parts);
    summary("nodes", "", 1.0, &nodes);
    summary("leaves", "", 1.0, &leaves);
    summary("blocks", "", 1.0, &blocks);

    let pc = |x: u64| x as f64 / n as f64 * 100.0;
    println!("\n=== tipo de resolução ===");
    println!(
        "primary_hit (só partição primária):  {primary_hits:>7} ({:.2}%)",
        pc(primary_hits)
    );
    println!(
        "early_hit   (parou cedo, com probe):  {early_hits:>7} ({:.2}%)",
        pc(early_hits)
    );
    println!(
        "full_scan   (sondou tudo, sem early): {full_scans:>7} ({:.2}%)",
        pc(full_scans)
    );
    println!("erros de detecção (FP+FN):            {errors:>7}");

    recs.sort_unstable_by(|a, b| b.ns.cmp(&a.ns));
    println!("\n=== 20 queries mais lentas ===");
    println!(
        "{:>10}  {:>5}  {:>7}  {:>7}  {:>7}  {:>4}  {:>5}  exp/got",
        "us", "parts", "nodes", "leaves", "blocks", "prim", "early"
    );
    for w in recs.iter().take(20) {
        println!(
            "{:>10.2}  {:>5}  {:>7}  {:>7}  {:>7}  {:>4}  {:>5}  {}/{}",
            w.ns as f64 / 1000.0,
            w.st.partitions,
            w.st.nodes,
            w.st.leaves,
            w.st.blocks,
            w.st.primary_hit,
            w.st.early_hit,
            w.expected_approved,
            w.approved,
        );
    }
}

#[cfg(not(all(feature = "builder", feature = "knn_stats")))]
fn main() {
    eprintln!("knn-tail requer as features 'builder' e 'knn_stats'");
    std::process::exit(1);
}
