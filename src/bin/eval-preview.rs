#[cfg(feature = "builder")]
fn main() {
    use detecta_fraude::index::IndexReader;
    use detecta_fraude::parse::parse_payload;
    use detecta_fraude::vectorize::vectorize_q;
    use serde_json::Value;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("uso: {} <index.bin> <test-data.json>", args[0]);
        std::process::exit(2);
    }
    let index_path = PathBuf::from(&args[1]);
    let test_path = PathBuf::from(&args[2]);

    let idx = IndexReader::open(&index_path).expect("abrir índice");
    eprintln!("índice: {} pontos", idx.n_points());

    let raw = fs::read(&test_path).expect("ler test-data");
    let v: Value = serde_json::from_slice(&raw).expect("parse test-data");
    let entries = v["entries"].as_array().expect("entries");

    let mut tp = 0u64;
    let mut tn = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    let mut total_ns = 0u128;
    let mut max_ns = 0u128;
    let total = entries.len();
    for entry in entries {
        let req_json = serde_json::to_vec(&entry["request"]).unwrap();
        let expected_approved = entry["expected_approved"].as_bool().unwrap();
        let t0 = Instant::now();
        let payload = parse_payload(&req_json).expect("parse payload");
        let q = vectorize_q(&payload);
        let fraud_count = idx.fraud_count(&q) as u32;
        let score = fraud_count as f32 / 5.0;
        let approved = score < 0.6;
        let dt = t0.elapsed().as_nanos();
        total_ns += dt;
        if dt > max_ns {
            max_ns = dt;
        }
        if approved == expected_approved {
            if approved {
                tn += 1;
            } else {
                tp += 1;
            }
        } else if approved {
            fn_ += 1;
        } else {
            fp += 1;
        }
    }
    let n = total as u64;
    let failures = fp + fn_;
    println!(
        "total={} tp={} tn={} fp={} fn={} failure_rate={:.3}% avg={:.2}us max={:.2}us",
        n,
        tp,
        tn,
        fp,
        fn_,
        (failures as f64) / (n as f64) * 100.0,
        (total_ns as f64 / n as f64) / 1000.0,
        max_ns as f64 / 1000.0,
    );
}

#[cfg(not(feature = "builder"))]
fn main() {
    eprintln!("eval-preview requires feature 'builder'");
    std::process::exit(1);
}
