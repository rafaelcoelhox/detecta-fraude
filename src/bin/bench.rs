// Micro-benchmark dos componentes do hot path: parse, vetorização, k-NN.

use detecta_fraude::index::IndexReader;
use detecta_fraude::parse::parse_payload;
use detecta_fraude::vectorize::vectorize_q;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

const SAMPLES: &[&[u8]] = &[
    br#"{"id":"a","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}"#,
    br#"{"id":"b","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#,
    br#"{"id":"c","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}"#,
    br#"{"id":"d","transaction":{"amount":4368.82,"installments":8,"requested_at":"2026-03-17T02:04:06Z"},"customer":{"avg_amount":68.88,"tx_count_24h":18,"known_merchants":["MERC-004","MERC-015","MERC-017","MERC-007"]},"merchant":{"id":"MERC-062","mcc":"7801","avg_amount":25.55},"terminal":{"is_online":true,"card_present":false,"km_from_home":881.61},"last_transaction":{"timestamp":"2026-03-17T01:58:06Z","km_from_current":660.92}}"#,
];

fn main() {
    let args: Vec<String> = env::args().collect();
    let index_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "target/idx/index.bin".to_string()
    };
    let n_iter: usize = if args.len() > 2 {
        args[2].parse().unwrap()
    } else {
        100_000
    };

    let idx = IndexReader::open(&PathBuf::from(&index_path)).expect("open");
    eprintln!("índice: {} pontos", idx.n_points());

    // 1) Parse + vectorize
    let t = Instant::now();
    for i in 0..n_iter {
        let s = SAMPLES[i % SAMPLES.len()];
        let p = parse_payload(s).unwrap();
        let v = vectorize_q(&p);
        std::hint::black_box(v);
    }
    let parse_vec = t.elapsed();
    println!(
        "parse+vec: {} iter -> {:.2} us/iter",
        n_iter,
        parse_vec.as_nanos() as f64 / n_iter as f64 / 1000.0
    );

    // 2) KNN puro (vetor pré-construído)
    let vecs: Vec<_> = SAMPLES
        .iter()
        .map(|s| {
            let p = parse_payload(s).unwrap();
            vectorize_q(&p)
        })
        .collect();
    let t = Instant::now();
    let mut sum = 0u64;
    for i in 0..n_iter {
        let q = &vecs[i % vecs.len()];
        sum += idx.fraud_count(q) as u64;
    }
    let knn = t.elapsed();
    println!(
        "knn: {} iter -> {:.2} us/iter (sum={})",
        n_iter,
        knn.as_nanos() as f64 / n_iter as f64 / 1000.0,
        sum
    );

    // 3) Pipeline completo
    let t = Instant::now();
    let mut sum = 0u64;
    for i in 0..n_iter {
        let s = SAMPLES[i % SAMPLES.len()];
        let p = parse_payload(s).unwrap();
        let v = vectorize_q(&p);
        sum += idx.fraud_count(&v) as u64;
    }
    let full = t.elapsed();
    println!(
        "full:      {} iter -> {:.2} us/iter (sum={})",
        n_iter,
        full.as_nanos() as f64 / n_iter as f64 / 1000.0,
        sum
    );
}
