# detecta-fraude — Rinha de Backend 2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), o desafio de detecção de fraude por busca vetorial.

## Arquitetura

```
   k6  --tcp 9999-->  lb (C, FD passing via SCM_RIGHTS)
                          |
                          +--> api1 (Rust, 2 workers epoll, mmap index)
                          +--> api2 (Rust, 2 workers epoll, mmap index)
```

- **`lb`** — load balancer minimalista em C. Abre TCP em `:9999`, `accept4`, alterna round-robin entre 4 Unix sockets (2 por API, um por worker) e entrega o FD via `SCM_RIGHTS`. Nunca lê bytes da conexão, nunca inspeciona payload.
- **`api1` / `api2`** — Rust com 2 workers (threads independentes) cada. Cada worker tem seu próprio epoll, recebe FDs pelo seu UDS dedicado e serve HTTP/1.1 com respostas pré-renderizadas. Índice carregado via `mmap` (`MAP_POPULATE`).
- **`index.bin`** — índice k-NN particionado sobre os 3 milhões de vetores quantizados em `i16` com escala 10 000. Cada partição tem uma KD-tree com bounding boxes por nó; a busca poda subárvores por lower-bound e varre folhas em layout SoA de 8 vetores com AVX2. O índice também usa parada antecipada quando os 5 vizinhos já estão suficientemente próximos.

## Decisão de detecção

Implementa exatamente a especificação oficial:

1. Vetoriza a transação em 14 dimensões, conforme [REGRAS_DE_DETECCAO.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/REGRAS_DE_DETECCAO.md).
2. Busca os 5 vizinhos mais próximos no índice particionado, com poda por bounding box e varredura AVX2 nas folhas.
3. `fraud_score = fraudes_no_top5 / 5` e `approved = fraud_score < 0.6`.

Sem lookup por payload, sem heurísticas derivadas de `test/test-data.json`, sem threshold customizado. O `test-data.json` é usado apenas para regressão local via `eval-preview`.

## Performance medida

- Parse JSON + vetorização: **~0.44 µs**
- KNN puro (índice mmap, AVX2): **~0.42 µs** nos samples do `bench`
- Pipeline completo no `bench`: **~0.92 µs**
- k6 oficial localmente: **final_score 6000**, **p99 0.39 ms**, **0 FP / 0 FN / 0 HTTP errors**

## Layout do projeto

```
.
├── Cargo.toml
├── Dockerfile.api
├── Dockerfile.lb
├── docker-compose.yml
├── info.json
├── native/fd-lb.c
├── resources/
└── src/
    ├── lib.rs
    ├── consts.rs
    ├── time.rs
    ├── parse.rs
    ├── vectorize.rs
    ├── index.rs
    ├── response.rs
    ├── server.rs
    └── bin/
        ├── fraud-api.rs
        ├── index-builder.rs
        ├── eval-preview.rs
        └── bench.rs
```

## Como rodar localmente

```bash
cp /caminho/para/rinha-de-backend-2026/resources/references.json.gz resources/

docker compose build

docker compose up -d

cd /caminho/para/rinha-de-backend-2026/test
docker compose --profile smoke up
docker compose --profile test up
```

## Testes unitários e regressão local

```bash
RUSTFLAGS="-C target-cpu=haswell" cargo test --release --features builder

RUSTFLAGS="-C target-cpu=haswell" cargo build --release --features builder \
    --bin index-builder --bin eval-preview --bin bench
./target/release/index-builder resources/references.json.gz target/idx/index.bin
./target/release/bench target/idx/index.bin 50000
./target/release/eval-preview target/idx/index.bin /caminho/para/test-data.json
```

`bench` reporta latência por etapa do hot path; `eval-preview` reporta `tp/tn/fp/fn` e latência média/máxima da parte CPU (parse + vetorização + k-NN), sem envolver a stack HTTP.

## Submissão

Branch `main` traz o código completo. A branch `submission` deve conter apenas `docker-compose.yml`, `info.json` e o restante necessário para o avaliador subir a stack — sem `src/`, `native/`, `Cargo.toml` etc.

Licença: MIT.
