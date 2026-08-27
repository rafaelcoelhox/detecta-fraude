# detecta-fraude — Rinha de Backend 2026

Winning solution for the [Rinha de Backend 2026](https://rinhadebackend.com.br), built for high-performance fraud detection using vector similarity search.

## Overview

`detecta-fraude` transforms transaction data into compact numerical vectors and classifies transactions by searching for similar examples in a custom vector index.

Each request is converted into a **14-dimensional feature vector** containing information such as transaction amount, customer behavior, time, location, merchant risk, card presence, and recent transaction history.

The resulting vector is searched against a custom **partitioned k-d tree**, using the **5 nearest neighbors** to determine the fraud classification.

The entire system was designed around low latency, predictable memory access, and minimal runtime overhead.

## Architecture

```text
                    ┌──────────────┐
                    │    Client    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ C Load       │
                    │ Balancer     │
                    └──────┬───────┘
                           │ SCM_RIGHTS
                    ┌──────▼───────┐
                    │ Rust Workers │
                    │    epoll     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Vectorizer   │
                    └──────┬───────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Partitioned k-d tree│
                │     AVX2 search     │
                └─────────┬───────────┘
                          │
                          ▼
                   Fraud decision
```

## Performance-oriented design

* **Rust + C** for low-level control and predictable performance
* Custom **partitioned k-d tree** for nearest-neighbor search
* **AVX2 SIMD** instructions for parallel distance calculations
* Quantized integer vectors to reduce memory and computation costs
* **`epoll`**-based event handling
* **`SCM_RIGHTS`** for passing accepted socket file descriptors between processes
* Dedicated C load balancer distributing connections across Rust workers
* Pre-built in-memory index optimized for read-heavy workloads

## Stack

`Rust` · `C` · `k-d tree` · `k-NN` · `AVX2` · `epoll` · `SCM_RIGHTS` · `Docker`

## Contact

[coelho38r@proton.me](mailto:coelho38r@proton.me)

## License

This project is licensed under the [MIT License](LICENSE).
