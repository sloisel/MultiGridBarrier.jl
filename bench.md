# MultiGridBarrier benchmarks

## CPU AMG vs GPU AMG (0.12.0+)

Machine: Heriot-Watt DMOG cluster, single A40 GPU (48 GB), Julia 1.11.6.
These runs used the 0.12-series API; the `structured=` knob no longer exists (FEM
geometries now always carry `BlockDiag` operators), and today's equivalent of a GPU AMG
solve is `mgb_solve(assemble(amg(subdivide(fem*(), L))); device=CUDADevice)`. The
`structured-ops` column is the path that lights up `BlockColumn' * Diagonal * BlockColumn
→ BlockHessian` batched-gemm Hessian assembly. The `sparse-ops` column forced sparse
operators on the GPU, so the GPU did everything as `SpGEMM` — a control to isolate the
batched-gemm win.

### `fem2d_P2` (block size 7)

| L | n DOFs | CPU AMG | GPU sparse | GPU structured | sparse vs CPU | structured vs CPU | structured vs sparse (GPU) |
|---|---|---|---|---|---|---|---|
| 4 |    896 |   0.369s |   1.921s |   0.664s | 1/5.20× | 1/1.80× | 2.89× |
| 5 |   3584 |   1.723s |   3.859s |   1.039s | 1/2.24× | 1.66×   | 3.71× |
| 6 |  14336 |  10.522s |   9.975s |   1.851s | 1.05×   | 5.68×   | 5.39× |
| 7 |  57344 |  66.509s |  45.840s |   5.122s | 1.45×   | 12.98×  | 8.95× |

### `fem2d_P1` (block size 3)

| L | n DOFs | CPU AMG | GPU sparse | GPU structured | sparse vs CPU | structured vs CPU | structured vs sparse (GPU) |
|---|---|---|---|---|---|---|---|
| 4 |    384 |   0.125s |   0.904s |   0.888s | 1/7.23×  | 1/7.10× | 1.02× |
| 5 |   1536 |   0.352s |   4.993s |   1.276s | 1/14.18× | 1/3.62× | 3.91× |
| 6 |   6144 |   3.504s |   4.246s |   3.781s | 1/1.21×  | 1/1.08× | 1.12× |
| 7 |  24576 |  25.795s |  13.204s |  13.152s | 1.95×    | 1.96×   | 1.00× |
| 8 |  98304 | 121.477s |  60.164s |  60.045s | 2.02×    | 2.02×   | 1.00× |

Block size 3 is too small for batched gemm to outperform `SpGEMM`; the structured and
sparse GPU paths are essentially identical from L=6 onward. GPU AMG buys ~2× over CPU.

### `fem3d` (Q_k, k=3, block size 64)

4-way comparison at L=2 (only L value with interior corners on the default unit cube).
Adds geometric MG (V/HBlockDiag transfers all the way down) to the matrix.

| L | n DOFs | CPU AMG | GPU AMG | CPU geometric_mg | GPU geometric_mg |
|---|---|---|---|---|---|
| 2 | 512 | 4.580s | **0.790s** | 3.076s | **0.541s** |
| speedup vs CPU AMG | — | 1.00× | 5.80× | 1.49× | 8.47× |

Block size 64 is the sweet spot for batched gemm on the A40. GPU geometric_mg edges out
GPU AMG by ~1.5× because its V/HBlockDiag refine/coarsen avoid the `SpGEMM` step in the
restriction/prolongation cycle.

### Reproducing

Bench script: `tools/bench_cuda_vs_native.jl`. From the package root:

```
julia --project=. tools/bench_cuda_vs_native.jl
```

Driver env vars:

```
BENCH_FEM=fem2d_P2|fem2d_P1|fem3d|fem1d   # default fem2d_P2
BENCH_L=N                                  # single L
BENCH_L_MIN=A BENCH_L_MAX=B                # range
BENCH_SKIP_NATIVE=1                        # skip CPU for large L
BENCH_SRT_ONLY=1                           # GPU structured only
```

Date: 2026-05-12.

---

## Legacy: CPU structured vs unstructured (pre-0.12.0 API)

The numbers below were collected before the 0.12.0 Geometry/MultiGrid split. The
function names referenced (`geometric_fem2d_P2_solve(L=...)`, `geometric_fem3d_solve(L=...)`)
no longer exist; the equivalent today is
`mgb_solve(assemble(geometric_mg(fem2d_P2(), L)))` (block operators are always on;
there is no unstructured switch). Kept here for the CPU memory column (which the
new GPU bench doesn't measure).

Machine: Apple Silicon (darwin, aarch64), Julia 1.12.2, single-threaded.

### geometric_fem2d_P2 (block size 7×7)

`geometric_fem2d_P2_solve(L=..., p=1.0)`, unstructured (sparse) vs structured (block-diagonal).

| L | Unstructured (s) | GC % | Memory (GiB) | Structured (s) | GC % | Memory (GiB) | Speedup | Memory reduction |
|---|---|---|---|---|---|---|---|---|
| 6 | 17.2 | 38.5 | 69.8 | 7.5 | 13.1 | 25.2 | 2.29× | 2.77× |
| 7 | 60.5 | 32.2 | 259.5 | 28.8 | 13.4 | 98.2 | 2.10× | 2.64× |
| 8 | 357.3 | 22.0 | 1809.8 | 171.3 | 6.2 | 644.2 | 2.09× | 2.81× |

### geometric_fem3d (block size 64×64, k=3)

`geometric_fem3d_solve(L=..., p=1.0)`, unstructured (sparse) vs structured (block-diagonal).

| L | Unstructured (s) | GC % | Memory (GiB) | Structured (s) | GC % | Memory (GiB) | Speedup | Memory reduction |
|---|---|---|---|---|---|---|---|---|
| 3 | 8.3 | 39.1 | 34.8 | 9.2 | 15.5 | 19.7 | 0.90× | 1.77× |
| 4 | 138.2 | 25.6 | 705.6 | 159.7 | 5.8 | 359.2 | 0.86× | 1.97× |

Date: 2026-02-09
