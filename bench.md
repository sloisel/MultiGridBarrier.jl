# Structured Block Matrices Benchmark

Machine: Apple Silicon (darwin, aarch64), Julia 1.12.2, single-threaded.

## fem2d (block size 7x7)

`fem2d_solve(L=..., p=1.0)`, unstructured (sparse) vs structured (block-diagonal).

| L | Unstructured (s) | GC % | Memory (GiB) | Structured (s) | GC % | Memory (GiB) | Speedup | Memory reduction |
|---|---|---|---|---|---|---|---|---|
| 6 | 17.2 | 38.5 | 69.8 | 7.5 | 13.1 | 25.2 | 2.29x | 2.77x |
| 7 | 60.5 | 32.2 | 259.5 | 28.8 | 13.4 | 98.2 | 2.10x | 2.64x |
| 8 | 357.3 | 22.0 | 1809.8 | 171.3 | 6.2 | 644.2 | 2.09x | 2.81x |

## fem3d (block size 64x64, k=3)

`fem3d_solve(L=..., p=1.0)`, unstructured (sparse) vs structured (block-diagonal).

| L | Unstructured (s) | GC % | Memory (GiB) | Structured (s) | GC % | Memory (GiB) | Speedup | Memory reduction |
|---|---|---|---|---|---|---|---|---|
| 3 | 8.3 | 39.1 | 34.8 | 9.2 | 15.5 | 19.7 | 0.90x | 1.77x |
| 4 | 138.2 | 25.6 | 705.6 | 159.7 | 5.8 | 359.2 | 0.86x | 1.97x |

Date: 2026-02-09
