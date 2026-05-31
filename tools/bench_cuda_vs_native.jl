#!/usr/bin/env julia
#
# Benchmark: CPU vs CUDA for the structured AMG solve.
#
# Assembles an `MGBProblem` once on the CPU, then solves it on `CPUDevice` and on
# `CUDADevice` (the GPU path routes through `native_to_cuda(::MGBProblem)`), times
# both, and checks the solutions agree. FEM geometries carry `BlockDiag` operators,
# so the GPU solve is the structured batched-GEMM path; spectral geometries are
# dense. Requires CUDA + CUDSS_jll.
#
# (Historical note: an earlier version compared a third "GPU sparse-operator" path.
# That distinction was removed when the solver became type-preserving — whether a
# solve is structured is decided once, at geometry construction, not at transfer —
# so there is no longer a sparse-operator FEM geometry to benchmark against.)
#
# Run from the package root:
#   julia --project=. tools/bench_cuda_vs_native.jl
#
# Environment variables:
#   BENCH_FEM         - "fem2d_P2" (default), "fem2d_P1", "fem3d", or "fem1d"
#   BENCH_L           - single L value (overrides L_MIN/L_MAX)
#   BENCH_L_MIN       - minimum L (default 4 for fem2d_*, 2 for fem3d, 1 for fem1d)
#   BENCH_L_MAX       - maximum L (default 6 for fem2d_*, 2 for fem3d, 7 for fem1d)
#   BENCH_SKIP_NATIVE - "1" to skip the CPU solve (useful at large L)

println("Loading packages…")
t_load_start = time()

using CUDA, CUDSS_jll
using MultiGridBarrier
using Printf

t_load = time() - t_load_start
println("  package load: $(round(t_load, digits=2))s")

const FEM = get(ENV, "BENCH_FEM", "fem2d_P2")

# Build the fine `Geometry` for the chosen discretization, subdivided L levels.
function _build_geom(::Type{T}, L::Int) where {T}
    if FEM == "fem2d_P2"
        geometric_mg(fem2d_P2(T), L).geometry
    elseif FEM == "fem2d_P1"
        geometric_mg(fem2d_P1(T), L).geometry
    elseif FEM == "fem3d"
        geometric_mg(fem3d(T), L).geometry
    elseif FEM == "fem1d"
        nodes = collect(range(-T(1), T(1), length=2^L + 1))
        geometric_mg(fem1d(T; nodes=nodes), L).geometry
    else
        error("Unknown BENCH_FEM=$FEM. Use fem1d / fem2d_P1 / fem2d_P2 / fem3d.")
    end
end

# Default L range per discretization.
default_L_min, default_L_max = if FEM == "fem3d"
    2, 2
elseif FEM == "fem1d"
    1, 7
else
    4, 6
end

const SKIP_NATIVE = get(ENV, "BENCH_SKIP_NATIVE", "0") == "1"

if haskey(ENV, "BENCH_L")
    const L_MIN = parse(Int, ENV["BENCH_L"])
    const L_MAX = L_MIN
else
    const L_MIN = parse(Int, get(ENV, "BENCH_L_MIN", string(default_L_min)))
    const L_MAX = parse(Int, get(ENV, "BENCH_L_MAX", string(default_L_max)))
end

println("\n" * "="^78)
println("Benchmark: $FEM — CPU vs CUDA (structured AMG solve)")
println("  GPU: $(CUDA.name(CUDA.device()))")
println("  L range: $L_MIN .. $L_MAX")
SKIP_NATIVE && println("  *** SKIP_NATIVE: CPU solve disabled ***")
println("="^78)

# ============================================================================
# Warmup (L = L_MIN, both enabled paths)
# ============================================================================
warmup_L = max(L_MIN, FEM == "fem3d" ? 1 : (FEM == "fem1d" ? 1 : 2))
println("\nWarmup (L=$warmup_L)…")
let prob = assemble(amg(_build_geom(Float64, warmup_L)))
    SKIP_NATIVE || (_ = mgb_solve(prob; device=CPUDevice, verbose=false))
    _ = mgb_solve(prob; device=CUDADevice, verbose=false)
    clear_cudss_cache!(); GC.gc(true); CUDA.reclaim()
end
println("  done.")

# ============================================================================
# Results storage
# ============================================================================
struct BenchResult
    L::Int
    n::Int
    t_native::Float64
    t_cuda::Float64
    diff::Float64
end

results = BenchResult[]

for L in L_MIN:L_MAX
    geom = _build_geom(Float64, L)
    prob = assemble(amg(geom))
    n = size(geom.x, 1)
    println("\n" * "-"^78)
    println("L=$L  (n=$n fine-mesh nodes)")
    println("-"^78)

    # --- 1. CPU solve ---
    if SKIP_NATIVE
        println("  [1/2] CPU (UMFPACK)…  SKIPPED")
        t_native = NaN
        sol_native = nothing
    else
        println("  [1/2] CPU (UMFPACK)…")
        GC.gc(true)
        b = @timed mgb_solve(prob; device=CPUDevice, verbose=false)
        t_native = b.time
        sol_native = b.value
        @printf("        %.3fs\n", t_native)
    end

    # --- 2. GPU solve (structured batched-gemm fine-level Hessian) ---
    println("  [2/2] CUDA (batched-gemm fine-level Hessian)…")
    GC.gc(true); CUDA.reclaim(); clear_cudss_cache!()
    b = @timed mgb_solve(prob; device=CUDADevice, verbose=false)
    CUDA.synchronize()
    t_cuda = b.time
    sol_cuda = b.value
    if sol_native === nothing
        diff = NaN
        @printf("        %.3fs\n", t_cuda)
    else
        diff = maximum(abs.(sol_native.z .- sol_cuda.z))
        @printf("        %.3fs  (diff vs CPU: %.2e)\n", t_cuda, diff)
    end

    push!(results, BenchResult(L, n, t_native, t_cuda, diff))
end

# ============================================================================
# Summary table
# ============================================================================
println("\n" * "="^78)
println("Summary: $FEM — CPU vs CUDA")
println("="^78)
println()
@printf("  %-3s  %8s  %10s  %10s  %10s  %10s\n",
        "L", "n", "CPU", "CUDA", "speedup", "diff")
@printf("  %-3s  %8s  %10s  %10s  %10s  %10s\n",
        "---", "--------", "----------", "----------", "----------", "----------")

for r in results
    native_str = isnan(r.t_native) ? "N/A" : @sprintf("%9.3fs", r.t_native)
    if isnan(r.t_native)
        sp_str = "N/A"
        diff_str = "N/A"
    else
        sp = r.t_native / r.t_cuda
        sp_str = sp >= 1 ? @sprintf("%.2fx", sp) : @sprintf("1/%.2fx", 1/sp)
        diff_str = @sprintf("%.1e", r.diff)
    end
    @printf("  %-3d  %8d  %10s  %9.3fs  %10s  %10s\n",
            r.L, r.n, native_str, r.t_cuda, sp_str, diff_str)
end
println()
println("="^78)
