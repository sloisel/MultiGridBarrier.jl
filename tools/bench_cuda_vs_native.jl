#!/usr/bin/env julia
#
# Benchmark: CPU AMG vs GPU AMG (sparse fine ops) vs GPU AMG (structured fine ops).
#
# Runs `mgb_solve(amg(subdivide(<fem>, L)))` on three matrix backings to isolate
# the win from structured (BlockDiag-on-GPU) fine-level Hessian assembly.
# Requires CUDA + CUDSS_jll.
#
# Run from the package root:
#   julia --project=. tools/bench_cuda_vs_native.jl
#
# Environment variables:
#   BENCH_FEM         - "fem2d_P2" (default), "fem2d_P1", "fem3d", or "fem1d"
#   BENCH_L           - single L value (overrides L_MIN/L_MAX)
#   BENCH_L_MIN       - minimum L (default 4 for fem2d_*, 2 for fem3d, 1 for fem1d)
#   BENCH_L_MAX       - maximum L (default 6 for fem2d_*, 2 for fem3d, 7 for fem1d)
#   BENCH_SKIP_NATIVE - "1" to skip CPU solve (useful at large L)
#   BENCH_SKIP_UNSRT  - "1" to skip the GPU sparse-operator path
#   BENCH_SRT_ONLY    - "1" to only run the GPU structured path

println("Loading packages…")
t_load_start = time()

using CUDA, CUDSS_jll
using MultiGridBarrier
using Printf

t_load = time() - t_load_start
println("  package load: $(round(t_load, digits=2))s")

const FEM = get(ENV, "BENCH_FEM", "fem2d_P2")

# Build a "fine Geometry" for the chosen discretization, subdivided L levels.
# Returns (geom_struct, geom_sparse) where geom_struct has BlockDiag operators
# and geom_sparse has SparseMatrixCSC operators. Both describe the same mesh.
function _build_geoms(::Type{T}, L::Int) where {T}
    if FEM == "fem2d_P2"
        geom_struct = geometric_mg(fem2d_P2(T), L; structured=true).geometry
        geom_sparse = geometric_mg(fem2d_P2(T), L; structured=false).geometry
    elseif FEM == "fem2d_P1"
        geom_struct = geometric_mg(fem2d_P1(T), L; structured=true).geometry
        geom_sparse = geometric_mg(fem2d_P1(T), L; structured=false).geometry
    elseif FEM == "fem3d"
        geom_struct = geometric_mg(fem3d(T), L; structured=true).geometry
        geom_sparse = geometric_mg(fem3d(T), L; structured=false).geometry
    elseif FEM == "fem1d"
        # fem1d's structured form comes via geometric_mg(fem1d(...), L).
        # geometric_mg(fem1d) currently builds its own default node mesh; we just
        # ask for that on both sides.
        nodes = collect(range(-T(1), T(1), length=2^L + 1))
        geom_struct = geometric_mg(fem1d(T; nodes=nodes), L; structured=true).geometry
        geom_sparse = geometric_mg(fem1d(T; nodes=nodes), L; structured=false).geometry
    else
        error("Unknown BENCH_FEM=$FEM. Use fem1d / fem2d_P1 / fem2d_P2 / fem3d.")
    end
    return geom_struct, geom_sparse
end

# Default L range per discretization.
default_L_min, default_L_max = if FEM == "fem3d"
    2, 2
elseif FEM == "fem1d"
    1, 7
else
    4, 6
end

const SRT_ONLY = get(ENV, "BENCH_SRT_ONLY", "0") == "1"
const SKIP_NATIVE = SRT_ONLY || get(ENV, "BENCH_SKIP_NATIVE", "0") == "1"
const SKIP_UNSRT = SRT_ONLY || get(ENV, "BENCH_SKIP_UNSRT", "0") == "1"

if haskey(ENV, "BENCH_L")
    const L_MIN = parse(Int, ENV["BENCH_L"])
    const L_MAX = L_MIN
else
    const L_MIN = parse(Int, get(ENV, "BENCH_L_MIN", string(default_L_min)))
    const L_MAX = parse(Int, get(ENV, "BENCH_L_MAX", string(default_L_max)))
end

println("\n" * "="^78)
println("Benchmark: $FEM — CPU AMG vs GPU AMG sparse-ops vs GPU AMG structured-ops")
println("  GPU: $(CUDA.name(CUDA.device()))")
println("  L range: $L_MIN .. $L_MAX")
SKIP_NATIVE && println("  *** SKIP_NATIVE: CPU solve disabled ***")
SKIP_UNSRT  && println("  *** SKIP_UNSRT:  GPU sparse-operator solve disabled ***")
println("="^78)

# ============================================================================
# Warmup (L = L_MIN, all enabled paths)
# ============================================================================
warmup_L = max(L_MIN, FEM == "fem3d" ? 1 : (FEM == "fem1d" ? 1 : 2))
println("\nWarmup (L=$warmup_L)…")
let (g_struct, g_sparse) = _build_geoms(Float64, warmup_L)
    SKIP_NATIVE || (_ = mgb_solve(amg(g_struct); verbose=false))
    if !SKIP_UNSRT
        _ = mgb_solve(native_to_cuda(amg(g_sparse); structured=false); verbose=false)
        clear_cudss_cache!(); GC.gc(true); CUDA.reclaim()
    end
    _ = mgb_solve(native_to_cuda(amg(g_struct); structured=false); verbose=false)
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
    t_cuda_unsrt::Float64
    t_cuda_srt::Float64
    diff_unsrt::Float64
    diff_srt::Float64
end

results = BenchResult[]

for L in L_MIN:L_MAX
    g_struct, g_sparse = _build_geoms(Float64, L)
    n = size(g_struct.x, 1)
    println("\n" * "-"^78)
    println("L=$L  (n=$n fine-mesh nodes)")
    println("-"^78)

    # --- 1. CPU AMG (operators: BlockDiag on CPU) ---
    if SKIP_NATIVE
        println("  [1/3] CPU AMG (UMFPACK)…  SKIPPED")
        t_native = NaN
        sol_native = nothing
    else
        println("  [1/3] CPU AMG (UMFPACK)…")
        GC.gc(true)
        b = @timed mgb_solve(amg(g_struct); verbose=false)
        t_native = b.time
        sol_native = b.value
        @printf("        %.3fs\n", t_native)
    end

    # --- 2. GPU AMG with sparse operators ---
    if SKIP_UNSRT
        println("  [2/3] GPU AMG sparse-ops…  SKIPPED")
        t_cuda_unsrt = NaN
        diff_unsrt = NaN
    else
        println("  [2/3] GPU AMG sparse-ops (all SpGEMM)…")
        GC.gc(true); CUDA.reclaim(); clear_cudss_cache!()
        b = @timed mgb_solve(native_to_cuda(amg(g_sparse); structured=false); verbose=false)
        CUDA.synchronize()
        t_cuda_unsrt = b.time
        sol_unsrt = cuda_to_native(b.value)
        if sol_native === nothing
            diff_unsrt = NaN
            @printf("        %.3fs\n", t_cuda_unsrt)
        else
            diff_unsrt = maximum(abs.(sol_native.z .- sol_unsrt.z))
            @printf("        %.3fs  (diff vs CPU: %.2e)\n", t_cuda_unsrt, diff_unsrt)
        end
    end

    # --- 3. GPU AMG with structured (BlockDiag) operators ---
    println("  [3/3] GPU AMG structured-ops (batched-gemm fine-level Hessian)…")
    GC.gc(true); CUDA.reclaim(); clear_cudss_cache!()
    b = @timed mgb_solve(native_to_cuda(amg(g_struct); structured=false); verbose=false)
    CUDA.synchronize()
    t_cuda_srt = b.time
    sol_srt = cuda_to_native(b.value)
    if sol_native === nothing
        diff_srt = NaN
        @printf("        %.3fs\n", t_cuda_srt)
    else
        diff_srt = maximum(abs.(sol_native.z .- sol_srt.z))
        @printf("        %.3fs  (diff vs CPU: %.2e)\n", t_cuda_srt, diff_srt)
    end

    push!(results, BenchResult(L, n, t_native, t_cuda_unsrt, t_cuda_srt, diff_unsrt, diff_srt))
end

# ============================================================================
# Summary table
# ============================================================================
println("\n" * "="^78)
println("Summary: $FEM — CPU AMG vs GPU AMG (sparse) vs GPU AMG (structured)")
println("="^78)
println()
@printf("  %-3s  %8s  %10s  %10s  %10s  %10s  %10s\n",
        "L", "n", "CPU", "GPU sparse", "GPU srt", "sparse spdup", "srt spdup")
@printf("  %-3s  %8s  %10s  %10s  %10s  %10s  %10s\n",
        "---", "--------", "----------", "----------", "----------", "----------", "----------")

for r in results
    native_str = isnan(r.t_native) ? "N/A" : @sprintf("%9.3fs", r.t_native)
    unsrt_str = isnan(r.t_cuda_unsrt) ? "N/A" : @sprintf("%9.3fs", r.t_cuda_unsrt)
    if isnan(r.t_native) || isnan(r.t_cuda_unsrt)
        sp_unsrt_str = "N/A"
    else
        sp_unsrt = r.t_native / r.t_cuda_unsrt
        sp_unsrt_str = sp_unsrt >= 1 ? @sprintf("%.2fx", sp_unsrt) : @sprintf("1/%.2fx", 1/sp_unsrt)
    end
    if isnan(r.t_native)
        sp_srt_str = "N/A"
    else
        sp_srt = r.t_native / r.t_cuda_srt
        sp_srt_str = sp_srt >= 1 ? @sprintf("%.2fx", sp_srt) : @sprintf("1/%.2fx", 1/sp_srt)
    end
    @printf("  %-3d  %8d  %10s  %10s  %9.3fs  %10s  %10s\n",
            r.L, r.n, native_str, unsrt_str, r.t_cuda_srt, sp_unsrt_str, sp_srt_str)
end

println()
println("  Structured vs sparse on GPU:")
for r in results
    if isnan(r.t_cuda_unsrt)
        @printf("    L=%d: sparse skipped\n", r.L)
    else
        ratio = r.t_cuda_unsrt / r.t_cuda_srt
        ratio_str = ratio >= 1 ? @sprintf("%.2fx faster", ratio) : @sprintf("%.2fx slower", 1/ratio)
        if isnan(r.diff_unsrt)
            @printf("    L=%d: structured is %s\n", r.L, ratio_str)
        else
            @printf("    L=%d: structured is %s  (diff: sparse=%.1e, srt=%.1e)\n",
                    r.L, ratio_str, r.diff_unsrt, r.diff_srt)
        end
    end
end
println()
println("="^78)
