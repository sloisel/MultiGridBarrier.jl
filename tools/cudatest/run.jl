# Diagnostic CUDA driver: mirrors test/test_cuda.jl but runs each case in
# isolation, reports max|Δ| and full stacktraces per case, and never aborts the
# whole run on the first failure. Run with:
#   julia --project=tools/cudatest tools/cudatest/run.jl
ENV["MPLBACKEND"] = "Agg"
using MultiGridBarrier
using CUDA
using CUDSS_jll
using Printf

@info "CUDA.functional()" funcional=CUDA.functional()
if CUDA.functional()
    dev = CUDA.device()
    @info "device" name=CUDA.name(dev) capability=CUDA.capability(dev)
    @info "toolkit" runtime=CUDA.runtime_version() driver=CUDA.driver_version()
else
    error("CUDA not functional; aborting")
end

cases = [
    ("fem1d geometric_mg",    () -> assemble(geometric_mg(fem1d(; nodes=collect(range(-1.0, 1.0, length=9))), 3))),
    ("fem2d_P2 geometric_mg", () -> assemble(geometric_mg(fem2d_P2(), 3))),
    ("fem3d geometric_mg",    () -> assemble(geometric_mg(fem3d(; k=3), 2))),
    ("spectral1d",            () -> assemble(amg(spectral1d(; n=8)))),
    ("spectral2d",            () -> assemble(amg(spectral2d(; n=5)))),
    ("fem1d AMG",             () -> assemble(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=5)))); p=1.0)),
    ("fem2d_P1 AMG",          () -> assemble(amg(subdivide(fem2d_P1(), 2)); p=1.0)),
    ("fem2d_P2 AMG",          () -> assemble(amg(subdivide(fem2d_P2(), 2)); p=1.0)),
    ("fem3d AMG",             () -> assemble(amg(subdivide(fem3d(; k=1), 2)); p=1.0)),
]

# Allow running a subset:  julia ... run.jl 3 5   (1-based case indices)
sel = isempty(ARGS) ? (1:length(cases)) : parse.(Int, ARGS)

results = Tuple{String,Symbol,Float64,Any}[]
for i in sel
    (name, build) = cases[i]
    println("\n" * "="^70)
    @info "CASE $i: $name"
    local status, maxdiff, err
    status, maxdiff, err = :unknown, NaN, nothing
    try
        prob     = build()
        sol_cpu  = mgb_solve(prob; device=CPUDevice,  verbose=false)
        sol_cuda = mgb_solve(prob; device=CUDADevice, verbose=false)
        maxdiff  = maximum(abs.(sol_cpu.z .- sol_cuda.z))
        status   = maxdiff < 1e-8 ? :pass : :MISMATCH
        @printf("    max|Δ| = %.3e   ->  %s\n", maxdiff, status)
    catch e
        status = :ERROR
        err = e
        @error "CASE $i threw" exception=(e, catch_backtrace())
    end
    push!(results, (name, status, maxdiff, err))
end

println("\n" * "#"^70)
println("SUMMARY")
for (i, (name, status, maxdiff, _)) in zip(sel, results)
    @printf("  [%d] %-24s %-9s max|Δ|=%.3e\n", i, name, String(status), maxdiff)
end
npass = count(r -> r[2] == :pass, results)
@printf("\n%d/%d passed\n", npass, length(results))
println("RUN_DONE")
