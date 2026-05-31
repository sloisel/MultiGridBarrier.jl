cuda_ok = try
    using CUDA
    using CUDSS_jll
    CUDA.functional()
catch
    false
end

if cuda_ok
    # Each case assembles a CPU `MGBProblem` once, then solves it on the CPU and on
    # CUDA via the `device=` keyword (the GPU path routes through
    # `native_to_cuda(::MGBProblem)`), and checks the two solutions agree. Structured
    # FEM operators stay block on the GPU; spectral operators are dense; AMG
    # prolongations are sparse CSR.
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
    @testset "CUDA extension" begin
        for (name, build) in cases
            @testset "$name: CPU vs CUDA" begin
                prob     = build()
                sol_cpu  = mgb_solve(prob; device=CPUDevice,  verbose=false)
                sol_cuda = mgb_solve(prob; device=CUDADevice, verbose=false)
                @test maximum(abs.(sol_cpu.z .- sol_cuda.z)) < 1e-8
            end
        end
    end
else
    @info "Skipping CUDA tests: no functional GPU or CUDA packages unavailable"
end
