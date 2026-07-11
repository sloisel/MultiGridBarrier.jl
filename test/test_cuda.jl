cuda_ok = try
    using CUDA
    using CUDSS_jll
    CUDA.functional()
catch
    false
end

using StaticArrays

@testset "barrier closures are isbits (GPU-compilable)" begin
    # Regression: a convex_linear function-scope local named `ni` shadowed by
    # assignments inside the barrier closures made Julia box the variable
    # (Core.Box); the closures were then no longer isbits and could not be
    # compiled into CUDA kernels (KernelError: passing non-bitstype argument).
    # This check needs no GPU, so it runs everywhere.
    mg1 = amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=9))))
    Qlin = convex_linear(; mg=mg1, idx=SVector(1,3),
                         A=x->SMatrix{1,2}(1.0,0.0), b=x->SVector(2.0))
    Qpw = intersect(mg1, Qlin,
                    convex_linear(; mg=mg1, idx=SVector(1,3),
                                  A=x->SMatrix{1,2}(-1.0,0.0), b=x->SVector(2.0)))
    for Q in (Qlin, Qpw), fs in (Q.barrier, Q.cobarrier, (Q.slack,)), f in fs
        @test isbits(f)
    end
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

        @testset "map_rows_gpu with ≥6 CuMatrix args (intersect barrier)" begin
            # Regression: the piecewise barrier of intersect(mg, Qa, Qb)
            # evaluates map_rows_gpu on select + A1,b1,A2,b2 grids + Dz — six
            # CuMatrix arguments — and the phase-I slack map adds z0 and w for
            # seven. The old per-arity kernels stopped at five, so both were
            # MethodErrors on GPU; the variadic kernel has no arity limit.
            # Problem: minimize ∫u subject to 1 ≤ u ≤ 5 (solution u ≡ 1), from
            # the infeasible start u ≡ 0 so the phase-I slack path also runs.
            mgq = amg(fem2d_P2())
            Qa = convex_linear(; mg=mgq, idx=SVector(1),
                               A=x->SMatrix{1,1}(1.0),  b=x->SVector(-1.0))
            Qb = convex_linear(; mg=mgq, idx=SVector(1),
                               A=x->SMatrix{1,1}(-1.0), b=x->SVector(5.0))
            probpw = assemble(mgq; state_variables=[:u :full], D=[:u :id],
                              f=x->SVector(1.0), g=x->SVector(0.0),
                              Q=intersect(mgq, Qa, Qb))
            sol_cpu  = mgb_solve(probpw; device=CPUDevice,  verbose=false)
            sol_cuda = mgb_solve(probpw; device=CUDADevice, verbose=false)
            @test sol_cuda.SOL_feasibility !== nothing   # phase I ran (7-arg slack map)
            @test maximum(abs.(sol_cpu.z .- sol_cuda.z)) < 1e-8
        end
    end
else
    @info "Skipping CUDA tests: no functional GPU or CUDA packages unavailable"
end
