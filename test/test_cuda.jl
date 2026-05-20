cuda_ok = try
    using CUDA
    using CUDSS_jll
    CUDA.functional()
catch
    false
end

if cuda_ok
    @testset "CUDA extension" begin
        @testset "fem1d CUDA geometric_mg vs native" begin
            sol_native = mgb_solve(geometric_mg(fem1d(; nodes=collect(range(-1.0, 1.0, length=9))), 3); verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(geometric_mg(fem1d(; nodes=collect(range(-1.0, 1.0, length=9))), 3; structured=false)); verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem2d_P2 CUDA geometric_mg vs native" begin
            sol_native = mgb_solve(geometric_mg(fem2d_P2(), 3); verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(geometric_mg(fem2d_P2(), 3; structured=false)); verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem3d CUDA geometric_mg vs native" begin
            sol_native = mgb_solve(geometric_mg(fem3d(; k=3), 2); verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(geometric_mg(fem3d(; k=3), 2; structured=false)); verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "spectral1d CUDA vs native" begin
            sol_native = mgb_solve(amg(spectral1d(; n=8)); verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(spectral1d(; n=8))); verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "spectral2d CUDA vs native" begin
            sol_native = mgb_solve(amg(spectral2d(; n=5)); verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(spectral2d(; n=5))); verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end

        # AMG-FEM CUDA front-ends via native_to_cuda(amg(...)). The algebraic
        # prolongations stay sparse on the GPU; structured operators (subdivide)
        # are preserved as block by the type-preserving native_to_cuda.
        nodes5 = collect(range(-1.0, 1.0, length=5))

        @testset "fem1d CUDA AMG vs native AMG" begin
            sol_native = mgb_solve(amg(fem1d(; nodes=nodes5)); p=1.0, verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(fem1d(; nodes=nodes5))); p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem2d_P1 CUDA AMG vs native AMG" begin
            sol_native = mgb_solve(amg(subdivide(fem2d_P1(), 2)); p=1.0, verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(subdivide(fem2d_P1(), 2))); p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem2d_P2 CUDA AMG vs native AMG" begin
            sol_native = mgb_solve(amg(subdivide(fem2d_P2(), 2)); p=1.0, verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(subdivide(fem2d_P2(), 2))); p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem3d CUDA AMG vs native AMG" begin
            sol_native = mgb_solve(amg(subdivide(fem3d(; k=1), 2)); p=1.0, verbose=false)
            sol_cuda = cuda_to_native(mgb_solve(native_to_cuda(amg(subdivide(fem3d(; k=1), 2))); p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
    end
else
    @info "Skipping CUDA tests: no functional GPU or CUDA packages unavailable"
end
