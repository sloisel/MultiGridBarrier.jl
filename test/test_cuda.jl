cuda_ok = try
    using CUDA
    using CUDSS_jll
    CUDA.functional()
catch
    false
end

if cuda_ok
    @testset "CUDA extension" begin
        @testset "geometric_fem1d CUDA vs native" begin
            sol_native = geometric_fem1d_solve(; L=3, verbose=false)
            sol_cuda = cuda_to_native(geometric_fem1d_cuda_solve(; L=3, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "geometric_fem2d_P2 CUDA vs native" begin
            sol_native = geometric_fem2d_P2_solve(; L=3, verbose=false)
            sol_cuda = cuda_to_native(geometric_fem2d_P2_cuda_solve(; L=3, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "geometric_fem3d CUDA vs native" begin
            sol_native = geometric_fem3d_solve(; L=2, verbose=false)
            sol_cuda = cuda_to_native(geometric_fem3d_cuda_solve(; L=2, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "spectral1d CUDA vs native" begin
            sol_native = spectral1d_solve(; n=8, verbose=false)
            sol_cuda = cuda_to_native(spectral1d_cuda_solve(; n=8, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "spectral2d CUDA vs native" begin
            sol_native = spectral2d_solve(; n=5, verbose=false)
            sol_cuda = cuda_to_native(spectral2d_cuda_solve(; n=5, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end

        # Algebraic-AMG FEM CUDA front-ends. The L= kwarg subdivides internally,
        # so CPU and GPU build identical hierarchies from the same coarse K default.
        nodes5 = collect(range(-1.0, 1.0, length=5))

        @testset "fem1d CUDA vs native" begin
            sol_native = fem1d_solve(; nodes=nodes5, p=1.0, verbose=false)
            sol_cuda = cuda_to_native(fem1d_cuda_solve(; nodes=nodes5, p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem2d_P1 CUDA vs native" begin
            sol_native = fem2d_P1_solve(; L=2, p=1.0, verbose=false)
            sol_cuda = cuda_to_native(fem2d_P1_cuda_solve(; L=2, p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem2d_P2 CUDA vs native" begin
            sol_native = fem2d_P2_solve(; L=2, p=1.0, verbose=false)
            sol_cuda = cuda_to_native(fem2d_P2_cuda_solve(; L=2, p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
        @testset "fem3d CUDA vs native" begin
            sol_native = fem3d_solve(; L=2, k=1, p=1.0, verbose=false)
            sol_cuda = cuda_to_native(fem3d_cuda_solve(; L=2, k=1, p=1.0, verbose=false))
            @test maximum(abs.(sol_native.z .- sol_cuda.z)) < 1e-8
        end
    end
else
    @info "Skipping CUDA tests: no functional GPU or CUDA packages unavailable"
end
