using MultiGridBarrier
using Test
using SparseArrays
using LinearAlgebra
using StaticArrays

# Import internal functions for testing
import MultiGridBarrier: illinois, newton, linesearch_illinois,
        _scatter_cobarrier_gradient, _scatter_cobarrier_hessian

@testset "AlgebraicMultiGridBarrier Coverage Tests" begin

    @testset "convex_linear function" begin
        # Test the completely uncovered convex_linear function
        T = Float64

        # Create a MultiGrid for convex_linear (required by new API)
        nodes4 = collect(range(-1.0, 1.0, length=5))
        mg = amg(subdivide(fem1d(; nodes=nodes4), 2))

        # Test basic linear constraints Ax + b ≤ 0
        # A(x) must return SMatrix, b(x) must return SVector
        A_func = x -> @SMatrix [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b_func = x -> @SVector [1.0, 1.0, 1.0, 1.0]

        # convex_linear returns a single Convex
        linear_domain = convex_linear(T; mg=mg, A=A_func, b=b_func)

        @test linear_domain.barrier isa Tuple
        @test linear_domain.cobarrier isa Tuple
        @test linear_domain.slack !== nothing
        @test linear_domain.args isa Tuple

        # A = x -> I (the default) materializes to a concrete identity when the constraint
        # size is known from an explicit SVector idx; construction must succeed.
        @test convex_linear(T; mg=mg, idx=SVector{1,Int}(1)) isa Convex
        # A UniformScaling A with idx = Colon() is ambiguous (size undeterminable) and errors.
        @test_throws ArgumentError convex_linear(T; mg=mg, idx=Colon())

        # Index validation happens before a barrier can reach its @inbounds access:
        # positivity at convex construction, and the D-row upper bound at assembly.
        @test_throws ArgumentError convex_linear(T; mg=mg, idx=SVector{1,Int}(0))
        q_bad_idx = convex_linear(T; mg=mg, idx=SVector{1,Int}(3))
        @test_throws ArgumentError assemble(mg;
            state_variables=[:u :full], D=[:u :id; :u :dx], Q=q_bad_idx)
    end

    @testset "rectangular linear cobarrier Hessian" begin
        T = Float64
        mg = amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=5))))
        A0 = @SMatrix [1.0 0.0; 0.0 1.0; 1.0 1.0]
        b0 = @SVector [2.0, 2.0, 3.0]
        q = convex_linear(T; mg, idx=Colon(), A=x->A0, b=x->b0)
        yhat = @SVector [0.1, -0.2, 0.5]
        H = reshape(collect(q.cobarrier[3](SVector{6,T}(vec(A0)), b0, yhat)), 3, 3)
        F = A0 * yhat[1:2] .+ b0 .+ yhat[3]
        B = hcat(Matrix(A0), ones(T, 3))
        @test H ≈ B' * Diagonal(one(T) ./ (F .^ 2)) * B
    end

    @testset "cobarrier scatter (Colon index)" begin
        # Directly exercise the Colon-index branch of the feasibility (cobarrier) scatters.
        grad = SVector(1.0, 2.0, 3.0)
        g = _scatter_cobarrier_gradient(Colon(), grad, 0.5, Val(4))
        @test g == SVector(1.0, 2.0, 3.0, 0.5)

        Hflat = SVector{9,Float64}(ntuple(i -> Float64(i), 9))   # 3×3, column-major
        cross = SVector(0.1, 0.2, 0.3)
        Hm = reshape(Array(_scatter_cobarrier_hessian(Colon(), Hflat, cross, 0.9, Val(3), Val(4))), 4, 4)
        @test Hm[1:3, 1:3] == reshape(Array(Hflat), 3, 3)
        @test Hm[1:3, 4]  == Array(cross)
        @test Hm[4, 1:3]  == Array(cross)
        @test Hm[4, 4]    == 0.9
    end

    @testset "Illinois algorithm edge cases" begin
        # Test illinois function edge cases

        # fa == 0 case
        f_zero = x -> x - 1.0
        result = illinois(f_zero, 1.0, 2.0)
        @test result ≈ 1.0
    end

    @testset "Newton early convergence" begin
        T = Float64

        F0 = x -> sum(x.^2)
        F1 = x -> 2*x
        F2 = x -> 2.0*sparse(I, length(x), length(x))

        x0 = zeros(T, 2)

        result = newton(SparseMatrixCSC{T,Int}, T, F0, F1, F2, x0; maxit=10, printlog=(args...)->nothing)
        @test result.converged == true
        @test result.k == 1  # Should converge in first iteration due to inc <= 0
    end

    @testset "Error handling in line search" begin
        T = Float64

        x = T[1.0, 1.0]
        y = 1.0
        g = T[1.0, 1.0]
        n = T[0.1, 0.1]

        F0_error = function(s)
            if any(s .< 0.5)
                throw(DomainError("Test error"))
            end
            return sum(s.^2)
        end
        F1_error = s -> 2*s

        result = linesearch_illinois(Float64; beta=0.5)(x, y, g, n, F0_error, F1_error, printlog=(args...)->nothing)
        @test length(result) == 3
    end

    @testset "Convergence failure exceptions" begin
        T = Float64

        # Force failure with tight tolerance and 1 iteration
        @test_throws MGBConvergenceFailure mgb_solve(
            assemble(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3)))));
            tol=1e-50, maxit=1, verbose=false)
    end


    @testset "structured assembly dimension checks" begin
        T = Float64
        block = MultiGridBarrier.BlockDiag(ones(T, 2, 2, 1))
        blocks = Matrix{Union{typeof(block),Nothing}}(nothing, 1, 1)
        blocks[1, 1] = block

        H = MultiGridBarrier.BlockHessian{T,Array{T,3}}(blocks, 2, 1, [2])
        R_tall = sparse(ones(T, 3, 1))
        @test_throws DimensionMismatch R_tall' * H * R_tall

        H_bad_layout = MultiGridBarrier.BlockHessian{T,Array{T,3}}(
            blocks, 2, 1, [3])
        @test_throws DimensionMismatch R_tall' * H_bad_layout * R_tall
    end

    @testset "continuation target equality" begin
        prob = assemble(amg(fem1d(; nodes=[-1.0, 0.0, 1.0])); p=1.0,
                        g=x->SVector(x[1], 2.0))
        seen_progress = Float64[]
        sol = mgb_solve(prob; verbose=false, tol=0.125, t=8.0,
                        progress=x->push!(seen_progress, x))
        @test all(isfinite, sol.z)
        @test !isempty(seen_progress)
        @test all(isfinite, seen_progress)
        @test seen_progress[end] == 1.0

        # Reaching the target t is not success if its centered/finalized Newton
        # solve itself failed.
        @test_throws MGBConvergenceFailure mgb_solve(
            prob; verbose=false, tol=0.125, t=8.0, maxit=1)
    end

    println("AlgebraicMultiGridBarrier coverage tests completed!")
end
