using MultiGridBarrier
using Test
using SparseArrays
using LinearAlgebra
using StaticArrays

# Import internal functions for testing
import MultiGridBarrier: mgb_phase1, mgb_core, illinois, newton, linesearch_illinois

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

        # convex_linear now returns Vector{Convex{T}}, one per multigrid level
        linear_domains = convex_linear(T; mg=mg, A=A_func, b=b_func)
        @test length(linear_domains) >= 1

        # Get the finest level for testing
        linear_domain = linear_domains[end]

        @test linear_domain.barrier isa Tuple
        @test linear_domain.cobarrier isa Tuple
        @test linear_domain.slack !== nothing
        @test linear_domain.args isa Tuple
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
            amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3))));
            tol=1e-50, maxit=1, verbose=false)
    end

    println("AlgebraicMultiGridBarrier coverage tests completed!")
end
