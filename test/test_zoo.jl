using MultiGridBarrier
using Test
using LinearAlgebra

@testset "Zoo problems (smoke tests)" begin
    nodes3 = collect(range(-1.0, 1.0, length=3))                 # 2 elements
    mg1 = amg(fem1d(; nodes=nodes3))
    mg2 = amg(fem2d_P1())

    finite(sol) = all(isfinite, sol.z)

    @testset "elastoplastic_torsion" begin
        problem = Zoo.elastoplastic_torsion(mg1)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end

    @testset "minimal_surface" begin
        problem = Zoo.minimal_surface(mg1)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end

    @testset "p_harmonic" begin
        problem = Zoo.p_harmonic(mg2)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end

    @testset "norton_hoff" begin
        problem = Zoo.norton_hoff(mg2)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end

    @testset "rof" begin
        problem = Zoo.rof(mg1)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end

    @testset "two_sided_obstacle" begin
        problem = Zoo.two_sided_obstacle(mg1)
        sol = mgb_solve(; problem..., verbose=false, tol=1e-3)
        @test finite(sol)
    end
end
