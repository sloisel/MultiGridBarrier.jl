# Feasibility phase: bounding-box barrier, R-escalation, and the
# activity-based infeasibility test (see _feasibility_convex / mgb_driver).
using MultiGridBarrier
using Test
using LinearAlgebra
using StaticArrays

@testset "feasibility phase" begin
    # A 1-component, 1-operator problem: minimize ∫u subject to u ≥ lower,
    # from the infeasible start u ≡ 0. Small enough that the phase-I t-ramp
    # rounds are cheap.
    function lower_bound_problem(lower; nodes=5)
        mgobj = amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=nodes))))
        Q = convex_linear(Float64; mg=mgobj, idx=SVector(1),
            A=x->SMatrix{1,1,Float64}(1.0), b=x->SVector(-lower))
        assemble(mgobj;
            state_variables=[:u :full],
            D=[:u :id],
            f=x->SVector(1.0),
            g=x->SVector(0.0),
            Q)
    end

    @testset "far feasible set: box escalation + warm start" begin
        # u ≥ 50 is outside the initial box R = 10, so phase I must press the
        # wall, grow R, and then find a strictly feasible point; the main
        # phase then drives u down onto the constraint.
        sol = mgb_solve(lower_bound_problem(50.0); verbose=false)
        @test sol.SOL_feasibility !== nothing   # phase 1 ran
        @test maximum(abs.(sol.z .- 50.0)) < 1e-3
        # Escalation actually happened: a second round with the tenfold box was
        # launched (the first round either converges wall-pressed or dies
        # numerically at the wall; both must grow the box).
        @test occursin("bounding box R=100", sol.log)
        # Every solve records its backend in the log (the CUDA autodetection is
        # otherwise silent).
        @test occursin("mgb_solve: device = CPUDevice", sol.log)
    end

    @testset "infeasible: interior phase-I minimizer is certified" begin
        # u ≥ 1 and u ≤ 0 simultaneously: infeasible, with the phase-I
        # minimizer at u ≈ 1/2, far inside the box, so the activity test
        # reports infeasibility instead of escalating to Rmax.
        mgobj = amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=5))))
        Q = convex_linear(Float64; mg=mgobj, idx=SVector(1),
            A=x->SMatrix{2,1,Float64}(1.0, -1.0), b=x->SVector(-1.0, 0.0))
        prob = assemble(mgobj;
            state_variables=[:u :full],
            D=[:u :id],
            f=x->SVector(1.0),
            g=x->SVector(0.0),
            Q)
        err = try
            mgb_solve(prob; verbose=false)
            nothing
        catch e
            e
        end
        @test err isa MGBConvergenceFailure
        @test err.code === :infeasible
        @test occursin("appears to be infeasible", err.message)
        @test occursin("inside the bounding box", err.message)
    end

    @testset "feasible only beyond Rmax: honest failure report" begin
        # u ≥ 10^6 is feasible, but not within feasibility_Rmax = 1000: the
        # minimizer presses the wall at every R, and the escalation must stop
        # at the cap with a message naming it.
        err = try
            mgb_solve(lower_bound_problem(1.0e6); verbose=false,
                feasibility_Rmax=1000.0)
            nothing
        catch e
            e
        end
        @test err isa MGBConvergenceFailure
        @test err.code === :feasibility_Rmax
        @test occursin("feasibility_Rmax", err.message)
    end

    @testset "feasible start: phase 1 not triggered" begin
        # Regression guard: the box machinery must not perturb problems whose
        # initial point is already strictly feasible.
        sol = mgb_solve(lower_bound_problem(-50.0); verbose=false)
        @test sol.SOL_feasibility === nothing
        @test maximum(abs.(sol.z .+ 50.0)) < 1e-3
    end
end
