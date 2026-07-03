# JuMP front-end cross-validation. Rebuilds all six Zoo problems (2D, default
# parameters) through the JuMP modeling extension and compares component-wise
# against the classical constructors `mgb_solve(Zoo.<problem>(amg(geom)))` on the
# same geometry. The discrete problems are identical up to D-row / barrier-piece
# ordering, so solutions must agree to solver tolerance (empirically bit-identical,
# or one ulp). A final testset exercises region-restricted cones (On),
# convex_piecewise lowering, and the feasibility phase.
#
# `using JuMP` loads MultiGridBarrierJuMPExt automatically; MGBModel, Coef, On,
# EpiPower, deriv, integral, set_start, mgb_solution come from MultiGridBarrier.
# JuMP is a test-only dependency ([extras]/[targets] in Project.toml); the package
# itself only weak-depends on it.
using MultiGridBarrier, JuMP, LinearAlgebra, Test

const _JUMP_MATCH_TOL = 1e-8

_quiet!(m) = set_attribute(m, "verbose", false)
_maxdiff(ref, cols...) =
    maximum(maximum(abs.(value(v) .- ref.z[:, i])) for (i, v) in enumerate(cols))

@testset "JuMP front end" begin
    geom = subdivide(fem2d_P2(), 2)
    mg_ref = amg(geom)
    bd = find_boundary(geom)

    @testset "p_harmonic" begin
        g1 = x -> x[1] * x[2]
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u1); @variable(m, u2); @variable(m, s, Broken())
        set_start(u1, g1); set_start(u2, 0.0); set_start(s, 100.0)
        @constraint(m, u1 == Coef(m, g1), On(bd))
        @constraint(m, u2 == Coef(m, 0.0), On(bd))
        @constraint(m, [deriv(u1, :dx); deriv(u1, :dy);
                        deriv(u2, :dx); deriv(u2, :dy); s] in EpiPower(1.5))
        @objective(m, Min, integral(Coef(m, 0.5) * u1 + Coef(m, 0.5) * u2 + s))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.p_harmonic(mg_ref); verbose = false), u1, u2, s)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "minimal_surface" begin
        # constant row inside the cone: s ≥ ‖(∇u, 1)‖ (square zero-padding path)
        gu = x -> 0.5 * (x[1]^2 - x[2]^2)
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken())
        set_start(u, gu); set_start(s, 10.0)
        @constraint(m, u == Coef(m, gu), On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); Coef(m, 1.0); s] in EpiPower(1.0))
        @objective(m, Min, integral(1.0 * s))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.minimal_surface(mg_ref); verbose = false), u, s)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "norton_hoff" begin
        # expression rows and nc < ni zero-row padding
        g1 = x -> x[1] * x[2]
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u1); @variable(m, u2); @variable(m, s, Broken())
        set_start(u1, g1); set_start(u2, 0.0); set_start(s, 100.0)
        @constraint(m, u1 == Coef(m, g1), On(bd))
        @constraint(m, u2 == Coef(m, 0.0), On(bd))
        @constraint(m, [deriv(u1, :dx); deriv(u2, :dy);
                        (deriv(u1, :dy) + deriv(u2, :dx)) / sqrt(2.0);
                        s] in EpiPower(1.5))
        @objective(m, Min, integral(Coef(m, 0.5) * u1 + Coef(m, 0.5) * u2 + s))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.norton_hoff(mg_ref); verbose = false), u1, u2, s)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "rof" begin
        # spatial data inside a cone: r ≥ (u - f_data)²
        fdata = x -> 0.5 * tanh(5 * x[1])
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken()); @variable(m, r, Broken())
        set_start(u, fdata); set_start(s, 10.0); set_start(r, 10.0)
        fd = Coef(m, fdata)
        @constraint(m, u == fd, On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.0))
        @constraint(m, [u - fd; r] in EpiPower(2.0))
        @objective(m, Min, integral(s + Coef(m, 0.5) * r))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.rof(mg_ref); verbose = false), u, s, r)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "two_sided_obstacle" begin
        # global scalar bounds with Number right-hand sides
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken())
        set_start(u, 0.0); set_start(s, 10.0)
        @constraint(m, u == Coef(m, 0.0), On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
        @constraint(m, u >= -0.1)
        @constraint(m, u <= 1.0)
        @objective(m, Min, integral(Coef(m, 2.0) * u + Coef(m, 0.5) * s))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.two_sided_obstacle(mg_ref); verbose = false), u, s)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "elastoplastic_torsion" begin
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken())
        set_start(u, 0.0); set_start(s, 0.5)
        @constraint(m, u == Coef(m, 0.0), On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
        @constraint(m, s <= 1.0)
        @objective(m, Min, integral(Coef(m, 4.0) * u + Coef(m, 0.5) * s))
        optimize!(m)
        d = _maxdiff(mgb_solve(Zoo.elastoplastic_torsion(mg_ref); verbose = false), u, s)
        @test d < _JUMP_MATCH_TOL
    end

    @testset "region-restricted cone (On) + feasibility phase" begin
        # Obstacle imposed only on the left half: exercises the convex_piecewise
        # selector lowering and, via the infeasible start, the feasibility phase.
        # The obstacle must bind on the region and be genuinely absent off it.
        geom3 = subdivide(fem2d_P2(), 3)
        Vn, Nn = size(geom3.x, 1), size(geom3.x, 2)
        left = [(v, e) for e in 1:Nn for v in 1:Vn if geom3.x[v, e, 1] < 0]
        m = MGBModel(geom3); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken())
        set_start(s, 100.0)
        @constraint(m, u == Coef(m, 0.0), On(find_boundary(geom3)))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
        @constraint(m, u >= Coef(m, x -> 0.25 - x[1]^2 - x[2]^2), On(left))
        @objective(m, Min, integral(Coef(m, -1.0) * u + s))
        optimize!(m)
        @test termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        @test mgb_solution(m).SOL_feasibility !== nothing   # phase 1 ran
        phi = value(Coef(m, x -> 0.25 - x[1]^2 - x[2]^2))
        zu = value(u)
        lin = [v + (e - 1) * Vn for (v, e) in left]
        rgt = setdiff(1:length(zu), lin)
        gapL = zu[lin] .- phi[lin]
        @test minimum(gapL) > -1e-8            # obstacle holds on the region
        @test count(<(1e-4), gapL) > 0         # ... with actual contact
        @test minimum(zu[rgt] .- phi[rgt]) < 0 # ... and is absent off it
    end
end
