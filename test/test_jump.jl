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

    @testset "Continuous() slack + merged linear pieces" begin
        # Explicitly-tagged conforming slack (never differentiated, no Dirichlet
        # -- previously inexpressible: inference would make it broken). For
        # affine boundary data the optimal |∇u|^p is constant, so the optimal
        # slack is a constant, which the continuous space contains: the solve
        # stays linear-exact. The two redundant global bounds also exercise
        # _merge_nonneg (they lower to ONE stacked convex_linear piece).
        gl = x -> 1 + 2x[1] + 3x[2]
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Continuous())
        set_start(u, gl); set_start(s, 100.0)
        @constraint(m, u == Coef(m, gl), On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
        @constraint(m, u >= -100.0)
        @constraint(m, u <= 100.0)
        @objective(m, Min, integral(1.0 * s))
        optimize!(m)
        @test termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        xf = reshape(geom.x, :, 2)
        @test maximum(abs.(value(u) .- [gl(xf[i, :]) for i in 1:size(xf, 1)])) < 1e-6
    end

    @testset "region-restricted cone (On) + feasibility phase" begin
        # Obstacle imposed only on the left half: exercises the convex_piecewise
        # selector lowering and, via the infeasible start, the feasibility phase.
        # The obstacle must bind on the region and be genuinely absent off it.
        # The region is stated twice — (vertex, element) pairs and the
        # grid-level Bool mask — which must lower identically.
        geom3 = subdivide(fem2d_P2(), 3)
        Vn, Nn = size(geom3.x, 1), size(geom3.x, 2)
        left = [(v, e) for e in 1:Nn for v in 1:Vn if geom3.x[v, e, 1] < 0]
        mask = reshape(geom3.x, :, 2)[:, 1] .< 0

        function obstacle(region::On)
            m = MGBModel(geom3); _quiet!(m)
            @variable(m, u); @variable(m, s, Broken())
            set_start(s, 100.0)
            @constraint(m, u == Coef(m, 0.0), On(find_boundary(geom3)))
            @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
            @constraint(m, u >= Coef(m, x -> 0.25 - x[1]^2 - x[2]^2), region)
            @objective(m, Min, integral(Coef(m, -1.0) * u + s))
            optimize!(m)
            m, u
        end
        m, u = obstacle(On(left))
        @test termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        @test mgb_solution(m).SOL_feasibility !== nothing   # phase 1 ran
        phi = value(Coef(m, x -> 0.25 - x[1]^2 - x[2]^2))
        zu = value(u)
        gapL = zu[mask] .- phi[mask]
        @test minimum(gapL) > -1e-8              # obstacle holds on the region
        @test count(<(1e-4), gapL) > 0           # ... with actual contact
        @test minimum(zu[.!mask] .- phi[.!mask]) < 0 # ... and is absent off it

        # On(geom, mask) is eager sugar for the same node set
        @test On(geom3, mask).pairs == On(left).pairs
        m2, u2 = obstacle(On(geom3, mask))
        @test value(u2) == zu
        @test_throws ArgumentError On(geom3, trues(5))
    end

    @testset "nodal vectors are the fundamental data form" begin
        # Every data entry point (Coef, set_start, the EpiPower exponent)
        # accepts a raw per-node vector; functions and constants are sugar for
        # it. The same problem stated both ways lowers to identical grids, so
        # the solves must agree bit-for-bit.
        xf = reshape(geom.x, :, size(geom.x, 3))
        n = size(xf, 1)
        gfun = x -> x[1]^2 + x[2]^2
        gvec = [gfun(xf[i, :]) for i in 1:n]

        function build(g_u, s0, obj_c, pexp)
            m = MGBModel(geom); _quiet!(m)
            @variable(m, u); @variable(m, s, Broken())
            set_start(u, g_u); set_start(s, s0)
            @constraint(m, u == Coef(m, g_u), On(bd))
            @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(pexp))
            @objective(m, Min, integral(Coef(m, obj_c) * u + s))
            optimize!(m)
            m, u, s
        end
        ma, ua, sa = build(gfun, 100.0, 0.5, 1.5)                            # sugar
        mb, ub, sb = build(gvec, fill(100.0, n), fill(0.5, n), fill(1.5, n)) # vectors
        @test value(ub) == value(ua)
        @test value(sb) == value(sa)
        @test objective_value(mb) == objective_value(ma)
        @test value(Coef(ma, gvec)) == value(Coef(ma, gfun))

        # wrong-length vectors are rejected where they enter
        @test_throws ArgumentError Coef(ma, ones(n + 1))
        @test_throws ArgumentError set_start(ua, ones(n - 1))
        mc = MGBModel(geom); _quiet!(mc)
        @variable(mc, uc); @variable(mc, sc, Broken())
        @test_throws ArgumentError @constraint(mc,
            [deriv(uc, :dx); deriv(uc, :dy); sc] in EpiPower(ones(n + 1)))

        # solutions round-trip into starts (same nodal ordering): warm-start
        # from the previous optimum, with the slack lifted to stay interior
        set_start(ua, value(ua)); set_start(sa, value(sa) .+ 1.0)
        optimize!(ma)
        @test termination_status(ma) == JuMP.MOI.LOCALLY_SOLVED
        @test maximum(abs.(value(ua) .- value(ub))) < 1e-6
    end

    @testset "variable/expression interface + model printing" begin
        # The small JuMP/Base interface surface: identity, hashing, zero/one,
        # convert, broadcasting, string forms, objective plumbing, and the
        # native-JuMP-style model printer. No solve — this is all modeling-time.
        mp = MGBModel(geom); _quiet!(mp)
        @variable(mp, u); @variable(mp, s, Broken())
        E = typeof(zero(u))                      # MGBExpr{Float64}

        # identity / accessors
        @test JuMP.num_variables(mp) == 2
        @test JuMP.owner_model(u) === mp
        @test JuMP.name(u) == "u"
        @test JuMP.name(deriv(u, :dx)) == "deriv(u, :dx)"
        @test deriv(u, :id) == u && u != s
        @test copy(u) == u
        @test JuMP.isequal_canonical(u, u)
        @test Dict(u => 1)[deriv(u, :id)] == 1   # hash + == in real use
        @test sprint(show, deriv(u, :dy)) == "deriv(u, :dy)"
        @test occursin("MGBModel over", sprint(show, mp))

        # zero / one / convert / broadcastable
        @test sprint(show, zero(u)) == "0.0"
        @test sprint(show, zero(typeof(u))) == "0.0"
        @test sprint(show, zero(E)) == "0.0" && sprint(show, zero(zero(u))) == "0.0"
        @test sprint(show, one(typeof(u))) == "1.0"
        @test sprint(show, convert(E, Coef(mp, 2.5))) == "2.5"
        @test sprint(show, convert(E, 1.5)) == "1.5"
        @test (2.0 .* u) isa E && ((u + s) .+ 1.0) isa E && (Coef(mp, 1.0) .* 2.0) isa E
        es = sprint(show, 2.0 * u + Coef(mp, x -> x[1]) * s + Coef(mp, x -> x[1]))
        @test occursin("2.0*", es) && occursin("⟨coef⟩", es)

        # MutableArithmetics promotion (the Real variants; the macros need these
        # on some JuMP versions, so exercise them through the ext's own alias)
        MA = JuMP._MA
        @test MA.promote_operation(+, typeof(u), Float64) == E
        @test MA.promote_operation(*, Float64, typeof(u)) == E
        @test MA.promote_operation(-, typeof(Coef(mp, 1.0)), Float64) == E
        @test MA.promote_operation(+, E, Float64) == E

        # objective plumbing: sense accessors, split sense/function route,
        # and the not-an-integral error fallbacks
        JuMP.set_objective_sense(mp, JuMP.MOI.MAX_SENSE)
        @test JuMP.objective_sense(mp) == JuMP.MOI.MAX_SENSE
        JuMP.set_objective_sense(mp, JuMP.MOI.MIN_SENSE)
        JuMP.set_objective_function(mp, integral(1.0 * s + 1.0))
        @test_throws ArgumentError JuMP.set_objective_function(mp, s)
        @test_throws ArgumentError @objective(mp, Min, s)
        @objective(mp, Min, integral(1.0 * s + 1.0))

        # printing with no constraints yet, then with every constraint flavor
        @test occursin("∫(", sprint(print, mp))
        cr = @constraint(mp, u == Coef(mp, 0.0), On(bd))
        @test sprint(show, cr) isa String
        @test_throws ArgumentError JuMP.dual(cr)
        @constraint(mp, u >= -10.0)
        @constraint(mp, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(x -> 1.5 + 0.1 * x[1]^2))
        @constraint(mp, [s; deriv(u, :dx); deriv(u, :dy)] in JuMP.MOI.SecondOrderCone(3))
        str = sprint(print, mp)
        @test occursin("== ⟨data⟩", str) && occursin("on ", str)   # Dirichlet + region
        @test occursin("≥ 0", str)                                 # nonneg row
        @test occursin("^⟨coef⟩", str) && occursin("^1.0", str)    # spatial + uniform exponent
        @test occursin("∫(", str) && occursin("+ 1.0", str)        # objective with constant

        # a fresh model prints "0" for the missing objective
        m0 = MGBModel(geom)
        @variable(m0, w0)
        @test occursin("0", sprint(print, m0))

        # RegionConstraint is a JuMP.AbstractConstraint: function/set accessors
        rc = JuMP.build_constraint(error, 1.0 * u, JuMP.MOI.GreaterThan(0.0), On(bd))
        @test JuMP.jump_function(rc) isa E
        @test JuMP.moi_set(rc) isa JuMP.MOI.GreaterThan{Float64}

        # vector comparisons (with and without On) must land in the same
        # region/cone machinery as their scalar spellings
        @constraint(mp, [u, u] >= [Coef(mp, -5.0), Coef(mp, -6.0)], On(bd))
        @constraint(mp, [u, u] <= [Coef(mp, 5.0), Coef(mp, 6.0)], On(bd))
        @constraint(mp, [u, u] >= [Coef(mp, -7.0), Coef(mp, -8.0)])
        @test_throws ArgumentError @constraint(mp, [u, u] == [Coef(mp, 0.0), Coef(mp, 0.0)], On(bd))
        @test JuMP.num_constraints(mp) == 7

        # rejections with explanatory errors
        @test_throws ArgumentError u * s                              # nonlinear
        @test_throws ArgumentError @constraint(mp, -1.0 <= u <= 1.0)  # Interval set
        @test_throws ArgumentError @constraint(mp, [u, s, u] in JuMP.MOI.ExponentialCone())
    end

    @testset "lowering guards" begin
        # a Broken variable cannot carry a Dirichlet condition
        mb = MGBModel(geom); _quiet!(mb)
        @variable(mb, ub, Broken()); @variable(mb, sb, Broken())
        @constraint(mb, ub == Coef(mb, 0.0), On(bd))
        @constraint(mb, [ub; sb] in EpiPower(2.0))
        @objective(mb, Min, integral(1.0 * sb))
        @test_throws ArgumentError optimize!(mb)

        # Dirichlet-but-never-differentiated is legal but warns (likely accident)
        mw = MGBModel(geom); _quiet!(mw)
        @variable(mw, uw); @variable(mw, sw, Broken())
        set_start(sw, 10.0)
        @constraint(mw, uw == Coef(mw, 0.0), On(bd))
        @constraint(mw, [uw; sw] in EpiPower(2.0))
        @objective(mw, Min, integral(1.0 * sw))
        @test_logs (:warn, r"never differentiated") match_mode = :any optimize!(mw)
        @test termination_status(mw) == JuMP.MOI.LOCALLY_SOLVED
    end

    @testset "scalar-arithmetic sugar lowers to the classical problem" begin
        # Objective spelled with every affine-sugar form (unary +/-, Real on
        # either side, /); algebraically it is 0.5u + s, i.e. the package's
        # default p-Laplacian, so the solve must match the classical reference.
        g1 = x -> x[1]^2 + x[2]^2
        m = MGBModel(geom); _quiet!(m)
        @variable(m, u); @variable(m, s, Broken())
        set_start(u, g1)
        JuMP.set_start_value(s, 100.0)           # JuMP-native start API delegates
        @constraint(m, u == Coef(m, g1), On(bd))
        @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
        @objective(m, Min,
            integral(+(u / 2) - (-s) + (u + 1.0) + (1.0 + u) + (1.0 - u) - u - 3.0))
        optimize!(m)
        sol_ref = mgb_solve(assemble(amg(geom); p = 1.5); verbose = false)
        @test maximum(abs.(value(u) .- sol_ref.z[:, 1])) < _JUMP_MATCH_TOL
        # success-side status accessors
        @test occursin("converged", JuMP.raw_status(m))
        @test JuMP.solve_time(m) > 0
        @test JuMP.primal_status(m) == JuMP.MOI.FEASIBLE_POINT
        @test JuMP.dual_status(m) == JuMP.MOI.NO_SOLUTION
    end

    @testset "region-restricted cone + Uniform variable" begin
        # ROF-style fidelity cone imposed only on the left half (a :power piece
        # inside convex_piecewise), with a global r >= 0 keeping the slack
        # bounded off the region.
        fdata = x -> 0.5 * tanh(5 * x[1])
        mask = reshape(geom.x, :, 2)[:, 1] .< 0
        mr = MGBModel(geom); _quiet!(mr)
        @variable(mr, w); @variable(mr, sw, Broken()); @variable(mr, r, Broken())
        set_start(w, fdata); set_start(sw, 10.0); set_start(r, 10.0)
        fd = Coef(mr, fdata)
        @constraint(mr, w == fd, On(bd))
        @constraint(mr, [deriv(w, :dx); deriv(w, :dy); sw] in EpiPower(1.0))
        @constraint(mr, [w - fd; r] in EpiPower(2.0), On(geom, mask))
        @constraint(mr, r >= 0.0)
        @objective(mr, Min, integral(sw + Coef(mr, 0.5) * r))
        optimize!(mr)
        @test termination_status(mr) == JuMP.MOI.LOCALLY_SOLVED
        rv = value(r); gap2 = (value(w) .- value(fd)) .^ 2
        @test maximum(rv[.!mask]) < 1e-3               # cone absent off-region
        @test all(rv[mask] .>= gap2[mask] .- 1e-8)     # epigraph holds on-region

        # Uniform(): one global constant c with c >= u pointwise; minimizing a
        # small weight on c drives it to max(u), and its column is constant.
        g0 = x -> x[1]^2 + x[2]^2
        mu = MGBModel(geom); _quiet!(mu)
        @variable(mu, uu); @variable(mu, su, Broken()); @variable(mu, c, Uniform())
        set_start(uu, g0); set_start(su, 100.0); set_start(c, 10.0)
        @constraint(mu, uu == Coef(mu, g0), On(bd))
        @constraint(mu, [deriv(uu, :dx); deriv(uu, :dy); su] in EpiPower(2.0))
        @constraint(mu, c >= uu)
        @objective(mu, Min, integral(Coef(mu, 0.5) * uu + su + Coef(mu, 0.1) * c))
        optimize!(mu)
        @test termination_status(mu) == JuMP.MOI.LOCALLY_SOLVED
        cv = value(c)
        @test maximum(abs.(cv .- cv[1])) < 1e-8        # genuinely one global dof
        @test cv[1] > maximum(value(uu)) - 1e-6        # dominates u ...
        @test cv[1] < maximum(value(uu)) + 1e-2        # ... and binds
    end

    @testset "convergence failure surfaces as OTHER_ERROR" begin
        # maxit too small for the barrier to reach its target t: optimize! must
        # catch MGBConvergenceFailure, report it, and refuse to hand out values.
        g1 = x -> x[1]^2 + x[2]^2
        mf = MGBModel(geom); _quiet!(mf)
        set_attribute(mf, "maxit", 2)
        @variable(mf, u); @variable(mf, s, Broken())
        set_start(u, g1); set_start(s, 100.0)
        @constraint(mf, u == Coef(mf, g1), On(bd))
        @constraint(mf, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
        @objective(mf, Min, integral(Coef(mf, 0.5) * u + s))
        optimize!(mf)                                   # must not throw
        @test termination_status(mf) == JuMP.MOI.OTHER_ERROR
        @test occursin("onvergence failure", JuMP.raw_status(mf))
        @test JuMP.primal_status(mf) == JuMP.MOI.NO_SOLUTION
        @test_throws ArgumentError value(u)
        @test_throws ArgumentError objective_value(mf)
        @test_throws ArgumentError mgb_solution(mf)
        @test_throws ArgumentError solver_log(mf)
    end
end
