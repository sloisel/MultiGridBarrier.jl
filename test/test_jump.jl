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

_quiet!(m) = set_silent(m)
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
        @test JuMP.result_count(m) == 1

        # A successful structural mutation invalidates the previous result.
        @variable(m, unused_after_solve)
        @test termination_status(m) == JuMP.MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.result_count(m) == 0
        @test_throws JuMP.OptimizeNotCalled value(u1)
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
        @constraint(m, u >= -1.0e6)
        @test termination_status(m) == JuMP.MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.result_count(m) == 0
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
        @objective(m, Min,
            integral(Coef(m, 0.5) * u1 + Coef(m, 0.5) * u2 + s))
        @test termination_status(m) == JuMP.MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.result_count(m) == 0
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
        set_attribute(m, "tol", 1.0e-7)
        @test termination_status(m) == JuMP.MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.result_count(m) == 0
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
        @test termination_status(m) == JuMP.MOI.OPTIMAL
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
        @test termination_status(m) == JuMP.MOI.OPTIMAL
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

    @testset "infeasible model reports MOI.INFEASIBLE" begin
        # u ≥ 1 and u ≤ 0 simultaneously: the feasibility phase certifies
        # infeasibility (interior phase-I minimizer with positive violation),
        # which must surface as an idiomatic MOI termination status with the
        # solver's diagnostic preserved in raw_status.
        mi = MGBModel(geom); _quiet!(mi)
        @variable(mi, u)
        @constraint(mi, u == Coef(mi, 0.0), On(bd))
        @constraint(mi, u >= Coef(mi, 1.0))
        @constraint(mi, u <= Coef(mi, 0.0))
        @objective(mi, Min, integral(1.0 * u))
        optimize!(mi)
        @test termination_status(mi) == JuMP.MOI.INFEASIBLE
        @test occursin("infeasible", JuMP.raw_status(mi))
        @test JuMP.primal_status(mi) == JuMP.MOI.NO_SOLUTION
        @test_throws ArgumentError value(u)   # no solution to query
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

        # Solutions round-trip into starts (same nodal ordering). Cache every
        # value before the first mutation invalidates the previous result.
        ua_start, sa_start = value(ua), value(sa) .+ 1.0
        set_start(ua, ua_start); set_start(sa, sa_start)
        @test termination_status(ma) == JuMP.MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.primal_status(ma) == JuMP.MOI.NO_SOLUTION
        @test JuMP.result_count(ma) == 0
        @test isnan(JuMP.solve_time(ma))
        @test_throws JuMP.OptimizeNotCalled value(ua)
        optimize!(ma)
        @test termination_status(ma) == JuMP.MOI.OPTIMAL
        @test JuMP.result_count(ma) == 1
        @test maximum(abs.(value(ua) .- value(ub))) < 1e-6
    end

    @testset "model ownership" begin
        m1 = MGBModel(geom)
        m2 = MGBModel(geom)
        @variable(m1, u1)
        @variable(m2, u2)

        # Atom dictionaries are keyed only by component/operator, so mixing
        # variables from two models must be rejected before they are merged.
        @test_throws JuMP.VariableNotOwned u1 + u2
        @test_throws JuMP.VariableNotOwned u1 - u2
        @test_throws JuMP.VariableNotOwned u1 * u2

        # Coef carries model-owned spatial data and gets a targeted error.
        c1 = Coef(m1, 1.0)
        c2 = Coef(m2, 2.0)
        @test_throws ArgumentError u1 + c2
        @test_throws ArgumentError c1 * u2
        @test_throws ArgumentError c1 + c2

        # Insertion checks remain necessary for expressions built wholly from
        # one model and then submitted to another model.
        @test_throws JuMP.VariableNotOwned @constraint(m2, u1 >= 0.0)
        @test_throws ArgumentError @constraint(m2, u2 >= c1)
        @test_throws JuMP.VariableNotOwned @objective(m2, Min, integral(1.0 * u1))
        @test_throws ArgumentError @objective(m2, Min, integral(c1))
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
        @test_throws ArgumentError @constraint(mp, 1.0 <= u <= 1.0)   # equal-bounds interval = empty interior
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

        # Dirichlet-but-never-differentiated is legal but likely a modeling
        # accident: the diagnostic goes into the solve log, never the console
        # (nothing but the opt-in progress bar may write to stdout/stderr).
        mw = MGBModel(geom); _quiet!(mw)
        @variable(mw, uw); @variable(mw, sw, Broken())
        set_start(sw, 10.0)
        @constraint(mw, uw == Coef(mw, 0.0), On(bd))
        @constraint(mw, [uw; sw] in EpiPower(2.0))
        @objective(mw, Min, integral(1.0 * sw))
        @test_logs optimize!(mw)   # no log records at all
        @test termination_status(mw) == JuMP.MOI.OPTIMAL
        @test occursin("never differentiated", solver_log(mw))
        @test occursin("never differentiated", mgb_solution(mw).log)
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

    @testset "standard JuMP accessors and idioms" begin
        # attribute keys are validated; valid-but-unset keys read as nothing
        mq = MGBModel(geom); _quiet!(mq)
        @test_throws ArgumentError set_attribute(mq, "verbsoe", false)
        @test_throws ArgumentError get_attribute(mq, "nope")
        @test get_attribute(mq, "tol") === nothing

        # set_silent/unset_silent drive the "verbose" attribute
        set_silent(mq)
        @test get_attribute(mq, "verbose") === false
        unset_silent(mq)
        @test get_attribute(mq, "verbose") === true
        set_silent(mq)

        # @variable's start kwarg takes all three Coef data forms; start_value
        # reads the start back as a nodal vector (defaults to 0, never nothing)
        xf = reshape(geom.x, :, 2)
        n = size(xf, 1)
        gms = x -> 0.5 * (x[1]^2 - x[2]^2)
        @variable(mq, uq, start = gms)
        @variable(mq, sq, Broken(), start = 10.0)
        @test start_value(sq) == fill(10.0, n)
        @test start_value(uq) == [gms(xf[i, :]) for i in 1:n]
        @test all_variables(mq) == [uq, sq]
        @test !has_values(mq)

        # plain numbers in cone vectors (JuMP promotes them to GenericAffExpr,
        # which must fold back into rows): the minimal surface with its
        # constant row spelled 1.0 instead of Coef(mq, 1.0). The objective
        # exercises Integral linearity: 2∫sq - ∫sq == ∫(1.0*sq).
        @constraint(mq, uq == Coef(mq, gms), On(bd))
        @constraint(mq, [deriv(uq, :dx); deriv(uq, :dy); 1.0; sq] in EpiPower(1.0))
        @objective(mq, Min, 2.0 * integral(sq) - integral(sq))
        optimize!(mq)
        @test termination_status(mq) == JuMP.MOI.OPTIMAL
        ref = mgb_solve(Zoo.minimal_surface(mg_ref); verbose = false)
        @test _maxdiff(ref, uq, sq) < _JUMP_MATCH_TOL

        # post-solve citizenship accessors
        @test has_values(mq)
        if isdefined(JuMP, :is_solved_and_feasible)
            @test is_solved_and_feasible(mq)
            @test is_solved_and_feasible(mq; allow_local = false)   # OPTIMAL, not LOCALLY_SOLVED
        end
        if isdefined(JuMP, :assert_is_solved_and_feasible)
            @test JuMP.assert_is_solved_and_feasible(mq) === nothing
        end
        @test value(uq + sq) == value(uq) .+ value(sq)              # expression value
        @test value(2.0 * uq - 1.0) == 2.0 .* value(uq) .- 1.0
        @test value(uq; result = 1) == value(uq)
        @test objective_value(mq; result = 1) == objective_value(mq)
        @test JuMP.primal_status(mq; result = 2) == JuMP.MOI.NO_SOLUTION
        @test_throws JuMP.MOI.ResultIndexBoundsError value(uq; result = 2)
        @test_throws JuMP.MOI.ResultIndexBoundsError objective_value(mq; result = 2)
        @test_throws JuMP.MOI.ResultIndexBoundsError solution_summary(mq; result = 2)
        ss = sprint(show, solution_summary(mq))
        @test occursin("MultiGridBarrier", ss) && occursin("OPTIMAL", ss) &&
              occursin("Objective value", ss)
        @test length(sprint(show, solution_summary(mq; verbose = true))) > length(ss)

        # a two-sided interval is the two one-sided constraints, lowered
        # identically (same stacked rows after the same-region merge)
        function _boxsolve(twosided::Bool)
            mB = MGBModel(geom); _quiet!(mB)
            @variable(mB, wB, Broken(), start = 0.0)
            if twosided
                @constraint(mB, -50.0 <= wB <= 50.0)
            else
                @constraint(mB, wB >= -50.0)
                @constraint(mB, wB <= 50.0)
            end
            @objective(mB, Min, integral(1.0 * wB))
            optimize!(mB)
            value(mB[:wB])
        end
        @test _boxsolve(true) == _boxsolve(false)

        # plain-Real Dirichlet data: `u == 0.25, On(bd)` is the Coef spelling;
        # the constrained nodes carry the constant exactly
        mD = MGBModel(geom); _quiet!(mD)
        @variable(mD, uD); @variable(mD, sD, Broken(), start = 100.0)
        @constraint(mD, uD == 0.25, On(bd))
        @constraint(mD, [deriv(uD, :dx); deriv(uD, :dy); sD] in EpiPower(2.0))
        @objective(mD, Min, integral(-1.0 * uD + sD))
        optimize!(mD)
        V = size(geom.x, 1)
        bdidx = [v + (e - 1) * V for (v, e) in bd]
        @test all(value(uD)[bdidx] .== 0.25)
    end

    @testset "spatial-data sugar in expression algebra" begin
        # a raw Function or nodal vector adjacent to a model scalar means
        # Coef(model, data); the p-Laplacian spelled all three ways lowers
        # identically
        g2 = x -> x[1]^2 + x[2]^2
        xf2 = reshape(geom.x, :, 2)
        n2 = size(xf2, 1)
        gvals = [g2(xf2[i, :]) for i in 1:n2]
        function pl(mode::Symbol)
            m = MGBModel(geom); _quiet!(m)
            @variable(m, u); @variable(m, s, Broken(), start = 100.0)
            set_start(u, g2)
            if mode === :fn
                @constraint(m, u == g2, On(bd))
                @objective(m, Min, integral((x -> 0.5) * u + s))
            elseif mode === :vec
                @constraint(m, u == gvals, On(bd))
                @objective(m, Min, integral(fill(0.5, n2) * u + s))
            else
                @constraint(m, u == Coef(m, g2), On(bd))
                @objective(m, Min, integral(Coef(m, 0.5) * u + s))
            end
            @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
            optimize!(m)
            value(m[:u])
        end
        zref = pl(:coef)
        @test pl(:fn) == zref
        @test pl(:vec) == zref

        mf = MGBModel(geom); _quiet!(mf)
        @variable(mf, uf); @variable(mf, rf, Broken())
        @test (uf - g2) isa JuMP.AbstractJuMPScalar           # direct algebra
        @test (g2 * uf + rf) isa JuMP.AbstractJuMPScalar
        @test (uf - gvals) isa JuMP.AbstractJuMPScalar        # vector = nodal data ...
        @test (gvals * uf) isa JuMP.AbstractJuMPScalar        # ... ONE field product, not container scaling
        @constraint(mf, [uf - g2; rf] in EpiPower(2.0))       # cone rows, no Coef
        @constraint(mf, uf >= gvals, On(bd))
        @test_throws ArgumentError @constraint(mf, uf == g2)  # still needs On
        # validation stays eager, with clear errors
        @test_throws ArgumentError uf - ((a, b) -> a + b)     # wrong arity
        @test_throws ArgumentError uf + (x -> "hi")           # wrong return type
        @test_throws ArgumentError uf + ones(n2 + 1)          # wrong length
        # JuMP variable containers have non-Real eltype: NOT data, JuMP's own
        # scalar-array guidance error still applies
        @variable(mf, xs[1:2])
        @test_throws ErrorException uf + xs
        # broadcasting is untouched: u .+ v is still n elementwise expressions
        @test (uf .+ gvals) isa Vector && length(uf .+ gvals) == n2
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
        @test termination_status(mr) == JuMP.MOI.OPTIMAL
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
        @test termination_status(mu) == JuMP.MOI.OPTIMAL
        cv = value(c)
        @test maximum(abs.(cv .- cv[1])) < 1e-8        # genuinely one global dof
        @test cv[1] > maximum(value(uu)) - 1e-6        # dominates u ...
        @test cv[1] < maximum(value(uu)) + 1e-2        # ... and binds
    end

    @testset "spectral geometries (whole-boundary Dirichlet)" begin
        # Spectral hierarchies build their Dirichlet subspace by basis
        # truncation, so the JuMP path maps whole-boundary equality constraints
        # onto :dirichlet and rejects partial-boundary sets. The 2D model
        # mirrors the classical assemble defaults, so it must match to solver
        # tolerance.
        gs = spectral2d(n = 8)
        bs = find_boundary(gs)
        g2 = x -> x[1]^2 + x[2]^2
        ms = MGBModel(gs); _quiet!(ms)
        @variable(ms, us); @variable(ms, ss, Broken())
        set_start(us, g2); set_start(ss, 100.0)
        @constraint(ms, us == Coef(ms, g2), On(bs))
        @constraint(ms, [deriv(us, :dx); deriv(us, :dy); ss] in EpiPower(1.5))
        @objective(ms, Min, integral(Coef(ms, 0.5) * us + ss))
        optimize!(ms)
        ref = mgb_solve(assemble(amg(gs); p = 1.5); verbose = false)
        @test maximum(abs.(value(us) .- ref.z[:, 1])) < _JUMP_MATCH_TOL

        # 1D, with the same data given explicitly on both sides
        g1d = spectral1d(n = 8)
        m1 = MGBModel(g1d); _quiet!(m1)
        @variable(m1, w); @variable(m1, r, Broken())
        set_start(w, x -> x[1]^2); set_start(r, 100.0)
        @constraint(m1, w == Coef(m1, x -> x[1]^2), On(find_boundary(g1d)))
        @constraint(m1, [deriv(w, :dx); r] in EpiPower(1.5))
        @objective(m1, Min, integral(Coef(m1, 0.5) * w + r))
        optimize!(m1)
        ref1 = mgb_solve(assemble(amg(g1d); p = 1.5,
            f = x -> (0.5, 0.0, 1.0), g = x -> (x[1]^2, 100.0)); verbose = false)
        @test maximum(abs.(value(w) .- ref1.z[:, 1])) < _JUMP_MATCH_TOL

        # partial-boundary Dirichlet is rejected with the truncation
        # explanation, and so is the prolongator attribute
        mpb = MGBModel(gs); _quiet!(mpb)
        @variable(mpb, up); @variable(mpb, sp, Broken())
        set_start(sp, 100.0)
        @constraint(mpb, up == Coef(mpb, 0.0), On(bs[1:5]))
        @constraint(mpb, [deriv(up, :dx); deriv(up, :dy); sp] in EpiPower(2.0))
        @objective(mpb, Min, integral(1.0 * sp))
        @test_throws ArgumentError optimize!(mpb)
        err = try optimize!(mpb); "" catch e; sprint(showerror, e) end
        @test occursin("whole boundary", err)
        set_attribute(mpb, "prolongator", :unused)
        @test_throws ArgumentError optimize!(mpb)
    end

    @testset "convergence failure surfaces as ITERATION_LIMIT" begin
        # maxit too small for the barrier to reach its target t: optimize! must
        # catch MGBConvergenceFailure, map its :iteration_limit code to the
        # matching MOI status, and refuse to hand out values.
        g1 = x -> x[1]^2 + x[2]^2
        mf = MGBModel(geom); _quiet!(mf)
        set_attribute(mf, "maxit", 2)
        @variable(mf, u); @variable(mf, s, Broken())
        set_start(u, g1); set_start(s, 100.0)
        @constraint(mf, u == Coef(mf, g1), On(bd))
        @constraint(mf, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
        @objective(mf, Min, integral(Coef(mf, 0.5) * u + s))
        optimize!(mf)                                   # must not throw
        @test termination_status(mf) == JuMP.MOI.ITERATION_LIMIT
        @test occursin("onvergence failure", JuMP.raw_status(mf))
        @test JuMP.primal_status(mf) == JuMP.MOI.NO_SOLUTION
        @test_throws ArgumentError value(u)
        @test_throws ArgumentError objective_value(mf)
        @test_throws ArgumentError mgb_solution(mf)
        @test_throws ArgumentError solver_log(mf)
    end
end
