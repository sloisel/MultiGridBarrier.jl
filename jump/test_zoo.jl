# Zoo cross-validation for MultiGridBarrierJuMP.
#
# Rebuilds all six Zoo problems (2D, default parameters) through the JuMP
# modeling layer and compares every state component against the classical
# constructors `mgb_solve(Zoo.<problem>(amg(geom)))` on the same geometry.
# The discrete problems are identical up to D-row / barrier-piece ordering,
# so solutions must agree to solver tolerance.
#
# Run:  julia --project=<env with MultiGridBarrier (dev) + JuMP> test_zoo.jl
# Exits nonzero on failure.

using MultiGridBarrier, JuMP, LinearAlgebra
include("MultiGridBarrierJuMP.jl")
using .MultiGridBarrierJuMP

const TOL_MATCH = 1e-6
geom = subdivide(fem2d_P2(), 2)
mg_ref = amg(geom)
bd = find_boundary(geom)

results = Tuple{String,Bool,Float64}[]   # (label, pass, worst diff)

function check!(name, m, ref, comps::Vector{Int})
    worst = 0.0
    for (mycol, refcol) in enumerate(comps)
        d = maximum(abs.(value(m[Symbol(_compname(m, mycol))]) .- ref.z[:, refcol]))
        worst = max(worst, d)
    end
    push!(results, (name, worst < TOL_MATCH, worst))
    println(rpad(name, 24), worst < TOL_MATCH ? "PASS" : "FAIL",
            "   max component diff = ", worst)
end
_compname(m, k) = m.comps[k].name   # declaration order == MGBSOL column order

quiet!(m) = set_attribute(m, "verbose", false)
refsolve(prob) = mgb_solve(prob; verbose = false)

# ---------------------------------------------------------------------------
# 1. p_harmonic (vectorial p-Laplacian): min ∫ 0.5 u₁ + 0.5 u₂ + s,
#    s ≥ ‖(∇u₁, ∇u₂)‖_F^p, uᵢ = g on ∂Ω.  p = 1.5.
# ---------------------------------------------------------------------------
let
    g1 = x -> x[1] * x[2]
    m = MGBModel(geom); quiet!(m)
    @variable(m, u1); @variable(m, u2); @variable(m, s, Broken())
    set_start(u1, g1); set_start(u2, 0.0); set_start(s, 100.0)
    @constraint(m, u1 == Coef(m, g1), On(bd))
    @constraint(m, u2 == Coef(m, 0.0), On(bd))
    @constraint(m, [deriv(u1, :dx); deriv(u1, :dy);
                    deriv(u2, :dx); deriv(u2, :dy); s] in EpiPower(1.5))
    @objective(m, Min, integral(Coef(m, 0.5) * u1 + Coef(m, 0.5) * u2 + s))
    optimize!(m)
    check!("p_harmonic", m, refsolve(Zoo.p_harmonic(mg_ref)), [1, 2, 3])
end

# ---------------------------------------------------------------------------
# 2. minimal_surface: min ∫ s,  s ≥ √(1 + |∇u|²)  (q = (∇u, 1), p = 1),
#    u = ½(x² − y²) on ∂Ω.  Exercises a constant row inside EpiPower
#    (rectangular system → square zero-padding).
# ---------------------------------------------------------------------------
let
    gu = x -> 0.5 * (x[1]^2 - x[2]^2)
    m = MGBModel(geom); quiet!(m)
    @variable(m, u); @variable(m, s, Broken())
    set_start(u, gu); set_start(s, 10.0)
    @constraint(m, u == Coef(m, gu), On(bd))
    @constraint(m, [deriv(u, :dx); deriv(u, :dy); Coef(m, 1.0); s] in EpiPower(1.0))
    @objective(m, Min, integral(1.0 * s))
    optimize!(m)
    check!("minimal_surface", m, refsolve(Zoo.minimal_surface(mg_ref)), [1, 2])
end

# ---------------------------------------------------------------------------
# 3. norton_hoff (power-law elasticity): min ∫ 0.5 u₁ + 0.5 u₂ + s,
#    s ≥ |ε(u)|_F^p with q = (ε₁₁, ε₂₂, (∂u₁/∂y + ∂u₂/∂x)/√2).  p = 1.5.
#    Exercises expression rows and nc < ni zero-row padding.
# ---------------------------------------------------------------------------
let
    g1 = x -> x[1] * x[2]
    m = MGBModel(geom); quiet!(m)
    @variable(m, u1); @variable(m, u2); @variable(m, s, Broken())
    set_start(u1, g1); set_start(u2, 0.0); set_start(s, 100.0)
    @constraint(m, u1 == Coef(m, g1), On(bd))
    @constraint(m, u2 == Coef(m, 0.0), On(bd))
    @constraint(m, [deriv(u1, :dx); deriv(u2, :dy);
                    (deriv(u1, :dy) + deriv(u2, :dx)) / sqrt(2.0);
                    s] in EpiPower(1.5))
    @objective(m, Min, integral(Coef(m, 0.5) * u1 + Coef(m, 0.5) * u2 + s))
    optimize!(m)
    check!("norton_hoff", m, refsolve(Zoo.norton_hoff(mg_ref)), [1, 2, 3])
end

# ---------------------------------------------------------------------------
# 4. rof (TV denoising): min ∫ s + (λ/2) r,  s ≥ |∇u|,  r ≥ (u − f_data)²,
#    u = f_data on ∂Ω.  λ = 1.  Exercises x-dependent affine shift inside a
#    power cone ([u - Coef; r] with p = 2).
# ---------------------------------------------------------------------------
let
    fdata = x -> 0.5 * tanh(5 * x[1])
    m = MGBModel(geom); quiet!(m)
    @variable(m, u); @variable(m, s, Broken()); @variable(m, r, Broken())
    set_start(u, fdata); set_start(s, 10.0); set_start(r, 10.0)
    fd = Coef(m, fdata)
    @constraint(m, u == fd, On(bd))
    @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.0))
    @constraint(m, [u - fd; r] in EpiPower(2.0))
    @objective(m, Min, integral(s + Coef(m, 0.5) * r))
    optimize!(m)
    check!("rof", m, refsolve(Zoo.rof(mg_ref)), [1, 2, 3])
end

# ---------------------------------------------------------------------------
# 5. two_sided_obstacle: min ∫ 2u + 0.5 s,  s ≥ |∇u|²,  −0.1 ≤ u ≤ 1,
#    u = 0 on ∂Ω.  Exercises global scalar bounds with Number rhs.
# ---------------------------------------------------------------------------
let
    m = MGBModel(geom); quiet!(m)
    @variable(m, u); @variable(m, s, Broken())
    set_start(u, 0.0); set_start(s, 10.0)
    @constraint(m, u == Coef(m, 0.0), On(bd))
    @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
    @constraint(m, u >= -0.1)
    @constraint(m, u <= 1.0)
    @objective(m, Min, integral(Coef(m, 2.0) * u + Coef(m, 0.5) * s))
    optimize!(m)
    check!("two_sided_obstacle", m, refsolve(Zoo.two_sided_obstacle(mg_ref)), [1, 2])
end

# ---------------------------------------------------------------------------
# 6. elastoplastic_torsion: min ∫ 4u + 0.5 s,  s ≥ |∇u|²,  s ≤ smax² = 1,
#    u = 0 on ∂Ω.
# ---------------------------------------------------------------------------
let
    m = MGBModel(geom); quiet!(m)
    @variable(m, u); @variable(m, s, Broken())
    set_start(u, 0.0); set_start(s, 0.5)
    @constraint(m, u == Coef(m, 0.0), On(bd))
    @constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(2.0))
    @constraint(m, s <= 1.0)
    @objective(m, Min, integral(Coef(m, 4.0) * u + Coef(m, 0.5) * s))
    optimize!(m)
    check!("elastoplastic_torsion", m, refsolve(Zoo.elastoplastic_torsion(mg_ref)), [1, 2])
end

# ---------------------------------------------------------------------------
println()
npass = count(r -> r[2], results)
println(npass, " / ", length(results), " Zoo problems match the classical API")
exit(npass == length(results) ? 0 : 1)
