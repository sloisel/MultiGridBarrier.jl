# Demo for the MultiGridBarrierJuMP draft.
# Requires JuMP in the active environment (it is NOT a dependency of
# MultiGridBarrier): julia> import Pkg; Pkg.add("JuMP")
#
# Run from this directory with the package's environment plus JuMP, e.g.
#   julia --project=@. -e 'include("demo.jl")'

using MultiGridBarrier, JuMP
include("MultiGridBarrierJuMP.jl")
using .MultiGridBarrierJuMP

# ---------------------------------------------------------------------------
# Demo 1: p-Laplacian. This is exactly the package default problem
#   min ∫ 0.5*u + s   s.t.  s ≥ ‖∇u‖^p,  u = x²+y² on ∂Ω
# so it should reproduce mgb_solve(assemble(amg(...); p = 1.5)).
# ---------------------------------------------------------------------------

geom = subdivide(fem2d_P2(), 2)
p = 1.5

m = MGBModel(geom)
@variable(m, u)
@variable(m, s, Broken())
set_start(u, x -> x[1]^2 + x[2]^2)   # smooth lift, mirrors the package default
set_start(s, 100.0)                  # comfortably feasible slack start

@constraint(m, u == Coef(m, x -> x[1]^2 + x[2]^2), On(find_boundary(geom)))
@constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(p))
@objective(m, Min, integral(Coef(m, 0.5) * u + s))

optimize!(m)
println("demo 1 (p-Laplace):   ", termination_status(m),
        ", objective = ", objective_value(m))

# Cross-check against the classical API on the same hierarchy.
sol_ref = mgb_solve(assemble(amg(geom); p = p); verbose = false)
zu = value(u)
println("  max |u - u_ref| = ", maximum(abs.(zu .- sol_ref.z[:, 1])))

# ---------------------------------------------------------------------------
# Demo 2: an obstacle active only on the left half of the domain.
# Region selection is plain data preparation: build the (vertex, element)
# pairs yourself, then pass them with On(...).
# (Finer mesh than demo 1: coarser grids have no left node where the obstacle
# exceeds the free solution, so contact would be vacuous.)
# ---------------------------------------------------------------------------

geom2 = subdivide(fem2d_P2(), 3)
Vn, Nn = size(geom2.x, 1), size(geom2.x, 2)
left = [(v, e) for e in 1:Nn for v in 1:Vn if geom2.x[v, e, 1] < 0]

m2 = MGBModel(geom2)
@variable(m2, u2)
@variable(m2, s2, Broken())
set_start(s2, 100.0)
@constraint(m2, u2 == Coef(m2, 0.0), On(find_boundary(geom2)))
@constraint(m2, [deriv(u2, :dx); deriv(u2, :dy); s2] in EpiPower(2.0))
@constraint(m2, u2 >= Coef(m2, x -> 0.25 - x[1]^2 - x[2]^2), On(left))
@objective(m2, Min, integral(Coef(m2, -1.0) * u2 + s2))

optimize!(m2)
println("demo 2 (half-obstacle): ", termination_status(m2),
        ", objective = ", objective_value(m2))

# The obstacle must hold on the left with actual contact (gap ≈ 0 somewhere),
# and be genuinely absent on the right (where the membrane dips below φ).
zu2 = value(u2)
phi = value(Coef(m2, x -> 0.25 - x[1]^2 - x[2]^2))
lin = [v + (e - 1) * Vn for (v, e) in left]
rgt = setdiff(1:length(zu2), lin)
gapL = zu2[lin] .- phi[lin]
println("  min(u - φ) on left  (expect ≥ 0):  ", minimum(gapL))
println("  contact nodes on left (expect > 0): ", count(<(1e-4), gapL))
println("  min(u - φ) on right (expect < 0):  ", minimum(zu2[rgt] .- phi[rgt]))

# plot(mgb_solution(m2))   # uncomment to visualize
