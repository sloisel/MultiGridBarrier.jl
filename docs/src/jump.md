```@meta
CurrentModule = MultiGridBarrier
```

# Modeling with JuMP

`MultiGridBarrierJuMP` lets you state convex variational problems in
[JuMP](https://jump.dev) syntax and solve them with the MultiGridBarrier
multigrid interior-point method. It is a `JuMP.AbstractModel` extension: the
standard macros (`@variable`, `@constraint`, `@objective`) and accessors
(`value`, `objective_value`, `termination_status`) work unchanged, but no MOI
model is ever built — `optimize!` lowers the model directly to the classical
pipeline `amg` → `assemble` → `mgb_solve`. The AMG hierarchy is constructed
automatically from the geometry and the Dirichlet constraints; it is never
user-visible.

!!! note "Requires JuMP"
    The front end is a package extension (`MultiGridBarrierJuMPExt`) that loads
    automatically once both `MultiGridBarrier` and `JuMP` are imported. JuMP is
    not a dependency of MultiGridBarrier, so add it to your environment first
    (`pkg> add JuMP`). The modeling API (`MGBModel`, `Coef`, `EpiPower`,
    `deriv`, `integral`, `set_start`, `On`, `Broken`, `Uniform`) is then
    exported from `MultiGridBarrier`.

## Setup

```@example jump
using MultiGridBarrier, JuMP, PyPlot
nothing # hide
```

## Quick tour: the p-Laplacian

```math
\min \int \tfrac{1}{2} u + s \, dx
\quad \text{s.t.} \quad s \geq \|\nabla u\|^{1.5},
\qquad u = x_1^2 + x_2^2 \text{ on } \partial\Omega .
```

A model is built over a fixed discretization, so every piece of spatial data
(`Coef`) is resolved to per-node values at modeling time (see
[the data model](@ref jump-data-model) below). Derivatives
are written `deriv(u, :dx)` where the symbol is a key of `geom.operators`;
the epigraph cone `[q...; slack] in EpiPower(p)` means
`slack ≥ ‖q‖₂ᵖ` pointwise (slack **last**).

```@example jump
geom = subdivide(fem2d_P2(), 2)
m = MGBModel(geom)
set_attribute(m, "verbose", false)
@variable(m, u)                          # conforming (inferred)
@variable(m, s, Broken())                # broken slack: one dof per node
set_start(u, x -> x[1]^2 + x[2]^2)       # initial iterate & Dirichlet lift
set_start(s, 100.0)
@constraint(m, u == Coef(m, x -> x[1]^2 + x[2]^2), On(find_boundary(geom)))
@constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
@objective(m, Min, integral(Coef(m, 0.5) * u + s))
optimize!(m)
termination_status(m)
```

```@example jump
plot(mgb_solution(m)); savefig("jump_plaplace.svg"); nothing  # hide
close()  # hide
```
![](jump_plaplace.svg)

This is exactly the package's default problem, so we can compare against the
classical API on the same geometry. The lowering produces the identical
discrete problem, so the solutions agree bit-for-bit:

```@example jump
sol_ref = mgb_solve(assemble(amg(geom); p = 1.5); verbose = false)
maximum(abs.(value(u) .- sol_ref.z[:, 1]))
```

## [The data model: nodal vectors, with sugar](@id jump-data-model)

Because the discretization is fixed when the model is built, every piece of
spatial data — coefficients, Dirichlet values, starts, obstacle heights,
variable exponents — boils down to **a vector with one value per broken
node**. That vector is the fundamental form, and every data entry point
accepts it directly:

- `Coef(m, vals::AbstractVector)`
- `set_start(u, vals::AbstractVector)`
- `EpiPower(pvals::AbstractVector)`
- `On(mask::AbstractVector{Bool})` — region membership, one Bool per node

Node `i` is vertex `v` of element `e` with `i = v + (e-1)V` where
`V = size(geom.x, 1)`: the rows of `reshape(geom.x, :, d)` are the node
coordinates in this ordering, `value` returns solutions in it, and an `On`
pair `(v, e)` selects entry `v + (e-1)V`. Functions and constants are
syntactic sugar, resolved eagerly at modeling time: `Coef(m, f)` equals
`Coef(m, [f(x) at every node coordinate x])`, and `Coef(m, 0.5)` equals
`Coef(m, fill(0.5, n))`.

```@example jump
xf = reshape(geom.x, :, 2)                       # broken-node coordinates
n  = size(xf, 1)
gv = [xf[i, 1]^2 + xf[i, 2]^2 for i in 1:n]      # nodal data, directly
value(Coef(m, gv)) == value(Coef(m, x -> x[1]^2 + x[2]^2))
```

Since data goes in and solutions come out in the same nodal ordering, vectors
round-trip: `set_start(u, value(u))` warm-starts a model from a previous
solve, and measured or precomputed nodal data (an image for ROF denoising,
say) drops in directly, without wrapping it in an interpolating closure.

## Regions: constraints on part of the domain

A constraint holds everywhere by default; adding `On(region)` restricts it to
a node set. The region is either a Bool mask over the nodal vectors (one
entry per broken node, in the ordering of
[the data model](@ref jump-data-model)) or a vector of `(vertex, element)`
pairs — the format of [`find_boundary`](@ref) and the low-level
`dirichlet_nodes` API; a mask is sugar that resolves to the pair set when the
constraint is added. Equality + `On` is a Dirichlet condition;
inequality/cone + `On` becomes a piecewise barrier, active only on the
region. Region *selection* is ordinary data preparation — a comparison on the
node coordinates gives the mask.

Here is a membrane pushed upward by a uniform load, with an obstacle imposed
only on the left half of the domain:

```@example jump
geom2 = subdivide(fem2d_P2(), 3)
left = reshape(geom2.x, :, 2)[:, 1] .< 0     # Bool mask: nodes with x₁ < 0

m2 = MGBModel(geom2)
set_attribute(m2, "verbose", false)
@variable(m2, u2); @variable(m2, s2, Broken())
set_start(s2, 100.0)
@constraint(m2, u2 == Coef(m2, 0.0), On(find_boundary(geom2)))
@constraint(m2, [deriv(u2, :dx); deriv(u2, :dy); s2] in EpiPower(2.0))
@constraint(m2, u2 >= Coef(m2, x -> 0.25 - x[1]^2 - x[2]^2), On(left))
@objective(m2, Min, integral(Coef(m2, -1.0) * u2 + s2))
optimize!(m2)
plot(mgb_solution(m2)); savefig("jump_obstacle.svg"); nothing  # hide
close()  # hide
```
![](jump_obstacle.svg)

The obstacle binds on its region (the infeasible start is handled by the
feasibility phase automatically) and is genuinely absent elsewhere — the mask
indexes solution vectors directly:

```@example jump
phi = value(Coef(m2, x -> 0.25 - x[1]^2 - x[2]^2))
println("min(u - φ) on the obstacle region:  ", minimum(value(u2)[left] .- phi[left]))
println("min(u - φ) off the region:          ", minimum(value(u2)[.!left] .- phi[.!left]))
```

## The Zoo, restated in JuMP

Every [Zoo](zoo.md) problem is a few lines in this syntax; `jump/test_zoo.jl`
checks all six against the classical constructors. Two examples. The minimal
surface uses a *constant row* inside the cone — `s ≥ ‖(∇u, 1)‖` is the
shifted Lorentz cone:

```@example jump
gu = x -> 0.5 * (x[1]^2 - x[2]^2)
ms = MGBModel(geom)
set_attribute(ms, "verbose", false)
@variable(ms, v); @variable(ms, sv, Broken())
set_start(v, gu); set_start(sv, 10.0)
@constraint(ms, v == Coef(ms, gu), On(find_boundary(geom)))
@constraint(ms, [deriv(v, :dx); deriv(v, :dy); Coef(ms, 1.0); sv] in EpiPower(1.0))
@objective(ms, Min, integral(1.0 * sv))
optimize!(ms)
ref = mgb_solve(Zoo.minimal_surface(amg(geom)); verbose = false)
maximum(abs.(value(v) .- ref.z[:, 1]))
```

Rudin–Osher–Fatemi denoising uses spatial *data inside a cone* — the
fidelity slack is `r ≥ (u - f_data)²`:

```@example jump
fdata = x -> 0.5 * tanh(5 * x[1])
mr = MGBModel(geom)
set_attribute(mr, "verbose", false)
@variable(mr, w); @variable(mr, sw, Broken()); @variable(mr, r, Broken())
set_start(w, fdata); set_start(sw, 10.0); set_start(r, 10.0)
fd = Coef(mr, fdata)
@constraint(mr, w == fd, On(find_boundary(geom)))
@constraint(mr, [deriv(w, :dx); deriv(w, :dy); sw] in EpiPower(1.0))   # s ≥ |∇u|
@constraint(mr, [w - fd; r] in EpiPower(2.0))                          # r ≥ (u-f)²
@objective(mr, Min, integral(sw + Coef(mr, 0.5) * r))
optimize!(mr)
ref = mgb_solve(Zoo.rof(amg(geom)); verbose = false)
maximum(abs.(value(w) .- ref.z[:, 1]))
```

## What lowers to what

| model content | classical object |
|---|---|
| geometry passed to `MGBModel` | `Geometry` |
| variables, kinds, Dirichlet constraints | `state_variables` + `dirichlet_nodes` → `amg(geom; dirichlet_nodes)` |
| distinct atoms `(component, operator)` used anywhere | the `D` table |
| each cone constraint | one `Convex` piece (`convex_linear` / `convex_Euclidian_power`); same-region scalar inequalities are merged into a single stacked piece |
| `On` regions on cones | `convex_piecewise` selector columns |
| `integral(...)` objective | the cost grid `f_grid` |
| starts + Dirichlet data | the initial/lift grid `g_grid` |

Untagged variables are conforming if differentiated or Dirichlet-constrained
and broken otherwise; `Broken()` / `Continuous()` / `Uniform()` override. The
notion of continuity is the geometry's connectivity `geom.t`, so slit domains
built with explicit connectivity keep their slits. The model must be in
conic form (epigraph slacks are yours to declare, as with any conic solver);
pointwise equality requires `On`; variable bounds and products of variable
expressions are rejected with explanatory errors. `dual` and spectral
geometries are not wired up yet.

## API reference

```@docs
MGBModel
Coef
deriv
integral
EpiPower
On
Broken
Continuous
Uniform
set_start
mgb_solution
solver_log
```
