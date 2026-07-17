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
    `deriv`, `integral`, `set_start`, `On`, `Broken`, `Continuous`, `Uniform`,
    `mgb_solution`, `solver_log`) is then exported from `MultiGridBarrier`.

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
— a `Function` of the coordinates, a nodal vector, or a constant — is
resolved to per-node values at modeling time; next to a model variable it
appears directly in the algebra (`u == g`), and [`Coef`](@ref) is the
explicit wrapper (see [the data model](@ref jump-data-model) below).
Derivatives are written `deriv(u, :dx)` where the symbol is a key of
`geom.operators`; the epigraph cone `[q...; slack] in EpiPower(p)` means
`slack ≥ ‖q‖₂ᵖ` pointwise (slack **last**).

```@example jump
g = x -> x[1]^2 + x[2]^2                     # boundary data
geom = subdivide(fem2d_P2(), 2)
m = MGBModel(geom)
set_silent(m)
@variable(m, u)                              # conforming (inferred)
@variable(m, s, Broken(), start = 100.0)     # broken slack: one dof per node
set_start(u, g)                              # initial iterate & Dirichlet lift
@constraint(m, u == g, On(find_boundary(geom)))
@constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))
@objective(m, Min, integral(0.5 * u + s))
optimize!(m)
termination_status(m)
```

```@example jump
plot(mgb_solution(m))
savefig("jump_plaplace.svg"); nothing  # hide
close()  # hide
```
![](jump_plaplace.svg)

`mgb_solution(m)` returns the underlying classical solution object (the same
`MGBSOL` that `mgb_solve` produces), so all of [Plotting](plotting.md)
applies. Equivalently, `value` evaluates variables, `deriv` atoms, `Coef`
data, and affine expressions of them as plain nodal vectors on `geom`, so
anything in the nodal ordering plots directly:

```julia
plot(mgb_solution(m))                # component 1: u
plot(mgb_solution(m), 2)             # component 2: the slack s
plot(geom, value(u))                 # same picture as plot(mgb_solution(m))
plot(geom, value(deriv(u, :dx)))     # the derivative field ∂u/∂x
plot(geom, value(u - g))             # u minus the boundary data
```

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
- `On(geom, mask::AbstractVector{Bool})` — region membership, one Bool per node

Node `i` is vertex `v` of element `e` with `i = v + (e-1)V` where
`V = size(geom.x, 1)`: the rows of `reshape(geom.x, :, d)` are the node
coordinates in this ordering, `value` returns solutions in it, and an `On`
pair `(v, e)` selects entry `v + (e-1)V`. Functions and constants are
syntactic sugar, resolved eagerly at modeling time: `Coef(m, f)` equals
`Coef(m, [f(x) at every node coordinate x])`, and `Coef(m, 0.5)` equals
`Coef(m, fill(0.5, n))`.

Next to a model variable, data needs no wrapper at all: `u == g`,
`u >= phi_vals`, `a * u`, and `w - image_vec` resolve the `Function` or nodal
vector through the adjacent operand's model, exactly as if wrapped in `Coef`.
This gives Real vectors *field* semantics in the scalar algebra — `u + v` is
one expression — while the broadcast `u .+ v` keeps its usual meaning (`n`
elementwise expressions). In particular write `@constraint(m, u >= vals)`,
not `u .>= vals`: the broadcast form makes `n` copies of a *global*
constraint, one per entry. Matrices are not data (nodal data is the flat
length-`V*N` vector), and a Bool vector next to a variable is rejected as
ambiguous — `On(geom, mask)` for a region, `Coef(m, v)` for genuine 0/1
data. The explicit `Coef(m, ...)` form remains for positions with no
adjacent variable, such as a pure-data cone row built from a function.

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

A constraint holds everywhere by default; adding `On(...)` restricts it to a
node set of `(vertex, element)` pairs — the format of [`find_boundary`](@ref)
and the low-level `dirichlet_nodes` API. `On(geom, mask)` is grid-level sugar
for the same thing: a Bool mask over the nodal vectors (one entry per broken
node, in the ordering of [the data model](@ref jump-data-model)), converted
eagerly to the pair set. Equality + `On` is a Dirichlet condition;
inequality/cone + `On` becomes a piecewise barrier, active only on the
region. Region *selection* is ordinary data preparation — a comparison on the
node coordinates gives the mask.

Here is a membrane pushed upward by a uniform load, with an obstacle imposed
only on the left half of the domain:

```@example jump
geom2 = subdivide(fem2d_P2(), 3)
left = reshape(geom2.x, :, 2)[:, 1] .< 0     # Bool mask: nodes with x₁ < 0
phi = x -> 0.25 - x[1]^2 - x[2]^2            # the obstacle

m2 = MGBModel(geom2)
set_silent(m2)
@variable(m2, u2); @variable(m2, s2, Broken(), start = 100.0)
crBC  = @constraint(m2, u2 == 0.0, On(find_boundary(geom2)))
@constraint(m2, [deriv(u2, :dx); deriv(u2, :dy); s2] in EpiPower(2.0))
crObs = @constraint(m2, u2 >= phi, On(geom2, left))
@objective(m2, Min, integral(-1.0 * u2 + s2))
optimize!(m2)
plot(mgb_solution(m2))
savefig("jump_obstacle.svg"); nothing  # hide
close()  # hide
```
![](jump_obstacle.svg)

The obstacle binds on its region (the infeasible start is handled by the
feasibility phase automatically) and is genuinely absent elsewhere — `value`
evaluates the gap `u - φ` as an expression, and the mask indexes it directly:

```@example jump
gap = value(u2 - phi)
println("min(u - φ) on the obstacle region:  ", minimum(gap[left]))
println("min(u - φ) off the region:          ", minimum(gap[.!left]))
```

## Duals

After `optimize!`, `dual(cr)` returns the constraint's Lagrange multiplier as
a nodal vector, in the same ordering as [`value`](@ref jump-data-model). For
inequality and cone constraints it is a *density* with respect to the volume
measure — for the obstacle above, the contact pressure — read off the barrier
gradient at the final barrier parameter, and accurate to `O(tol)` like the
primal. It is zero off an `On` region, zero at nodes with zero quadrature
weight (where the constraint is not collocated), nonnegative for `>=`,
nonpositive for `<=` (the MOI convention; signs flip for `Max`, as in MOI).
For `EpiPower` and `SecondOrderCone` rows, `dual` reports the multiplier of
the pointwise epigraph inequality `s ≥ ‖q‖ᵖ` (not an MOI dual-cone vector).

One normalization rule to keep in mind: an inequality's dual is a density
with respect to the *volume* measure, whether or not the constraint carries
an `On` region. Its pointwise values are meaningful when the region has
positive volume (an obstacle on a subdomain); on a lower-dimensional node
set — a Signorini-type `u >= 0, On(find_boundary(geom))` — the true
multiplier is a boundary measure, and the volume-density values grow like
the reciprocal mesh weight under refinement. Integrals remain correct in
every case: `sum(geom.w .* dual(cr) .* φ)` is the pairing `⟨multiplier, φ⟩`
regardless of the region's dimension. Equality duals (below) are the same
object reported the other way — the multiplier's raw node masses,
undivided — so *their* sums are the meaningful quantity.

```@example jump
pressure = dual(crObs)                       # contact pressure, one value per node
println("pressure off the region:       ", maximum(pressure[.!left]))
println("min pressure on the region:    ", minimum(pressure[left]))
println("complementarity ⟨pressure,gap⟩: ", sum(geom2.w .* pressure .* gap))
```

The dual of a Dirichlet equality is the *reaction*: the leftover objective
gradient at the pinned coordinates (for a membrane, the boundary force
holding the solution at `g`; discretely, the flux `∂u/∂ν`). It is reported as
**raw per-broken-node forces** — element shares, in the nodal ordering — not
as a boundary density: an `On` set need not be a manifold, so no boundary
measure is assumed. Coincident broken copies of a glued node sum to the
physical nodal force, and the sum over the region is the total force:

```@example jump
reaction = dual(crBC)
println("total boundary reaction: ", sum(reaction))
```

## The Zoo, restated in JuMP

Every [Zoo](zoo.md) problem is a few lines in this syntax; `jump/test_zoo.jl`
checks all six against the classical constructors. Two examples. The minimal
surface uses a *constant row* inside the cone — `s ≥ ‖(∇u, 1)‖` is the
shifted Lorentz cone (a plain `1.0` works; spatial data would be a `Coef`):

```@example jump
gu = x -> 0.5 * (x[1]^2 - x[2]^2)
ms = MGBModel(geom)
set_silent(ms)
@variable(ms, v, start = gu); @variable(ms, sv, Broken(), start = 10.0)
@constraint(ms, v == gu, On(find_boundary(geom)))
@constraint(ms, [deriv(v, :dx); deriv(v, :dy); 1.0; sv] in EpiPower(1.0))
@objective(ms, Min, integral(sv))
optimize!(ms)
ref = mgb_solve(Zoo.minimal_surface(amg(geom)); verbose = false)
maximum(abs.(value(v) .- ref.z[:, 1]))
```

Rudin–Osher–Fatemi denoising uses spatial *data inside a cone* — the
fidelity slack is `r ≥ (u - f_data)²`:

```@example jump
fdata = x -> 0.5 * tanh(5 * x[1])
mr = MGBModel(geom)
set_silent(mr)
@variable(mr, w, start = fdata)
@variable(mr, sw, Broken(), start = 10.0); @variable(mr, r, Broken(), start = 10.0)
@constraint(mr, w == fdata, On(find_boundary(geom)))
@constraint(mr, [deriv(w, :dx); deriv(w, :dy); sw] in EpiPower(1.0))   # s ≥ |∇u|
@constraint(mr, [w - fdata; r] in EpiPower(2.0))                       # r ≥ (u-f)²
@objective(mr, Min, integral(sw + 0.5 * r))
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
pointwise equality requires `On`; `@variable` bounds and products of variable
expressions are rejected with explanatory errors (spell bounds as pointwise
constraints — the two-sided `@constraint(m, lo <= u <= hi)` works and lowers
to two stacked inequalities). Spectral geometries work
too, with one restriction inherited from their hierarchy: the spectral
Dirichlet subspace is built by basis truncation, so a Dirichlet condition
there must cover exactly the whole boundary (`find_boundary(geom)`).
Constraint duals are available after the solve — see [Duals](@ref).

## Standard JuMP accessors

The usual JuMP workflow works unchanged: `set_silent`/`unset_silent` (they
drive the `"verbose"` attribute), `solution_summary`, `is_solved_and_feasible`,
`termination_status` (`MOI.OPTIMAL` on success — the problems are convex),
`objective_value`, `value` (on variables, `deriv` atoms, `Coef` data, and
affine expressions, always as a nodal vector), `dual`/`has_duals`/`dual_status`
(see [Duals](@ref)), `all_variables`, `start_value`,
`has_values`, `raw_status`, `solve_time`, and the `result` keyword (validated —
there is exactly one result). `@variable(m, u, start = data)` accepts the same
three data forms as [`set_start`](@ref), and the `integral` objective is
linear: `2*integral(u) - integral(s)` equals `integral(2*u - s)`.
`is_solved_and_feasible(m)` is the recommended success check, and
`solution_summary(m; verbose = true)` appends the solver iteration log (the
same text as [`solver_log`](@ref)). JuMP's `SecondOrderCone` is accepted in
its own epigraph-**first** convention: `[s; q...] in SecondOrderCone()`
lowers identically to `[q...; s] in EpiPower(1.0)`.

Three deliberate departures. Spatial data rides the scalar algebra: a
`Function` or Real-eltype vector adjacent to a model variable is nodal data
forming *one* field expression (`u == g`, `u + v`, `phi_vals * u`), where
generic JuMP throws use-broadcasting errors for `+`/`-` and container-scales
for `*`; broadcasting itself (`u .+ v`) keeps its usual elementwise meaning.
Models are add-only: `delete`, `fix`, and the
`set_normalized_*` modification API are not implemented — models are cheap, so
rebuild instead (any structural mutation invalidates the previous result
anyway). And the exported name `MGBModel` is the constructor *function*, not
the model's type (the type lives in the extension because it must subtype
`JuMP.AbstractModel`); annotate helper functions with
`f(m::JuMP.AbstractModel)` if you need dispatch.

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
