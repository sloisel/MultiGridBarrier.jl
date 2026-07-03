# MultiGridBarrierJuMP — JuMP front end for MultiGridBarrier.jl

Model convex variational problems in JuMP syntax; solve them with the
MultiGridBarrier multigrid interior-point method. This is a
`JuMP.AbstractModel` extension: JuMP's macros (`@variable`, `@constraint`,
`@objective`) and standard accessors (`value`, `objective_value`,
`termination_status`) work unchanged, but no MOI model is ever built —
`optimize!` lowers the model directly to the classical pipeline
`amg` → `assemble` → `mgb_solve`. The AMG hierarchy is constructed
automatically from the geometry and the Dirichlet constraints and is never
user-visible.

**Status: working draft.** Validated against JuMP v1.30.1 / Julia 1.12.6:
all six Zoo problems (`p_harmonic`, `minimal_surface`, `norton_hoff`, `rof`,
`two_sided_obstacle`, `elastoplastic_torsion`) rebuilt in JuMP syntax
reproduce the classical constructors' solutions component-wise — five of six
**bit-identically** (max diff 0.0), `norton_hoff` to one ulp (4.4e-16); see
`test_zoo.jl`. `demo.jl` additionally exercises region-restricted constraints
(real contact on the region, none off it) and the feasibility phase.
Not yet a registered package; see *Packaging* below.

## Quick start

```julia
using MultiGridBarrier, JuMP
include("MultiGridBarrierJuMP.jl"); using .MultiGridBarrierJuMP

geom = subdivide(fem2d_P2(), 2)                 # any FEM Geometry
m = MGBModel(geom)
@variable(m, u)                                 # conforming (inferred)
@variable(m, s, Broken())                       # broken slack, one dof per node
set_start(u, x -> x[1]^2 + x[2]^2)              # initial iterate / Dirichlet lift
set_start(s, 100.0)
@constraint(m, u == Coef(m, x -> x[1]^2 + x[2]^2), On(find_boundary(geom)))
@constraint(m, [deriv(u, :dx); deriv(u, :dy); s] in EpiPower(1.5))  # s ≥ ‖∇u‖^1.5
@objective(m, Min, integral(Coef(m, 0.5) * u + s))
optimize!(m)
termination_status(m), objective_value(m)
value(u)                                        # values at the broken nodes
plot(mgb_solution(m))                           # underlying MGBSOL
```

## Modeling reference

### Model and data

| construct | meaning |
|---|---|
| `MGBModel(geom::Geometry)` | model over a fixed discretization; all spatial data is evaluated **eagerly** at the broken nodes at modeling time |
| `Coef(m, f::Function)` | spatial data; `f` maps a coordinate vector to a `Real`, evaluated at every node on construction |
| `Coef(m, r::Real)` | constant spatial data |
| `deriv(u, :dx)` | operator atom; the symbol must be a key of `geom.operators` (`:id`, `:dx`, `:dy`, …). `deriv(u, :id)` is `u`. Operators do not compose |
| `integral(expr)` | the objective functional `∫ expr dx`; `expr` must be affine in the atoms |
| `On(pairs)` | constraint region: broken nodes as `(vertex, element)` pairs — the same format as `find_boundary(geom)` and the low-level `dirichlet_nodes` API. Boundary or interior. Region *selection* is plain data preparation (filter the pairs yourself); the modeling API takes only the resolved array |

### Variables

- `@variable(m, u)` — kind inferred at `optimize!`: **conforming** if the
  variable is differentiated or Dirichlet-constrained, **broken** otherwise.
- `@variable(m, s, Broken())` / `@variable(m, c, Uniform())` — explicit kinds.
  Broken variables live one dof per broken node (`:full`); `Uniform()` is the
  global-constant subspace (`:uniform`).
- `set_start(u, f_or_number)` — initial iterate for the component. For
  Dirichlet-constrained variables the start doubles as the lift away from the
  constrained nodes, so passing the boundary-data function is the natural
  choice. Give slacks a comfortably feasible start (large enough); an
  infeasible start is legal and triggers the feasibility phase automatically.
- Variable bounds / integrality in `@variable` are **rejected** — write
  pointwise constraints instead.

### Constraints

All constraints hold pointwise at the (broken) quadrature nodes; with
`On(pairs)`, only at those nodes.

| syntax | lowers to |
|---|---|
| `@constraint(m, u == g_, On(pairs))` | Dirichlet condition: node mask (`dirichlet_nodes`) + lift from `g_` values. Equality **requires** `On` (pointwise equality has empty interior) and must involve exactly one variable with constant coefficient |
| `@constraint(m, expr >= 0)` (or `<=`, or `expr >= Coef(...)`) | pointwise linear barrier (`convex_linear`) |
| `@constraint(m, exprs in MOI.Nonnegatives(k))` | k pointwise linear barriers |
| `@constraint(m, [q...; s] in EpiPower(p))` | `s ≥ ‖q‖₂^p` (`convex_Euclidian_power`). **Slack last.** Rows are affine expressions; constant rows (e.g. `Coef(m, 1.0)` for the minimal surface) are allowed. `p` is a `Real` or a spatial function `x -> p(x) ≥ 1` |
| `@constraint(m, [t; x...] in SecondOrderCone())` | JuMP's t-first convention, reordered internally to `EpiPower(1)` |
| any inequality/cone + `On(pairs)` | the constraint holds only on the region (`convex_piecewise` selector) |

Products of two variable expressions are rejected (the model must be in conic
form; introduce a slack + cone, as in every conic solver).

### Objective

`@objective(m, Min, integral(affine expr))` (or `Max`). The objective must be
an integral of an atom-affine expression; nonlinear integrands are expressed
by epigraph slacks, exactly as with Mosek/Clarabel through JuMP.

### Solver options and results

Options via `set_attribute(m, key, value)` with string keys, forwarded to
`amg` / `mgb_solve`: `"prolongator"`, `"tol"`, `"t"`, `"t_feasibility"`,
`"maxit"`, `"kappa"`, `"max_newton"`, `"verbose"`, `"device"`.

Results: `value(u)` and `value(deriv(u, :dx))` (vectors over broken nodes,
flat `reshape(geom.x, V*N, d)` row order), `objective_value`,
`termination_status` (`LOCALLY_SOLVED` / `OTHER_ERROR` /
`OPTIMIZE_NOT_CALLED`), `raw_status`, `solve_time`, `primal_status`;
`mgb_solution(m)` returns the underlying `MGBSOL` (so `plot` works),
`solver_log(m)` the iteration log. Conforming components have equal values on
coincident node copies; broken components genuinely may not (`value` reports
the honest per-copy values).

## How the lowering works

At `optimize!`, the model compiles to the classical API with no algorithmic
changes — the theory of the underlying method transfers verbatim:

| model content | classical MultiGridBarrier object |
|---|---|
| geometry passed to `MGBModel` | `Geometry` |
| variables + kinds + Dirichlet constraints | `state_variables` matrix + `dirichlet_nodes` masks → `amg(geom; dirichlet_nodes)` (hierarchy built here, automatically) |
| the set of distinct atoms `(component, operator)` appearing anywhere | the `D` table, in first-appearance order |
| each cone constraint | one `Convex` piece (`convex_linear` / `convex_Euclidian_power`), with per-node `A_grid`/`b_grid`/`p_grid` built from the (eager) coefficients |
| regions (`On`) on cones | `convex_piecewise` `select_grid` indicator columns |
| `integral(...)` objective | the cost grid `f_grid` (per-node rows over D) |
| starts + Dirichlet data | the initial/lift grid `g_grid` |
| — | `assemble(mg; state_variables, D, f_grid, g_grid, Q)` then `mgb_solve` |

One MultiGridBarrier-side invariant the lowering enforces:
`convex_Euclidian_power` kernels reconstruct a **square** per-node `A` with
`length(idx) == size(A, 1)`; rectangular systems (constant rows, or more atoms
than rows) are zero-padded — padded zero q-rows extend `q` by zeros, which is
norm-neutral, and the slack row stays last.

## Tests

- `demo.jl` — p-Laplacian (cross-checked against
  `mgb_solve(assemble(amg(geom); p = 1.5))`, bit-identical) and a
  region-restricted obstacle (contact on the region, none off it;
  feasibility phase exercised).
- `test_zoo.jl` — all six Zoo problems rebuilt in JuMP syntax and compared
  component-wise against the classical constructors on the same geometry.

Environment for running (JuMP is *not* a dependency of MultiGridBarrier):

```julia
import Pkg
Pkg.activate("jumpenv"; shared = false)
Pkg.develop(path = "path/to/MultiGridBarrier.jl")
Pkg.add("JuMP")
```

then `julia --project=jumpenv jump/test_zoo.jl`.

## JuMP version notes

Validated on JuMP v1.30.1. The version-sensitive seam is the constraint
macro: scalar comparisons whose sides are not plain `Number`s are routed
through *variadic* `build_constraint` methods on JuMP marker sets (`Zeros`,
`Nonnegatives`, `Nonpositives`, and on newer versions `GreaterThanZero` /
`LessThanZero` / `EqualToZero`); each needs an explicit disambiguating method
against the `On` extra-argument hook (present in the module, guarded by
`isdefined`). Additionally `Base.broadcastable` must be defined for the model
type (JuMP's `model_convert` broadcasts over it), and expressions like
`s + Coef(m, λ)*r` inside the macros require `MutableArithmetics`
`promote_operation` / `Base.convert` methods for the scalar types (present in
the module, reached through `JuMP._MA`).

One MultiGridBarrier-side kernel invariant worth knowing: the
`convex_Euclidian_power` gradient/Hessian scatter is **last-match-wins**, so
`idx` entries must be distinct — the lowering pads rectangular systems with
distinct spare D rows (guaranteed to exist by the `:id`-row seeding), never
with repeated indices.

## Deliberately missing (v1 scope)

- `dual` — not wired up yet (available in principle from the barrier gradient
  at the solution; for the obstacle problem it is the contact pressure).
- Spectral geometries (`spectral1d/2d`) — the Dirichlet-subspace naming
  differs; FEM geometries only for now.
- Variable bounds in `@variable`, constraint deletion, feasibility-only
  models, anywhere-evaluation of solutions (report at nodes; use
  `interpolate` from the main package if needed).

## Packaging

Currently a single `include`-able module for iteration. The plan of record:
promote to a small glue package (`MultiGridBarrierJuMP.jl`, possibly a
separately-registered subdirectory of the main repo) with a hard JuMP
dependency — JuMP has been semver-stable at 1.x since 2022, so unlike a
hypothetical InfiniteOpt coupling this does not need weak-dependency
quarantine. MultiGridBarrier itself gains no dependencies.
