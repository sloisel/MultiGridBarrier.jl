# JuMP front-end API surface: stubs + marker types.
#
# The JuMP modeling front end is a package extension (ext/MultiGridBarrierJuMPExt),
# loaded automatically when both MultiGridBarrier and JuMP are imported. Its core
# types must subtype JuMP abstract types (AbstractModel, AbstractJuMPScalar,
# AbstractVectorSet), so they are defined in the extension — but their user-facing
# NAMES must be owned here so that `using MultiGridBarrier` exposes them. This file
# declares those names: function stubs whose methods the extension supplies
# (MGBModel, Coef, EpiPower, deriv, integral, set_start, mgb_solution, solver_log),
# plus the plain marker/data types that need no JuMP (On, Broken, Uniform). Calling
# a stub without JuMP loaded raises a MethodError; see the "Modeling with JuMP"
# manual page for behavior.

"""
    MGBModel(geom::Geometry)

Construct a JuMP model over a fixed MultiGridBarrier discretization. Requires
`using JuMP` (which loads the modeling extension). Build it from any FEM
`Geometry` (e.g. `fem2d_P2()`, `subdivide(fem2d_P1(), 4)`), then use the standard
JuMP macros (`@variable`, `@constraint`, `@objective`); `optimize!` lowers the
model to `amg` → `assemble` → `mgb_solve`, constructing the AMG hierarchy
automatically from the geometry and the Dirichlet constraints. All spatial data
([`Coef`](@ref)) is evaluated eagerly at the broken nodes at modeling time.

Solver options via `set_attribute(m, key, value)` with string keys
`"prolongator"`, `"tol"`, `"t"`, `"t_feasibility"`, `"maxit"`, `"kappa"`,
`"max_newton"`, `"verbose"`, `"device"`.
"""
function MGBModel end

"""
    Coef(m, f::Function)
    Coef(m, r::Real)

Spatial data for a [`MGBModel`](@ref), evaluated eagerly at the broken nodes on
construction. `f` receives a node coordinate as a `Vector` and returns a `Real`;
`r` is a constant.
"""
function Coef end

"""
    EpiPower(p)

The pointwise epigraph set `{ [q; s] : s ≥ ‖q‖₂^p }`, used as
`@constraint(m, [q...; s] in EpiPower(p))` — slack LAST (the
`convex_Euclidian_power` `[q; s]` convention). `p` is a `Real` or a spatial
function `x -> p(x) ≥ 1`.
"""
function EpiPower end

"""
    deriv(u, op::Symbol)

The atom `op` applied to a model variable `u`, e.g. `deriv(u, :dx)`. `op` must be
a key of the geometry's `operators` (`:id`, `:dx`, `:dy`, ...). `deriv(u, :id)`
is `u`.
"""
function deriv end

"""
    integral(expr)

The objective functional `∫ expr dx`, where `expr` is affine in the model atoms;
use inside `@objective(m, Min, integral(...))`.
"""
function integral end

"""
    set_start(u, f_or_number)

Initial iterate for component `u` of a [`MGBModel`](@ref), as a spatial function
or a constant. For Dirichlet-constrained variables the start doubles as the lift
away from the constrained nodes; give slacks a comfortably feasible (large) start.
"""
function set_start end

"""
    mgb_solution(m::MGBModel) -> MGBSOL

The underlying MultiGridBarrier solution object after `optimize!` (for
`plot(sol)`, logs, etc.).
"""
function mgb_solution end

"""
    solver_log(m::MGBModel) -> String

The MGB solver iteration log after `optimize!`.
"""
function solver_log end

"""
    On(pairs::Vector{Tuple{Int,Int}})

Constraint region for a [`MGBModel`](@ref): a set of broken nodes as
`(vertex, element)` pairs — the same format as [`find_boundary`](@ref) and the
low-level `dirichlet_nodes` API. `@constraint(m, u == g, On(pairs))` is a
Dirichlet condition; an inequality/cone with `On` holds only on those nodes.
"""
struct On
    pairs::Vector{Tuple{Int,Int}}
end

"""
    Broken()

`@variable` tag for a [`MGBModel`](@ref): the variable lives in the broken space
(one degree of freedom per broken node) — the natural space for epigraph slacks.
Untagged variables are broken by default when never differentiated or
Dirichlet-constrained.
"""
struct Broken end

"""
    Uniform()

`@variable` tag for a [`MGBModel`](@ref): the variable is a single global constant.
"""
struct Uniform end

export MGBModel, Coef, EpiPower, deriv, integral, set_start, mgb_solution,
       solver_log, On, Broken, Uniform
