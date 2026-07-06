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
`using JuMP` (which loads the modeling extension). Build it from any FEM or
spectral `Geometry` (e.g. `fem2d_P2()`, `subdivide(fem2d_P1(), 4)`,
`spectral2d(n = 16)`; spectral Dirichlet conditions must cover the whole
boundary), then use the standard
JuMP macros (`@variable`, `@constraint`, `@objective`); `optimize!` lowers the
model to `amg` → `assemble` → `mgb_solve`, constructing the AMG hierarchy
automatically from the geometry and the Dirichlet constraints. All spatial data
([`Coef`](@ref)) is per-broken-node vectors, resolved eagerly at modeling time
(functions and constants are sugar for their nodal samples).

Solver options via `set_attribute(m, key, value)` with string keys
`"prolongator"`, `"tol"`, `"t"`, `"t_feasibility"`, `"maxit"`, `"kappa"`,
`"max_newton"`, `"verbose"`, `"device"`.
"""
function MGBModel end

"""
    Coef(m, v::AbstractVector{<:Real})
    Coef(m, f::Function)
    Coef(m, r::Real)

Spatial data for a [`MGBModel`](@ref): one value per broken node. The nodal
vector `v` is the fundamental form; it must have length `V*N` (`V` vertices
per element, `N` elements), where entry `i = v + (e-1)V` belongs to vertex `v`
of element `e` — the ordering of the rows of `reshape(geom.x, :, d)`, of the
vectors returned by `value`, and of the `(v, e)` pairs used by [`On`](@ref).
The other two forms are syntactic sugar, resolved eagerly on construction:
`Coef(m, f)` is `Coef(m, [f(x_i) for every node])`, where `f` receives a node
coordinate as a `Vector` and returns a `Real`, and `Coef(m, r)` is the
constant vector `Coef(m, fill(r, V*N))`.
"""
function Coef end

"""
    EpiPower(p)

The pointwise epigraph set `{ [q; s] : s ≥ ‖q‖₂^p }`, used as
`@constraint(m, [q...; s] in EpiPower(p))` — slack LAST (the
`convex_Euclidian_power` `[q; s]` convention). The exponent `p ≥ 1` is spatial
data in the same three forms as [`Coef`](@ref): a per-node vector (the
fundamental form), a spatial function `x -> p(x)`, or a constant `Real`.
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
    set_start(u, v::AbstractVector{<:Real})
    set_start(u, f::Function)
    set_start(u, r::Real)

Initial iterate for component `u` of a [`MGBModel`](@ref), as per-node data in
the same three forms as [`Coef`](@ref): a nodal vector (the fundamental form),
or a spatial function / constant as sugar for it. `value(u)` returns solutions
in the same nodal ordering, so `set_start(u, value(u))` warm-starts from a
previous solve. For Dirichlet-constrained variables the start doubles as the
lift away from the constrained nodes; give slacks a comfortably feasible
(large) start.
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
    On(geom::Geometry, mask::AbstractVector{Bool})

Constraint region for a [`MGBModel`](@ref): a set of broken nodes as
`(vertex, element)` pairs — the same format as [`find_boundary`](@ref) and the
low-level `dirichlet_nodes` API. The second form is grid-level sugar: a Bool
mask over the nodal vectors of [`Coef`](@ref), converted eagerly to the pair
set (`mask[v + (e-1)V]` selects vertex `v` of element `e`; the geometry
supplies `V`). A mask composes directly with grid-level data:
`On(geom, reshape(geom.x, :, d)[:, 1] .< 0)` is the left half of the domain.
`@constraint(m, u == g, On(pairs))` is a Dirichlet condition; an
inequality/cone with `On` holds only on those nodes.
"""
struct On
    pairs::Vector{Tuple{Int,Int}}
    On(pairs::AbstractVector{<:Tuple{Integer,Integer}}) =
        new(Tuple{Int,Int}[(Int(v), Int(e)) for (v, e) in pairs])
end
function On(geom::Geometry, mask::AbstractVector{Bool})
    V, N = size(geom.x, 1), size(geom.x, 2)
    length(mask) == V * N || throw(ArgumentError(
        "region mask has length $(length(mask)) but the geometry has $(V * N) " *
        "broken nodes; entry i is vertex v of element e with i = v + (e-1)V, " *
        "V = $V"))
    On(Tuple{Int,Int}[(mod1(i, V), cld(i, V)) for i in findall(mask)])
end

"""
    Broken()

`@variable` tag for a [`MGBModel`](@ref): the variable lives in the broken space
(one degree of freedom per broken node) — the natural space for epigraph slacks.
Untagged variables are broken by default when never differentiated or
Dirichlet-constrained. See also [`Continuous`](@ref).
"""
struct Broken end

"""
    Continuous()

`@variable` tag for a [`MGBModel`](@ref): the variable is a conforming
(continuous) finite-element function, glued across elements. This is what
untagged variables become when they are differentiated or Dirichlet-constrained;
use the tag to force it for a variable that is neither — e.g. a slack you want
continuous. (Note that constraining a slack to be continuous genuinely changes
the optimum relative to [`Broken`](@ref): the pointwise epigraph reformulation
is exact only for broken slacks.)

The *notion* of continuity is the geometry's, not the tag's: DOFs are glued
according to the connectivity `geom.t`. A slit/branch-cut domain built with an
explicit connectivity (`tensor_dofmap`, the `t=` keyword of the mesh
constructors, or `gmsh_import`'s exact node tags) keeps coincident-but-distinct
nodes separate, and `Continuous()` respects that gluing automatically —
"continuous" then means continuous everywhere except across the slit.
"""
struct Continuous end

"""
    Uniform()

`@variable` tag for a [`MGBModel`](@ref): the variable is a single global constant.
"""
struct Uniform end

export MGBModel, Coef, EpiPower, deriv, integral, set_start, mgb_solution,
       solver_log, On, Broken, Continuous, Uniform
