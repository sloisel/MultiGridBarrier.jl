# MultiGridBarrierJuMPExt — the JuMP modeling front end for MultiGridBarrier,
# as a package extension. Loads automatically when both MultiGridBarrier and JuMP
# are imported (`using MultiGridBarrier, JuMP`). It reuses JuMP's macros
# (@variable, @constraint, @objective) and accessors (value, objective_value,
# termination_status), but never builds an MOI model — `optimize!` lowers the
# model directly to `amg` → `assemble` → `mgb_solve`, constructing the AMG
# hierarchy automatically from the geometry and the Dirichlet constraints.
#
# The user-facing names (MGBModel, Coef, EpiPower, deriv, integral, set_start,
# mgb_solution, solver_log, On, Broken, Uniform) are declared in MultiGridBarrier
# itself (src/jump_frontend.jl): the marker/data types On/Broken/Uniform live
# there directly; the rest are function stubs whose methods are supplied here
# (the concrete MGBModel/Coef/EpiPower structs must subtype JuMP abstract types,
# so they are defined in this extension). See the "Modeling with JuMP" manual
# page for the reference.
module MultiGridBarrierJuMPExt

using JuMP
using StaticArrays
using LinearAlgebra
import MultiGridBarrier
import MultiGridBarrier: Geometry, MultiGrid, Convex, amg, assemble, mgb_solve,
    convex_linear, convex_Euclidian_power, convex_piecewise, MGBConvergenceFailure,
    On, Broken, Continuous, Uniform, deriv, integral, set_start, mgb_solution,
    solver_log, find_boundary
const MOI = JuMP.MOI

# ---------------------------------------------------------------------------
# Coefficient values: a uniform scalar or a per-node vector, as two concrete
# types under one abstract supertype (dispatch instead of a Union{Nothing,...}
# sentinel field). All spatial data is eager. The `CoefVal{T}(...)` convenience
# constructors pick the concrete type, so call sites read like before.
# ---------------------------------------------------------------------------

abstract type CoefVal{T} end
struct UniformCoef{T} <: CoefVal{T}
    scalar::T
end
struct NodalCoef{T} <: CoefVal{T}
    vals::Vector{T}
end
CoefVal{T}(x::Real) where {T} = UniformCoef{T}(T(x))
CoefVal{T}(v::Vector{T}) where {T} = NodalCoef{T}(v)

_isuniform(::UniformCoef) = true
_isuniform(::NodalCoef) = false
_materialize(c::UniformCoef{T}, n::Int) where {T} = fill(c.scalar, n)
_materialize(c::NodalCoef{T}, n::Int) where {T} = c.vals
_getnode(c::UniformCoef, i::Int) = c.scalar
_getnode(c::NodalCoef, i::Int) = c.vals[i]
_iszeroval(c::UniformCoef) = iszero(c.scalar)
_iszeroval(::NodalCoef) = false

# Length of the nodal operand (at least one of a, b is nodal when this is called).
_nodallength(a::NodalCoef, b::CoefVal) = length(a.vals)
_nodallength(a::UniformCoef, b::NodalCoef) = length(b.vals)

for op in (:+, :-, :*)
    @eval Base.$op(a::UniformCoef{T}, b::UniformCoef{T}) where {T} =
        UniformCoef{T}($op(a.scalar, b.scalar))
    @eval function Base.$op(a::CoefVal{T}, b::CoefVal{T}) where {T}
        n = _nodallength(a, b)
        NodalCoef{T}(($op).(_materialize(a, n), _materialize(b, n)))
    end
end
Base.:-(a::UniformCoef{T}) where {T} = UniformCoef{T}(-a.scalar)
Base.:-(a::NodalCoef{T}) where {T} = NodalCoef{T}(-a.vals)

# ---------------------------------------------------------------------------
# Model, variable references, expressions
# ---------------------------------------------------------------------------

mutable struct CompInfo{T}
    name::Symbol
    kind::Symbol                 # :auto | :broken | :uniform
    start::CoefVal{T}
end

# One normalized constraint row: an affine functional of the atoms.
# terms maps (component index, operator symbol) => coefficient.
struct Row{T}
    terms::Dict{Tuple{Int,Symbol},CoefVal{T}}
    constant::CoefVal{T}
end

# settag: (:eq, rhs::CoefVal) | (:nonneg,) | (:power, p::CoefVal)
struct ConRecord{T}
    name::String
    rows::Vector{Row{T}}
    settag::Union{Tuple{Symbol},Tuple{Symbol,CoefVal{T}}}
    pairs::Union{Nothing,Vector{Tuple{Int,Int}}}
end

mutable struct MGBModel{T} <: JuMP.AbstractModel
    geometry::Geometry           # concrete Geometry{T,...} of the model
    coords::Matrix{T}            # (V*N) x d broken-node coordinates
    nnodes::Int
    comps::Vector{CompInfo{T}}
    compnames::Dict{Symbol,Int}
    cons::Vector{ConRecord{T}}
    objsense::MOI.OptimizationSense
    objexpr::Any                 # Union{Nothing,MGBExpr{T}}; Any because MGBExpr
                                 # references MGBModel back (mutually recursive)
    attrs::Dict{String,Any}
    objdict::Dict{Symbol,Any}
    lowered::Any                 # NamedTuple from _lower, set by optimize!
    sol::Any                     # MGBSOL
    status::MOI.TerminationStatusCode
    rawstatus::String
    solvetime::Float64
end

function MultiGridBarrier.MGBModel(geom::Geometry{T}) where {T}
    x3 = geom.x
    coords = Matrix{T}(reshape(x3, :, size(x3, 3)))
    MGBModel{T}(geom, coords, size(coords, 1),
        CompInfo{T}[], Dict{Symbol,Int}(), ConRecord{T}[],
        MOI.MIN_SENSE, nothing, Dict{String,Any}(), Dict{Symbol,Any}(),
        nothing, nothing, MOI.OPTIMIZE_NOT_CALLED, "optimize! not called", NaN)
end

function _invalidate_solution!(m::MGBModel)
    m.lowered = nothing
    m.sol = nothing
    m.status = MOI.OPTIMIZE_NOT_CALLED
    m.rawstatus = "optimize! not called"
    m.solvetime = NaN
    return nothing
end

struct MGBVarRef{T} <: JuMP.AbstractVariableRef
    model::MGBModel{T}
    comp::Int
    op::Symbol
end

mutable struct MGBExpr{T} <: JuMP.AbstractJuMPScalar
    model::Union{Nothing,MGBModel{T}}
    terms::Dict{Tuple{Int,Symbol},CoefVal{T}}
    constant::CoefVal{T}
end
MGBExpr{T}() where {T} =
    MGBExpr{T}(nothing, Dict{Tuple{Int,Symbol},CoefVal{T}}(), CoefVal{T}(zero(T)))

struct Coef{T} <: JuMP.AbstractJuMPScalar
    model::MGBModel{T}
    val::CoefVal{T}
end

# Canonical conversion for ALL user-supplied spatial data (Coef, set_start, the
# EpiPower exponent). The per-node vector is the fundamental form; a Function
# or Real is syntactic sugar for it (nodal samples / a constant vector, the
# latter kept in the compressed uniform representation).
const _NodalData = Union{AbstractVector{<:Real},Function,Real}
function _nodal(m::MGBModel{T}, v::AbstractVector{<:Real}) where {T}
    length(v) == m.nnodes || _argerror(
        "nodal data has length $(length(v)) but the geometry has $(m.nnodes) " *
        "broken nodes; entry i is vertex v of element e with i = v + (e-1)V, " *
        "V = $(size(m.geometry.x, 1))")
    CoefVal{T}(Vector{T}(v))
end
function _nodal(m::MGBModel{T}, f::Function) where {T}
    vals = Vector{T}(undef, m.nnodes)
    try
        for i in 1:m.nnodes
            vals[i] = T(f(m.coords[i, :]))
        end
    catch e
        _argerror("a spatial function must accept a coordinate Vector and " *
                  "return a Real; sampling it failed with: " *
                  first(split(sprint(showerror, e), '\n')))
    end
    CoefVal{T}(vals)
end
_nodal(m::MGBModel{T}, r::Real) where {T} = CoefVal{T}(T(r))

MultiGridBarrier.Coef(m::MGBModel{T}, data::_NodalData) where {T} =
    Coef{T}(m, _nodal(m, data))

function deriv(v::MGBVarRef, op::Symbol)
    v.op === :id || _argerror("deriv: cannot apply :$op to $(JuMP.name(v)); operators do not compose")
    ops = v.model.geometry.operators
    haskey(ops, op) || _argerror("deriv: unknown operator :$op; available: $(collect(keys(ops)))")
    MGBVarRef(v.model, v.comp, op)
end

struct Integral{T}
    expr::MGBExpr{T}
end

integral(x::Union{MGBVarRef,MGBExpr,Coef}) = Integral(_to_expr(x))

# The integral is linear, so Integrals close under +/- and Real scaling:
# `2*integral(u) - integral(s)` is the same objective as `integral(2*u - s)`.
# (No Real +/-: `integral(u) + 3` has no ∫-form and stays a MethodError.)
Base.:+(a::Integral{T}, b::Integral{T}) where {T} = Integral(a.expr + b.expr)
Base.:-(a::Integral{T}, b::Integral{T}) where {T} = Integral(a.expr - b.expr)
Base.:+(a::Integral) = a
Base.:-(a::Integral) = Integral(-a.expr)
Base.:*(r::Real, a::Integral) = Integral(a.expr * r)
Base.:*(a::Integral, r::Real) = Integral(a.expr * r)
Base.:/(a::Integral, r::Real) = Integral(a.expr / r)

_argerror(msg) = throw(ArgumentError(msg))

# --- ref/expr identity ------------------------------------------------------

Base.:(==)(a::MGBVarRef, b::MGBVarRef) =
    a.model === b.model && a.comp == b.comp && a.op == b.op
Base.hash(v::MGBVarRef, h::UInt) = hash((objectid(v.model), v.comp, v.op), h)
Base.copy(v::MGBVarRef) = v
Base.broadcastable(v::MGBVarRef) = Ref(v)
JuMP.owner_model(v::MGBVarRef) = v.model
JuMP.isequal_canonical(a::MGBVarRef, b::MGBVarRef) = a == b
function JuMP.name(v::MGBVarRef)
    base = String(v.model.comps[v.comp].name)
    v.op === :id ? base : "deriv($base, :$(v.op))"
end
Base.show(io::IO, v::MGBVarRef) = print(io, JuMP.name(v))

_atomkey(v::MGBVarRef) = (v.comp, v.op)

function Base.show(io::IO, e::MGBExpr)
    first = true
    for (k, c) in e.terms
        print(io, first ? "" : " + ", _coef_string(c), "*", "atom", k)
        first = false
    end
    print(io, first ? "" : " + ", _coef_string(e.constant))
end

# --- conversions and arithmetic ---------------------------------------------

_to_expr(x::MGBExpr) = x
function _to_expr(v::MGBVarRef{T}) where {T}
    e = MGBExpr{T}()
    e.model = v.model
    e.terms[_atomkey(v)] = CoefVal{T}(one(T))
    e
end
function _to_expr(c::Coef{T}) where {T}
    e = MGBExpr{T}()
    e.model = c.model
    e.constant = c.val
    e
end
_to_expr(::Type{T}, r::Real) where {T} =
    (e = MGBExpr{T}(); e.constant = CoefVal{T}(r); e)

_coefval(::Type{T}, r::Real) where {T} = CoefVal{T}(r)
_coefval(::Type{T}, c::Coef{T}) where {T} = c.val

const _Scalar = Union{MGBVarRef,MGBExpr,Coef}

_exprmodel(v::MGBVarRef) = v.model
_exprmodel(e::MGBExpr) = e.model
_exprmodel(c::Coef) = c.model

_hasvars(::MGBVarRef) = true
_hasvars(e::MGBExpr) = !isempty(e.terms)
_hasvars(::Coef) = false

_firstvar(v::MGBVarRef) = v
function _firstvar(e::MGBExpr)
    key = first(keys(e.terms))
    return MGBVarRef(e.model, key[1], key[2])
end

function _check_same_model(x::_Scalar, y::_Scalar)
    mx, my = _exprmodel(x), _exprmodel(y)
    (mx === nothing || my === nothing || mx === my) && return nothing
    xvars, yvars = _hasvars(x), _hasvars(y)
    if xvars && yvars
        throw(JuMP.VariableNotOwned(_firstvar(y)))
    end
    _argerror("cannot combine Coef data from different MGBModel objects")
end

function _check_belongs_to_model(e::MGBExpr, m::MGBModel)
    (e.model === nothing || e.model === m) && return nothing
    isempty(e.terms) || throw(JuMP.VariableNotOwned(_firstvar(e)))
    _argerror("cannot use Coef data from another MGBModel in this model")
end

function JuMP.check_belongs_to_model(e::MGBExpr, m::MGBModel)
    _check_belongs_to_model(e, m)
end
function JuMP.check_belongs_to_model(c::Coef, m::MGBModel)
    c.model === m ||
        _argerror("cannot use Coef data from another MGBModel in this model")
    return nothing
end

function _merge!(a::MGBExpr{T}, b::MGBExpr{T}, sgn::T) where {T}
    _check_same_model(a, b)
    a.model = a.model === nothing ? b.model : a.model
    for (k, c) in b.terms
        cc = sgn == one(T) ? c : CoefVal{T}(sgn) * c
        a.terms[k] = haskey(a.terms, k) ? a.terms[k] + cc : cc
    end
    a.constant = a.constant + (sgn == one(T) ? b.constant : CoefVal{T}(sgn) * b.constant)
    a
end

function Base.:+(x::_Scalar, y::_Scalar)
    _check_same_model(x, y)
    T = _eltype(x, y)
    _merge!(_copyexpr(_to_expr(x)), _to_expr(y), one(T))
end
function Base.:-(x::_Scalar, y::_Scalar)
    _check_same_model(x, y)
    T = _eltype(x, y)
    _merge!(_copyexpr(_to_expr(x)), _to_expr(y), -one(T))
end
Base.:+(x::_Scalar, y::Real) = x + _to_expr(_scalartype(x), y)
Base.:+(x::Real, y::_Scalar) = y + x
Base.:-(x::_Scalar, y::Real) = x - _to_expr(_scalartype(x), y)
Base.:-(x::Real, y::_Scalar) = _to_expr(_scalartype(y), x) - y
Base.:-(x::_Scalar) = (-1) * x
Base.:+(x::_Scalar) = x

_scalartype(::MGBVarRef{T}) where {T} = T
_scalartype(::MGBExpr{T}) where {T} = T
_scalartype(::Coef{T}) where {T} = T
_eltype(x::_Scalar, y::_Scalar) = _scalartype(x)

function _copyexpr(e::MGBExpr{T}) where {T}
    MGBExpr{T}(e.model, copy(e.terms), e.constant)
end

function _scale(e::MGBExpr{T}, c::CoefVal{T}) where {T}
    out = MGBExpr{T}()
    out.model = e.model
    for (k, v) in e.terms
        out.terms[k] = v * c
    end
    out.constant = e.constant * c
    out
end

function Base.:*(x::_Scalar, y::_Scalar)
    _check_same_model(x, y)
    T = _eltype(x, y)
    if _hasvars(x) && _hasvars(y)
        _argerror("nonlinear: cannot multiply two expressions that both contain variables; " *
                  "introduce a slack variable and a cone constraint instead")
    end
    if _hasvars(y)
        x, y = y, x
    end
    ye = _to_expr(y)
    _scale(_to_expr(x), ye.constant)
end
Base.:*(x::_Scalar, y::Real) = _scale(_to_expr(x), _coefval(_scalartype(x), y))
Base.:*(x::Real, y::_Scalar) = y * x
Base.:/(x::_Scalar, y::Real) = x * (1 / y)

# Spatial-data sugar: a raw Function or nodal vector (AbstractVector{<:Real})
# next to a model-carrying scalar means Coef(model, data), resolved eagerly
# through the adjacent operand's model — so `u == g`, `u >= phi_vals`, and
# `a * u` work in constraints and objectives without an explicit Coef. For
# Real vectors this deliberately claims the scalar-array arithmetic slots:
# our scalars are fields, and a Real vector beside one is nodal data forming
# ONE field expression — unlike generic JuMP, where +/- throw
# use-broadcasting errors and v*x is container scaling. Broadcasting is
# untouched (`u .+ v` is still n elementwise expressions), and JuMP variable
# containers have non-Real eltype, so they keep JuMP semantics.
const _SpatialData = Union{Function,AbstractVector{<:Real}}
function _sugarmodel(x::_Scalar)
    m = _exprmodel(x)
    m === nothing && _argerror(
        "cannot resolve spatial data (a Function or nodal vector) in an " *
        "expression with no variables; wrap it in Coef(m, ...)")
    m
end
for op in (:+, :-, :*)
    @eval Base.$op(x::_Scalar, d::_SpatialData) =
        $op(x, MultiGridBarrier.Coef(_sugarmodel(x), d))
    @eval Base.$op(d::_SpatialData, x::_Scalar) =
        $op(MultiGridBarrier.Coef(_sugarmodel(x), d), x)
end

Base.zero(::Type{MGBVarRef{T}}) where {T} = MGBExpr{T}()
Base.zero(v::MGBVarRef{T}) where {T} = MGBExpr{T}()
Base.zero(::Type{MGBExpr{T}}) where {T} = MGBExpr{T}()
Base.zero(e::MGBExpr{T}) where {T} = MGBExpr{T}()
Base.one(::Type{MGBVarRef{T}}) where {T} = _to_expr(T, one(T))

# MutableArithmetics interop: the @constraint/@objective macros rewrite
# `a + c*b` into MA.add_mul, whose JuMP fallback needs promote_operation and
# convert for our scalar types (otherwise `s + Coef(m, λ)*r` throws).
const _MA = JuMP._MA   # MutableArithmetics, as imported by JuMP
const _OurT{T} = Union{MGBVarRef{T}, MGBExpr{T}, Coef{T}}
_expr_type(::Type{MGBVarRef{T}}) where {T} = MGBExpr{T}
_expr_type(::Type{MGBExpr{T}}) where {T} = MGBExpr{T}
_expr_type(::Type{Coef{T}}) where {T} = MGBExpr{T}
for op in (:+, :-, :*)
    @eval _MA.promote_operation(::typeof(Base.$op), ::Type{A},
            ::Type{B}) where {T, A<:_OurT{T}, B<:_OurT{T}} = MGBExpr{T}
    @eval _MA.promote_operation(::typeof(Base.$op), ::Type{A},
            ::Type{<:Real}) where {A<:_Scalar} = _expr_type(A)
    @eval _MA.promote_operation(::typeof(Base.$op), ::Type{<:Real},
            ::Type{A}) where {A<:_Scalar} = _expr_type(A)
end
Base.convert(::Type{MGBExpr{T}}, x::MGBVarRef{T}) where {T} = _to_expr(x)
Base.convert(::Type{MGBExpr{T}}, x::Coef{T}) where {T} = _to_expr(x)
Base.convert(::Type{MGBExpr{T}}, r::Real) where {T} = _to_expr(T, r)

# JuMP intercepts MutableArithmetics' add_mul/sub_mul between a JuMP scalar
# and an AbstractArray to throw its use-broadcasting error, which would defeat
# the nodal-vector sugar inside @constraint/@objective (Base's +/- are never
# consulted on that path); restore the field-data meaning for Real vectors
# adjacent to our scalars.
const _AddSub = Union{typeof(_MA.add_mul),typeof(_MA.sub_mul)}
_MA.operate!!(op::_AddSub, x::_Scalar, d::AbstractVector{<:Real}) =
    op === _MA.add_mul ? x + d : x - d
_MA.operate!!(op::_AddSub, d::AbstractVector{<:Real}, x::_Scalar) =
    op === _MA.add_mul ? d + x : d - x

# JuMP's macro plumbing asks expression types for their variable type (e.g. when
# parsing interval constraints lb <= expr <= ub); without these the user gets an
# internal JuMP error instead of our "unsupported set" rejection.
JuMP.variable_ref_type(::Type{MGBExpr{T}}) where {T} = MGBVarRef{T}
JuMP.variable_ref_type(::Type{Coef{T}}) where {T} = MGBVarRef{T}

# ---------------------------------------------------------------------------
# JuMP model interface: variables
# ---------------------------------------------------------------------------

JuMP.object_dictionary(m::MGBModel) = m.objdict
JuMP.num_variables(m::MGBModel) = length(m.comps)
JuMP.all_variables(m::MGBModel{T}) where {T} =
    MGBVarRef{T}[MGBVarRef(m, k, :id) for k in 1:length(m.comps)]
Base.broadcastable(m::MGBModel) = Ref(m)
Base.broadcastable(e::MGBExpr) = Ref(e)
Base.broadcastable(c::Coef) = Ref(c)
Base.show(io::IO, m::MGBModel) =
    print(io, "MGBModel over $(typeof(m.geometry.discretization)) with ",
          length(m.comps), " variable(s), ", length(m.cons), " constraint(s)")

struct KindedVariable
    info::JuMP.VariableInfo
    kind::Symbol
end

JuMP.build_variable(err::Function, info::JuMP.VariableInfo, ::Broken) =
    KindedVariable(info, :broken)
JuMP.build_variable(err::Function, info::JuMP.VariableInfo, ::Continuous) =
    KindedVariable(info, :conforming)
JuMP.build_variable(err::Function, info::JuMP.VariableInfo, ::Uniform) =
    KindedVariable(info, :uniform)

function _check_info(info::JuMP.VariableInfo)
    (info.has_lb || info.has_ub) &&
        _argerror("variable bounds are not supported; write a pointwise constraint, e.g. @constraint(m, u >= 0) — or u >= Coef(m, lo) for spatial data")
    info.has_fix && _argerror("fixed variables are not supported; use an equality constraint with On(pairs)")
    (info.binary || info.integer) && _argerror("integrality is not supported")
    nothing
end

function _add_comp(m::MGBModel{T}, info::JuMP.VariableInfo, kind::Symbol,
                   name::String) where {T}
    _check_info(info)
    sym = isempty(name) ? Symbol("_anon", length(m.comps) + 1) : Symbol(name)
    haskey(m.compnames, sym) && _argerror("a variable named $sym already exists")
    s0 = info.has_start ? something(info.start) : zero(T)
    s0 isa _NodalData || _argerror(
        "start must be a Real, a spatial Function, or a nodal vector, got $(typeof(s0))")
    start = _nodal(m, s0)
    push!(m.comps, CompInfo{T}(sym, kind, start))
    m.compnames[sym] = length(m.comps)
    ref = MGBVarRef(m, length(m.comps), :id)
    _invalidate_solution!(m)
    return ref
end

JuMP.add_variable(m::MGBModel, v::JuMP.ScalarVariable, name::String = "") =
    _add_comp(m, v.info, :auto, name)
JuMP.add_variable(m::MGBModel, v::KindedVariable, name::String = "") =
    _add_comp(m, v.info, v.kind, name)

function set_start(v::MGBVarRef{T}, data::_NodalData) where {T}
    v.op === :id || _argerror("set_start applies to variables, not deriv atoms")
    start = _nodal(v.model, data)
    v.model.comps[v.comp].start = start
    _invalidate_solution!(v.model)
    return nothing
end
JuMP.set_start_value(v::MGBVarRef, r::Real) = set_start(v, r)
# starts always exist (they default to 0), so this returns the nodal vector,
# never `nothing`
function JuMP.start_value(v::MGBVarRef{T}) where {T}
    v.op === :id || _argerror("start_value applies to variables, not deriv atoms")
    Vector{T}(_materialize(v.model.comps[v.comp].start, v.model.nnodes))
end

# ---------------------------------------------------------------------------
# JuMP model interface: constraints
# ---------------------------------------------------------------------------

struct EpiPower{P} <: JuMP.AbstractVectorSet
    p::P
end
MultiGridBarrier.EpiPower(p::_NodalData) = EpiPower(p)

struct MOIEpiPower{P} <: MOI.AbstractVectorSet
    p::P
    dim::Int
end
MOI.dimension(s::MOIEpiPower) = s.dim
JuMP.moi_set(s::EpiPower, dim::Int) = MOIEpiPower(s.p, dim)

# region-restricted constraint: wrap the underlying constraint. On is already
# resolved to (vertex, element) pairs (masks convert eagerly in On(geom, mask)).
struct RegionConstraint{C} <: JuMP.AbstractConstraint
    con::C
    pairs::Vector{Tuple{Int,Int}}
end
JuMP.jump_function(rc::RegionConstraint) = JuMP.jump_function(rc.con)
JuMP.moi_set(rc::RegionConstraint) = JuMP.moi_set(rc.con)

JuMP.build_constraint(err::Function, f, set, on::On) =
    RegionConstraint(JuMP.build_constraint(err, f, set), on.pairs)

# JuMP's macro maps comparisons with non-Number sides to shortcut sets through
# VARIADIC build_constraint methods, which are ambiguous with the On method
# above: classic Zeros/Nonnegatives/Nonpositives, and (on newer JuMP versions)
# the internal *Zero marker sets. Cover whichever exist, normalizing to the
# MOI scalar/vector sets ourselves; vector comparisons arrive here too (f is
# then a Vector), hence the branch.
for (marker, moiset, vecset) in
        ((:Zeros,           :(MOI.EqualTo(0.0)),     :(MOI.Zeros)),
         (:Nonnegatives,    :(MOI.GreaterThan(0.0)), :(MOI.Nonnegatives)),
         (:Nonpositives,    :(MOI.LessThan(0.0)),    :(MOI.Nonpositives)),
         (:GreaterThanZero, :(MOI.GreaterThan(0.0)), :(MOI.Nonnegatives)),
         (:LessThanZero,    :(MOI.LessThan(0.0)),    :(MOI.Nonpositives)),
         (:EqualToZero,     :(MOI.EqualTo(0.0)),     :(MOI.Zeros)))
    if isdefined(JuMP, marker)
        @eval function JuMP.build_constraint(err::Function, f,
                                             set::JuMP.$marker, on::On)
            inner = f isa AbstractVector ?
                JuMP.build_constraint(err, f, $vecset(length(f))) :
                JuMP.build_constraint(err, f, $moiset)
            RegionConstraint(inner, on.pairs)
        end
    end
end

# Defensive entry points, in case the JuMP generics for AbstractJuMPScalar do
# not cover our types on some JuMP version. More specific than JuMP's own
# methods on our types, so no ambiguities.
JuMP.build_constraint(err::Function, f::Union{MGBVarRef,MGBExpr},
                      set::MOI.AbstractScalarSet) = JuMP.ScalarConstraint(f, set)
JuMP.build_constraint(err::Function, f::AbstractVector{<:_Scalar},
                      set::JuMP.AbstractVectorSet) =
    JuMP.VectorConstraint(collect(f), JuMP.moi_set(set, length(f)))
JuMP.build_constraint(err::Function, f::AbstractVector{<:_Scalar},
                      set::MOI.AbstractVectorSet) =
    JuMP.VectorConstraint(collect(f), set)

# lightweight constraint reference (deliberately not JuMP.ConstraintRef)
struct MGBConRef
    model::MGBModel
    idx::Int
end
Base.show(io::IO, cr::MGBConRef) =
    print(io, "MGBConRef(", cr.model.cons[cr.idx].name, ")")

function _row(m::MGBModel{T}, x) where {T}
    e = x isa MGBExpr{T} ? x :
        x isa _Scalar ? _to_expr(x) :
        x isa Real ? _to_expr(T, x) :
        x isa JuMP.GenericAffExpr ? _from_affexpr(m, x) :
        _argerror("cannot use $(typeof(x)) inside a constraint; wrap plain data in Coef(m, ...)")
    _check_belongs_to_model(e, m)
    Row{T}(copy(e.terms), e.constant)
end

# JuMP promotes plain numbers sitting next to our scalars in vector constraints
# into GenericAffExpr{C, MGBVarRef} (e.g. the constant row in
# `[dx; dy; 1.0; s] in EpiPower(1.0)`); fold those back into our form.
function _from_affexpr(m::MGBModel{T},
                       a::JuMP.GenericAffExpr{<:Real,MGBVarRef{T}}) where {T}
    e = MGBExpr{T}()
    e.model = m
    for (c, v) in JuMP.linear_terms(a)
        v.model === m || throw(JuMP.VariableNotOwned(v))
        key = _atomkey(v)
        cc = CoefVal{T}(T(c))
        e.terms[key] = haskey(e.terms, key) ? e.terms[key] + cc : cc
    end
    e.constant = CoefVal{T}(T(a.constant))
    e
end
_from_affexpr(m::MGBModel, a::JuMP.GenericAffExpr) =
    _argerror("constraint mixes variables from a different model type ($(typeof(a)))")

_negrow(r::Row{T}) where {T} =
    Row{T}(Dict(k => -c for (k, c) in r.terms), -r.constant)

function _shiftrow(r::Row{T}, delta::CoefVal{T}) where {T}
    Row{T}(r.terms, r.constant + delta)
end

function _push_con!(m::MGBModel{T}, rows::Vector{Row{T}}, settag::Tuple,
                    pairs, name::String) where {T}
    push!(m.cons, ConRecord{T}(isempty(name) ? "c$(length(m.cons)+1)" : name,
                               rows, settag, pairs))
    ref = MGBConRef(m, length(m.cons))
    _invalidate_solution!(m)
    return ref
end

# --- scalar constraints ---
function _add_scalar(m::MGBModel{T}, func, set, pairs, name) where {T}
    r = _row(m, func)
    if set isa MOI.EqualTo
        pairs === nothing &&
            _argerror("pointwise equality has empty interior; Dirichlet-style equality needs a node set: @constraint(m, u == g, On(pairs))")
        # normalize to: single atom (comp, :id) with coefficient a; values (rhs - const)/a
        length(r.terms) == 1 ||
            _argerror("Dirichlet equality must involve exactly one variable, got $(length(r.terms)) atoms")
        (key, a) = first(r.terms)
        key[2] === :id ||
            _argerror("Dirichlet equality applies to a variable, not deriv(..., :$(key[2]))")
        _isuniform(a) || _argerror("Dirichlet equality: variable coefficient must be a constant scalar")
        iszero(a.scalar) && _argerror("Dirichlet equality: zero coefficient")
        rhs = (CoefVal{T}(T(set.value)) - r.constant) * CoefVal{T}(inv(a.scalar))
        return _push_con!(m, [Row{T}(Dict(key => CoefVal{T}(one(T))), CoefVal{T}(zero(T)))],
                          (:eq, rhs), pairs, name)
    elseif set isa MOI.GreaterThan
        rr = _shiftrow(r, CoefVal{T}(-T(set.lower)))
        return _push_con!(m, [rr], (:nonneg,), pairs, name)
    elseif set isa MOI.LessThan
        rr = _shiftrow(_negrow(r), CoefVal{T}(T(set.upper)))
        return _push_con!(m, [rr], (:nonneg,), pairs, name)
    elseif set isa MOI.Interval
        # lo <= expr <= hi: two stacked :nonneg rows, exactly what the two
        # one-sided constraints would merge into
        lo, hi = T(set.lower), T(set.upper)
        lo == hi && _argerror(
            "an interval with equal bounds is a pointwise equality, which has " *
            "empty interior; use @constraint(m, u == g, On(pairs))")
        rows = Row{T}[]
        isfinite(lo) && push!(rows, _shiftrow(r, CoefVal{T}(-lo)))
        isfinite(hi) && push!(rows, _shiftrow(_negrow(r), CoefVal{T}(hi)))
        isempty(rows) && _argerror("interval constraint with two infinite bounds is vacuous")
        return _push_con!(m, rows, (:nonneg,), pairs, name)
    else
        _argerror("unsupported scalar constraint set $(typeof(set))")
    end
end

# --- vector constraints ---
function _add_vector(m::MGBModel{T}, funcs, set, pairs, name) where {T}
    rows = Row{T}[_row(m, f) for f in funcs]
    if set isa MOI.Nonnegatives
        return _push_con!(m, rows, (:nonneg,), pairs, name)
    elseif set isa MOI.Nonpositives
        return _push_con!(m, map(_negrow, rows), (:nonneg,), pairs, name)
    elseif set isa MOI.SecondOrderCone
        # JuMP convention [t; x...]; zoo convention [q...; s] with s last, p = 1
        return _push_con!(m, [rows[2:end]; rows[1:1]], (:power, _nodal(m, one(T))), pairs, name)
    elseif set isa MOIEpiPower
        # normalize the exponent to nodal data here: a wrong-length vector fails
        # at @constraint time, and downstream code sees one form (a CoefVal,
        # like the :eq settag) regardless of how the user spelled p
        return _push_con!(m, rows, (:power, _nodal(m, set.p)), pairs, name)
    elseif set isa MOI.Zeros
        _argerror("pointwise vector equality has empty interior; use scalar equalities with On(pairs)")
    else
        _argerror("unsupported vector constraint set $(typeof(set))")
    end
end

function JuMP.add_constraint(m::MGBModel, c::JuMP.ScalarConstraint,
                             name::String = "")
    _add_scalar(m, JuMP.jump_function(c), JuMP.moi_set(c), nothing, name)
end
function JuMP.add_constraint(m::MGBModel, c::JuMP.VectorConstraint,
                             name::String = "")
    _add_vector(m, JuMP.jump_function(c), JuMP.moi_set(c), nothing, name)
end
function JuMP.add_constraint(m::MGBModel, rc::RegionConstraint, name::String = "")
    c = rc.con
    if c isa JuMP.ScalarConstraint
        _add_scalar(m, JuMP.jump_function(c), JuMP.moi_set(c), rc.pairs, name)
    elseif c isa JuMP.VectorConstraint
        _add_vector(m, JuMP.jump_function(c), JuMP.moi_set(c), rc.pairs, name)
    else
        _argerror("unsupported constraint type $(typeof(c)) with On(...)")
    end
end
JuMP.num_constraints(m::MGBModel; kwargs...) = length(m.cons)

# ---------------------------------------------------------------------------
# JuMP model interface: objective
# ---------------------------------------------------------------------------

function JuMP.set_objective(m::MGBModel{T}, sense::MOI.OptimizationSense,
                            f::Integral{T}) where {T}
    sense == MOI.FEASIBILITY_SENSE &&
        _argerror("feasibility-only models are not supported yet; minimize integral(s) of a slack instead")
    _check_belongs_to_model(f.expr, m)
    m.objsense = sense
    m.objexpr = f.expr
    _invalidate_solution!(m)
    return nothing
end
JuMP.set_objective(m::MGBModel, ::MOI.OptimizationSense, f) =
    _argerror("the objective must be integral(affine expr), got $(typeof(f))")
function JuMP.set_objective_sense(m::MGBModel, s::MOI.OptimizationSense)
    m.objsense = s
    _invalidate_solution!(m)
    return nothing
end
JuMP.objective_sense(m::MGBModel) = m.objsense
# some JuMP versions route @objective through sense + function separately
function JuMP.set_objective_function(m::MGBModel{T}, f::Integral{T}) where {T}
    _check_belongs_to_model(f.expr, m)
    m.objexpr = f.expr
    _invalidate_solution!(m)
    return nothing
end
JuMP.set_objective_function(m::MGBModel, f) =
    _argerror("the objective must be integral(affine expr), got $(typeof(f))")

# ---------------------------------------------------------------------------
# Lowering: model -> (state_variables, D, dirichlet_nodes, f_grid, g_grid, Q)
# ---------------------------------------------------------------------------

_pairs_to_linear(pairs, V) = Int[v + (e - 1) * V for (v, e) in pairs]

# Merge :nonneg constraint records that share the same region into a single
# stacked record. Each record becomes one convex_linear piece, and pieces carry
# per-node piecewise-kernel overhead, so k same-region scalar inequalities as
# one k-row piece beats k one-row pieces (and matches what a hand-written
# intersect(...) formulation would build). The barrier is a sum either way, so
# the math is unchanged. :power records are single cones and never merge.
function _merge_nonneg(cones::Vector{ConRecord{T}}) where {T}
    out = ConRecord{T}[]
    slot = Dict{Any,Int}()               # region key => index in out
    for c in cones
        if c.settag[1] === :nonneg
            key = c.pairs === nothing ? nothing : sort(c.pairs)
            j = get(slot, key, 0)
            if j == 0
                push!(out, c)
                slot[key] = length(out)
            else
                p = out[j]
                out[j] = ConRecord{T}(p.name * "+" * c.name,
                                      vcat(p.rows, c.rows), (:nonneg,), p.pairs)
            end
        else
            push!(out, c)
        end
    end
    return out
end

function _lower(m::MGBModel{T}) where {T}
    isempty(m.comps) && _argerror("model has no variables")
    m.objexpr === nothing && _argerror("model has no objective; a barrier method needs one")
    geom = m.geometry
    n = m.nnodes
    V = size(geom.x, 1)
    ncomp = length(m.comps)

    cones = _merge_nonneg([c for c in m.cons if c.settag[1] !== :eq])
    diris = [c for c in m.cons if c.settag[1] === :eq]
    isempty(cones) &&
        _argerror("model has no inequality/cone constraints; a barrier method needs a barrier")

    # -- resolve kinds ---------------------------------------------------
    differentiated = falses(ncomp)
    used = falses(ncomp)
    function _scan(terms)
        for ((comp, op), _) in terms
            used[comp] = true
            op === :id || (differentiated[comp] = true)
        end
    end
    for c in cones, r in c.rows
        _scan(r.terms)
    end
    _scan(m.objexpr.terms)
    hasdiri = falses(ncomp)
    for c in diris
        comp = first(keys(c.rows[1].terms))[1]
        hasdiri[comp] = true
        used[comp] = true
    end
    for k in 1:ncomp
        used[k] || _argerror("variable $(m.comps[k].name) appears in no constraint or objective; its Hessian block would be singular")
    end
    kinds = Vector{Symbol}(undef, ncomp)
    notes = String[]
    for k in 1:ncomp
        ck = m.comps[k]
        kinds[k] = ck.kind === :auto ?
            ((differentiated[k] || hasdiri[k]) ? :conforming : :broken) : ck.kind
        if hasdiri[k] && kinds[k] !== :conforming
            _argerror("variable $(ck.name) has a Dirichlet constraint but kind $(kinds[k]); Dirichlet conditions require a conforming variable")
        end
        if hasdiri[k] && !(differentiated[k]) && ck.kind === :auto
            # Legal but almost certainly a modeling accident (silenced when the
            # user explicitly tagged the variable Continuous()). A lowering
            # note in the solve log, NOT a console @warn: the package writes
            # nothing to stdout/stderr besides the opt-in progress bar.
            push!(notes, "lowering: variable $(ck.name) is Dirichlet-constrained " *
                "but never differentiated (tag it Continuous() if intentional)")
        end
    end

    # -- state_variables + dirichlet_nodes -------------------------------
    state_variables = Matrix{Symbol}(undef, ncomp, 2)
    dirichlet_nodes = Dict{Symbol,Vector{Tuple{Int,Int}}}()
    for k in 1:ncomp
        ck = m.comps[k]
        state_variables[k, 1] = ck.name
        if kinds[k] === :broken
            state_variables[k, 2] = :full
        elseif kinds[k] === :uniform
            state_variables[k, 2] = :uniform
        else
            sub = Symbol("dirichlet_", ck.name)
            state_variables[k, 2] = sub
            dirichlet_nodes[sub] = Tuple{Int,Int}[]
        end
    end
    for c in diris
        comp = first(keys(c.rows[1].terms))[1]
        append!(dirichlet_nodes[state_variables[comp, 2]], c.pairs)
    end
    for (k, v) in dirichlet_nodes
        dirichlet_nodes[k] = unique(v)
    end

    # -- D table: every component's :id row first (matching the classical
    # default_D convention; unused rows cost nothing), then remaining atoms in
    # first-appearance order. The :id seeding also guarantees spare D rows for
    # the square-padding of Euclidian-power constraints: the barrier kernels'
    # gradient scatter is last-match-wins, so padded idx entries must be
    # DISTINCT rows (with zero A columns), never repeats.
    atoms = Tuple{Int,Symbol}[(k, :id) for k in 1:ncomp]
    atomidx = Dict{Tuple{Int,Symbol},Int}((k, :id) => k for k in 1:ncomp)
    function _visit(terms)
        for k in keys(terms)
            if !haskey(atomidx, k)
                push!(atoms, k)
                atomidx[k] = length(atoms)
            end
        end
    end
    for c in cones, r in c.rows
        _visit(r.terms)
    end
    _visit(m.objexpr.terms)
    nD = length(atoms)
    D = Matrix{Symbol}(undef, nD, 2)
    for (i, (comp, op)) in enumerate(atoms)
        D[i, 1] = m.comps[comp].name
        D[i, 2] = op
    end

    # -- f_grid (cost rows over D), g_grid (start/Dirichlet per component) -
    sgn = m.objsense == MOI.MAX_SENSE ? -one(T) : one(T)
    f_grid = zeros(T, n, nD)
    for (k, c) in m.objexpr.terms
        f_grid[:, atomidx[k]] .= sgn .* _materialize(c, n)
    end
    obj_offset = m.objexpr.constant   # user-sense constant, added back in objective_value

    g_grid = zeros(T, n, ncomp)
    for k in 1:ncomp
        g_grid[:, k] .= _materialize(m.comps[k].start, n)
    end
    for c in diris
        comp = first(keys(c.rows[1].terms))[1]
        rhs = c.settag[2]
        for i in _pairs_to_linear(c.pairs, V)
            g_grid[i, comp] = _getnode(rhs, i)
        end
    end

    (; state_variables, dirichlet_nodes, D, atoms, atomidx,
       f_grid, g_grid, obj_offset, cones, kinds, nD, ncomp, notes)
end

# one Convex piece per cone record
function _piece(m::MGBModel{T}, mg::MultiGrid, low, c::ConRecord{T}) where {T}
    n = m.nnodes
    # selected atoms for this constraint, in D order
    sel = sort!(unique(reduce(vcat,
        [collect(keys(r.terms)) for r in c.rows]; init = Tuple{Int,Symbol}[]));
        by = k -> low.atomidx[k])
    ni = length(sel)
    ni == 0 && _argerror("constraint $(c.name) contains no variables")
    pos = Dict(key => j for (j, key) in enumerate(sel))
    nc = length(c.rows)

    if c.settag[1] === :nonneg
        # convex_linear reconstructs a rectangular SMatrix{nc,ni} per node;
        # A_grid rows are vec(A) in column-major order.
        idx = SVector{ni,Int}((low.atomidx[key] for key in sel)...)
        A_grid = zeros(T, n, nc * ni)
        b_grid = zeros(T, n, nc)
        for (r, row) in enumerate(c.rows)
            b_grid[:, r] .= _materialize(row.constant, n)
            for (key, coef) in row.terms
                j = pos[key]
                A_grid[:, (j - 1) * nc + r] .= _materialize(coef, n)
            end
        end
        return convex_linear(T; mg, idx, A_grid, b_grid)
    elseif c.settag[1] === :power
        # convex_Euclidian_power reconstructs a SQUARE k×k matrix per node with
        # length(idx) == k. Pad to square: extra idx entries must be DISTINCT
        # spare D rows with zero A columns (the kernels' gradient scatter is
        # last-match-wins, so repeated idx entries would drop gradients); extra
        # zero q-rows (with b = 0) extend q by zeros, which leaves ‖q‖
        # unchanged. The slack row must stay LAST.
        k = max(nc, ni)
        selidx = Int[low.atomidx[key] for key in sel]
        if ni < k
            spare = [j for j in 1:low.nD if !(j in selidx)]
            length(spare) >= k - ni ||
                _argerror("constraint $(c.name): needs $(k - ni) padding D row(s) but only $(length(spare)) are unused")
            append!(selidx, spare[1:(k - ni)])
        end
        idxfull = SVector{k,Int}(selidx...)
        A_grid = zeros(T, n, k * k)
        b_grid = zeros(T, n, k)
        rowpos(r) = r == nc ? k : r      # q rows keep position, s row goes last
        for (r, row) in enumerate(c.rows)
            rp = rowpos(r)
            b_grid[:, rp] .= _materialize(row.constant, n)
            for (key, coef) in row.terms
                j = pos[key]
                A_grid[:, (j - 1) * k + rp] .= _materialize(coef, n)
            end
        end
        p_grid = _materialize(c.settag[2], n)
        return convex_Euclidian_power(T; mg, idx = idxfull, A_grid, b_grid, p_grid)
    else
        _argerror("internal: unknown settag $(c.settag)")
    end
end

function JuMP.optimize!(m::MGBModel{T}) where {T}
    _invalidate_solution!(m)
    t0 = time()
    low = _lower(m)
    geom = m.geometry
    n = m.nnodes
    V = size(geom.x, 1)

    sv = low.state_variables
    if hasmethod(amg, Tuple{typeof(geom)}, (:dirichlet_nodes,))
        amg_kw = Dict{Symbol,Any}(:dirichlet_nodes => low.dirichlet_nodes)
        haskey(m.attrs, "prolongator") && (amg_kw[:prolongator] = m.attrs["prolongator"])
        isempty(low.dirichlet_nodes) && pop!(amg_kw, :dirichlet_nodes)
        mg = amg(geom; amg_kw...)
    else
        # Basis-truncation hierarchies (the spectral families): the Dirichlet
        # subspace is the fixed zero-trace space, so per-variable node sets
        # cannot be honored. Accept exactly whole-boundary conditions and map
        # them onto the hierarchy's :dirichlet / :full subspaces.
        haskey(m.attrs, "prolongator") &&
            _argerror("the \"prolongator\" attribute is not supported by this geometry's amg")
        bdset = Set(find_boundary(geom))
        sv = copy(low.state_variables)
        for i in 1:size(sv, 1)
            sub = sv[i, 2]
            startswith(String(sub), "dirichlet_") || continue
            pairs = low.dirichlet_nodes[sub]
            if isempty(pairs)
                sv[i, 2] = :full
            elseif Set(pairs) == bdset
                sv[i, 2] = :dirichlet
            else
                _argerror("this geometry builds its Dirichlet subspace by basis " *
                    "truncation, so a Dirichlet condition must cover exactly the " *
                    "whole boundary (find_boundary(geom)); variable $(sv[i, 1]) is " *
                    "constrained on $(length(pairs)) of $(length(bdset)) boundary nodes")
            end
        end
        mg = amg(geom)
    end

    pieces = Convex{T}[_piece(m, mg, low, c) for c in low.cones]
    if length(pieces) == 1 && low.cones[1].pairs === nothing
        Q = pieces[1]
    else
        np = length(pieces)
        select_grid = zeros(T, n, np)
        for (j, c) in enumerate(low.cones)
            if c.pairs === nothing
                select_grid[:, j] .= one(T)
            else
                select_grid[_pairs_to_linear(c.pairs, V), j] .= one(T)
            end
        end
        Q = convex_piecewise(T; Q = Tuple(pieces), mg, select_grid)
    end

    prob = assemble(mg;
        state_variables = sv,
        D = low.D,
        f_grid = low.f_grid,
        g_grid = low.g_grid,
        Q)

    solver_kw = Dict{Symbol,Any}()
    for k in _SOLVER_ATTRS
        haskey(m.attrs, k) && (solver_kw[Symbol(k)] = m.attrs[k])
    end

    m.lowered = low
    try
        sol = mgb_solve(prob; solver_kw...)
        # Lowering diagnostics go into the solve's own log (solver_log /
        # mgb_solution(m).log), never onto the console.
        isempty(low.notes) || (sol = MultiGridBarrier.MGBSOL(
            sol.z, sol.SOL_feasibility, sol.SOL_main,
            join(low.notes, "\n") * "\n" * sol.log, sol.geometry))
        m.sol = sol
        m.status = MOI.OPTIMAL   # convex problem, solved to interior-point tolerance
        m.rawstatus = "mgb_solve converged" *
            (sol.SOL_feasibility === nothing ? "" : " (feasibility phase was required)")
    catch e
        if e isa MGBConvergenceFailure
            m.sol = nothing
            # Map the solver's machine-readable failure code (see
            # MGBConvergenceFailure) to the corresponding MOI status:
            # certified infeasibility, the feasibility_Rmax search limit,
            # a stalled t-ramp, and everything else.
            m.status = e.code === :infeasible       ? MOI.INFEASIBLE :
                       e.code === :feasibility_Rmax ? MOI.OTHER_LIMIT :
                       e.code === :stall            ? MOI.SLOW_PROGRESS :
                       e.code === :iteration_limit  ? MOI.ITERATION_LIMIT :
                       MOI.OTHER_ERROR
            m.rawstatus = sprint(showerror, e)
        else
            rethrow()
        end
    end
    m.solvetime = time() - t0
    nothing
end

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

function _checksolved(m::MGBModel)
    if m.status == MOI.OPTIMIZE_NOT_CALLED
        throw(JuMP.OptimizeNotCalled())
    elseif m.sol === nothing
        _argerror("no solution available (status: $(m.rawstatus)); call optimize!(m) first")
    end
    nothing
end

# Exactly one result is ever produced. The standard `result` keyword is
# accepted everywhere JuMP defines it: result 1 is the solution, other indices
# throw the usual MOI bounds error (status queries return NO_SOLUTION instead,
# per the MOI convention).
function _check_result(m::MGBModel, attr, result::Int)
    result == 1 || throw(MOI.ResultIndexBoundsError(attr, JuMP.result_count(m)))
    return nothing
end

function JuMP.value(v::MGBVarRef{T}; result::Int = 1) where {T}
    m = v.model
    _check_result(m, MOI.VariablePrimal(result), result)
    _checksolved(m)
    col = m.sol.z[:, v.comp]
    v.op === :id ? Vector{T}(col) : Vector{T}(m.geometry.operators[v.op] * col)
end
JuMP.value(c::Coef{T}; result::Int = 1) where {T} = _materialize(c.val, c.model.nnodes)
function JuMP.value(e::MGBExpr{T}; result::Int = 1) where {T}
    m = e.model
    m === nothing &&
        _argerror("cannot evaluate an expression with no variables and no model")
    _check_result(m, MOI.VariablePrimal(result), result)
    _checksolved(m)
    out = Vector{T}(_materialize(e.constant, m.nnodes))
    for ((comp, op), c) in e.terms
        out .+= _materialize(c, m.nnodes) .* JuMP.value(MGBVarRef(m, comp, op))
    end
    out
end

function JuMP.objective_value(m::MGBModel{T}; result::Int = 1) where {T}
    _check_result(m, MOI.ObjectiveValue(result), result)
    _checksolved(m)
    low = m.lowered
    w = m.geometry.w
    sgn = m.objsense == MOI.MAX_SENSE ? -one(T) : one(T)
    acc = zero(T)
    for (j, (comp, op)) in enumerate(low.atoms)
        cj = @view low.f_grid[:, j]
        all(iszero, cj) && continue
        vals = op === :id ? m.sol.z[:, comp] :
               m.geometry.operators[op] * m.sol.z[:, comp]
        acc += dot(w, cj .* vals)
    end
    sgn * acc + dot(w, _materialize(low.obj_offset, m.nnodes))
end

JuMP.termination_status(m::MGBModel) = m.status
JuMP.result_count(m::MGBModel) = m.sol === nothing ? 0 : 1
JuMP.raw_status(m::MGBModel) = m.rawstatus
JuMP.solve_time(m::MGBModel) = m.solvetime
JuMP.primal_status(m::MGBModel; result::Int = 1) =
    (result == 1 && m.sol !== nothing) ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
JuMP.dual_status(m::MGBModel; result::Int = 1) = MOI.NO_SOLUTION   # duals not wired up yet
JuMP.has_values(m::MGBModel; result::Int = 1) =
    JuMP.primal_status(m; result) == MOI.FEASIBLE_POINT
JuMP.dual(::MGBConRef) =
    _argerror("dual extraction is not implemented yet (it is available in principle from the barrier gradient)")

# These two exist on newer JuMP only (>= 1.17 / 1.22); the compat bound is
# JuMP = "1", so define them when the host JuMP has them.
if isdefined(JuMP, :is_solved_and_feasible)
    function JuMP.is_solved_and_feasible(m::MGBModel; dual::Bool = false,
            allow_local::Bool = true, allow_almost::Bool = false, result::Int = 1)
        ts = JuMP.termination_status(m)
        ok = ts == MOI.OPTIMAL || (allow_local && ts == MOI.LOCALLY_SOLVED)
        ok &= JuMP.primal_status(m; result) == MOI.FEASIBLE_POINT
        dual && (ok &= JuMP.dual_status(m; result) == MOI.FEASIBLE_POINT)
        return ok
    end
end
if isdefined(JuMP, :assert_is_solved_and_feasible)
    function JuMP.assert_is_solved_and_feasible(m::MGBModel; kwargs...)
        JuMP.is_solved_and_feasible(m; kwargs...) || error(
            "The model was not solved correctly: termination status " *
            "$(m.status); $(m.rawstatus)")
        return nothing
    end
end

# solution_summary holds a live view of the model; `show` renders the standard
# JuMP summary layout. verbose=true appends the solver iteration log (printing
# per-node primal vectors, as JuMP's verbose summary would, is not useful).
struct MGBSolutionSummary
    model::MGBModel
    verbose::Bool
end
function JuMP.solution_summary(m::MGBModel; result::Int = 1, verbose::Bool = false)
    _check_result(m, MOI.PrimalStatus(result), result)
    MGBSolutionSummary(m, verbose)
end
function Base.show(io::IO, s::MGBSolutionSummary)
    m = s.model
    println(io, "* Solver : MultiGridBarrier")
    println(io)
    println(io, "* Status")
    println(io, "  Result count       : ", JuMP.result_count(m))
    println(io, "  Termination status : ", JuMP.termination_status(m))
    println(io, "  Message from the solver:")
    println(io, "  \"", JuMP.raw_status(m), "\"")
    println(io)
    println(io, "* Candidate solution (result #1)")
    println(io, "  Primal status      : ", JuMP.primal_status(m))
    println(io, "  Dual status        : ", JuMP.dual_status(m))
    m.sol === nothing ||
        println(io, "  Objective value    : ", JuMP.objective_value(m))
    println(io)
    println(io, "* Work counters")
    print(io,   "  Solve time (sec)   : ", JuMP.solve_time(m))
    s.verbose && m.sol !== nothing &&
        print(io, "\n\n* Solver log\n", solver_log(m))
    return nothing
end

mgb_solution(m::MGBModel) = (_checksolved(m); m.sol)

solver_log(m::MGBModel) = (_checksolved(m); m.sol.log)

# attributes (string keys, validated): "prolongator" is consumed by amg, the
# rest forward to mgb_solve. Unknown keys throw here rather than being
# silently ignored at solve time.
const _SOLVER_ATTRS = ("tol", "t", "t_feasibility", "feasibility_Rmax", "maxit",
                       "kappa", "max_newton", "verbose", "device", "logfile")
const _MODEL_ATTRS = ("prolongator", _SOLVER_ATTRS...)
_check_attr(k::String) = k in _MODEL_ATTRS ? nothing : _argerror(
    "unknown attribute \"$k\"; supported attributes: " * join(_MODEL_ATTRS, ", "))
function JuMP.set_attribute(m::MGBModel, k::String, v)
    _check_attr(k)
    m.attrs[k] = v
    _invalidate_solution!(m)
    return nothing
end
JuMP.get_attribute(m::MGBModel, k::String) = (_check_attr(k); get(m.attrs, k, nothing))

# the standard JuMP silencing idiom, driving the "verbose" attribute (the
# solver's only output is the opt-in progress bar)
JuMP.set_silent(m::MGBModel) = JuMP.set_attribute(m, "verbose", false)
JuMP.unset_silent(m::MGBModel) = JuMP.set_attribute(m, "verbose", true)

# ---------------------------------------------------------------------------
# Pretty-printing. JuMP's `print(model)` / `show(::MIME"text/plain")` route
# through `JuMP._print_model`, which requires an AbstractModel to implement
# these three string hooks. Implementing them (rather than only Base.show)
# means a bare `m` at the REPL renders as a native JuMP-style formulation
# instead of a MethodError. Per-node (spatial) coefficients have no scalar
# form, so they print as `⟨coef⟩`.
# ---------------------------------------------------------------------------

_atom_string(m::MGBModel, key::Tuple{Int,Symbol}) =
    key[2] === :id ? String(m.comps[key[1]].name) :
                     "deriv($(m.comps[key[1]].name), :$(key[2]))"

_coef_string(c::UniformCoef) = string(c.scalar)
_coef_string(::NodalCoef) = "⟨coef⟩"

function _terms_string(m::MGBModel, terms, constant)
    parts = String[_coef_string(c) * "*" * _atom_string(m, k) for (k, c) in terms]
    s = isempty(parts) ? "0" : join(parts, " + ")
    (constant === nothing || _iszeroval(constant)) ? s : s * " + " * _coef_string(constant)
end
_row_string(m::MGBModel, r::Row) = _terms_string(m, r.terms, r.constant)

function JuMP.objective_function_string(::MIME, m::MGBModel{T}) where {T}
    m.objexpr === nothing && return "0"
    "∫(" * _terms_string(m, m.objexpr.terms, m.objexpr.constant) * ") dx"
end

function JuMP.constraints_string(::MIME, m::MGBModel{T}) where {T}
    out = String[]
    for c in m.cons
        region = c.pairs === nothing ? "" : " on $(length(c.pairs)) node(s)"
        if c.settag[1] === :eq
            comp = first(keys(c.rows[1].terms))[1]
            push!(out, "$(m.comps[comp].name) == ⟨data⟩" * region)
        elseif c.settag[1] === :nonneg
            for r in c.rows
                push!(out, _row_string(m, r) * " ≥ 0" * region)
            end
        elseif c.settag[1] === :power
            pdesc = _coef_string(c.settag[2])
            q = String[_row_string(m, r) for r in c.rows[1:end-1]]
            push!(out, "($(_row_string(m, c.rows[end]))) ≥ ‖($(join(q, ", ")))‖^$pdesc" * region)
        end
    end
    return out
end

JuMP._nl_subexpression_string(::MIME, ::MGBModel) = String[]

end # module
