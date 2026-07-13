# Convex-set machinery: the Barrier/Convex types, intersect, and barrier(Q).
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

@kwdef struct Barrier{F0,F1,F2}
    f0::F0
    f1::F1
    f2::F2
end


# Helper for convex_piecewise: extract args slice and call piece barrier
# Uses @generated to create compile-time efficient code for tuple slicing
@generated function _call_piece_barrier(f::F, all_rows_and_y::Tuple, ::Val{a}, ::Val{b}) where {F, a, b}
    M = fieldcount(all_rows_and_y)
    # Extract args at positions a to b, plus y at position M
    arg_exprs = [:(all_rows_and_y[$i]) for i in a:b]
    y_expr = :(all_rows_and_y[$M])
    quote
        @inline f($(arg_exprs...), $y_expr)
    end
end

@doc raw"""
    Convex{T}

Container for a convex constraint set used by the MGB solver.

Fields:
- barrier: (F0, F1, F2) - value, gradient, Hessian functions
- cobarrier: (F0, F1, F2) - with slack for feasibility
- slack: initial slack function
- args: tuple of parameter arrays, splatted to map_rows_gpu
- input_spec: construction-time validation of the number of rows in `D`

Barrier functions receive `(args_rows..., y)` where args_rows are per-vertex
parameter data (via broadcasting), and y is the solution SVector.
This enables true GPU execution without scalar indexing.

Construct via helpers like `convex_linear`, `convex_Euclidian_power`, `convex_piecewise`, or `intersect`.
These helpers return a single `Convex{T}` (the barrier is evaluated only at the fine level).
"""
abstract type _ConvexInputSpec end
struct _UncheckedInputs <: _ConvexInputSpec end
struct _AtLeastInputs <: _ConvexInputSpec
    n::Int
end
struct _ExactInputs <: _ConvexInputSpec
    n::Int
end
struct _AllInputSpecs{S<:Tuple} <: _ConvexInputSpec
    specs::S
end

_validate_input_spec(::_UncheckedInputs, ::Int) = nothing
function _validate_input_spec(spec::_AtLeastInputs, nD::Int)
    spec.n <= nD || throw(ArgumentError(
        "convex constraint indexes input row $(spec.n), but D has only $nD row(s)"))
    nothing
end
function _validate_input_spec(spec::_ExactInputs, nD::Int)
    spec.n == nD || throw(ArgumentError(
        "convex constraint with idx = Colon() expects exactly $(spec.n) D row(s), " *
        "but D has $nD row(s)"))
    nothing
end
function _validate_input_spec(spec::_AllInputSpecs, nD::Int)
    foreach(s -> _validate_input_spec(s, nD), spec.specs)
    nothing
end

function _input_spec(idx, n::Int)
    if idx isa Colon
        return _ExactInputs(n)
    end
    isempty(idx) && throw(ArgumentError("idx must contain at least one input row"))
    all(>(0), idx) || throw(ArgumentError("idx entries must be positive; got $(collect(idx))"))
    _AtLeastInputs(maximum(idx))
end

struct Convex{T, Args<:Tuple, B<:Tuple, CB<:Tuple, S, I<:_ConvexInputSpec}
    barrier::B      # (F0, F1, F2) - value, gradient, Hessian (any callable)
    cobarrier::CB   # (F0, F1, F2) - value, gradient, Hessian (any callable)
    slack::S        # slack function (any callable)
    args::Args      # Tuple of parameter arrays for this level
    input_spec::I   # expected input-row layout, checked by assemble
end

# Outer constructor: infer all type parameters
function Convex{T}(barrier::B, cobarrier::CB, slack::S, args::Args,
                   input_spec::I) where {T, B<:Tuple, CB<:Tuple, S,
                                        Args<:Tuple, I<:_ConvexInputSpec}
    Convex{T, Args, B, CB, S, I}(barrier, cobarrier, slack, args, input_spec)
end
Convex{T}(barrier, cobarrier, slack, args) where {T} =
    Convex{T}(barrier, cobarrier, slack, args, _UncheckedInputs())

_validate_convex_inputs(Q::Convex, nD::Int) = _validate_input_spec(Q.input_spec, nD)

# Helper: A' * Diagonal(d) * A for SMatrix, returns flattened SVector.
@inline function _At_diag_A(A::SMatrix{M,N,T}, d::SVector{M,T}) where {M,N,T}
    # (A'DA)[i,j] = sum_k A[k,i] * d[k] * A[k,j]
    H = A' * Diagonal(d) * A
    SVector(H)
end

# GPU-compatible index types: Colon (all) or SVector of Int (static indices)
const GPUIndex = Union{Colon, SVector{<:Any, Int}}


@doc raw"""
    intersect(mg::MultiGrid, U::Convex{T}, rest...) where {T}

Return the intersection of convex domains as a single `Convex{T}`.
Equivalent to `convex_piecewise` with all pieces active at all vertices.
"""
function intersect(mg::MultiGrid, U::Convex{T}, rest::Convex{T}...) where {T}
    pieces = (U, rest...)
    n = length(pieces)
    # All pieces always active
    select_all(x) = ntuple(_ -> true, Val(n))
    convex_piecewise(T; Q=pieces, mg=mg, select=select_all)
end

@doc raw"""    apply_D(D,z) = hcat([D[k]*z for k in 1:length(D)]...)"""
apply_D(D,z) = hcat([D[k]*z for k in 1:length(D)]...)

"""
    barrier(Q::Convex{T}; barrier_weights=nothing) -> Barrier

Create a Barrier from a Convex constraint specification.

The Convex's barrier functions receive row data via broadcasting:
`F0(args_rows..., y)` where `args_rows` are per-vertex parameter data
(from Q.args) and `y` is the solution SVector at that vertex.

This enables true GPU execution without scalar indexing - Q.args
are splatted to map_rows_gpu which broadcasts them together.

With `barrier_weights === nothing` the barrier is the flat average
`(1/n)Σ_k F(Dz_k)` over all nodes. Otherwise `barrier_weights` is a
per-node weight vector (see `_barrier_weights`) and the barrier is
`Σ_k bw_k F(Dz_k)`; nodes with `bw_k == 0` are excluded outright — their
barrier terms are dropped *before* any arithmetic, so an infeasible value
there (`F = ±Inf`, which `0 * Inf` would turn into `NaN`) cannot poison
the sum. Excluded nodes are therefore genuinely unconstrained.
"""
function barrier(Q::Convex{T}; barrier_weights=nothing)::Barrier where T
    (F0, F1, F2) = Q.barrier
    args = Q.args

    if barrier_weights !== nothing
        return _masked_barrier(Q, barrier_weights)
    end

    function f0(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        # Splat Q.args to map_rows_gpu - barriers receive (args_rows..., y)
        y = map_rows_gpu(F0, args..., Dz)
        # Flat-averaged barrier (1/n)Σ F(Dz); the linear term keeps the physical
        # quadrature weights w (paper.tex, "discretization by averaging").
        invn = inv(T(length(w)))
        result = invn*sum(y) + sum(w .* map_rows_gpu(dot, c, Dz))
        result
    end

    function f1(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        # Splat Q.args to map_rows_gpu
        grad_barrier = map_rows_gpu(F1, args..., Dz)
        # Barrier gradient flat-averaged (1/n); linear coefficient c on weights w.
        invn = inv(T(length(w)))
        y = invn .* grad_barrier .+ w .* c
        ret = D[1]' * y[:, 1]
        for k = 2:n
            ret += D[k]' * y[:, k]
        end
        R' * ret
    end

    function f2(z::W, w::W, c, R::Mat, D, z0) where {W, Mat}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        # Splat Q.args to map_rows_gpu
        y = map_rows_gpu(F2, args..., Dz)
        # Barrier Hessian flat-averaged (1/n); the linear term has none.
        invn = inv(T(length(w)))
        # Accumulate with `_hess_add!`, which is in place for BlockHessian
        # operands (see BlockMatrices.jl): every operand below is a fresh
        # temporary owned by this loop, so the mutation/aliasing is safe.
        foo = mgb_diag(D[1], invn .* y[:, 1])
        ret = D[1]' * foo * D[1]
        for j = 2:n
            foo = mgb_diag(D[1], invn .* y[:, (j - 1) * n + j])
            ret = _hess_add!(ret, D[j]' * foo * D[j])
            for k = 1:j-1
                foo = mgb_diag(D[1], invn .* y[:, (j - 1) * n + k])
                ret = _hess_add!(ret, _hess_add!(D[j]' * foo * D[k], D[k]' * foo * D[j]))
            end
        end
        R' * ret * R
    end

    Barrier(; f0, f1, f2)
end

# The `barrier_weights` variant of `barrier(Q)`: barrier terms weighted by the
# per-node vector `bw` instead of the flat 1/n. Structurally identical to the
# flat-average closures above except that every `invn .* y` becomes
# `ifelse.(bz, 0, bw .* y)`: the fused ifelse drops excluded nodes before
# multiplying, because an excluded node may sit outside the barrier domain
# (F = ±Inf there) and `0 * Inf = NaN` would otherwise contaminate the sum.
function _masked_barrier(Q::Convex{T}, bw)::Barrier where T
    (F0, F1, F2) = Q.barrier
    args = Q.args
    bz = iszero.(bw)

    function f0(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        y = map_rows_gpu(F0, args..., Dz)
        result = sum(ifelse.(bz, zero(T), bw .* y)) +
            sum(w .* map_rows_gpu(dot, c, Dz))
        result
    end

    function f1(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        grad_barrier = map_rows_gpu(F1, args..., Dz)
        y = ifelse.(bz, zero(T), bw .* grad_barrier) .+ w .* c
        ret = D[1]' * y[:, 1]
        for k = 2:n
            ret += D[k]' * y[:, k]
        end
        R' * ret
    end

    function f2(z::W, w::W, c, R::Mat, D, z0) where {W, Mat}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        y = map_rows_gpu(F2, args..., Dz)
        # Same `_hess_add!` aliasing discipline as the flat-average f2 above.
        foo = mgb_diag(D[1], ifelse.(bz, zero(T), bw .* y[:, 1]))
        ret = D[1]' * foo * D[1]
        for j = 2:n
            foo = mgb_diag(D[1], ifelse.(bz, zero(T), bw .* y[:, (j - 1) * n + j]))
            ret = _hess_add!(ret, D[j]' * foo * D[j])
            for k = 1:j-1
                foo = mgb_diag(D[1], ifelse.(bz, zero(T), bw .* y[:, (j - 1) * n + k]))
                ret = _hess_add!(ret, _hess_add!(D[j]' * foo * D[k], D[k]' * foo * D[j]))
            end
        end
        R' * ret * R
    end

    Barrier(; f0, f1, f2)
end

"""
    _barrier_weights(w, barrier_nodes) -> nothing | weight vector

Resolve the user-facing `barrier_nodes` selection to the per-node weight
vector consumed by `barrier(Q; barrier_weights)`. The barrier is the flat
average over the *selected* nodes: `indicator(selected) / count(selected)`.
Returns `nothing` whenever the selection is all nodes, which routes
`barrier(Q)` through the exact historical `(1/n)Σ` code path (bit-identical
results on every discretization whose quadrature weights are all nonzero).

Accepted `barrier_nodes` values:
- `AbstractVector{Bool}`: nodal mask, `length(w)` entries. `mgb_driver`'s
  default is the mask of nodes with nonzero quadrature weight,
  `.!iszero.(w)`: all nodes on every standard discretization; on pure-P2
  geometries (`fem2d_P2(bubble=false)`) the triangle corners have exactly
  zero midpoint-rule weight and drop out, collocating the constraints at
  the edge midpoints only.
- `Colon()`: all nodes — the historical flat average, forced.
- `AbstractVector{<:Integer}`: node indices into `1:length(w)`.
"""
_barrier_weights(w, ::Colon) = nothing
function _barrier_weights(w, sel::AbstractVector{Bool})
    length(sel) == length(w) || throw(DimensionMismatch(
        "barrier_nodes mask has length $(length(sel)) but the mesh has $(length(w)) nodes"))
    T = eltype(w)
    _normalize_barrier_weights(w, T.(sel))
end
function _barrier_weights(w, sel::AbstractVector{<:Integer})
    n = length(w)
    isempty(sel) && throw(ArgumentError("barrier_nodes must select at least one node"))
    all(i -> 1 <= i <= n, sel) || throw(ArgumentError(
        "barrier_nodes indices must lie in 1:$n"))
    T = eltype(w)
    v = zeros(T, n)
    v[sel] .= one(T)
    _normalize_barrier_weights(w, v)
end
# Normalize a 0/1 indicator to mean-one-over-selection weights; `nothing` when
# the selection is everything (legacy path). `oftype` moves a CPU-built
# indicator to `w`'s array family (e.g. CuVector) — a no-op when types match.
function _normalize_barrier_weights(w, nz)
    m = sum(nz)
    m > 0 || throw(ArgumentError("barrier_nodes selects no nodes"))
    m == length(nz) && return nothing
    oftype(w, nz ./ m)
end
