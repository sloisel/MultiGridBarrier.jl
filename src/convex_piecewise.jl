# convex_piecewise: piecewise-active constraints and barrier callables.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

# ----------------------------------------------------------------------------
# Piecewise barrier/cobarrier/slack callables.
#
# Top-level functor structs (not inner-function closures) so the GPU broadcast
# kernel can specialize: the piece count `N` is a type parameter, making
# `Val(N)` in the call body a compile-time constant, and `_call_piece_barrier`
# is `@generated` with static `Val{a}`/`Val{b}` slicing. The barrier and
# cobarrier variants share the same structs — the callable never inspects
# which triple it holds.
# ----------------------------------------------------------------------------

struct PiecewiseBarrierF0{N, BFs, ARGV}
    barrier_f0s::BFs      # NTuple{N, ...}
    arg_ranges_val::ARGV  # NTuple{N, Tuple{Val{Int}, Val{Int}}}
end
function (b::PiecewiseBarrierF0{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    y       = all_rows_and_y[M]
    TT      = eltype(y)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? zero(TT) :
            _call_piece_barrier(b.barrier_f0s[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    sum(vals)
end

struct PiecewiseBarrierF1{N, BFs, ARGV}
    barrier_f1s::BFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseBarrierF1{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    y       = all_rows_and_y[M]
    NY      = length(y)
    TT      = eltype(y)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? SVector(ntuple(i -> zero(TT), Val(NY))) :
            _call_piece_barrier(b.barrier_f1s[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    reduce(+, vals)
end

struct PiecewiseBarrierF2{N, BFs, ARGV}
    barrier_f2s::BFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseBarrierF2{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    y       = all_rows_and_y[M]
    NY      = length(y)
    TT      = eltype(y)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? SVector(ntuple(i -> zero(TT), Val(NY*NY))) :
            _call_piece_barrier(b.barrier_f2s[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    reduce(+, vals)
end

struct PiecewiseSlack{N, SFs, ARGV}
    slack_fns::SFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseSlack{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    y       = all_rows_and_y[M]
    TT      = eltype(y)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? typemin(TT) :
            _call_piece_barrier(b.slack_fns[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    maximum(vals)
end

@doc raw"""
    convex_piecewise(::Type{T}=Float64; Q::Tuple{Vararg{Convex{T}}}, mg, select::Function=x->(true,...), select_grid=<sampled from select>) where {T}

Build a single `Convex{T}` that combines multiple convex domains with spatial selectivity.

# Arguments
- `Q::Tuple{Vararg{Convex{T}}}`: tuple of convex pieces, one `Convex{T}` each.
- `mg::MultiGrid`: multigrid hierarchy (provides the fine grid).
- `select::Function`: a function `x -> Tuple{Bool,...}` indicating which pieces are active at spatial position `x`.
- `select_grid`: pre-computed fine selection grid; defaults to sampling `select`
  at the mesh nodes, so pass it to skip the closure entirely.

# Semantics
At each vertex, over the pieces `k` that are active there (`select(x)[k]` true):
- `barrier   = ∑ₖ Q[k].barrier`
- `cobarrier = ∑ₖ Q[k].cobarrier`
- `slack     = maxₖ Q[k].slack`

The slack is the maximum over active pieces, ensuring a single slack value suffices for feasibility.

# Examples
```julia
# Intersection of two convex domains
U = convex_Euclidian_power(Float64; mg=mg, idx=SVector(1, 3), p=x->2)
V = convex_linear(Float64; mg=mg, A=x->A_matrix, b=x->b_vector)
select_both(x) = (true, true)
Qint = convex_piecewise(Float64; Q=(U, V), mg=mg, select=select_both)

# Region-dependent constraints
Q_left = convex_Euclidian_power(Float64; mg=mg, p=x->1.5)
Q_right = convex_Euclidian_power(Float64; mg=mg, p=x->2.0)
select(x) = (x[1] < 0, x[1] >= 0)
Qreg = convex_piecewise(Float64; Q=(Q_left, Q_right), mg=mg, select=select)
```

See also: [`intersect`](@ref), [`convex_linear`](@ref), [`convex_Euclidian_power`](@ref).
"""
function convex_piecewise(::Type{T}=Float64;
        Q::Tuple{Vararg{Convex{T}}},
        mg::MultiGrid,
        select::Function = x -> ntuple(_ -> true, length(Q)),
        # select_grid is an N × n matrix indicating which pieces are active,
        # stored as T (not Bool) for MPI compatibility; defaults to sampling
        # `select` at the mesh nodes.
        select_grid = map_rows(xi -> SVector{length(Q),T}(T.(select(xi))), _xflat(mg))) where {T}

    n = length(Q)  # Number of pieces

    # Extract all barrier functions into tuples (one entry per piece)
    barrier_f0s = ntuple(k -> Q[k].barrier[1], Val(n))
    barrier_f1s = ntuple(k -> Q[k].barrier[2], Val(n))
    barrier_f2s = ntuple(k -> Q[k].barrier[3], Val(n))
    cobarrier_f0s = ntuple(k -> Q[k].cobarrier[1], Val(n))
    cobarrier_f1s = ntuple(k -> Q[k].cobarrier[2], Val(n))
    cobarrier_f2s = ntuple(k -> Q[k].cobarrier[3], Val(n))
    slack_fns = ntuple(k -> Q[k].slack, Val(n))

    # Collect args from all pieces; each piece's args is a tuple, concatenate them.
    piece_args = ntuple(k -> Q[k].args, Val(n))

    # Compute cumulative arg lengths for slicing
    # arg_lengths[k] = number of args for piece k
    arg_lengths = map(length, piece_args)

    # Compute start indices for each piece's args (1-based, after sel)
    # sel is at position 1, so piece args start at position 2
    # Piece 1: starts at 2
    # Piece 2: starts at 2 + arg_lengths[1]
    # Piece k: starts at 2 + sum(arg_lengths[1:k-1])
    arg_starts = ntuple(Val(n)) do k
        2 + sum(arg_lengths[1:k-1]; init=0)
    end
    arg_ends = ntuple(Val(n)) do k
        arg_starts[k] + arg_lengths[k] - 1
    end

    # Store ranges as tuples of Val for compile-time slicing
    arg_ranges_val = ntuple(Val(n)) do k
        (Val(arg_starts[k]), Val(arg_ends[k]))
    end

    # Flatten all args into combined_args tuple
    all_args_flat = reduce((a, b) -> (a..., b...), piece_args; init=())
    combined_args = (select_grid, all_args_flat...)

    # Combined barrier functions receive row data via broadcasting
    # Signature: (sel_row, piece1_args_rows..., piece2_args_rows..., ..., y).
    # Note: sel_row contains T values (not Bool) for MPI compatibility; use
    # !iszero for tests. The cobarrier callables reuse the barrier structs
    # with the cobarrier tuples inside (the callable never inspects which).
    barrier_f0_l    = PiecewiseBarrierF0{n, typeof(barrier_f0s), typeof(arg_ranges_val)}(barrier_f0s, arg_ranges_val)
    barrier_f1_l    = PiecewiseBarrierF1{n, typeof(barrier_f1s), typeof(arg_ranges_val)}(barrier_f1s, arg_ranges_val)
    barrier_f2_l    = PiecewiseBarrierF2{n, typeof(barrier_f2s), typeof(arg_ranges_val)}(barrier_f2s, arg_ranges_val)
    cobarrier_f0_l  = PiecewiseBarrierF0{n, typeof(cobarrier_f0s), typeof(arg_ranges_val)}(cobarrier_f0s, arg_ranges_val)
    cobarrier_f1_l  = PiecewiseBarrierF1{n, typeof(cobarrier_f1s), typeof(arg_ranges_val)}(cobarrier_f1s, arg_ranges_val)
    cobarrier_f2_l  = PiecewiseBarrierF2{n, typeof(cobarrier_f2s), typeof(arg_ranges_val)}(cobarrier_f2s, arg_ranges_val)
    slack_l         = PiecewiseSlack{n, typeof(slack_fns), typeof(arg_ranges_val)}(slack_fns, arg_ranges_val)

    return Convex{T}(
        (barrier_f0_l, barrier_f1_l, barrier_f2_l),
        (cobarrier_f0_l, cobarrier_f1_l, cobarrier_f2_l),
        slack_l,
        combined_args,  # Combined args tuple - splatted to map_rows_gpu
        _AllInputSpecs(map(q -> q.input_spec, Q))
    )
end
