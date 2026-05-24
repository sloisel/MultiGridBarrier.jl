# convex_piecewise: piecewise-active constraints and barrier callables.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

# ----------------------------------------------------------------------------
# Piecewise barrier/cobarrier/slack callables.
#
# These seven structs replace the previous inner-function closures in
# `convex_piecewise`. The piece count `N` is in the type parameter (not a
# closure-captured Int), so `Val(N)` inside the call body is a compile-time
# constant and the GPU broadcast kernel can lift the `ntuple(Val(N))`
# specialization through the call site. The remaining work — `_call_piece_barrier`
# is already `@generated` with static `Val{a}`/`Val{b}` slicing — is
# GPU-compatible as a result.
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

struct PiecewiseCobarrierF0{N, BFs, ARGV}
    cobarrier_f0s::BFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseCobarrierF0{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    yhat    = all_rows_and_y[M]
    TT      = eltype(yhat)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? zero(TT) :
            _call_piece_barrier(b.cobarrier_f0s[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    sum(vals)
end

struct PiecewiseCobarrierF1{N, BFs, ARGV}
    cobarrier_f1s::BFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseCobarrierF1{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    yhat    = all_rows_and_y[M]
    NY      = length(yhat)
    TT      = eltype(yhat)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? SVector(ntuple(i -> zero(TT), Val(NY))) :
            _call_piece_barrier(b.cobarrier_f1s[k], all_rows_and_y, b.arg_ranges_val[k]...)
    end
    reduce(+, vals)
end

struct PiecewiseCobarrierF2{N, BFs, ARGV}
    cobarrier_f2s::BFs
    arg_ranges_val::ARGV
end
function (b::PiecewiseCobarrierF2{N})(all_rows_and_y::Vararg{Any,M}) where {N,M}
    sel_row = all_rows_and_y[1]
    yhat    = all_rows_and_y[M]
    NY      = length(yhat)
    TT      = eltype(yhat)
    vals    = ntuple(Val(N)) do k
        iszero(sel_row[k]) ? SVector(ntuple(i -> zero(TT), Val(NY*NY))) :
            _call_piece_barrier(b.cobarrier_f2s[k], all_rows_and_y, b.arg_ranges_val[k]...)
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
    convex_piecewise(::Type{T}=Float64; Q::Tuple{Vararg{Convex{T}}}, mg, select::Function=x->(true,...)) where {T}

Build a single `Convex{T}` that combines multiple convex domains with spatial selectivity.

# Arguments
- `Q::Tuple{Vararg{Convex{T}}}`: tuple of convex pieces, one `Convex{T}` each.
- `mg::MultiGrid`: multigrid hierarchy (provides the fine grid).
- `select::Function`: a function `x -> Tuple{Bool,...}` indicating which pieces are active at spatial position `x`.
- `select_grid`: (optional) pre-computed fine selection grid. If not provided, computed from `select`.

# Semantics
The resulting `Convex` has:
- `barrier(j, y) = ∑(Q[k].barrier(j, y) for k where sel[j][k])`
- `cobarrier(j, yhat) = ∑(Q[k].cobarrier(j, yhat) for k where sel[j][k])`
- `slack(j, y) = max(Q[k].slack(j, y) for k where sel[j][k])`

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
        select_grid = nothing) where {T}

    n = length(Q)  # Number of pieces
    x_fine = _xflat(mg)

    # Pre-compute the fine-level select grid if not provided.
    # select_grid is an N × n matrix indicating which pieces are active.
    # Use T instead of Bool for MPI compatibility.
    if select_grid === nothing
        select_grid = map_rows(xi -> SVector{n,T}(T.(select(xi))), x_fine)
    end

    let
        sel_l = select_grid

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
        combined_args = (sel_l, all_args_flat...)

        # Combined barrier functions receive row data via broadcasting
        # Signature: (sel_row, piece1_args_rows..., piece2_args_rows..., ..., y).
        # Note: sel_row contains T values (not Bool) for MPI compatibility; use
        # !iszero for tests. The seven Piecewise* structs above carry the piece
        # count N as a type parameter so GPU broadcast can lift the
        # `ntuple(Val(N))` specialization through the kernel.
        Nv = Val(n)
        barrier_f0_l    = PiecewiseBarrierF0{n, typeof(barrier_f0s), typeof(arg_ranges_val)}(barrier_f0s, arg_ranges_val)
        barrier_f1_l    = PiecewiseBarrierF1{n, typeof(barrier_f1s), typeof(arg_ranges_val)}(barrier_f1s, arg_ranges_val)
        barrier_f2_l    = PiecewiseBarrierF2{n, typeof(barrier_f2s), typeof(arg_ranges_val)}(barrier_f2s, arg_ranges_val)
        cobarrier_f0_l  = PiecewiseCobarrierF0{n, typeof(cobarrier_f0s), typeof(arg_ranges_val)}(cobarrier_f0s, arg_ranges_val)
        cobarrier_f1_l  = PiecewiseCobarrierF1{n, typeof(cobarrier_f1s), typeof(arg_ranges_val)}(cobarrier_f1s, arg_ranges_val)
        cobarrier_f2_l  = PiecewiseCobarrierF2{n, typeof(cobarrier_f2s), typeof(arg_ranges_val)}(cobarrier_f2s, arg_ranges_val)
        slack_l         = PiecewiseSlack{n, typeof(slack_fns), typeof(arg_ranges_val)}(slack_fns, arg_ranges_val)

        return Convex{T}(
            (barrier_f0_l, barrier_f1_l, barrier_f2_l),
            (cobarrier_f0_l, cobarrier_f1_l, cobarrier_f2_l),
            slack_l,
            combined_args  # Combined args tuple - splatted to map_rows_gpu
        )
    end
end

