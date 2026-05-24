# Convex-set machinery: the Barrier/Convex types, intersect, and barrier(Q).
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

@kwdef struct Barrier
    f0::Function
    f1::Function
    f2::Function
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

Barrier functions receive `(args_rows..., y)` where args_rows are per-vertex
parameter data (via broadcasting), and y is the solution SVector.
This enables true GPU execution without scalar indexing.

Construct via helpers like `convex_linear`, `convex_Euclidian_power`, `convex_piecewise`, or `intersect`.
These helpers return a single `Convex{T}` (the barrier is evaluated only at the fine level).
"""
struct Convex{T, Args<:Tuple, B<:Tuple, CB<:Tuple, S}
    barrier::B      # (F0, F1, F2) - value, gradient, Hessian (any callable)
    cobarrier::CB   # (F0, F1, F2) - value, gradient, Hessian (any callable)
    slack::S        # slack function (any callable)
    args::Args      # Tuple of parameter arrays for this level
end

# Outer constructor: infer all type parameters
function Convex{T}(barrier::B, cobarrier::CB, slack::S, args::Args) where {T, B<:Tuple, CB<:Tuple, S, Args<:Tuple}
    Convex{T, Args, B, CB, S}(barrier, cobarrier, slack, args)
end

# Helper: A' * Diagonal(d) * A for SMatrix or UniformScaling, returns flattened SVector
@inline function _At_diag_A(::UniformScaling, d::SVector{N,T}) where {N,T}
    # I' * Diag(d) * I = Diag(d), flattened column-major
    SVector(ntuple(i -> (i - 1) ÷ N + 1 == (i - 1) % N + 1 ? d[(i - 1) % N + 1] : zero(T), Val(N * N)))
end

@inline function _At_diag_A(A::SMatrix{M,N,T}, d::SVector{M,T}) where {M,N,T}
    # (A'DA)[i,j] = sum_k A[k,i] * d[k] * A[k,j]
    H = A' * Diagonal(d) * A
    SVector(H)
end

# Helper: A' * v for SMatrix or UniformScaling
@inline _At_mul(::UniformScaling, v::SVector) = v
@inline _At_mul(A::SMatrix, v::SVector) = A' * v

# Helper: A * v + b for SMatrix or UniformScaling
@inline _A_mul_plus_b(::UniformScaling, y::SVector, b::SVector) = y .+ b
@inline _A_mul_plus_b(::UniformScaling, y::SVector, b::T) where {T<:Number} = y .+ b
@inline _A_mul_plus_b(A::SMatrix, y::SVector, b::SVector) = A * y .+ b
@inline _A_mul_plus_b(A::SMatrix, y::SVector, b::T) where {T<:Number} = A * y .+ b

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
    barrier(Q::Convex{T}) -> Barrier

Create a Barrier from a Convex constraint specification.

The Convex's barrier functions receive row data via broadcasting:
`F0(args_rows..., y)` where `args_rows` are per-vertex parameter data
(from Q.args) and `y` is the solution SVector at that vertex.

This enables true GPU execution without scalar indexing - Q.args
are splatted to map_rows_gpu which broadcasts them together.
"""
function barrier(Q::Convex{T})::Barrier where T
    (F0, F1, F2) = Q.barrier
    args = Q.args

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
        ret = 0
        for k = 1:n
            foo = D[k]' * y[:, k]
            if k > 1
                ret += foo
            else
                ret = foo
            end
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
        ret = D[1]
        for j = 1:n
            foo = mgb_diag(D[1], invn .* y[:, (j - 1) * n + j])
            bar = (D[j])' * foo * D[j]
            if j > 1
                ret += bar
            else
                ret = bar
            end
            for k = 1:j-1
                foo = mgb_diag(D[1], invn .* y[:, (j - 1) * n + k])
                ret += D[j]' * foo * D[k] + D[k]' * foo * D[j]
            end
        end
        R' * ret * R
    end

    Barrier(; f0, f1, f2)
end
