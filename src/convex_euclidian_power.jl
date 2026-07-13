# convex_Euclidian_power: Euclidian-power constraints and barrier functors.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

# =============================================================================
# GPU-Compatible Barrier Functors for Euclidian Power Constraints
# =============================================================================
#
# These functor structs encode compile-time constants (NZ, IDX) as type parameters
# so the GPU compiler can resolve all dimensions without heap allocation.

"""
    _ep_get_z_and_parts(Ax, bx, idx, y, nz_m1_val)

GPU-compatible helper to compute z = Ax * y[idx] + bx and return its
(q, s) = (leading rows, last row) split. Dispatches on idx type for optimal
GPU code generation.
"""
@inline function _ep_get_z_and_parts(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                      ::Colon, y::SVector{N,TT},
                                      ::Val{NZM1}) where {NZ,NZM1,N,TT}
    z = Ax * y + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ]
    return q, s
end

@inline function _ep_get_z_and_parts(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                      idx::SVector{M,Int}, y::SVector{N,TT},
                                      ::Val{NZM1}) where {NZ,NZM1,M,N,TT}
    # Extract y[idx] using static indexing
    y_idx = SVector{NZ,TT}(ntuple(i -> @inbounds(y[idx[i]]), Val(NZ)))
    z = Ax * y_idx + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ]
    return q, s
end

# Cobarrier version that pops yhat first
@inline function _ep_get_z_and_parts_cobarrier(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                                ::Colon, yhat::SVector{NP1,TT},
                                                ::Val{NZM1}) where {NZ,NZM1,NP1,TT}
    # Pop slack from yhat
    y = _static_pop(yhat, Val(NP1 - 1))
    slack = yhat[NP1]
    z = Ax * y + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ] + slack
    return q, s
end

@inline function _ep_get_z_and_parts_cobarrier(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                                idx::SVector{M,Int}, yhat::SVector{NP1,TT},
                                                ::Val{NZM1}) where {NZ,NZM1,M,NP1,TT}
    # Pop slack from yhat
    y = _static_pop(yhat, Val(NP1 - 1))
    slack = yhat[NP1]
    # Extract y[idx] using static indexing
    y_idx = SVector{NZ,TT}(ntuple(i -> @inbounds(y[idx[i]]), Val(NZ)))
    z = Ax * y_idx + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ] + slack
    return q, s
end

"""
    EuclidianPowerBarrier{NZ,NZM1,IDX}

GPU-compatible functor for barrier function evaluation.
Encodes dimension NZ and index type IDX as type parameters.
"""
struct EuclidianPowerBarrier{NZ,NZM1,IDX}
    idx::IDX  # Store the actual index value
end

EuclidianPowerBarrier(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerBarrier{NZ,NZM1,IDX}(idx)

# Barrier f0 (value)
@inline function (b::EuclidianPowerBarrier{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    α = TT(2) / p0
    -Log(_safe_pow(s, α) - normsquared(q)) - μ * Log(s)
end

"""
    EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}

GPU-compatible functor for barrier gradient evaluation.
"""
struct EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}
    idx::IDX
    core_grad::CoreGrad
end

EuclidianPowerBarrierGrad(::Val{NZ}, ::Val{NZM1}, idx::IDX, cg::CoreGrad) where {NZ,NZM1,IDX,CoreGrad} =
    EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}(idx, cg)

@inline function (b::EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX,CoreGrad}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    grad_z = b.core_grad(q, s, p0, μ)
    grad_idx = Ax' * grad_z
    return _scatter_gradient(b.idx, grad_idx, Val(N))
end

"""
    EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}

GPU-compatible functor for barrier Hessian evaluation.
"""
struct EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}
    idx::IDX
    core_hess::CoreHess
end

EuclidianPowerBarrierHess(::Val{NZ}, ::Val{NZM1}, idx::IDX, ch::CoreHess) where {NZ,NZM1,IDX,CoreHess} =
    EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}(idx, ch)

@inline function (b::EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX,CoreHess}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    H_z_flat = b.core_hess(q, s, p0, μ)
    H_z = reshape(SVector(H_z_flat), Size(NZ, NZ))
    H_idx = Ax' * H_z * Ax
    return _scatter_hessian(b.idx, H_idx, Val(N))
end

"""
    EuclidianPowerCobarrier{NZ,NZM1,IDX}

GPU-compatible functor for cobarrier function evaluation.
"""
struct EuclidianPowerCobarrier{NZ,NZM1,IDX}
    idx::IDX
end

EuclidianPowerCobarrier(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerCobarrier{NZ,NZM1,IDX}(idx)

@inline function (b::EuclidianPowerCobarrier{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    α = TT(2) / p0
    -Log(_safe_pow(s, α) - normsquared(q)) - μ * Log(s)
end

"""
    EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}

GPU-compatible functor for cobarrier gradient evaluation.
"""
struct EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}
    idx::IDX
    core_grad::CoreGrad
end

EuclidianPowerCobarrierGrad(::Val{NZ}, ::Val{NZM1}, idx::IDX, cg::CoreGrad) where {NZ,NZM1,IDX,CoreGrad} =
    EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}(idx, cg)

@inline function (b::EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX,CoreGrad}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    grad_z = b.core_grad(q, s, p0, μ)
    grad_idx = Ax' * grad_z
    return _scatter_cobarrier_gradient(b.idx, grad_idx, grad_z[NZ], Val(NP1))
end

"""
    EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}

GPU-compatible functor for cobarrier Hessian evaluation.
"""
struct EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}
    idx::IDX
    core_hess::CoreHess
end

EuclidianPowerCobarrierHess(::Val{NZ}, ::Val{NZM1}, idx::IDX, ch::CoreHess) where {NZ,NZM1,IDX,CoreHess} =
    EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}(idx, ch)

@inline function (b::EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX,CoreHess}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    q, s = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    H_z_flat = b.core_hess(q, s, p0, μ)
    H_z = reshape(SVector(H_z_flat), Size(NZ, NZ))

    H_idx = Ax' * H_z * Ax
    cross = Ax' * H_z[:, NZ]
    H_ss = H_z[NZ, NZ]

    return _scatter_cobarrier_hessian(b.idx, SVector(H_idx), cross, H_ss, Val(NZ), Val(NP1))
end

"""
    EuclidianPowerSlack{NZ,NZM1,IDX}

GPU-compatible functor for slack computation.
"""
struct EuclidianPowerSlack{NZ,NZM1,IDX}
    idx::IDX
end

EuclidianPowerSlack(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerSlack{NZ,NZM1,IDX}(idx)

@inline function (b::EuclidianPowerSlack{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)

    q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    q_sq = normsquared(q)
    -min(s - _safe_pow(q_sq, p0 / TT(2)), s)
end

# Per-node constraint dimension nz: from idx when given, else from the width of
# A_grid (the flattened per-node nz×nz matrix). Deriving nz from the grid means
# a caller who supplies A_grid never needs the A closure at all.
function _ep_nz(A_grid, idx)
    idx isa Colon || return length(idx)
    ncols = size(A_grid, 2)
    nz = isqrt(ncols)
    nz * nz == ncols || throw(ArgumentError(
        "A_grid has $ncols columns per node; expected a square count nz^2 " *
        "(the per-node A of convex_Euclidian_power is nz×nz, stored flattened)"))
    return nz
end

# Default A grid: the flattened per-node nz×nz matrix sampled from the closure
# `A`; a per-node `UniformScaling` is materialized to the concrete (scaled-)
# identity. When idx is Colon the dimension comes from a one-node sample of `A`
# (which must then be matrix-valued).
function _ep_A_grid(::Type{T}, mg::MultiGrid, A::Function, idx) where {T}
    nz = if idx isa Colon
        A_sample = _sample_node(A, mg)
        A_sample isa UniformScaling && throw(ArgumentError(
            "a UniformScaling A (e.g. the default A = x -> I) with idx = Colon() " *
            "cannot determine the constraint dimension; pass an explicit SVector " *
            "idx, or a matrix-valued A."))
        size(A_sample, 1)
    else
        length(idx)
    end
    map_rows(xi -> begin
            Ax = A(xi)
            if Ax isa UniformScaling
                SVector{nz*nz,T}(vec(Matrix{T}(Ax, nz, nz)))
            else
                SVector{nz*nz,T}(vec(Ax))
            end
        end, _xflat(mg))
end

# Default b grid: a vector-valued `b` fills all nz slots; a scalar-valued `b`
# (e.g. the default `b = x -> T(0)`) lands in the last slot (the `s` row of
# `[q; s]`), zeros elsewhere.
function _ep_b_grid(::Type{T}, mg::MultiGrid, b::Function, A_grid, idx) where {T}
    nz = _ep_nz(A_grid, idx)
    map_rows(xi -> begin
            bx = b(xi)
            if bx isa Number
                SVector{nz,T}(ntuple(i -> i == nz ? T(bx) : zero(T), Val(nz)))
            else
                SVector{nz,T}(bx)
            end
        end, _xflat(mg))
end

@doc raw"""
    convex_Euclidian_power(::Type{T}=Float64; mg, idx=Colon(), A=(x)->I, b=(x)->T(0), p=x->T(2), ...)

Create a convex set defined by Euclidean norm power constraints, with GPU support.

Constructs a `Convex{T}` representing the power cone:
`{y : s ≥ ‖q‖₂^p}` where `[q; s] = A(x)*y[idx] + b(x)`

This is the fundamental constraint for p-Laplace problems where we need
`s ≥ ‖∇u‖^p` for some scalar field u.

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `mg::MultiGrid`: Required. The multigrid hierarchy (provides the fine grid).
- `idx=Colon()`: Indices of y to which transformation applies
- `A::Function`: Matrix function `x -> A(x)` for linear transformation
- `b::Function`: Vector function `x -> b(x)` for affine shift
- `p::Function`: Exponent function `x -> p(x)` where p(x) ≥ 1
- `A_grid`, `b_grid`, `p_grid`: Pre-computed fine grids; they default to sampling
  `A`, `b`, `p` at the mesh nodes, so pass them to skip the closures entirely.

# Returns
A `Convex{T}` whose `args` carry the pre-computed fine parameter grids
(`A`, `b`, `p`, `μ`); the barrier functors receive `(args_rows..., y)` per
vertex via broadcasting (see [`Convex`](@ref)).

# Mathematical Details
The barrier function is `-log(s^(2/p) - ‖q‖²) - μ(p)*log(s)`,
where μ(p) = 0 if p∈{1,2}, 1 if p<2, 2 if p>2. In particular:
- For p = 1: `-log(s² - ‖q‖²)`
- For p = 2: `-log(s - ‖q‖²)`

# Examples
```julia
# Standard p-Laplace constraint with GPU support. With the default 1D layout
# Dz = (u, ∂u/∂x, s), idx = SVector(2, 3) selects (q, s) = (∂u/∂x, s).
mg = amg(fem1d(Float32; nodes=collect(range(-1f0, 1f0, length=33))))
Q = convex_Euclidian_power(Float32; mg=mg, idx=SVector(2, 3), p=x->1.5f0)

# Q is a single Convex{Float32}
```
"""
function convex_Euclidian_power(::Type{T}=Float64;
        mg::MultiGrid,
        idx::GPUIndex=Colon(),
        A::Function=(x)->I,
        b::Function=(x)->T(0),
        p::Function=x->T(2),
        # Pre-computed grids; default to sampling the closures at the mesh nodes.
        A_grid = _ep_A_grid(T, mg, A, idx),
        b_grid = _ep_b_grid(T, mg, b, A_grid, idx),
        p_grid = map_rows(xi -> T(p(xi)), _xflat(mg))) where {T}

    nz = _ep_nz(A_grid, idx)
    input_spec = _input_spec(idx, nz)

    # Shape consistency: the barrier functors reconstruct the per-node A as an
    # nz×nz SMatrix and require b of length nz. Catch mismatches here rather
    # than as an inscrutable StaticArrays error (or a functor MethodError) at
    # solve time.
    size(A_grid, 2) == nz * nz || throw(ArgumentError(
        "A_grid has $(size(A_grid, 2)) columns per node but nz = $nz requires " *
        "nz^2 = $(nz * nz) (the per-node A is nz×nz, stored flattened)"))
    ncb = b_grid isa AbstractVector ? 1 : size(b_grid, 2)
    ncb == nz || throw(ArgumentError(
        "b_grid has $ncb value(s) per node but the per-node constraint vector " *
        "[q; s] has nz = $nz components"))

    # Pre-compute mu grid on CPU (eliminates conditional in GPU barrier)
    # mu = 0 for p=1 or p=2, mu = 1 for p<2, mu = 2 for p>2
    mu_func(p0) = (p0 == 2 || p0 == 1) ? T(0) : (p0 < 2 ? T(1) : T(2))
    mu_grid = map_rows(p_val -> mu_func(T(p_val)), p_grid)

    # Compile-time constant for static pop operations
    nz_m1_val = Val(nz - 1)

    # Core gradient w.r.t. (q, s) - GPU compatible, receives μ as parameter
    @inline function core_grad(q::SVector{NQ,TT}, s::TT, p0::TT, μ::TT) where {NQ,TT}
        α = TT(2) / p0
        q_sq = normsquared(q)
        s_α = _safe_pow(s, α)
        r = s_α - q_sq
        inv_r = one(TT) / r
        grad_q = (TT(2) * inv_r) .* q
        s_α_m1 = _safe_pow(s, α - one(TT))
        grad_s = -α * s_α_m1 * inv_r - μ / s
        return push(grad_q, grad_s)
    end

    # Core Hessian - GPU compatible, receives μ as parameter
    @inline function core_hess(q::SVector{NQ,TT}, s::TT, p0::TT, μ::TT) where {NQ,TT}
        α = TT(2) / p0
        q_sq = normsquared(q)
        s_α = _safe_pow(s, α)
        r = s_α - q_sq
        inv_r = one(TT) / r
        inv_r2 = inv_r * inv_r
        s_α_m1 = _safe_pow(s, α - one(TT))
        coef_qs = -TT(2) * α * s_α_m1 * inv_r2
        s_α_m2 = _safe_pow(s, α - TT(2))
        s_2α_m2 = _safe_pow(s, TT(2) * α - TT(2))
        H_ss = -α * (α - one(TT)) * s_α_m2 * inv_r + α * α * s_2α_m2 * inv_r2 + μ / (s * s)

        nz_local = NQ + 1
        H = SMatrix{nz_local, nz_local, TT}(ntuple(Val(nz_local * nz_local)) do k
            i = (k - 1) % nz_local + 1
            j = (k - 1) ÷ nz_local + 1
            result = zero(TT)
            if i <= NQ && j <= NQ
                result = TT(4) * q[i] * q[j] * inv_r2
                if i == j
                    result += TT(2) * inv_r
                end
            elseif i <= NQ && j == nz_local
                result = coef_qs * q[i]
            elseif i == nz_local && j <= NQ
                result = coef_qs * q[j]
            else
                result = H_ss
            end
            result
        end)
        return SVector(H)
    end

    # Build the Convex using GPU-compatible functors.
    # Functors encode nz and idx as type parameters for GPU compilation
    nz_val = Val(nz)
    barrier_f0 = EuclidianPowerBarrier(nz_val, nz_m1_val, idx)
    barrier_f1 = EuclidianPowerBarrierGrad(nz_val, nz_m1_val, idx, core_grad)
    barrier_f2 = EuclidianPowerBarrierHess(nz_val, nz_m1_val, idx, core_hess)
    cobarrier_f0 = EuclidianPowerCobarrier(nz_val, nz_m1_val, idx)
    cobarrier_f1 = EuclidianPowerCobarrierGrad(nz_val, nz_m1_val, idx, core_grad)
    cobarrier_f2 = EuclidianPowerCobarrierHess(nz_val, nz_m1_val, idx, core_hess)
    slack_f = EuclidianPowerSlack(nz_val, nz_m1_val, idx)

    return Convex{T}(
        (barrier_f0, barrier_f1, barrier_f2),
        (cobarrier_f0, cobarrier_f1, cobarrier_f2),
        slack_f,
        (A_grid, b_grid, p_grid, mu_grid),  # args tuple - includes pre-computed μ values
        input_spec
    )
end
