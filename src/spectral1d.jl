export spectral1d, SPECTRAL1D

"""
    SPECTRAL1D{T}

1D spectral discretization descriptor (Chebyshev). Field: `n::Int` (nodes).
"""
struct SPECTRAL1D{T}
    n::Int
end

amg_dim(::SPECTRAL1D{T}) where {T} = 1

function _chebyshev_values(::Type{T}, x::Real, n::Int) where {T}
    n >= 1 || throw(ArgumentError("the Chebyshev basis must contain at least one term"))
    xx = T(x)
    values = Vector{T}(undef, n)
    values[1] = one(T)
    if n >= 2
        values[2] = xx
        @inbounds for j in 3:n
            values[j] = 2 * xx * values[j-1] - values[j-2]
        end
    end
    return values
end

chebfun(c::AbstractVector{T}, x::Real) where {T} =
    dot(c, _chebyshev_values(T, x, length(c)))

function chebfun(c::AbstractMatrix{T}, x::Real) where {T}
    return transpose(c) * _chebyshev_values(T, x, size(c, 1))
end

function chebfun(c::AbstractVector{T}, x::AbstractArray{<:Real}) where {T}
    out = Array{T}(undef, size(x))
    @inbounds for i in eachindex(x, out)
        out[i] = chebfun(c, x[i])
    end
    return out
end

function derivative(::Type{T},n::Integer) where {T}
    D = zeros(T,(n,n))
    for j=1:n-1
        for k=j+1:2:n
            D[j,k] = 2*(k-1)
        end
    end
    D[1,:]/=2
    D
end

function evaluation(xs::AbstractArray{T}, n::Integer) where {T}
    M = Matrix{T}(undef, length(xs), n)
    @inbounds for (j, x) in enumerate(xs)
        M[j, :] = _chebyshev_values(T, x, Int(n))
    end
    return M
end

# Internal: returns the MultiGrid hierarchy (and the fine-level Chebyshev nodes/operators).
function _spectral1d_mg(::Type{T}, n::Integer) where {T}
    L = Int(ceil(log2(n)))
    ls = [min(n,2^k) for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    dirichlet = Array{Array{T,2},1}(undef,(L,))
    full = Array{Array{T,2},1}(undef,(L,))
    uniform = Array{Array{T,2},1}(undef,(L,))
    refine = Array{Array{T,2},1}(undef,(L,))
    # w and M keep their last-level values after the loop.
    local w::Vector{T}
    local M::Matrix{T}
    for l=1:L
        Q = ClenshawCurtisQuadrature(T,ls[l])
        nodes,weights = Q.nodes,Q.weights
        w = 2 .* weights
        x[l] = reshape(2 .* nodes .- 1,(length(w),1))   # internal flat (n_l, 1) view
        M = evaluation(x[l],ls[l])
        @assert size(M,1)==size(M,2)
        CI = M[:,3:end]
        for k=1:2:size(CI,2)
            CI[:,k] -= M[:,1]
        end
        for k=2:2:size(CI,2)
            CI[:,k] -= M[:,2]
        end
        dirichlet[l] = CI
        full[l] = M
        uniform[l] = ones(T,(size(x[l],1),1))
    end
    D0 = derivative(T,ls[L])
    @assert size(D0,1)==size(D0,2)
    dx = M*D0/M
    id = Matrix{T}(I,ls[L],ls[L])
    refine[L] = id
    for l=1:L-1
        refine[l] = evaluation(x[l+1],ls[l])/full[l]
    end
    subspaces = Dict{Symbol,Vector{Matrix{T}}}(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict{Symbol,Matrix{T}}(:id => id, :dx => dx)
    disc = SPECTRAL1D{T}(n)
    # Spectral has no natural element structure; wrap the flat (n, 1) into a
    # single-element 3-tensor (n, 1, 1).
    x_fine = reshape(x[end], size(x[end], 1), 1, 1)
    geom = Geometry{T,Array{T,3},Vector{T},Matrix{T},SPECTRAL1D{T}}(
        disc, x_fine, w, operators)
    return MultiGrid(geom, subspaces, refine)
end

"""
    spectral1d(::Type{T}=Float64; n=16) -> Geometry

Construct a 1D spectral single-level `Geometry` with `n` Chebyshev nodes. Use
`amg(geom)` to attach a multigrid hierarchy.
"""
spectral1d(::Type{T}=Float64;n=16) where {T} = _spectral1d_mg(T, n).geometry

"""
    find_boundary(geom::Geometry{...,SPECTRAL1D{T}}) -> Vector{Tuple{Int,Int}}

The two `(v, t)` pairs `[(1, 1), (n, 1)]` for the endpoint Chebyshev nodes.
Spectral geometries use a single notional element (`size(geom.x, 2) == 1`)
since there is no real per-element structure; the flattened view has one
row per unique node. Informational only: the spectral `amg` builds the
zero-trace subspace by basis truncation, not node masking; it does not
accept `dirichlet_nodes`.
"""
find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL1D{T}}) where {T} =
    [(1, 1), (geom.discretization.n, 1)]

# amg(::Geometry{SPECTRAL1D}) -> MultiGrid
amg(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL1D{T}}) where {T} =
    _spectral1d_mg(T, geom.discretization.n)

# Spectral has no meaningful geometric subdivision distinct from amg, so geometric_mg
# returns the same hierarchy.
geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL1D{T}}, L::Int) where {T} =
    _spectral1d_mg(T, geom.discretization.n)

# Internal spectral interpolation function
function _spectral1d_coefficients(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}},
        y::AbstractVector{T}) where {T}
    n = length(MM.w)
    length(y) == n || throw(DimensionMismatch(
        "spectral1d interpolation needs $n values (got $(length(y)))"))
    M = evaluation(_xflat(MM.x),n)
    @assert size(M, 1) == size(M, 2)
    return M \ y
end

function spectral1d_interp(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}},
        y::AbstractVector{T}, x::Real) where {T}
    return chebfun(_spectral1d_coefficients(MM, y), x)
end

function spectral1d_interp(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}},
        y::AbstractVector{T}, x::AbstractArray{<:Real}) where {T}
    return chebfun(_spectral1d_coefficients(MM, y), x)
end

interpolate(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}},
            z::AbstractVector{T}, t) where {T} = spectral1d_interp(M, z, t)

# plot(::Geometry{...SPECTRAL1D}, y) lives in MultiGridBarrierPyPlotExt.
