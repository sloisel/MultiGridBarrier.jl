export spectral1d, SPECTRAL1D

"""
    SPECTRAL1D{T}

1D spectral discretization descriptor (Chebyshev). Field: `n::Int` (nodes).
"""
struct SPECTRAL1D{T}
    n::Int
end

amg_dim(::SPECTRAL1D{T}) where {T} = 1

function chebfun(c::Array{T,2}, x::T) where {T}
    n = size(c,1)-1
    if x>1
        return c'*cosh.((0:n).*acosh(x))
    elseif x>=-1
        return c'*cos.((0:n).*acos(x))
    end
    s = ones(T,n+1)
    s[2:2:n+1] .= T(-1)
    return c'*(s.*cosh.((0:n).*acosh(-x)))
end
function chebfun(c::Array{T}, x::Array{T}) where {T}
    sc = size(c)
    sx = size(x)
    c = reshape(c,(sc[1],:))
    m = size(c,2)
    n = prod(sx)
    x = reshape(x,n)
    y = zeros(T,n,m)
    for k=1:n
        y[k,:] = chebfun(c,x[k])
    end
    if length(sc)==1
        return reshape(y,sx)
    end
    return reshape(y,(sx...,sc[2:end]...))
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

function evaluation(xs::Array{T},n::Integer) where {T}
    m = size(xs,1)
    n = n-1
    M = zeros(T,(m,n+1))
    for j=1:m
        x = xs[j]
        if x>1
            M[j,:] = cosh.((0:n).*acosh(x))
        elseif x>=-1
            M[j,:] = cos.((0:n).*acos(x))
        else
            s = ones(T,n+1)
            s[2:2:n+1] .= T(-1)
            M[j,:] = s.*cosh.((0:n).*acosh(-x))
        end
    end
    M
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
function spectral1d_interp(MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}}, y::Array{T,1},x) where {T}
    n = length(MM.w)
    M = evaluation(_xflat(MM.x),n)
    m1 = size(M,1)
    @assert m1==size(M,2)
    sz = size(y)
    y1 = reshape(y,(m1,:))
    z = chebfun(M\y1,x)
    if length(sz)==1
        ret = z
    else
        ret = reshape(z,(size(x)...,sz[2:end]...))
    end
    ret
end

interpolate(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}}, z::Vector{T}, t) where {T} =
    spectral1d_interp(M,z,t)

# plot(::Geometry{...SPECTRAL1D}, y) lives in MultiGridBarrierPyPlotExt.
