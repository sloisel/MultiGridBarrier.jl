export spectral2d, SPECTRAL2D

"""
    SPECTRAL2D{T}

2D spectral discretization descriptor (tensor Chebyshev). Field: `n::Int` (nodes per dim).
"""
struct SPECTRAL2D{T}
    n::Int
end

amg_dim(::SPECTRAL2D{T}) where {T} = 2

# Internal: returns the MultiGrid for SPECTRAL2D.
function _spectral2d_mg(::Type{T}, n::Integer) where {T}
    M = _spectral1d_mg(T, n)
    w = M.geometry.w
    N = length(w)
    w2 = reshape(w,(N,1))
    w2 = w2*(w2')
    w2 = reshape(w2,(N*N,))
    # Tensor (2D) prolongations from the 1D ones: by the Kronecker mixed-product
    # identity, R2d[X][l] = kron(R1d[X][l], R1d[X][l]).
    R = Dict(X => [kron(M.R[X][l], M.R[X][l]) for l in 1:length(M.R[X])]
             for X in keys(M.R))
    xl = _xflat(M.geometry.x)   # flat (N1, 1) Chebyshev nodes
    N1 = size(xl, 1)
    y = reshape(repeat(xl,outer=(1,N1)),(N1*N1,1))
    z = reshape(repeat(xl,outer=(1,N1))',(N1*N1,1))
    x = hcat(y,z)
    ID = M.geometry.operators[:id]
    DX = M.geometry.operators[:dx]
    id = kron(ID,ID)
    dx = kron(DX,ID)
    dy = kron(ID,DX)
    operators = Dict{Symbol,Matrix{T}}(:id => id, :dx => dx, :dy => dy)
    disc = SPECTRAL2D{T}(n)
    x_fine = reshape(x, N1*N1, 1, 2)   # single-element 3-tensor for spectral
    geom = Geometry{T,Array{T,3},Vector{T},Matrix{T},SPECTRAL2D{T}}(
        disc, x_fine, w2, operators)
    return MultiGrid(geom, R)
end

"""
    spectral2d(::Type{T}=Float64; n=4) -> Geometry

Construct a 2D tensor-Chebyshev spectral single-level `Geometry`. Use `amg(geom)` to attach
the spectral multigrid hierarchy.
"""
spectral2d(::Type{T}=Float64;n=4,rest...) where {T} = _spectral2d_mg(T, n).geometry

"""
    find_boundary(geom::Geometry{...,SPECTRAL2D{T}}) -> Vector{Tuple{Int,Int}}

`(v, 1)` pairs (single notional element) for every Chebyshev node on the
perimeter of the tensor-product spectral 2D mesh. Informational only:
the spectral `amg` builds the zero-trace subspace by basis truncation, not
node masking; it does not accept `dirichlet_nodes`.
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL2D{T}}) where {T}
    n = geom.discretization.n
    out = Tuple{Int,Int}[]
    @inbounds for j in 1:n, i in 1:n
        if i == 1 || i == n || j == 1 || j == n
            push!(out, ((j - 1) * n + i, 1))
        end
    end
    return out
end

amg(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL2D{T}}) where {T} =
    _spectral2d_mg(T, geom.discretization.n)

geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,SPECTRAL2D{T}}, L::Int) where {T} =
    _spectral2d_mg(T, geom.discretization.n)


# Internal 2D spectral interpolation function
function spectral2d_interp(MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},z::Array{T,1},x::Array{T,2}) where {T}
    m1 = Int(sqrt(size(MM.x,1)))
    M = spectral1d(T, n=m1)
    Z0 = zeros(T,m1)
    function interp0(z::Array{T,1},x::T,y::T)
        ZW = reshape(z,(m1,m1))
        for k=1:m1
            Z0[k] = spectral1d_interp(M,ZW[:,k],x)[1]
        end
        spectral1d_interp(M,Z0,y)[1]
    end
    function interp1(z::Array{T,1},x::T,y::T)
        ZZ = reshape(z,(m1*m1,:))
        ret1 = zeros(T,size(ZZ,2))
        for k1=1:size(ZZ,2)
            ret1[k1] = interp0(ZZ[:,k1],x,y)
        end
        ret1
    end
    function interp(z::Array{T,1},x::Array{T,2})
        m = Int(size(z,1)/(m1*m1))
        ret2 = zeros(T,(size(x,1),m))
        for k2=1:size(x,1)
            foo = interp1(z,x[k2,1],x[k2,2])
            ret2[k2,:] = foo
        end
        ret2[:]
    end
    interp(z,x)
end

interpolate(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}}, z::Vector{T}, t) where {T} =
    spectral2d_interp(M,z,t)

# plot(::Geometry{...SPECTRAL2D}, z) lives in MultiGridBarrierPyPlotExt.
