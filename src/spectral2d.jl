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
spectral2d(::Type{T}=Float64;n=4) where {T} = _spectral2d_mg(T, n).geometry

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


function _spectral2d_coefficients(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},
        z::AbstractVector{T}) where {T}
    n = MM.discretization.n
    length(z) == n^2 || throw(DimensionMismatch(
        "spectral2d interpolation needs $(n^2) values (got $(length(z)))"))
    nodes = @view MM.x[1:n, 1, 1]
    V = evaluation(nodes, n)
    return V \ reshape(z, n, n) / transpose(V)
end

function _spectral2d_eval(C::AbstractMatrix{T}, x::Real, y::Real) where {T}
    bx = _chebyshev_values(T, x, size(C, 1))
    by = _chebyshev_values(T, y, size(C, 2))
    return dot(bx, C * by)
end

function spectral2d_interp(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},
        z::AbstractVector{T}, point::AbstractVector{<:Real}) where {T}
    length(point) == 2 || throw(DimensionMismatch(
        "a spectral2d interpolation point must have two coordinates"))
    C = _spectral2d_coefficients(MM, z)
    return _spectral2d_eval(C, point[1], point[2])
end

function spectral2d_interp(
        MM::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},
        z::AbstractVector{T}, points::AbstractMatrix{<:Real}) where {T}
    size(points, 2) == 2 || throw(DimensionMismatch(
        "spectral2d interpolation points must form an N-by-2 matrix"))
    C = _spectral2d_coefficients(MM, z)
    out = Vector{T}(undef, size(points, 1))
    @inbounds for i in axes(points, 1)
        out[i] = _spectral2d_eval(C, points[i, 1], points[i, 2])
    end
    return out
end

interpolate(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},
            z::AbstractVector{T}, t) where {T} = spectral2d_interp(M, z, t)

# plot(::Geometry{...SPECTRAL2D}, z) lives in MultiGridBarrierPyPlotExt.
