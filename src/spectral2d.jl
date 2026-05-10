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
    L = Int(ceil(log2(n)))
    ls = [min(n,2^k) for k=1:L]
    w = M.geometry.w
    N = length(w)
    w2 = reshape(w,(N,1))
    w2 = w2*(w2')
    w2 = reshape(w2,(N*N,))
    dirichlet = Array{Array{T,2},1}(undef,(L,))
    full = Array{Array{T,2},1}(undef,(L,))
    uniform = Array{Array{T,2},1}(undef,(L,))
    refine = Array{Array{T,2},1}(undef,(L,))
    coarsen = Array{Array{T,2},1}(undef,(L,))
    for l=1:L
        S = M.subspaces
        dirichlet[l] = kron(S[:dirichlet][l],S[:dirichlet][l])
        full[l] = kron(S[:full][l],S[:full][l])
        uniform[l] = kron(S[:uniform][l],S[:uniform][l])
        refine[l] = kron(M.refine[l],M.refine[l])
        coarsen[l] = kron(M.coarsen[l],M.coarsen[l])
    end
    xl = M.geometry.x
    N1 = size(xl)[1]
    y = reshape(repeat(xl,outer=(1,N1)),(N1*N1,1))
    z = reshape(repeat(xl,outer=(1,N1))',(N1*N1,1))
    x = hcat(y,z)
    ID = M.geometry.operators[:id]
    DX = M.geometry.operators[:dx]
    id = kron(ID,ID)
    dx = kron(DX,ID)
    dy = kron(ID,DX)
    subspaces = Dict{Symbol,Vector{Matrix{T}}}(:dirichlet => dirichlet, :full => full, :uniform=>uniform)
    operators = Dict{Symbol,Matrix{T}}(:id => id, :dx => dx, :dy => dy)
    disc = SPECTRAL2D{T}(n)
    geom = Geometry{T,Matrix{T},Vector{T},Matrix{T},Matrix{T},SPECTRAL2D{T}}(
        disc, x, w2,
        Dict{Symbol,Matrix{T}}(:dirichlet => dirichlet[end], :full => full[end], :uniform => uniform[end]),
        operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

"""
    spectral2d(::Type{T}=Float64; n=4) -> Geometry

Construct a 2D tensor-Chebyshev spectral single-level `Geometry`. Use `amg(geom)` to attach
the spectral multigrid hierarchy.
"""
spectral2d(::Type{T}=Float64;n=4,rest...) where {T} = _spectral2d_mg(T, n).geometry

amg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,SPECTRAL2D{T}}) where {T} =
    _spectral2d_mg(T, geom.discretization.n)

geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,SPECTRAL2D{T}}, L::Int;
             structured::Bool=false, kwargs...) where {T} =
    _spectral2d_mg(T, geom.discretization.n)


# Internal 2D spectral interpolation function
function spectral2d_interp(MM::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,SPECTRAL2D{T}},z::Array{T,1},x::Array{T,2}) where {T}
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

interpolate(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,SPECTRAL2D{T}}, z::Vector{T}, t) where {T} =
    spectral2d_interp(M,z,t)

function plot(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,SPECTRAL2D{T}},z::Array{T,1};x=-1:T(0.01):1,y=-1:T(0.01):1,rest...) where {T}
    X = repeat(x,1,length(y))
    Y = repeat(y,1,length(x))'
    sz = (length(x),length(y))
    Z = reshape(interpolate(M,z,hcat(X[:],Y[:])),(length(x),length(y)))
    gcf().add_subplot(projection="3d")
    dx = maximum(x)-minimum(x)
    dy = maximum(y)-minimum(y)
    lw = max(dx,dy)*0.002
    plot_surface(Float64.(x), Float64.(y), Float64.(Z); rcount=50, ccount=50, antialiased=false, edgecolor=:black, linewidth=Float64(lw), rest...)
end
