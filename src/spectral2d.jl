export spectral2d, SPECTRAL2D, spectral2d_solve

"""
    SPECTRAL2D

2D spectral geometry descriptor (tensor Chebyshev). Field: `n::Int` (nodes per dim).
Use with `amgb`.
"""
struct SPECTRAL2D{T}
    n::Int
end

"""
    spectral2d_solve(::Type{T}=Float64;rest...) where {T} = amgb(spectral2d(T;rest...);rest...)
"""
spectral2d_solve(::Type{T}=Float64;rest...) where {T} = amgb(spectral2d(T;rest...);rest...)

amg_dim(::SPECTRAL2D{T}) where {T} = 2


"""
    spectral2d(::Type{T}=Float64; n=4, kwargs...)

Construct 2D spectral geometry with n×n Chebyshev nodes on [-1,1]^2.
Returns SPECTRAL2D{T}; use with subdivide and amgb.
"""
spectral2d(::Type{T}=Float64;n=4,rest...) where {T} = subdivide(SPECTRAL2D{T}(n))

# subdivide method for SPECTRAL2D - generates the multigrid hierarchy
function subdivide(discretization::SPECTRAL2D{T}) where {T}
    n = discretization.n
    M = spectral1d_(T,n)
    L = Int(ceil(log2(n)))
    ls = [min(n,2^k) for k=1:L]
    w = M.w
    N = length(w)
    w = reshape(w,(N,1))
    w = w*(w')
    w = reshape(w,(N*N,))
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
    xl = M.x
    N = size(xl)[1]
    y = reshape(repeat(xl,outer=(1,N)),(N*N,1))
    z = reshape(repeat(xl,outer=(1,N))',(N*N,1))
    x = hcat(y,z)
    ID = M.operators[:id]
    DX = M.operators[:dx]
    id = kron(ID,ID)
    dx = kron(DX,ID)
    dy = kron(ID,DX)
    subspaces = Dict{Symbol,Array{Array{T,2},1}}(:dirichlet => dirichlet, :full => full, :uniform=>uniform)
    operators = Dict{Symbol,Array{T,2}}(:id => id, :dx => dx, :dy => dy)
    return Geometry{T,Matrix{T},SPECTRAL2D{T}}(discretization,
        x,w,subspaces,operators,refine,coarsen)
end


# Internal 2D spectral interpolation function
function spectral2d_interp(MM::Geometry{T,Mat,SPECTRAL2D{T}},z::Array{T,1},x::Array{T,2}) where {T,Mat}
#    n = MM.n
#    M = spectralmesh(T,n)
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
# Implementation of interpolate for SPECTRAL2D
interpolate(M::Geometry{T,Mat,SPECTRAL2D{T}}, z::Vector{T}, t) where {T,Mat} = spectral2d_interp(M,z,t)

function plot(M::Geometry{T,Mat,SPECTRAL2D{T}},z::Array{T,1};x=-1:T(0.01):1,y=-1:T(0.01):1,rest...) where {T,Mat}
    X = repeat(x,1,length(y))
    Y = repeat(y,1,length(x))'
    sz = (length(x),length(y))
    Z = reshape(interpolate(M,z,hcat(X[:],Y[:])),(length(x),length(y)))
    gcf().add_subplot(projection="3d")
    dx = maximum(x)-minimum(x)
    dy = maximum(y)-minimum(y)
    lw = max(dx,dy)*0.002
    plot_surface(Float64.(x), Float64.(y), Float64.(Z); rcount=50, ccount=50, antialiased=false, edgecolor=:black, linewidth=Float64(lw), rest...)
#        plot_wireframe(x,y,Z; rcount=10, ccount=10, color=:white, edgecolor=:black)
end
