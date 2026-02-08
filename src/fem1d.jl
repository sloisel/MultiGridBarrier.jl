export fem1d, FEM1D, fem1d_solve

"""
    FEM1D{T}

1D FEM geometry descriptor. Field: `L::Int` (levels). Use with `amgb`.
"""
struct FEM1D{T}
    L::Int
end

amg_dim(::FEM1D) = 1

"""
    fem1d_solve(::Type{T}=Float64;rest...) where {T} = amgb(fem1d(T;rest...);rest...)
"""
fem1d_solve(::Type{T}=Float64;rest...) where {T} = amgb(fem1d(T;rest...);rest...)

"""
    fem1d(::Type{T}=Float64; L=4, kwargs...)

Construct 1D FEM geometry (piecewise linear) on [-1, 1].
Returns a Geometry suitable for use with `amgb`. Keyword `L` sets 2^L elements.
"""
fem1d(::Type{T}=Float64;L=4,rest...) where {T} = subdivide(FEM1D{T}(L))

# subdivide method for FEM1D - generates the multigrid hierarchy
function subdivide(discretization::FEM1D{T}) where {T}
    L = discretization.L
    ls = [2^k for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))',(2*n0,1)) .* 2 .- 1
        N = size(x[l])[1]
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T,I,N,N)
        uniform[l] = sparse(ones(T,(N,1)))
    end
    N = size(x[L])[1]
    w = repeat([T(2)/N],outer=(N,))
    id = sparse(T,I,N,N)
    dx = blockdiag(repeat([sparse(T[-2^(L-1) 2^(L-1)
                                    -2^(L-1) 2^(L-1)])],outer=(2^L,))...)
    refine[L] = id
    coarsen[L] = id
    for l=1:L-1
        n0 = 2^l
        refine[l] = blockdiag(
            repeat([sparse(T[1.0 0.0
                     0.5 0.5
                     0.5 0.5
                     0.0 1.0])],outer=(n0,))...)
        coarsen[l] = blockdiag(
            repeat([sparse(T[1 0 0 0
                     0 0 0 1])],outer=(n0,))...)
    end
    subspaces = Dict(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict(:id => id, :dx => dx)
    return Geometry{T,Matrix{T},Vector{T},SparseMatrixCSC{T,Int},FEM1D{T}}(discretization,
        x[end],w,subspaces,operators,refine,coarsen)
end

# Internal interpolation function for piecewise linear functions
function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::T) where{T}
    b = length(x)
    if t<x[1]
        return y[1]
    elseif t>x[b]
        return y[b]
    end
    a = 1
    while b-a>1
        c = (a+b)÷2
        if x[c]<=t
            a=c
        else
            b=c
        end
    end
    w = (t-x[a])/(x[b]-x[a])
    return w*y[b]+(1-w)*y[a]
end

# Vector version of fem1d_interp
function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}
    [fem1d_interp(x,y,t[k]) for k=1:length(t)]
end

# Implementation of interpolate for FEM1D
interpolate(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,<:Any,<:Any,FEM1D{T}}, z::Vector{T}, t) where {T} = fem1d_interp(reshape(M.x,(:,)),z,t)

plot(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,<:Any,<:Any,FEM1D{T}}, z::Vector{T}; kwargs...) where {T} = plot(M.x,z; kwargs...)
