export fem1d, FEM1D, fem1d_solve

"    abstract type FEM1D end"
abstract type FEM1D end

"    amg_dim(::Type{FEM1D}) = 1"
amg_dim(::Type{FEM1D}) = 1
"    amg_construct(::Type{T},::Type{FEM1D};rest...) where {T} = fem1d(T;rest...)"
amg_construct(::Type{T},::Type{FEM1D};rest...) where {T} = fem1d(T;rest...)
"    fem1d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM1D,rest...)"
fem1d_solve(::Type{T}=Float64;rest...) where {T} = amgb(T;method=FEM1D,rest...)

"""
    fem1d(::Type{T}=Float64; L::Int=4, n=nothing, K=nothing,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}

Construct an `AMG` object for a 1d piecewise linear finite element grid. The interval is [-1,1]. Parameters are:
* `L`: divide the interval into 2^L subintervals (L for Levels).
* `state_variables`: the "state vector" consists of functions, by default this is `u(x)` and `s(x)`, on the finite element grid.
* `D`: the set of differential operators. The barrier function `F` will eventually be called with the parameters `F(x,Dz)`, where `z` is the state vector. By default, this results in `F(x,u,ux,s)`, where `ux` is the derivative of `u`.
* `generate_feasibility`: if `true`, returns a pair `M` of `AMG` objects. `M[1]` is the `AMG` object for the main problem, and `M[2]` is for the feasibility subproblem.
"""
function fem1d(::Type{T}=Float64; L::Int=4, n=nothing, K=nothing,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}
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
    return amg(FEM1D,x=x[L],w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen,
        generate_feasibility=generate_feasibility)
end

"""
    fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::T) where{T}

Interpolate a 1d piecewise linear function at the given `t` value. If `u(xi)` is the piecewise linear function such that `u(x[k])=y[k]` then this function returns `u(t)`.
"""
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

"""
    fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}

Returns `[fem1d_interp(x,y,t[k]) for k=1:length(t)]`.
"""
function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}
    [fem1d_interp(x,y,t[k]) for k=1:length(t)]
end

"    amg_plot(M::AMG{T,Mat,FEM1D}, z::Vector{T}) where {T,Mat} = plot(M.x,z)"
amg_plot(M::AMG{T,Mat,FEM1D}, z::Vector{T}) where {T,Mat} = plot(M.x,z)
