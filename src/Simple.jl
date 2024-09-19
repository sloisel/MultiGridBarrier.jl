export amg_construct, amg_plot, amg_solve, amg_dim, spectral1d_solve, spectral2d_solve, fem1d_solve, fem2d_solve

default_f(T) = [(x)->T[0.5,0.0,1.0],(x)->T[0.5,0.0,0.0,1.0]]
default_g(T) = [(x)->T[x[1],2],(x)->T[x[1]^2+x[2]^2,100.0]]
default_D = [[:u :id 
              :u :dx
              :s :id],
             [:u :id
              :u :dx
              :u :dy
              :s :id]]

"""
    function amg_solve(::Type{T}=Float64; 
        L::Integer=2, n=nothing,
        method=FEM1D,
        K = nothing,
        state_variables::Matrix{Symbol} = [:u :dirichlet
                           :s :full],
        dim::Integer = amg_dim(method),
        D::Matrix{Symbol} = default_D[dim],
        M = amg_construct(T,method,L=L,n=n,K=K,state_variables=state_variables,D=D),
        p::T = T(1.0),
        g::Union{Function,Matrix{T}} = default_g(T)[dim],
        f::Union{Function,Matrix{T}} = default_f(T)[dim],
        Q::Convex{T} = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        show=true,         
        return_details=false, rest...) where {T}

A simplified interface for module MultiGridBarrier to "quickly get started". To solve a p-Laplace problem, do: `amg_solve()`.

Different behaviors can be obtained by supplying various optional keyword arguments, as follows.

* `L=2`: the number of times to subdivide the base mesh.
* The `n` parameter is only used in `spectral` methods, in which case, if `n` is an integer, then `L` is disregarded. `n` is the number of quadrature nodes along each axis.
* `method=FEM1D`: this must be either `FEM1D`, `FEM2D`, `SPECTRAL1D` or `SPECTRAL2D`. This parameter is used twice: once to choose the constructor for the `M` parameter, and again to plot the solution if `show` is `true`. If `show` is `false` and if `M` is constructed "manually", not by its default value, then the `method` parameter is ignored.
* `K`: In most cases, this is `nothing`, but in the `fem2d` case, `K` is the initial mesh.
* `state_variables = [:u :dirichlet ; :s :full]`: the names of the components of the solution vector `z`.
* `dim = size(M[1].x[end],2)`, the dimension of the problem, should be 1 or 2. This is only used in the default values of the `g`, `f`, `Q`, `D` parameters, and is ignored if these parameters do not use default values.
* `D`: the differential operators, see below for defaults.
* `M`: a mesh obtained by one of the constructors `fem1d`, `fem2d`, `spectral1d` or `spectral2d`, corresponding to the `method` parameter.
* `x = M[1].x`: a matrix, same number of rows as `M[1].x`. This matrix will be passed, row by row, to the barrier function, as the x parameter.
* `p = T(1.0)`: the parameter of the p-Laplace operator. This is only relevant if the default value is used for the `Q` parameter, and is ignored otherwise.
* `g`: the "boundary conditions" function. See below for defaults.
* `f`: the "forcing" or "cost functional" to be minimized. See below for defaults.
* `Q`: the convex domain to which the solution should belong. Defaults to `convex_Euclidian_power(T,idx=2:dim+2,p=x->p)`, which corresponds to p-Laplace problems. Change this to solve other variational problems.
* `show=true`: if `true`, plot the solution.
* `return_details=false`: if `false`, return a `Vector{T}` of the solution. If `true`, returned a named tuple with some more details about the solution process.
* `rest...`: any further keyword arguments are passed on to `amgb`.

The default values for the parameters `f`, `g`, `D` are as follows

| `dim` | 1                     | 2                             |
|:------|:----------------------|:------------------------------|
| `f`   | `(x)->T[0.5,0.0,1.0]` | `(x)->T[0.5,0.0,0.0,1.0]`     |
| `g`   | `(x)->T[x[1],2]`      | `(x)->T[x[1]^2+x[2]^2,100.0]` |
| `D`   | `[:u :id`             | `[:u :id`                     |
|       | ` :u :dx`             | ` :u :dx`                     |
|       | ` :s :id]`            | ` :u :dy`                     |
|       |                       | ` :s :id]`                    |
"""
function amg_solve(::Type{T}=Float64; 
        L::Integer=2, n=nothing,
        method=FEM1D,
        K = nothing,
        state_variables::Matrix{Symbol} = [:u :dirichlet
                           :s :full],
        dim::Integer = amg_dim(method),
        D::Matrix{Symbol} = default_D[dim],
        M = amg_construct(T,method,L=L,n=n,K=K,state_variables=state_variables,D=D),
        p::T = T(1.0),
        g::Union{Function,Matrix{T}} = default_g(T)[dim],
        f::Union{Function,Matrix{T}} = default_f(T)[dim],
        Q::Convex{T} = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        show=true,         
        return_details=false, rest...) where {T}
    SOL=amgb(M,f, g, Q,;return_details=return_details,rest...)
    if show
        z = if return_details SOL.z else SOL end
        amg_plot(M[1],z[:,1])
    end
    SOL
end

"    amg_dim(::Type{FEM1D}) = 1"
amg_dim(::Type{FEM1D}) = 1
"    amg_construct(::Type{T},::Type{FEM1D};rest...) where {T} = fem1d(T;rest...)"
amg_construct(::Type{T},::Type{FEM1D};rest...) where {T} = fem1d(T;rest...)
"    fem1d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM1D,rest...)"
fem1d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM1D,rest...)

"    fem2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM2D,rest...)"
fem2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM2D,rest...)
"    amg_dim(::Type{FEM2D}) = 2"
amg_dim(::Type{FEM2D}) = 2
"    amg_construct(::Type{T},::Type{FEM2D};rest...) where {T} = fem2d(T;rest...)"
amg_construct(::Type{T},::Type{FEM2D};rest...) where {T} = fem2d(T;rest...)

"    amg_dim(::Type{SPECTRAL1D}) = 1"
amg_dim(::Type{SPECTRAL1D}) = 1
"    amg_construct(::Type{T},::Type{SPECTRAL1D};rest...) where {T} = spectral1d(T;rest...)"
amg_construct(::Type{T},::Type{SPECTRAL1D};rest...) where {T} = spectral1d(T;rest...)
"    spectral1d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=SPECTRAL1D,rest...)"
spectral1d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=SPECTRAL1D,rest...)

"    spectral2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=SPECTRAL2D,rest...)"
spectral2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=SPECTRAL2D,rest...)
"    amg_dim(::Type{SPECTRAL2D}) = 2"
amg_dim(::Type{SPECTRAL2D}) = 2
"    amg_construct(::Type{T},::Type{SPECTRAL2D},L,n,K) where {T} = spectral2d(T,n=n,L=L)"
amg_construct(::Type{T},::Type{SPECTRAL2D};rest...) where {T} = spectral2d(T;rest...)
