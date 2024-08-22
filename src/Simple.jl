export FEM1D, FEM2D, SPECTRAL1D, SPECTRAL2D, simple_construct, simple_plot, simple_solve

default_f(T) = [(x)->T[0.5,0.0,1.0],(x)->T[0.5,0.0,0.0,1.0]]
default_g(T) = [(x)->T[x[1],2],(x)->T[x[1]^2+x[2]^2,100.0]]
default_D = [[:u :id 
              :u :dx
              :s :id],
             [:u :id
              :u :dx
              :u :dy
              :s :id]]
abstract type FEM1D end
abstract type FEM2D end
abstract type SPECTRAL1D end
abstract type SPECTRAL2D end

"""
    function simple_solve(::Type{T}=Float64; 
        L=2, n=nothing,
        method::Function=FEM1D,
        K = nothing,
        M = simple_construct(T,method,L=L,n=n,K=K),
        p = T(1.0),
        dim = size(M[1].x[end],2),
        g = default_g(T)[dim],
        f = default_f(T)[dim],
        Q = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        show=true, tol=sqrt(eps(T)),
        t=T(0.1), kappa=T(10), maxit=10000,
        state_variables = [:u :dirichlet
                           :s :full],
        D = default_D[dim],
        verbose=true,
        return_details=false) where {T}

A simplified interface for module MultiGridBarrier to "quickly get started". To solve a p-Laplace problem, do: `simple_solve()`.

Different behaviors can be obtained by supplying various optional keyword arguments, as follows.

* `L=2`: the number of times to subdivide the base mesh.
* The `n` parameter is only used in `spectral` methods, in which case, if `n` is an integer, then `L` is disregarded. `n` is the number of quadrature nodes along each axis.
* `method=FEM1D`: this must be either `FEM1D`, `FEM2D`, `SPECTRAL1D` or `SPECTRAL2D`. This parameter is used twice: once to choose the constructor for the `M` parameter, and again to plot the solution if `show` is `true`. If `show` is `false` and if `M` is constructed "manually", not by its default value, then the `method` parameter is ignored.
* `K`: In most cases, this is `nothing`, but in the `fem2d` case, `K` is the initial mesh.
* `M`: a mesh obtained by one of the constructors `fem1d`, `fem2d`, `spectral1d` or `spectral2d`, corresponding to the `method` parameter.
* `p = T(1.0)`: the parameter of the p-Laplace operator. This is only relevant if the default value is used for the `Q` parameter, and is ignored otherwise.
* `dim = size(M[1].x[end],2)`, the dimension of the problem, should be 1 or 2. This is only used in the default values of the `g`, `f`, `Q`, `D` parameters, and is ignored if these parameters do not use default values.
* `g`: the "boundary conditions" function. See below for defaults.
* `f`: the "forcing" or "cost functional" to be minimized. See below for defaults.
* `Q`: the convex domain to which the solution should belong. Defaults to `convex_Euclidian_power(T,idx=2:dim+2,p=x->p)`, which corresponds to p-Laplace problems. Change this to solve other variational problems.
* `show=true`: if `true`, plot the solution.
* `tol=sqrt(eps(T))`: the stopping criterion for `amgb`.
* `t=T(0.1)`: the initial value of the barrier parameter.
* `kappa=T(10)`: the initial growth factor of the barrier parameter.
* `maxit=10000`: can be used to limit the number of t-iterations.
* `state_variables = [:u :dirichlet ; :s :full]`: the names of the components of the solution vector `z`.
* `D`: the differential operators, see below for defaults.
* `verbose=true`: if `true`, use a progress bar.
* `return_details=false`: if `false`, return a `Vector{T}` of the solution. If `true`, returned a named tuple with some more details about the solution process.

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
function simple_solve(::Type{T}=Float64; 
        L=2, n=nothing,
        method=FEM1D,
        K = nothing,
        M = simple_construct(T,method,L,n,K),
        p = T(1.0),
        dim = size(M[1].x[end],2),
        g = default_g(T)[dim],
        f = default_f(T)[dim],
        Q = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        show=true, tol=sqrt(eps(T)),
        t=T(0.1), kappa=T(10), maxit=10000,
        state_variables = [:u :dirichlet
                           :s :full],
        D = default_D[dim],
        verbose=true,
        return_details=false) where {T}
    SOL=amgb(M,f, g, Q,
            tol=tol,t=t,maxit=maxit,kappa=kappa,verbose=verbose,return_details=return_details)
    if show
        z = if return_details SOL.z else SOL end
        simple_plot(method,M[1],z[:,1])
    end
    SOL
end

"    simple_plot(::Type{FEM1D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = plot(M.x[end],z)"
simple_plot(::Type{FEM1D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = plot(M.x[end],z)
"    simple_construct(::Type{T},::Type{FEM1D},L,n,K) where {T} = fem1d(T,L=L)"
simple_construct(::Type{T},::Type{FEM1D},L,n,K) where {T} = fem1d(T,L=L)

"    simple_plot(::Type{FEM2D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = fem2d_plot(M,z)"
simple_plot(::Type{FEM2D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = fem2d_plot(M,z)
"    simple_construct(::Type{T},::Type{FEM2D},L,n,K) where {T} = fem2d(T,L=L,K=K)"
simple_construct(::Type{T},::Type{FEM2D},L,n,K) where {T} = fem2d(T,L=L,K=K)

"    simple_plot(::Type{SPECTRAL1D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = spectral1d_plot(M,Array(-1:T(0.01):1),z)"
simple_plot(::Type{SPECTRAL1D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = spectral1d_plot(M,Array(-1:T(0.01):1),z)
"    simple_construct(::Type{T},::Type{SPECTRAL1D},L,n,K) where {T} = spectral1d(T,n=n,L=L)"
simple_construct(::Type{T},::Type{SPECTRAL1D},L,n,K) where {T} = spectral1d(T,n=n,L=L)

"    simple_plot(::Type{SPECTRAL2D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = spectral2d_plot(M,-1:T(0.01):1,-1:T(0.01):1,z;cmap=:jet)"
simple_plot(::Type{SPECTRAL2D}, M::AMG{T,Mat}, z::Vector{T}) where {T,Mat} = spectral2d_plot(M,-1:T(0.01):1,-1:T(0.01):1,z;cmap=:jet)
"    simple_construct(::Type{T},::Type{SPECTRAL2D},L,n,K) where {T} = spectral2d(T,n=n,L=L)"
simple_construct(::Type{T},::Type{SPECTRAL2D},L,n,K) where {T} = spectral2d(T,n=n,L=L)
