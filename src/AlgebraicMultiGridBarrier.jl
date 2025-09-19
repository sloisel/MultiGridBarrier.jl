export Barrier, AMG, barrier, amgb, amg, newton, illinois, Convex, convex_linear, convex_Euclidian_power, AMGBConvergenceFailure, amgb_core, amg_construct, amg_plot, amgb_solve, amg_dim, apply_D, linesearch_illinois, linesearch_backtracking, stopping_exact, stopping_inexact


function blkdiag(M...)
    Mat = typeof(M[1])
    Mat(blockdiag((sparse.(M))...))
end

macro debug(args...)
    escargs = map(esc, args)
    return :($(esc(:printlog))(nameof($(esc(:(var"#self#")))), ":", $(escargs...)))
end

struct AMGBConvergenceFailure <: Exception
    message
end

Base.showerror(io::IO, e::AMGBConvergenceFailure) = print(io, "AMGBConvergenceFailure:\n", e.message)

"""
    Barrier

A type for holding barrier functions. Fields are:

    f0::Function
    f1::Function
    f2::Function

`f0` is the barrier function itself, while `f1` is its gradient and
`f2` is the Hessian.
"""
@kwdef struct Barrier
    f0::Function
    f1::Function
    f2::Function
end

"""
    @kwdef struct AMG{T,M,Geometry}
        ...
    end

Objects of this type should probably be assembled by the constructor `amg()`.

A multigrid with `L` levels. Denote by `l` between 1 and `L`, a grid level.
Fields are:
* `x::Matrix{T}` the vertices of the fine grid.
* `w::Vector{T}` corresponding quadrature weights.
* `R_fine::Array{M,1}` an array of `L` matrices. The columns of `R_fine[l]` are basis functions for the function space on grid level `l`, interpolated to the fine grid.
* `R_coarse::Array{M,1}` an array of `L` matrices. The columns of `R_coarse[l]` are basis functions for the function space on grid level `l`. Unlike `R_fine[l]`, these basis functions are on grid level `l`, not interpolated to the fine grid.
* `D::Array{M,2}` an array of differential operators. For example, if the barrier parameters are to be `u,ux,s`, with `ux` the derivative of `u`, then `D[l,:] = [I,Dx,I]`, where `Dx` is a numerical differentiation operator on grid level `l`.  
* `refine_u::Array{M,1}` an array of `L` grid refinement matrices. If `x[l]` has `n[l]` vertices, then `refine_u[l]` is `n[l+1]` by `n[l]`.
* `coarsen_u::Array{M,1}` an array of `L` grid coarsening matrices. `coarsen_u[l]` is `n[l]` by `n[l+1]`.
* `refine_z::Array{M,1}` an array of `L` grid refining matrices for the "state vector" `z`. For example, if `z` contains the state functions `u` and `s`, then there are `k=2` state functions, and `refine_z[l]` is `k*n[l+1]` by `k*n[l]`.
* `coarsen_z::Array{M,1}` an array of `L` grid coarsening matrices for the "state vector" `z`. `coarsen_z[l]` is `k*n[l]` by `k*n[l+1]`.

These various matrices must satisfy a wide variety of algebraic relations. For this reason, it is recommended to use the constructor `amg()`.
"""
@kwdef struct AMG{T,M,Geometry}
    x::Matrix{T}
    w::Vector{T}
    R_fine::Array{M,1}
    R_coarse::Array{M,1}
    D::Array{M,2}
    refine_u::Array{M,1}
    coarsen_u::Array{M,1}
    refine_z::Array{M,1}
    coarsen_z::Array{M,1}
end

function amg_helper(::Type{Geometry},
        x::Matrix{T},
        w::Vector{T},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        subspaces::Dict{Symbol,Vector{M}},
        operators::Dict{Symbol,M},
        refine::Vector{M},
        coarsen::Vector{M}) where {T,M,Geometry}
    L = length(refine)
    @assert size(w) == (size(x)[1],) && size(refine)==(L,) && size(coarsen)==(L,)
    for l=1:L
        @assert norm(coarsen[l]*refine[l]-I)<sqrt(eps(T))
    end
    refine_fine = Array{M,1}(undef,(L,))
    refine_fine[L] = refine[L]
    coarsen_fine = Array{M,1}(undef,(L,))
    coarsen_fine[L] = coarsen[L]
    for l=L-1:-1:1
        refine_fine[l] = refine_fine[l+1]*refine[l]
        coarsen_fine[l] = coarsen[l]*coarsen_fine[l+1]
    end
    R_coarse = Array{M,1}(undef,(L,))
    R_fine = Array{M,1}(undef,(L,))
    nu = size(state_variables)[1]
    @assert size(state_variables)[2] == 2
    for l=1:L
        foo = [sparse(subspaces[state_variables[k,2]][l]) for k=1:nu]
        R_coarse[l] = M(blockdiag(foo...))
        foo = [sparse(refine_fine[l]*subspaces[state_variables[k,2]][l]) for k=1:nu]
        R_fine[l] = M(blockdiag(foo...)) 
    end
    nD = size(D)[1]
    @assert size(D)[2]==2
    bar = Dict{Symbol,Int}()
    for k=1:nu
        bar[state_variables[k,1]] = k
    end
    D0 = Array{M,2}(undef,(L,nD))
    for l=1:L
        n = size(coarsen_fine[l],1)
        Z = M(spzeros(T,n,n))
        for k=1:nD
            foo = [Z for j=1:nu]
            foo[bar[D[k,1]]] = coarsen_fine[l]*operators[D[k,2]]*refine_fine[l]
            D0[l,k] = hcat(foo...)
        end
    end
    refine_z = [blkdiag([refine[l] for k=1:nu]...) for l=1:L]
    coarsen_z = [blkdiag([coarsen[l] for k=1:nu]...) for l=1:L]
    AMG{T,M,Geometry}(x=x,w=w,R_fine=R_fine,R_coarse=R_coarse,D=D0,
        refine_u=refine,coarsen_u=coarsen,refine_z=refine_z,coarsen_z=coarsen_z)
end

"""
    function amg(::Type{Geometry};
        x::Matrix{T},
        w::Vector{T},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        subspaces::Dict{Symbol,Vector{M}},
        operators::Dict{Symbol,M},
        refine::Vector{M},
        coarsen::Vector{M},
        full_space=:full,
        id_operator=:id,
        feasibility_slack=:feasibility_slack,
        generate_feasibility=true) where {T,M,Geometry}

Construct an `AMG` object for use with the `amgb` solver. In many cases, this constructor is not called directly by the user. For 1d and 2d finite elements, use the `fem1d()` or `fem2d()`. For 1d and 2d spectral elements, use  `spectral1d()` or `spectral2d()`. You use `amg()` directly if you are implementing your own function spaces.

The `AMG` object shall represent all `L` grid levels of the multigrid hierarchy. Parameters are:
* `x`: the vertices of the fine grid.
* `w`: the quadrature weights for the fine grid.
* `state_variables`: a matrix of symbols. The first column indicates the names of the state vectors or functions, and the second column indicates the names of the corresponding subspaces. A typical example is: `state_variables = [:u :dirichlet; :s :full]`. This would define the solution as being functions named u(x) and s(x). The u function would lie in the space `:dirichlet`, presumably consisting of functions with homogeneous Dirichlet conditions. The s function would lie in the space `:full`, presumably being the full function space, without boundary conditions.
* `D`: a matrix of symbols. The first column indicates the names of various state variables, and the second column indicates the corresponding differentiation operator(s). For example: `D = [:u :id ; :u :dx ; :s :id]`. This would indicate that the barrier should be called as `F(x,y)` with `y = [u,ux,s]`, where `ux` denotes the derivative of `u` with respect to the space variable `x`.
* `subspaces`: a `Dict` mapping each subspace symbol to an array of `L` matrices, e.g. for each `l`, `subspaces[:dirichlet][l]` is a matrix whose columns span the homogeneous Dirichlet subspace of grid level `l`.
* `operators`: a `Dict` mapping each differential operator symbol to a matrix, e.g. `operators[:id]` is an identity matrix, while `operators[:dx]` is a numerical differentiation matrix, on the fine grid level `L`.
* `refine`: an array of length `L` of matrices. For each `l`, `refine[l]` interpolates from grid level `l` to grid level `l+1`. `refine[L]` should be the identity, and `coarsen[l]*refine[l]` should be the identity.
* `coarsen`: an array of length `L` of matrices. For each `l`, `coarsen[l]` interpolates or projects from grid level `l+1` to grid level `l`. `coarsen[L]` should be the identity.
* `generate_feasibility`: if true, `amg()` returns a pair `M` of `AMG` objects. `M[1]` is an `AMG` object to be used for the main optimization problem, while `M[2]` is an `AMG` object for the preliminary feasibility sub problem. In this case, `amg()` also needs to be provided with the following additional information: `feasibility_slack` is the name of a special slack variable that must be unique to the feasibility subproblem (default: `:feasibility_slack`); `full_space` is the name of the "full" vector space (i.e. no boundary conditions, default: `:full`); and `id_operator` is the name of the identity operator (default: `:id`).
"""
function amg(::Type{Geometry};
        x::Matrix{T},
        w::Vector{T},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        subspaces::Dict{Symbol,Vector{M}},
        operators::Dict{Symbol,M},
        refine::Vector{M},
        coarsen::Vector{M},
        full_space=:full,
        id_operator=:id,
        feasibility_slack=:feasibility_slack,
        generate_feasibility=true) where {T,M,Geometry}
    M1 = amg_helper(Geometry,x,w,state_variables,D,subspaces,operators,refine,coarsen)
    if !generate_feasibility
        return M1
    end
    s1 = vcat(state_variables,[feasibility_slack full_space])
    D1 = vcat(D,[feasibility_slack id_operator])
    M2 = amg_helper(Geometry,x,w,s1,D1,subspaces,operators,refine,coarsen)
    return M1,M2
end

@doc raw"""
    struct Convex
        barrier::Function
        cobarrier::Function
        slack::Function
    end

The `Convex` data structure represents a convex domain $Q$ implicitly by way of three functions. The `barrier` function is a barrier for $Q$. `cobarrier` is a barrier for the feasibility subproblem, and `slack` is a function that initializes a valid slack value for the feasibility subproblem. The various `convex_` functions can be used to generate various convex domains.

These functions are called as follows: `barrier(x,y)`. `x` is a vertex in a grid, as per the `AMG` object. `y` is some vector. For each fixed `x` variable, `y -> barrier(x,y)` defines a barrier for a convex set in `y`.
"""
struct Convex{T}
    barrier::Function
    cobarrier::Function
    slack::Function
end

"""
    function convex_linear(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0)) where {T}

Generate a `Convex` structure corresponding to the convex domain A(x,k)*y[idx] .+ b(x,k) ≤ 0.
"""
function convex_linear(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0)) where {T}
    F(x,y) = A(x)*y[idx] .+ b(x)
    barrier_linear(x,y) = -sum(log.(F(x,y)))
    cobarrier_linear(x,yhat) = -sum(log.(F(x,yhat[1:end-1]) .+ yhat[end]))
    slack_linear(x,y) = -minimum(F(x,y))
    return Convex{T}(barrier_linear,cobarrier_linear,slack_linear)
end

normsquared(z) = dot(z,z)

@doc raw"""
    function convex_Euclidian_power(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0),p::Function=x->T(2)) where {T}

Generate a `Convex` object corresponding to the convex set defined by $z[end] \geq \|z[1:end-1]\|_2^p$ where $z = A(x)*y[idx] .+ b(x)$.
"""
function convex_Euclidian_power(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0),p::Function=x->T(2)) where {T}
    F(x,y) = A(x)*y[idx] .+ b(x)
    mu = p->(if (p==2 || p==1) 0 elseif p<2 1 else 2 end)
    function barrier_Euclidian_power(x,y) 
        z = F(x,y)
        p0 = p(x) ::T
        return -log(z[end]^(2/p0)-normsquared(z[1:end-1]))-mu(p0)*log(z[end])
    end
    function cobarrier_Euclidian_power(x,yhat)
        z = F(x,yhat[1:end-1])
        z[end] += yhat[end]
        p0 = p(x) ::T
        return -log(z[end]^(2/p0)-normsquared(z[1:end-1]))-mu(p0)*log(z[end])
    end
    function slack_Euclidian_power(x,y)
        z = F(x,y)
        p0 = p(x) ::T
        return -min(z[end]-normsquared(z[1:end-1])^(p0/2),z[end])
    end
    return Convex{T}(barrier_Euclidian_power,cobarrier_Euclidian_power,slack_Euclidian_power)
end

@doc raw"""
    convex_piecewise(::Type{T}=Float64; Q::Vector{Convex{T}}, select::Function=(tr=fill(true,length(Q));x->tr)) where {T}

Build a `Convex{T}` that combines multiple convex domains with spatial selectivity.

# Arguments
- `Q::Vector{Convex{T}}`: a vector of convex pieces to be combined.
- `select::Function`: a function `x -> Vector{Bool}` indicating which pieces are active at `x`.
  Default: all pieces active everywhere (equivalent to intersection).

# Semantics
For `sel = select(x)`, the resulting convex domain has:
- `barrier(x, y) = ∑(Q[k].barrier(x, y) for k where sel[k])`
- `cobarrier(x, yhat) = ∑(Q[k].cobarrier(x, yhat) for k where sel[k])`  
- `slack(x, y) = max(Q[k].slack(x, y) for k where sel[k])`

The slack is the maximum over active pieces, ensuring a single slack value
suffices for feasibility at each `x`.

# Use cases
1. **Intersections** (default): All pieces active everywhere creates `Q₁ ∩ Q₂ ∩ ...`
2. **Spatial switching**: Different constraints in different regions
3. **Conditional constraints**: Activate constraints based on solution state

# Examples
```julia
# Intersection (using default select)
U = convex_Euclidian_power(Float64; idx=[1, 3], p = x->2)
V = convex_linear(Float64; A = x->A_matrix, b = x->b_vector)
Qint = convex_piecewise(Float64; Q = [U, V])  # U ∩ V everywhere

# Region-dependent constraints
Q_left = convex_Euclidian_power(Float64; p = x->1.5)  
Q_right = convex_Euclidian_power(Float64; p = x->2.0)
select(x) = [x[1] < 0, x[1] >= 0]  # left half vs right half
Qreg = convex_piecewise(Float64; Q = [Q_left, Q_right], select = select)

# Conditional activation
Q_base = convex_linear(Float64; A = x->I, b = x->-ones(2))
Q_extra = convex_Euclidian_power(Float64; p = x->3)
select(x) = [true, norm(x) > 0.5]  # extra constraint outside radius 0.5
Qcond = convex_piecewise(Float64; Q = [Q_base, Q_extra], select = select)
```

See also: [`Base.intersect`](@ref), [`convex_linear`](@ref), [`convex_Euclidian_power`](@ref).
"""
function convex_piecewise(::Type{T}=Float64;Q::Vector{Convex{T}}, select::Function=(tr=fill(true,length(Q));x->tr)) where{T}
    n = length(Q)
    function barrier_piecewise(x,y)
        ret = T(0)
        sel = select(x)
        for k=1:n
            if sel[k]
                ret += Q[k].barrier(x,y)
            end
        end
        return ret
    end
    function cobarrier_piecewise(x,y)
        ret = T(0)
        sel = select(x)
        for k=1:n
            if sel[k]
                ret += Q[k].cobarrier(x,y)
            end
        end
        return ret
    end
    function slack_piecewise(x,y)
        ret = T(0)
        sel = select(x)
        for k=1:n
            if sel[k]
                ret = max(ret,Q[k].slack(x,y))
            end
        end
        return ret
    end
    return Convex{T}(barrier_piecewise,cobarrier_piecewise,slack_piecewise)
end

@doc raw"""
    Base.intersect(U::Convex{T}, rest...) where {T}

Intersection of arbitrarily many convex domains.

Returns a `Convex{T}` that enforces all given domains at each `x`. Internally this
is implemented via `convex_piecewise` with `select(x) = [true, true, ...]`, so that:
- `barrier(x, y) = U.barrier(x, y) + ∑ rest[k].barrier(x, y)`
- `cobarrier(x, yhat) = U.cobarrier(x, yhat) + ∑ rest[k].cobarrier(x, yhat)`
- `slack(x, y) = max(U.slack(x, y), max(rest[k].slack(x, y) for k))`

This lets you compose constraints in a natural way: the resulting domain equals
`U ∩ V₁ ∩ V₂ ∩ ...`.

# Examples
```julia
# Intersect two domains
U = convex_Euclidian_power(Float64; idx=[1, 2+dim], p = x->2)
V = convex_Euclidian_power(Float64; idx=vcat(2:1+dim, 3+dim), p = x->p)
Q = U ∩ V  # same as intersect(U, V)

# Intersect three or more domains
W = convex_linear(Float64; A = x->A_matrix, b = x->b_vector)
Q3 = U ∩ V ∩ W  # same as intersect(U, V, W)

# Works with single domain (returns it unchanged)
Q1 = intersect(U)  # effectively returns U
```

See also: [`convex_piecewise`](@ref).
"""
Base.intersect(U::Convex{T}, rest...) where {T} = convex_piecewise(T;Q=[U,rest...])

@doc raw"""    apply_D(D,z::Vector{T}) where {T} = hcat([D[k]*z for k in 1:length(D)]...)"""
apply_D(D,z::Vector{T}) where {T} = hcat([D[k]*z for k in 1:length(D)]...)

@doc raw"""
    function barrier(F;
        F1=(x,y)->ForwardDiff.gradient(z->F(x,z),y),
        F2=(x,y)->ForwardDiff.hessian(z->F(x,z),y))

Constructor for barriers.

* `F` is the actual barrier function. It should take parameters `(x,y)`.
* `F1` is the gradient of `F` with respect to `y`.
* `F2` is the Hessian of `F` with  respect to `y`.

By default, `F1` and `F2` are automatically generated by the module `ForwardDiff`.

A more specific description of the Barrier object is as follows. The function `Barrier.f0` has parameters:

    function Barrier.f0(z,x,w,c,R,D,z0)

Here, `R` is a matrix and `D` is an array of matrices; `x` is a matrix of quadrature nodes with weights `w`, and `c` is a matrix describing the functional we seek to minimize. The value of `Barrier.f0` is given by:
```
        p = length(w)
        n = length(D)
        Rz = z0+R*z
        Dz = hcat([D[k]*Rz for k=1:n]...)
        y = [F(x[k,:],Dz[k,:]) for k=1:p]
        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])
```
Thus, `Barrier.f0` can be regarded as a quadrature approximation of the integral
```math
\int_{\Omega} \left(\sum_{k=1}^nc_k(x)v_k(x)\right) + F(x,v_1(x),\ldots,v_n(x)) \, dx \text{ where } v_k = D_k(z_0 + Rz).
```

Functions `Barrier.f1` and `Barrier.f2` are the gradient and Hessian, respectively, of `Barrier.f0`, with respect to the `z` parameter. If the underlying matrices are sparse, then sparse arithmetic is used for `Barrier.f2`.
"""
function barrier(F;
        F1=(x,y)->ForwardDiff.gradient(z->F(x,z),y),
        F2=(x,y)->ForwardDiff.hessian(z->F(x,z),y))::Barrier
    function f0(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = [F(x[k,:],Dz[k,:]) for k=1:p]
        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])
    end
    function f1(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = Array{T,2}(undef,(p,n))
        for k=1:p
            y[k,:] = F1(x[k,:],Dz[k,:])
        end
        y += c
        m0 = size(D[1],2)
        ret = zeros(T,(m0,))
        for k=1:n
            ret += D[k]'*(w.*y[:,k])
        end
        R'*ret
    end
    function f2(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = Array{T,3}(undef,(p,n,n))
        for k=1:p
            y[k,:,:] = F2(x[k,:],Dz[k,:])
        end
        m0 = size(D[1],2)
        ret = spzeros(T,m0,m0)
        for j=1:n
            foo = spdiagm(0=>w.*y[:,j,j])
            ret += (D[j])'*foo*D[j]
            for k=1:j-1
                foo = spdiagm(0=>w.*y[:,j,k])
                ret += D[j]'*foo*D[k] + D[k]'*foo*D[j]
            end
        end
        R'*ret*R
    end
    Barrier(;f0,f1,f2)
end
function divide_and_conquer(eta,j,J)
    if eta(j,J) return true end
    jmid = (j+J)÷2
    if jmid==j || jmid==J return false end
    return divide_and_conquer(eta,j,jmid) && divide_and_conquer(eta,jmid,J)
end
function amgb_phase1(B::Barrier,
        M::AMG{T,Mat,Geometry},
        x::Matrix{T},
        z::Vector{T},
        c::Matrix{T};
        maxit,
        max_newton,
        stopping_criterion,
        line_search,
        printlog,
        args...
        ) where {T,Mat,Geometry}
    @debug("start")
    L = length(M.R_fine)
    cm = Vector{Matrix{T}}(undef,L)
    cm[L] = c
    zm = Vector{Vector{T}}(undef,L)
    zm[L] = z
    xm = Vector{Matrix{T}}(undef,L)
    xm[L] = x
    wm = Vector{Vector{T}}(undef,L)
    wm[L] = M.w
    passed = falses((L,))
    for l=L-1:-1:1
        cm[l] = M.coarsen_u[l]*cm[l+1]
        xm[l] = M.coarsen_u[l]*xm[l+1]
        zm[l] = M.coarsen_z[l]*zm[l+1]
        wm[l] = M.refine_u[l]'*wm[l+1]
    end
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    function zeta(j,J)
        @debug("j=",j," J=",J)
        x = xm[J]
        w = wm[J]
        R = M.R_coarse[J]
        D = M.D[J,:]
        z0 = zm[J]
        c0 = cm[J]
        s0 = zeros(T,(size(R)[2],))
        mi = if J-j==1 maxit else max_newton end
        SOL = newton(Mat,
                s->f0(s,x,w,c0,R,D,z0),
                s->f1(s,x,w,c0,R,D,z0),
                s->f2(s,x,w,c0,R,D,z0),
                s0,
                maxit=mi,
                stopping_criterion=stopping_criterion,
                ;line_search,printlog)
        if !SOL.converged
            if J-j>1 return false end
            it = SOL.k
            throw(AMGBConvergenceFailure("Damped Newton iteration failed to converge at level $J during phase 1 ($it iterations, maxit=$maxit)."))
        end
        znext = copy(zm)
        s = R*SOL.x
        znext[J] = zm[J]+s
        try
            for k=J+1:L
                s = M.refine_z[k-1]*s
                znext[k] = zm[k]+s
                s0 = zeros(T,(size(M.R_coarse[k])[2],))
                y0 = f0(s0,xm[k],wm[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])::T
                y1 = f1(s0,xm[k],wm[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])
                @assert isfinite(y0) && all(isfinite.(y1))
            end
            zm = znext
            passed[J] = true
        catch
        end
        return true
    end
    if !divide_and_conquer(zeta,0,L) || !passed[end]
            throw(AMGBConvergenceFailure("Phase 1 failed to converge."))
    end
    (;z=zm[L],its,passed)
end
function amgb_step(B::Barrier,
        M::AMG{T,Mat,Geometry},
        x::Matrix{T},
        z::Vector{T},
        c::Matrix{T};
        early_stop,
        maxit,
        max_newton,
        line_search,
        stopping_criterion,
        finalize,
        printlog,
        args...
        ) where {T,Mat,Geometry}
    L = length(M.R_fine)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D[L,:]
    function eta(j,J,sc,maxit,ls)
        @debug("j=",j," J=",J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = zeros(T,(size(R)[2],))
        SOL = newton(Mat,
            s->f0(s,x,w,c,R,D,z),
            s->f1(s,x,w,c,R,D,z),
            s->f2(s,x,w,c,R,D,z),
            s0,
            ;maxit,
            stopping_criterion=sc,
            line_search=ls,
            printlog)
        its[J] += SOL.k
        if SOL.converged
            z = z + R*SOL.x
        end
        return SOL.converged
    end
    converged = divide_and_conquer((j,J)->eta(j,J,stopping_criterion,max_newton,line_search),0,L)
    z_unfinalized = z
    if finalize!=false
        @debug("finalize")
        foo = eta(L-1,L,finalize,maxit,line_search)
        converged = converged && foo
    end
    @debug("converged=",converged)
    return (;z,z_unfinalized,its,converged)
end

"""
    illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}

Find a root of `f` between `a` and `b` using the Illinois algorithm. If `f(a)*f(b)>=0`, returns `b`.
"""
function illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}
    @assert isfinite(fa) && isfinite(fb)
    if fa==0
        return a
    end
    if fa*fb>=0
        return b
    end
    for k=1:maxit
        c = (a*fb-b*fa)/(fb-fa)
        fc = f(c)
        @assert isfinite(fc)
        if c<=min(a,b) || c>=max(a,b) || fc*fa==0 || fc*fb==0
            return c
        end
        if fb*fc<0
            a,fa = b,fb
        else
            fa /= 2
        end
        b,fb = c,fc
    end
    throw("Illinois solver failed to converge.")
end

raw"""
    linesearch_illinois(::Type{T}=Float64; beta=T(0.5)) where {T}

Create an Illinois-based line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction when Illinois fails (default: 0.5).

# Returns
A line search function `ls(x, y, g, n, F0, F1; log)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : Newton direction (typically H\g where H is the Hessian).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
The Illinois algorithm finds a root of `φ(s) = ⟨∇F(x - s*n), n⟩`, which corresponds to
the exact line search condition. If the Illinois solver fails or encounters numerical
issues, the step size is reduced by factor `beta` and the process repeats.

# Notes
This line search strategy aims for the exact minimizer along the search direction,
making it potentially more aggressive than backtracking but also more expensive per iteration.
"""
function linesearch_illinois(::Type{T}=Float64;beta=T(0.5)) where {T}
    function ls_illinois(x::Vector{T},y::T,g::Vector{T},
        n::Vector{T},F0,F1;printlog)
        s = T(1)
        test_s = true
        xnext = x
        ynext = y
        gnext = g
        inc = dot(g,n)
        while s>T(0) && test_s
            @debug("s=",s)
            try
                function phi(s)
                    xn = x-s*n
                    @assert(isfinite(F0(xn)))
                    return dot(F1(xn),n)
                end
                s = illinois(phi,T(0),s,fa=inc)
                xnext = x-s*n
                test_s = any(xnext != x)
                ynext,gnext = F0(xnext)::T,F1(xnext)
                @assert isfinite(ynext) && all(isfinite.(gnext))
                break
            catch e
                @debug(e.msg)
            end
            s = s*beta
        end
        return (xnext,ynext,gnext)
    end
    return ls_illinois
end

raw"""
    linesearch_backtracking(::Type{T}=Float64; beta=T(0.5)) where {T}

Create a backtracking line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction (default: 0.5).

# Returns
A line search function `ls(x, y, g, n, F0, F1; log)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : search direction (typically Newton direction H\g).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
Implements the Armijo backtracking line search with sufficient decrease condition:
`F(x - s*n) ≤ F(x) - c₁ * s * ⟨∇F(x), n⟩` where `c₁ = 0.1`.
The step size starts at `s = 1` and is reduced by factor `beta` until the condition
is satisfied or numerical limits are reached.

# Notes
This is a robust and commonly used line search that guarantees sufficient decrease
in the objective function, making it suitable for general nonlinear optimization.
"""
function linesearch_backtracking(::Type{T}=Float64;beta = T(0.5)) where {T}
    function ls_backtracking(x::Vector{T},y::T,g::Vector{T},
        n::Vector{T},F0,F1;printlog)
        s = T(1)
        test_s = true
        xnext = x
        ynext = y
        gnext = g
        inc = dot(g,n)
        while s>T(0) && test_s
            @debug("s=",s)
            try
                xnext = x-s*n
                test_s = any(xnext != x)
                ynext,gnext = F0(xnext)::T,F1(xnext)
                @assert isfinite(ynext) && all(isfinite.(gnext))
                if ynext<=y-0.1*inc*s
                    break
                end
            catch e
                @debug(e.msg)
            end
            s = s*beta
        end
        return (xnext,ynext,gnext)
    end
    return ls_backtracking
end

"""
    stopping_exact(theta::T) where {T}

Create an exact stopping criterion for Newton methods.

# Arguments
* `theta` : tolerance parameter for gradient norm relative decrease (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement.

# Algorithm
Returns `true` (stop) if both conditions hold:
1. No objective improvement: `ynext ≥ ymin`
2. Gradient norm stagnation: `‖gnext‖ ≥ theta * gmin`

# Notes
This criterion is "exact" in the sense that it requires both objective and gradient
stagnation before stopping, making it suitable for high-precision optimization.
Typical values of `theta` are in the range [0.1, 0.9].
"""
stopping_exact(theta::T) where {T} = (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->ynext>=ymin && norm(gnext)>=theta*gmin
"""
    stopping_inexact(lambda_tol::T, theta::T) where {T}

Create an inexact stopping criterion for Newton methods that combines Newton decrement
and exact stopping conditions.

# Arguments
* `lambda_tol` : tolerance for the Newton decrement (type T).
* `theta` : tolerance parameter for the exact stopping criterion (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement (√(gᵀH⁻¹g)).

# Algorithm
Returns `true` (stop) if either condition holds:
1. Newton decrement condition: `ndec < lambda_tol`
2. Exact stopping condition: `stopping_exact(theta)` is satisfied

# Notes
This criterion is "inexact" because it allows early termination based on the Newton
decrement, which provides a quadratic convergence estimate. The Newton decrement
`λ = √(gᵀH⁻¹g)` approximates the distance to the optimum in the Newton metric.
Typical values: `lambda_tol ∈ [1e-6, 1e-3]`, `theta ∈ [0.1, 0.9]`.
"""
function stopping_inexact(lambda_tol::T,theta::T) where {T} 
    exact_stop = stopping_exact(theta)
    (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->((ndec<lambda_tol || exact_stop(ymin,ynext,gmin,gnext,n,ndecmin,ndec)))
end

"""
    newton(::Type{Mat},
           F0::Function,
           F1::Function,
           F2::Function,
           x::Array{T,1};
           maxit=10000,
           stopping_criterion=stopping_exact(T(0.1)),
           line_search=linesearch_illinois(T),
           printlog) where {T,Mat}

Damped Newton iteration for unconstrained minimization of a differentiable function.

# Arguments
* `F0` : objective function.
* `F1` : gradient of `F0`.
* `F2` : Hessian of `F0` (must return a matrix of type `Mat`).
* `x`  : starting point (vector of type `T`).

# Keyword arguments
* `maxit` : maximum number of iterations (default: 10,000).
* `stopping_criterion` : user-defined predicate deciding when to stop.
  The default is `stopping_exact(T(0.1))` which checks whether the objective decreased and the gradient norm fell sufficiently.
* `line_search` : line search strategy (default: `linesearch_illinois`). The alternative is `linesearch_backtracking`.
* `printlog` : logging function.

# Notes
The iteration stops if the `stopping_criterion` is satisfied or if `maxit` iterations are exceeded.
"""
function newton(::Type{Mat},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::Array{T,1};
                       maxit=10000,
                       stopping_criterion=stopping_exact(T(0.1)),
                       printlog,
                       line_search=linesearch_illinois(T),
        ) where {T,Mat}
    ss = T[]
    ys = T[]
    @assert all(isfinite.(x))
    y = F0(x) ::T
    @assert isfinite(y)
    ymin = y
    push!(ys,y)
    converged = false
    k = 0
    g = F1(x) ::Array{T,1}
    @assert all(isfinite.(g))
    ynext,xnext,gnext=y,x,g
    gmin = norm(g)
    incmin = T(Inf)
    while k<maxit && !converged
        k+=1
        H = F2(x) ::Mat
        n = (H\g)::Array{T,1}
        @assert all(isfinite.(n))
        inc = dot(g,n)
        @debug("k=",k," y=",y," ‖g‖=",norm(g), " λ^2=",inc)
        if inc<=0
            converged = true
            break
        end
        (xnext,ynext,gnext) = line_search(x,y,g,n,F0,F1;printlog)
        if stopping_criterion(ymin,ynext,gmin,gnext,n,sqrt(incmin),sqrt(inc)) #ynext>=ymin && norm(gnext)>=theta*norm(g)
            @debug("converged: ymin=",ymin," ynext=",ynext," ‖gnext‖=",norm(gnext)," λ=",sqrt(inc)," λmin=",sqrt(incmin))
            converged = true
        end
        x,y,g = xnext,ynext,gnext
        gmin = min(gmin,norm(g))
        ymin = min(ymin,y)
        incmin = min(inc,incmin)
        push!(ys,y)
    end
    if !converged
        @debug("diverge")
    end
    return (;x,y,k,converged,ys)
end

"""
    amgb_core(B::Barrier,
              M::AMG{T,Mat,Geometry},
              x::Matrix{T},
              z::Array{T,1},
              c::Array{T,2};
              tol,
              t,
              maxit,
              kappa,
              early_stop,
              progress,
              c0,
              max_newton,
              finalize,
              printlog,
              args...) where {T,Mat,Geometry}

Algebraic MultiGrid Barrier (AMGB) method.

# Arguments
* `B` : a `Barrier` object.
* `M` : an `AMG` hierarchy.
* `x` : a matrix with the same number of rows as `M.x`. Typically `x = M.x`.
* `z` : initial iterate, which must be admissible (`B.f0(z) < ∞`).
* `c` : objective functional to minimize. Concretely, the method minimizes the integral of `c .* (D*z)` (with `D` the differential operator in `M`), subject to barrier feasibility.

# Keyword arguments
* `tol` : stopping tolerance; the method stops once `1/t < tol`.
* `t` : initial barrier parameter.
* `maxit` : maximum number of barrier iterations.
* `kappa` : initial step size multiplier for `t`. Adapted dynamically but never exceeds this initial value.
* `early_stop` : function `z -> Bool`; if `true`, the iteration halts early (e.g. in feasibility mode).
* `progress` : callback receiving a scalar in `[0,1]` for reporting progress (default: no-op).
* `c0` : base offset added to the objective (`c0 + t*c`).
* `max_newton` : maximum Newton iterations per inner solve (default depends on problem data).
* `finalize` : finalization stopping criterion for the last Newton solve.
* `printlog` : logging function.
* `args...` : extra keyword arguments passed to inner routines (`amgb_phase1`, `amgb_step`).

# Returns
A named tuple `SOL` with fields:
* `z` : the computed solution.
* `z_unfinalized`: the solution before finalization.
* `c` : the input functional.
* `its` : iteration counts across levels and barrier steps.
* `ts` : sequence of barrier parameters `t`.
* `kappas` : step size multipliers used.
* `times` : wall-clock timestamps of iterations.
* `M` : the AMG hierarchy.
* `t_begin`, `t_end`, `t_elapsed` : timing information.
* `passed` : whether phase 1 succeeded.
* `c_dot_Dz` : recorded values of ⟨c, D*z⟩ at each iteration.

Throws `AMGBConvergenceFailure` if convergence is not achieved.
"""
function amgb_core(B::Barrier,
        M::AMG{T,Mat,Geometry},
        x::Matrix{T},
        z::Array{T,1},
        c::Array{T,2};
        tol=sqrt(eps(T)),
        t=T(0.1),
        maxit=10000,
        kappa=T(10.0),
        early_stop=z->false,
        progress=x->nothing,
        c0=T(0),
        max_newton= Int(ceil((log2(-log2(eps(T))))+2)),
        printlog,
        finalize,
        args...) where {T,Mat,Geometry}
    t_begin = time()
    tinit = t
    kappa0 = kappa
    L = length(M.R_fine)
    its = zeros(Int,(L,maxit))
    ts = zeros(T,(maxit,))
    kappas = zeros(T,(maxit,))
    times = zeros(Float64,(maxit,))
    c_dot_Dz = zeros(T,(maxit,))
    k = 1
    times[k] = time()
    SOL = amgb_phase1(B,M,x,z,c0 .+ t*c;maxit,max_newton,printlog,args...)
    @debug("phase 1 success")
    passed = SOL.passed
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    z_unfinalized = z
    c_dot_Dz[k] = dot(repeat(M.w,1,size(c,2)).*c,apply_D(M.D[end,:],z))
#    mi = Int(ceil(log2(-log2(eps(T)))))+2
    while t<=1/tol && kappa > 1 && k<maxit && !early_stop(z)
        k = k+1
        its[:,k] .= 0
        times[k] = time()
        prog = ((log(t)-log(tinit))/(log(1/tol)-log(tinit)))
        progress(prog)
        while kappa > 1
            t1 = kappa*t
            @debug("k=",k," t=",t," kappa=",kappa," t1=",t1)
            fin = (t1>1/tol) ? finalize : false
            SOL = amgb_step(B,M,x,z,c0 .+ t1*c;
                max_newton,early_stop,maxit,printlog,finalize=fin,args...)
            its[:,k] += SOL.its
            if SOL.converged
                if maximum(SOL.its)<=max_newton*0.5
                    @debug("increasing t step size?")
                    kappa = min(kappa0,kappa^2)
                end
                z = SOL.z
                z_unfinalized = SOL.z_unfinalized
                t = t1
                break
            end
            @debug("t refinement failed, shrinking kappa")
            kappa = sqrt(kappa)
        end
        ts[k] = t
        kappas[k] = kappa
        c_dot_Dz[k] = dot(repeat(M.w,1,size(c,2)).*c,apply_D(M.D[end,:],z))
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        throw(AMGBConvergenceFailure("Convergence failure in amgb at t=$t, k=$k, kappa=$kappa."))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    @debug("success. t=",t," tol=",tol)
    return (;z,z_unfinalized,c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],M,
            t_begin,t_end,t_elapsed,times=times[1:k],
            passed,c_dot_Dz=c_dot_Dz[1:k])
end

"""
    amgb(M::Tuple{AMG{T,Mat,Geometry}, AMG{T,Mat,Geometry}},
         f::Union{Function, Matrix{T}},
         g::Union{Function, Matrix{T}},
         Q::Convex;
         x::Matrix{T}=M[1].x,
         t=T(0.1),
         t_feasibility=t,
         verbose=true,
         return_details=false,
         stopping_criterion,
         line_search,
         finalize,
         logfile=devnull,
         rest...) where {T,Mat,Geometry}

Algebraic MultiGrid Barrier (AMGB) driver.

High-level wrapper around [`amgb_core`](@ref) that:
1. Builds the initial guess `z0` and cost functional `c0` from `g` and `f`.
2. Solves a feasibility subproblem on `M[2]` if needed.
3. Solves the main optimization problem on `M[1]`.
4. Optionally reports progress and logs diagnostics.

# Arguments
* `M`: a tuple `(M_main, M_feas)` of `AMG` hierarchies.
  * `M[1]` encodes the main problem.
  * `M[2]` encodes the feasibility subproblem.
* `f`: objective functional to minimize. May be a function (evaluated at rows of `x`)
  or a precomputed matrix.
* `g`: boundary/initial data. May be a function (evaluated at rows of `x`)
  or a precomputed matrix.
* `Q`: convex domain describing admissible states.

# Keyword arguments
* `x`: mesh/sample points where `f` and `g` are evaluated when they are functions
  (default: `M[1].x`).
* `t`: initial barrier parameter for the main solve.
* `t_feasibility`: initial barrier parameter for the feasibility solve.
* `verbose`: show a progress bar if `true`.
* `return_details`: if `true`, return detailed results from both solves.
* `stopping_criterion`: stopping criterion for the Newton solver (has a default based on mesh parameters).
* `line_search`: line search strategy (default: `linesearch_backtracking`).
* `finalize`: finalization stopping criterion (default: `stopping_exact(T(0.1))`).
* `logfile`: IO stream for logging (default: `devnull`).
* `rest...`: additional keyword arguments forwarded to [`amgb_core`](@ref) (e.g.,
  tolerances, other options).

# Initialization
If `f`/`g` are functions, `c0` and `z0` are built by evaluating on each row of `x`:
```julia
m = size(M[1].x, 1)
for k in 1:m
    z0[k, :] .= g(x[k, :])
    c0[k, :] .= f(x[k, :])
end
```

If `f/g` are matrices, they are used directly (their shapes must match the
discretization implied by `M[1]`).

# Feasibility handling

The routine checks barrier admissibility. If any point is infeasible under `Q`,
a feasibility problem is automatically constructed and solved on `M[2]`
(using an internal slack augmentation and an early-stop criterion). If feasibility
is already satisfied, this step is skipped.

# Returns
If `return_details == false` (default):
* returns `z`, an `m × n` matrix, where `m = size(x,1)` and `n` is the number of
  state variables in the discretization.

If `return_details == true`:
* returns a named tuple `(z, SOL_feasibility, SOL_main,log)`
  where `SOL_feasibility` is nothing if no feasibility step was needed. The `SOL_*`
  objects are the detailed results returned by `amgb_core`. `log` is a string log of
  the iterations, useful for debugging purposes.

# Errors
Throws AMGBConvergenceFailure if either the feasibility or main solve fails to
converge.
"""
function amgb(M::Tuple{AMG{T,Mat,Geometry},AMG{T,Mat,Geometry}},
              f::Matrix{T},
              g::Matrix{T}, 
              Q::Convex;
              x::Matrix{T} = M[1].x,
              t=T(0.1),
              t_feasibility=t,
              progress = x->nothing,
              return_details=false,
              stopping_criterion=stopping_inexact(sqrt(minimum(M[1].w))/2,T(0.5)),
              printlog = (args...)->nothing,
              line_search=linesearch_backtracking(T),
              finalize=stopping_exact(T(0.5)),
              rest...) where {T,Mat,Geometry}
    D0 = M[1].D[end,1]
    m = size(M[1].x,1)
    ns = Int(size(D0,2)/m)
    nD = size(M[1].D,2)
    c0 = f
    z0 = g
    z2 = reshape(z0,(:,))
    w = hcat([M[1].D[end,k]*z2 for k=1:nD]...)
    pbarfeas = 0.0
    SOL_feasibility=nothing
    try
        for k=1:m
            @assert(isfinite(Q.barrier(x[k,:],w[k,:])::T))
        end
    catch
        pbarfeas = 0.1
        z1 = hcat(z0,[2*max(Q.slack(x[k,:],w[k,:]),1) for k=1:m])
        b = 2*max(1,maximum(z1[:,end]))
        c1 = zeros(T,(m,nD+1))
        c1[:,end] .= 1
        B1 = barrier((x,y)->dot(y,y)+Q.cobarrier(x,y)-log(b^2-y[end]^2))
        z1 = reshape(z1,(:,))
        early_stop(z) = all(z[end-m+1:end] .< 0)
        try
            SOL_feasibility = amgb_core(B1,M[2],x,z1,c1;t=t_feasibility,
                progress=x->progress(pbarfeas*x),
                early_stop,
                printlog,
                stopping_criterion,
                line_search,
                finalize,
                rest...)
            @assert early_stop(SOL_feasibility.z)
        catch e
            if isa(e,AMGBConvergenceFailure)
                throw(AMGBConvergenceFailure("Could not solve the feasibility subproblem, probem may be infeasible. Failure was: "*e.message))
            end
            throw(e)
        end
        z2 = reshape((reshape(SOL_feasibility.z,(m,ns+1)))[:,1:end-1],(:,))
    end
    B = barrier(Q.barrier)
    SOL_main = amgb_core(B,M[1],x,z2,c0;
        t,
        progress=x->progress((1-pbarfeas)*x+pbarfeas),
        printlog,
        stopping_criterion,
        line_search,
        finalize,
        rest...)
    z = reshape(SOL_main.z,(m,:))
    return (;z,SOL_feasibility,SOL_main)
end

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
    amgb_solve([::Type{T}=Float64]; kwargs...) where {T}

Convenience interface to the MultiGridBarrier module for solving nonlinear convex optimization problems 
in function spaces using multigrid barrier methods.

This function builds a discretization and calls `amgb` with appropriate defaults. It is the main high-level 
entry point for solving p-Laplace and related problems.

# Arguments

- `T::Type = Float64`: Numeric type for computations

# Keyword Arguments

## Discretization Parameters
- `L::Integer = 2`: Number of mesh refinement levels (grid has 2^L subdivisions)
- `n::Union{Nothing,Integer} = nothing`: Number of quadrature nodes per axis (spectral methods only; overrides `L`)
- `method = FEM1D`: Discretization method. Options: `FEM1D`, `FEM2D`, `SPECTRAL1D`, `SPECTRAL2D`
- `K::Union{Nothing,Matrix} = nothing`: Initial triangular mesh for `FEM2D` (3n×2 matrix for n triangles)

## Problem Specification
- `state_variables::Matrix{Symbol} = [:u :dirichlet; :s :full]`: Solution components and their function spaces
- `D::Matrix{Symbol} = default_D[dim]`: Differential operators to apply to state variables
- `p::T = 1.0`: Exponent for p-Laplace operator (p ≥ 1)
- `g::Union{Function,Matrix{T}}`: Boundary conditions/initial guess (function of spatial coordinates or matrix)
- `f::Union{Function,Matrix{T}}`: Forcing term/cost functional (function of spatial coordinates or matrix)
- `Q::Convex{T} = convex_Euclidian_power(...)`: Convex constraint set for the variational problem

## Solver Control
- `M`: Pre-built AMG hierarchy (constructed automatically if not provided)
- `verbose::Bool = true`: Display progress bar during solving
- `logfile = devnull`: IO stream for logging (default: no file logging)

## Output Control
- `show::Bool = true`: Plot the computed solution using PyPlot (requires PyPlot.jl)
- `return_details::Bool = false`: 
  - `false`: Return only the solution matrix `z`
  - `true`: Return full solution object with fields `z`, `SOL_feasibility`, `SOL_main`, and `log`

## Additional Parameters
- `dim::Integer = amg_dim(method)`: Problem dimension (1 or 2), auto-detected from method
- `rest...`: Additional keyword arguments passed to `amgb` (e.g., `tol`, `maxiter`)

# Default Values

The defaults for `f`, `g`, and `D` depend on the problem dimension:

## 1D Problems
- `f(x) = [0.5, 0.0, 1.0]` - Forcing term
- `g(x) = [x[1], 2]` - Boundary conditions
- `D = [:u :id; :u :dx; :s :id]` - Identity, derivative, identity

## 2D Problems  
- `f(x) = [0.5, 0.0, 0.0, 1.0]` - Forcing term
- `g(x) = [x[1]²+x[2]², 100.0]` - Boundary conditions
- `D = [:u :id; :u :dx; :u :dy; :s :id]` - Identity, x-derivative, y-derivative, identity

# Returns

- If `return_details=false` (default): Matrix of size `(n_nodes, n_components)` containing the solution
- If `return_details=true`: NamedTuple with fields:
  - `z`: Solution matrix
  - `SOL_feasibility`: Feasibility phase solution (if applicable)
  - `SOL_main`: Main phase solution details
  - `log`: String containing solver log

# See Also
- [`fem1d_solve`](@ref), [`fem2d_solve`](@ref), [`spectral1d_solve`](@ref), [`spectral2d_solve`](@ref): 
  Convenience wrappers for specific discretizations
- [`amgb`](@ref): Lower-level solver function
- [`amg`](@ref): AMG hierarchy construction
"""
function amgb_solve(::Type{T}=Float64;
        L::Integer=2, n=nothing,
        method=FEM1D,
        K = nothing,
        state_variables::Matrix{Symbol} = [:u :dirichlet
                           :s :full],
        dim::Integer = amg_dim(method),
        D::Matrix{Symbol} = default_D[dim],
        M = amg_construct(T,method;L,n,K,state_variables,D),
        x = M[1].x,
        p::T = T(1.0),
        g::Union{Function, Matrix{T}} = default_g(T)[dim],
        f::Union{Function, Matrix{T}} = default_f(T)[dim],
        g_grid::Matrix{T} = vcat([g(x[k,:])' for k in 1:size(x,1)]...),
        f_grid::Matrix{T} = vcat([f(x[k,:])' for k in 1:size(x,1)]...),
        Q::Convex{T} = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        show=true,
        verbose=true,
        return_details=false, 
        logfile=devnull,
        rest...) where {T}
    progress = x->nothing
    pbar = 0
    if verbose
        pbar = Progress(1000000; dt=1.0)
        finished = false
        function _progress(x)
            if !finished
                fooz = Int(floor(1000000*x))
                update!(pbar,fooz)
                if fooz==1000000
                    finished = true
                end
            end
        end
        progress = _progress
    end
    log_buffer = IOBuffer()
    function printlog(args...)
        println(log_buffer,args...)
        println(logfile,args...)
    end
    SOL=amgb(M,f_grid, g_grid, Q;x,progress,printlog,rest...)
    if show
        amg_plot(M[1],SOL.z[:,1])
    end
    if return_details
        return (;SOL...,log=String(take!(log_buffer)))
    end
    return SOL.z
end

function amg_precompile()
    fem1d_solve(L=1)
    fem1d_solve(L=1;line_search=linesearch_illinois(Float64))
    fem1d_solve(L=1;line_search=linesearch_illinois(Float64),stopping_criterion=stopping_exact(0.1),finalize=false)
    fem2d_solve(L=1)
    spectral1d_solve(L=2)
    spectral2d_solve(L=2)
end

precompile(amg_precompile,())
