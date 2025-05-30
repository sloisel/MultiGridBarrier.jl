export Barrier, AMG, barrier, amgb, amg, newton, illinois, Convex, convex_linear, convex_Euclidian_power, AMGBConvergenceFailure, amgb_core, amg_construct, amg_plot, amg_solve, amg_dim

function blkdiag(M...)
    Mat = typeof(M[1])
    Mat(blockdiag((sparse.(M))...))
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

A multigrid with `L` level. Denote by `l` between 1 and `L`, a grid level.
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

These function are called as follows: `barrier(x,y)`. `x` is a vertex in a grid, as per the `AMG` object. `y` is some vector. For each fixed `x` variable, `y -> barrier(x,y)` defines a barrier for a convex set in `y`.
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

function convex_piecewise(::Type{T}=Float64;select::Function,Q::Vector{Convex{T}}) where{T}
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

Base.intersect(U::Convex{T}, V::Convex{T}) where {T} = convex_piecewise(T;select=x->[true,true],Q=[U,V])

@doc raw"""
    function barrier(F;
        F1=(x,y)->ForwardDiff.gradient(z->F(x,z),y),
        F2=(x,y)->ForwardDiff.hessian(z->F(x,z),y))::Barrier

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
    function apply_D(z::Vector{T},x,w,R,D,z0) where {T}
        @assert all(isfinite.(z))
        p = length(w)
        n = length(D)
        Dz = Array{T,2}(undef,(p,n))
        Rz = z0+R*z
        for k=1:n
            Dz[:,k] = D[k]*Rz
        end
        return Dz
    end
    function f0(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(z,x,w,R,D,z0)
        p = length(w)
        n = length(D)
        y = [F(x[k,:],Dz[k,:]) for k=1:p]
        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])
    end
    function f1(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(z,x,w,R,D,z0)
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
        Dz = apply_D(z,x,w,R,D,z0)
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
    Barrier(f0=f0,f1=f1,f2=f2)
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
        maxit=10000,
        maxit2=Int(ceil(log2(-log2(eps(T)))))+2) where {T,Mat,Geometry}
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
        x = xm[J]
        w = wm[J]
        R = M.R_coarse[J]
        D = M.D[J,:]
        z0 = zm[J]
        c0 = cm[J]
        s0 = zeros(T,(size(R)[2],))
        mi = if J-j==1 maxit else maxit2 end
        SOL = newton(Mat,
                s->f0(s,x,w,c0,R,D,z0),
                s->f1(s,x,w,c0,R,D,z0),
                s->f2(s,x,w,c0,R,D,z0),
                s0,
                maxit=mi)
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
    (z=zm[L],its=its,passed=passed)
end
function amgb_step(B::Barrier,
        M::AMG{T,Mat,Geometry},
        x::Matrix{T},
        z::Vector{T},
        c::Matrix{T};
        maxit=Int(ceil(log2(-log2(eps(T)))))+2,
        early_stop=z->false) where {T,Mat,Geometry}
    L = length(M.R_fine)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D[L,:]
    function eta(j,J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = zeros(T,(size(R)[2],))
        SOL = newton(Mat,
            s->f0(s,x,w,c,R,D,z),
            s->f1(s,x,w,c,R,D,z),
            s->f2(s,x,w,c,R,D,z),
            s0,
            maxit=maxit)
        its[J] += SOL.k
        if SOL.converged
            z = z + R*SOL.x
        end
        return SOL.converged
    end
    converged = divide_and_conquer(eta,0,L)
    return (z=z,its=its,converged=converged)
end

"""
    function illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}

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

"""
    function newton(::Type{Mat},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::Array{T,1};
                       maxit=10000,
                       theta=T(0.1),
                       beta=T(0.1),
                       tol=eps(T)) where {T,Mat}

Damped Newton iteration for minimizing a function.

* `F0` the objective function
* `F1` and `F2` are the gradient and Hessian of `F0`, respectively.
* `x` the starting point of the minimization procedure.

The Hessian `F2` return value should be of type `Mat`.

The optional parameters are:
* `maxit`, the iteration aborts with a failure message if convergence is not achieved within `maxit` iterations.
* `tol` is used as a stopping criterion. We stop when the decrement in the objective is sufficiently small.
"""
function newton(::Type{Mat},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::Array{T,1};
                       maxit=10000,
                       theta=T(0.1),
                       beta=T(0.1),
                       tol=eps(T)) where {T,Mat}
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
    while k<maxit && !converged
        k+=1
        H = F2(x) ::Mat
        n = ((H+I*norm(H)*eps(T))\g)::Array{T,1}
        @assert all(isfinite.(n))
        inc = dot(g,n)
        if inc<=0
            converged = true
            break
        end
        s = T(1)
        while s>T(0)
            try
                function phi(s)
                    xn = x-s*n
                    @assert(isfinite(F0(xn)))
                    return dot(F1(xn),n)
                end
                s = illinois(phi,T(0),s,fa=inc)
                xnext = x-s*n
                ynext,gnext = F0(xnext)::T,F1(xnext)
                @assert isfinite(ynext) && all(isfinite.(gnext))
                break
            catch
            end
            s = s*beta
        end
        if ynext>=ymin && norm(gnext)>=theta*norm(g)
            converged = true
        end
        x,y,g = xnext,ynext,gnext
        ymin = min(ymin,y)
        push!(ss,s)
        push!(ys,y)
    end
    return (x=x,y=y,k=k,converged=converged,ss=ss,ys=ys)
end

"""
    function amgb_core(B::Barrier,
        M::AMG{T,Mat,Geometry},
        x::Matrix{T},
        z::Array{T,1},
        c::Array{T,2};
        tol=(eps(T)),
        t=T(0.1),
        maxit=10000,
        kappa=T(10.0),
        early_stop=z->false,
        verbose=true) where {T,Mat,Geometry}

The "Algebraic MultiGrid Barrier" method.

* `B` a Barrier object.
* `M` an AMG object.
* `x` a matrix with the same number of rows as `M.x`. This is passed as the `x` parameter of the barrier. Often, `x = M.x`.
* `z` a starting point for the minimization, which should be admissible, i.e. `B.f0(z)<∞`.
* `c` an objective functional to minimize. Concretely, we minimize the integral of `c.*(D*z)`, as computed by the finest quadrature in `M`, subject to `B.f0(z)<∞`. Here, `D` is the differential operator provided in `M`.

Optional parameters:

* `t`: the initial value of `t`
* `tol`: we stop when `1/t<tol`.
* `maxit`: the maximum number of `t` steps.
* `kappa`: the initial size of the t-step. Stepsize adaptation is used in the AMGB algorithm, where the t-step size may be made smaller or large, but it will never exceed the initial size provided here.
* `verbose`: set to `true` to see a progress bar.
* `early_stop`: if `early_stop(z)` is `true` then the minimization is stopped early. This is used when solving the preliminary feasibility problem.

Return value is a named tuple `SOL` with the following fields:
* `SOL.converged` is `true` if convergence was obtained, else it is `false`.
* `SOL.z` the computed solution.
Further `SOL` fields contain various statistics about the solve process.
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
        divide_and_conquer=true) where {T,Mat,Geometry}
    t_begin = time()
    tinit = t
    kappa0 = kappa
    L = length(M.R_fine)
    its = zeros(Int,(L,maxit))
    ts = zeros(T,(maxit,))
    kappas = zeros(T,(maxit,))
    times = zeros(Float64,(maxit,))
    k = 1
    times[k] = time()
    SOL = amgb_phase1(B,M,x,z,c0 .+ t*c,maxit=maxit)
    passed = SOL.passed
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    mi = Int(ceil(log2(-log2(eps(T)))))+2
    while t<=1/tol && kappa > 1 && k<maxit && !early_stop(z)
        k = k+1
        its[:,k] .= 0
        times[k] = time()
        prog = ((log(t)-log(tinit))/(log(1/tol)-log(tinit)))
        progress(prog)
        while kappa > 1
            t1 = kappa*t
            SOL = amgb_step(B,M,x,z,c0 .+ t1*c,maxit=mi,early_stop=early_stop)
            its[:,k] += SOL.its
            if SOL.converged
                if maximum(SOL.its)<=mi*0.5
                    kappa = min(kappa0,kappa^2)
                end
                z = SOL.z
                t = t1
                break
            end
            kappa = sqrt(kappa)
        end
        ts[k] = t
        kappas[k] = kappa
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        throw(AMGBConvergenceFailure("Convergence failure in amgb at t=$t, k=$k, kappa=$kappa."))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    return (z=z,c=c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],M=M,
            t_begin=t_begin,t_end=t_end,t_elapsed=t_elapsed,times=times[1:k],passed=passed)
end

"""
    function amgb(M::Tuple{AMG{T,Mat,Geometry},AMG{T,Mat,Geometry}},
              f::Union{Function,Matrix{T}}, 
              g::Union{Function,Matrix{T}}, 
              Q::Convex;
              x::Matrix{T} = M[1].x,
              t=T(0.1),
              t_feasibility=t,
              verbose=true,
              return_details=false,
              rest...) where {T,Mat,Geometry}

A thin wrapper around `amgb_core()`. Parameters are:

* `M`: obtained from the `amg` constructor, a pair of `AMG` structures. `M[1]` is the main problem while `M[2]` is the feasibility problem.
* `f`: the functional to minimize.
* `g`: the "boundary conditions".
* `Q`: a `Convex` domain for the convex optimization problem.
* `rest...`: any further named arguments are passed on to `amgb_core`.

The initial `z0` guess, and the cost functional `c0`, are computed as follows:

    m = size(M[1].x,1)
    for k=1:m
        z0[k,:] .= g(M[1].x[k,:])
        c0[k,:] .= f(M[1].x[k,:])
    end

By default, the return value `z` is an `m×n` matrix, where `n` is the number of `state_variables`, see either `fem1d()`, `fem2d()`, `spectral1d()` or `spectral2d()`. If `return_details=true` then the return value is a named tuple with fields `z`, `SOL_feasibility` and `SOL_main`; the latter two fields are named tuples with detailed information regarding the various solves.
"""
function amgb(M::Tuple{AMG{T,Mat,Geometry},AMG{T,Mat,Geometry}},
              f::Union{Function,Matrix{T}}, 
              g::Union{Function,Matrix{T}}, 
              Q::Convex;
              x::Matrix{T} = M[1].x,
              t=T(0.1),
              t_feasibility=t,
              verbose=true,
              return_details=false,
              rest...) where {T,Mat,Geometry}
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
    M0 = M[1]
    D0 = M0.D[end,1]
    xend = M0.x
    m = size(xend,1)
    ns = Int(size(D0,2)/m)
    nD = size(M0.D,2)
    w = zeros(T,(m,nD))
    c0 = f
    if f isa Function
        c0 = zeros(T,(m,nD))
        for k=1:m
            c0[k,:] .= f(x[k,:])
        end
    end
    z0 = g
    if g isa Function
        z0 = zeros(T,(m,ns))
        for k=1:m
            z0[k,:] .= g(x[k,:])
        end
    end
    wend = M0.w
    z2 = reshape(z0,(:,))
    for k=1:nD
        w[:,k] = M0.D[end,k]*z2
    end
    pbarfeas = 0.0
    feasible = true
    SOL1=nothing
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
            SOL1 = amgb_core(B1,M[2],x,z1,c1,t=t_feasibility,
                progress=x->progress(pbarfeas*x),
                early_stop=early_stop, rest...)#,c0=hcat(t*c0,zeros(T,(m,1))))
            @assert early_stop(SOL1.z)
        catch e
            if isa(e,AMGBConvergenceFailure)
                throw(AMGBConvergenceFailure("Could not solve the feasibility subproblem, probem may be infeasible. Failure was: "*e.message))
            end
            throw(e)
        end
        z2 = reshape((reshape(SOL1.z,(m,ns+1)))[:,1:end-1],(:,))
    end
    B = barrier(Q.barrier)
    SOL2 = amgb_core(B,M0,x,z2,c0,t=t,
        progress=x->progress((1-pbarfeas)*x+pbarfeas),rest...)
    z = reshape(SOL2.z,(m,:))
    if return_details
        return (z=z,SOL_feasibility=SOL1,SOL_main=SOL2)
    end
    return z
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

function amg_precompile()
    fem1d_solve(L=1)
    fem2d_solve(L=1)
    spectral1d_solve(L=1)
    spectral2d_solve(L=1)
end

precompile(amg_precompile,())
