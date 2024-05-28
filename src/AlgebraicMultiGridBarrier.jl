export Barrier, AMG, barrier, amgb, amg

function blkdiag(M...)
    Mat = typeof(M[1])
    Mat(blockdiag((sparse.(M))...))
end


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
    @kwdef struct AMG{T,M}
        ...
    end

Objects of this type should probably be assembled by the constructor `amg()`.

A multigrid with `L` level. Denote by `l` between 1 and `L`, a grid level.
Fields are:
* `x::Array{Array{T,2},1}` an array of `L` matrices. `x[l]` stores the vertices of the grid at multigrid level `l`.
* `w::Array{Array{T,1},1}` an array of `L` quadrature weights. `w[l]` corresponds to `x[l]`.
* `R_fine::Array{M,1}` an array of `L` matrices. The columns of `R_fine[l]` are basis functions for the function space on grid level `l`, interpolated to the fine grid.
* `R_coarse::Array{M,1}` an array of `L` matrices. The columns of `R_coarse[l]` are basis functions for the function space on grid level `l`. Unlike `R_fine[l]`, these basis functions are on grid level `l`, not interpolated to the fine grid.
* `D::Array{M,2}` an array of differential operators. For example, if the barrier parameters are to be `u,ux,s`, with `ux` the derivative of `u`, then `D[l,:] = [I,Dx,I]`, where `Dx` is a numerical differentiation operator on grid level `l`.  
* `refine_u::Array{M,1}` an array of `L` grid refinement matrices. If `x[l]` has `n[l]` vertices, then `refine_u[l]` is `n[l+1]` by `n[l]`.
* `coarsen_u::Array{M,1}` an array of `L` grid coarsening matrices. `coarsen_u[l]` is `n[l]` by `n[l+1]`.
* `refine_z::Array{M,1}` an array of `L` grid refining matrices for the "state vector" `z`. For example, if `z` contains the state functions `u` and `s`, then there are `k=2` state functions, and `refine_z[l]` is `k*n[l+1]` by `k*n[l]`.
* `coarsen_z::Array{M,1}` an array of `L` grid coarsening matrices for the "state vector" `z`. `coarsen_z[l]` is `k*n[l]` by `k*n[l+1]`.

These various matrices must satisfy a wide variety of algebraic relations. For this reason, it is recommended to use the constructor `amg()`.
"""
@kwdef struct AMG{T,M}
    x::Array{Array{T,2},1}
    w::Array{Array{T,1},1}
    R_fine::Array{M,1}
    R_coarse::Array{M,1}
    D::Array{M,2}
    refine_u::Array{M,1}
    coarsen_u::Array{M,1}
    refine_z::Array{M,1}
    coarsen_z::Array{M,1}
end

"""
    function amg(;
        x::Array{Array{T,2},1},
        w::Array{T,1},
        state_variables::Array{Symbol,2},
        D::Array{Symbol,2},
        subspaces::Dict{Symbol,Array{M,1}},
        operators::Dict{Symbol,M},
        refine::Array{M,1},
        coarsen::Array{M,1}) where {T,M}

Construct an `AMG` object for use with the `amgb` solver. In many cases, this constructor is not called directly by the user. For 1d and 2d finite elements, use the `fem1d()` or `fem2d()`. For 1d and 2d spectral elements, use  `spectral1d()` or `spectral2d()`. You use `amg()` directly if you are implementing your own function spaces.

The `AMG` object shall represent all `L` grid levels of the multigrid hierarchy. Parameters are:
* `x`: an array of `L` matrices. `x[l]` has the vertices of grid level `l`, one vertex per row.
* `w`: an array of `L` vectors. `w[l]` has the quadrature weights for grid level `l`.
* `state_variables`: a matrix of symbols. The first column indicates the names of the state vectors or functions, and the second column indicates the names of the corresponding subspaces. A typical example is: `state_variables = [:u :dirichlet; :s :full]`. This would define the solution as being functions named u(x) and s(x). The u function would lie in the space `:dirichlet`, presumably consisting of functions with homogeneous Dirichlet conditions. The s function would lie in the space `:full`, presumably being the full function space, without boundary conditions.
* `D`: a matrix of symbols. The first column indicates the names of various state variables, and the second column indicates the corresponding differentiation operator(s). For example: `D = [:u :id ; :u :dx ; :s :id]`. This would indicate that the barrier should be called as `F(x,y)` with `y = [u,ux,s]`, where `ux` denotes the derivative of `u` with respect to the space variable `x`.
* `subspaces`: a `Dict` mapping each subspace symbol to an array of `L` matrices, e.g. for each `l`, `subspaces[:dirichlet][l]` is a matrix whose columns span the homogeneous Dirichlet subspace of grid level `l`.
* `operators`: a `Dict` mapping each differential operator symbol to a matrix, e.g. `operators[:id]` is an identity matrix, while `operators[:dx]` is a numerical differentiation matrix, on the fine grid level `L`.
* `refine`: an array of length `L` of matrices. For each `l`, `refine[l]` interpolates from grid level `l` to grid level `l+1`. `refine[L]` should be the identity, and `coarsen[l]*refine[l]` should be the identity.
* `coarsen`: an array of length `L` of matrices. For each `l`, `coarsen[l]` interpolates or projects from grid level `l+1` to grid level `l`. `coarsen[L]` should be the identity.
"""
function amg(;
        x::Array{Array{T,2},1},
        w::Array{T,1},
        state_variables::Array{Symbol,2},
        D::Array{Symbol,2},
        subspaces::Dict{Symbol,Array{M,1}},
        operators::Dict{Symbol,M},
        refine::Array{M,1},
        coarsen::Array{M,1}) where {T,M}
    L = length(x)
    @assert size(w) == (size(x[L])[1],) && size(refine)==(L,) && size(coarsen)==(L,)
    nx = size(x[1])[2]
    for l=1:L
        @assert size(x[l],2) == nx
        @assert norm(coarsen[l]*refine[l]-I)<sqrt(eps(T))
    end
    refine_fine = Array{M,1}(undef,(L,))
    refine_fine[L] = refine[L]
    coarsen_fine = Array{M,1}(undef,(L,))
    coarsen_fine[L] = coarsen[L]
    w0 = Array{Vector{T},1}(undef,(L,))
    w0[L] = w
    for l=L-1:-1:1
        refine_fine[l] = refine_fine[l+1]*refine[l]
        coarsen_fine[l] = coarsen[l]*coarsen_fine[l+1]
        w0[l] = refine_fine[l]'*w
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
        n = length(w0[l])
        Z = M(spzeros(T,n,n))
        for k=1:nD
            foo = [Z for j=1:nu]
            foo[bar[D[k,1]]] = coarsen_fine[l]*operators[D[k,2]]*refine_fine[l]
            D0[l,k] = hcat(foo...)
        end
    end
    refine_z = [blkdiag([refine[l] for k=1:nu]...) for l=1:L]
    coarsen_z = [blkdiag([coarsen[l] for k=1:nu]...) for l=1:L]
    AMG{T,M}(x=x,w=w0,R_fine=R_fine,R_coarse=R_coarse,D=D0,
        refine_u=refine,coarsen_u=coarsen,refine_z=refine_z,coarsen_z=coarsen_z)
end



"""
    function barrier(f;
        f1=(x,y)->ForwardDiff.gradient(z->f(x,z),y),
        f2=(x,y)->ForwardDiff.hessian(z->f(x,z),y))::Barrier

Constructor for barriers.

* `f` is the actual barrier function. It should take parameters `(x,y)`.
* `f1` is the gradient of `f` with respect to `y`.
* `f2` is the Hessian of `f` with  respect to `y`.

By default, `f1` and `f2` are automatically generated by the module `ForwardDiff`.

A more specific description of the Barrier object is as follows. The function `Barrier.f0` has parameters:

    function Barrier.f0(z,x,w,c,R,D,z0)

Here, `R` is a matrix and `D` is an array of matrices. Define `Rz = z0+R*z`, then `Dz[:,k] = D[k]*Rz`. Then, the value of `Barrier.f0` is given by:
```
        p = length(w)
        n = length(D)
        y = [f(x[k,:],Dz[k,:]) for k=1:p]
        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])
```

Functions `Barrier.f1` and `Barrier.f2` are the gradient and Hessian, respectively, of `Barrier.f0`, with respect to the `z` parameter.
"""
function barrier(f;
        f1=(x,y)->ForwardDiff.gradient(z->f(x,z),y),
        f2=(x,y)->ForwardDiff.hessian(z->f(x,z),y))::Barrier
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
    function F0(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(z,x,w,R,D,z0)
        p = length(w)
        n = length(D)
        y = [f(x[k,:],Dz[k,:]) for k=1:p]
        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])
    end
    function F1(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(z,x,w,R,D,z0)
        p = length(w)
        n = length(D)
        y = Array{T,2}(undef,(p,n))
        for k=1:p
            y[k,:] = f1(x[k,:],Dz[k,:])
        end
        y += c
        m0 = size(D[1],2)
        ret = zeros(T,(m0,))
        for k=1:n
            ret += D[k]'*(w.*y[:,k])
        end
        R'*ret
    end
    function F2(z::Vector{T},x,w,c,R,D,z0) where {T}
        Dz = apply_D(z,x,w,R,D,z0)
        p = length(w)
        n = length(D)
        y = Array{T,3}(undef,(p,n,n))
        for k=1:p
            y[k,:,:] = f2(x[k,:],Dz[k,:])
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
    Barrier(f0=F0,f1=F1,f2=F2)
end
function amgb_phase1(B::Barrier,
        M::AMG{T,Mat},
        z::Vector{T},
        c::Matrix{T};
        maxit=10000,
        alpha=T(0.5),
        beta=T(0.1)) where {T,Mat}
    L = length(M.w)
    cm = Vector{Matrix{T}}(undef,L)
    cm[L] = c
    zm = Vector{Vector{T}}(undef,L)
    zm[L] = z
    passed = falses((L,))
    for l=L-1:-1:1
        cm[l] = M.coarsen_u[l]*cm[l+1]
        zm[l] = M.coarsen_z[l]*zm[l+1]
    end
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    converged = false
    message = ""
    for l=1:L
        x = M.x[l]
        w = M.w[l]
        R = M.R_coarse[l]
        D = M.D[l,:]
        z0 = zm[l]
        c0 = cm[l]
        s0 = zeros(T,(size(R)[2],))
        SOL = damped_newton(s->f0(s,x,w,c0,R,D,z0),
                s->f1(s,x,w,c0,R,D,z0),
                s->f2(s,x,w,c0,R,D,z0),
                s0,
                maxit=maxit,alpha=alpha,beta=beta)
        converged = SOL.converged
        if !converged
            it = SOL.k
            message = "damped Newton iteration failed to converge at level $l during phase 1 ($it iterations, maxit=$maxit)"
            break
        end
        its[l] = SOL.k
        znext = copy(zm)
        s = R*SOL.x
        try
            for k=l+1:L
                s = M.refine_z[k-1]*s
                znext[k] = zm[k]+s
                s0 = zeros(T,(size(M.R_coarse[k])[2],))
                y0 = f0(s0,M.x[k],M.w[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])
                y1 = f1(s0,M.x[k],M.w[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])
                @assert isfinite(y0) && all(isfinite.(y1))
            end
            zm = znext
            passed[l] = true
        catch
        end
    end
    (z=zm[L],its=its,converged=converged,passed=passed,message=message)
end
function amgb_step(B::Barrier,
        M::AMG{T,Mat},
        z::Vector{T},
        c::Matrix{T};
        maxit=Int(ceil(log2(-log2(eps(T)))))+2,
        alpha=T(0.5),
        beta=T(0.1),
        greedy_step=true) where {T,Mat}
    L = length(M.w)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L+1,))
    converged = false
    x = M.x[L]
    w = M.w[L]
    D = M.D[L,:]
    message = ""
    if greedy_step
        R = M.R_fine[L]
        s0 = zeros(T,(size(R)[2],))
        SOL = damped_newton(s->f0(s,x,w,c,R,D,z),
                s->f1(s,x,w,c,R,D,z),
                s->f2(s,x,w,c,R,D,z),
                s0,
                maxit=maxit,alpha=alpha,beta=beta)
        its[1] = SOL.k
        if SOL.converged
            z = z+R*SOL.x
            converged = true
        end
    end
    if !converged
        converged = true
        for l=1:L
            R = M.R_fine[l]
            s0 = zeros(T,(size(R)[2],))
            SOL = damped_newton(s->f0(s,x,w,c,R,D,z),
                s->f1(s,x,w,c,R,D,z),
                s->f2(s,x,w,c,R,D,z),
                s0,
                maxit=maxit,alpha=alpha,beta=beta)
            its[l+1] = SOL.k
            if !SOL.converged
                converged = false
                message = "damped Newton iteration failed to converge at level $l during amgb step"
                break
            end
            z = z+R*SOL.x
        end
    end
    return (z=z,its=its,converged=converged,message=message)
end

"""
    function damped_newton(F0::Function,
                       F1::Function,
                       F2::Function,
                       x::Array{T,1};
                       maxit=10000,
                       alpha=T(0.5),
                       beta=T(0.1)) where {T}

Damped Newton iteration for minimizing a function.

* `F0` the objective function
* `F1` and `F2` are the gradient and Hessian of `F0`, respectively.
* `x` the starting point of the minimization procedure.

The optional parameters are:
* `maxit`, the iteration aborts with a failure message if convergence is not achieved within `maxit` iterations.
* `alpha` and `beta` are the parameters of the backtracking line search.
* `tol` is used as a stopping criterion. We stop when the decrement in the objective is sufficiently small.
"""
function damped_newton(F0::Function,
                       F1::Function,
                       F2::Function,
                       x::Array{T,1};
                       maxit=10000,
                       alpha=T(0.5),
                       beta=T(0.1)) where {T}
    ss = T[]
    ys = T[]
    @assert all(isfinite.(x))
    y = F0(x) ::T
    @assert isfinite(y)
    push!(ys,y)
    converged = false
    k = 0
    gnext = x
    g = F1(x) ::Array{T,1}
    @assert all(isfinite.(g))
    xnext = x
    ynext = y
    (xmid,ymid,gmid) = (x,y,x)
    message = ""
    while k<maxit && !converged
        k+=1
        H = F2(x)
        n = ((H+I*norm(H)*eps(T))\g)::Array{T,1}
        @assert all(isfinite.(n))
        inc = dot(g,n)
        if inc<=0
            converged = true
            break
        end
        s = T(1)
        (xnext,ynext,gnext) = (x,y,g)
        while s>T(0)
            try
                xnext = x-s*n ::Array{T,1}
                ynext = F0(xnext) ::T
                if isfinite(ynext) && (y-beta*s*inc>=ynext || xnext==x)
                    break
                end
            catch
            end
            s = s*beta
        end
        gnext = F1(xnext) ::Array{T,1}
        if (y<=ynext && alpha*norm(g) <= norm(gnext)) || s==T(0) || x==xnext
            converged = true
        end
        (x,y,g) = (xnext, ynext, gnext)
        push!(ss,s)
        push!(ys,y)
    end
    return (x=x,y=y,k=k,converged=converged,ss=ss,ys=ys)
end

"""
    function amgb(B::Barrier,
        M::AMG{T,Mat},
        z::Array{T,1},
        c::Array{T,2};
        tol=(eps(T)),
        t=T(0.1),
        maxit=10000,
        alpha=T(0.5),
        beta=T(0.1),
        kappa=T(10.0),
        verbose=true) where {T,Mat}

The "Algebraic MultiGrid Barrier" method.

* `B` a Barrier object.
* `M` an AMG object.
* `z` a starting point for the minimization, which should be admissible, i.e. `B.f0(z)<∞`.
* `c` an objective functional to minimize. Concretely, we minimize the integral of `c.*(D*z)`, as computed by the finest quadrature in `M`, subject to `B.f0(z)<∞`. Here, `D` is the differential operator provided in `M`.

Optional parameters:

* `t`: the initial value of `t`
* `tol`: we stop when `1/t<tol`.
* `maxit`: the maximum number of `t` steps.
* `alpha`, `beta`: parameters of the backtracking line search.
* `kappa`: the initial size of the t-step. Stepsize adaptation is used in the AMGB algorithm, where the t-step size may be made smaller or large, but it will never exceed the initial size provided here.
* `verbose`: set to `true` to see a progress bar.

Return value is a named tuple `SOL` with the following fields:
* `SOL.converged` is `true` if convergence was obtained, else it is `false`.
* `SOL.z` the computed solution.
Further `SOL` fields contain various statistics about the solve process.

The following "example usage" is an extremely convoluted way of minimizing x in the interval [-1,1]:
```
using AlgebraicMultiGridBarrier
M = amg(x = [[-1.0 ; 1.0 ;;]],
        w = [1.0,1.0],
        state_variables = [:u :space],
        D = [:u :id],
        subspaces = Dict(:space => [[1.0 ; -1.0 ;;]]),
        operators = Dict(:id => [1.0 0.0;0.0 1.0]),
        refine = [[1.0 0.0 ; 0.0 1.0]],
        coarsen = [[1.0 0.0 ; 0.0 1.0]])
B = barrier((x,y)->-log(1-x[1]*y[1]))
amgb(B,M,[0.0,0.0],[1.0 ; 0.0 ;;])
```
"""
function amgb(B::Barrier,
        M::AMG{T,Mat},
        z::Array{T,1},
        c::Array{T,2};
        tol=(eps(T)),
        t=T(0.1),
        maxit=10000,
        alpha=T(0.5),
        beta=T(0.1),
        kappa=T(10.0),
        verbose=true) where {T,Mat}
    t_begin = time()
    pbar = 0
    tinit = t
    if verbose
        pbar = Progress(1000000; dt=1.0)
    end
    kappa0 = kappa
    converged = false
    L = length(M.R_fine)
    its = zeros(Int,(L+1,maxit))
    ts = zeros(T,(maxit,))
    kappas = zeros(T,(maxit,))
    times = zeros(Float64,(maxit,))
    k = 1
    times[k] = time()
    SOL = amgb_phase1(B,M,z,t*c,maxit=maxit,alpha=alpha,beta=beta)
    passed = SOL.passed
    its[2:end,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    message = ""
    if SOL.converged
        z = SOL.z
        mi = Int(ceil(log2(-log2(eps(T)))))+2
        while t<=1/tol && kappa > 1 && k<maxit
            k = k+1
            its[:,k] .= 0
            times[k] = time()
            if verbose
                permil = 1000000*((log(t)-log(tinit))/(log(1/tol)-log(tinit)))
                update!(pbar,Int(floor(permil)))
            end
            greedy_step = true
            while kappa > 1
                t1 = kappa*t
                SOL = amgb_step(B,M,z,t1*c,greedy_step=greedy_step,maxit=mi,alpha=alpha,beta=beta)
                its[:,k] += SOL.its
                if SOL.converged
                    if greedy_step && SOL.its[1]<=mi*0.5
                        kappa = min(kappa0,kappa^2)
                    end
                    z = SOL.z
                    t = t1
                    break
                end
                kappa = sqrt(kappa)
                greedy_step = false
            end
            ts[k] = t
            kappas[k] = kappa
        end
        converged = (t>1/tol)
        if !converged
            message = "convergence failure in amgb at t=$t, k=$k, kappa=$kappa"
        end
    else
        message = SOL.message
    end
    if verbose
        update!(pbar,100)
        finish!(pbar)
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    return (z=z,c=c,converged=converged,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],M=M,
            t_begin=t_begin,t_end=t_end,t_elapsed=t_elapsed,times=times[1:k],passed=passed,
            message=message)
end

function amgb_precompile(::Type{T}) where {T}
    M = amg(x = [T[-1.0 ; 1.0 ;;]],
        w = T[1.0,1.0],
        state_variables = [:u :space],
        D = [:u :id],
        subspaces = Dict(:space => [T[1.0 ; -1.0 ;;]]),
        operators = Dict(:id => T[1.0 0.0;0.0 1.0]),
        refine = [T[1.0 0.0 ; 0.0 1.0]],
        coarsen = [T[1.0 0.0 ; 0.0 1.0]])
    B = barrier((x,y)->-log(1-x[1]*y[1]))
    amgb(B,M,T[0.0,0.0],T[1.0 ; 0.0 ;;],verbose=false,tol=T(0.1))
end
precompile(amgb_precompile,(Float64,))
precompile(amgb_precompile,(BigFloat,))
