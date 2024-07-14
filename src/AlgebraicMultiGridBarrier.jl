export Barrier, AMG, barrier, amgb, amg, newton, illinois

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
        x::Matrix{T},
        w::Vector{T},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        subspaces::Dict{Symbol,Vector{M}},
        operators::Dict{Symbol,M},
        refine::Vector{M},
        coarsen::Vector{M}) where {T,M}

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
"""
function amg(;
        x::Matrix{T},
        w::Vector{T},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        subspaces::Dict{Symbol,Vector{M}},
        operators::Dict{Symbol,M},
        refine::Vector{M},
        coarsen::Vector{M}) where {T,M}
    L = length(refine)
    @assert size(w) == (size(x)[1],) && size(refine)==(L,) && size(coarsen)==(L,)
    x0 = x
    x = Vector{Matrix{T}}(undef,L)
    x[L] = x0
    for l=L-1:-1:1
        x[l] = coarsen[l]*x[l+1]
    end
    for l=1:L
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
function amgb_phase1(B::Barrier,
        M::AMG{T,Mat},
        z::Vector{T},
        c::Matrix{T};
        maxit=10000) where {T,Mat}
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
        SOL = newton(Mat,
                s->f0(s,x,w,c0,R,D,z0),
                s->f1(s,x,w,c0,R,D,z0),
                s->f2(s,x,w,c0,R,D,z0),
                s0,
                maxit=maxit)
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
        maxit=Int(ceil(log2(-log2(eps(T)))))+2) where {T,Mat}
    L = length(M.w)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    converged = false
    x = M.x[L]
    w = M.w[L]
    D = M.D[L,:]
    message = ""
    function step(j,J)
        R = M.R_fine[J]
        s0 = zeros(T,(size(R)[2],))
        while true
            SOL = newton(Mat,
                s->f0(s,x,w,c,R,D,z),
                s->f1(s,x,w,c,R,D,z),
                s->f2(s,x,w,c,R,D,z),
                s0,
                maxit=maxit)
            its[J] += SOL.k
            if SOL.converged
                z = z+R*SOL.x
                return true
            end
            jmid = (j+J)÷2
            if jmid==j
                message = "amgb step failed at level $j"
                return false
            end
            if !step(j,jmid)
                return false
            end
            j = jmid
        end
    end
    if step(0,L)
        converged = true
    end
    return (z=z,its=its,converged=converged,message=message)
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
    error("illinois solver failed to converge")
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
                ynext,gnext = F0(xnext),F1(xnext)
                @assert isfinite(ynext) && all(isfinite.(gnext))
                break
            catch
            end
            s = s*beta
        end
        if ynext>=y && norm(gnext)>=theta*norm(g)
            converged = true
        end
        x,y,g = xnext,ynext,gnext
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
* `kappa`: the initial size of the t-step. Stepsize adaptation is used in the AMGB algorithm, where the t-step size may be made smaller or large, but it will never exceed the initial size provided here.
* `verbose`: set to `true` to see a progress bar.

Return value is a named tuple `SOL` with the following fields:
* `SOL.converged` is `true` if convergence was obtained, else it is `false`.
* `SOL.z` the computed solution.
Further `SOL` fields contain various statistics about the solve process.

The following "example usage" is an extremely convoluted way of minimizing x in the interval [-1,1]:
```jldoctest
using MultiGridBarrier
M = amg(x = [-1.0 ; 1.0 ;;],
        w = [1.0,1.0],
        state_variables = [:u :space],
        D = [:u :id],
        subspaces = Dict(:space => [[1.0 ; -1.0 ;;]]),
        operators = Dict(:id => [1.0 0.0;0.0 1.0]),
        refine = [[1.0 0.0 ; 0.0 1.0]],
        coarsen = [[1.0 0.0 ; 0.0 1.0]])
B = barrier((x,y)->-log(1-x[1]*y[1]))
amgb(B,M,[0.0,0.0],[1.0 ; 0.0 ;;],verbose=false).z[1]

# output

-0.9999999999999998
```
"""
function amgb(B::Barrier,
        M::AMG{T,Mat},
        z::Array{T,1},
        c::Array{T,2};
        tol=(eps(T)),
        t=T(0.1),
        maxit=10000,
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
    its = zeros(Int,(L,maxit))
    ts = zeros(T,(maxit,))
    kappas = zeros(T,(maxit,))
    times = zeros(Float64,(maxit,))
    k = 1
    times[k] = time()
    SOL = amgb_phase1(B,M,z,t*c,maxit=maxit)
    passed = SOL.passed
    its[:,k] = SOL.its
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
            while kappa > 1
                t1 = kappa*t
                SOL = amgb_step(B,M,z,t1*c,maxit=mi)
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

"""
    function amgb(;
              M::AMG{T,Mat},
              f::Function, g::Function, F::Function,
              tol=sqrt(eps(T)),
              t=T(0.1),
              maxit=10000,
              kappa=T(10.0),
              verbose=true) where {T,Mat}

This is a thin wrapper around the function call
```
    amgb(B,M,reshape(z0,(m*ns,)),c,
        kappa=kappa,maxit=maxit,verbose=verbose,tol=tol)
```
The initial value `z0` and the functional `c` are calculated as follows:
```
    for k=1:m
        z0[k,:] .= g(xend[k,:]...)
        c[k,:] .= f(xend[k,:]...)
    end
```
The `Barrier` object `B` is constructed from `F`.
"""
function amgb(;
              M::AMG{T,Mat},
              f::Function, g::Function, F::Function,
              tol=sqrt(eps(T)),
              t=T(0.1),
              maxit=10000,
              kappa=T(10.0),
              verbose=true) where {T,Mat}
    D0 = M.D[end,1]
    xend = M.x[end]
    m = size(xend,1)
    ns = Int(size(D0,2)/m)
    nD = size(M.D,2)
    z0 = zeros(T,(m,ns))
    c = zeros(T,(m,nD))
    for k=1:m
        z0[k,:] .= g(xend[k,:]...)
        c[k,:] .= f(xend[k,:]...)
    end
    B = barrier((x,y)->F(x...,y...))
    amgb(B,M,reshape(z0,(m*ns,)),c,
        kappa=kappa,maxit=maxit,verbose=verbose,tol=tol)
end


function amgb_precompile(::Type{T}) where {T}
    M = amg(x = T[-1.0 ; 1.0 ;;],
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
