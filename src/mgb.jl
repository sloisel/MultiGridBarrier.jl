# The MGB V-cycle: mgb_step, mgb_core, mgb_driver, MGBSOL, and mgb_solve.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

function divide_and_conquer(eta,j,J)
    if eta(j,J) return true end
    jmid = (j+J)÷2
    if jmid==j || jmid==J return false end
    return divide_and_conquer(eta,j,jmid) && divide_and_conquer(eta,jmid,J)
end
function mgb_step(Q::Convex{T},
        M::AMG{X,W,M_sub,<:Any,<:Any},
        z::W,
        c::X;
        early_stop,
        maxit,
        max_newton,
        line_search,
        stopping_criterion,
        finalize,
        printlog,
        args...
        ) where {T,X,W,M_sub}
    L = length(M.R_fine)
    B = barrier(Q)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D_fine
    function eta(j,J,sc,maxit,ls)
        @debug("j=",j," J=",J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = mgb_zeros(W, size(R, 2))
        SOL = newton(M_sub,T,
            s->f0(s,w,c,R,D,z),
            s->f1(s,w,c,R,D,z),
            s->f2(s,w,c,R,D,z),
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


function mgb_core(Q::Convex{T},
        M::AMG{X,W,M_sub,<:Any,<:Any},
        z::W,
        c::X;
        tol=sqrt(eps(T)),
        t=T(0.1),
        maxit=10000,
        kappa=T(10.0),
        early_stop=z->false,
        progress=x->nothing,
#        c0=T(0),
        max_newton= Int(ceil((log2(-log2(eps(T))))+2)),
        printlog,
        finalize,
        args...) where {T,X,W,M_sub}
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
    # Initial return-to-center at t, on the fine quadrature rule. The main
    # MGBstep multigrid sweep handles centring from the (feasible) starting
    # point; no coarse/hierarchical quadrature is used. For the feasibility
    # subproblem `early_stop` halts the t-ramp as soon as strict feasibility
    # is reached.
    SOL = mgb_step(Q,M,z,t*c;max_newton,early_stop,maxit,printlog,finalize=false,args...)
    @debug("initial centering done")
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    z_unfinalized = z
    Dz = apply_D(M.D_fine, z)
    c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:length(M.D_fine)])
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
            SOL = mgb_step(Q,M,z,t1*c;
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
        #c_dot_Dz[k] = dot(M.w .* c, apply_D(M.D_fine, z))
        Dz = apply_D(M.D_fine, z)
        c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:length(M.D_fine)])
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        throw(MGBConvergenceFailure("Convergence failure in mgb_solve at t=$t, k=$k, kappa=$kappa, tol=$tol, maxit=$maxit."))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    @debug("success. t=",t," tol=",tol)
    return (;z,z_unfinalized,c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],
            t_begin,t_end,t_elapsed,times=times[1:k],
            c_dot_Dz=c_dot_Dz[1:k])
end

function mgb_driver(M::Tuple{AMG{X,W,M_sub,<:Any,<:Any},AMG{X,W,M_sub,<:Any,<:Any}},
              f::X,
              g::X,
              Q::Convex{T};
              t=T(0.1),
              t_feasibility=t,
              progress = x->nothing,
              # Newton-decrement tolerance for the inexact (central-path) stopping.
              # The flat-averaged barrier (1/n)Σ F is self-concordant with parameter
              # μ = C_RH ~ n (see paper.tex, "self-concordance on the sublevel set"),
              # so the decrement threshold scales like η/μ ~ 1/n (not 1/√n).
              stopping_criterion=stopping_inexact(inv(T(length(M[1].w)))/2,T(0.5)),
              printlog = (args...)->nothing,
              line_search=linesearch_backtracking(T),
              finalize=stopping_exact(T(0.5)),
              rest...) where {T,X,W,M_sub}
    D0 = M[1].D_fine[1]
    m = size(M[1].x,1)
    ns = Int(size(D0,2)/m)
    nD = length(M[1].D_fine)
    L = length(M[1].R_fine)  # Number of multigrid levels
    c0 = f
    z0 = g
    Z = mgb_zeros(M[1].D_fine[1],m,m)
    # GPU-compatible: use `one` instead of closure capturing T
    ONES = map_rows_gpu(one, M[1].w)
    II = mgb_diag(M[1].D_fine[1],ONES)
    RR2 = []
    for k = 1:size(z0,2)
        foo = fill(Z,size(z0,2))
        foo[k] = II
        push!(RR2,hcat(foo...))
    end
    z2 = vcat([z0[:,k] for k=1:size(z0,2)]...)
    w = hcat([M[1].D_fine[k]*z2 for k=1:nD]...)
    pbarfeas = 0.0
    SOL_feasibility=nothing
    Q_L = Q
    barrier_f0_fn = Q_L.barrier[1]
    slack_fn = Q_L.slack
    cobar_f0 = Q_L.cobarrier[1]
    cobar_f1 = Q_L.cobarrier[2]
    cobar_f2 = Q_L.cobarrier[3]
    args_L = Q_L.args  # Args for finest level, splatted to map_rows_gpu
    try
        # GPU-compatible: splat Q.args to map_rows_gpu
        foo = map_rows_gpu(barrier_f0_fn, args_L..., w)
        @assert mgb_all_isfinite(foo)
    catch
        pbarfeas = 0.1
        # GPU-compatible: use slack_fn with args splatted
        z1 = map_rows_gpu((args_and_vecs...)->begin
            # Last two are z0_j and w_j, rest are args
            w_j = args_and_vecs[end]
            z0_j = args_and_vecs[end-1]
            args_j = args_and_vecs[1:end-2]
            push(z0_j, 2*max(slack_fn(args_j..., w_j), 1))
        end, args_L..., z0, w)
        b = 2*max(1,maximum(z1[:,size(z1,2)]))
        foo = zeros(T,(nD+1,)); foo[end] = 1
        foo_sv = SVector(Tuple(foo))
        c1 = map_rows_gpu(k->foo_sv, M[1].w)

        # Feasibility barrier: dot(y,y) + Q.cobarrier(args...,y) - log(b² - y[end]²)
        # GPU-compatible: barriers receive (args_rows..., yy)
        function feas_f0(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            u = yy[end]
            dot(yy, yy) + cobar_f0(args_j..., yy) - log(b^2 - u^2)
        end
        function feas_f1(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            N = length(yy)
            TT = eltype(yy)
            u = yy[end]
            denom = b^2 - u^2
            # GPU-compatible: use ntuple instead of zeros(MVector)
            g_extra = SVector(ntuple(i -> i == N ? TT(2) * u / denom : zero(TT), Val(N)))
            TT(2) .* yy .+ cobar_f1(args_j..., yy) .+ g_extra
        end
        function feas_f2(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            N = length(yy)
            TT = eltype(yy)
            u = yy[end]
            denom = b^2 - u^2
            # Get cobarrier Hessian (flattened)
            H_cobar_flat = cobar_f2(args_j..., yy)
            # H_extra has only (N,N) entry = 2*(b² + u²)/denom²
            H_extra_nn = TT(2) * (b^2 + u^2) / denom^2
            # GPU-compatible: build H directly with ntuple instead of zeros(MMatrix)
            H = SMatrix{N,N,TT}(ntuple(Val(N*N)) do k
                i = (k - 1) % N + 1
                jj = (k - 1) ÷ N + 1
                val = H_cobar_flat[k]  # cobarrier contribution
                if i == jj
                    val += TT(2)  # 2*I diagonal
                end
                if i == N && jj == N
                    val += H_extra_nn
                end
                val
            end)
            SVector(H)  # Flatten
        end
        # Feasibility-subproblem Convex: reuse Q's args (the problem data).
        Q_feas = Convex{T}((feas_f0, feas_f1, feas_f2), (feas_f0, feas_f1, feas_f2), slack_fn, Q.args)

        z1 = vcat([z1[:,k] for k=1:size(z1,2)]...)
        foo = fill(Z,size(z0,2)+1)
        foo[end] = II
        WW = hcat(foo...)
        early_stop(z) = (maximum(WW*z)<0)
        try
            SOL_feasibility = mgb_core(Q_feas,M[2],z1,c1;t=t_feasibility,
                progress=x->progress(pbarfeas*x),
                early_stop,
                printlog,
                stopping_criterion,
                line_search,
                finalize,
                rest...)
            @assert early_stop(SOL_feasibility.z)
        catch e
            if isa(e,MGBConvergenceFailure)
                throw(MGBConvergenceFailure("Could not solve the feasibility subproblem, probem may be infeasible. Failure was: "*e.message))
            end
            throw(e)
        end
        # Extract main-problem components, dropping the feasibility slack
        z2 = SOL_feasibility.z[1:length(z2)]
    end
    SOL_main = mgb_core(Q,M[1],z2,c0;
        t,
        progress=x->progress((1-pbarfeas)*x+pbarfeas),
        printlog,
        stopping_criterion,
        line_search,
        finalize,
        rest...)
    z = hcat([RR2[k]*SOL_main.z for k=1:size(z0,2)]...)
    return (;z,SOL_feasibility,SOL_main)
end

# GPU-compatible default functions - return SVector, infer type from input x
default_f(T,::Val{1}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,::Val{2}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,::Val{3}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,k::Int) = default_f(T,Val(k))
default_g(T,::Val{1}) = (x)->SVector(x[1], oftype(x[1],2))
default_g(T,::Val{2}) = (x)->SVector(x[1]^2+x[2]^2, oftype(x[1],100))
default_g(T,::Val{3}) = (x)->SVector(x[1]^2+x[2]^2+x[3]^2, oftype(x[1],100))
default_g(T,k::Int) = default_g(T,Val(k))
default_D(::Val{1}) = [:u :id
              :u :dx
              :s :id]
default_D(::Val{2}) = [:u :id
              :u :dx
              :u :dy
              :s :id]
default_D(::Val{3}) = [:u :id
              :u :dx
              :u :dy
              :u :dz
              :s :id]
default_D(k::Int) = default_D(Val(k))

# Static indices for GPU-compatible indexing: idx = 2:dim+2 as SVector
default_idx(::Val{1}) = SVector(2, 3)
default_idx(::Val{2}) = SVector(2, 3, 4)
default_idx(::Val{3}) = SVector(2, 3, 4, 5)
default_idx(k::Int) = default_idx(Val(k))

"""
    MGBSOL{T,X,W,Discretization,G}

Solution object returned by `mgb_solve` and `parabolic_solve`.

# Type Parameters
- `T`: scalar numeric type
- `X`: solution/point matrix type (e.g. `Matrix{T}`, `CuMatrix{T}`)
- `W`: weight vector type
- `Discretization`: geometry descriptor (e.g. `FEM2D_P2{T}`, `SPECTRAL1D{T}`)
- `G`: full `Geometry` type

# Fields
- `z::X`: solution matrix of size `(n_nodes, n_components)`
- `SOL_feasibility`: feasibility phase diagnostics (`nothing` if initial point was feasible)
- `SOL_main`: main optimization phase diagnostics (NamedTuple)
- `log::String`: detailed iteration log
- `geometry::G`: the input `Geometry`

Supports `plot(sol)` to visualize the first solution component.
"""
struct MGBSOL{T,X,W,Discretization,G}
    z::X
    SOL_feasibility
    SOL_main
    log::String
    geometry::G
end
function MGBSOL(z::X, sf, sm, log::String, geometry::Geometry{T,<:Any,W,<:Any,Discretization}) where {T,X,W,Discretization}
    MGBSOL{T,X,W,Discretization,typeof(geometry)}(z, sf, sm, log, geometry)
end
plot(sol::MGBSOL,k::Int=1;kwargs...) = plot(sol.geometry,sol.z[:,k];kwargs...)

"""
    mgb_solve(mg::MultiGrid; kwargs...) -> MGBSOL

MultiGrid Barrier (MGB) solver for nonlinear convex optimization problems on a multigrid
hierarchy. Operates in a feasibility phase followed by a main optimization phase, with
damped Newton inner solves and line search.

# Keyword Arguments

## Problem Specification
- `dim::Integer = amg_dim(mg.geometry.discretization)`: spatial dimension; auto-detected.
- `state_variables = [:u :dirichlet; :s :full]`: solution components and their function spaces.
- `D = default_D(dim)`: differential operators to apply to state variables.
- `x = _xflat(mg)`: sample points where `f`/`g` are evaluated when given as
  functions; a `(V·N, D)` flat view of the mesh tensor, one row per node.

## Problem Data
- `p::T = T(1.0)`: exponent for the p-Laplace term.
- `g`/`g_grid`, `f`/`f_grid`: boundary/initial data and forcing.
- `Q::Convex{T}`: convex constraint specification; defaults to a
  p-Laplace power-cone barrier.

## Output Control
- `verbose::Bool = true`: progress bar.
- `logfile = devnull`: optional log stream.

## Solver Control (forwarded internally)
- `tol`, `t`, `t_feasibility`, `maxit`, `kappa`, `early_stop`, `max_newton`,
  `stopping_criterion`, `line_search`, `finalize`, `progress`, `printlog`.

# Returns
An `MGBSOL` whose `z` is the fine-level solution matrix and whose `geometry` is the
`MultiGrid`'s fine-level `Geometry` (the `MultiGrid` itself is not stored).

# Examples
```julia
sol = mgb_solve(amg(fem1d(; nodes = collect(range(-1.0, 1.0, length=33)))); p = 1.5)
sol = mgb_solve(amg(subdivide(fem2d_P2(), 3)); p = 1.5)
sol = mgb_solve(amg(spectral2d(n = 8)); p = 2.0)
```
"""
function mgb_solve(mg::MultiGrid{T};
        dim::Integer = amg_dim(mg.geometry.discretization),
        state_variables = [:u :dirichlet ; :s :full],
        D = default_D(dim),
        M = _prepare_amg(mg;state_variables,D),
        x = _xflat(mg),
        p::T = T(1.0),
        g::Function = default_g(T,dim),
        f::Function = default_f(T,dim),
        g_grid = map_rows(xi->SVector(Tuple(g(xi))),x),
        f_grid = map_rows(xi->SVector(Tuple(f(xi))),x),
        Q::Convex{T} = convex_Euclidian_power(T; mg=mg, idx=default_idx(dim), p=xi->p),
        verbose=true,
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
    SOL = mgb_driver(M, f_grid, g_grid, Q; progress, printlog, rest...)
    return mgb_cleanup(MGBSOL(SOL.z, SOL.SOL_feasibility, SOL.SOL_main, String(take!(log_buffer)), mg.geometry))
end

"""
    mgb_solve(; mg::MultiGrid, kwargs...) -> MGBSOL

Keyword-only convenience method. Lets callers splat a `NamedTuple` produced by
`Zoo` problem constructors: `mgb_solve(; problem...)`.
"""
function mgb_solve(; mg::MultiGrid, kwargs...)
    mgb_solve(mg; kwargs...)
end

