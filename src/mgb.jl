# The MGB V-cycle: mgb_step, mgb_core, mgb_driver, MGBSOL, and mgb_solve.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

# Internal marker for "skip the finalize pass". The public `finalize` keyword
# accepts either a stopping criterion or `false`; `false` is normalized to
# `NoFinalize()` on entry (mgb_driver) so the internal plumbing carries a single
# non-Bool convention.
struct NoFinalize end

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
        initial_step=false,
        args...
        ) where {T,X,W,M_sub}
    L = length(M.R_fine)
    B = barrier(Q)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D_fine
    function eta(j,J,sc,maxit,ls)
        @mgblog("j=",j," J=",J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = mgb_zeros(W, size(R, 2))
        # Snapshot: `z` is reassigned below, so capturing it directly would box
        # it and untype every objective/gradient/Hessian call in the hot loop.
        zJ = z
        SOL = newton(M_sub,T,
            s->f0(s,w,c,R,D,zJ),
            s->f1(s,w,c,R,D,zJ),
            s->f2(s,w,c,R,D,zJ),
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
    # Newton-iteration cap per bounded attempt. A multi-level jump (J-j>1) is
    # capped at `max_newton` so a failure triggers divide-and-conquer bisection.
    # For the *initial step*, a single-level transfer (J-j==1) is guaranteed by
    # the analysis (Thm p1:cost) to converge in O(1) Newton iterations, but that
    # O(1) can exceed `max_newton` (the cap budgets only the loglog(eps) quadratic
    # phase, not the damped phase); since there is no finer level to bisect to, we
    # let it run to the global `maxit`. Generic steps keep `max_newton` throughout
    # (the kappa adaptation, not extra Newton iterations, handles their failures).
    mn(j,J) = (initial_step && J-j==1) ? maxit : max_newton
    converged = divide_and_conquer((j,J)->eta(j,J,stopping_criterion,mn(j,J),line_search),0,L)
    z_unfinalized = z
    if !(finalize isa NoFinalize)
        @mgblog("finalize")
        foo = eta(L-1,L,finalize,maxit,line_search)
        converged = converged && foo
    end
    @mgblog("converged=",converged)
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
    SOL = mgb_step(Q,M,z,t*c;max_newton,early_stop,maxit,printlog,finalize=NoFinalize(),initial_step=true,args...)
    @mgblog("initial centering done")
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    z_unfinalized = z
    Dz = apply_D(M.D_fine, z)
    c_dot_Dz[k] = sum(dot(M.w .* c[:,j], Dz[:,j]) for j = 1:length(M.D_fine))
    while t<=1/tol && kappa > 1 && k<maxit && !early_stop(z)
        k = k+1
        its[:,k] .= 0
        times[k] = time()
        prog = ((log(t)-log(tinit))/(log(1/tol)-log(tinit)))
        progress(prog)
        while kappa > 1
            t1 = kappa*t
            @mgblog("k=",k," t=",t," kappa=",kappa," t1=",t1)
            fin = (t1>1/tol) ? finalize : NoFinalize()
            SOL = mgb_step(Q,M,z,t1*c;
                max_newton,early_stop,maxit,printlog,finalize=fin,args...)
            its[:,k] += SOL.its
            if SOL.converged
                if maximum(SOL.its)<=max_newton*0.5
                    @mgblog("increasing t step size?")
                    kappa = min(kappa0,kappa^2)
                end
                z = SOL.z
                z_unfinalized = SOL.z_unfinalized
                t = t1
                break
            end
            @mgblog("t refinement failed, shrinking kappa")
            kappa = sqrt(kappa)
        end
        ts[k] = t
        kappas[k] = kappa
        Dz = apply_D(M.D_fine, z)
        c_dot_Dz[k] = sum(dot(M.w .* c[:,j], Dz[:,j]) for j = 1:length(M.D_fine))
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        # Two distinct failure exits from the t-ramp: the kappa refinement
        # collapsed (no t-step succeeds anymore: a stall), or the outer
        # iteration cap was reached first.
        code = kappa <= 1 ? :stall : :iteration_limit
        throw(MGBConvergenceFailure("Convergence failure in mgb_solve at t=$t, k=$k, kappa=$kappa, tol=$tol, maxit=$maxit.",code))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    @mgblog("success. t=",t," tol=",tol)
    return (;z,z_unfinalized,c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],
            t_begin,t_end,t_elapsed,times=times[1:k],
            c_dot_Dz=c_dot_Dz[1:k])
end

# Slice the leading NC entries of yy: the `(user D rows..., slack)` part that
# the problem's cobarrier consumes. GPU-compatible (static sizes only).
@inline _feas_head(yy::SVector{NF,TT}, ::Val{NC}) where {NF,TT,NC} =
    SVector{NC,TT}(ntuple(i -> @inbounds(yy[i]), Val(NC)))

@doc raw"""
    _feasibility_convex(Q::Convex{T}, b::T, R::T, ::Val{NC}) -> Convex{T}

Build the phase-I (feasibility) barrier for `Q`, restricted to a bounding box
of radius `R`.

The feasibility operator list is `(user D rows..., slack, component values...)`
(see `_prepare_amg`), so each per-node argument `yy` carries the cobarrier's
input `(D rows..., slack u)` in its leading `NC` entries and the state
components' nodal values `v_i` in the trailing entries. The barrier at each
node is

    cobarrier(yy[1:NC]) - log(b-u) - log(b+u) - Σ_i [ log(R-v_i) + log(R+v_i) ]

Every term is a self-concordant barrier, so the sum is a self-concordant
barrier (with parameter grown by 2 per boxed entry) for the *bounded* domain:
the relaxed constraint set intersected with `|v_i| < R` and `|u| < b`. On a
bounded domain the phase-I minimizer and the central path exist, as the
concrete path-following theory requires. A plain quadratic regularizer (the
previous `dot(yy,yy)` term) is self-concordant but not a barrier of finite
parameter, and its restraint is fixed while the t-ramp's pull grows, so
iterates drift to a t-dependent scale; a box wall does not move with t.

The factored form `log(R-v) + log(R+v)` is used instead of `log(R²-v²)`
because the latter cancels catastrophically near the walls, which is exactly
where the barrier must be accurate.
"""
function _feasibility_convex(Q::Convex{T}, b::T, R::T, ncval::Val{NC}) where {T,NC}
    cobar_f0 = Q.cobarrier[1]
    cobar_f1 = Q.cobarrier[2]
    cobar_f2 = Q.cobarrier[3]
    # GPU-compatible: barriers receive (args_rows..., yy); captures are isbits
    # scalars, a Val, and the cobarrier callables.
    function feas_f0(args_and_y::Vararg{Any,M}) where M
        yy = args_and_y[M]
        args_j = args_and_y[1:M-1]
        TT = eltype(yy)
        NF = length(yy)
        bb = TT(b)
        RR = TT(R)
        yc = _feas_head(yy, ncval)
        u = yc[NC]
        ret = cobar_f0(args_j..., yc) - Log(bb-u) - Log(bb+u)
        ret + sum(ntuple(Val(NF-NC)) do i
            v = @inbounds yy[NC+i]
            -Log(RR-v) - Log(RR+v)
        end)
    end
    function feas_f1(args_and_y::Vararg{Any,M}) where M
        yy = args_and_y[M]
        args_j = args_and_y[1:M-1]
        TT = eltype(yy)
        NF = length(yy)
        bb = TT(b)
        RR = TT(R)
        yc = _feas_head(yy, ncval)
        u = yc[NC]
        gc = cobar_f1(args_j..., yc)
        gs = inv(bb-u) - inv(bb+u)
        SVector{NF,TT}(ntuple(Val(NF)) do i
            if i < NC
                @inbounds gc[i]
            elseif i == NC
                gc[NC] + gs
            else
                v = @inbounds yy[i]
                inv(RR-v) - inv(RR+v)
            end
        end)
    end
    function feas_f2(args_and_y::Vararg{Any,M}) where M
        yy = args_and_y[M]
        args_j = args_and_y[1:M-1]
        TT = eltype(yy)
        NF = length(yy)
        bb = TT(b)
        RR = TT(R)
        yc = _feas_head(yy, ncval)
        u = yc[NC]
        Hc = cobar_f2(args_j..., yc)
        hs = inv((bb-u)^2) + inv((bb+u)^2)
        H = SMatrix{NF,NF,TT}(ntuple(Val(NF*NF)) do k
            i = (k-1) % NF + 1
            j = (k-1) ÷ NF + 1
            if i <= NC && j <= NC
                val = @inbounds Hc[(j-1)*NC + i]
                (i == NC && j == NC) ? val + hs : val
            elseif i == j
                v = @inbounds yy[i]
                inv((RR-v)^2) + inv((RR+v)^2)
            else
                zero(TT)
            end
        end)
        SVector(H)
    end
    Convex{T}((feas_f0,feas_f1,feas_f2),(feas_f0,feas_f1,feas_f2),Q.slack,Q.args)
end

function mgb_driver(M::Tuple{AMG{X,W,M_sub,<:Any,<:Any},AMG{X,W,M_sub,<:Any,<:Any}},
              f::X,
              g::X,
              Q::Convex{T};
              t=T(0.1),
              t_feasibility=t,
              # Cap on the phase-I bounding box (see `_feasibility_convex` and the
              # R-escalation loop below). Beyond ~1/√eps the box barrier is
              # numerically inert in the interior, where it must restrain the
              # drift toward infinity: its curvature ~1/R² falls below roundoff
              # of the O(1) Hessian rows, and the solve degenerates to the
              # unbounded-domain failure the box exists to prevent. Raise it only
              # for problems whose feasible points genuinely have huge nodal
              # values (better: rescale the problem).
              feasibility_Rmax=one(T)/sqrt(eps(T)),
              progress = x->nothing,
              # Newton-decrement (central-path) stopping tolerance.
              # The flat-averaged barrier (1/n)Σ F has self-concordance *constant* √n:
              # the summed barrier Σ F is standard self-concordant (constant 1), and
              # scaling by 1/n multiplies the constant by (1/n)^{-1/2} = √n. With the
              # sqrt decrement λ = √(gᵀH⁻¹g), standardizing (×n, which recovers Σ F)
              # gives λ_Σ = √n·λ, so the standard criterion λ_Σ < η (η ≤ 1/4) becomes
              #     λ < η/√n,     here η = 1/4.
              # (The reverse-Hölder constant C_RH ~ n is a *looser* s.c.-constant
              # bound, via L^∞/L^1 instead of L^∞/L^2; using it would give the
              # over-conservative 1/n.)  The second argument is the gradient-stagnation
              # factor of the roundoff fallback (stopping_exact); keep it near 1 so the
              # decrement criterion, not the fallback, is the binding stop.
              stopping_criterion=stopping_inexact(T(0.25)/sqrt(T(length(M[1].w))),T(0.9)),
              printlog = (args...)->nothing,
              line_search=linesearch_backtracking(T),
              finalize=stopping_exact(T(0.9)),
              rest...) where {T,X,W,M_sub}
    # Public API: `finalize = false` means "skip the finalize pass". Normalize
    # to the internal NoFinalize marker so downstream code never mixes Bools
    # and stopping criteria.
    finalize === false && (finalize = NoFinalize())
    m = size(M[1].x,1)
    nD = length(M[1].D_fine)
    c0 = f
    z0 = g
    Z = mgb_zeros(M[1].D_fine[1],m,m)
    # GPU-compatible: use `one` instead of closure capturing T
    ONES = map_rows_gpu(one, M[1].w)
    II = mgb_diag(M[1].D_fine[1],ONES)
    # RR2[k] is the m × (m*c) selection matrix [0 … I … 0] picking component k
    # out of the stacked solution vector; `RR2[k]*SOL_main.z` below reimplements
    # `reshape(SOL_main.z, m, :)[:, k]`, and the analogous WW in the feasibility
    # branch reimplements a `maximum` over a view of the last block. This matrix
    # formulation is deliberate, not vestigial: backends only have to provide
    # matvec/hcat/vcat/getindex(:,k)/maximum. The distributed MPI backend
    # (HPCSparseArrays' row-partitioned VectorMPI/MatrixMPI) defines exactly
    # those and has NO reshape/vec/view — and a reshape of a row-partitioned
    # stacked vector would not be a rank-local operation anyway, whereas these
    # matvecs go through the backend's communication plans. Do not "simplify"
    # to reshape/vec/views; that breaks every backend that is not
    # view-friendly. (On CuArray reshape/vec happen to work; irrelevant here.)
    RR2 = map(1:size(z0,2)) do k
        foo = fill(Z,size(z0,2))
        foo[k] = II
        hcat(foo...)
    end
    z2 = vcat([z0[:,k] for k=1:size(z0,2)]...)
    w = hcat([M[1].D_fine[k]*z2 for k=1:nD]...)
    pbarfeas = 0.0
    SOL_feasibility=nothing
    Q_L = Q
    barrier_f0_fn = Q_L.barrier[1]
    slack_fn = Q_L.slack
    args_L = Q_L.args  # Args for finest level, splatted to map_rows_gpu
    try
        # GPU-compatible: splat Q.args to map_rows_gpu
        foo = map_rows_gpu(barrier_f0_fn, args_L..., w)
        mgb_all_isfinite(foo) || error("initial point is on or outside the barrier domain")
    catch e
        # Broad on purpose: there is no fixed protocol for a barrier to signal
        # that z0 escapes its domain (a DomainError from log, non-finite
        # values, an InexactError, complex results, ...), so any failure here
        # is treated as an infeasible start and routed to the feasibility
        # phase. Interrupts still abort.
        e isa InterruptException && rethrow()
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
        ncomp = size(z0,2)
        # Phase-I cost: minimize ∫slack. The feasibility operator list is
        # (user D rows..., slack id, component id rows...) — see _prepare_amg —
        # so the cost selects row nD+1.
        foo = zeros(T,(nD+1+ncomp,)); foo[nD+1] = 1
        foo_sv = SVector(Tuple(foo))
        c1 = map_rows_gpu(k->foo_sv, M[1].w)

        z1 = vcat([z1[:,k] for k=1:size(z1,2)]...)
        foo = fill(Z,ncomp+1)
        foo[end] = II
        WW = hcat(foo...)
        early_stop(z) = (maximum(WW*z)<0)
        # Per-component nodal-value selectors, same matvec-only style as WW/RR2
        # (see the backend comment above RR2).
        VV = map(1:ncomp) do k
            sel = fill(Z,ncomp+1)
            sel[k] = II
            hcat(sel...)
        end
        # Phase I minimizes ∫slack over the constraint relaxation intersected
        # with the box |values| < Rbox (see _feasibility_convex: without the
        # box, an unbounded domain makes the barrier unbounded below and the
        # t-ramp chases the false minimum at infinity). The box must contain
        # the initial point strictly; it is grown geometrically whenever the
        # phase-I minimizer presses against it.
        Rbox = max(T(10), T(10)*maximum(abs.(z2)))
        Rmax = max(T(feasibility_Rmax), Rbox)
        while true
            printlog("mgb_driver: feasibility phase with bounding box R=",Rbox)
            Q_feas = _feasibility_convex(Q_L, T(b), Rbox, Val(nD+1))
            failure = nothing
            try
                # `rest...` may carry a user-supplied `early_stop` meant for the main
                # problem; the rightmost duplicate keyword wins, so the feasibility
                # phase's own strict-feasibility stop must come after `rest...`.
                SOL_feasibility = mgb_core(Q_feas,M[2],z1,c1;t=t_feasibility,
                    progress=x->progress(pbarfeas*x),
                    printlog,
                    stopping_criterion,
                    line_search,
                    finalize,
                    rest...,
                    early_stop)
            catch e2
                # Broad on purpose, like the line-search trial rejection and the
                # infeasible-start routing above: a wall-pressed phase-I ramp can
                # die in many ways (MGBConvergenceFailure, a SingularException
                # from the Newton solve at huge t, a non-finite barrier value,
                # ...). Each round is a probe; any numerical death is answered
                # by growing the box, and the terminal error below reports the
                # last underlying failure. Interrupts still abort.
                e2 isa InterruptException && rethrow()
                failure = e2
            end
            if failure === nothing
                early_stop(SOL_feasibility.z) && break  # strictly feasible point found
                # The t-ramp completed without reaching strict feasibility, so
                # SOL_feasibility.z approximates the phase-I minimizer over the
                # boxed domain. Activity test: a minimizer well inside the box
                # is (up to tolerance) an interior minimizer of the unrestricted
                # phase-I problem — growing the box cannot lower it further, so
                # the positive violation certifies infeasibility. Only a
                # minimizer pressing the walls warrants a larger box.
                zf = SOL_feasibility.z
                vmax = maximum(map(V->maximum(abs.(V*zf)),VV))
                smax = maximum(WW*zf)
                if vmax <= Rbox/2
                    throw(MGBConvergenceFailure(
                        "The problem appears to be infeasible: the feasibility subproblem "*
                        "converged to a minimizer with positive constraint violation "*
                        "(max slack ≈ $smax) strictly inside the bounding box "*
                        "(max |nodal value| ≈ $vmax ≤ R/2 with R = $Rbox).",
                        :infeasible))
                end
                printlog("mgb_driver: phase-I minimizer presses the box (max |nodal value|=",
                    vmax,", max slack=",smax,"); growing R")
                # No warm start: a completed (wall-pressed) ramp terminates within
                # ~1/t of the old box wall AND of the relaxed-constraint boundary,
                # where the barrier Hessian carries 1/slack² ~ t² entries; seeding
                # the next round there makes its Newton systems numerically
                # singular. Every round restarts from the pristine z1, which is
                # strictly interior for all rounds since the box only grows.
            else
                printlog("mgb_driver: feasibility solve failed at R=",Rbox,": ",
                    sprint(showerror,failure))
            end
            Rnext = 10*Rbox
            if Rnext > Rmax
                reason = failure === nothing ?
                    "the phase-I minimizer still presses against the bounding box" :
                    "the last attempt failed with: "*sprint(showerror,failure)
                throw(MGBConvergenceFailure(
                    "Could not find a strictly feasible point with nodal values bounded "*
                    "by R = $Rbox (cap feasibility_Rmax ≈ $Rmax); "*reason*". "*
                    "The problem is infeasible, or its feasible points have nodal values "*
                    "exceeding the cap (rescale the problem, or raise feasibility_Rmax).",
                    :feasibility_Rmax))
            end
            Rbox = Rnext
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
default_f(::Val{1}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(::Val{2}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(::Val{3}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(k::Int) = default_f(Val(k))
default_g(::Val{1}) = (x)->SVector(x[1], oftype(x[1],2))
default_g(::Val{2}) = (x)->SVector(x[1]^2+x[2]^2, oftype(x[1],100))
default_g(::Val{3}) = (x)->SVector(x[1]^2+x[2]^2+x[3]^2, oftype(x[1],100))
default_g(k::Int) = default_g(Val(k))
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
    MGBSOL{T,X,W,Discretization,G,SF,SM}

Solution object returned by `mgb_solve` and `parabolic_solve`.

# Type Parameters
- `T`: scalar numeric type
- `X`: solution/point matrix type (e.g. `Matrix{T}`, `CuMatrix{T}`)
- `W`: weight vector type
- `Discretization`: geometry descriptor (e.g. `FEM2D_P2{T}`, `SPECTRAL1D{T}`)
- `G`: full `Geometry` type
- `SF`, `SM`: types of the feasibility/main diagnostics (`Nothing` or a NamedTuple)

# Fields
- `z::X`: solution matrix of size `(n_nodes, n_components)`
- `SOL_feasibility`: feasibility phase diagnostics (`nothing` if initial point was feasible)
- `SOL_main`: main optimization phase diagnostics (NamedTuple)
- `log::String`: detailed iteration log
- `geometry::G`: the input `Geometry`

Supports `plot(sol)` to visualize the first solution component.
"""
struct MGBSOL{T,X,W,Discretization,G,SF,SM}
    z::X
    SOL_feasibility::SF
    SOL_main::SM
    log::String
    geometry::G
end
function MGBSOL(z::X, sf, sm, log::String, geometry::Geometry{T,<:Any,W,<:Any,Discretization}) where {T,X,W,Discretization}
    MGBSOL{T,X,W,Discretization,typeof(geometry),typeof(sf),typeof(sm)}(z, sf, sm, log, geometry)
end
# plot(sol::MGBSOL, k) lives in MultiGridBarrierPyPlotExt.

"""
    MGBProblem{T,MT,FT,GT,QT,GeoT}

A fully assembled, array-valued, **closure-free** convex problem ready for the MGB
solver. Produced by `assemble`; consumed by `mgb_solve`. Every closure that
specifies the problem (`f`, `g`, and the constraint data `A`, `b`, `p`, `select`)
has been lowered to per-vertex grids during assembly, so an `MGBProblem` is pure
data: moving it between backends (`native_to_device`) is plain array movement, and
a `MGBProblem{T,...,CuArray,...}` differs from its CPU sibling only by array type.

# Fields
- `M`: the `(main, feasibility)` AMG hierarchy pair (`_prepare_amg` output).
- `f`: linear-term grid (`f_grid`).
- `g`: Dirichlet/initial-data grid (`g_grid`).
- `Q::Convex`: the convex constraint (its `args` are per-vertex grids).
- `geometry`: the fine-level `Geometry` (for the returned solution / plotting).
"""
struct MGBProblem{T,MT,FT,GT,QT,GeoT}
    M::MT
    f::FT
    g::GT
    Q::QT
    geometry::GeoT
end
MGBProblem{T}(M, f, g, Q, geometry) where {T} =
    MGBProblem{T,typeof(M),typeof(f),typeof(g),typeof(Q),typeof(geometry)}(M, f, g, Q, geometry)

"""
    assemble(mg::MultiGrid{T}; kwargs...) -> MGBProblem

Lower a problem specification to a closure-free, **CPU** `MGBProblem`. This is the single
place where the problem closures are evaluated to grids: `f`/`g` via `map_rows`, and the
constraint closures inside `Q` at its construction. The result is a backend-agnostic,
native data structure; to solve on a GPU, hand it to `mgb_solve(prob; device=CUDADevice)`,
which moves it to the device and the solution back.

The five `MGBProblem` fields are built from `mg` and the keyword arguments as:

| field      | value                                                                 |
|:-----------|:----------------------------------------------------------------------|
| `M`        | `_prepare_amg(mg; state_variables, D)` — the `(main, feasibility)` AMG pair |
| `f`        | `f_grid`, default `map_rows(f, x)` — the linear-term closure sampled at `x` |
| `g`        | `g_grid`, default `map_rows(g, x)` — the Dirichlet/initial-data closure at `x` |
| `Q`        | `Q`, default `convex_Euclidian_power(T; mg, idx=default_idx(dim), p=xi->p)` |
| `geometry` | `mg.geometry` — the fine-level `Geometry`                              |

# Keyword Arguments
- `dim::Integer = amg_dim(mg.geometry.discretization)`: spatial dimension; auto-detected.
- `state_variables = [:u :dirichlet; :s :full]`, `D = default_D(dim)`: solution components /
  function spaces and the differential operators applied to them; together they define `M`.
- `x = _xflat(mg)`: sample points where `f`/`g` are evaluated, one row per node.
- `p::T = T(1.0)`: p-Laplace exponent for the default `Q`.
- `f`/`f_grid`, `g`/`g_grid`: forcing and boundary/initial data, as closures (lowered via
  `map_rows`) or as pre-built grids.
- `Q::Convex{T}`: convex constraint; defaults to a p-Laplace power-cone barrier.
- `M`: supply an AMG hierarchy pair directly instead of building it from `mg`.

Any extra (solver-control) keywords are accepted and ignored.
"""
function assemble(mg::MultiGrid{T};
        dim::Integer = amg_dim(mg.geometry.discretization),
        state_variables = [:u :dirichlet ; :s :full],
        D = default_D(dim),
        x = _xflat(mg),
        p::T = T(1.0),
        g::Function = default_g(dim),
        f::Function = default_f(dim),
        g_grid = map_rows(xi->SVector(Tuple(g(xi))),x),
        f_grid = map_rows(xi->SVector(Tuple(f(xi))),x),
        Q::Convex{T} = convex_Euclidian_power(T; mg=mg, idx=default_idx(dim), p=xi->p),
        M = _prepare_amg(mg;state_variables,D),
        rest...) where {T}
    MGBProblem{T}(M, f_grid, g_grid, Q, mg.geometry)
end

"""
    mgb_solve(prob::MGBProblem; device=default_device(), kwargs...) -> MGBSOL

MultiGrid Barrier (MGB) solver for nonlinear convex optimization problems on a multigrid
hierarchy. Operates in a feasibility phase followed by a main optimization phase, with
damped Newton inner solves and line search.

Solves an assembled [`MGBProblem`](@ref) — build one with [`assemble`](@ref) (where the
problem-specification keywords `p`, `f`, `g`, `Q`, `state_variables`, `D`, … live), or take
one from the [`Zoo`](@ref). The problem is moved to `device`, solved there, and the solution
moved back, so the returned `MGBSOL` is always in native CPU types regardless of `device`.

# Keyword Arguments

## Backend
- `device::Type{<:Device} = default_device()`: compute backend, `CPUDevice` (default) or
  `CUDADevice` (requires `using CUDA, CUDSS_jll`). The problem is moved to the device,
  solved there, and the solution moved back; the returned `MGBSOL` is always native.

## Output Control
- `verbose::Bool = true`: progress bar.
- `logfile = devnull`: optional log stream.

## Solver Control (forwarded to `mgb_driver`)
- `tol`, `t`, `t_feasibility`, `feasibility_Rmax`, `maxit`, `kappa`, `early_stop`,
  `max_newton`, `stopping_criterion`, `line_search`, `finalize`, `progress`, `printlog`.

## Feasibility phase
If the initial point is infeasible, a phase-I subproblem minimizes the integral of a
slack variable inside a bounding box `|nodal values| < R` (the box keeps the phase-I
domain bounded, so its barrier has a minimizer; without it, iterates can drift to
infinity). `R` starts at `max(10, 10·max|g|)` and is grown tenfold whenever the
phase-I minimizer presses against the box, up to `feasibility_Rmax`
(default `1/√eps(T)`). If the phase-I minimizer has positive violation but sits
strictly inside the box, the problem is reported infeasible via
`MGBConvergenceFailure` (for a convex problem, an interior phase-I minimizer is
global, so a larger box cannot help).

# Returns
An `MGBSOL` whose `z` is the fine-level solution matrix and whose `geometry` is the
fine-level `Geometry` (the `MultiGrid` itself is not stored).

# Examples
```julia
sol = mgb_solve(assemble(amg(fem1d(; nodes = collect(range(-1.0, 1.0, length=33)))); p = 1.5))
sol = mgb_solve(assemble(amg(subdivide(fem2d_P2(), 3)); p = 1.5))
sol = mgb_solve(assemble(amg(spectral2d(n = 8)); p = 2.0); device = CPUDevice)
sol = mgb_solve(Zoo.p_harmonic(amg(fem2d_P2()); p = 1.5); tol = 1e-4)
```
"""
function mgb_solve(prob::MGBProblem{T};
        device::Type{<:Device} = default_device(),
        verbose=true,
        logfile=devnull,
        rest...) where {T}
    # Move the assembled (CPU) problem to the requested backend (identity on CPU),
    # solve there, and move the solution back to native types.
    prob = native_to_device(device, prob)
    progress = x->nothing
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
    SOL = try
        mgb_driver(prob.M, prob.f, prob.g, prob.Q; progress, printlog, rest...)
    catch
        # The success path flushes backend caches via mgb_cleanup(sol) below; a
        # throwing solve (MGBConvergenceFailure, interrupt, ...) must flush them
        # too, or cached plans and factorizations stay resident until the next
        # successful solve. There is no MGBSOL here, so dispatch on the device.
        mgb_cleanup(device)
        rethrow()
    end
    sol = mgb_cleanup(MGBSOL(SOL.z, SOL.SOL_feasibility, SOL.SOL_main, String(take!(log_buffer)), prob.geometry))
    return device_to_native(device, sol)
end

