export amgb, Geometry, Convex, convex_linear, convex_Euclidian_power, AMGBConvergenceFailure, apply_D, linesearch_illinois, linesearch_backtracking, stopping_exact, stopping_inexact, interpolate, intersect, plot


@doc raw"""
    interpolate(M::Geometry, z::Vector, t)

Interpolate a solution vector at specified points.

Given a solution `z` on the mesh `M`, evaluates the solution at new points `t`
using the appropriate interpolation method for the discretization.

Supported discretizations
- 1D FEM (`FEM1D`): piecewise-linear interpolation
- 1D spectral (`SPECTRAL1D`): spectral polynomial interpolation
- 2D spectral (`SPECTRAL2D`): tensor-product spectral interpolation

Note: 2D FEM interpolation is not currently provided.

# Arguments
- `M::Geometry`: The geometry containing grid and basis information
- `z::Vector`: Solution vector on the finest grid (length must match number of DOFs)
- `t`: Evaluation points. Format depends on dimension:
  - 1D: scalar or `Vector{T}` of x-coordinates
  - 2D spectral: `Matrix{T}` where each row is `[x, y]`

# Returns
Interpolated values at the specified points. Shape matches input `t`.

# Examples
```julia
# 1D interpolation (FEM)
geom = fem1d(L=3)
z = sin.(π .* vec(geom.x))
y = interpolate(geom, z, 0.5)
y_vec = interpolate(geom, z, [-0.5, 0.0, 0.5])

# 2D interpolation (spectral)
geom = spectral2d(n=4)
z = exp.(-geom.x[:,1].^2 .- geom.x[:,2].^2)
points = [0.0 0.0; 0.5 0.5; -0.5 0.5]
vals = interpolate(geom, z, points)
```
""" interpolate

@doc raw"""
    plot(sol::AMGBSOL, k::Int=1; kwargs...)
    plot(sol::ParabolicSOL, k::Int=1; kwargs...)
    plot(M::Geometry, z::Vector; kwargs...)
    plot(M::Geometry, ts::AbstractVector, U::Matrix; frame_time=..., embed_limit=..., printer=...)

Visualize solutions and time sequences on meshes.

- 1D problems: Line plot. For spectral methods, you can specify evaluation points with `x=-1:0.01:1`.
- 2D FEM: Triangulated surface plot using the mesh structure.
- 2D spectral: 3D surface plot. You can specify evaluation grids with `x=-1:0.01:1, y=-1:0.01:1`.

Time sequences (animation):
- Call `plot(M, ts, U; frame_time=1/30, printer=anim->nothing)` where `U` has columns as frames and `ts` are absolute times in seconds (non-uniform allowed).
- Or simply call `plot(sol)` where `sol` is a `ParabolicSOL` returned by `parabolic_solve` (uses `sol.ts`).
- Animation advances at a fixed frame rate given by `frame_time` (seconds per video frame). For irregular `ts`, each video frame shows the latest data frame with timestamp ≤ current video time.
- The `printer` callback receives the Matplotlib animation object; use it to display or save (e.g., `anim.save("out.mp4")`).
- `embed_limit` controls the maximum embedded HTML5 video size in megabytes.

When `sol` is a solution object returned by `amgb` or the `*_solve` helpers, `plot(sol,k)` plots
the kth component `sol.z[:, k]` using `sol.geometry`. `plot(sol)` uses the default k=1.

All other keyword arguments are passed to the underlying `PyPlot` functions.
""" plot

amgb_zeros(::SparseMatrixCSC{T,Int}, m,n) where {T} = spzeros(T,m,n)
amgb_zeros(::LinearAlgebra.Adjoint{T, SparseArrays.SparseMatrixCSC{T, Int64}},m,n) where {T} = spzeros(T,m,n)
amgb_zeros(::Matrix{T}, m,n) where {T} = zeros(T,m,n)
amgb_zeros(::LinearAlgebra.Adjoint{T, Matrix{T}},m,n) where {T} = zeros(T,m,n)
amgb_all_isfinite(z::Vector{T}) where {T} = all(isfinite.(z))

amgb_hcat(A...) = hcat(A...)

amgb_diag(::SparseMatrixCSC{T,Int}, z::Vector{T},m=length(z),n=length(z)) where {T} = spdiagm(m,n,0=>z)
amgb_diag(::Matrix{T}, z::Vector{T},m=length(z),n=length(z)) where {T} = diagm(m,n,0=>z)

amgb_blockdiag(args::SparseMatrixCSC{T,Int}...) where {T} = blockdiag(args...)
amgb_blockdiag(args::Matrix{T}...) where {T} = Matrix{T}(blockdiag((sparse(args[k]) for k=1:length(args))...))

# makes a matrix (if f returns adjoint vectors) or a vector (if f returns scalars)
map_rows(f,A...) = vcat((f.((eachrow.(A))...))...)

# Type-stable linear solver
solve(A, b) = A \ b

macro debug(args...)
    escargs = map(esc, args)
    return :($(esc(:printlog))(nameof($(esc(:(var"#self#")))), ":", $(escargs...)))
end

"""
    AMGBConvergenceFailure <: Exception

Thrown when the AMGB solver fails to converge (feasibility or main phase).
Includes a descriptive message about the failure.
"""
struct AMGBConvergenceFailure <: Exception
    message
end

Base.showerror(io::IO, e::AMGBConvergenceFailure) = print(io, "AMGBConvergenceFailure:\n", e.message)

@kwdef struct Barrier
    f0::Function
    f1::Function
    f2::Function
end

"""
    Geometry{T,X,W,M,Discretization}

Container for discretization geometry and the multigrid transfer machinery used by AMGB.

Constructed by high-level front-ends like `fem1d`, `fem2d`, `spectral1d`, and `spectral2d`. It
collects the physical/sample points, quadrature weights, per-level subspace embeddings, discrete
operators (e.g. identity and derivatives), and intergrid transfer operators (refine/coarsen).

Type parameters
- `T`: scalar numeric type (e.g. Float64)
- `X`: type of the point storage for `x` (typically `Matrix{T}`; may be any AbstractArray with size (n_nodes, dim))
- `W`: type of the weight storage for `w` (typically `Vector{T}`; may be any AbstractVector)
- `M`: matrix type used for linear operators (e.g. `SparseMatrixCSC{T,Int}` or `Matrix{T}`)
- `Discretization`: front-end descriptor (e.g. `FEM1D{T}`, `FEM2D{T}`, `SPECTRAL1D{T}`, `SPECTRAL2D{T}`)

Fields
- `discretization::Discretization`: Discretization descriptor that encodes dimension and grid construction
- `x::X`: Sample/mesh points on the finest level; size is (n_nodes, dim)
- `w::W`: Quadrature weights matching `x` (length n_nodes)
- `subspaces::Dict{Symbol,Vector{M}}`: Per-level selection/embedding matrices for function spaces
  (keys commonly include `:dirichlet`, `:full`, `:uniform`). Each value is a vector of length L
  with one matrix per level.
- `operators::Dict{Symbol,M}`: Discrete operators defined on the finest level (e.g. `:id`, `:dx`, `:dy`).
  Operators at other levels are obtained via `coarsen_fine * operator * refine_fine` inside `amg`.
- `refine::Vector{M}`: Level-to-level refinement (prolongation) matrices for the primary state space
- `coarsen::Vector{M}`: Level-to-level coarsening (restriction) matrices for the primary state space

Notes
- `Geometry` is consumed by `amg` to build an `AMG` hierarchy and by utilities like `interpolate` and `plot`.
- The length of `refine`/`coarsen` equals the number of levels L; the last entry is typically the identity.
"""
struct Geometry{T,X,W,M,Discretization}
    discretization::Discretization
    x::X
    w::W
    subspaces::Dict{Symbol,Vector{M}}
    operators::Dict{Symbol,M}
    refine::Vector{M}
    coarsen::Vector{M}
end

@kwdef struct AMG{T,X,W,M,Discretization}
    geometry::Geometry{T,X,W,M,Discretization}
    x::X
    w::W
    R_fine::Array{M,1}
    R_coarse::Array{M,1}
    D::Array{M,2}
    refine_u::Array{M,1}
    coarsen_u::Array{M,1}
    refine_z::Array{M,1}
    coarsen_z::Array{M,1}
end

function amg_helper(geometry::Geometry{T,X,W,M,Discretization},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol}) where {T,X,W,M,Discretization}
    x = geometry.x
    w = geometry.w
    subspaces = geometry.subspaces
    operators = geometry.operators
    refine = geometry.refine
    coarsen = geometry.coarsen
    L = length(refine)
    @assert size(w) == (size(x)[1],) && size(refine)==(L,) && size(coarsen)==(L,)
    for l=1:L
        @assert norm(coarsen[l]*refine[l]-I)<sqrt(eps(T))
    end
    refine_fine = Array{M,1}(undef,(L,))
    refine_fine[L] = refine[L]
    coarsen_fine = Array{M,1}(undef,(L,))
    coarsen_fine[L] = geometry.coarsen[L]
    for l=L-1:-1:1
        refine_fine[l] = refine_fine[l+1]*refine[l]
        coarsen_fine[l] = coarsen[l]*coarsen_fine[l+1]
    end
    R_coarse = Array{M,1}(undef,(L,))
    R_fine = Array{M,1}(undef,(L,))
    nu = size(state_variables)[1]
    @assert size(state_variables)[2] == 2
    for l=1:L
        R_coarse[l] = amgb_blockdiag((subspaces[state_variables[k,2]][l] for k=1:nu)...)
        R_fine[l] = amgb_blockdiag((refine_fine[l]*subspaces[state_variables[k,2]][l] for k=1:nu)...)
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
        Z = amgb_zeros(coarsen_fine[l],n,n)
        for k=1:nD
            foo = [Z for j=1:nu]
            foo[bar[D[k,1]]] = coarsen_fine[l]*operators[D[k,2]]*refine_fine[l]
            D0[l,k] = amgb_hcat(foo...)
        end
    end
    refine_z = [amgb_blockdiag([refine[l] for k=1:nu]...) for l=1:L]
    coarsen_z = [amgb_blockdiag([coarsen[l] for k=1:nu]...) for l=1:L]
    AMG{T,X,W,M,Discretization}(geometry=geometry,x=x,w=w,R_fine=R_fine,R_coarse=R_coarse,D=D0,
        refine_u=refine,coarsen_u=coarsen,refine_z=refine_z,coarsen_z=coarsen_z)
end

function amg(geometry::Geometry{T,X,W,M,Discretization};
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        full_space=:full,
        id_operator=:id,
        feasibility_slack=:feasibility_slack
        ) where {T,X,W,M,Discretization}                
    M1 = amg_helper(geometry,state_variables,D)
    s1 = vcat(state_variables,[feasibility_slack full_space])
    D1 = vcat(D,[feasibility_slack id_operator])
    M2 = amg_helper(geometry,s1,D1)
    return M1,M2
end

@doc raw"""
    Convex{T}

Container for a convex constraint set used by AMGB.

Fields:
- barrier(x, y): barrier of the set
- cobarrier(x, yhat): barrier with slack for feasibility
- slack(x, y): initial slack value

Construct via helpers like `convex_linear`, `convex_Euclidian_power`, `convex_piecewise`, or `intersect`.
"""
struct Convex{T}
    barrier::Function
    cobarrier::Function
    slack::Function
end

"""
    convex_linear(::Type{T}=Float64; idx=Colon(), A=(x)->I, b=(x)->T(0))

Create a convex set defined by linear inequality constraints.

Defines `F(x, y) = A(x) * y[idx] + b(x)`. The interior of the set is given by
`F(x, y) > 0` (a logarithmic barrier is applied to each component of `F`). The boundary
`F(x, y) = 0` corresponds to constraint activation.

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `idx=Colon()`: Indices of y to which constraints apply (default: all)
- `A::Function`: Matrix function `x -> A(x)` for constraint coefficients
- `b::Function`: Vector function `x -> b(x)` for constraint bounds

# Returns
`Convex{T}` object with appropriate barrier functions

# Examples
```julia
# Box constraints in 2D: -1 ≤ y ≤ 1
A_box(x) = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
b_box(x) = [1.0, 1.0, 1.0, 1.0]
Q = convex_linear(; A=A_box, b=b_box, idx=1:2)

# Single linear constraint: y[1] + 2*y[2] ≤ 3
# Choose F = 3 - (y1 + 2*y2) > 0
A_single(x) = [-1.0 -2.0]
b_single(x) = [3.0]
Q = convex_linear(; A=A_single, b=b_single, idx=1:2)
```
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
    convex_Euclidian_power(::Type{T}=Float64; idx=Colon(), A=(x)->I, b=(x)->T(0), p=x->T(2))

Create a convex set defined by Euclidean norm power constraints.

Constructs a `Convex{T}` object representing the power cone:
`{y : s ≥ ‖q‖₂^p}` where `[q; s] = A(x)*y[idx] + b(x)`

This is the fundamental constraint for p-Laplace problems where we need
`s ≥ ‖∇u‖^p` for some scalar field u.

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `idx=Colon()`: Indices of y to which transformation applies
- `A::Function`: Matrix function `x -> A(x)` for linear transformation
- `b::Function`: Vector function `x -> b(x)` for affine shift
- `p::Function`: Exponent function `x -> p(x)` where p(x) ≥ 1

# Returns
`Convex{T}` object with logarithmic barrier for the power cone

# Mathematical Details
The barrier function is:
- For p = 2: `-log(s² - ‖q‖²)`
- For p ≠ 2: `-log(s^(2/p) - ‖q‖²) - μ(p)*log(s)`
  where μ(p) = 0 if p∈{1,2}, 1 if p<2, 2 if p>2

# Examples
```julia
# Standard p-Laplace constraint: s ≥ ‖∇u‖^p
Q = convex_Euclidian_power(; idx=2:4, p=x->1.5)

# Spatially varying exponent
p_var(x) = 1.0 + 0.5 * x[1]  # variable p
Q = convex_Euclidian_power(; p=p_var)

# Second-order cone constraint: s ≥ ‖q‖₂
Q = convex_Euclidian_power(; p=x->1.0)
```
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

See also: [`intersect`](@ref), [`convex_linear`](@ref), [`convex_Euclidian_power`](@ref).
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
    intersect(U::Convex{T}, rest...) where {T}

Return the intersection of convex domains as a single `Convex{T}`.
Equivalent to `convex_piecewise` with all pieces active.
"""
intersect(U::Convex{T}, rest...) where {T} = convex_piecewise(T;Q=[U,rest...])

@doc raw"""    apply_D(D,z) = amgb_hcat([D[k]*z for k in 1:length(D)]...)"""
apply_D(D,z) = amgb_hcat([D[k]*z for k in 1:length(D)]...)

function barrier(F,::Type{T}=Float64;
        F1=(x,y)->ForwardDiff.gradient(z->F(x,z),y),
        F2=(x,y)->ForwardDiff.hessian(z->F(x,z),y))::Barrier where {T}
    function f0(z::W,x::X,w::W,c,R,D,z0) where {X,W}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = map_rows(F,x,Dz)
        dot(w,y)+sum(map_rows((w,c,Dz)->w*dot(c,Dz),w,c,Dz))
    end
    function f1(z::W,x::X,w::W,c::X,R,D,z0) where {X,W}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = map_rows((x,q,c)->(F1(x,q)+c)',x,Dz,c)
        m0 = size(D[1],2)
        ret = 0
        for k=1:n
            foo = D[k]'*(w.*y[:,k])
            if k>1
                ret += foo
            else
                ret = foo
            end
        end
        R'*ret
    end
    function f2(z::W,x::X,w::W,c,R::Mat,D,z0) where {X, W, Mat}
        Dz = apply_D(D,z0+R*z)
        p = length(w)
        n = length(D)
        y = map_rows((x,q)->F2(x,q)[:]',x,Dz)
        m0 = size(D[1],2)
        ret = D[1]
        for j=1:n
            foo = amgb_diag(D[1],w.*y[:,(j-1)*n+j])
            bar = (D[j])'*foo*D[j]
            if j>1
                ret += bar
            else
                ret = bar
            end
            for k=1:j-1
                foo = amgb_diag(D[1],w.*y[:,(j-1)*n+k])
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
        M::AMG{T,X,W,Mat,Geometry},
        x::X,
        z::W,
        c::X;
        maxit,
        max_newton,
        stopping_criterion,
        line_search,
        printlog,
        args...
        ) where {T,X,W,Mat,Geometry}
    @debug("start")
    L = length(M.R_fine)
    cm = Vector{X}(undef,L)
    cm[L] = c
    zm = Vector{W}(undef,L)
    zm[L] = z
    xm = Vector{X}(undef,L)
    xm[L] = x
    wm = Vector{W}(undef,L)
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
        s0 = map_rows(k->T(0),R')
        mi = if J-j==1 maxit else max_newton end
        SOL = newton(Mat,T,
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
                s0 = amgb_zeros(W,size(M.R_coarse[k])[2])
                y0 = f0(s0,xm[k],wm[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])::T
                y1 = f1(s0,xm[k],wm[k],cm[k],M.R_coarse[k],M.D[k,:],znext[k])
                @assert isfinite(y0) && amgb_all_isfinite(y1)
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
        M::AMG{T,X,W,Mat,Geometry},
        x::X,
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
        ) where {T,X,W,Mat,Geometry}
    L = length(M.R_fine)
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D[L,:]
    function eta(j,J,sc,maxit,ls)
        @debug("j=",j," J=",J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = map_rows(k->T(0),R')
        SOL = newton(Mat,T,
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
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
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
    function ls_illinois(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
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
                ynext,gnext = F0(xnext)::T, F1(xnext)
                @assert isfinite(ynext) && amgb_all_isfinite(gnext)
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
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
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
    function ls_backtracking(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
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
                ynext,gnext = F0(xnext)::T, F1(xnext)
                @assert isfinite(ynext) && amgb_all_isfinite(gnext)
                if ynext <= y - T(0.1)*inc*s
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

function newton(::Type{Mat}, ::Type{T},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::V;
                       maxit=10000,
                       stopping_criterion=nothing,
                       printlog,
                       line_search=nothing,
        ) where {T,Mat,V}
    stopping_criterion = stopping_criterion === nothing ? stopping_exact(T(0.1)) : stopping_criterion
    line_search = line_search === nothing ? linesearch_illinois(T) : line_search
    ss = T[]
    ys = T[]
    @assert amgb_all_isfinite(x)
    y = F0(x) ::T
    @assert isfinite(y)
    ymin = y
    push!(ys,y)
    converged = false
    k = 0
    g = F1(x)
    @assert amgb_all_isfinite(g)
    ynext,xnext,gnext=y,x,g
    gmin = norm(g)
    incmin = T(Inf)
    while k<maxit && !converged
        k+=1
        H = F2(x) ::Mat
        n = solve(H, g)
        @assert amgb_all_isfinite(n)
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

# Backward-compatible wrapper inferring T from x
function newton(::Type{Mat},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::V; kwargs...) where {Mat,V}
    T = eltype(x)
    return newton(Mat, T, F0, F1, F2, x; kwargs...)
end

function amgb_core(B::Barrier,
        M::AMG{T,X,W,Mat,Geometry},
        x::X,
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
        args...) where {T,X,W,Mat,Geometry}
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
    SOL = amgb_phase1(B,M,x,z,t*c;maxit,max_newton,printlog,args...)
    @debug("phase 1 success")
    passed = SOL.passed
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    z_unfinalized = z
    Dz = apply_D(M.D[end,:], z)
    c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:size(M.D,2)])
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
            SOL = amgb_step(B,M,x,z,t1*c;
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
        #c_dot_Dz[k] = dot(M.w .* c, apply_D(M.D[end,:], z))
        Dz = apply_D(M.D[end,:], z)
        c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:size(M.D,2)])
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        throw(AMGBConvergenceFailure("Convergence failure in amgb at t=$t, k=$k, kappa=$kappa."))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    @debug("success. t=",t," tol=",tol)
    return (;z,z_unfinalized,c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],
            t_begin,t_end,t_elapsed,times=times[1:k],
            passed,c_dot_Dz=c_dot_Dz[1:k])
end

function amgb_driver(M::Tuple{AMG{T,X,W,Mat,Geometry},AMG{T,X,W,Mat,Geometry}},
              f::X,
              g::X, 
              Q::Convex;
              x::X = M[1].x,
              t=T(0.1),
              t_feasibility=t,
              progress = x->nothing,
              stopping_criterion=stopping_inexact(sqrt(minimum(M[1].w))/2,T(0.5)),
              printlog = (args...)->nothing,
              line_search=linesearch_backtracking(T),
              finalize=stopping_exact(T(0.5)),
              rest...) where {T,X,W,Mat,Geometry}
    D0 = M[1].D[end,1]
    m = size(M[1].x,1)
    ns = Int(size(D0,2)/m)
    nD = size(M[1].D,2)
    c0 = f
    z0 = g
    Z = amgb_zeros(M[1].D[end,1],m,m)
    ONES = map_rows(x->T(1),M[1].D[end,1])
    II = amgb_diag(M[1].D[end,1],ONES)
    RR2 = []
    for k = 1:size(z0,2)
        foo = fill(Z,size(z0,2))
        foo[k] = II
        push!(RR2,amgb_hcat(foo...))
    end
    z2 = vcat([z0[:,k] for k=1:size(z0,2)]...)
    w = amgb_hcat([M[1].D[end,k]*z2 for k=1:nD]...)
    pbarfeas = 0.0
    SOL_feasibility=nothing
    try
        foo = map_rows(Q.barrier,x,w)
        @assert amgb_all_isfinite(foo)
    catch
        pbarfeas = 0.1
        z1 = map_rows((z0,x,w)->vcat(z0,2*max(Q.slack(x,w),1))',z0,x,w)
        b = 2*max(1,maximum(z1[:,end]))
        foo = zeros(T,(nD+1,)); foo[end] = 1
        c1 = map_rows(k->foo',x)
        B1 = barrier((x,y)->dot(y,y)+Q.cobarrier(x,y)-log(b^2-y[end]^2),T)
        z1 = vcat([z1[:,k] for k=1:size(z1,2)]...)
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
        ONES2 = map_rows(x->T(1),z2)
        II2 = amgb_diag(M[1].D[end,1],ONES2,size(ONES2,1),size(SOL_feasibility.z,1))
        z2 = II2*SOL_feasibility.z
    end
    B = barrier(Q.barrier,T)
    SOL_main = amgb_core(B,M[1],x,z2,c0;
        t,
        progress=x->progress((1-pbarfeas)*x+pbarfeas),
        printlog,
        stopping_criterion,
        line_search,
        finalize,
        rest...)
    z = amgb_hcat([RR2[k]*SOL_main.z for k=1:size(z0,2)]...)
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

struct AMGBSOL{T,X,W,Mat,Discretization}
    z::X
    SOL_feasibility
    SOL_main
    log::String
    geometry::Geometry{T,X,W,Mat,Discretization}
end
plot(sol::AMGBSOL{T,X,W,Mat,Discretization},k::Int=1;kwargs...) where {T,X,W,Mat,Discretization} = plot(sol.geometry,sol.z[:,k];kwargs...)

"""
    amgb(geometry::Geometry{T,X,W,Mat,Discretization}=fem1d(); kwargs...) where {T, X, W, Mat, Discretization}

Algebraic MultiGrid Barrier (AMGB) solver for nonlinear convex optimization problems
in function spaces using multigrid barrier methods.

This is the main high-level entry point for solving p-Laplace and related problems using
the barrier method with multigrid acceleration. The solver operates in two phases:
1. Feasibility phase: Finds an interior point for the constraint set (if needed)
2. Main optimization phase: Solves the barrier-augmented optimization problem

# Arguments

- `geometry`: Discretization geometry (default: `fem1d()`). Options:
  - `fem1d(L=n)`: 1D finite elements with 2^L elements
  - `fem2d(L=n, K=mesh)`: 2D finite elements
  - `spectral1d(n=m)`: 1D spectral with m nodes
  - `spectral2d(n=m)`: 2D spectral with m×m nodes

# Keyword Arguments

## Problem Specification
- `dim::Integer = amg_dim(geometry.discretization)`: Problem dimension (1 or 2), auto-detected from geometry
- `state_variables::Matrix{Symbol} = [:u :dirichlet; :s :full]`: Solution components and their function spaces
- `D::Matrix{Symbol} = default_D[dim]`: Differential operators to apply to state variables
- `x = geometry.x`: Mesh/sample points where `f` and `g` are evaluated when they are functions

## Problem Data
- `p::T = 1.0`: Exponent for p-Laplace operator (p ≥ 1)
- `g::Function = default_g(T)[dim]`: Boundary conditions/initial guess (function of spatial coordinates)
- `g_grid::X` (same container type as `x`): Alternative to `g`, directly provide values on grid (default: `g` evaluated at `x`)
- `f::Function = default_f(T)[dim]`: Forcing term/cost functional (function of spatial coordinates)
- `f_grid::X` (same container type as `x`): Alternative to `f`, directly provide values on grid (default: `f` evaluated at `x`)
- `Q::Convex{T} = convex_Euclidian_power(T, idx=2:dim+2, p=x->p)`: Convex constraint set

## Output Control
- `verbose::Bool = true`: Display progress bar during solving
- `logfile = devnull`: IO stream for logging (default: no file logging)

## Solver Control (passthrough)
Additional keyword arguments are forwarded to the internal solvers:
- `tol = sqrt(eps(T))`: Barrier tolerance; the method stops once `1/t > 1/tol`
- `t = T(0.1)`: Initial barrier parameter for the main solve
- `t_feasibility = t`: Initial barrier parameter for the feasibility solve
- `maxit = 10000`: Maximum number of outer (barrier) iterations
- `kappa = T(10.0)`: Multiplier controlling barrier growth (`t ← min(kappa*t, ...)`)
- `early_stop = z->false`: Callback `z -> Bool`; if `true`, halt early (e.g. after feasibility is reached)
- `max_newton = ceil((log2(-log2(eps(T))))+2)`: Max Newton iterations per inner solve
- `stopping_criterion = stopping_inexact(sqrt(minimum(M[1].w))/2, T(0.5))`: Newton stopping rule
  - use `stopping_exact(theta)` or `stopping_inexact(lambda_tol, theta)`
- `line_search = linesearch_backtracking(T)`: Line search strategy (`linesearch_backtracking` or `linesearch_illinois`)
- `finalize = stopping_exact(T(0.5))`: Extra final Newton pass with stricter stopping
- `progress = x->nothing`: Progress callback in [0,1]; when `verbose=true` this is set internally
- `printlog = (args...)->nothing`: Logging callback; overridden by `logfile` when using `amgb`

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

Solution object with fields:
- `z`: Solution matrix of size `(n_nodes, n_components)` containing the computed solution
- `SOL_feasibility`: Feasibility phase results (`nothing` if the initial point was already feasible), otherwise a solution object (see below)
- `SOL_main`: Main optimization phase results as a solution object (see below)
- `log`: String containing detailed iteration log for debugging
- `geometry`: The input `geometry` object

Each solution object (`SOL_feasibility` and `SOL_main`) is a NamedTuple containing:
- `z`: Solution vector (flattened; for feasibility phase includes auxiliary slack variable)
- `z_unfinalized`: Solution before final refinement step
- `c`: Cost functional used in this phase
- `its`: Iteration counts across levels and barrier steps (L×k matrix where L is number of levels, k is number of barrier iterations)
- `ts`: Sequence of barrier parameters t used (length k)
- `kappas`: Step size multipliers used at each iteration (length k)
- `times`: Wall-clock timestamps for each iteration (length k)
- `t_begin`, `t_end`, `t_elapsed`: Timing information for this phase
- `passed`: Boolean array indicating phase 1 success at each level
- `c_dot_Dz`: Values of ⟨c, D*z⟩ at each barrier iteration (length k)

# Algorithm Overview

The AMGB method combines:
1. Interior point method: Uses logarithmic barriers to handle constraints
2. Multigrid acceleration: Solves on hierarchy of grids from coarse to fine
3. Damped Newton iteration: Inner solver with line search for robustness

The solver automatically handles:
- Construction of appropriate discretization and multigrid hierarchy
- Feasibility restoration when initial point is infeasible
- Adaptive barrier parameter updates with step size control
- Convergence monitoring across multiple grid levels
- Progress reporting (when `verbose=true`) and logging (to `logfile` if specified)

# Errors

Throws `AMGBConvergenceFailure` if:
- The feasibility problem cannot be solved (problem may be infeasible)
- The main optimization fails to converge within `maxit` iterations
- Newton iteration fails at any grid level

# Examples

```julia
# Solve 1D p-Laplace problem with p=1.5 using FEM
z = amgb(fem1d(L=4); p=1.5).z

# Solve 2D problem with spectral elements
z = amgb(spectral2d(n=8); p=2.0).z

# Custom boundary conditions
g_custom(x) = [sin(π*x[1])*sin(π*x[2]), 10.0]
z = amgb(fem2d(L=3); g=g_custom).z

# Get detailed solution information
sol = amgb(fem1d(L=3); verbose=true)
println("Iterations: ", sum(sol.SOL_main.its))
println("Final barrier parameter: ", sol.SOL_main.ts[end])

# Visualize the first component
plot(sol)

# Log iterations to a file
open("solver.log", "w") do io
    amgb(fem2d(L=2); logfile=io, verbose=false)
end
```

# See Also
- [`fem1d_solve`](@ref), [`fem2d_solve`](@ref), [`spectral1d_solve`](@ref), [`spectral2d_solve`](@ref):
  Convenience wrappers for specific discretizations
- [`Convex`](@ref): Constraint set specification type
"""
function amgb(geometry::Geometry{T,X,W,Mat,Discretization}=fem1d();
        dim::Integer = amg_dim(geometry.discretization),
        state_variables = [:u :dirichlet ; :s :full],
        D = default_D[dim],
        x = geometry.x,
        p::T = T(1.0),
        g::Function = default_g(T)[dim],
        f::Function = default_f(T)[dim],
        g_grid::X = map_rows(x->g(x)',x),
        f_grid::X = map_rows(x->f(x)',x),
        Q::Convex{T} = convex_Euclidian_power(T,idx=2:dim+2,p=x->p),
        verbose=true,
        logfile=devnull,
        rest...) where {T,X,W,Mat,Discretization}
    M = amg(geometry;state_variables,D)
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
    SOL=amgb_driver(M,f_grid, g_grid, Q;x,progress,printlog,rest...)
    return AMGBSOL(SOL.z,SOL.SOL_feasibility,SOL.SOL_main,String(take!(log_buffer)),geometry)
end

