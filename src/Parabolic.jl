export parabolic_solve

default_D_parabolic(::Val{1}) = [:u  :id
     :u  :dx
     :s1 :id
     :s2 :id]
default_D_parabolic(::Val{2}) = [:u  :id
     :u  :dx
     :u  :dy
     :s1 :id
     :s2 :id]
default_D_parabolic(::Val{3}) = [:u  :id
     :u  :dx
     :u  :dy
     :u  :dz
     :s1 :id
     :s2 :id]
default_D_parabolic(k::Int) = default_D_parabolic(Val(k))
default_f_parabolic(::Val{1}) = (f1,w1,w2)->[f1,0,w1,w2]
default_f_parabolic(::Val{2}) = (f1,w1,w2)->[f1,0,0,w1,w2]
default_f_parabolic(::Val{3}) = (f1,w1,w2)->[f1,0,0,0,w1,w2]
default_f_parabolic(k::Int) = default_f_parabolic(Val(k))

default_g_parabolic(::Val{1}) = (t,x)->[x[1],0,0]
default_g_parabolic(::Val{2}) = (t,x)->[x[1]^2+x[2]^2,0,0]
default_g_parabolic(::Val{3}) = (t,x)->[x[1]^2+x[2]^2+x[3]^2,0,0]
default_g_parabolic(k::Int) = default_g_parabolic(Val(k))

# Parabolic indices for convex constraints - static SVector versions for GPU compatibility
# idx1 corresponds to [1, 2+dim] - indices for first constraint (s1 >= u^2)
parabolic_idx1(::Val{1}) = SVector(1, 3)
parabolic_idx1(::Val{2}) = SVector(1, 4)
parabolic_idx1(::Val{3}) = SVector(1, 5)
parabolic_idx1(k::Int) = parabolic_idx1(Val(k))

# idx2 corresponds to vcat(2:1+dim, 3+dim) - indices for second constraint (s2 >= ||∇u||^p)
parabolic_idx2(::Val{1}) = SVector(2, 4)
parabolic_idx2(::Val{2}) = SVector(2, 3, 5)
parabolic_idx2(::Val{3}) = SVector(2, 3, 4, 6)
parabolic_idx2(k::Int) = parabolic_idx2(Val(k))

@doc raw"""
    ParabolicSOL{T,X,W,Discretization,G}

Solution structure returned by `parabolic_solve`.

# Fields
- `geometry::G`: the discretization geometry.
- `ts::Vector{T}`: time stamps for each solution snapshot.
- `u::Vector{X}`: list of solution matrices, one per timestep. Each matrix has size `(n_nodes, n_components)`.

# Plotting
Use `plot(sol)` to create an HTML5 animation, or `plot(sol, k)` to animate component `k`.
"""
struct ParabolicSOL{T,X,W,Discretization,G}
    geometry::G
    ts::Vector{T}
    u::Vector{X}
end
function ParabolicSOL(geometry::Geometry{T,<:Any,W,<:Any,Discretization}, ts, u::Vector{X}) where {T,X,W,Discretization}
    ParabolicSOL{T,X,W,Discretization,typeof(geometry)}(geometry, collect(T, ts), u)
end

# plot(sol::ParabolicSOL, k) and the (M, ts, U) animation live in
# MultiGridBarrierPyPlotExt.

"""
    HTML5anim

Wrapper around an HTML5 `<video>` produced by the parabolic-animation `plot`
methods. It carries the rendered video markup in its `anim::String` field and
defines `show(io, MIME"text/html"(), ::HTML5anim)`, so returning one as the last
value of a Jupyter/Pluto cell (or in Documenter output) embeds the animation
inline.
"""
struct HTML5anim
    anim::String
end

# Render inline in Jupyter/Documenter/etc.
function Base.show(io::IO, ::MIME"text/html", A::HTML5anim)
    print(io, A.anim)
end

@doc raw"""
    parabolic_solve(mg::MultiGrid; kwargs...) -> ParabolicSOL

Solve time-dependent p-Laplace problems using implicit Euler timestepping on the multigrid
hierarchy `mg`.

# Keyword Arguments

## Discretization
- `state_variables = [:u :dirichlet; :s1 :full; :s2 :full]`.
- `D`: differential operators (default depends on spatial dimension).
- `dim::Int`: spatial dimension (auto-detected from `mg`).

## Time Integration
- `t0=T(0)`, `t1=T(1)`, `h=T(0.2)`, `ts=t0:h:t1`.

## Problem Parameters
- `p::T=T(1)`: exponent for the p-Laplacian.
- `f1`: source term `(t, x) -> T` (default: `(t,x)->T(0.5)`).
- `g`: initial/boundary conditions `(t, x) -> Vector{T}`.
- `Q`: convex constraints.

## Advanced (grid-level) overrides
- `f1_grid`: per-node samples of `f1` at every `ts[j]` (one column per time).
- `g_grid`: a function `j -> grid`, the sampled `g` at time `ts[j]`.
- `f_grid`: a function `(z, j) -> grid`, the linear-term grid for step `j` given
  the previous state `z`; together with `f_default` (the per-row layout of that
  linear functional) the default implements implicit Euler for the p-Laplace flow.

## Output Control
- `verbose::Bool=true`: progress bar.

## Additional Parameters
- `rest...`: forwarded to `mgb_solve` for each time step.

# Examples
```julia
sol = parabolic_solve(amg(fem2d_P2()); p=2.0, h=0.1)
sol = parabolic_solve(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=33)))); p=1.5, h=0.05)
```
"""
function parabolic_solve(mg::MultiGrid{T} =
                              amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3))));
        state_variables = [:u  :dirichlet
                           :s1 :full
                           :s2 :full],
        dim = amg_dim(mg.geometry.discretization),
        f1 = (t,x)->T(0.5),
        f_default = default_f_parabolic(dim),
        p = T(1),
        h = T(0.2),
        t0 = T(0),
        t1 = T(1),
        ts = t0:h:t1,
        f1_grid = map_rows(x->SVector(Tuple([f1(ts[j], x) for j=1:length(ts)])),_xflat(mg)),
        f_grid = (z, j)->map_rows((z,f1_grid)->SVector(Tuple(f_default((ts[j]-ts[j-1]) * f1_grid[j] - z[1], T(0.5), (ts[j]-ts[j-1]) / p))),z,f1_grid),
        g = default_g_parabolic(dim),
        g_grid = j->map_rows(x->SVector(Tuple(g(ts[j], x))),_xflat(mg)),
        D = default_D_parabolic(dim),
        Q = intersect(mg,
              convex_Euclidian_power(T; mg=mg, idx=parabolic_idx1(dim), p=x->T(2)),
              convex_Euclidian_power(T; mg=mg, idx=parabolic_idx2(dim), p=x->p)),
        verbose = true,
        rest...) where {T}
    n = length(ts)
    U = [g_grid(k) for k in 1:n]
    # The (main, feasibility) AMG hierarchy pair depends only on
    # (state_variables, D): build it once and reuse it for every time step
    # instead of letting assemble rebuild it per step.
    M = _prepare_amg(mg; state_variables, D)
    prog = k->nothing
    if verbose
        pbar = Progress(n; dt=1.0)
        prog = k->update!(pbar,k)
    end
    for k=1:n-1
        prog(k-1)
        prob = assemble(mg; M=M, g_grid=U[k+1], f_grid=f_grid(U[k], k+1), Q=Q)
        sol = mgb_solve(prob; verbose=false, rest...)
        U[k+1] = sol.z
    end
    if verbose
        finish!(pbar)
    end
    ret = ParabolicSOL(mg.geometry, ts, U)
    return ret
end


