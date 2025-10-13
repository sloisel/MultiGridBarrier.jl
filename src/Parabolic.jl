export parabolic_solve

default_D_parabolic = [
    [:u  :id
     :u  :dx
     :s1 :id
     :s2 :id],
    [:u  :id
     :u  :dx
     :u  :dy
     :s1 :id
     :s2 :id]
    ]
default_f_parabolic = [
    (f1,w1,w2)->[f1,0,w1,w2],
    (f1,w1,w2)->[f1,0,0,w1,w2]
]
default_g_parabolic = [
    (t,x)->[x[1],0,0],
    (t,x)->[x[1]^2+x[2]^2,0,0],
]

struct ParabolicSOL{T,Mat,Discretization}
    geometry::Geometry{T,Mat,Discretization}
    ts::Vector{T}
    u::Array{T,3}
end

# Fixed-FPS parabolic animation: advance logical frame index based on ts and frame_time
plot(sol::ParabolicSOL{T,Mat,Discretization}, k::Int=1; kwargs...) where {T,Mat,Discretization} =
    plot(sol.geometry, sol.ts, sol.u[:, k, :]; kwargs...)

struct HTML5anim
    anim::String
end

# Render inline in Jupyter/Documenter/etc.
function Base.show(io::IO, ::MIME"text/html", A::HTML5anim)
    print(io, A.anim)
end

# Nice fallback in non-HTML contexts (terminal, logs)
function Base.show(io::IO, ::MIME"text/plain", A::HTML5anim)
    print(io, "HTML5 animation — use display(\"text/html\", obj) to embed.")
end

function plot(M::Geometry{T, Mat, Discretization}, ts::AbstractVector{T}, U::Matrix{T};
        frame_time::Real = max(1/1000.0, reduce(gcd, max.(1, round.(Int, 1_000_000 .* diff(ts)))) / 1_000_000),
        embed_limit=200.0,
        printer=(animation)->nothing
        ) where {T,Mat,Discretization}
    anim = pyimport("matplotlib.animation")
    m0 = minimum(U)
    m1 = maximum(U)
    dim = amg_dim(M.discretization)
    nframes = size(U, 2)

    if length(ts) != nframes
        error("length(ts)=$(length(ts)) must equal number of frames=$(nframes)")
    end
    if any(diff(ts) .< 0)
        error("ts must be nondecreasing")
    end

    # Build a fixed-FPS timeline and advance current data frame according to ts
    ts0 = ts .- ts[1]                     # relative times starting at 0
    total_time = ts0[end]
    Δ = T(frame_time)
    n_video_frames = max(1, Int(floor(total_time / Δ)) + 1)

    # State refs for animation closure
    current_idx = Ref(1)                  # 1-based index into U columns

    function draw_frame(j)
        # j is 0-based by our design (FuncAnimation will call with 0..n_video_frames-1)
        t = min(T(j) * Δ, total_time)
        # Advance to the latest data frame not exceeding time t
        while current_idx[] < nframes && ts0[current_idx[] + 1] <= t
            current_idx[] += 1
        end

        clf()
        ret = plot(M, U[:, current_idx[]])
        ax = plt.gca()
        if dim == 1
            ax.set_ylim([m0, m1])
            return ret
        end
        ax.axes.set_zlim3d(bottom=m0, top=m1)
        return [ret,]
    end

    init() = draw_frame(0)
    fig = figure()
    interval_ms = Int(round(1000 * Float64(frame_time)))
    myanim = anim.FuncAnimation(fig, draw_frame, frames=n_video_frames, init_func=init, interval=interval_ms, blit=true)
    printer(myanim)
    ret = HTML5anim(myanim.to_html5_video(embed_limit=embed_limit))
    plt.close(fig)
    return ret
end

@doc raw"""
    parabolic_solve(geometry::Geometry{T,Mat,Discretization}=fem2d(); kwargs...)

Solve time-dependent p-Laplace problems using implicit Euler timestepping.

Solves the parabolic PDE:
```math
u_t - \nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = -f_1
```
using implicit Euler discretization and barrier methods.

# Arguments
- `geometry`: Discretization geometry (default: `fem2d()`).

# Keyword Arguments

## Discretization
- `state_variables`: State variables (default: `[:u :dirichlet; :s1 :full; :s2 :full]`).
- `D`: Differential operators (default depends on spatial dimension).
- `dim::Int`: Spatial dimension (auto-detected from `geometry`).

## Time Integration
- `t0::T=0`: Initial time.
- `t1::T=1`: Final time.
- `h::T=0.2`: Time step size.
- `ts::AbstractVector{T}=t0:h:t1`: Time grid; override to provide a custom, nonuniform, nondecreasing sequence.

## Problem Parameters
- `p::T=1`: Exponent for the p-Laplacian.
- `f1`: Source term function of signature `(t, x) -> T` (default: `(t,x)->T(0.5)`).
- `g`: Initial/boundary conditions function of signature `(t, x) -> Vector{T}` (default depends on dimension).
- `Q`: Convex constraints (default: appropriate for p-Laplace).

## Output Control
- `verbose::Bool=true`: Show a progress bar during time stepping.

## Additional Parameters
- `rest...`: Passed through to `amgb` for each time step.

# Returns
A `ParabolicSOL` with fields:
- `geometry`: the `Geometry` used.
- `ts::Vector{T}`: time stamps (seconds).
- `u::Array{T,3}`: solution tensor of size `(n_nodes, n_components, n_timesteps)`.

Animate with `plot(sol)` (or `plot(sol, k)` for component `k`).
To save to a file, use the plotting printer, e.g. `plot(sol; printer=anim->anim.save("out.mp4"))`.

# Mathematical Formulation

The implicit Euler scheme ``u_t \approx (u_{k+1}-u_k)/h`` gives:
```math
u_{k+1} - h\nabla \cdot (\|\nabla u_{k+1}\|^{p-2}\nabla u_{k+1}) = u_k - h f_1.
```

We minimize the functional:
```math
J(u) = \int_\Omega \tfrac{1}{2}u^2 + \tfrac{h}{p}\|\nabla u\|^p + (h f_1 - u_k)u \, dx.
```

With slack variables ``s_1 \ge u^2`` and ``s_2 \ge \|\nabla u\|^p``, this becomes:
```math
\min \int_\Omega \tfrac{1}{2}s_1 + \tfrac{h}{p}s_2 + (h f_1 - u_k)u \, dx.
```

# Examples
```julia
# Basic 2D heat equation (p=2)
sol = parabolic_solve(; p=2.0, h=0.1)

# 1D p-Laplace with custom parameters
sol = parabolic_solve(fem1d(L=5); p=1.5, h=0.05, t1=2.0)

# Spectral discretization
sol = parabolic_solve(spectral2d(n=8); verbose=true)

# Custom initial condition
g_init(t, x) = [exp(-10*(x[1]^2 + x[2]^2)), 0, 0]
sol = parabolic_solve(; g=g_init)
```

# See Also
- [`amgb`](@ref): Single time step solver
- [`plot`](@ref): Animation and plotting function
"""
function parabolic_solve(geometry::Geometry{T,Mat,Discretization}=fem2d();
        state_variables = [:u  :dirichlet
                           :s1 :full
                           :s2 :full],
        dim = amg_dim(geometry.discretization),
        f1 = (t,x)->T(0.5),
        f_default = default_f_parabolic[dim],
        p = T(1),
        h = T(0.2),
        t0 = T(0),
        t1 = T(1),
        ts = t0:h:t1,
        f1_grid = hcat([[f1(ts[j], geometry.x[k, :]) for k=1:size(geometry.x,1)] for j=1:length(ts)]...),
        f_grid = (z, j)->vcat([f_default((ts[j]-ts[j-1]) * f1_grid[k, j] - z[k, 1], T(0.5), (ts[j]-ts[j-1]) / p)' for k=1:size(geometry.x,1)]...),
        g = default_g_parabolic[dim],
        g_grid = j->vcat([g(ts[j], geometry.x[k, :])' for k=1:size(geometry.x,1)]...),
        D = default_D_parabolic[dim],
        Q = (convex_Euclidian_power(;idx=[1,2+dim],p=x->T(2)) 
            ∩ convex_Euclidian_power(;idx=vcat(2:1+dim,3+dim),p=x->p)),
        verbose = true,
        rest...) where {T,Mat,Discretization}
    n = length(ts)
    m = size(geometry.x,1)
    U = cat((g_grid(k) for k in 1:n)...; dims=3)
    pbar = 0
    prog = k->nothing
    if verbose
        pbar = Progress(n; dt=1.0)
        prog = k->update!(pbar,k)
    end
    for k=1:n-1
        prog(k-1)
        sol = amgb(geometry;D=D,state_variables=state_variables,x=hcat(geometry.x,U[:,:,k]),g_grid=U[:,:,k+1],f_grid = f_grid(U[:,:,k],k+1),Q=Q,verbose=false,rest...)
        U[:,:,k+1] = sol.z
    end
    if verbose
        finish!(pbar)
    end
    ret = ParabolicSOL{T,Mat,Discretization}(geometry,ts,U)
    return ret
end


