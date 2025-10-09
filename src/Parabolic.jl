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

@doc raw"""
    parabolic_solve(geometry::Geometry{T,Mat,Discretization}=fem2d(); kwargs...)

Solve time-dependent p-Laplace problems using implicit Euler timestepping.

Solves the parabolic PDE:
```math
u_t - \nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = -f_1
```
using implicit Euler discretization and barrier methods.

# Arguments
- `geometry`: Discretization geometry (default: `fem2d()`)

# Keyword Arguments

## Discretization
- `state_variables`: State variables (default: `[:u :dirichlet; :s1 :full; :s2 :full]`)
- `D`: Differential operators (default depends on dimension)
- `dim::Int`: Spatial dimension (auto-detected from geometry)

## Time Integration
- `t0::T=0`: Initial time
- `t1::T=1`: Final time
- `h::T=0.2`: Time step size

## Problem Parameters
- `p::T=1`: Exponent for p-Laplacian
- `f1`: Source term function (default: `x->T(0.5)`)
- `f`: Full forcing function (derived from f1 by default)
- `g`: Initial/boundary conditions (default depends on dimension)
- `Q`: Convex constraints (default: appropriate for p-Laplace)

## Output Control
- `verbose::Bool=true`: Show progress bar
- `show::Bool=true`: Show animation after solving (calls `plot(M, ts, U[:,1,:]; printer=...)`)
- `printer`: Function to display the animation produced by `plot`. Takes a single argument `animation::matplotlib.animation.FuncAnimation`
  and displays it. Default: `(animation)->display("text/html", animation.to_html5_video(embed_limit=200.0))`.
  Custom printers can save to file (e.g., `(anim)->anim.save("output.mp4")`) or use alternative display methods.

## Additional Parameters
- `rest...`: Passed to `amgb` for each time step

# Returns
ParabolicSOL with fields:
- `geometry`: the Geometry used
- `ts::Vector{T}`: time stamps (seconds)
- `u::Array{T,3}`: solution tensor of size `(n_nodes, n_components, n_timesteps)`

You can animate directly with `plot(sol)` (or `plot(sol, k)` for component k).
To save, pass a printer, e.g. `plot(sol; printer=anim->anim.save("out.mp4"))`.

# Mathematical Formulation

The implicit Euler scheme ``u_t ≈ (u_{k+1}-u_k)/h`` gives:
```math
u_{k+1} - h\nabla \cdot (\|\nabla u_{k+1}\|^{p-2}\nabla u_{k+1}) = u_k - hf_1
```

We minimize the functional:
```math
J(u) = \int_Ω \frac{1}{2}u² + \frac{h}{p}\|\nabla u\|^p + (hf_1 - u_k)u \, dx
```

With slack variables ``s_1 ≥ u²`` and ``s_2 ≥ \|\nabla u\|^p``, this becomes:
```math
\min \int_Ω \frac{1}{2}s_1 + \frac{h}{p}s_2 + (hf_1 - u_k)u \, dx
```

# Examples
```julia
# Basic 2D heat equation (p=2)
U = parabolic_solve(; p=2.0, h=0.1)

# 1D p-Laplace with custom parameters
U = parabolic_solve(fem1d(L=5); p=1.5, h=0.05, t1=2.0)

# Spectral discretization without animation
U = parabolic_solve(spectral2d(n=8); show=false, verbose=true)

# Custom initial condition
g_init(t, x) = [exp(-10*(x[1]^2 + x[2]^2)), 0, 0]
U = parabolic_solve(; g=g_init)
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
        show = true,
        interval = 200,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=200.0)),
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

plot(sol::ParabolicSOL{T,Mat,Discretization},k::Int=1;kwargs...) where {T,Mat,Discretization} = plot(sol.geometry,sol.ts,sol.u[:,k,:];kwargs...)

function plot(M::Geometry{T, Mat, Discretization}, ts::AbstractVector{T}, U::Matrix{T};
        embed_limit=200.0,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=embed_limit)),
        anim_duration=ts[end]-ts[1]
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

    # Rescale to requested animation duration (seconds)
    if ts[end] == ts[1]
        scaled = fill(zero(T), length(ts))
    else
        scaled = (ts .- ts[1]) .* (anim_duration / (ts[end] - ts[1]))
    end

    # Convert to integer milliseconds and keep one frame per unique millisecond
    ms = round.(Int, 1000 .* scaled)
    keep = nframes == 0 ? Int[] : vcat(1, 1 .+ findall(diff(ms) .!= 0))
    ms_unique = ms[keep]
    nsteps = length(keep)

    # Animation function; dynamically set interval for the NEXT frame
    event_ref = Ref{Any}(nothing)
    function animate(i)
        clf()
        j = i + 1               # 1-based into kept frames
        src_idx = keep[j]       # original frame index
        if event_ref[] !== nothing && j < nsteps
            next_ms = ms_unique[j+1] - ms_unique[j]
            setproperty!(event_ref[], :interval, next_ms)
        end
        ret = plot(M, U[:, src_idx])
        ax = plt.gca()
        if dim == 1
            ax.set_ylim([m0, m1])
            return ret
        end
        ax.axes.set_zlim3d(bottom=m0, top=m1)
        return [ret,]
    end

    init() = animate(0)
    fig = figure()
    initial_ms = nsteps >= 2 ? (ms_unique[2] - ms_unique[1]) : 1000
    myanim = anim.FuncAnimation(fig, animate, frames=nsteps, init_func=init, interval=initial_ms, blit=true)
    event_ref[] = myanim.event_source
    printer(myanim)
    plt.close(fig)
    return nothing
end

