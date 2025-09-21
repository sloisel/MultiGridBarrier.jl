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
@doc raw"""
    parabolic_solve(geometry=fem2d(), ::Type{T}=get_T(geometry); kwargs...)

Solve time-dependent p-Laplace problems using implicit Euler timestepping.

Solves the parabolic PDE:
```math
u_t - \nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = -f_1
```
using implicit Euler discretization and barrier methods.

# Arguments
- `geometry`: Discretization geometry (default: `fem2d()`)
- `T::Type`: Numeric type (inferred from geometry)

# Keyword Arguments

## Discretization
- `state_variables`: State variables (default: `[:u :dirichlet; :s1 :full; :s2 :full]`)
- `D`: Differential operators (default depends on dimension)
- `dim::Int`: Spatial dimension (auto-detected from geometry)
- `M`: Pre-built AMG hierarchy (constructed if not provided)

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
- `show::Bool=true`: Animate solution after solving
- `interval::Int=200`: Animation frame interval (ms)
- `printer`: Function to display animation

## Additional Parameters
- `rest...`: Passed to `amgb` for each time step

# Returns
3D array `U` of size `(n_nodes, n_components, n_timesteps)` containing
the solution at each time step.

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
- [`PyPlot.plot`](@ref): Animation function for time-dependent solutions
"""
function parabolic_solve(geometry=fem2d(),::Type{T}=get_T(geometry);
        state_variables = [:u  :dirichlet
                           :s1 :full
                           :s2 :full],
        dim = amg_dim(geometry),
        f1 = x->T(0.5),
        f_default = default_f_parabolic[dim],
        p = T(1),
        h = T(0.2),
        f = (t,x)->f_default(h*f1(x)-x[1+dim],T(0.5),h/p),
        g = default_g_parabolic[dim],
        D = default_D_parabolic[dim],
        t0 = T(0),
        t1 = T(1),
        M = subdivide(geometry;D=D,state_variables=state_variables),
        Q = (convex_Euclidian_power(;idx=[1,2+dim],p=x->T(2)) 
            ∩ convex_Euclidian_power(;idx=vcat(2:1+dim,3+dim),p=x->p)),
        verbose = true,
        show = true,
        interval = 200,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=200.0)),
        rest...) where {T}
    ts = t0:h:t1
    n = length(ts)
    m = size(M[1].x,1)
    g0 = g
    if g isa Function
        foo = g(t0,M[1].x[1,:])
        d = length(foo)
        g0 = zeros(T,(m,d,n))
        for j=1:n
            for k=1:m
                g0[k,:,j] = g(ts[j],M[1].x[k,:])
            end
        end
    end
    d = size(g0,2)
    U = g0
    pbar = 0
    prog = k->nothing
    if verbose
        pbar = Progress(n; dt=1.0)
        prog = k->update!(pbar,k)
    end
    for k=1:n-1
        prog(k-1)
        z = amgb(geometry;M=M,x=hcat(M[1].x,U[:,:,k]),g_grid=U[:,:,k+1],f=x->f(ts[k+1],x),Q=Q,show=false,verbose=false,rest...)
        U[:,:,k+1] = z
    end
    if verbose
        finish!(pbar)
    end
    if show
        plot(M[1],U[:,1,:],interval=interval,printer=printer)
    end
    return U
end

# Implementation of PyPlot.plot for time-dependent solutions - creates animation
function PyPlot.plot(M::AMG{T, Mat,Geometry}, U::Matrix{T};
        interval=200, embed_limit=200.0,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=embed_limit))) where {T,Mat,Geometry}
    anim = pyimport("matplotlib.animation")
#    anim = matplotlib.animation
    m0 = minimum(U)
    m1 = maximum(U)
    dim = amg_dim(M.geometry)
    function animate(i)
        clf()
        ret = plot(M,U[:,i+1])
        ax = plt.gca()
        if dim==1
            ax.set_ylim([m0, m1])
            return ret
        end
        ax.axes.set_zlim3d(bottom=m0, top=m1)
        return [ret,]
    end

    init()=animate(0)
    fig = figure()
    myanim = anim.FuncAnimation(fig, animate, frames=size(U,2), init_func=init, interval=interval, blit=true)
    printer(myanim)
    plt.close(fig)
    return nothing
end

function parabolic_precompile()
    parabolic_solve(geometry=fem1d(L=1),h=0.5)
    parabolic_solve(geometry=fem2d(L=1),h=0.5)
    parabolic_solve(geometry=spectral1d(n=4),h=0.5)
    parabolic_solve(geometry=spectral2d(n=4),h=0.5)
end

precompile(parabolic_precompile,())
