export parabolic_solve, parabolic_plot

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
    parabolic_solve(::Type{T}=Float64;
        method = FEM2D,
        state_variables = [:u  :dirichlet
                           :s1 :full
                           :s2 :full],
        dim = amg_dim(method),
        f1 = x->T(0.5),
        f_default = default_f_parabolic[dim],
        p = T(1),
        h = T(0.2),
        f = (t,x)->f_default(h*f1(x)-x[1+dim],T(0.5),h/p),
        g = default_g_parabolic[dim],
        D = default_D_parabolic[dim],
        L = 2,
        t0 = T(0),
        t1 = T(1),
        M = amg_construct(T,method,L=L,D=D,state_variables=state_variables),
        Q = (convex_Euclidian_power(;idx=[1,2+dim],p=x->T(2)) 
            ∩ convex_Euclidian_power(;idx=vcat(2:1+dim,3+dim),p=x->p)),
        verbose = true,
        show = true,
        interval = 200,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=200.0)),
        rest...) where {T}

Solves a parabolic (i.e. time-dependent) p-Laplace problem of the form:
```math
u_t - \nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = -f_1.
```
We use the implicit Euler scheme ``u_t \approx (u_{k+1}-u_k)/h`` to arrive at:
```math
u_{k+1} - h\nabla \cdot (\|\nabla u_{k+1}\|_2^{p-2}\nabla u_{k+1}) = u_k-hf_1.
```
According to the calculus of variation, we look for a weak solution minimizing
```math
J(u) = \int_{\Omega}{1 \over 2} u^2 + h {1 \over p} \|\nabla u\|_2^p + (hf_1-u_k)u \, dx
```
We introduce the slack functions ``s_1 \geq u^2`` and ``s_2 \geq \|\nabla u\|_2^p`` and minimize instead
```math
\int_{\Omega} {1 \over 2}s_1 + {h \over p} s_2 + (hf_1-u_k)u \, dx.
```
The canonical form is:
```math
z = \begin{bmatrix} u \\ s_1 \\ s_2 \end{bmatrix}
\qquad
f^T = \left[hf_1-u_k,0,0,{1 \over 2},{h \over p}\right]
\qquad
Dz = \begin{bmatrix} u \\ u_x \\ u_y \\ s_1 \\ s_2 \end{bmatrix}
\qquad
g = \begin{bmatrix} g_1 \\ 0 \\ 0 \end{bmatrix}.
```
Here, ``g_1`` encodes boundary conditions for ``u``. Then we minimize:
```math
\int_{\Omega} f^TDz
```

The named arguments `rest...` are passed verbatim to `amg_solve`.
"""
function parabolic_solve(::Type{T}=Float64, geometry=fem2d();
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
        M = subdivide(T,geometry;D=D,state_variables=state_variables),
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
        z = amgb(T,geometry;M=M,x=hcat(M[1].x,U[:,:,k]),g_grid=U[:,:,k+1],f=x->f(ts[k+1],x),Q=Q,show=false,verbose=false,rest...)
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

"""
    parabolic_plot(method,M::AMG{T, Mat,Geometry}, U::Matrix{T};
        interval=200, embed_limit=200.0,
        printer=(animation)->display("text/html", animation.to_html5_video(embed_limit=embed_limit))) where {T,Mat,Geometry}

Animate the solution of the parabolic problem.
"""
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
