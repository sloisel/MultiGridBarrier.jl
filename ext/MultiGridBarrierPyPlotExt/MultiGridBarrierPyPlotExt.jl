# MultiGridBarrierPyPlotExt — all MultiGridBarrier visualization, as a package
# extension. Loads automatically when both MultiGridBarrier and PyPlot are
# imported (`using MultiGridBarrier, PyPlot`; PyCall arrives with PyPlot). The
# solver core has no Python dependency without it.
#
# Every method here extends `PyPlot.plot` (and `PyPlot.savefig`), so `plot` is
# one function whether you hand it matplotlib arrays or MultiGridBarrier
# solutions:
#   * 1D / 2D FEM and spectral geometries -> matplotlib (this file);
#   * FEM3D volumes, embedded surfaces and curves -> PyVista (plot3d.jl,
#     imported lazily through PyCall on the first 3D plot);
#   * ParabolicSOL / (geometry, ts, U) time series -> HTML5 animations
#     (matplotlib.animation in 1D/2D, ffmpeg-encoded PyVista frames in 3D).
#
# The result types `MGB3DFigure` and `HTML5anim` are plain data (PNG bytes /
# HTML text) and live in the parent so they exist without Python.
module MultiGridBarrierPyPlotExt

using PyPlot
import PyPlot: plot, savefig
using PyCall
import MultiGridBarrier
import MultiGridBarrier: Geometry, MGBSOL, ParabolicSOL, HTML5anim, MGB3DFigure,
    FEM2D_P1, FEM2D_P2, SPECTRAL1D, SPECTRAL2D, TensorFEM, FEM3D,
    interpolate, amg_dim, _xflat

@doc raw"""
    plot(sol::MGBSOL, k::Int=1; kwargs...)
    plot(sol::ParabolicSOL, k::Int=1; kwargs...)
    plot(M::Geometry, z::Vector; kwargs...)
    plot(M::Geometry, ts::AbstractVector, U::Matrix; frame_time=..., embed_limit=..., printer=...)

Visualize solutions and time sequences on meshes. Provided by the
`MultiGridBarrierPyPlotExt` extension: load PyPlot
(`using MultiGridBarrier, PyPlot`) and these methods attach to `PyPlot.plot`.

- 1D problems: Line plot. For spectral methods, you can specify evaluation points with `x=-1:0.01:1`.
- 2D FEM: Triangulated surface plot using the mesh structure.
- 2D spectral: 3D surface plot. You can specify evaluation grids with `x=-1:0.01:1, y=-1:0.01:1`.
- 3D FEM, embedded surfaces and curves: PyVista render returning an `MGB3DFigure` (PNG).

Time sequences (animation):
- Call `plot(M, ts, U; frame_time=1/30, printer=anim->nothing)` where `U` has columns as frames and `ts` are absolute times in seconds (non-uniform allowed).
- Or simply call `plot(sol)` where `sol` is a `ParabolicSOL` returned by `parabolic_solve` (uses `sol.ts`).
- Animation advances at a fixed frame rate given by `frame_time` (seconds per video frame). For irregular `ts`, each video frame shows the latest data frame with timestamp ≤ current video time.
- The `printer` callback receives the Matplotlib animation object; use it to display or save (e.g., `anim.save("out.mp4")`).
- `embed_limit` controls the maximum embedded HTML5 video size in megabytes.

When `sol` is a solution object returned by `mgb_solve`, `plot(sol,k)` plots the kth
component `sol.z[:, k]` using `sol.geometry`. `plot(sol)` uses the default k=1.

All other keyword arguments are passed to the underlying `PyPlot` functions.
""" plot

# ---------------------------------------------------------------------------
# Solution objects
# ---------------------------------------------------------------------------

plot(sol::MGBSOL, k::Int=1; kwargs...) = plot(sol.geometry, sol.z[:, k]; kwargs...)

# Fixed-FPS parabolic animation: advance logical frame index based on ts and frame_time
plot(sol::ParabolicSOL, k::Int=1; kwargs...) =
    plot(sol.geometry, sol.ts, stack(sol.u[j][:, k] for j in eachindex(sol.ts)); kwargs...)

# ---------------------------------------------------------------------------
# 1D / 2D FEM geometries (matplotlib)
# ---------------------------------------------------------------------------

# The one 3D axes of the current figure. PyPlot's top-level 3D wrappers
# (plot_trisurf, plot_surface, ...) locate their target through gca(), which on
# an axes-less figure manufactures a default *2D* axes; matplotlib ≥ 3.5 no
# longer removes an axes displaced by another at the same slot, so its empty
# frame lingers behind the 3D plot. Target the 3D axes explicitly instead:
# reuse the current axes when it is already 3D (overlays, animation redraws),
# else add one to the current figure.
function _axes3d()
    fig = gcf()
    (!isempty(fig.axes) && gca().name == "3d") ? gca() : fig.add_subplot(projection="3d")
end

function plot(M::Geometry{T, Array{T,3}, Vector{T}, <:Any, FEM2D_P1{T}}, z::Vector{T}; kwargs...) where {T}
    Xf = _xflat(M.x)
    x = Xf[:,1]
    y = Xf[:,2]
    N = size(M.x, 2)
    S = reshape(0:(3*N-1), 3, N)'
    _axes3d().plot_trisurf(x, y, z, triangles=S; kwargs...)
end

function plot(M::Geometry{T, Array{T,3}, Vector{T}, <:Any, <:FEM2D_P2{T}}, z::Vector{T}; kwargs...) where {T}
    Xf = _xflat(M.x)
    x = Xf[:,1]
    y = Xf[:,2]
    V = size(M.x, 1)
    # Bubble layout: fan around the centroid node 7. Pure P2: the standard
    # 4-subtriangle split on corners (1,3,5) and edge midpoints (2,4,6).
    S = V == 7 ? [1 2 7
                  2 3 7
                  3 4 7
                  4 5 7
                  5 6 7
                  6 1 7] :
                 [1 2 6
                  2 3 4
                  4 5 6
                  2 4 6]
    N = size(M.x, 2)
    S = vcat([S.+(V*k) for k=0:N-1]...)
    _axes3d().plot_trisurf(x,y,z,triangles=S .- 1; kwargs...)
end

function plot(M::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{1,1,T}}, z::Vector{T}; kwargs...) where {T}
    xv = vec(_xflat(M.x))
    perm = sortperm(xv)
    plot(xv[perm], z[perm]; kwargs...)
end

# 2D: triangulate each quad's (s × s) node grid into 2(s-1)^2 triangles.
function plot(M::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{2,2,T}}, z::Vector{T}; kwargs...) where {T}
    k = M.discretization.k
    s = k + 1; n = s^2; N = size(M.x, 2)
    Xf = _xflat(M.x)
    xs = Xf[:, 1]; ys = Xf[:, 2]
    tris = Vector{NTuple{3,Int}}()
    lin(ix, iy, e) = (e-1)*n + (iy-1)*s + ix       # axis-1 (x) fastest
    for e in 1:N, iy in 1:s-1, ix in 1:s-1
        a = lin(ix,   iy,   e); b = lin(ix+1, iy,   e)
        c = lin(ix,   iy+1, e); dd = lin(ix+1, iy+1, e)
        push!(tris, (a, b, dd)); push!(tris, (a, dd, c))
    end
    S = reduce(vcat, ([t[1] t[2] t[3]] for t in tris))
    _axes3d().plot_trisurf(xs, ys, z, triangles = S .- 1; kwargs...)
end

# ---------------------------------------------------------------------------
# Spectral geometries (matplotlib, sampled through `interpolate`)
# ---------------------------------------------------------------------------

function plot(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL1D{T}},y::Vector{T};x=Array(-1:T(0.01):1),rest...) where {T}
    plot(Float64.(x),Float64.(interpolate(M,y,x));rest...)
end

function plot(M::Geometry{T,Array{T,3},Vector{T},<:Any,SPECTRAL2D{T}},z::Array{T,1};x=-1:T(0.01):1,y=-1:T(0.01):1,rest...) where {T}
    X = repeat(x,1,length(y))
    Y = repeat(y,1,length(x))'
    Z = reshape(interpolate(M,z,hcat(X[:],Y[:])),(length(x),length(y)))
    dx = maximum(x)-minimum(x)
    dy = maximum(y)-minimum(y)
    lw = max(dx,dy)*0.002
    _axes3d().plot_surface(Float64.(x), Float64.(y), Float64.(Z); rcount=50, ccount=50, antialiased=false, edgecolor=:black, linewidth=Float64(lw), rest...)
end

# ---------------------------------------------------------------------------
# 1D/2D time-series animation (matplotlib.animation -> HTML5anim)
# ---------------------------------------------------------------------------

# Fixed-FPS timeline shared by the 2D (matplotlib) and 3D (ffmpeg) animators:
# validates ts against the frame count and returns (n_video_frames, frame_of),
# where frame_of(j, cur) advances cur to the latest data frame whose timestamp
# is <= the j-th video time (j is 0-based).
function _anim_timeline(ts, nframes::Int, frame_time)
    length(ts) == nframes ||
        error("length(ts)=$(length(ts)) must equal number of frames=$(nframes)")
    any(diff(ts) .< 0) && error("ts must be nondecreasing")
    ts0 = ts .- ts[1]                     # relative times starting at 0
    total_time = ts0[end]
    n_video_frames = max(1, Int(floor(total_time / frame_time)) + 1)
    frame_of = function (j, cur)
        t = min(j * frame_time, total_time)
        while cur < nframes && ts0[cur + 1] <= t
            cur += 1
        end
        return cur
    end
    return n_video_frames, frame_of
end

function plot(M::Geometry{T,X,W,<:Any,Discretization}, ts::AbstractVector{T}, U::AbstractMatrix{T};
        frame_time::Real = max(0.001, minimum(diff(ts))),
        embed_limit=200.0,
        printer=(animation)->nothing
        ) where {T,X,W,Discretization}
    anim = pyimport("matplotlib.animation")
    m0 = minimum(U)
    m1 = maximum(U)
    dim = amg_dim(M.discretization)
    nframes = size(U, 2)
    n_video_frames, frame_of = _anim_timeline(ts, nframes, frame_time)

    # State ref for the animation closure
    current_idx = Ref(1)                  # 1-based index into U columns

    function draw_frame(j)
        # j is 0-based (FuncAnimation calls with 0..n_video_frames-1)
        current_idx[] = frame_of(j, current_idx[])
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

# 3D (FEM3D = TensorFEM{3}) PyVista plotting + animation
include("plot3d.jl")

end # module
