# plot3d.jl -- PyVista (VTK) plotting for the TensorFEM family:
#   * FEM3D = TensorFEM{3} volume render (isosurfaces / slices) + ffmpeg HTML5 animation;
#   * TensorFEM{2,3} surfaces in ℝ³ (colored quad mesh);
#   * TensorFEM{1,e} curves in ℝ²/ℝ³ (tube — graphed as height for e=2, colored for e=3).
# Relocated from the former Mesh3d submodule. Geometry/MGBSOL/ParabolicSOL/HTML5anim
# are the enclosing module's types (in scope).

using PyCall
import PyPlot: savefig
using PNGFiles
using Base64
using FFMPEG: ffmpeg

# VTK Cell Types
const VTK_HEXAHEDRON = 12

const pv = PyNULL()
const _plotter = Ref{Any}(nothing)

function ensure_pyvista()
    if ispynull(pv)
        copy!(pv, pyimport_conda("pyvista", "pyvista", "conda-forge"))
    end
end

"""
    MGB3DFigure

A rendered 3D plot returned by the FEM3D `plot` methods; the `png` field holds the
PNG bytes. Displays inline as `image/png` (Jupyter, Documenter, …); write it to a
file with `savefig(fig, "out.png")`.
"""
struct MGB3DFigure
    png::Vector{UInt8}
end

function Base.show(io::IO, ::MIME"image/png", fig::MGB3DFigure)
    write(io, fig.png)
end

"""
    savefig(fig::MGB3DFigure, filename::String)

Save the figure to a file (e.g., "plot.png"). Extends `PyPlot.savefig`.
"""
function savefig(fig::MGB3DFigure, filename::String)
    write(filename, fig.png)
end

# Shared PyVista plumbing for the FEM3D / surface / curve plotters.
# Reuse one off-screen plotter (avoids the macOS "Context leak detected" error),
# cleared and re-lit on each call.
function _mgb_plotter(plotter::NamedTuple)
    if isnothing(_plotter[])
        _plotter[] = pv.Plotter(off_screen=true; plotter...)
    end
    pl = _plotter[]
    pl.clear()
    pl.remove_bounds_axes()
    pl.enable_lightkit()           # clear() removes lights; restore default lighting
    return pl
end

# Screenshot the live plotter into an MGB3DFigure (PNG); keeps the plotter alive.
function _mgb_screenshot(pl)
    img_array = convert(Array, pl.screenshot(return_img=true))
    buf = IOBuffer()
    PNGFiles.save(buf, img_array)
    seekstart(buf)
    return MGB3DFigure(take!(buf))
end

"""
    plot(geo::Geometry{...,FEM3D}, u::Vector; kwargs...)

Plot a 3D Q_k solution `u` on a hexahedral geometry using PyVista (volume render +
isosurfaces, with optional slices). Returns an `MGB3DFigure` (PNG). See the keyword
arguments below for volume/isosurface/slice options.

# Keyword Arguments
- `plotter=(window_size=(800, 600),)`: options passed to `pv.Plotter()`.
- `volume=(;)`: options for `add_volume`; pass `nothing` to disable.
- `scalar_bar_args=...`: options for `add_scalar_bar`; pass `nothing` to hide.
- `isosurfaces`: isosurface values (default: 5 evenly spaced; `[]` to disable).
- `contour_mesh`, `slice`, `slice_orthogonal`, `slice_along_axis`, `show_grid`,
  `camera_position`: see the slicing/grid options.
"""
function plot(geo::Geometry{T,X,W,<:Any,FEM3D{E,T}}, u::Vector{T};
                       plotter::NamedTuple=(window_size=(800, 600),),
                       volume=(;),
                       scalar_bar_args=(title="",position_x=0.6,position_y=0.0,width=0.35,height=0.05,use_opacity=false),
                       isosurfaces=[0.1,0.3,0.5,0.7,0.9]*(maximum(u)-minimum(u)).+minimum(u),
                       contour_mesh=(;),
                       slice_orthogonal=nothing,
                       slice_orthogonal_mesh=(;),
                       slice=nothing,
                       slice_mesh=(;),
                       slice_along_axis=nothing,
                       slice_along_axis_mesh=(;),
                       show_grid=true,
                       camera_position=nothing,
                       kwargs...) where {T,X,W,E}

    ensure_pyvista()
    k = geo.discretization.k

    # Internal name for the scalar field
    u_name = "u"

    # Create unstructured grid — flatten the 3-tensor geom.x to (n_nodes, 3)
    points = reshape(geo.x, :, size(geo.x, 3))
    cells, cell_types = create_vtk_cells(k, size(points, 1))

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Add data
    grid.point_data.set_scalars(u, u_name)

    pl = _mgb_plotter(plotter)

    if !isnothing(volume)
        pl.add_volume(grid, scalars=u_name; show_scalar_bar=false, volume...)
    end

    if length(isosurfaces)>0
        contours = grid.contour(isosurfaces=collect(isosurfaces), scalars=u_name)
        if contours.n_points > 0
            pl.add_mesh(contours; show_scalar_bar=false, contour_mesh...)
        else
            @warn "Isosurface generation resulted in an empty mesh. Check if isosurface values are within the range of the solution."
        end
    end

    if !isnothing(slice_orthogonal)
        sl = grid.slice_orthogonal(; slice_orthogonal...)
        pl.add_mesh(sl; show_scalar_bar=false, slice_orthogonal_mesh...)
    end

    if !isnothing(slice)
        sl = grid.slice(; slice...)
        pl.add_mesh(sl; show_scalar_bar=false, slice_mesh...)
    end

    if !isnothing(slice_along_axis)
        sl = grid.slice_along_axis(; slice_along_axis...)
        pl.add_mesh(sl; show_scalar_bar=false, slice_along_axis_mesh...)
    end

    if show_grid
        pl.show_grid()
    end

    if !isnothing(camera_position)
        pl.camera_position = camera_position
    else
        pl.reset_camera()  # Fit camera to new geometry
    end

    if !isnothing(scalar_bar_args)
        pl.add_scalar_bar(;scalar_bar_args...)
    end

    return _mgb_screenshot(pl)
end

# Subdivide each Q_k hex into k^3 linear VTK hexes over the (k+1)^3 tensor nodes
# (axis-1 fastest), matching the TensorFEM node ordering.
function create_vtk_cells(k::Int, n_total_nodes::Int)
    n_nodes_per_elem = (k+1)^3
    n_elems = div(n_total_nodes, n_nodes_per_elem)
    n_subcells = n_elems * k^3

    cells = Vector{Int}(undef, n_subcells * 9)
    cell_types = fill(UInt8(VTK_HEXAHEDRON), n_subcells)

    idx(ix, iy, iz) = (iz-1)*(k+1)^2 + (iy-1)*(k+1) + ix

    ptr = 1
    for e in 1:n_elems
        base_node_idx = (e-1) * n_nodes_per_elem
        for iz in 1:k, iy in 1:k, ix in 1:k
            cells[ptr]   = 8
            cells[ptr+1] = base_node_idx + idx(ix,   iy,   iz)   - 1
            cells[ptr+2] = base_node_idx + idx(ix+1, iy,   iz)   - 1
            cells[ptr+3] = base_node_idx + idx(ix+1, iy+1, iz)   - 1
            cells[ptr+4] = base_node_idx + idx(ix,   iy+1, iz)   - 1
            cells[ptr+5] = base_node_idx + idx(ix,   iy,   iz+1) - 1
            cells[ptr+6] = base_node_idx + idx(ix+1, iy,   iz+1) - 1
            cells[ptr+7] = base_node_idx + idx(ix+1, iy+1, iz+1) - 1
            cells[ptr+8] = base_node_idx + idx(ix,   iy+1, iz+1) - 1
            ptr += 9
        end
    end
    return cells, cell_types
end

# VTK cell types for the embedded-manifold (surface / curve) plotters.
const VTK_LINE = 3
const VTK_QUAD = 9

# Subdivide each Q_k quad into k^2 linear VTK quads over its (k+1)^2 tensor nodes
# (axis-1 fastest), matching the TensorFEM node ordering; corners counterclockwise.
function create_vtk_quad_cells(k::Int, n_total_nodes::Int)
    s = k + 1; npe = s^2
    n_elems = div(n_total_nodes, npe)
    cells = Vector{Int}(undef, n_elems * k^2 * 5)
    cell_types = fill(UInt8(VTK_QUAD), n_elems * k^2)
    idx(ix, iy) = (iy-1)*s + ix                    # axis-1 (x) fastest
    ptr = 1
    for e in 1:n_elems
        base = (e-1) * npe
        for iy in 1:k, ix in 1:k
            cells[ptr]   = 4
            cells[ptr+1] = base + idx(ix,   iy)   - 1
            cells[ptr+2] = base + idx(ix+1, iy)   - 1
            cells[ptr+3] = base + idx(ix+1, iy+1) - 1
            cells[ptr+4] = base + idx(ix,   iy+1) - 1
            ptr += 5
        end
    end
    return cells, cell_types
end

# Subdivide each Q_k line element into k VTK line segments over its (k+1) nodes.
# Returns the flat VTK connectivity [2,i,j, 2,i,j, …] (0-based) for `PolyData.lines`.
function create_vtk_line_connectivity(k::Int, n_total_nodes::Int)
    s = k + 1
    n_elems = div(n_total_nodes, s)
    cells = Vector{Int}(undef, n_elems * k * 3)
    ptr = 1
    for e in 1:n_elems
        base = (e-1) * s
        for i in 1:k
            cells[ptr] = 2; cells[ptr+1] = base + i - 1; cells[ptr+2] = base + i
            ptr += 3
        end
    end
    return cells
end

"""
    plot(geo::Geometry{...,TensorFEM{2,3,T}}, z::Vector; kwargs...)

Plot a scalar field `z` on a Q_k **surface in ℝ³** (a `fem2d(…; ambient=Val(3))`
geometry) with PyVista — the quad mesh drawn as a colored surface. Returns an
`MGB3DFigure` (PNG).

# Keyword Arguments
- `plotter=(window_size=(800,600),)`: options passed to `pv.Plotter()`.
- `mesh=(;)`: extra options for `add_mesh` (e.g. `show_edges=true`, `cmap="viridis"`).
- `scalar_bar_args=…`: options for `add_scalar_bar`; pass `nothing` to hide.
- `show_grid=true`, `camera_position=nothing`.
"""
function plot(geo::Geometry{T,X,W,<:Any,TensorFEM{2,3,T}}, z::Vector{T};
              plotter::NamedTuple=(window_size=(800,600),),
              mesh=(;),
              scalar_bar_args=(title="",position_x=0.6,position_y=0.0,width=0.35,height=0.05,use_opacity=false),
              show_grid=true,
              camera_position=nothing,
              kwargs...) where {T,X,W}
    ensure_pyvista()
    k = geo.discretization.k
    points = Matrix{T}(reshape(geo.x, :, 3))
    cells, cell_types = create_vtk_quad_cells(k, size(points, 1))
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data.set_scalars(z, "u")

    pl = _mgb_plotter(plotter)
    pl.add_mesh(grid, scalars="u"; show_scalar_bar=false, mesh...)
    show_grid && pl.show_grid()
    isnothing(camera_position) ? pl.reset_camera() : (pl.camera_position = camera_position)
    isnothing(scalar_bar_args) || pl.add_scalar_bar(; scalar_bar_args...)
    return _mgb_screenshot(pl)
end

"""
    plot(geo::Geometry{...,TensorFEM{1,e,T}}, z::Vector; tube=(;), height_scale=1, kwargs...)

Plot a scalar field `z` on a Q_k **curve** with PyVista, drawn as a tube colored by
`z`. Following the package's "graph the solution" convention (cf. the d=1 line and
d=2 surface plots), the solution occupies the first free ambient dimension:

- a curve **in ℝ²** (`ambient=Val(2)`) is rendered as the height-graph `(x, y, z)` —
  the solution becomes the third coordinate (use `height_scale` to exaggerate or
  compress it relative to the spatial extent; default `1`, i.e. 1:1 with the data);
- a curve **in ℝ³** (`ambient=Val(3)`) is rendered in place — there is no spare
  dimension, so `z` shows up as color only.

Returns an `MGB3DFigure` (PNG). Pass `tube=nothing` for bare lines, or
`tube=(radius=r,)` to set the radius (default ≈ 1% of the bounding-box diagonal).
"""
plot(geo::Geometry{T,X,W,<:Any,TensorFEM{1,2,T}}, z::Vector{T}; kwargs...) where {T,X,W} =
    _tf_plot_curve(geo, z, Val(2); kwargs...)
plot(geo::Geometry{T,X,W,<:Any,TensorFEM{1,3,T}}, z::Vector{T}; kwargs...) where {T,X,W} =
    _tf_plot_curve(geo, z, Val(3); kwargs...)

function _tf_plot_curve(geo::Geometry{T}, z::Vector{T}, ::Val{e};
              plotter::NamedTuple=(window_size=(800,600),),
              tube=(;),
              height_scale::Real=1,
              mesh=(;),
              scalar_bar_args=(title="",position_x=0.6,position_y=0.0,width=0.35,height=0.05,use_opacity=false),
              show_grid=true,
              camera_position=nothing,
              kwargs...) where {T,e}
    ensure_pyvista()
    k = geo.discretization.k
    pts = Matrix{T}(reshape(geo.x, :, e))
    # PyVista points are 3D. e=2: graph the solution as the third coordinate (height);
    # e=3: plot the curve in place (the field shows up as color only).
    points = e == 2 ? hcat(pts, reshape(T(height_scale) .* z, :, 1)) : pts
    poly = pv.PolyData(points)
    poly.lines = create_vtk_line_connectivity(k, size(points, 1))
    poly.point_data.set_scalars(z, "u")

    drawn = poly
    if !isnothing(tube)
        ext = vec(maximum(points, dims=1) .- minimum(points, dims=1))
        rad = T(0.01) * sqrt(sum(abs2, ext))
        drawn = poly.tube(; merge((radius=rad,), NamedTuple(tube))...)
    end

    pl = _mgb_plotter(plotter)
    pl.add_mesh(drawn, scalars="u"; show_scalar_bar=false, mesh...)
    show_grid && pl.show_grid()
    isnothing(camera_position) ? pl.reset_camera() : (pl.camera_position = camera_position)
    isnothing(scalar_bar_args) || pl.add_scalar_bar(; scalar_bar_args...)
    return _mgb_screenshot(pl)
end

"""
    plot(M::Geometry{...,FEM3D}, ts::AbstractVector, U::Matrix; frame_time, kwargs...)

Animate a time series of 3D solutions (columns of `U` at times `ts`) into an
`HTML5anim` MP4 via ffmpeg. Color limits and isosurfaces are fixed across frames
from the global range of `U`.
"""
function plot(M::Geometry{T,X,W,<:Any,FEM3D{E,T}}, ts::AbstractVector, U::Matrix{T};
              frame_time::Real = max(0.001, minimum(diff(ts))),
              kwargs...) where {T,X,W,E}

    nframes = size(U, 2)
    if length(ts) != nframes
        error("length(ts)=$(length(ts)) must equal number of frames=$(nframes)")
    end
    if any(diff(ts) .< 0)
        error("ts must be nondecreasing")
    end

    global_min = minimum(U)
    global_max = maximum(U)
    clim = (global_min, global_max)

    plot_kwargs = Dict{Symbol,Any}(kwargs)
    if !haskey(plot_kwargs, :isosurfaces)
        plot_kwargs[:isosurfaces] = [0.1, 0.3, 0.5, 0.7, 0.9] .* (global_max - global_min) .+ global_min
    end

    volume_kw = get(plot_kwargs, :volume, (;))
    if !isnothing(volume_kw)
        volume_kw = NamedTuple(volume_kw)
        if !haskey(volume_kw, :clim)
            plot_kwargs[:volume] = merge(volume_kw, (clim=clim,))
        end
    end

    contour_kw = get(plot_kwargs, :contour_mesh, (;))
    contour_kw = NamedTuple(contour_kw)
    if !haskey(contour_kw, :clim)
        plot_kwargs[:contour_mesh] = merge(contour_kw, (clim=clim,))
    end

    ts0 = ts .- ts[1]
    total_time = ts0[end]
    fps = 1.0 / frame_time
    n_video_frames = max(1, Int(floor(total_time / frame_time)) + 1)

    ffmpeg_cmd = `$(ffmpeg()) -y -loglevel quiet -f image2pipe -framerate $fps -i pipe:0 -c:v libx264 -pix_fmt yuv420p -movflags frag_keyframe+empty_moov -f mp4 pipe:1`

    mp4_bytes = UInt8[]
    open(ffmpeg_cmd, "r+") do proc
        writer = @async begin
            current_idx = 1
            for j in 0:n_video_frames-1
                t = min(j * frame_time, total_time)
                while current_idx < nframes && ts0[current_idx + 1] <= t
                    current_idx += 1
                end
                fig = plot(M, U[:, current_idx]; plot_kwargs...)
                write(proc, fig.png)
            end
            close(proc.in)
        end
        mp4_bytes = read(proc)
        wait(writer)
    end

    b64 = base64encode(mp4_bytes)
    html = """<video controls autoplay loop>
        <source src="data:video/mp4;base64,$b64" type="video/mp4">
    </video>"""

    return HTML5anim(html)
end

"""
    plot(sol::ParabolicSOL, k::Int=1; kwargs...)

Animate a 3D parabolic solution (component `k`).
"""
function plot(sol::ParabolicSOL{T,X,W,<:Any,<:Geometry{T,<:Any,<:Any,<:Any,FEM3D{E,T}}}, k::Int=1; kwargs...) where {T,X,W,E}
    return plot(sol.geometry, collect(sol.ts), hcat([sol.u[j][:, k] for j=1:length(sol.ts)]...); kwargs...)
end
