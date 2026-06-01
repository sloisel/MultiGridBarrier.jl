# plot3d.jl -- 3D (FEM3D = TensorFEM{3}) PyVista volume/isosurface/slice plotting
# and ffmpeg-based HTML5 animation. Relocated from the former Mesh3d submodule;
# dispatches on the TensorFEM{3,T} discretization. Geometry/MGBSOL/ParabolicSOL/
# HTML5anim are the enclosing module's types (in scope).

using PyCall
import PyPlot: savefig
using FileIO
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
function plot(geo::Geometry{T,X,W,<:Any,FEM3D{T}}, u::Vector{T};
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
                       kwargs...) where {T,X,W}

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

    # Reuse plotter to avoid macOS "Context leak detected" error
    if isnothing(_plotter[])
        _plotter[] = pv.Plotter(off_screen=true; plotter...)
    end
    pl = _plotter[]

    # Clear previous actors and reset state
    pl.clear()
    pl.remove_bounds_axes()
    pl.enable_lightkit()  # Re-enable default lighting (clear() removes lights)

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

    # Render to numpy array (use screenshot instead of show to keep plotter alive)
    img_np = pl.screenshot(return_img=true)
    img_array = convert(Array, img_np)

    buf = IOBuffer()
    PNGFiles.save(buf, img_array)
    seekstart(buf)
    png_data = take!(buf)

    return MGB3DFigure(png_data)
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

"""
    plot(M::Geometry{...,FEM3D}, ts::AbstractVector, U::Matrix; frame_time, kwargs...)

Animate a time series of 3D solutions (columns of `U` at times `ts`) into an
`HTML5anim` MP4 via ffmpeg. Color limits and isosurfaces are fixed across frames
from the global range of `U`.
"""
function plot(M::Geometry{T,X,W,<:Any,FEM3D{T}}, ts::AbstractVector, U::Matrix{T};
              frame_time::Real = max(0.001, minimum(diff(ts))),
              kwargs...) where {T,X,W}

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
function plot(sol::ParabolicSOL{T,X,W,<:Any,<:Geometry{T,<:Any,<:Any,<:Any,FEM3D{T}}}, k::Int=1; kwargs...) where {T,X,W}
    return plot(sol.geometry, collect(sol.ts), hcat([sol.u[j][:, k] for j=1:length(sol.ts)]...); kwargs...)
end
