module Plotting

using PyCall
import PyPlot: plot, savefig
using ..Mesh3d: FEM3D
using ...MultiGridBarrier: Geometry, AMGBSOL, ParabolicSOL, HTML5anim
using FileIO
using PNGFiles
using Base64
using FFMPEG: ffmpeg

export plot, savefig, MGB3DFigure, HTML5anim

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

Save the figure to a file (e.g., "plot.png").
Extends `PyPlot.savefig`.
"""
function savefig(fig::MGB3DFigure, filename::String)
    write(filename, fig.png)
end

"""
    plot(sol::AMGBSOL, k=1; kwargs...)
    plot(geo::Geometry, u::Vector; kwargs...)

Plot a 3D solution using PyVista. The first form plots column `k` of the solution matrix
`sol.z` (default: first column). The second form plots the vector `u` directly on the geometry.

# Keyword Arguments
- `plotter=(window_size=(800, 600),)`: Options passed to `pv.Plotter()`.
- `volume=(;)`: Options for `add_volume` (e.g., `(cmap="viridis", opacity="sigmoid")`). Pass `nothing` to disable.
- `scalar_bar_args=(title="", ...)`: Options for `add_scalar_bar`. Pass `nothing` to hide.
- `isosurfaces`: Vector of isosurface values. Defaults to 5 evenly spaced values across the solution range. Pass `[]` to disable.
- `contour_mesh=(;)`: Options for `add_mesh` for the isosurfaces.
- `slice_orthogonal=nothing`: Options for `slice_orthogonal` filter (e.g., `(x=0.5, y=0.5)`).
- `slice_orthogonal_mesh=(;)`: Options for `add_mesh` for orthogonal slices.
- `slice=nothing`: Options for `slice` filter (e.g., `(normal="z",)` or `(normal=[1,1,1], origin=[0,0,0])`).
- `slice_mesh=(;)`: Options for `add_mesh` for slices.
- `slice_along_axis=nothing`: Options for `slice_along_axis` filter.
- `slice_along_axis_mesh=(;)`: Options for `add_mesh` for slices along axis.
- `show_grid=true`: Show the coordinate axes grid.
- `camera_position=nothing`: Camera position.

# Returns
- `MGB3DFigure`: Displays as PNG in Jupyter. Use `savefig(fig, "file.png")` to save.

# Examples

```julia
# Default: volume rendering with 5 isosurfaces
plot(sol)

# Plot the second solution column
plot(sol, 2)

# Volume rendering only (no isosurfaces)
plot(sol; isosurfaces=[])

# Volume rendering with custom colormap and opacity
plot(sol; volume=(cmap="magma", opacity="linear"))

# Disable volume rendering, show only isosurfaces
plot(sol; volume=nothing)

# Custom isosurface values
plot(sol; isosurfaces=[0.1, 0.5], contour_mesh=(color="black", opacity=0.5))

# Orthogonal slices at specific coordinates
plot(sol; slice_orthogonal=(x=0.0, y=0.0, z=0.0))

# Slice along a normal vector
plot(sol; slice=(normal=[1,1,1], origin=[0,0,0]))

# High resolution output
plot(sol; plotter=(window_size=(1600, 1200),))
```
"""
function plot(geo::Geometry{T,X,W,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}, u::Vector{T};
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

    # Create unstructured grid
    points = geo.x
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
            pl.add_mesh(contours; show_scalar_bar=false, contour_mesh...) #, opacity=0.5, show_scalar_bar=false, cmap=cmap)
        else
            @warn "Isosurface generation resulted in an empty mesh. Check if isosurface values are within the range of the solution."
        end
    end

    # New slicing options passed as dictionaries
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

    # Convert to Julia array
    img_array = convert(Array, img_np)

    # Save to IOBuffer as PNG
    buf = IOBuffer()
    PNGFiles.save(buf, img_array)
    seekstart(buf)
    png_data = take!(buf)

    return MGB3DFigure(png_data)
end



function create_vtk_cells(k::Int, n_total_nodes::Int)
    # Subdivide each Q_k element into k^3 linear hexes
    n_nodes_per_elem = (k+1)^3
    n_elems = div(n_total_nodes, n_nodes_per_elem)

    # Number of sub-cells per element = k^3
    n_subcells = n_elems * k^3

    # VTK cell array format: [n_nodes, id1, id2, ..., n_nodes, id1, ...]
    # For Hex, n_nodes = 8.
    # Total size = n_subcells * 9

    cells = Vector{Int}(undef, n_subcells * 9)
    cell_types = fill(UInt8(VTK_HEXAHEDRON), n_subcells)

    # Helper to get local index in (k+1)^3 array
    idx(ix, iy, iz) = (iz-1)*(k+1)^2 + (iy-1)*(k+1) + ix

    # VTK Hex ordering relative to (0,0,0)-(1,1,1) box
    # 0: 000, 1: 100, 2: 110, 3: 010
    # 4: 001, 5: 101, 6: 111, 7: 011

    # Map to our local indices (1-based)
    # (ix, iy, iz) from 1 to k
    # corners:
    # 0: (ix,   iy,   iz)
    # 1: (ix+1, iy,   iz)
    # 2: (ix+1, iy+1, iz)
    # 3: (ix,   iy+1, iz)
    # 4: (ix,   iy,   iz+1)
    # 5: (ix+1, iy,   iz+1)
    # 6: (ix+1, iy+1, iz+1)
    # 7: (ix,   iy+1, iz+1)

    ptr = 1
    for e in 1:n_elems
        base_node_idx = (e-1) * n_nodes_per_elem

        for iz in 1:k
            for iy in 1:k
                for ix in 1:k
                    cells[ptr] = 8 # Number of nodes

                    # VTK 0
                    cells[ptr+1] = base_node_idx + idx(ix, iy, iz) - 1 # 0-based for PyVista/VTK?
                    # VTK 1
                    cells[ptr+2] = base_node_idx + idx(ix+1, iy, iz) - 1
                    # VTK 2
                    cells[ptr+3] = base_node_idx + idx(ix+1, iy+1, iz) - 1
                    # VTK 3
                    cells[ptr+4] = base_node_idx + idx(ix, iy+1, iz) - 1
                    # VTK 4
                    cells[ptr+5] = base_node_idx + idx(ix, iy, iz+1) - 1
                    # VTK 5
                    cells[ptr+6] = base_node_idx + idx(ix+1, iy, iz+1) - 1
                    # VTK 6
                    cells[ptr+7] = base_node_idx + idx(ix+1, iy+1, iz+1) - 1
                    # VTK 7
                    cells[ptr+8] = base_node_idx + idx(ix, iy+1, iz+1) - 1

                    ptr += 9
                end
            end
        end
    end

    return cells, cell_types
end

"""
    plot(M::Geometry{T,X,W,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}, ts::AbstractVector, U::Matrix{T}; kwargs...)

Create an animated 3D visualization from a time series of solutions.

# Arguments
- `M`: The geometry (must be FEM3D)
- `ts`: Vector of timestamps (length must equal number of columns in U)
- `U`: Matrix of solutions, each column is a frame (n_nodes × n_frames)

# Keyword Arguments
- `frame_time::Real`: Time between video frames in seconds (default: `max(0.001, minimum(diff(ts)))`)
- All other kwargs are passed to the static `plot` function for each frame.

# Returns
- `HTML5anim`: An HTML5 video that displays in Jupyter notebooks.

# Notes
The animation uses ffmpeg to encode PNG frames into an MP4 video. Frames are
piped directly to ffmpeg without temporary files. The video is embedded as
base64-encoded data in an HTML5 video tag.

Color limits (`clim`) and `isosurfaces` are automatically computed from the global
range of U across all frames to ensure consistent visualization throughout the animation.
"""
function plot(M::Geometry{T,X,W,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}, ts::AbstractVector, U::Matrix{T};
              frame_time::Real = max(0.001, minimum(diff(ts))),
              kwargs...) where {T,X,W}

    nframes = size(U, 2)

    if length(ts) != nframes
        error("length(ts)=$(length(ts)) must equal number of frames=$(nframes)")
    end
    if any(diff(ts) .< 0)
        error("ts must be nondecreasing")
    end

    # Compute global color limits across ALL frames for consistent visualization
    global_min = minimum(U)
    global_max = maximum(U)
    clim = (global_min, global_max)

    # Build plot kwargs with fixed limits (user can override)
    plot_kwargs = Dict{Symbol,Any}(kwargs)

    # Set fixed isosurfaces from global range (unless user provided)
    if !haskey(plot_kwargs, :isosurfaces)
        plot_kwargs[:isosurfaces] = [0.1, 0.3, 0.5, 0.7, 0.9] .* (global_max - global_min) .+ global_min
    end

    # Inject clim into volume kwargs (unless user provided or volume=nothing)
    volume_kw = get(plot_kwargs, :volume, (;))
    if !isnothing(volume_kw)
        volume_kw = NamedTuple(volume_kw)
        if !haskey(volume_kw, :clim)
            plot_kwargs[:volume] = merge(volume_kw, (clim=clim,))
        end
    end

    # Inject clim into contour_mesh kwargs (unless user provided)
    contour_kw = get(plot_kwargs, :contour_mesh, (;))
    contour_kw = NamedTuple(contour_kw)
    if !haskey(contour_kw, :clim)
        plot_kwargs[:contour_mesh] = merge(contour_kw, (clim=clim,))
    end

    # Build a fixed-FPS timeline
    ts0 = ts .- ts[1]  # relative times starting at 0
    total_time = ts0[end]
    fps = 1.0 / frame_time
    n_video_frames = max(1, Int(floor(total_time / frame_time)) + 1)

    # Setup ffmpeg pipeline: PNG input via stdin, MP4 output via stdout
    ffmpeg_cmd = `$(ffmpeg()) -y -loglevel quiet -f image2pipe -framerate $fps -i pipe:0 -c:v libx264 -pix_fmt yuv420p -movflags frag_keyframe+empty_moov -f mp4 pipe:1`

    mp4_bytes = UInt8[]

    open(ffmpeg_cmd, "r+") do proc
        # Writer task: generate frames and write to ffmpeg stdin
        writer = @async begin
            current_idx = 1
            for j in 0:n_video_frames-1
                t = min(j * frame_time, total_time)
                # Advance to the latest data frame not exceeding time t
                while current_idx < nframes && ts0[current_idx + 1] <= t
                    current_idx += 1
                end
                # Generate frame with fixed limits
                fig = plot(M, U[:, current_idx]; plot_kwargs...)
                write(proc, fig.png)
            end
            close(proc.in)
        end

        # Read MP4 output
        mp4_bytes = read(proc)
        wait(writer)
    end

    # Create HTML5 video tag with embedded base64 data
    b64 = base64encode(mp4_bytes)
    html = """<video controls autoplay loop>
        <source src="data:video/mp4;base64,$b64" type="video/mp4">
    </video>"""

    return HTML5anim(html)
end

"""
    plot(sol::ParabolicSOL, k::Int=1; kwargs...)

Plot an animated 3D visualization of a parabolic solution (FEM3D).

# Arguments
- `sol`: The parabolic solution from `parabolic_solve`
- `k`: Which solution component to plot (default: 1)
- `kwargs...`: Passed to the animation plot method

# Returns
- `HTML5anim`: An HTML5 video that displays in Jupyter notebooks.
"""
function plot(sol::ParabolicSOL{T,X,W,<:Any,<:Geometry{T,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}}, k::Int=1; kwargs...) where {T,X,W}
    return plot(sol.geometry, collect(sol.ts), hcat([sol.u[j][:, k] for j=1:length(sol.ts)]...); kwargs...)
end

end # module
