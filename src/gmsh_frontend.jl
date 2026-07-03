# Gmsh import API surface: stub only.
#
# Mesh import from Gmsh is a package extension (ext/MultiGridBarrierGmshExt),
# loaded automatically when both MultiGridBarrier and Gmsh are imported. The
# extension supplies the methods of the stub declared here. Calling the stub
# without Gmsh loaded raises a MethodError; see the "Meshes from Gmsh" manual
# page for usage.

"""
    gmsh_import(path::AbstractString; verbose=false, T=Float64) -> (; geometry, regions)
    gmsh_import(; verbose=false, T=Float64)                     -> (; geometry, regions)

Import a Gmsh mesh as a MultiGridBarrier `Geometry` (requires `using Gmsh`,
which loads the `MultiGridBarrierGmshExt` extension). The first form opens a
`.msh`/`.geo` file; the zero-argument form reads the **current** Gmsh model
(script your geometry through the `gmsh` API, call
`gmsh.model.mesh.generate(dim)`, then `gmsh_import()`).

Returns a named tuple:

- `geometry::Geometry` — ready for `amg(geometry)`. The FEM family is chosen
  from the mesh's elements: 3-node triangles → `fem2d_P1`, 6-node triangles →
  `fem2d_P2` (isoparametric, curved edges supported), `(k+1)²`-node quadrilaterals
  → tensor `fem2d` of **any order k**, `(k+1)³`-node hexahedra → `fem3d` of **any
  order k**. High-order quad/hex geometry is obtained by resampling each element
  at MultiGridBarrier's reference nodes, so curved elements of any order import
  correctly. Quadrilateral surface meshes with non-planar coordinates become
  embedded 2-manifolds (`ambient = Val(3)`). The mesh must consist of a single
  element type; serendipity (incomplete high-order) elements, order-≥3 triangles
  (MultiGridBarrier has only P1/P2 triangles), tetrahedra, prisms and pyramids are
  rejected with instructions (e.g. `Mesh.SecondOrderIncomplete = 0`, or
  `Mesh.SubdivisionAlgorithm = 2` to turn tetrahedra into hexahedra).
- `regions::Dict{String,Vector{Tuple{Int,Int}}}` — one entry per Gmsh
  **physical group**, mapping its name to the `(vertex, element)` node pairs of
  the volume mesh that lie on the group. This is the same format as
  [`find_boundary`](@ref), so entries plug directly into `amg`'s
  `dirichlet_nodes` and the JuMP front end's [`On`](@ref):

```julia
using MultiGridBarrier, Gmsh
gm  = gmsh_import("domain.msh")
mg  = amg(gm.geometry; dirichlet_nodes = Dict(:dirichlet => gm.regions["clamped"]))
sol = mgb_solve(assemble(mg; p = 1.5))
```

Physical groups of the volume dimension select subdomains (useful with `On`);
lower-dimensional groups select boundary/interface node sets. Unnamed groups
get the key `"dim<d>_tag<t>"`. Element connectivity is taken from the Gmsh node
tags (shared nodes glue exactly; no coordinate tolerance), and elements with
negative orientation are flipped automatically.
"""
function gmsh_import end

export gmsh_import
