# MultiGridBarrierGmshExt — Gmsh mesh import for MultiGridBarrier, as a package
# extension. Loads automatically when both MultiGridBarrier and Gmsh are imported
# (`using MultiGridBarrier, Gmsh`). Supplies the methods of the parent stub
# `gmsh_import` (src/gmsh_frontend.jl).
#
# What it does: reads the highest-dimensional elements of a Gmsh mesh, chooses the
# matching MultiGridBarrier FEM family, builds the `K` coordinate tensor in that
# family's local node order (with orientation fixing), builds exact connectivity
# from the Gmsh node tags, constructs the `Geometry`, and converts every Gmsh
# physical group into a `(vertex, element)` node-pair list in the same format as
# `find_boundary` (directly usable as `dirichlet_nodes` values and in `On(...)`).
#
# Family selection (single element type required):
#   3-node triangles  -> fem2d_P1                      (K = corners)
#   6-node triangles  -> fem2d_P2                      (K = c1,e12,c2,e23,c3,e31,centroid;
#                                                       curved edges supported; the centroid
#                                                       is placed at the P2 map's barycenter)
#   4/9-node quads    -> fem2d, k = 1/2                (tensor order, t via tensor_dofmap;
#                                                       non-planar quad meshes -> ambient=Val(3))
#   8/27-node hexes   -> fem3d, k = 1/2                (tensor order, t via tensor_dofmap)
#
# Orders >= 3 are rejected: Gmsh places high-order nodes equispaced, which only
# coincides with the tensor family's reference nodes up to k = 2 ({-1,0,1}).
# Serendipity elements (8-node quad, 20-node hex), tetrahedra, prisms and pyramids
# are rejected with actionable messages.
module MultiGridBarrierGmshExt

using Gmsh: gmsh
import MultiGridBarrier
import MultiGridBarrier: Geometry, fem2d_P1, fem2d_P2, fem2d, fem3d, tensor_dofmap

_gmsh_err(msg) = throw(ArgumentError(msg))

# ---------------------------------------------------------------------------
# Element-type table (Gmsh element type codes)
# ---------------------------------------------------------------------------

# code => (family, dim, order, nnodes)
const _SUPPORTED = Dict{Int,Tuple{Symbol,Int,Int,Int}}(
    2  => (:tri,  2, 1, 3),
    9  => (:tri,  2, 2, 6),
    3  => (:quad, 2, 1, 4),
    10 => (:quad, 2, 2, 9),
    5  => (:hex,  3, 1, 8),
    12 => (:hex,  3, 2, 27),
)

function _reject_element(etype::Int)
    name = try
        first(gmsh.model.mesh.getElementProperties(etype))
    catch
        "element type $etype"
    end
    if etype in (16, 17)      # 8-node quad / 20-node hex (serendipity)
        _gmsh_err("gmsh_import: $name is a serendipity element; generate complete " *
                  "elements with gmsh.option.setNumber(\"Mesh.SecondOrderIncomplete\", 0)")
    elseif etype in (4, 11, 29, 30, 31)   # tetrahedra
        _gmsh_err("gmsh_import: $name is not supported (MultiGridBarrier has no " *
                  "simplicial 3D element). Mesh with hexahedra instead: transfinite/swept " *
                  "volumes, or gmsh.option.setNumber(\"Mesh.SubdivisionAlgorithm\", 2) " *
                  "to subdivide tetrahedra into hexahedra.")
    elseif etype in (6, 7, 13, 14, 18, 19)  # prisms / pyramids
        _gmsh_err("gmsh_import: $name is not supported; use an all-quad (2D) or " *
                  "all-hex (3D) mesh.")
    else
        _gmsh_err("gmsh_import: $name is not supported. Supported: 3/6-node triangles, " *
                  "4/9-node quadrilaterals, 8/27-node hexahedra (orders 1 and 2; for " *
                  "order-2 meshes call gmsh.model.mesh.setOrder(2) with " *
                  "Mesh.SecondOrderIncomplete = 0).")
    end
end

# ---------------------------------------------------------------------------
# Node-order permutations (Gmsh local order -> MultiGridBarrier local order)
# ---------------------------------------------------------------------------

# Tensor families: match Gmsh's reference-element node coordinates (from
# getElementProperties) against the (k+1)^d tensor grid over {-1,0,1}^d in
# MultiGridBarrier's order (axis 1 fastest). Computing the permutation
# numerically avoids hardcoding Gmsh's node-numbering tables.
function _tensor_perm(etype::Int, k::Int, d::Int)
    props = gmsh.model.mesh.getElementProperties(etype)
    # props = (name, dim, order, numNodes, localNodeCoord, numPrimaryNodes)
    local_coords = props[5]          # flattened, `dim` parametric coords per node
    nn = (k + 1)^d
    length(local_coords) == d * nn ||
        _gmsh_err("gmsh_import: internal error: element type $etype has " *
                  "$(length(local_coords) ÷ d) reference nodes, expected $nn")
    ref = reshape(Float64.(local_coords), d, nn)   # gmsh reference in [-1,1]^d
    coords1d = k == 1 ? [-1.0, 1.0] : [-1.0, 0.0, 1.0]
    perm = zeros(Int, nn)
    for (i, c) in enumerate(Iterators.product(ntuple(_ -> coords1d, d)...))
        target = collect(c)          # MGB node i sits at `target` (axis 1 fastest)
        j = findfirst(col -> maximum(abs.(ref[:, col] .- target)) < 1e-8, 1:nn)
        j === nothing && _gmsh_err("gmsh_import: internal error matching reference " *
                                   "nodes of element type $etype (node $i at $target)")
        perm[i] = j
    end
    allunique(perm) || _gmsh_err("gmsh_import: internal error: non-bijective node match")
    return perm                      # MGB local v  <-  gmsh local perm[v]
end

# Reversal of tensor axis 1 (used to fix negatively-oriented elements):
# lexicographic index (i, rest...) -> (k+2-i, rest...).
function _flip_axis1(k::Int, d::Int)
    s = k + 1
    n = s^d
    p = zeros(Int, n)
    for lin in 1:n
        i0 = (lin - 1) % s
        rest = (lin - 1) ÷ s
        p[lin] = 1 + (s - 1 - i0) + rest * s
    end
    return p
end

# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

# The single volume element block of the mesh: (etype, family, dim, order, conn)
# where conn is (nnodes_gmsh_order, N) of gmsh node tags.
function _volume_block()
    dim = gmsh.model.getDimension()
    while dim > 0
        types, _, nodetags = gmsh.model.mesh.getElements(dim, -1)
        isempty(types) && (dim -= 1; continue)
        length(types) > 1 && _gmsh_err(
            "gmsh_import: the mesh mixes element types in dimension $dim " *
            "(MultiGridBarrier needs a single element type). For quads, use full " *
            "recombination (Mesh.RecombinationAlgorithm) or " *
            "Mesh.SubdivisionAlgorithm = 1.")
        etype = Int(types[1])
        haskey(_SUPPORTED, etype) || _reject_element(etype)
        family, edim, order, nn = _SUPPORTED[etype]
        conn = reshape(Int.(nodetags[1]), nn, :)
        return etype, family, edim, order, conn
    end
    _gmsh_err("gmsh_import: the model has no mesh elements; call " *
              "gmsh.model.mesh.generate(dim) first")
end

# tag -> coordinates (3 columns; gmsh always stores xyz)
function _node_coords()
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = reshape(Float64.(coords), 3, :)
    d = Dict{Int,NTuple{3,Float64}}()
    for (j, t) in enumerate(tags)
        d[Int(t)] = (xyz[1, j], xyz[2, j], xyz[3, j])
    end
    return d
end

# ---------------------------------------------------------------------------
# Family builders: (K, conn_mgb) where conn_mgb is (V, N) gmsh node tags in
# MultiGridBarrier local order (after orientation fixes).
# ---------------------------------------------------------------------------

_signed_area(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

function _build_tri(::Type{T}, order::Int, conn::Matrix{Int}, xyz) where {T}
    N = size(conn, 2)
    planar = all(t -> abs(xyz[t][3]) < 1e-12, vec(conn))
    planar || _gmsh_err("gmsh_import: triangle meshes must be planar (z = 0); " *
                        "for surface meshes in 3D use quadrilaterals (tensor fem2d " *
                        "supports ambient = Val(3)).")
    if order == 1
        V = 3
        K = Array{T,3}(undef, 3, N, 2)
        conn_mgb = Matrix{Int}(undef, 3, N)
        for e in 1:N
            t = conn[:, e]
            a, b, c = xyz[t[1]], xyz[t[2]], xyz[t[3]]
            if _signed_area(a, b, c) < 0
                t = t[[1, 3, 2]]
            end
            conn_mgb[:, e] = t
            for v in 1:3, dd in 1:2
                K[v, e, dd] = T(xyz[t[v]][dd])
            end
        end
        return K, conn_mgb
    else
        # gmsh 6-node triangle: c1 c2 c3 e12 e23 e31; MGB P2+bubble layout:
        # c1, e12, c2, e23, c3, e31, centroid. Curved edges pass through; the
        # centroid (gmsh has none) is the P2 map's image of the barycenter:
        # (-1/9)(c1+c2+c3) + (4/9)(e12+e23+e31).
        K = Array{T,3}(undef, 7, N, 2)
        conn_mgb = Matrix{Int}(undef, 6, N)   # tags for the 6 boundary-relevant nodes
        for e in 1:N
            t = conn[:, e]
            if _signed_area(xyz[t[1]], xyz[t[2]], xyz[t[3]]) < 0
                t = t[[1, 3, 2, 6, 5, 4]]     # swap c2<->c3: e12<->e31, e23 fixed
            end
            mgb = t[[1, 4, 2, 5, 3, 6]]       # c1 e12 c2 e23 c3 e31
            conn_mgb[:, e] = mgb
            for v in 1:6, dd in 1:2
                K[v, e, dd] = T(xyz[mgb[v]][dd])
            end
            for dd in 1:2
                K[7, e, dd] = (-(K[1, e, dd] + K[3, e, dd] + K[5, e, dd]) +
                               4 * (K[2, e, dd] + K[4, e, dd] + K[6, e, dd])) / 9
            end
        end
        return K, conn_mgb
    end
end

function _build_tensor(::Type{T}, etype::Int, k::Int, d::Int, conn::Matrix{Int},
                       xyz) where {T}
    N = size(conn, 2)
    nn = (k + 1)^d
    perm = _tensor_perm(etype, k, d)
    flip = _flip_axis1(k, d)
    # ambient dimension: quads may live in 3D (embedded surface); hexes are 3D.
    ambient = d == 3 ? 3 :
              (all(t -> abs(xyz[t][3]) < 1e-12, vec(conn)) ? 2 : 3)
    K = Array{T,3}(undef, nn, N, ambient)
    conn_mgb = Matrix{Int}(undef, nn, N)
    s = k + 1
    # corner linear indices in MGB tensor order (positions with coord in {-1,1})
    cornerlin = d == 2 ? [1, s, s * (s - 1) + 1, s * s] :
                         [1, s, s * (s - 1) + 1, s * s,
                          s^2 * (s - 1) + 1, s^2 * (s - 1) + s,
                          s^2 * (s - 1) + s * (s - 1) + 1, s^3]
    for e in 1:N
        t = conn[perm, e]             # gmsh -> MGB tensor order
        c = [xyz[t[cl]] for cl in cornerlin]
        # orientation from the corner (multi)linear map at the element center
        neg = if d == 2 && ambient == 2
            _signed_area(c[1], c[2], c[3]) < 0
        elseif d == 3
            u = (c[2][1] - c[1][1], c[2][2] - c[1][2], c[2][3] - c[1][3])
            v = (c[3][1] - c[1][1], c[3][2] - c[1][2], c[3][3] - c[1][3])
            w = (c[5][1] - c[1][1], c[5][2] - c[1][2], c[5][3] - c[1][3])
            det3 = u[1] * (v[2] * w[3] - v[3] * w[2]) -
                   u[2] * (v[1] * w[3] - v[3] * w[1]) +
                   u[3] * (v[1] * w[2] - v[2] * w[1])
            det3 < 0
        else
            false                     # embedded surface: orientation is chartwise
        end
        neg && (t = t[flip])
        conn_mgb[:, e] = t
        for v in 1:nn, dd in 1:ambient
            K[v, e, dd] = T(xyz[t[v]][dd])
        end
    end
    return K, conn_mgb, ambient
end

# ---------------------------------------------------------------------------
# Physical groups -> (vertex, element) pairs
# ---------------------------------------------------------------------------

function _regions(conn_mgb::Matrix{Int})
    tag2pairs = Dict{Int,Vector{Tuple{Int,Int}}}()
    V, N = size(conn_mgb)
    for e in 1:N, v in 1:V
        push!(get!(() -> Tuple{Int,Int}[], tag2pairs, conn_mgb[v, e]), (v, e))
    end
    regions = Dict{String,Vector{Tuple{Int,Int}}}()
    for (dim, ptag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(dim, ptag)
        isempty(name) && (name = "dim$(dim)_tag$(ptag)")
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, ptag)
        pairs = Tuple{Int,Int}[]
        for t in tags
            append!(pairs, get(tag2pairs, Int(t), Tuple{Int,Int}[]))
        end
        sort!(pairs)
        regions[name] = pairs
    end
    return regions
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

function _import_current(::Type{T}) where {T}
    etype, family, d, order, conn = _volume_block()
    xyz = _node_coords()
    if family === :tri
        K, conn_mgb = _build_tri(T, order, conn, xyz)
        geom = order == 1 ? fem2d_P1(T; K) : fem2d_P2(T; K)
        return (; geometry = geom, regions = _regions(conn_mgb))
    else
        K, conn_mgb, ambient = _build_tensor(T, etype, order, d, conn, xyz)
        # exact connectivity from the gmsh corner tags (no coordinate dedup)
        s = order + 1
        cornerlin = d == 2 ? [1, s, s * (s - 1) + 1, s * s] :
                             [1, s, s * (s - 1) + 1, s * s,
                              s^2 * (s - 1) + 1, s^2 * (s - 1) + s,
                              s^2 * (s - 1) + s * (s - 1) + 1, s^3]
        corner_tags = conn_mgb[cornerlin, :]
        compact = Dict{Int,Int}()
        t_corner = Matrix{Int}(undef, size(corner_tags)...)
        for j in eachindex(corner_tags)
            t_corner[j] = get!(compact, corner_tags[j], length(compact) + 1)
        end
        t_full = tensor_dofmap(t_corner, order, Val(d))
        geom = if d == 2
            fem2d(T; k = order, K, t = t_full, ambient = Val(ambient))
        else
            fem3d(T; k = order, K, t = t_full)
        end
        return (; geometry = geom, regions = _regions(conn_mgb))
    end
end

function MultiGridBarrier.gmsh_import(; verbose::Bool = false, T::Type = Float64)
    Bool(gmsh.isInitialized()) ||
        _gmsh_err("gmsh_import(): gmsh is not initialized; script your geometry " *
                  "(gmsh.initialize(); ...; gmsh.model.mesh.generate(dim)) or pass " *
                  "a file path: gmsh_import(\"mesh.msh\")")
    gmsh.option.setNumber("General.Terminal", verbose ? 1 : 0)
    return _import_current(T)
end

function MultiGridBarrier.gmsh_import(path::AbstractString; verbose::Bool = false,
                                      T::Type = Float64)
    isfile(path) || _gmsh_err("gmsh_import: file not found: $path")
    we_initialized = !Bool(gmsh.isInitialized())
    we_initialized && gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", verbose ? 1 : 0)
        gmsh.open(path)
        if endswith(lowercase(path), ".geo") || isempty(gmsh.model.mesh.getNodes()[1])
            gmsh.model.mesh.generate(gmsh.model.getDimension())
        end
        return _import_current(T)
    finally
        we_initialized && gmsh.finalize()
    end
end

end # module
