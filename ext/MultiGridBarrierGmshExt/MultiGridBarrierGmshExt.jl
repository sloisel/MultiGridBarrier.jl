# MultiGridBarrierGmshExt — Gmsh mesh import for MultiGridBarrier, as a package
# extension. Loads automatically when both MultiGridBarrier and Gmsh are imported
# (`using MultiGridBarrier, Gmsh`). Supplies the methods of the parent stub
# `gmsh_import` (src/gmsh_frontend.jl).
#
# What it does: reads the highest-dimensional elements of a Gmsh mesh, chooses the
# matching MultiGridBarrier FEM family, builds the `K` coordinate tensor in that
# family's local node order, builds exact connectivity from the Gmsh node tags,
# constructs the `Geometry`, and converts every Gmsh physical group into a
# `(vertex, element)` node-pair list (the same format as `find_boundary`, so it
# feeds `dirichlet_nodes` and the JuMP front end's `On`).
#
# Family selection (single element type required):
#   3-node triangles  -> fem2d_P1
#   6-node triangles  -> fem2d_P2   (curved edges; the bubble node is placed at the
#                                    P2 map's image of the barycenter)
#   (k+1)^2-node quads -> fem2d, order k  (any order k; curved; non-planar quad
#                                          meshes become embedded ambient=Val(3) surfaces)
#   (k+1)^3-node hexes -> fem3d, order k  (any order k; curved)
#
# The tensor families use Chebyshev-Lobatto reference nodes (-cos(iπ/k)), which do
# NOT coincide with Gmsh's equispaced high-order nodes for k >= 3. Rather than
# permuting Gmsh's node tags (correct only through k = 2), the coordinates are
# RESAMPLED: `getJacobians` evaluates each element's isoparametric geometry at
# MultiGridBarrier's Chebyshev reference nodes, so any order imports correctly.
# Because Gmsh's high-order boundary nodes then no longer coincide with the DOFs,
# physical-group membership is decided by a corner-subset test (see `_regions_tensor`)
# rather than by node identity. Triangles (P1/P2) keep the direct node mapping.
#
# Serendipity elements (incomplete high-order quad/hex), P3+ triangles,
# tetrahedra, prisms, pyramids, line elements and mixed-type meshes are rejected
# with actionable messages.
module MultiGridBarrierGmshExt

using Gmsh: gmsh
import MultiGridBarrier
import MultiGridBarrier: Geometry, fem2d_P1, fem2d_P2, fem2d, fem3d, tensor_dofmap

_gmsh_err(msg) = throw(ArgumentError(msg))

# 1D Chebyshev-Lobatto reference nodes, matching TensorFEM's `_tf_nodes`.
_cheb(k::Int) = Float64[-cos(i * π / k) for i in 0:k]

# ---------------------------------------------------------------------------
# Tensor-grid helpers (MultiGridBarrier local order: axis-1 fastest)
# ---------------------------------------------------------------------------

# Linear index of the tensor node whose per-axis indices are `idx` (0-based).
_lin(idx, s) = 1 + sum(idx[a] * s^(a - 1) for a in 1:length(idx))

# Linear indices (in the (k+1)^d grid) of the 2^d corners, in MultiGridBarrier
# corner order (axis-1 fastest over {0,k}).
_corner_linidx(k::Int, d::Int) =
    Int[_lin(ntuple(a -> ((m >> (a - 1)) & 1) * k, d), k + 1) for m in 0:(2^d - 1)]

# Reversal of tensor axis 1: lexicographic (i, rest...) -> (k+2-i, rest...).
# Used to flip a negatively-oriented element. Works on any (k+1)^d grid, and on
# the 2^d corner grid when called with k = 1.
function _flip_axis1(k::Int, d::Int)
    s = k + 1
    n = s^d
    p = Vector{Int}(undef, n)
    for lin in 1:n
        i0 = (lin - 1) % s
        rest = (lin - 1) ÷ s
        p[lin] = 1 + (s - 1 - i0) + rest * s
    end
    return p
end

# Permutation matching MultiGridBarrier's 2^d corner slots to Gmsh's first 2^d
# (primary) element nodes: perm[mgb_slot] = gmsh_corner_index. Computed from the
# reference-node coordinates so no Gmsh numbering table is hardcoded.
function _corner_perm(etype::Int, d::Int)
    props = gmsh.model.mesh.getElementProperties(etype)
    ref = reshape(Float64.(props[5]), d, :)     # (d, numNodes); props[5] = localNodeCoord
    ncorner = 2^d
    perm = Vector{Int}(undef, ncorner)
    for (mgb, idx) in enumerate(Iterators.product(ntuple(_ -> (-1.0, 1.0), d)...))
        target = collect(idx)                   # first axis varies fastest
        j = findfirst(c -> maximum(abs.(ref[:, c] .- target)) < 1e-9, 1:ncorner)
        j === nothing && _gmsh_err("gmsh_import: cannot match corner $mgb of element type $etype")
        perm[mgb] = j
    end
    allunique(perm) || _gmsh_err("gmsh_import: corner permutation is not bijective")
    return perm
end

# Negative orientation from the resampled corner coordinates `cc` (2^d tuples in
# MGB order). Embedded surfaces (d=2 in R^3) are chartwise -- never flipped.
function _neg_orient(cc, d::Int, ambient::Int)
    (d == 2 && ambient == 3) && return false
    if d == 2
        a, b, c = cc[1], cc[2], cc[3]
        return (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]) < 0
    else
        a = cc[1]
        u = (cc[2][1] - a[1], cc[2][2] - a[2], cc[2][3] - a[3])
        v = (cc[3][1] - a[1], cc[3][2] - a[2], cc[3][3] - a[3])
        w = (cc[5][1] - a[1], cc[5][2] - a[2], cc[5][3] - a[3])
        det = u[1] * (v[2] * w[3] - v[3] * w[2]) - u[2] * (v[1] * w[3] - v[3] * w[1]) +
              u[3] * (v[1] * w[2] - v[2] * w[1])
        return det < 0
    end
end

# ---------------------------------------------------------------------------
# Element-type classification (single element type per mesh)
# ---------------------------------------------------------------------------

# Returns (etype, family, d, order, numnodes, numprimary, conn) for the highest
# non-empty dimension. conn is (numnodes, N) of Gmsh node tags. Rejects
# unsupported / mixed meshes with actionable messages.
function _volume_block()
    dim = gmsh.model.getDimension()
    while dim > 0
        types, _, nodetags = gmsh.model.mesh.getElements(dim, -1)
        isempty(types) && (dim -= 1; continue)
        length(types) > 1 && _gmsh_err(
            "gmsh_import: the mesh mixes element types in dimension $dim " *
            "(MultiGridBarrier needs a single element type). For quads use full " *
            "recombination (Mesh.RecombinationAlgorithm) or Mesh.SubdivisionAlgorithm = 1.")
        etype = Int(types[1])
        props = gmsh.model.mesh.getElementProperties(etype)
        name, order, numnodes, numprimary = props[1], Int(props[3]), Int(props[4]), Int(props[6])
        conn = reshape(Int.(nodetags[1]), numnodes, :)
        if occursin("Triangle", name)
            order <= 2 || _gmsh_err("gmsh_import: order-$order triangles are not supported " *
                "(MultiGridBarrier has P1 and P2 triangles only). Use order-2 triangles, " *
                "or quadrilaterals for higher order.")
            return etype, :tri, 2, order, numnodes, numprimary, conn
        elseif occursin("Quadrilateral", name)
            numnodes == (order + 1)^2 || _gmsh_err("gmsh_import: $name is a serendipity " *
                "quadrilateral; generate complete elements with " *
                "gmsh.option.setNumber(\"Mesh.SecondOrderIncomplete\", 0).")
            return etype, :quad, 2, order, numnodes, numprimary, conn
        elseif occursin("Hexahedron", name)
            numnodes == (order + 1)^3 || _gmsh_err("gmsh_import: $name is a serendipity " *
                "hexahedron; generate complete elements with " *
                "gmsh.option.setNumber(\"Mesh.SecondOrderIncomplete\", 0).")
            return etype, :hex, 3, order, numnodes, numprimary, conn
        elseif occursin("Tetrahedron", name)
            _gmsh_err("gmsh_import: $name is not supported (MultiGridBarrier has no simplicial " *
                "3D element). Mesh with hexahedra instead: transfinite/swept volumes, or " *
                "gmsh.option.setNumber(\"Mesh.SubdivisionAlgorithm\", 2) to subdivide " *
                "tetrahedra into hexahedra.")
        elseif occursin("Line", name)
            _gmsh_err("gmsh_import: line elements ($name) are not supported; import a 2D or 3D mesh.")
        else
            _gmsh_err("gmsh_import: $name is not supported; use triangles, quadrilaterals, " *
                "or hexahedra.")
        end
    end
    _gmsh_err("gmsh_import: the model has no mesh elements; call gmsh.model.mesh.generate(dim) first")
end

# tag -> (x, y, z)
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
# Triangle families (P1 / P2): direct node mapping (Gmsh nodes coincide with the
# MultiGridBarrier DOFs at these orders).
# ---------------------------------------------------------------------------

_signed_area(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

function _build_tri(::Type{T}, order::Int, conn::Matrix{Int}, xyz) where {T}
    N = size(conn, 2)
    all(t -> abs(xyz[t][3]) < 1e-12, vec(conn)) ||
        _gmsh_err("gmsh_import: triangle meshes must be planar (z = 0); for surfaces in 3D use " *
                  "quadrilaterals (tensor fem2d supports ambient = Val(3)).")
    if order == 1
        K = Array{T,3}(undef, 3, N, 2)
        conn_mgb = Matrix{Int}(undef, 3, N)
        for e in 1:N
            t = conn[:, e]
            _signed_area(xyz[t[1]], xyz[t[2]], xyz[t[3]]) < 0 && (t = t[[1, 3, 2]])
            conn_mgb[:, e] = t
            for v in 1:3, dd in 1:2
                K[v, e, dd] = T(xyz[t[v]][dd])
            end
        end
        return K, conn_mgb
    else
        # gmsh 6-node triangle: c1 c2 c3 e12 e23 e31; MGB P2+bubble layout is
        # c1, e12, c2, e23, c3, e31, centroid. The centroid is the P2 map's image
        # of the barycenter: (-1/9)(Σ corners) + (4/9)(Σ edge nodes).
        K = Array{T,3}(undef, 7, N, 2)
        conn_mgb = Matrix{Int}(undef, 6, N)
        for e in 1:N
            t = conn[:, e]
            _signed_area(xyz[t[1]], xyz[t[2]], xyz[t[3]]) < 0 && (t = t[[1, 3, 2, 6, 5, 4]])
            mgb = t[[1, 4, 2, 5, 3, 6]]
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

# ---------------------------------------------------------------------------
# Tensor families (quad / hex, any order): resample geometry at the Chebyshev
# reference nodes, so Gmsh's node placement never has to match ours.
# ---------------------------------------------------------------------------

# Returns (K, tcorner, ambient) where tcorner is (2^d, N) of Gmsh corner tags in
# MultiGridBarrier corner order (after the per-element orientation flip).
function _build_tensor(::Type{T}, etype::Int, k::Int, d::Int, conn::Matrix{Int}) where {T}
    N = size(conn, 2)
    s = k + 1
    nn = s^d
    cheb = _cheb(k)
    # MGB reference nodes (axis-1 fastest) as a flat gmsh localCoord (u,v,w per point)
    loc = Float64[]
    for idx in Iterators.product(ntuple(_ -> cheb, d)...)
        push!(loc, idx[1], d >= 2 ? idx[2] : 0.0, d >= 3 ? idx[3] : 0.0)
    end
    _, _, coord = gmsh.model.mesh.getJacobians(etype, loc)
    C = reshape(Float64.(coord), 3, nn, N)               # C[dd, node, elem]
    ambient = d == 3 ? 3 :
              (all(abs(C[3, v, e]) < 1e-10 for v in 1:nn, e in 1:N) ? 2 : 3)

    cperm = _corner_perm(etype, d)
    cornerlin = _corner_linidx(k, d)
    flip_full = _flip_axis1(k, d)
    flip_corner = _flip_axis1(1, d)
    ncorner = 2^d

    K = Array{T,3}(undef, nn, N, ambient)
    tcorner = Matrix{Int}(undef, ncorner, N)
    for e in 1:N
        Ke = Matrix{T}(undef, nn, ambient)
        for v in 1:nn, dd in 1:ambient
            Ke[v, dd] = T(C[dd, v, e])
        end
        cg = conn[1:ncorner, e][cperm]                   # corner tags in MGB order
        cc = ntuple(m -> ntuple(dd -> Ke[cornerlin[m], dd], ambient), ncorner)
        if _neg_orient(cc, d, ambient)
            Ke = Ke[flip_full, :]
            cg = cg[flip_corner]
        end
        K[:, e, :] = Ke
        tcorner[:, e] = cg
    end
    return K, tcorner, ambient
end

# ---------------------------------------------------------------------------
# Physical groups -> (vertex, element) pairs
# ---------------------------------------------------------------------------

# Triangle path: Gmsh nodes coincide with the DOFs, so map tags directly.
function _regions_nodes(conn_mgb::Matrix{Int})
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

_issubset_sorted(a, b) = all(x -> insorted(x, b), a)

# Tensor path: a DOF lies on a physical group iff the global corner tags of its
# reference sub-entity (corner / edge / face) are all contained in a single facet
# of the group. This is exact at any order and needs no node coincidence:
# corners are shared exactly, and a DOF's sub-entity is an element sub-entity, so
# it is on the group iff it sits inside one of the group's facets.
function _regions_tensor(k::Int, d::Int, ncorner::Int, tcorner::Matrix{Int})
    N = size(tcorner, 2)
    s = k + 1
    nn = s^d
    # For each local DOF, the corner slots (1..2^d) spanning its reference sub-entity.
    dof_slots = Vector{Vector{Int}}(undef, nn)
    for v in 1:nn
        idx = ntuple(a -> ((v - 1) ÷ s^(a - 1)) % s, d)       # 0-based per-axis
        fixed = ntuple(a -> idx[a] == 0 ? 0 : (idx[a] == k ? 1 : -1), d)  # side, -1 = free
        slots = Int[]
        for m in 0:(ncorner - 1)
            ok = true
            for a in 1:d
                if fixed[a] != -1 && ((m >> (a - 1)) & 1) != fixed[a]
                    ok = false
                    break
                end
            end
            ok && push!(slots, m + 1)
        end
        dof_slots[v] = slots
    end

    vtypes, vtags, _ = gmsh.model.mesh.getElements(d, -1)
    tag2e = Dict{Int,Int}(Int(t) => e for (e, t) in enumerate(vtags[1]))

    regions = Dict{String,Vector{Tuple{Int,Int}}}()
    for (pg_dim, ptag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(pg_dim, ptag)
        isempty(name) && (name = "dim$(pg_dim)_tag$(ptag)")
        pairs = Tuple{Int,Int}[]
        if pg_dim == d
            # subdomain: every DOF of the group's volume elements
            elems = Set{Int}()
            for ent in gmsh.model.getEntitiesForPhysicalGroup(pg_dim, ptag)
                _, etags, _ = gmsh.model.mesh.getElements(pg_dim, ent)
                for eg in etags, t in eg
                    haskey(tag2e, Int(t)) && push!(elems, tag2e[Int(t)])
                end
            end
            for e in elems, v in 1:nn
                push!(pairs, (v, e))
            end
        else
            # lower-dim group: index its facets by corner tag, then subset-test.
            fac_by_corner = Dict{Int,Vector{Vector{Int}}}()
            for ent in gmsh.model.getEntitiesForPhysicalGroup(pg_dim, ptag)
                ftypes, _, fconn = gmsh.model.mesh.getElements(pg_dim, ent)
                for (ft, fc) in zip(ftypes, fconn)
                    fp = gmsh.model.mesh.getElementProperties(Int(ft))
                    fnn, fnp = Int(fp[4]), Int(fp[6])
                    fcm = reshape(Int.(fc), fnn, :)
                    for c in 1:size(fcm, 2)
                        cs = sort(fcm[1:fnp, c])
                        for t in cs
                            push!(get!(() -> Vector{Int}[], fac_by_corner, t), cs)
                        end
                    end
                end
            end
            for e in 1:N, v in 1:nn
                T = sort([tcorner[m, e] for m in dof_slots[v]])
                cands = get(fac_by_corner, T[1], Vector{Int}[])
                any(f -> _issubset_sorted(T, f), cands) && push!(pairs, (v, e))
            end
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
    etype, family, d, order, numnodes, numprimary, conn = _volume_block()
    if family === :tri
        K, conn_mgb = _build_tri(T, order, conn, _node_coords())
        geom = order == 1 ? fem2d_P1(T; K) : fem2d_P2(T; K)
        return (; geometry = geom, regions = _regions_nodes(conn_mgb))
    else
        K, tcorner, ambient = _build_tensor(T, etype, order, d, conn)
        compact = Dict{Int,Int}()
        t_corner = Matrix{Int}(undef, size(tcorner)...)
        for j in eachindex(tcorner)
            t_corner[j] = get!(compact, tcorner[j], length(compact) + 1)
        end
        t_full = tensor_dofmap(t_corner, order, Val(d))
        geom = d == 2 ? fem2d(T; k = order, K, t = t_full, ambient = Val(ambient)) :
                        fem3d(T; k = order, K, t = t_full)
        return (; geometry = geom, regions = _regions_tensor(order, d, 2^d, tcorner))
    end
end

function MultiGridBarrier.gmsh_import(; verbose::Bool = false, T::Type = Float64)
    Bool(gmsh.isInitialized()) ||
        _gmsh_err("gmsh_import(): gmsh is not initialized; script your geometry " *
                  "(gmsh.initialize(); ...; gmsh.model.mesh.generate(dim)) or pass a file " *
                  "path: gmsh_import(\"mesh.msh\")")
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
