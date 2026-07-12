export FEM2D_P2, fem2d_P2

"""
    FEM2D_P2{T,B}

2D simplicial P2 FEM discretization descriptor. The `Bool` parameter `B`
selects the space: `B = true` is P2 enriched with a per-element cubic bubble
(7 nodes per triangle; the historical default, whose nodal quadrature weights
are strictly positive), `B = false` is plain Lagrange P2 (6 nodes per
triangle; the nodal quadrature is the edge-midpoint rule, whose **corner
weights are zero**). `FEM2D_P2{T}` (either variant) is the family; concrete
descriptors are `FEM2D_P2{T,true}` / `FEM2D_P2{T,false}`.

Stores the geometry's own full node mesh `Kfull` of shape `(V, N, 2)`,
`V = B ? 7 : 6` — equal to the enclosing `Geometry`'s mesh tensor `x`, so
after `subdivide` it is the fine mesh, not the original input — and the
corner triangulation `K` of shape `(3, N, 2)` extracted from it (local slots
1, 3, 5). Informational: the hierarchy builders work from the `Geometry`'s
`x` and connectivity `t`. The per-triangle node layout is
`corner1, edge(1,2), corner2, edge(2,3), corner3, edge(3,1)[, centroid]`.

For the pure-P2 variant the zero corner weights mean a variational problem's
slack must not live in the fully broken `:full` space: a corner slack value
would carry no cost and the central path escapes along it. `amg` and
`geometric_mg` therefore provide the `:broken_P1` subspace (per-element
*linear* functions, 3 DOFs per triangle, parametrized by the edge-midpoint
values), which `assemble` uses as the default `:s` space on pure-P2
geometries: every `:broken_P1` direction moves the positively weighted
midpoint values, so the slack is determined.
"""
struct FEM2D_P2{T,B}
    K::Array{T,3}
    Kfull::Array{T,3}
end

# Convenience: build from just the corner triangulation K (3 × N × 2); derive
# the full node mesh canonically (straight edges; bubble at the barycenter).
function FEM2D_P2{T,B}(K::Array{T,3}) where {T,B}
    size(K, 1) == 3 || throw(ArgumentError("K must have 3 vertices per triangle"))
    size(K, 3) == 2 || throw(ArgumentError("K must have spatial dim 2"))
    R = reference_triangle(T, Val(B))
    nn = size(K, 2)
    V = B ? 7 : 6
    Kf = Matrix{T}(kron(sparse(one(T) * I, nn, nn), R.K) * _xflat(K))   # (V*N, 2)
    FEM2D_P2{T,B}(K, reshape(Kf, V, nn, 2))
end

# Back-compat entry points: `FEM2D_P2{T}(K)` with corner input builds the
# bubble variant (the pre-`B` behavior); a full 6- or 7-node mesh infers the
# variant from its row count.
function FEM2D_P2{T}(K::Array{T,3}; bubble::Bool = true) where {T}
    if size(K, 1) == 3
        return FEM2D_P2{T,bubble}(K)
    elseif size(K, 1) == 6 || size(K, 1) == 7
        b = size(K, 1) == 7
        bubble == b || throw(ArgumentError(
            "bubble=$bubble contradicts the $(size(K, 1))-node mesh K"))
        return FEM2D_P2{T,b}(_extract_corner_mesh(K), K)
    end
    throw(ArgumentError("K must have 3 (corners), 6 (P2) or 7 (P2+bubble) rows"))
end
FEM2D_P2{T}(K::Array{T,3}, Kfull::Array{T,3}) where {T} =
    FEM2D_P2{T,size(Kfull, 1) == 7}(K, Kfull)

amg_dim(::FEM2D_P2) = 2

# The default `:s` search space for `assemble`: the fully broken space is fine
# whenever every nodal weight is positive, but pure P2 must use `:broken_P1`
# (zero corner weights leave `:full` corner slacks without a finite minimizer).
_default_slack_space(::FEM2D_P2{T,false}) where {T} = :broken_P1

reference_triangle(::Type{T}) where {T} = reference_triangle(T, Val(true))

function reference_triangle(::Type{T}, ::Val{true}) where {T}
    K = sparse(T[6 0 0
      3 3 0
      0 6 0
      0 3 3
      0 0 6
      3 0 3
      2 2 2]./6)
    w = T[3,8,3,8,3,8,27]./60
    dx =  sparse(T[  36    0   0    0   12  -48    0
   3   60  -9   12    3   12  -81
 -12   48   0  -48   12    0    0
  -3  -12   9  -60   -3  -12   81
 -12    0   0    0  -36   48    0
  12    0   0    0  -12    0    0
   4   16   0  -16   -4    0    0]./12)
    dy = sparse(T[  0   48  -12    0   12  -48    0
 -9   60    3   12    3   12  -81
  0    0   36  -48   12    0    0
  0    0   12    0  -12    0    0
  0    0  -12   48  -36    0    0
  9  -12   -3  -12   -3  -60   81
  0   16    4    0   -4  -16    0]./12)
    refine = sparse([2, 3, 4, 6, 7, 9, 13, 14, 18, 20, 21, 23, 25, 27, 4, 5, 6, 7, 8, 9, 13, 14, 20, 21, 22, 23, 25, 27, 4, 6, 7, 9, 10, 11, 13, 14, 16, 20, 21, 23, 25, 27, 6, 7, 11, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 27, 2, 6, 7, 11, 13, 14, 16, 17, 18, 20, 21, 23, 25, 27, 1, 2, 6, 7, 13, 14, 18, 19, 20, 21, 23, 25, 26, 27, 6, 7, 13, 14, 20, 21, 23, 25, 27, 28], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], T[243, 648, 243, 61, 180, -81, -20, -36, -81, -20, -36, -20, -20, 61, 486, 648, 80, 144, 648, 486, 80, 144, -82, -72, 648, 80, -82, 80, -81, -20, -36, 243, 648, 243, 61, 180, -81, -20, -36, 61, -20, -20, -82, -72, 486, 648, 80, 144, 648, 486, 80, 144, 80, 648, 80, -82, -81, -20, -36, -81, -20, -36, 243, 648, 243, 61, 180, -20, 61, -20, 648, 486, 80, 144, -82, -72, 486, 648, 80, 144, -82, 80, 648, 80, 549, 324, 549, 324, 549, 324, 549, 549, 549, 648]./648, 28, 7)
    return (K=K,w=w,dx=dx,dy=dy,refine=refine)
end

# Plain Lagrange P2 on the same reference frame as the bubble tables. Derived
# exactly (rational arithmetic) by the recipe that reproduces the P2+bubble
# K/w/dx/dy verbatim: nodal basis on the 6 nodes, weights `w_j = 2∫φ_j` (the
# edge-midpoint rule — corner weights exactly zero), `refine` = interpolation
# at the child nodes. Interpolation is canonical here: P2 restricted to a
# sub-triangle is P2, so the refined function *equals* the parent (the bubble
# table instead had to choose how to distribute the parent bubble, which the
# child spaces cannot represent; that freedom does not exist at pure P2).
function reference_triangle(::Type{T}, ::Val{false}) where {T}
    K = sparse(T[2 0 0
      1 1 0
      0 2 0
      0 1 1
      0 0 2
      1 0 1]./2)
    w = T[0,1,0,1,0,1]./3
    dx = sparse(T[ 3  0  0   0   1  -4
   1   2  0  -2   1  -2
  -1   4  0  -4   1   0
  -1   2  0  -2  -1   2
  -1   0  0   0  -3   4
   1   0  0   0  -1   0])
    dy = sparse(T[ 0  4  -1  0   1  -4
   0   2   1  -2   1  -2
   0   0   3  -4   1   0
   0   0   1   0  -1   0
   0   0  -1   4  -3   0
   0   2  -1   2  -1  -2])
    refine = sparse(T[ 0  0  0  0  0  8
   3   0   0   0  -1   6
   8   0   0   0   0   0
   3   6  -1   0   0   0
   0   8   0   0   0   0
   0   4  -1   2  -1   4
   0   8   0   0   0   0
  -1   6   3   0   0   0
   0   0   8   0   0   0
   0   0   3   6  -1   0
   0   0   0   8   0   0
  -1   4   0   4  -1   2
   0   0   0   8   0   0
   0   0  -1   6   3   0
   0   0   0   0   8   0
  -1   0   0   0   3   6
   0   0   0   0   0   8
  -1   2  -1   4   0   4
   0   8   0   0   0   0
  -1   4   0   4  -1   2
   0   0   0   8   0   0
  -1   2  -1   4   0   4
   0   0   0   0   0   8
   0   4  -1   2  -1   4]./8)
    return (K=K,w=w,dx=dx,dy=dy,refine=refine)
end

# Zero-trace continuous-P2(+bubble) embedding for the geometric_mg path. The
# supplied connectivity is authoritative; coordinates may coincide without
# representing the same topological node.
function continuous(t::AbstractMatrix{<:Integer}, ::Type{T}) where {T}
    labels = vec(t)
    bdry   = _p2_boundary_dedup_set(labels, size(t, 2))
    return _p2_continuous_subspace(labels, maximum(labels), bdry, T)
end

# Refine full P2(+bubble) connectivity in the same four-child order as
# `reference_triangle().refine`. Existing P2 edge nodes become child corners;
# new child-edge nodes are shared by topological edge key, while every cubic
# bubble (7-node layout only) remains element-local.
function _refine_p2_connectivity(t::AbstractMatrix{<:Integer})
    V = size(t, 1)
    N = size(t, 2)
    out = Matrix{Int}(undef, V, 4N)
    node_ids = Dict{Int,Int}()
    @inbounds for e in 1:N, v in 1:6
        id = Int(t[v, e])
        get!(node_ids, id, length(node_ids) + 1)
    end
    edge_nodes = Dict{Tuple{Int,Int},Int}()
    next_id = length(node_ids)

    @inbounds for e in 1:N
        a, ab, b, bc, c, ca = (node_ids[Int(t[v, e])] for v in 1:6)
        child_corners = ((ca, a, ab), (ab, b, bc),
                         (bc, c, ca), (ab, bc, ca))
        for (s, corners) in enumerate(child_corners)
            j = 4(e - 1) + s
            out[1, j], out[3, j], out[5, j] = corners
            for (slot, u, v) in ((2, corners[1], corners[2]),
                                 (4, corners[2], corners[3]),
                                 (6, corners[3], corners[1]))
                key = u < v ? (u, v) : (v, u)
                edge_id = get(edge_nodes, key, 0)
                if edge_id == 0
                    next_id += 1
                    edge_id = next_id
                    edge_nodes[key] = edge_id
                end
                out[slot, j] = edge_id
            end
            if V == 7
                next_id += 1
                out[7, j] = next_id
            end
        end
    end
    return out
end

# Default mesh: canonical expansion of the unit-square 2-triangle corner mesh.
function _default_Kfull(::Type{T}, ::Val{B}) where {T,B}
    R = reference_triangle(T, Val(B))
    K_corners_flat = T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1]
    nn = size(K_corners_flat, 1) ÷ 3
    V = B ? 7 : 6
    Kf = Matrix{T}(kron(sparse(one(T) * I, nn, nn), R.K) * K_corners_flat)
    return reshape(Kf, V, nn, 2)
end

# Extract the corner mesh (3, N, 2) tensor from a full 6/7-node mesh (slots 1, 3, 5).
function _extract_corner_mesh(Kfull::Array{T,3}) where {T}
    N = size(Kfull, 2)
    K_corners = Array{T,3}(undef, 3, N, 2)
    @inbounds for k in 1:N
        K_corners[1, k, :] .= Kfull[1, k, :]
        K_corners[2, k, :] .= Kfull[3, k, :]
        K_corners[3, k, :] .= Kfull[5, k, :]
    end
    return K_corners
end

"""
    fem2d_P2(::Type{T}=Float64; bubble=true, K=<default unit square>, t=<from K>) -> Geometry

Construct a **single-level** 2D simplicial P2 FEM `Geometry`. With
`bubble = true` (the default) the element is P2 enriched with a per-element
cubic bubble — 7 nodes per triangle, laid out as
`corner1, edge(1,2), corner2, edge(2,3), corner3, edge(3,1), centroid` —
whose nodal quadrature weights are strictly positive. With `bubble = false`
the element is plain Lagrange P2 (the same layout without the centroid) and
the nodal quadrature is the edge-midpoint rule, whose corner weights are
**zero**; the slack of a variational problem then lives in the `:broken_P1`
subspace (see [`FEM2D_P2`](@ref)), which `assemble` selects automatically.
Use `amg(geom)` to attach an algebraic-multigrid hierarchy. (The legacy
`geometric_mg(geom, L)` builds geometric-subdivision transfers instead.)

# Arguments
- `bubble::Bool`: selects the variant. When `K` is supplied its row count
  (7 or 6) determines the variant directly; passing a contradicting explicit
  `bubble` is an error.
- `K::Array{T,3}` (`V × N × 2`, `V = bubble ? 7 : 6`): per-triangle node mesh.
- `t::AbstractMatrix{<:Integer}` (`V × N`): optional full-node connectivity.
  By default it is recovered from `K` by coordinate deduplication. Pass it
  explicitly when coincident nodes must remain topologically distinct.

The element geometry is **isoparametric**: the map from the reference triangle
is built from all `V` node positions via the shape functions, so displacing
the edge (and centroid) nodes off the straight midpoints (barycenter)
genuinely curves the element (with a node-varying Jacobian). Triangles must be
**orientation-preserving and non-self-intersecting** — construction errors if
any element's `det J ≤ 0` at a node.
"""
function fem2d_P2(::Type{T}=Float64;
                  bubble::Union{Bool,Nothing} = nothing,
                  K::Union{Array{T,3},Nothing} = nothing,
                  t::Union{AbstractMatrix{<:Integer},Nothing} = nothing) where {T}
    b = bubble === nothing ? (K === nothing || size(K, 1) == 7) : bubble
    Kf = K === nothing ? _default_Kfull(T, Val(b)) : K
    V = b ? 7 : 6
    size(Kf, 1) == V || throw(ArgumentError(
        "K must have $V vertices per triangle for bubble=$b (size(K,1) = $(size(Kf, 1)))"))
    size(Kf, 3) == 2 ||
        throw(ArgumentError("K must have spatial dim 2 (size(K,3) = 2)"))
    tt = t === nothing ?
        reshape(_dedupe(_xflat(Kf))[2], size(Kf, 1), size(Kf, 2)) : t
    mg = _fem2d_P2_geometric_mg(T, Kf, tt, 1)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM2D_P2}) — AMG on continuous corners.
# ============================================================================

"""
    find_boundary(geom::Geometry{...,<:FEM2D_P2{T}}) -> Vector{Tuple{Int,Int}}

`(v, t)` index pairs into `geom.x` for every P2(+bubble) DOF (corner vertex or
edge midpoint) on `∂Ω`. Centroids (bubble variant) never appear (they are
interior by construction). Duplicates are present (a corner shared by `k`
triangles contributes its `k` pairs; a boundary-edge midpoint contributes its
single pair).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,<:FEM2D_P2{T}}) where {T}
    V, N   = size(geom.x, 1), size(geom.x, 2)
    full_labels = vec(geom.t)            # authoritative cached connectivity
    bdry_set = _p2_boundary_dedup_set(full_labels, N)
    pairs = Tuple{Int,Int}[]
    for t in 1:N, v in 1:V
        full_labels[V*(t-1) + v] in bdry_set && push!(pairs, (v, t))
    end
    return pairs
end

# Set of full-fine-deduplicated node IDs on the boundary of a P2(+bubble) mesh.
# Half-edge analysis: each P2 edge appears as two half-edges (corner, midpt) /
# (midpt, corner); an interior edge is shared by two triangles (each half-edge
# count = 2), a boundary edge by one (count = 1). Boundary DOFs are the dedup
# IDs that appear in any count-1 half-edge. The perimeter walk covers local
# slots 1-6 for both layouts (the bubble slot 7 is never on an edge).
function _p2_boundary_dedup_set(labels::AbstractVector{Int}, N::Int)
    V = length(labels) ÷ N
    t = reshape(labels, (V, N))
    halfedges = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1))
    counts = Dict{Tuple{Int,Int},Int}()
    @inbounds for e in 1:N, (a, b) in halfedges
        i, j = t[a, e], t[b, e]
        key = i < j ? (i, j) : (j, i)
        counts[key] = get(counts, key, 0) + 1
    end
    bdry = Set{Int}()
    for (k, c) in counts
        if c == 1
            push!(bdry, k[1])
            push!(bdry, k[2])
        end
    end
    return bdry
end

# Build the n × n_interior continuous-P2(+bubble) subspace matrix from an
# explicit Dirichlet node-id set (in the same indexing as `labels`).
function _p2_continuous_subspace(labels::AbstractVector{Int}, n_unique::Int,
                                 dirichlet_dedup_set, ::Type{T}) where {T}
    interior = setdiff(1:n_unique, dirichlet_dedup_set)
    pos = zeros(Int, n_unique)
    @inbounds for (j, c) in enumerate(interior)
        pos[c] = j
    end
    rows = Int[]; cols = Int[]; vals = T[]
    @inbounds for i in eachindex(labels)
        p = pos[labels[i]]
        if p != 0
            push!(rows, i); push!(cols, p); push!(vals, one(T))
        end
    end
    return sparse(rows, cols, vals, length(labels), length(interior))
end

# Embedding of the `:broken_P1` subspace (per-element linear functions) into
# the broken P2(+bubble) nodal basis. The 3 coefficients per element are the
# edge-midpoint values (μ12, μ23, μ31): midpoint slots are the identity,
# corner slots the P1 extrapolations (corner1 = μ12 − μ23 + μ31 and cyclic),
# and the bubble slot (V = 7) the barycenter value (μ12 + μ23 + μ31)/3.
# Every nonzero coefficient direction moves some midpoint value, which is why
# this space keeps pure-P2 slacks determined despite the zero corner weights.
function _broken_p1_embedding(::Type{T}, N::Int, V::Int) where {T}
    o = one(T); z = zero(T)
    slotcoef = ((o, -o, o),    # corner 1
                (o, z, z),     # edge(1,2)
                (o, o, -o),    # corner 2
                (z, o, z),     # edge(2,3)
                (-o, o, o),    # corner 3
                (z, z, o))     # edge(3,1)
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, (V == 7 ? 15 : 12) * N)
    third = o / 3
    @inbounds for e in 1:N
        br = V * (e - 1); bc = 3 * (e - 1)
        for s in 1:6, m in 1:3
            c = slotcoef[s][m]
            c == z && continue
            push!(rows, br + s); push!(cols, bc + m); push!(vals, c)
        end
        if V == 7
            for m in 1:3
                push!(rows, br + 7); push!(cols, bc + m); push!(vals, third)
            end
        end
    end
    return sparse(rows, cols, vals, V * N, 3 * N)
end

# Build an AMG hierarchy on the continuous-P1 stiffness restricted to
# `interior_set` (a subset of corner indices), with the level-K_amg bridge
# mapping back into the doubled P2(+bubble) broken basis. Used twice from
# `amg(geom)` — once with the user's Dirichlet-aware interior corners (for
# `:dirichlet`), once with `1:n_v` (for `:full`, giving the all-corners
# Neumann hierarchy).
function _fem2d_P2_hierarchy(tri_conn::Matrix{Int},
                              K_full::SparseMatrixCSC{T,Int},
                              interior_set::AbstractVector{<:Integer},
                              n_v::Int, n_doubled::Int,
                              prolongator, V::Int) where {T}
    interior_vec = collect(interior_set)
    K_loc  = K_full[interior_vec, interior_vec]
    P_amg  = _amg_prolongations(K_loc, T, prolongator)
    bridge = _interior_corners_to_doubled_p2(tri_conn, n_v, interior_vec, T, V)
    return _assemble_amg_ladder(P_amg, bridge, n_doubled)
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P2{T,B}};
             prolongator = amg_ruge_stuben(max_coarse=2),
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom))) where {T,B}
    x_fine    = _xflat(geom.x)
    V         = size(geom.x, 1)
    N         = size(geom.x, 2)
    n_doubled = V * N

    full_labels    = vec(geom.t)         # authoritative cached connectivity
    n_full_unique  = maximum(full_labels)

    # Corner connectivity + coordinates from `t` (no separate corner-coordinate dedup).
    corners, tri_conn = _extract_corners_and_connectivity(geom.t, x_fine)
    n_v = size(corners, 1)

    full_to_corner = Dict{Int,Int}()
    @inbounds for tri in 1:N, (j, local_pos) in enumerate((1, 3, 5))
        broken_row = (tri - 1) * V + local_pos
        full_to_corner[full_labels[broken_row]] = tri_conn[tri, j]
    end

    K_full = _assemble_p1_stiffness_full(corners, tri_conn)

    # :full hierarchy (all-corners Neumann); :uniform and :broken_P1 ride it.
    refine_full, sizes_full, L_full, K_amg_full =
        _fem2d_P2_hierarchy(tri_conn, K_full,
                             collect(1:n_v), n_v, n_doubled, prolongator, V)

    # One zero-trace continuous subspace per named dirichlet node set.
    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dirichlet_dedup_set  = Set{Int}(full_labels[r] for r in _pairs_to_linear(nodes, V))
        dirichlet_corner_set = Set{Int}(
            full_to_corner[fid] for fid in dirichlet_dedup_set if haskey(full_to_corner, fid))
        interior_corners = sort!(collect(setdiff(1:n_v, dirichlet_corner_set)))
        refine_dir, sizes_dir, L_dir, K_amg_dir =
            _fem2d_P2_hierarchy(tri_conn, K_full,
                                 interior_corners, n_v, n_doubled, prolongator, V)
        # Force the corner-only coarse search space to vanish at *every* Dirichlet
        # node (not just the corner DOFs the auxiliary problem represents): mask the
        # bridge so its P1 lift cannot leak nonzero values onto Dirichlet edge/
        # centroid nodes hosted on an edge with a free corner.
        refine_dir[K_amg_dir] =
            _mask_dirichlet_rows(refine_dir[K_amg_dir], full_labels, dirichlet_dedup_set)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for kk in 1:K_amg_dir
            sub[kk] = sparse(one(T) * I, sizes_dir[kk], sizes_dir[kk])
        end
        sub[L_dir] = _p2_continuous_subspace(full_labels, n_full_unique, dirichlet_dedup_set, T)
        return refine_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, sizes_full, L_full, K_amg_full, build_dirichlet;
        full_riders = Dict(:broken_P1 => _broken_p1_embedding(T, N, V)))
end

# ============================================================================
# geometric_mg(::Geometry{FEM2D_P2}, L) — reference-triangle subdivision.
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,<:FEM2D_P2{T}}, L::Int) where {T}
    _fem2d_P2_geometric_mg(T, geom.x, geom.t, L)
end

# Internal: geometric L-level multigrid for FEM2D_P2 (structured BlockDiag operators).
_fem2d_P2_geometric_mg(::Type{T}, Kfull::Array{T,3}, t::AbstractMatrix{<:Integer},
                       L::Int) where {T} = _fem2d_P2_structured(T, Kfull, t, L)

function _fem2d_P2_structured(::Type{T}, Kfull::Array{T,3}, t::AbstractMatrix{<:Integer},
                              L::Int) where {T}
    p = size(Kfull, 1)
    p == 6 || p == 7 ||
        throw(ArgumentError("K must have 6 (P2) or 7 (P2+bubble) vertices per triangle"))
    B = p == 7
    R = reference_triangle(T, Val(B))

    size(Kfull, 3) == 2 || throw(ArgumentError("K must have spatial dim 2"))
    size(t) == (p, size(Kfull, 2)) ||
        throw(ArgumentError("t must have size ($p, size(K,2))"))
    all(>(0), t) || throw(ArgumentError("t must contain positive node ids"))
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))

    nn = size(Kfull, 2)
    x = Array{Matrix{T}, 1}(undef, L)   # flat (p*N_l, 2) coordinates per level
    x[1] = _xflat(Kfull)
    topology = Vector{Matrix{Int}}(undef, L)
    topology[1] = Matrix{Int}(t)

    ref_dense = Matrix(R.refine)
    K_refine = 4

    N_blocks = nn * 4^(L-1)

    id_data = zeros(T, p, p, N_blocks)
    for i in 1:N_blocks
        for j in 1:p
            id_data[j, j, i] = one(T)
        end
    end
    id_vbd = _vblock_sparse(p, p, 1, N_blocks, id_data)

    refine = Vector{typeof(id_vbd)}(undef, L)

    for l in 1:L-1
        n_l = nn * 4^(l-1)
        ref_data = zeros(T, p, p, K_refine * n_l)
        for i in 1:n_l
            for s in 1:K_refine
                ref_data[:, :, (i-1)*K_refine + s] = ref_dense[(s-1)*p+1:s*p, :]
            end
        end
        refine[l] = _vblock_sparse(p, p, K_refine, n_l, ref_data)
        x[l+1] = refine[l] * x[l]
        topology[l+1] = _refine_p2_connectivity(topology[l])
    end

    refine[L] = id_vbd

    n = size(x[L], 1)
    N = Int(n / p)
    xL = reshape(x[L]', (2, p, N))

    R_dx = Matrix(R.dx)
    R_dy = Matrix(R.dy)

    id_block = zeros(T, p, p, N)
    dx_block = zeros(T, p, p, N)
    dy_block = zeros(T, p, p, N)
    w_vec = zeros(T, n)
    detJ_vec = zeros(T, n)

    # Isoparametric P2(+bubble) map: x(ξ,η) = Σ_i N_i(ξ,η) X_i over all p geometry
    # nodes, so the Jacobian J = [∂x/∂ξ ∂x/∂η; ∂y/∂ξ ∂y/∂η] varies node-to-node
    # whenever the element is curved (edge nodes off the midpoints, centroid off
    # the barycenter). The reference derivative rows give ∂N_i/∂ξ and ∂N_i/∂η at
    # each node, so (R_dx*X)[j] = ∂x/∂ξ at node j, etc. The physical-derivative
    # operator rows are J(node j)^{-T} applied to (R_dx, R_dy), and the quadrature
    # weight at node j is det J(node j) · R.w[j]. For a straight triangle J is
    # constant and this reduces to the previous affine map.
    for k in 1:N
        X = @view xL[1, :, k]
        Y = @view xL[2, :, k]
        x_xi = R_dx * X; x_eta = R_dy * X
        y_xi = R_dx * Y; y_eta = R_dy * Y
        for j in 1:p
            detJ = x_xi[j] * y_eta[j] - x_eta[j] * y_xi[j]
            invdet = one(T) / detJ
            for m in 1:p
                dx_block[j, m, k] = ( y_eta[j] * R_dx[j, m] - y_xi[j] * R_dy[j, m]) * invdet
                dy_block[j, m, k] = (-x_eta[j] * R_dx[j, m] + x_xi[j] * R_dy[j, m]) * invdet
            end
            id_block[j, j, k] = one(T)
            detJ_vec[(k-1)*p + j] = detJ
            w_vec[(k-1)*p + j] = detJ * R.w[j]
        end
    end

    # The barrier method requires nonnegative quadrature weights, which here is
    # det J > 0 at every node (geometric validity of the element map). For the
    # bubble space that is equivalent to strictly positive weights; for pure P2
    # the corner weights are exactly zero by construction, which is fine — only
    # a non-positive Jacobian is refused.
    if !all(>(zero(T)), detJ_vec)
        bad      = findall(<=(zero(T)), detJ_vec)
        badelems = sort!(unique((bad .- 1) .÷ p .+ 1))
        error("fem2d_P2: non-positive Jacobian at $(length(bad)) node(s) " *
              "across $(length(badelems)) element(s) (first few: $(first(badelems, 5))). " *
              "The isoparametric element map has det J ≤ 0 there — the curved triangle " *
              "is folded or has inverted (clockwise) vertex orientation. Supply " *
              "orientation-preserving, non-self-intersecting elements.")
    end

    id = BlockDiag(id_block)
    dx = BlockDiag(dx_block)
    dy = BlockDiag(dy_block)

    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    broken_P1 = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        dirichlet[l] = continuous(topology[l], T)
        full[l] = spdiagm(0 => ones(T, size(x[l], 1)))
        uniform[l] = sparse(ones(T, (size(x[l], 1), 1)))
        broken_P1[l] = _broken_p1_embedding(T, size(x[l], 1) ÷ p, p)
    end

    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => dirichlet, :full => full, :uniform => uniform,
        :broken_P1 => broken_P1)
    operators = Dict{Symbol, BlockDiag{T,Array{T,3}}}(:id => id, :dx => dx, :dy => dy)
    x_fine = reshape(x[end], p, N, 2)
    disc = FEM2D_P2{T,B}(_extract_corner_mesh(x_fine), x_fine)
    geom = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, FEM2D_P2{T,B}}(
        disc, topology[end], x_fine, w_vec, operators)
    return MultiGrid(geom, subspaces, refine)
end

# plot(::Geometry{...FEM2D_P2}, z) lives in MultiGridBarrierPyPlotExt.

# ============================================================================
# Helpers used by amg(::FEM2D_P2)
# ============================================================================

# Compact corner coordinates + triangle connectivity from the cached full-node
# connectivity `t` (shape (V,N)) and broken coords `x_fine` (V·N × 2). Corner local
# slots are (1,3,5). Numbering comes from `t` (first occurrence), not a corner-
# coordinate dedup — so it differs from the legacy ordering (solve-equivalent).
function _extract_corners_and_connectivity(t::AbstractMatrix{Int}, x_fine::AbstractMatrix{T}) where {T}
    corner_local = (1, 3, 5)
    V = size(t, 1)
    N = size(t, 2)
    labels, n_v = _corner_labels_from_t(t, corner_local)
    tri_conn = collect(transpose(reshape(labels, 3, N)))
    corners = Matrix{T}(undef, n_v, size(x_fine, 2))
    seen = falses(n_v)
    @inbounds for e in 1:N, ci in 1:3
        cc = labels[(e-1)*3 + ci]
        if !seen[cc]
            corners[cc, :] = @view x_fine[V*(e-1) + corner_local[ci], :]
            seen[cc] = true
        end
    end
    return corners, tri_conn
end

function _find_boundary_corners(tri_conn::Matrix{Int})
    N = size(tri_conn, 1)
    edge_count = Dict{Tuple{Int,Int}, Int}()
    @inbounds for k in 1:N
        a, b, c = tri_conn[k, 1], tri_conn[k, 2], tri_conn[k, 3]
        for (i, j) in ((a, b), (b, c), (c, a))
            key = i < j ? (i, j) : (j, i)
            edge_count[key] = get(edge_count, key, 0) + 1
        end
    end
    bset = Set{Int}()
    for (e, c) in edge_count
        if c == 1
            push!(bset, e[1])
            push!(bset, e[2])
        end
    end
    return sort!(collect(bset))
end

function _assemble_p1_stiffness_full(corners::Matrix{T}, tri_conn::Matrix{Int}) where {T}
    n_v = size(corners, 1)
    N   = size(tri_conn, 1)
    rows = Vector{Int}(undef, 9*N)
    cols = Vector{Int}(undef, 9*N)
    vals = Vector{T}(undef, 9*N)
    @inbounds for k in 1:N
        i1, i2, i3 = tri_conn[k, 1], tri_conn[k, 2], tri_conn[k, 3]
        x1, y1 = corners[i1, 1], corners[i1, 2]
        x2, y2 = corners[i2, 1], corners[i2, 2]
        x3, y3 = corners[i3, 1], corners[i3, 2]
        det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
        s = T(1) / (2 * abs(det2))
        bs = (b1, b2, b3); cs = (c1, c2, c3); idx = (i1, i2, i3)
        for i in 1:3, j in 1:3
            rows[9*(k-1) + 3*(i-1) + j] = idx[i]
            cols[9*(k-1) + 3*(i-1) + j] = idx[j]
            vals[9*(k-1) + 3*(i-1) + j] = (bs[i]*bs[j] + cs[i]*cs[j]) * s
        end
    end
    return sparse(rows, cols, vals, n_v, n_v)
end

# Direct bridge from interior-P1 corners to the doubled P2(+bubble) basis.
# Per triangle: 3 vertex DOFs at 1, 6 edge midpoints at 1/2, and (bubble
# layout only) 1 centroid at 1/3, with boundary-corner pushes dropped instead
# of zero-padded through a full-corners intermediate.
function _interior_corners_to_doubled_p2(tri_conn::Matrix{Int}, n_v::Int,
                                         interior_corners::Vector{Int},
                                         ::Type{T}, V::Int) where {T}
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    N = size(tri_conn, 1)
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, 12*N); sizehint!(cols, 12*N); sizehint!(vals, 12*N)
    half  = T(1) / 2
    third = T(1) / 3
    @inbounds for k in 1:N
        a, b, c = tri_conn[k, 1], tri_conn[k, 2], tri_conn[k, 3]
        ai, bi, ci = interior_idx[a], interior_idx[b], interior_idx[c]
        base = V*(k - 1)
        if ai != 0; push!(rows, base+1); push!(cols, ai); push!(vals, T(1)); end
        if bi != 0; push!(rows, base+3); push!(cols, bi); push!(vals, T(1)); end
        if ci != 0; push!(rows, base+5); push!(cols, ci); push!(vals, T(1)); end
        if ai != 0; push!(rows, base+2); push!(cols, ai); push!(vals, half); end
        if bi != 0; push!(rows, base+2); push!(cols, bi); push!(vals, half); end
        if bi != 0; push!(rows, base+4); push!(cols, bi); push!(vals, half); end
        if ci != 0; push!(rows, base+4); push!(cols, ci); push!(vals, half); end
        if ci != 0; push!(rows, base+6); push!(cols, ci); push!(vals, half); end
        if ai != 0; push!(rows, base+6); push!(cols, ai); push!(vals, half); end
        if V == 7
            if ai != 0; push!(rows, base+7); push!(cols, ai); push!(vals, third); end
            if bi != 0; push!(rows, base+7); push!(cols, bi); push!(vals, third); end
            if ci != 0; push!(rows, base+7); push!(cols, ci); push!(vals, third); end
        end
    end
    return sparse(rows, cols, vals, V*N, n_int)
end
