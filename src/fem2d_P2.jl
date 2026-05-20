export FEM2D_P2, fem2d_P2
using Random

"""
    FEM2D_P2{T}

2D FEM (P2+bubble) discretization descriptor. Stores the corner triangulation `K`
shape `(3, N, 2)` and the canonical P2+bubble mesh `K7` shape `(7, N, 2)`.
"""
struct FEM2D_P2{T}
    K::Array{T,3}
    K7::Array{T,3}
end

# Convenience: build FEM2D_P2 from just the corner triangulation K; derive K7 canonically.
function FEM2D_P2{T}(K::Array{T,3}) where {T}
    size(K, 1) == 3 || throw(ArgumentError("K must have 3 vertices per triangle"))
    size(K, 3) == 2 || throw(ArgumentError("K must have spatial dim 2"))
    R = reference_triangle(T)
    nn = size(K, 2)
    K7f = Matrix{T}(blockdiag([R.K for _ in 1:nn]...) * _xflat(K))   # (7*N, 2)
    K7  = reshape(K7f, 7, nn, 2)
    FEM2D_P2{T}(K, K7)
end

amg_dim(::FEM2D_P2{T}) where {T} = 2

function reference_triangle(::Type{T}) where {T}
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
    coarsen = sparse([6, 1, 2, 2, 3, 4, 4, 5, 6, 2, 4, 6, 7], [1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 28], T[1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3]./3, 7, 28)
    refine = sparse([2, 3, 4, 6, 7, 9, 13, 14, 18, 20, 21, 23, 25, 27, 4, 5, 6, 7, 8, 9, 13, 14, 20, 21, 22, 23, 25, 27, 4, 6, 7, 9, 10, 11, 13, 14, 16, 20, 21, 23, 25, 27, 6, 7, 11, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 27, 2, 6, 7, 11, 13, 14, 16, 17, 18, 20, 21, 23, 25, 27, 1, 2, 6, 7, 13, 14, 18, 19, 20, 21, 23, 25, 26, 27, 6, 7, 13, 14, 20, 21, 23, 25, 27, 28], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], T[243, 648, 243, 61, 180, -81, -20, -36, -81, -20, -36, -20, -20, 61, 486, 648, 80, 144, 648, 486, 80, 144, -82, -72, 648, 80, -82, 80, -81, -20, -36, 243, 648, 243, 61, 180, -81, -20, -36, 61, -20, -20, -82, -72, 486, 648, 80, 144, 648, 486, 80, 144, 80, 648, 80, -82, -81, -20, -36, -81, -20, -36, 243, 648, 243, 61, 180, -20, 61, -20, 648, 486, 80, 144, -82, -72, 486, 648, 80, 144, -82, 80, 648, 80, 549, 324, 549, 324, 549, 324, 549, 549, 549, 648]./648, 28, 7)
    return (K=K,w=w,dx=dx,dy=dy,coarsen=coarsen,refine=refine)
end

function continuous(x::Matrix{T};
                    tol=maximum(abs.(x))*10*eps(T)) where {T}
    n = size(x)[1]
    a = 1
    seed = hash(x)
    rng  = Xoshiro(seed)
    u = randn(rng,T,2)
    u = u/norm(u)
    p = x*u
    P = sortperm(p)
    labels = zeros(Int,n)
    count = 0
    while a<=n
        if labels[P[a]]==0
            count += 1
            labels[P[a]] = count
            b = a+1
            while b<=n && p[P[b]]<=p[P[a]]+tol
                b+=1
            end
            for k=a+1:b-1
                if norm(x[P[a],:]-x[P[k],:])<=tol
                    labels[P[k]] = count
                    x[P[k],:] = x[P[a],:]
                end
            end
        end
        a+=1
    end
    t = reshape(labels,(7,:))
    e = hcat(t[1:2,:],t[2:3,:],t[3:4,:],t[4:5,:],t[5:6,:],t[[6,1],:])'
    e = sort(e,dims=2)
    P = sortperm(1:size(e,1),lt=(j,k)->e[j,:]<e[k,:])
    w = e[P,:]
    J = cumsum(vcat(1,(w[1:end-1,1].!=w[2:end,1]) .|| (w[1:end-1,2].!=w[2:end,2])))
    J = J[invperm(P)]
    ne = maximum(J)
    ec = zeros(Int,ne)
    for k=1:length(J)
        ec[J[k]] += 1
    end
    idx = findall(ec[J] .== 1)
    e = e[idx,:]
    boundary = unique(reshape(e,(length(e),)))
    interior = setdiff(1:count,boundary)

    C = sparse(1:n,labels,ones(T,n),n,count)
    C[:,interior]
end

# Default 7-DOF mesh: canonical expansion of the unit-square 2-triangle corner mesh.
function _default_K7(::Type{T}) where {T}
    R = reference_triangle(T)
    K_corners_flat = T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1]
    nn = size(K_corners_flat, 1) ÷ 3
    K7f = Matrix{T}(blockdiag([R.K for _ in 1:nn]...) * K_corners_flat)
    return reshape(K7f, 7, nn, 2)
end

# Extract corner mesh (3, N, 2) tensor from a 7-DOF P2+bubble mesh (positions 1, 3, 5).
function _extract_corner_mesh_from_K7(K7::Array{T,3}) where {T}
    N = size(K7, 2)
    K_corners = Array{T,3}(undef, 3, N, 2)
    @inbounds for k in 1:N
        K_corners[1, k, :] .= K7[1, k, :]
        K_corners[2, k, :] .= K7[3, k, :]
        K_corners[3, k, :] .= K7[5, k, :]
    end
    return K_corners
end

"""
    fem2d_P2(::Type{T}=Float64; K=<default 7-DOF unit square>) -> Geometry

Construct a **single-level** 2D FEM `Geometry` on the doubled P2+bubble mesh `K`
(`7 × N × 2`). Use `amg(geom)` to attach an algebraic-multigrid hierarchy. (The
legacy `geometric_mg(geom, L)` builds geometric-subdivision transfers instead.)

# Arguments
- `K::Array{T,3}` (`7 × N × 2`): P2+bubble per-triangle mesh; the 7 vertices per
  triangle are laid out as
  `corner1, midpt(1,2), corner2, midpt(2,3), corner3, midpt(3,1), centroid`.
"""
function fem2d_P2(::Type{T}=Float64;
                  K::Array{T,3} = _default_K7(T),
                  rest...) where {T}
    size(K, 1) == 7 ||
        throw(ArgumentError("K must have 7 vertices per triangle (size(K,1) = 7)"))
    size(K, 3) == 2 ||
        throw(ArgumentError("K must have spatial dim 2 (size(K,3) = 2)"))

    K_corners = _extract_corner_mesh_from_K7(K)
    mg = _fem2d_P2_geometric_mg(T, K_corners, K, 1)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM2D_P2}) — AMG on continuous corners.
# ============================================================================

"""
    find_boundary(geom::Geometry{...,FEM2D_P2{T}}) -> Vector{Tuple{Int,Int}}

`(v, t)` index pairs into `geom.x` for every P2+bubble DOF (corner vertex or
edge midpoint) on `∂Ω`. Centroids never appear (they are interior by
construction). Duplicates are present (a corner shared by `k` triangles
contributes its `k` pairs; a boundary-edge midpoint contributes its single
pair).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P2{T}}) where {T}
    x_fine = _xflat(geom.x)
    N      = size(geom.x, 2)
    _, full_labels = _dedupe(x_fine)
    bdry_set = _p2_boundary_dedup_set(full_labels, N)
    pairs = Tuple{Int,Int}[]
    for t in 1:N, v in 1:7
        full_labels[7*(t-1) + v] in bdry_set && push!(pairs, (v, t))
    end
    return pairs
end

# Set of full-fine-deduplicated node IDs on the boundary of a P2+bubble mesh.
# Half-edge analysis: each P2 edge appears as two half-edges (corner, midpt) /
# (midpt, corner); an interior edge is shared by two triangles (each half-edge
# count = 2), a boundary edge by one (count = 1). Boundary DOFs are the dedup
# IDs that appear in any count-1 half-edge.
function _p2_boundary_dedup_set(labels::AbstractVector{Int}, N::Int)
    t = reshape(labels, (7, N))
    e = hcat(t[1:2,:], t[2:3,:], t[3:4,:], t[4:5,:], t[5:6,:], t[[6,1],:])'
    e = sort(e, dims=2)
    P = sortperm(1:size(e,1), lt=(j,k) -> e[j,:] < e[k,:])
    w = e[P,:]
    J = cumsum(vcat(1, (w[1:end-1,1] .!= w[2:end,1]) .|| (w[1:end-1,2] .!= w[2:end,2])))
    J = J[invperm(P)]
    ne = maximum(J)
    counts = zeros(Int, ne)
    @inbounds for k in 1:length(J)
        counts[J[k]] += 1
    end
    idx = findall(counts[J] .== 1)
    return Set{Int}(reshape(e[idx, :], :))
end

# Build the n × n_interior continuous-P2+bubble subspace matrix from an
# explicit Dirichlet-dedup set (in the same indexing as `labels = _dedupe(x)[2]`).
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

# Build an AMG hierarchy on the continuous-P1 stiffness restricted to
# `interior_set` (a subset of corner indices), with the level-K_amg bridge
# mapping back into the doubled P2+bubble broken basis. Used twice from
# `amg(geom)` — once with the user's Dirichlet-aware interior corners (for
# `:dirichlet`), once with `1:n_v` (for `:full`, giving the all-corners
# Neumann hierarchy).
function _fem2d_P2_hierarchy(corners::Matrix{T},
                              tri_conn::Matrix{Int},
                              K_full::SparseMatrixCSC{T,Int},
                              interior_set::AbstractVector{<:Integer},
                              n_v::Int, n_doubled::Int,
                              max_coarse::Int) where {T}
    interior_vec = collect(interior_set)
    K_loc  = K_full[interior_vec, interior_vec]
    n_loc  = length(interior_vec)

    P_amg       = _amg_prolongations(K_loc, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1
    L_total     = K_amg + 1

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
        coarsen[kk] = _amg_injection(P_amg[i])
    end
    refine[K_amg]  = _interior_corners_to_doubled_p2_bubble(tri_conn, n_v, interior_vec, T)
    coarsen[K_amg] = _doubled_to_interior_corners_pick(tri_conn, n_v, interior_vec, T)
    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_loc
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[L_total] = n_doubled

    return refine, coarsen, sizes, L_total, K_amg
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P2{T}};
             max_coarse::Int=2,
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom))) where {T}
    x_fine    = _xflat(geom.x)
    N         = size(geom.x, 2)
    n_doubled = 7 * N

    _, full_labels = _dedupe(x_fine)
    n_full_unique  = maximum(full_labels)

    corners, tri_conn = _extract_corners_and_connectivity(x_fine, N)
    n_v = size(corners, 1)

    full_to_corner = Dict{Int,Int}()
    @inbounds for tri in 1:N, (j, local_pos) in enumerate((1, 3, 5))
        broken_row = (tri - 1) * 7 + local_pos
        full_to_corner[full_labels[broken_row]] = tri_conn[tri, j]
    end

    K_full = _assemble_p1_stiffness_full(corners, tri_conn)

    # :full hierarchy (all-corners Neumann); :uniform rides it.
    refine_full, coarsen_full, sizes_full, L_full, K_amg_full =
        _fem2d_P2_hierarchy(corners, tri_conn, K_full,
                             collect(1:n_v), n_v, n_doubled, max_coarse)

    # One zero-trace continuous subspace per named dirichlet node set.
    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dirichlet_dedup_set  = Set{Int}(full_labels[r] for r in _pairs_to_linear(nodes, 7))
        dirichlet_corner_set = Set{Int}(
            full_to_corner[fid] for fid in dirichlet_dedup_set if haskey(full_to_corner, fid))
        interior_corners = sort!(collect(setdiff(1:n_v, dirichlet_corner_set)))
        refine_dir, coarsen_dir, sizes_dir, L_dir, K_amg_dir =
            _fem2d_P2_hierarchy(corners, tri_conn, K_full,
                                 interior_corners, n_v, n_doubled, max_coarse)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for kk in 1:K_amg_dir
            sub[kk] = sparse(one(T) * I, sizes_dir[kk], sizes_dir[kk])
        end
        sub[L_dir] = _p2_continuous_subspace(full_labels, n_full_unique, dirichlet_dedup_set, T)
        return refine_dir, coarsen_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, coarsen_full, sizes_full, L_full, K_amg_full, build_dirichlet)
end

# ============================================================================
# geometric_mg(::Geometry{FEM2D_P2}, L) — reference-triangle subdivision.
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P2{T}}, L::Int) where {T}
    K  = geom.discretization.K
    K7 = geom.discretization.K7
    _fem2d_P2_geometric_mg(T, K, K7, L)
end

# Internal: geometric L-level multigrid for FEM2D_P2 (structured BlockDiag operators).
_fem2d_P2_geometric_mg(::Type{T}, K::Array{T,3}, K7::Array{T,3}, L::Int) where {T} =
    _fem2d_P2_structured(T, K, K7, L)

function _fem2d_P2_structured(::Type{T}, K::Array{T,3}, K7::Array{T,3}, L::Int) where {T}
    R = reference_triangle(T)
    p = 7

    nn = size(K7, 2)
    x = Array{Matrix{T}, 1}(undef, L)   # flat (7*N_l, 2) coordinates per level
    x[1] = _xflat(K7)

    ref_dense = Matrix(R.refine)
    coar_dense = Matrix(R.coarsen)
    K_refine = 4

    N_blocks = nn * 4^(L-1)

    id_data = zeros(T, p, p, N_blocks)
    for i in 1:N_blocks
        for j in 1:p
            id_data[j, j, i] = one(T)
        end
    end
    id_vbd = _vblock_sparse(p, p, 1, N_blocks, id_data)
    id_hbd = _hblock_sparse(p, p, 1, N_blocks, copy(id_data))

    refine = Vector{typeof(id_vbd)}(undef, L)
    coarsen = Vector{typeof(id_hbd)}(undef, L)

    for l in 1:L-1
        n_l = nn * 4^(l-1)
        ref_data = zeros(T, p, p, K_refine * n_l)
        coar_data = zeros(T, p, p, K_refine * n_l)
        for i in 1:n_l
            for s in 1:K_refine
                ref_data[:, :, (i-1)*K_refine + s] = ref_dense[(s-1)*p+1:s*p, :]
                coar_data[:, :, (i-1)*K_refine + s] = coar_dense[:, (s-1)*p+1:s*p]
            end
        end
        refine[l] = _vblock_sparse(p, p, K_refine, n_l, ref_data)
        coarsen[l] = _hblock_sparse(p, p, K_refine, n_l, coar_data)
        x[l+1] = refine[l] * x[l]
    end

    refine[L] = id_vbd
    coarsen[L] = id_hbd

    n = size(x[L], 1)
    N = Int(n / p)
    xL = reshape(x[L]', (2, p, N))

    R_dx = Matrix(R.dx)
    R_dy = Matrix(R.dy)

    id_block = zeros(T, p, p, N)
    dx_block = zeros(T, p, p, N)
    dy_block = zeros(T, p, p, N)
    w_vec = zeros(T, n)

    for k in 1:N
        u = xL[:, 1, k] - xL[:, 5, k]
        v = xL[:, 3, k] - xL[:, 5, k]
        A = hcat(u, v)
        invA = inv(A)'
        dx_block[:, :, k] = invA[1, 1] * R_dx + invA[1, 2] * R_dy
        dy_block[:, :, k] = invA[2, 1] * R_dx + invA[2, 2] * R_dy
        for j in 1:p
            id_block[j, j, k] = one(T)
        end
        w_blk = abs(det(A)) * R.w
        w_vec[(k-1)*p+1:k*p] = w_blk
    end

    id = BlockDiag(id_block)
    dx = BlockDiag(dx_block)
    dy = BlockDiag(dy_block)

    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        dirichlet[l] = continuous(x[l])
        full[l] = spdiagm(0 => ones(T, size(x[l], 1)))
        uniform[l] = sparse(ones(T, (size(x[l], 1), 1)))
    end

    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict{Symbol, BlockDiag{T,Array{T,3}}}(:id => id, :dx => dx, :dy => dy)
    disc = FEM2D_P2{T}(K, K7)
    x_fine = reshape(x[end], 7, N, 2)
    geom = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, FEM2D_P2{T}}(
        disc, x_fine, w_vec, operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# ============================================================================
# Plotting
# ============================================================================
function plot(M::Geometry{T, Array{T,3}, Vector{T}, <:Any, FEM2D_P2{T}}, z::Vector{T}; kwargs...) where {T}
    Xf = _xflat(M.x)
    x = Xf[:,1]
    y = Xf[:,2]
    S = [1 2 7
         2 3 7
         3 4 7
         4 5 7
         5 6 7
         6 1 7]
    N = size(M.x, 2)
    S = vcat([S.+(7*k) for k=0:N-1]...)
    plot_trisurf(x,y,z,triangles=S .- 1; kwargs...)
end

# ============================================================================
# Helpers used by amg(::FEM2D_P2)
# ============================================================================

# Given fine doubled coordinates (7N × 2), return unique corners and triangle connectivity.
function _extract_corners_and_connectivity(x_fine::Matrix{T}, N::Int) where {T}
    corner_rows = Vector{Int}(undef, 3*N)
    @inbounds for k in 1:N
        corner_rows[3*(k-1) + 1] = 7*(k-1) + 1
        corner_rows[3*(k-1) + 2] = 7*(k-1) + 3
        corner_rows[3*(k-1) + 3] = 7*(k-1) + 5
    end
    x_corners = x_fine[corner_rows, :]
    unique_xy, labels = _dedupe(x_corners)
    tri_conn = collect(transpose(reshape(labels, 3, N)))
    return unique_xy, tri_conn
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

# Direct bridge from interior-P1 corners to doubled P2+bubble basis.
# Same per-triangle pattern as the corner-to-doubled lift (3 vertex DOFs, 6 edge
# midpoints @ 1/2, 1 centroid @ 1/3), with boundary-corner pushes dropped instead
# of zero-padded through a full-corners intermediate.
function _interior_corners_to_doubled_p2_bubble(tri_conn::Matrix{Int}, n_v::Int,
                                                interior_corners::Vector{Int},
                                                ::Type{T}) where {T}
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
        base = 7*(k - 1)
        if ai != 0; push!(rows, base+1); push!(cols, ai); push!(vals, T(1)); end
        if bi != 0; push!(rows, base+3); push!(cols, bi); push!(vals, T(1)); end
        if ci != 0; push!(rows, base+5); push!(cols, ci); push!(vals, T(1)); end
        if ai != 0; push!(rows, base+2); push!(cols, ai); push!(vals, half); end
        if bi != 0; push!(rows, base+2); push!(cols, bi); push!(vals, half); end
        if bi != 0; push!(rows, base+4); push!(cols, bi); push!(vals, half); end
        if ci != 0; push!(rows, base+4); push!(cols, ci); push!(vals, half); end
        if ci != 0; push!(rows, base+6); push!(cols, ci); push!(vals, half); end
        if ai != 0; push!(rows, base+6); push!(cols, ai); push!(vals, half); end
        if ai != 0; push!(rows, base+7); push!(cols, ai); push!(vals, third); end
        if bi != 0; push!(rows, base+7); push!(cols, bi); push!(vals, third); end
        if ci != 0; push!(rows, base+7); push!(cols, ci); push!(vals, third); end
    end
    return sparse(rows, cols, vals, 7*N, n_int)
end

# Injection restriction: for each interior corner, pick the first triangle that
# carries it and select that triangle's vertex DOF for the corner.
function _doubled_to_interior_corners_pick(tri_conn::Matrix{Int}, n_v::Int,
                                           interior_corners::Vector{Int},
                                           ::Type{T}) where {T}
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    N = size(tri_conn, 1)
    chosen = zeros(Int, n_int)
    @inbounds for k in 1:N
        for p in 1:3
            v = tri_conn[k, p]
            vi = interior_idx[v]
            if vi != 0 && chosen[vi] == 0
                chosen[vi] = 7*(k - 1) + (2*p - 1)
            end
        end
    end
    @assert all(chosen .> 0)
    return sparse(1:n_int, chosen, ones(T, n_int), n_int, 7*N)
end
