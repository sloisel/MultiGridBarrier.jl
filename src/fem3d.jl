export fem3d
using .Mesh3d: FEM3D, _geometric_fem3d_mg
import .Mesh3d
using Random: Xoshiro

# Default Q_k mesh: canonical expansion of the unit-cube single-hex Q1 mesh.
function _default_K_qk(::Type{T}, k::Int) where {T}
    K_q1 = T[-1.0 -1.0 -1.0;  1.0 -1.0 -1.0;
             -1.0  1.0 -1.0;  1.0  1.0 -1.0;
             -1.0 -1.0  1.0;  1.0 -1.0  1.0;
             -1.0  1.0  1.0;  1.0  1.0  1.0]
    Kf = Mesh3d.promote_to_Qk(K_q1, k)  # flat ((k+1)^3 * N, 3)
    s = k + 1
    sk3 = s^3
    N   = size(Kf, 1) ÷ sk3
    return reshape(Kf, sk3, N, 3)
end

# Extract the (8, N, 3) Q1 corner tensor from a Q_k Lagrange-Chebyshev mesh `K`
# of shape ((k+1)^3, N, 3).
function _extract_Kqk_corners(K::Array{T,3}, k::Int) where {T}
    s = k + 1
    sk3 = s^3
    @assert size(K, 1) == sk3
    N = size(K, 2)
    local_c = (1, s, s*(s-1) + 1, s^2,
               (s-1)*s^2 + 1, (s-1)*s^2 + s,
               (s-1)*s^2 + s*(s-1) + 1, s^3)
    out = Array{T,3}(undef, 8, N, 3)
    @inbounds for e in 1:N, c in 1:8
        out[c, e, :] .= K[local_c[c], e, :]
    end
    return out
end

"""
    fem3d(::Type{T}=Float64; k=3, K=<default unit-cube Q_k>) -> Geometry

Construct a **single-level** 3D Q_k FEM `Geometry` on the Lagrange-Chebyshev mesh
`K` of shape `((k+1)^3, N, 3)`. Use `amg(geom)` to attach an algebraic-multigrid
hierarchy. (The legacy `geometric_mg(geom, L)` builds geometric-subdivision
transfers instead.)

# Arguments
- `k::Int=3`: polynomial order of the Q_k basis.
- `K::Array{T,3}` (`(k+1)^3 × N × 3`): per-hex Q_k Lagrange-Chebyshev tensor;
  `K[v, e, d]` is coordinate `d` of Lagrange node `v` of hex `e` (tensor-product
  ordering: x fastest, then y, then z).

The element geometry is **isoparametric**: the map from the reference hex is built
from all `(k+1)^3` node positions via the Q_k Lagrange basis, so displacing the
edge/face/interior nodes genuinely curves the element (node-varying Jacobian).
Hexes must be **orientation-preserving and non-self-intersecting** — construction
errors if any element's `det J ≤ 0` at a quadrature node, since the barrier method
requires strictly positive weights.

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem3d(::Type{T}=Float64;
                k::Int=3,
                K::Array{T,3} = _default_K_qk(T, k),
                rest...) where {T}
    s   = k + 1
    sk3 = s^3
    size(K, 1) == sk3 ||
        throw(ArgumentError("K must have (k+1)^3 = $sk3 vertices per hex (size(K,1) = $sk3)"))
    size(K, 3) == 3 ||
        throw(ArgumentError("K must have spatial dim 3 (size(K,3) = 3)"))

    K_corners = _extract_Kqk_corners(K, k)
    mg = _geometric_fem3d_mg(T; L=1, K=_xflat(K_corners), K_qk=_xflat(K), k=k, structured=true)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM3D}) — AMG on Q1 corners with companion Galerkin stiffness.
# ============================================================================
"""
    find_boundary(geom::Geometry{...,FEM3D{T}}) -> Vector{Tuple{Int,Int}}

`(v, e)` index pairs into `geom.x` for every Q_k DOF on `∂Ω`. Boundary faces
are identified by use count (a face used by exactly one hex is on `∂Ω`);
every DOF lying on a boundary face is returned, including edge / face /
corner DOFs and (for `k ≥ 2`) the interior-of-face nodes. Duplicates are
present (one pair per (vertex, element) ownership).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,FEM3D{T}}) where {T}
    k    = geom.discretization.k
    s    = k + 1
    sk3  = s^3
    N    = size(geom.x, 2)
    bdry, _, node_map = Mesh3d.get_boundary_nodes(_xflat(geom.x), k)
    bdry_set = Set(bdry)
    pairs = Tuple{Int,Int}[]
    for e in 1:N, v in 1:sk3
        node_map[v + (e - 1)*sk3] in bdry_set && push!(pairs, (v, e))
    end
    return pairs
end

# Build an AMG hierarchy on the Q1-corner Galerkin stiffness restricted to
# `interior_set` (a subset of corner indices), with the level-K_amg bridge
# mapping back into the doubled Q_k broken basis. Used twice from `amg(geom)`
# — once with the user's Dirichlet-aware interior corners (for `:dirichlet`),
# once with `1:n_v` (for `:full`, giving the all-corners Neumann hierarchy).
function _fem3d_hierarchy(node_map_q1::Vector{Int}, k::Int,
                           A_doubled::SparseMatrixCSC{Float64,Int},
                           interior_set::AbstractVector{<:Integer},
                           n_v::Int, n_doubled::Int,
                           prolongator, ::Type{T}) where {T}
    interior_vec = collect(interior_set)
    n_loc = length(interior_vec)

    S_lift = _interior_q1_lift_to_doubled(node_map_q1, k, n_v, interior_vec, T)
    S64    = SparseMatrixCSC{Float64,Int}(S_lift)
    K_loc  = SparseMatrixCSC{T,Int}(S64' * A_doubled * S64)

    P_amg       = _amg_prolongations(K_loc, T, prolongator)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1
    L_total     = K_amg + 1

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
    end
    refine[K_amg]  = S_lift
    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_loc
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[L_total] = n_doubled

    return refine, sizes, L_total, K_amg
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM3D{T}};
             prolongator = amg_ruge_stuben(max_coarse=2),
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom))) where {T}
    k    = geom.discretization.k
    s    = k + 1
    sk3  = s^3
    K_qk = geom.x                         # 3-tensor (sk3, N, 3)
    N_in = size(K_qk, 2)

    K_corners      = _extract_Kqk_corners(K_qk, k)
    K_corners_flat = _xflat(K_corners)
    K_qk_flat      = _xflat(K_qk)
    mg_fem = _geometric_fem3d_mg(T; L=1, K=K_corners_flat, K_qk=K_qk_flat, k=k, structured=false)
    geom_fem = mg_fem.geometry
    x_fine = _xflat(geom_fem.x)           # flat (sk3 * N_in, 3) view used by sparse ops
    w_fine = geom_fem.w
    n_doubled = N_in * sk3
    @assert size(x_fine, 1) == n_doubled
    N = N_in

    _, full_labels, _ = Mesh3d.deduplicate_vertices(_xflat(K_qk))
    n_full_unique     = maximum(full_labels)

    unique_corners, node_map_q1 = _dedupe(K_corners_flat)
    n_v = size(unique_corners, 1)

    local_c = (1, s, s*(s-1) + 1, s^2,
               (s-1)*s^2 + 1, (s-1)*s^2 + s,
               (s-1)*s^2 + s*(s-1) + 1, s^3)
    full_to_corner = Dict{Int,Int}()
    @inbounds for h in 1:N, c in 1:8
        broken_full   = sk3*(h-1) + local_c[c]
        broken_corner = 8*(h-1) + c
        full_to_corner[full_labels[broken_full]] = node_map_q1[broken_corner]
    end

    dx = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dx])
    dy = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dy])
    dz = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dz])
    W  = spdiagm(0 => Float64.(w_fine))
    A_doubled = dx' * W * dx + dy' * W * dy + dz' * W * dz

    # :full hierarchy (all-corners Neumann); :uniform rides it.
    refine_full, sizes_full, L_full, K_amg_full =
        _fem3d_hierarchy(node_map_q1, k, A_doubled,
                          collect(1:n_v), n_v, n_doubled, prolongator, T)

    # One zero-trace continuous subspace per named dirichlet node set.
    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dirichlet_dedup_set  = Set{Int}(full_labels[r] for r in _pairs_to_linear(nodes, sk3))
        dirichlet_corner_set = Set{Int}(
            full_to_corner[fid] for fid in dirichlet_dedup_set if haskey(full_to_corner, fid))
        interior_corners = sort!(collect(setdiff(1:n_v, dirichlet_corner_set)))
        refine_dir, sizes_dir, L_dir, K_amg_dir =
            _fem3d_hierarchy(node_map_q1, k, A_doubled,
                              interior_corners, n_v, n_doubled, prolongator, T)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for kk in 1:K_amg_dir
            sub[kk] = sparse(one(T) * I, sizes_dir[kk], sizes_dir[kk])
        end
        sub[L_dir] = _p2_continuous_subspace(full_labels, n_full_unique, dirichlet_dedup_set, T)
        return refine_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, sizes_full, L_full, K_amg_full, build_dirichlet)
end

# ============================================================================
# geometric_mg(::Geometry{FEM3D}, L)
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM3D{T}}, L::Int) where {T}
    k = geom.discretization.k
    K_q1_flat = _xflat(geom.discretization.K)   # Mesh3d helpers operate on flat (8N, 3)
    # `_geometric_fem3d_mg`'s `structured` kwarg stays an internal detail: the
    # FEM3D AMG setup uses a sparse companion stiffness, but every user-facing
    # geometry is structured (BlockDiag operators).
    _geometric_fem3d_mg(T; L=L, K=K_q1_flat, k=k, structured=true)
end

# ============================================================================
# Helpers
# ============================================================================

# Generic d-dimensional dedup (random-projection sort + tol-bucket pairwise check).
function _dedupe(x::Matrix{T}) where {T}
    n, d = size(x)
    seed = hash(x)
    rng  = Xoshiro(seed)
    u    = randn(rng, T, d); u ./= norm(u)
    p    = x * u
    P    = sortperm(p)
    tol  = max(maximum(abs, x), one(T)) * 100 * eps(real(T))

    labels = zeros(Int, n)
    count  = 0
    a      = 1
    while a <= n
        if labels[P[a]] == 0
            count += 1
            labels[P[a]] = count
            b = a + 1
            while b <= n && p[P[b]] <= p[P[a]] + tol
                b += 1
            end
            for kk in a+1:b-1
                if norm(@view(x[P[a], :]) .- @view(x[P[kk], :])) <= tol
                    labels[P[kk]] = count
                end
            end
        end
        a += 1
    end

    unique_xy = zeros(T, count, d)
    seen = falses(count)
    @inbounds for kk in 1:n
        l = labels[kk]
        if !seen[l]
            unique_xy[l, :] .= @view x[kk, :]
            seen[l] = true
        end
    end
    return unique_xy, labels
end

# Local lift matrix: (k+1)^3 × 8 trilinear weight from 8 Q1 corners at each Q_k Lagrange node.
function _q1_lift_local(k::Int, ::Type{T}) where {T}
    s     = k + 1
    nodes = T[cos(j * π / k) for j in 0:k]
    L     = zeros(T, s^3, 8)
    @inbounds for iz in 1:s, iy in 1:s, ix in 1:s
        ξ, η, ζ = nodes[ix], nodes[iy], nodes[iz]
        row = (iz - 1) * s^2 + (iy - 1) * s + ix
        for c in 1:8
            a = ((c - 1) & 1) == 0 ? -one(T) : one(T)
            b = ((c - 1) & 2) == 0 ? -one(T) : one(T)
            d = ((c - 1) & 4) == 0 ? -one(T) : one(T)
            L[row, c] = (1 + a*ξ) * (1 + b*η) * (1 + d*ζ) / 8
        end
    end
    return L
end

# Direct bridge: interior-Q1 corners -> doubled Q_k. Same per-element trilinear
# lift pattern as the full corner-Q1 -> Q_k lift, with boundary-corner pushes
# dropped (column index remapped through interior_idx).
function _interior_q1_lift_to_doubled(node_map_q1::Vector{Int}, k::Int, n_v::Int,
                                      interior_corners::Vector{Int},
                                      ::Type{T}) where {T}
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    L_local = _q1_lift_local(k, T)
    s = k + 1
    nrow = s^3
    N = length(node_map_q1) ÷ 8

    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, N * nrow * 8)
    sizehint!(cols, N * nrow * 8)
    sizehint!(vals, N * nrow * 8)

    @inbounds for e in 1:N
        offset = (e - 1) * nrow
        cu  = ntuple(c -> node_map_q1[8*(e-1) + c], 8)
        cui = ntuple(c -> interior_idx[cu[c]], 8)
        for r in 1:nrow, c in 1:8
            v = L_local[r, c]
            if v != 0 && cui[c] != 0
                push!(rows, offset + r)
                push!(cols, cui[c])
                push!(vals, v)
            end
        end
    end
    return sparse(rows, cols, vals, N * nrow, n_int)
end

