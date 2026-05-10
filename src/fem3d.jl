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
    Mesh3d.promote_to_Qk(K_q1, k)
end

# Extract the 8N × 3 Q1 corner mesh from a Q_k Lagrange-Chebyshev mesh `K`.
function _extract_Kqk_corners(K::Matrix{T}, k::Int) where {T}
    s = k + 1
    sk3 = s^3
    N = size(K, 1) ÷ sk3
    local_c = (1, s, s*(s-1) + 1, s^2,
               (s-1)*s^2 + 1, (s-1)*s^2 + s,
               (s-1)*s^2 + s*(s-1) + 1, s^3)
    Matrix{T}(K[[sk3*(e-1) + p for e in 1:N for p in local_c], :])
end

"""
    fem3d(::Type{T}=Float64; k=3, K=<default unit-cube Q_k>) -> Geometry

Construct a **single-level** 3D Q_k FEM `Geometry` on the Lagrange-Chebyshev mesh `K`
(`(k+1)^3 N × 3`). Use `amg(geom)` for algebraic-MG or `geometric_mg(geom, L)` for a
geometric hierarchy.

# Arguments
- `k::Int=3`: polynomial order of the Q_k basis.
- `K::Matrix{T}` (`(k+1)^3 N × 3`): Q_k Lagrange-Chebyshev mesh; each `(k+1)^3`
  consecutive rows give one hex's tensor-product Lagrange nodes (x fastest, then y, then z).

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem3d(::Type{T}=Float64;
                k::Int=3,
                K::Matrix{T} = _default_K_qk(T, k),
                rest...) where {T}
    s   = k + 1
    sk3 = s^3
    size(K, 1) % sk3 == 0 ||
        throw(ArgumentError("K must have (k+1)^3 = $sk3 rows per hex (Q_k Lagrange layout)."))

    K_corners = _extract_Kqk_corners(K, k)
    mg = _geometric_fem3d_mg(T; L=1, K=K_corners, K_qk=K, k=k, structured=false)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM3D}) — AMG on Q1 corners with companion Galerkin stiffness.
# ============================================================================
function amg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM3D{T}};
             max_coarse::Int=2) where {T}
    k = geom.discretization.k
    s = k + 1
    sk3 = s^3
    K_qk = geom.x
    N_in = size(K_qk, 1) ÷ sk3

    K_corners = _extract_Kqk_corners(K_qk, k)
    mg_fem = _geometric_fem3d_mg(T; L=1, K=K_corners, K_qk=K_qk, k=k, structured=false)
    geom_fem = mg_fem.geometry
    x_fine = geom_fem.x
    w_fine = geom_fem.w
    n_doubled = N_in * sk3
    @assert size(x_fine, 1) == n_doubled
    N = N_in

    unique_corners, node_map_q1 = _dedupe(K_corners)
    n_v = size(unique_corners, 1)

    boundary_corners = _hex_boundary_corners(node_map_q1, N)
    interior_corners = setdiff(1:n_v, boundary_corners)
    n_int            = length(interior_corners)
    n_int >= 1 || throw(ArgumentError(
        "mesh has no interior corners; need at least one interior vertex"))

    lift = _global_q1_lift(node_map_q1, k, n_v, T)

    dx = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dx])
    dy = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dy])
    dz = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dz])
    W  = spdiagm(0 => Float64.(w_fine))
    A_doubled = dx' * W * dx + dy' * W * dy + dz' * W * dz
    K_corner  = SparseMatrixCSC{T,Int}(lift' * A_doubled * lift)
    K_int     = K_corner[interior_corners, interior_corners]

    P_amg       = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1

    L_total = K_amg + 2

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
        coarsen[kk] = _amg_injection(P_amg[i])
    end

    refine[K_amg]  = _interior_to_full_corners(n_v, interior_corners, T)
    coarsen[K_amg] = _full_to_interior_corners(n_v, interior_corners, T)

    refine[K_amg + 1]  = SparseMatrixCSC{T,Int}(lift)
    coarsen[K_amg + 1] = _doubled_to_corners_pick_q1(node_map_q1, k, n_v, T)

    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_int
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[K_amg + 1] = n_v
    sizes[L_total]   = n_doubled

    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for kk in 1:K_amg
        sub_dirichlet[kk] = sparse(one(T) * I, sizes[kk], sizes[kk])
        sub_full[kk]      = sparse(one(T) * I, sizes[kk], sizes[kk])
        sub_uniform[kk]   = sparse(ones(T, sizes[kk], 1))
    end

    sub_dirichlet[K_amg + 1] = _interior_to_full_corners(n_v, interior_corners, T)
    sub_full[K_amg + 1]      = sparse(one(T) * I, n_v, n_v)
    sub_uniform[K_amg + 1]   = sparse(ones(T, n_v, 1))

    sub_dirichlet[L_total] = SparseMatrixCSC{T,Int}(geom.subspaces[:dirichlet])
    sub_full[L_total]      = SparseMatrixCSC{T,Int}(geom.subspaces[:full])
    sub_uniform[L_total]   = SparseMatrixCSC{T,Int}(geom.subspaces[:uniform])

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => sub_dirichlet,
        :full      => sub_full,
        :uniform   => sub_uniform,
    )
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# ============================================================================
# geometric_mg(::Geometry{FEM3D}, L)
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}, L::Int;
                      structured::Bool=true) where {T}
    k = geom.discretization.k
    K_q1 = geom.discretization.K
    _geometric_fem3d_mg(T; L=L, K=K_q1, k=k, structured=structured)
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

# Boundary corners via face counting on Q1 hexes.
function _hex_boundary_corners(node_map_q1::Vector{Int}, N::Int)
    faces_local = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (1, 2, 5, 6),
        (3, 4, 7, 8),
        (1, 3, 5, 7),
        (2, 4, 6, 8),
    )
    face_count = Dict{NTuple{4, Int}, Int}()
    for e in 1:N
        base = 8 * (e - 1)
        for face in faces_local
            uni = (node_map_q1[base + face[1]], node_map_q1[base + face[2]],
                   node_map_q1[base + face[3]], node_map_q1[base + face[4]])
            sorted_uni = NTuple{4, Int}(sort!(collect(uni)))
            face_count[sorted_uni] = get(face_count, sorted_uni, 0) + 1
        end
    end
    bset = Set{Int}()
    for (face, c) in face_count
        if c == 1
            for v in face
                push!(bset, v)
            end
        end
    end
    return sort!(collect(bset))
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

# Global lift: corner-Q1 → doubled-Q_k.
function _global_q1_lift(node_map_q1::Vector{Int}, k::Int, n_v::Int, ::Type{T}) where {T}
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
        cu = ntuple(c -> node_map_q1[8*(e-1) + c], 8)
        for r in 1:nrow, c in 1:8
            v = L_local[r, c]
            if v != 0
                push!(rows, offset + r)
                push!(cols, cu[c])
                push!(vals, v)
            end
        end
    end
    return sparse(rows, cols, vals, N * nrow, n_v)
end

# Pick one corner-Lagrange node per Q1 corner so that coarsen * lift = I exactly.
function _corner_lagrange_indices(k::Int)
    s = k + 1
    idx(ix, iy, iz) = (iz - 1) * s^2 + (iy - 1) * s + ix
    [idx(s, s, s),
     idx(1, s, s),
     idx(s, 1, s),
     idx(1, 1, s),
     idx(s, s, 1),
     idx(1, s, 1),
     idx(s, 1, 1),
     idx(1, 1, 1)]
end

function _doubled_to_corners_pick_q1(node_map_q1::Vector{Int}, k::Int,
                                     n_v::Int, ::Type{T}) where {T}
    s = k + 1
    nrow = s^3
    N = length(node_map_q1) ÷ 8
    corner_lag = _corner_lagrange_indices(k)
    chosen = zeros(Int, n_v)
    @inbounds for e in 1:N
        for c in 1:8
            v = node_map_q1[8*(e-1) + c]
            if chosen[v] == 0
                chosen[v] = (e - 1) * nrow + corner_lag[c]
            end
        end
    end
    @assert all(chosen .> 0)
    return sparse(1:n_v, chosen, ones(T, n_v), n_v, N * nrow)
end
