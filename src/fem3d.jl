"""
    fem3d(::Type{T}=Float64; L=1, k=3, K, max_coarse=2) -> Geometry

Construct a 3D FEM geometry on the Q_k hex mesh `K` with **Q_k** (tensor-
product Lagrange-on-Chebyshev-Lobatto) elements. The result is passed to
`amgb` to solve a Dirichlet variational problem.

The output `Geometry`'s coordinate matrix equals `K` verbatim when `L=1`
(input mesh = output mesh). For `L>1` the corner mesh is subdivided
geometrically and the Q_k Lagrange nodes are regenerated for the new hexes.

# Arguments

- `L::Int=1`: number of geometric refinement levels. `L=1` keeps `K` as-is;
  `L>1` subdivides the Q1 corner mesh `L−1` times before regenerating the
  Q_k Lagrange nodes.

- `k::Int`: polynomial order of the Q_k Lagrange-Chebyshev basis. Each hex
  carries `(k+1)^3` doubled DOFs. Default `3`.

- `K::Matrix{T}` (size `(k+1)^3 · N × 3`): the Q_k Lagrange-Chebyshev mesh.
  Each `(k+1)^3` consecutive rows give one hex's tensor-product Lagrange
  nodes (x varying fastest, then y, then z). The Q1 corners — which AMG
  uses internally — sit at local block indices `(1, k+1, k(k+1)+1, (k+1)^2,
  (k-1)(k+1)^2 + … )`.

  See `geometric_fem3d` if you want to build a custom `K` from a
  coarse corner mesh.

- `max_coarse`: AMG stopping threshold; default `2`.

# Example
```julia
sol = fem3d_solve(L=3, k=3, p=1.5)      # default unit-cube, 2 subdivisions
```

# Caveat — Dirichlet only

The returned Geometry is intended for Dirichlet boundary conditions.
"""
function fem3d(::Type{T}=Float64;
                         L::Int=1,
                         k::Int=3,
                         K::Matrix{T} = MultiGridBarrier.geometric_fem3d(T;
                                            K=T[-1.0 -1.0 -1.0;  1.0 -1.0 -1.0;
                                                -1.0  1.0 -1.0;  1.0  1.0 -1.0;
                                                -1.0 -1.0  1.0;  1.0 -1.0  1.0;
                                                -1.0  1.0  1.0;  1.0  1.0  1.0],
                                            L=1, k=k, structured=false).x,
                         max_coarse::Int=2, rest...) where {T}
    s   = k + 1
    sk3 = s^3
    size(K, 1) % sk3 == 0 ||
        throw(ArgumentError("K must have (k+1)^3 = $sk3 rows per hex (Q_k Lagrange " *
                            "layout). Build one with `geometric_fem3d(K=corners, L=1, k=k).x`."))
    N_in = size(K, 1) ÷ sk3

    # 0. Extract the Q1 corner mesh from K. Local Q1 corner indices within each
    #    (k+1)^3 Lagrange-Chebyshev tensor block:
    local_c = (1, s, s*(s-1) + 1, s^2,
               (s-1)*s^2 + 1, (s-1)*s^2 + s,
               (s-1)*s^2 + s*(s-1) + 1, s^3)
    K_corners = K[[sk3*(e-1) + p for e in 1:N_in for p in local_c], :]

    # If L > 1, subdivide corners geometrically and regenerate the Q_k mesh on
    # the subdivided corners (intermediate Lagrange nodes are necessarily new).
    # For L = 1, K is preserved verbatim.
    if L > 1
        K_corners = MultiGridBarrier.geometric_fem3d(T; K=K_corners, L=L, k=1, structured=false).x
        K = MultiGridBarrier.geometric_fem3d(T; K=K_corners, L=1, k=k, structured=false).x
    end
    N = size(K_corners, 1) ÷ 8

    # 1. Fine doubled Q_k geometry. Pass K verbatim via K_qk so geom_fem.x = K
    #    when L = 1. structured=false: read geom_fem.operators as sparse.
    geom_fem  = MultiGridBarrier.geometric_fem3d(T; K=K_corners, K_qk=K, L=1, k=k, structured=false)
    x_fine    = geom_fem.x          # N·(k+1)^3 × 3
    w_fine    = geom_fem.w          # N·(k+1)^3
    s         = k + 1
    n_doubled = N * s^3
    @assert size(x_fine, 1) == n_doubled

    # 2. Dedupe the corner mesh; node_map_q1[r] = unique-corner index of K_corners row r.
    unique_corners, node_map_q1 = _dedupe(K_corners)
    n_v = size(unique_corners, 1)

    # 3. Boundary corners via face counting on Q1 hexes.
    boundary_corners = _hex_boundary_corners(node_map_q1, N)
    interior_corners = setdiff(1:n_v, boundary_corners)
    n_int            = length(interior_corners)
    n_int >= 1 || throw(ArgumentError(
        "mesh has no interior corners; need at least one interior vertex"))

    # 4. Trilinear lift: corners → doubled Q_k Lagrange nodes.
    lift = _global_q1_lift(node_map_q1, k, n_v, T)   # n_doubled × n_v

    # 5. P1 (Q1) companion stiffness on corners via Galerkin against geometric_fem3d's
    #    own dx/dy/dz/W. Cleaner than re-deriving Q1 hex stiffness from scratch.
    dx = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dx])
    dy = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dy])
    dz = SparseMatrixCSC{Float64,Int}(geom_fem.operators[:dz])
    W  = spdiagm(0 => Float64.(w_fine))
    A_doubled = dx' * W * dx + dy' * W * dy + dz' * W * dz
    K_corner  = SparseMatrixCSC{T,Int}(lift' * A_doubled * lift)
    K_int     = K_corner[interior_corners, interior_corners]

    # 6. AMG hierarchy on K_int.
    P_amg       = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1

    L_total = K_amg + 2

    # 7. refine[ℓ] / coarsen[ℓ].
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

    # 8. subspaces.
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

    sub_dirichlet[L_total] = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:dirichlet][end])
    sub_full[L_total]      = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:full][end])
    sub_uniform[L_total]   = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:uniform][end])

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => sub_dirichlet,
        :full      => sub_full,
        :uniform   => sub_uniform,
    )

    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => SparseMatrixCSC{T,Int}(geom_fem.operators[:id]),
        :dx => SparseMatrixCSC{T,Int}(geom_fem.operators[:dx]),
        :dy => SparseMatrixCSC{T,Int}(geom_fem.operators[:dy]),
        :dz => SparseMatrixCSC{T,Int}(geom_fem.operators[:dz]),
    )

    # disc.K is the Q1 corner mesh (informational); the doubled Q_k mesh lives
    # in the returned Geometry's x = x_fine = user's K (verbatim if L=1).
    disc = FEM3D{T}(k, K_corners, 1)
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, FEM3D{T}}(
        disc, x_fine, w_fine, subspaces, operators, refine, coarsen
    )
end

"""
    fem3d_solve(::Type{T}=Float64; rest...) -> AMGBSOL

Solve a 3D Dirichlet variational problem with Q_k hexahedral elements on
the mesh you supply. Equivalent to `amgb(fem3d(T; rest...); rest...)`:
keyword arguments are forwarded to both `fem3d` (mesh kwargs
`K`, `L`, `k`, `max_coarse`) and `amgb` (solver kwargs `p`, `f`, `g`,
`verbose`, …).

# Example
```julia
sol = fem3d_solve(L = 3, k = 3, p = 1.5)  # default unit-cube K, 2 subdivisions
```

See `amgb` for the full set of solver kwargs.
"""
fem3d_solve(::Type{T}=Float64; rest...) where {T} =
    amgb(fem3d(T; rest...); rest...)

# ============================================================================
# Helpers
# ============================================================================

# Generic d-dimensional dedup (replaces _dedupe_2d / would-be _dedupe_3d).
# Random-projection sort + tol-bucket pairwise check; mirrors geometric_fem2d_P2.continuous.
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

# Boundary corners via face counting. Each Q1 hex has 6 faces (4 corners each),
# in geometric_fem3d's tensor-product Q1 ordering.
function _hex_boundary_corners(node_map_q1::Vector{Int}, N::Int)
    # Local corner indices (1..8) on each face — derived from Q1 corner
    # ordering (-1,-1,-1), (1,-1,-1), (-1,1,-1), (1,1,-1), (-1,-1,1), (1,-1,1),
    # (-1,1,1), (1,1,1).
    faces_local = (
        (1, 2, 3, 4),   # z = -1   (bottom)
        (5, 6, 7, 8),   # z = +1   (top)
        (1, 2, 5, 6),   # y = -1
        (3, 4, 7, 8),   # y = +1
        (1, 3, 5, 7),   # x = -1
        (2, 4, 6, 8),   # x = +1
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

# Local lift matrix: (k+1)^3 × 8. Trilinear weight from each of the 8 Q1 corners
# at every Q_k Lagrange node.
function _q1_lift_local(k::Int, ::Type{T}) where {T}
    s     = k + 1
    nodes = T[cos(j * π / k) for j in 0:k]   # Chebyshev-Lobatto, descending
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

# Global lift: corner-Q1 → doubled-Q_k. Per-hex local lift composed with the
# hex's 8 unique-corner indices.
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

# Corner-Lagrange-node local indices: position in the (k+1)^3 block of the
# Lagrange node that coincides with each Q1 corner.
# Note: chebyshev_nodes is *descending* (1 → -1), so corner with reference
# coordinate -1 lives at axis index k+1, corner +1 at axis index 1.
function _corner_lagrange_indices(k::Int)
    s = k + 1
    idx(ix, iy, iz) = (iz - 1) * s^2 + (iy - 1) * s + ix
    [idx(s, s, s),     # corner 1 = (-1,-1,-1)
     idx(1, s, s),     # corner 2 = (+1,-1,-1)
     idx(s, 1, s),     # corner 3 = (-1,+1,-1)
     idx(1, 1, s),     # corner 4 = (+1,+1,-1)
     idx(s, s, 1),     # corner 5 = (-1,-1,+1)
     idx(1, s, 1),     # corner 6 = (+1,-1,+1)
     idx(s, 1, 1),     # corner 7 = (-1,+1,+1)
     idx(1, 1, 1)]     # corner 8 = (+1,+1,+1)
end

# Left inverse of `lift`: pick one corner Lagrange node per unique corner.
# Satisfies `coarsen * lift = I` exactly because at a corner Lagrange node,
# the trilinear weight from the matching Q1 corner is 1 and from all others 0.
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
