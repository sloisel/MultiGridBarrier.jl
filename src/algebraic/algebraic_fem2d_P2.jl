"""
    algebraic_fem2d_P2(::Type{T}=Float64; K, max_coarse=2) -> Geometry

Construct a 2D FEM geometry on the triangulation `K` with **P2 + cubic-
bubble** elements (7 DOFs per triangle: 3 vertices + 3 edge midpoints +
1 centroid). The result is passed to `amgb` to solve a Dirichlet
variational problem.

# Arguments

- `K::Matrix{T}` (size `3N × 2`): the fine triangulation. Each three
  consecutive rows give one triangle's three vertex coordinates `(x, y)`:
  ```
  row 3k-2:  (x, y) of vertex 1 of triangle k
  row 3k-1:  (x, y) of vertex 2 of triangle k
  row 3k  :  (x, y) of vertex 3 of triangle k
  ```
  Triangulation must cover the domain without gaps. Triangles share
  vertices by *coincident coordinates* (the package deduplicates them);
  there is no separate connectivity array. Vertex order within a triangle
  is unconstrained — orientation is not used.

- `max_coarse`: AMG keeps coarsening until the coarsest level has at most
  this many DOFs. Default `2` matches `MultiGridBarrier.fem2d_P2`'s geometric
  depth (a single global mode at the coarsest level).

# Example
```julia
# Two-triangle unit square on [-1, 1]²
K = Float64[-1 -1;  1 -1; -1  1;       # triangle 1
             1 -1;  1  1; -1  1]       # triangle 2
geom = algebraic_fem2d_P2(; K=K)
sol  = amgb(geom; p=1.5)               # solve p-Laplace
```

# Subdividing a coarser mesh

If you want a finer triangulation than your `K` provides, build it with
`MultiGridBarrier.fem2d_P2` (or any other mesher) and extract the fine
corners:

```julia
fine    = MultiGridBarrier.fem2d_P2(K=K_coarse, L=3)   # 3 levels of subdivision
N       = size(fine.x, 1) ÷ 7
K_fine  = fine.x[[7*(k-1) + p for k in 1:N for p in (1, 3, 5)], :]
geom    = algebraic_fem2d_P2(; K=K_fine)
```
The slice `(1, 3, 5)` are the local-DOF indices of the three triangle
corners in `fem2d_P2`'s 7-DOF block.

# Caveat — Dirichlet only

The returned Geometry is intended for Dirichlet boundary conditions. The
`:full` and `:uniform` subspaces are populated for API compatibility but
their semantics at coarse levels do not include boundary DOFs.
"""
function algebraic_fem2d_P2(::Type{T}=Float64;
                         K = T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1],
                         max_coarse::Int=2, rest...) where {T}
    K_T = collect(T, K)

    # 1. Build the doubled P2+bubble representation on K. We use fem2d_P2 with
    #    L=1 (no subdivision) — it does the per-element 7-DOF assembly and
    #    builds the operators (id, dx, dy) and quadrature for us.
    # structured=false: this code reads geom_fem.operators as sparse matrices.
    geom_fem = MultiGridBarrier.fem2d_P2(T; K=K_T, L=1, structured=false)
    x_fine   = geom_fem.x          # 7N × 2
    w_fine   = geom_fem.w          # 7N
    n_doubled = size(x_fine, 1)
    @assert n_doubled % 7 == 0
    N = n_doubled ÷ 7              # number of fine triangles

    # 2. Extract corner vertices and triangle connectivity from the doubled
    #    coordinates (positions 1, 3, 5 in each 7-block are corners).
    corners, tri_conn = _extract_corners_and_connectivity(x_fine, N)
    n_v = size(corners, 1)

    # 3. Identify boundary corners (corners on edges shared by ≤ 1 triangle).
    boundary_corners = _find_boundary_corners(tri_conn)
    interior_corners = setdiff(1:n_v, boundary_corners)
    n_int = length(interior_corners)
    n_int >= 1 || throw(ArgumentError(
        "mesh has no interior corners; need at least one interior vertex"))

    # 4. Continuous P1 Dirichlet stiffness on interior corners.
    K_full = _assemble_p1_stiffness_full(corners, tri_conn)
    K_int  = K_full[interior_corners, interior_corners]

    # 5. AMG hierarchy on K_int.
    P_amg       = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1     # number of AMG-interior levels (≥ 1)

    L_total = K_amg + 2                # + 2 bridge layers

    # 6. refine[ℓ] / coarsen[ℓ].
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for i in 1:n_amg_steps
        k = K_amg - i
        refine[k]  = P_amg[i]
        coarsen[k] = _amg_injection(P_amg[i])
    end

    refine[K_amg]  = _interior_to_full_corners(n_v, interior_corners, T)
    coarsen[K_amg] = _full_to_interior_corners(n_v, interior_corners, T)

    refine[K_amg + 1]  = _corners_to_doubled_p2_bubble(tri_conn, n_v, T)
    coarsen[K_amg + 1] = _doubled_to_corners_pick(tri_conn, n_v, T)

    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    # 7. subspaces.
    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_int
    for k in K_amg-1:-1:1
        sizes[k] = size(refine[k], 2)
    end
    sizes[K_amg + 1] = n_v
    sizes[L_total]   = n_doubled

    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    # AMG-interior levels: identity (level is already interior-only).
    for k in 1:K_amg
        sub_dirichlet[k] = sparse(one(T) * I, sizes[k], sizes[k])
        sub_full[k]      = sparse(one(T) * I, sizes[k], sizes[k])
        sub_uniform[k]   = sparse(ones(T, sizes[k], 1))
    end

    # Continuous-corner level: :dirichlet drops boundary corner cols.
    sub_dirichlet[K_amg + 1] = _interior_to_full_corners(n_v, interior_corners, T)
    sub_full[K_amg + 1]      = sparse(one(T) * I, n_v, n_v)
    sub_uniform[K_amg + 1]   = sparse(ones(T, n_v, 1))

    # Doubled fine: reuse fem2d_P2's own subspaces (continuity + zero-boundary
    # via `MultiGridBarrier.continuous(x[L])`, identity, and ones).
    sub_dirichlet[L_total] = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:dirichlet][end])
    sub_full[L_total]      = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:full][end])
    sub_uniform[L_total]   = SparseMatrixCSC{T,Int}(geom_fem.subspaces[:uniform][end])

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => sub_dirichlet,
        :full      => sub_full,
        :uniform   => sub_uniform,
    )

    # 8. Operators: reuse fem2d_P2's id/dx/dy (acting at the doubled fine level).
    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => SparseMatrixCSC{T,Int}(geom_fem.operators[:id]),
        :dx => SparseMatrixCSC{T,Int}(geom_fem.operators[:dx]),
        :dy => SparseMatrixCSC{T,Int}(geom_fem.operators[:dy]),
    )

    disc = FEM2D_P2{T}(K_T, 1)
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, FEM2D_P2{T}}(
        disc, x_fine, w_fine, subspaces, operators, refine, coarsen
    )
end

"""
    algebraic_fem2d_P2_solve(::Type{T}=Float64; rest...) -> AMGBSOL

Solve a 2D Dirichlet variational problem with P2+bubble triangular elements
on the triangulation you supply. Equivalent to
`amgb(algebraic_fem2d_P2(T; rest...); rest...)`: keyword arguments are
forwarded to both `algebraic_fem2d_P2` (mesh kwargs `K`, `max_coarse`) and
`amgb` (solver kwargs `p`, `f`, `g`, `verbose`, …).

# Example
```julia
sol = algebraic_fem2d_P2_solve(p = 1.5)        # default unit-square K
```

See `amgb` for the full set of solver kwargs.
"""
algebraic_fem2d_P2_solve(::Type{T}=Float64; rest...) where {T} =
    amgb(algebraic_fem2d_P2(T; rest...); rest...)

# ============================================================================
# Helpers
# ============================================================================

# Dimension-agnostic dedup is defined in algebraic_fem3d.jl (`_dedupe`).

# Given fine doubled coordinates (7N × 2), return unique corners and the
# triangle-corner connectivity (N × 3).
function _extract_corners_and_connectivity(x_fine::Matrix{T}, N::Int) where {T}
    # Corner positions in each 7-block: 1, 3, 5.
    corner_rows = Vector{Int}(undef, 3*N)
    @inbounds for k in 1:N
        corner_rows[3*(k-1) + 1] = 7*(k-1) + 1
        corner_rows[3*(k-1) + 2] = 7*(k-1) + 3
        corner_rows[3*(k-1) + 3] = 7*(k-1) + 5
    end
    x_corners = x_fine[corner_rows, :]                     # 3N × 2 with dups
    unique_xy, labels = _dedupe(x_corners)
    tri_conn = collect(transpose(reshape(labels, 3, N)))   # N × 3
    return unique_xy, tri_conn
end

# Boundary corners: corners on edges that appear in only one triangle.
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

# Continuous P1 Laplacian stiffness on the corner mesh (n_v × n_v).
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

# n_v × n_int matrix embedding interior-corner subspace into full corners.
function _interior_to_full_corners(n_v::Int, interior::Vector{Int}, ::Type{T}) where {T}
    n_int = length(interior)
    sparse(interior, 1:n_int, ones(T, n_int), n_v, n_int)
end

# n_int × n_v matrix selecting interior-corner rows.
function _full_to_interior_corners(n_v::Int, interior::Vector{Int}, ::Type{T}) where {T}
    n_int = length(interior)
    sparse(1:n_int, interior, ones(T, n_int), n_int, n_v)
end

# 7N × n_v: linear-function lift from continuous corner P1 to the doubled
# P2+bubble basis (per-element 7 DOFs).
#   pos 1, 3, 5: corner values
#   pos 2, 4, 6: edge-midpoint averages
#   pos 7      : centroid average
function _corners_to_doubled_p2_bubble(tri_conn::Matrix{Int}, n_v::Int, ::Type{T}) where {T}
    N = size(tri_conn, 1)
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, 12*N); sizehint!(cols, 12*N); sizehint!(vals, 12*N)
    half  = T(1) / 2
    third = T(1) / 3
    @inbounds for k in 1:N
        a, b, c = tri_conn[k, 1], tri_conn[k, 2], tri_conn[k, 3]
        base = 7*(k - 1)
        # vertices
        push!(rows, base+1); push!(cols, a); push!(vals, T(1))
        push!(rows, base+3); push!(cols, b); push!(vals, T(1))
        push!(rows, base+5); push!(cols, c); push!(vals, T(1))
        # edge midpoints
        push!(rows, base+2); push!(cols, a); push!(vals, half)
        push!(rows, base+2); push!(cols, b); push!(vals, half)
        push!(rows, base+4); push!(cols, b); push!(vals, half)
        push!(rows, base+4); push!(cols, c); push!(vals, half)
        push!(rows, base+6); push!(cols, c); push!(vals, half)
        push!(rows, base+6); push!(cols, a); push!(vals, half)
        # centroid
        push!(rows, base+7); push!(cols, a); push!(vals, third)
        push!(rows, base+7); push!(cols, b); push!(vals, third)
        push!(rows, base+7); push!(cols, c); push!(vals, third)
    end
    return sparse(rows, cols, vals, 7*N, n_v)
end

# n_v × 7N: pick one corner-position doubled DOF per unique corner so that
# coarsen * refine = I exactly.
function _doubled_to_corners_pick(tri_conn::Matrix{Int}, n_v::Int, ::Type{T}) where {T}
    N = size(tri_conn, 1)
    chosen = zeros(Int, n_v)
    @inbounds for k in 1:N
        for p in 1:3
            v = tri_conn[k, p]
            if chosen[v] == 0
                chosen[v] = 7*(k - 1) + (2*p - 1)   # positions 1, 3, 5
            end
        end
    end
    @assert all(chosen .> 0)
    return sparse(1:n_v, chosen, ones(T, n_v), n_v, 7*N)
end
