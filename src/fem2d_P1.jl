"""
    fem2d_P1(::Type{T}=Float64; K, L=1, max_coarse=2) -> Geometry

Construct a 2D FEM geometry on the triangulation `K` with **P1**
(piecewise-linear) elements (3 vertex DOFs per triangle, no edge midpoints
or bubble). The result is passed to `amgb`.

This is the lower-order alternative to `fem2d_P2`: same input
format, faster and less expressive. Use it when the higher-order P2+bubble
is not needed (e.g., when smoothness of the solution is limited by the
geometry or BCs).

# Arguments

- `K::Matrix{T}` (size `3N × 2`): the triangulation. Each three consecutive
  rows give one triangle's three vertex coordinates `(x, y)`:
  ```
  row 3k-2:  vertex 1 of triangle k
  row 3k-1:  vertex 2 of triangle k
  row 3k  :  vertex 3 of triangle k
  ```
  Vertices are deduplicated by coordinate. Vertex order within a triangle
  is unconstrained.

- `L::Int=1`: number of geometric refinement levels. `L=1` uses `K` as-is;
  `L>1` subdivides `K` geometrically `L−1` times via `geometric_fem2d_P1`
  to produce the fine mesh on which AMG transfer operators are then built.

- `max_coarse`: AMG stopping threshold; default `2` (deepest hierarchy).

# Example
```julia
# Build your fine 3-DOF corner triangulation `K` (or use the package default), then:
geom = fem2d_P1(; K=K)
sol  = amgb(geom; p=2.0)
```

# Caveat — Dirichlet only

Same as `fem2d_P2`: the Geometry is intended for Dirichlet BCs.
"""
function fem2d_P1(::Type{T}=Float64;
                            K::Matrix{T} = T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1],
                            L::Int=1,
                            max_coarse::Int=2, rest...) where {T}
    size(K, 1) % 3 == 0 ||
        throw(ArgumentError("K must have 3 rows per triangle (3N × 2)"))

    # 1. Fine geometry via geometric_fem2d_P1; reuse its operators (id, dx, dy),
    #    quadrature, and fine subspaces. L=1 means no subdivision; L>1 subdivides
    #    K geometrically L−1 times to produce the fine mesh on which AMG is built.
    geom_fem  = geometric_fem2d_P1(T; K=K, L=L)
    N = size(geom_fem.x, 1) ÷ 3
    x_fine    = geom_fem.x          # 3N × 2
    n_doubled = size(x_fine, 1)
    @assert n_doubled == 3*N

    # 2. Dedupe corners. x_fine IS the corner mesh in doubled-DOF order.
    unique_corners, labels = _dedupe(x_fine)
    n_v = size(unique_corners, 1)
    tri_conn = collect(transpose(reshape(labels, 3, N)))   # N × 3

    # 3. Boundary corners.
    boundary_corners = _find_boundary_corners(tri_conn)
    interior_corners = setdiff(1:n_v, boundary_corners)
    n_int            = length(interior_corners)
    n_int >= 1 || throw(ArgumentError(
        "mesh has no interior corners; need at least one interior vertex"))

    # 4. Continuous P1 Dirichlet stiffness on interior corners.
    K_full = _assemble_p1_stiffness_full(unique_corners, tri_conn)
    K_int  = K_full[interior_corners, interior_corners]

    # 5. AMG.
    P_amg       = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1

    L_total = K_amg + 2

    # 6. refine[ℓ] / coarsen[ℓ].
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
        coarsen[kk] = _amg_injection(P_amg[i])
    end

    refine[K_amg]  = _interior_to_full_corners(n_v, interior_corners, T)
    coarsen[K_amg] = _full_to_interior_corners(n_v, interior_corners, T)

    refine[K_amg + 1]  = _corners_to_doubled_p1(tri_conn, n_v, T)
    coarsen[K_amg + 1] = _doubled_to_corners_pick_p1(tri_conn, n_v, T)

    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    # 7. Subspaces.
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
    )

    disc = FEM2D_P1{T}(K, 1)
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, FEM2D_P1{T}}(
        disc, x_fine, geom_fem.w, subspaces, operators, refine, coarsen
    )
end

"""
    fem2d_P1_solve(::Type{T}=Float64; rest...) -> AMGBSOL

Solve a 2D Dirichlet variational problem with P1 triangular elements on
the triangulation you supply. Equivalent to
`amgb(fem2d_P1(T; rest...); rest...)`: keyword arguments are forwarded to
both `fem2d_P1` (geometry kwargs — see its docstring) and `amgb` (solver
kwargs `p`, `f`, `g`, `verbose`, …).
"""
fem2d_P1_solve(::Type{T}=Float64; rest...) where {T} =
    amgb(fem2d_P1(T; rest...); rest...)

# Doubling map (continuous corners → doubled per-element corners). For each
# triangle e, place the corner-c value at fine row 3(e-1)+c.
function _corners_to_doubled_p1(tri_conn::Matrix{Int}, n_v::Int, ::Type{T}) where {T}
    N = size(tri_conn, 1)
    rows = Vector{Int}(undef, 3*N)
    cols = Vector{Int}(undef, 3*N)
    @inbounds for e in 1:N
        rows[3*(e-1) + 1] = 3*(e-1) + 1; cols[3*(e-1) + 1] = tri_conn[e, 1]
        rows[3*(e-1) + 2] = 3*(e-1) + 2; cols[3*(e-1) + 2] = tri_conn[e, 2]
        rows[3*(e-1) + 3] = 3*(e-1) + 3; cols[3*(e-1) + 3] = tri_conn[e, 3]
    end
    return sparse(rows, cols, ones(T, 3*N), 3*N, n_v)
end

# Left inverse of the doubling map: pick one doubled copy per unique corner.
function _doubled_to_corners_pick_p1(tri_conn::Matrix{Int}, n_v::Int, ::Type{T}) where {T}
    N = size(tri_conn, 1)
    chosen = zeros(Int, n_v)
    @inbounds for e in 1:N, c in 1:3
        v = tri_conn[e, c]
        if chosen[v] == 0
            chosen[v] = 3*(e - 1) + c
        end
    end
    @assert all(chosen .> 0)
    return sparse(1:n_v, chosen, ones(T, n_v), n_v, 3*N)
end
