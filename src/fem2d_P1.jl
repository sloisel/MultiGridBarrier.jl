export FEM2D_P1, fem2d_P1

"""
    FEM2D_P1{T}

Discretization tag for 2D P1 triangular FEM. Stores the user's fine 3N×2 doubled
corner triangulation `K`.
"""
struct FEM2D_P1{T}
    K::Matrix{T}
end

MultiGridBarrier.amg_dim(::FEM2D_P1) = 2
_default_block_size(::FEM2D_P1) = 3

function plot(M::Geometry{T, Matrix{T}, Vector{T}, <:Any, <:Any, FEM2D_P1{T}}, z::Vector{T}; kwargs...) where {T}
    x = M.x[:,1]
    y = M.x[:,2]
    N = size(x,1) ÷ 3
    S = reshape(0:(3*N-1), 3, N)'
    plot_trisurf(x, y, z, triangles=S; kwargs...)
end

"""
    fem2d_P1(::Type{T}=Float64; K=<default unit-square>) -> Geometry

Construct a **single-level** 2D FEM P1 `Geometry` on the doubled-per-element fine
triangulation `K` (`3N × 2`). Use `amg(geom)` or `geometric_mg(geom, L)` to attach a
multigrid hierarchy.

# Arguments
- `K::Matrix{T}` (`3N × 2`): doubled-per-element fine triangulation; each three
  consecutive rows are one triangle's three vertex coordinates `(x, y)`.

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem2d_P1(::Type{T}=Float64;
                  K::Matrix{T} = T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1],
                  rest...) where {T}
    size(K, 1) % 3 == 0 ||
        throw(ArgumentError("K must have 3 rows per triangle (3N × 2)"))

    mg = _fem2d_P1_geometric_mg(T, K, 1)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM2D_P1}) — algebraic-MG hierarchy.
# ============================================================================
function amg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM2D_P1{T}};
             max_coarse::Int=2) where {T}
    x_fine    = geom.x
    n_doubled = size(x_fine, 1)
    N = n_doubled ÷ 3

    unique_corners, labels = _dedupe(x_fine)
    n_v = size(unique_corners, 1)
    tri_conn = collect(transpose(reshape(labels, 3, N)))

    boundary_corners = _find_boundary_corners(tri_conn)
    interior_corners = setdiff(1:n_v, boundary_corners)
    n_int            = length(interior_corners)
    n_int >= 1 || throw(ArgumentError(
        "mesh has no interior corners; need at least one interior vertex"))

    K_full = _assemble_p1_stiffness_full(unique_corners, tri_conn)
    K_int  = K_full[interior_corners, interior_corners]

    P_amg       = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1
    L_total = K_amg + 1

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
        coarsen[kk] = _amg_injection(P_amg[i])
    end

    refine[K_amg]  = _interior_corners_to_doubled_p1(tri_conn, n_v, interior_corners, T)
    coarsen[K_amg] = _doubled_to_interior_corners_pick_p1(tri_conn, n_v, interior_corners, T)

    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_int
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[L_total] = n_doubled

    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)

    for kk in 1:K_amg
        sub_dirichlet[kk] = sparse(one(T) * I, sizes[kk], sizes[kk])
        sub_full[kk]      = sparse(one(T) * I, sizes[kk], sizes[kk])
        sub_uniform[kk]   = sparse(ones(T, sizes[kk], 1))
    end

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
# geometric_mg(::Geometry{FEM2D_P1}, L)
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM2D_P1{T}}, L::Int;
                      structured::Bool=false) where {T}
    # Start from the (coarse) K stored on the discretization tag.
    _fem2d_P1_geometric_mg(T, geom.discretization.K, L)
end

# Internal: build geometric L-level multigrid for a P1 triangulation `K`.
function _fem2d_P1_geometric_mg(::Type{T}, K::Matrix{T}, L::Int) where {T}
    size(K, 1) % 3 == 0 ||
        throw(ArgumentError("K must have 3 rows per triangle (3n × 2)"))
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))

    nn = size(K, 1) ÷ 3

    R_K       = sparse(T(1) * I, 3, 3)
    R_refine  = _p1_reference_refine(T)
    R_coarsen = _p1_reference_coarsen(T)

    x       = Vector{Matrix{T}}(undef, L)
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    x[1] = blockdiag([R_K for _ in 1:nn]...) * K

    for l in 1:L-1
        n_tri      = nn * 4^(l - 1)
        refine[l]  = blockdiag([R_refine  for _ in 1:n_tri]...)
        coarsen[l] = blockdiag([R_coarsen for _ in 1:n_tri]...)
        x[l + 1]   = refine[l] * x[l]
    end

    n_doubled = size(x[L], 1)
    N_fine    = n_doubled ÷ 3

    id    = sparse(one(T) * I, n_doubled, n_doubled)
    dx_op, dy_op, w = _p1_assemble_operators(x[L], N_fine, T)

    refine[L]  = id
    coarsen[L] = id

    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    for l in 1:L
        sub_dirichlet[l] = _continuous_p1(x[l])
        sub_full[l]      = sparse(one(T) * I, size(x[l], 1), size(x[l], 1))
        sub_uniform[l]   = sparse(ones(T, size(x[l], 1), 1))
    end

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => sub_dirichlet,
        :full      => sub_full,
        :uniform   => sub_uniform,
    )
    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => id, :dx => dx_op, :dy => dy_op,
    )

    disc = FEM2D_P1{T}(K)
    geom = Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int}, FEM2D_P1{T}}(
        disc, x[L], w,
        Dict{Symbol,SparseMatrixCSC{T,Int}}(
            :dirichlet => sub_dirichlet[end],
            :full      => sub_full[end],
            :uniform   => sub_uniform[end]),
        operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# ============================================================================
# Helpers
# ============================================================================

# Doubling map (continuous corners → doubled per-element corners).
# Direct bridge from interior-P1 corners to doubled P1 basis: per triangle, each
# of the 3 doubled vertex DOFs receives its corner's interior coefficient (or no
# entry if the corner is on the Dirichlet boundary).
function _interior_corners_to_doubled_p1(tri_conn::Matrix{Int}, n_v::Int,
                                         interior_corners::Vector{Int},
                                         ::Type{T}) where {T}
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    N = size(tri_conn, 1)
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, 3*N); sizehint!(cols, 3*N); sizehint!(vals, 3*N)
    @inbounds for e in 1:N
        for c in 1:3
            vi = interior_idx[tri_conn[e, c]]
            if vi != 0
                push!(rows, 3*(e-1) + c); push!(cols, vi); push!(vals, T(1))
            end
        end
    end
    return sparse(rows, cols, vals, 3*N, n_int)
end

function _doubled_to_interior_corners_pick_p1(tri_conn::Matrix{Int}, n_v::Int,
                                              interior_corners::Vector{Int},
                                              ::Type{T}) where {T}
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    N = size(tri_conn, 1)
    chosen = zeros(Int, n_int)
    @inbounds for e in 1:N, c in 1:3
        vi = interior_idx[tri_conn[e, c]]
        if vi != 0 && chosen[vi] == 0
            chosen[vi] = 3*(e - 1) + c
        end
    end
    @assert all(chosen .> 0)
    return sparse(1:n_int, chosen, ones(T, n_int), n_int, 3*N)
end

# Refine map per parent triangle: 12×3.
function _p1_reference_refine(::Type{T}) where {T}
    sparse(T[
        1.0 0.0 0.0;     # child 0, corner 1 = P1
        0.5 0.5 0.0;     # child 0, corner 2 = M12
        0.5 0.0 0.5;     # child 0, corner 3 = M31
        0.5 0.5 0.0;     # child 1, corner 1 = M12
        0.0 1.0 0.0;     # child 1, corner 2 = P2
        0.0 0.5 0.5;     # child 1, corner 3 = M23
        0.5 0.0 0.5;     # child 2, corner 1 = M31
        0.0 0.5 0.5;     # child 2, corner 2 = M23
        0.0 0.0 1.0;     # child 2, corner 3 = P3
        0.5 0.5 0.0;     # child 3, corner 1 = M12
        0.0 0.5 0.5;     # child 3, corner 2 = M23
        0.5 0.0 0.5;     # child 3, corner 3 = M31
    ])
end

function _p1_reference_coarsen(::Type{T}) where {T}
    rows = [1, 2, 3]
    cols = [1, 5, 9]
    sparse(rows, cols, ones(T, 3), 3, 12)
end

function _p1_assemble_operators(x_fine::Matrix{T}, N::Int, ::Type{T}) where {T}
    rows_dx = Vector{Int}(undef, 9*N); cols_dx = Vector{Int}(undef, 9*N); vals_dx = Vector{T}(undef, 9*N)
    rows_dy = Vector{Int}(undef, 9*N); cols_dy = Vector{Int}(undef, 9*N); vals_dy = Vector{T}(undef, 9*N)
    w = zeros(T, 3*N)
    @inbounds for k in 1:N
        i1, i2, i3 = 3*(k-1) + 1, 3*(k-1) + 2, 3*(k-1) + 3
        x1, y1 = x_fine[i1, 1], x_fine[i1, 2]
        x2, y2 = x_fine[i2, 1], x_fine[i2, 2]
        x3, y3 = x_fine[i3, 1], x_fine[i3, 2]
        det2 = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
        area = abs(det2) / 2
        b = (y2 - y3, y3 - y1, y1 - y2)
        c = (x3 - x2, x1 - x3, x2 - x1)
        for i in 1:3, j in 1:3
            idx = 9*(k-1) + 3*(i-1) + j
            rows_dx[idx] = 3*(k-1) + i
            cols_dx[idx] = 3*(k-1) + j
            vals_dx[idx] = b[j] / det2
            rows_dy[idx] = 3*(k-1) + i
            cols_dy[idx] = 3*(k-1) + j
            vals_dy[idx] = c[j] / det2
        end
        w[i1] = area / 3
        w[i2] = area / 3
        w[i3] = area / 3
    end
    dx = sparse(rows_dx, cols_dx, vals_dx, 3*N, 3*N)
    dy = sparse(rows_dy, cols_dy, vals_dy, 3*N, 3*N)
    return dx, dy, w
end

function _continuous_p1(x::Matrix{T}) where {T}
    n = size(x, 1)
    @assert n % 3 == 0
    N = n ÷ 3

    unique_xy, labels = _dedupe(x)
    n_v = size(unique_xy, 1)

    tri_conn = collect(transpose(reshape(labels, 3, N)))
    boundary = _find_boundary_corners(tri_conn)
    interior = setdiff(1:n_v, boundary)
    n_int    = length(interior)

    interior_pos = Dict{Int, Int}()
    for (j, ui) in enumerate(interior)
        interior_pos[ui] = j
    end

    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, n); sizehint!(cols, n); sizehint!(vals, n)
    for i in 1:n
        ui = labels[i]
        if haskey(interior_pos, ui)
            push!(rows, i)
            push!(cols, interior_pos[ui])
            push!(vals, T(1))
        end
    end
    return sparse(rows, cols, vals, n, n_int)
end
