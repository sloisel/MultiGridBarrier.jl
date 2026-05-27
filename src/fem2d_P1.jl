export FEM2D_P1, fem2d_P1

"""
    FEM2D_P1{T}

Discretization tag for 2D P1 triangular FEM. Stores the user's per-element corner
tensor `K` of shape `(3, N, 2)` — `K[v, t, d]` is coordinate `d` of vertex `v` of
triangle `t`.
"""
struct FEM2D_P1{T}
    K::Array{T,3}
end

MultiGridBarrier.amg_dim(::FEM2D_P1) = 2

function plot(M::Geometry{T, Array{T,3}, Vector{T}, <:Any, FEM2D_P1{T}}, z::Vector{T}; kwargs...) where {T}
    Xf = _xflat(M.x)
    x = Xf[:,1]
    y = Xf[:,2]
    N = size(M.x, 2)
    S = reshape(0:(3*N-1), 3, N)'
    plot_trisurf(x, y, z, triangles=S; kwargs...)
end

"""
    fem2d_P1(::Type{T}=Float64; K=<default unit-square>) -> Geometry

Construct a **single-level** 2D FEM P1 `Geometry` on the doubled-per-element fine
triangulation `K`. Use `amg(geom)` to attach an algebraic-multigrid
hierarchy. (The legacy `geometric_mg(geom, L)` builds geometric-subdivision transfers
instead.)

# Arguments
- `K::Array{T,3}` (`3 × N × 2`): per-triangle corner tensor; `K[v, t, d]` is
  coordinate `d` of vertex `v` of triangle `t`.

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem2d_P1(::Type{T}=Float64;
                  K::Array{T,3} = reshape(T[-1 -1; 1 -1; -1 1; 1 -1; 1 1; -1 1], 3, 2, 2),
                  rest...) where {T}
    size(K, 1) == 3 ||
        throw(ArgumentError("K must have 3 vertices per triangle (size(K,1) = 3)"))
    size(K, 3) == 2 ||
        throw(ArgumentError("K must have spatial dim 2 (size(K,3) = 2)"))

    mg = _fem2d_P1_geometric_mg(T, K, 1)
    return mg.geometry
end

# ============================================================================
# amg(::Geometry{FEM2D_P1}) — algebraic-MG hierarchy.
# ============================================================================
"""
    find_boundary(geom::Geometry{...,FEM2D_P1{T}}) -> Vector{Tuple{Int,Int}}

`(v, t)` index pairs into `geom.x` for every vertex on `∂Ω`. A corner shared by
`k` triangles contributes its `k` pairs (one per triangle that owns it).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P1{T}}) where {T}
    x_fine = _xflat(geom.x)
    N      = size(geom.x, 2)
    _, labels = _dedupe(x_fine)
    tri_conn  = collect(transpose(reshape(labels, 3, N)))
    bdry_corner_set = Set(_find_boundary_corners(tri_conn))
    pairs = Tuple{Int,Int}[]
    for t in 1:N, v in 1:3
        labels[3*(t-1) + v] in bdry_corner_set && push!(pairs, (v, t))
    end
    return pairs
end

# Build an AMG hierarchy on the continuous-P1 stiffness restricted to
# `interior_set` (a subset of corner indices). Used twice from `amg(geom)` —
# once with the user's Dirichlet-aware interior set (for `:dirichlet`), once
# with `1:n_v` (for `:full`, giving the all-corners Neumann variant).
function _fem2d_P1_hierarchy(unique_corners::Matrix{T},
                              tri_conn::Matrix{Int},
                              K_full::SparseMatrixCSC{T,Int},
                              interior_set::AbstractVector{<:Integer},
                              n_v::Int, n_doubled::Int,
                              prolongator) where {T}
    K_loc  = K_full[interior_set, interior_set]
    n_loc  = length(interior_set)

    P_amg       = _amg_prolongations(K_loc, T, prolongator)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1
    L_total     = K_amg + 1

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    for i in 1:n_amg_steps
        kk = K_amg - i
        refine[kk]  = P_amg[i]
    end
    refine[K_amg]  = _interior_corners_to_doubled_p1(tri_conn, n_v, interior_set, T)
    refine[L_total]  = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_loc
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[L_total] = n_doubled

    return refine, sizes, L_total, K_amg
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P1{T}};
             prolongator = amg_ruge_stuben(max_coarse=2),
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom))) where {T}
    x_fine    = _xflat(geom.x)
    N         = size(geom.x, 2)
    n_doubled = 3 * N

    unique_corners, labels = _dedupe(x_fine)
    n_v = size(unique_corners, 1)
    tri_conn = collect(transpose(reshape(labels, 3, N)))

    K_full = _assemble_p1_stiffness_full(unique_corners, tri_conn)

    # :full hierarchy (all-corners Neumann); :uniform rides it.
    refine_full, sizes_full, L_full, K_amg_full =
        _fem2d_P1_hierarchy(unique_corners, tri_conn, K_full,
                             collect(1:n_v), n_v, n_doubled, prolongator)

    # One zero-trace continuous subspace per named dirichlet node set.
    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dirichlet_corner_set = Set(labels[r] for r in _pairs_to_linear(nodes, 3))
        interior_corners     = sort!(collect(setdiff(1:n_v, dirichlet_corner_set)))
        refine_dir, sizes_dir, L_dir, K_amg_dir =
            _fem2d_P1_hierarchy(unique_corners, tri_conn, K_full,
                                 interior_corners, n_v, n_doubled, prolongator)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for kk in 1:K_amg_dir
            sub[kk] = sparse(one(T) * I, sizes_dir[kk], sizes_dir[kk])
        end
        sub[L_dir] = SparseMatrixCSC{T,Int}(refine_dir[K_amg_dir])
        return refine_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, sizes_full, L_full, K_amg_full, build_dirichlet)
end

# ============================================================================
# geometric_mg(::Geometry{FEM2D_P1}, L)
# ============================================================================
function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM2D_P1{T}}, L::Int) where {T}
    # Start from the (coarse) K stored on the discretization tag.
    _fem2d_P1_geometric_mg(T, geom.discretization.K, L)
end

# Internal: build geometric L-level multigrid for a P1 triangulation `K`.
function _fem2d_P1_geometric_mg(::Type{T}, K::Array{T,3}, L::Int) where {T}
    size(K, 1) == 3 ||
        throw(ArgumentError("K must have 3 vertices per triangle (size(K,1) = 3)"))
    size(K, 3) == 2 ||
        throw(ArgumentError("K must have spatial dim 2 (size(K,3) = 2)"))
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))

    nn   = size(K, 2)
    Kf   = _xflat(K)               # (3*nn, 2) flat view used by the sparse matmuls

    R_K       = sparse(T(1) * I, 3, 3)
    R_refine  = _p1_reference_refine(T)

    x       = Vector{Matrix{T}}(undef, L)   # internal flat coordinates per level
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    x[1] = blockdiag([R_K for _ in 1:nn]...) * Kf

    for l in 1:L-1
        n_tri      = nn * 4^(l - 1)
        refine[l]  = blockdiag([R_refine  for _ in 1:n_tri]...)
        x[l + 1]   = refine[l] * x[l]
    end

    n_doubled = size(x[L], 1)
    N_fine    = n_doubled ÷ 3

    id    = sparse(one(T) * I, n_doubled, n_doubled)
    dx_op, dy_op, w = _p1_assemble_operators(x[L], N_fine, T)

    refine[L]  = id

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
    # P1 broken operators are element-block-diagonal (3 DOFs / triangle); store
    # them as BlockDiag so the Hessian assembly runs as batched dense GEMM.
    operators = Dict{Symbol, BlockDiag{T,Array{T,3}}}(
        :id => _extract_block_diag(id, 3),
        :dx => _extract_block_diag(dx_op, 3),
        :dy => _extract_block_diag(dy_op, 3),
    )

    disc   = FEM2D_P1{T}(K)
    x_fine = reshape(x[L], 3, N_fine, 2)   # store the fine mesh as a 3-tensor
    geom = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, FEM2D_P1{T}}(
        disc, x_fine, w, operators)
    return MultiGrid(geom, subspaces, refine)
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
