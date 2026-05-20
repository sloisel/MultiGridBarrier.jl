export FEM1D, fem1d

"""
    FEM1D{T}

1D FEM (P1) discretization descriptor (zero fields).
"""
struct FEM1D{T}
end

FEM1D(::Type{T}) where {T} = FEM1D{T}()

amg_dim(::FEM1D) = 1

"""
    fem1d(::Type{T}=Float64; nodes, K=<doubled-from-nodes>) -> Geometry

Construct a **single-level** 1D FEM (P1) `Geometry` on the doubled-per-element fine mesh.
Attach a multigrid hierarchy with `amg(geom)` (algebraic multigrid on the continuous P1
stiffness). The legacy `geometric_mg(geom, L)` builds a geometric-subdivision hierarchy
on `[-1, 1]` instead.

# Arguments
- `nodes::Vector{T}`: strictly increasing fine-mesh vertices; seeds the default `K`.
- `K::Array{T,3}` shape `(2, n_e, 1)`: per-element corner tensor; `K[v, e, 1]` is the
  `v`-th endpoint (1 = left, 2 = right) of element `e`. Defaults to a tensor built
  from successive `nodes` pairs.

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem1d(::Type{T}=Float64;
                         nodes::Vector{T},
                         K::Array{T,3} = reshape(T[nodes[k] for i in 1:length(nodes)-1 for k in (i, i+1)], 2, length(nodes)-1, 1),
                         rest...) where {T}
    @assert size(K, 1) == 2 "fem1d: K must have 2 vertices per element"
    @assert size(K, 3) == 1 "fem1d: K must have spatial dim 1"
    n_e   = size(K, 2)
    Kf    = _xflat(K)
    h     = [Kf[2k, 1] - Kf[2k-1, 1] for k in 1:n_e]
    n_doubled = 2 * n_e

    # Operators on fine doubled level
    id_op = sparse(one(T) * I, n_doubled, n_doubled)
    dx_op = _dx_doubled(h, T)

    # Quadrature
    x = K
    w = _doubled_weights(h)

    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => id_op,
        :dx => dx_op,
    )

    disc = FEM1D{T}()
    return Geometry{T, Array{T,3}, Vector{T}, SparseMatrixCSC{T,Int}, FEM1D{T}}(
        disc, x, w, operators)
end

# ============================================================================
# amg(::Geometry{FEM1D}) — algebraic-MG hierarchy on the fine doubled-DOF P1 mesh.
# ============================================================================

"""
    find_boundary(geom::Geometry{...,FEM1D{T}}) -> Vector{Tuple{Int,Int}}

`(v, e)` index pairs into `geom.x` for the two boundary nodes: vertex 1 of
element 1 (the left endpoint) and vertex 2 of element `n_e` (the right
endpoint).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,FEM1D{T}}) where {T}
    n_e = size(geom.x, 2)
    return [(1, 1), (2, n_e)]
end

# Map a set of broken-basis row indices (into the 2*n_e × 1 doubled mesh) to
# the set of underlying corner-vertex indices (1 .. n). Row `2k-1` is the left
# endpoint of element k (corner `k`); row `2k` is the right endpoint (corner `k+1`).
function _fem1d_broken_rows_to_corner_set(rows, n_e::Int)
    out = Set{Int}()
    for r in rows
        if iseven(r)
            push!(out, r ÷ 2 + 1)            # right endpoint of element r÷2
        else
            push!(out, (r + 1) ÷ 2)          # left endpoint of element (r+1)÷2
        end
    end
    return out
end

# Build a fem1d AMG hierarchy on the continuous-P1 stiffness restricted to
# `interior_set` (a subset of corner indices). Used twice from `amg(geom)` —
# once with the user's Dirichlet-aware interior set (for `:dirichlet`), once
# with `1:n` (for `:full`, giving the all-corners Neumann variant).
function _fem1d_p1_hierarchy(h::Vector{T}, n::Int, n_doubled::Int,
                             interior_set::AbstractVector{<:Integer},
                             max_coarse::Int) where {T}
    K_full = _assemble_p1_stiffness_full(h, T)
    K_loc  = K_full[interior_set, interior_set]
    n_loc  = length(interior_set)

    P_amg = _amg_prolongations(K_loc, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg = n_amg_steps + 1
    L = K_amg + 1

    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    for i in 1:n_amg_steps
        k = K_amg - i
        refine[k]  = P_amg[i]
        coarsen[k] = _amg_injection(P_amg[i])
    end
    refine[K_amg]  = _interior_continuous_to_doubled_p1(n, interior_set, T)
    coarsen[K_amg] = _doubled_p1_to_interior_continuous_pick(n, interior_set, T)
    refine[L]  = sparse(one(T) * I, n_doubled, n_doubled)
    coarsen[L] = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L)
    sizes[K_amg] = n_loc
    for k in K_amg-1:-1:1
        sizes[k] = size(refine[k], 2)
    end
    sizes[L] = n_doubled

    return refine, coarsen, sizes, L, K_amg
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM1D{T}};
             max_coarse::Int=2,
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom))) where {T}
    Kf  = _xflat(geom.x)
    n_e = size(geom.x, 2)
    n   = n_e + 1
    n_doubled = 2 * n_e
    h     = [Kf[2k, 1] - Kf[2k-1, 1] for k in 1:n_e]

    # :full hierarchy (all-corners P1, Neumann variant); :uniform rides it.
    refine_full, coarsen_full, sizes_full, L_full, K_amg_full =
        _fem1d_p1_hierarchy(h, n, n_doubled, collect(1:n), max_coarse)

    # One zero-trace continuous subspace per named dirichlet node set.
    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dirichlet_corners = _fem1d_broken_rows_to_corner_set(_pairs_to_linear(nodes, 2), n_e)
        interior          = sort!(collect(setdiff(1:n, dirichlet_corners)))
        refine_dir, coarsen_dir, sizes_dir, L_dir, K_amg_dir =
            _fem1d_p1_hierarchy(h, n, n_doubled, interior, max_coarse)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for k in 1:K_amg_dir
            sub[k] = sparse(one(T) * I, sizes_dir[k], sizes_dir[k])
        end
        sub[L_dir] = SparseMatrixCSC{T,Int}(refine_dir[K_amg_dir])
        return refine_dir, coarsen_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, coarsen_full, sizes_full, L_full, K_amg_full, build_dirichlet)
end

# ============================================================================
# geometric_mg(::Geometry{FEM1D}, L) — geometric subdivision hierarchy on [-1,1] / 2^L
# (replaces the user's nodes with the canonical geometric mesh).
# ============================================================================

function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,FEM1D{T}}, L::Int;
                      structured::Bool=true) where {T}
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))
    structured ? _geometric_fem1d_structured(T, L) : _geometric_fem1d_sparse(T, L)
end

function _geometric_fem1d_sparse(::Type{T}, L::Int) where {T}
    ls = [2^k for k=1:L]
    x = Array{Array{T,3},1}(undef,(L,))
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L
        n0 = 2^l
        x[l] = reshape((hcat((0:n0-1)./T(n0),(1:n0)./T(n0))' .* 2 .- 1), 2, n0, 1)
        N = 2*n0
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T,I,N,N)
        uniform[l] = sparse(ones(T,(N,1)))
    end
    N = 2*2^L
    w = repeat([T(2)/N],outer=(N,))
    id = sparse(T,I,N,N)
    dx = blockdiag(repeat([sparse(T[-2^(L-1) 2^(L-1)
                                    -2^(L-1) 2^(L-1)])],outer=(2^L,))...)
    refine[L] = id
    coarsen[L] = id
    for l=1:L-1
        n0 = 2^l
        refine[l] = blockdiag(
            repeat([sparse(T[1.0 0.0
                     0.5 0.5
                     0.5 0.5
                     0.0 1.0])],outer=(n0,))...)
        coarsen[l] = blockdiag(
            repeat([sparse(T[1 0 0 0
                     0 0 0 1])],outer=(n0,))...)
    end
    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict{Symbol,SparseMatrixCSC{T,Int}}(:id => id, :dx => dx)
    disc = FEM1D{T}()
    geom = Geometry{T,Array{T,3},Vector{T},SparseMatrixCSC{T,Int},FEM1D{T}}(
        disc, x[end], w, operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# Direct structured construction — builds block types without sparse intermediates
function _geometric_fem1d_structured(::Type{T}, L::Int) where {T}
    p = 2

    x = Array{Array{T,3},1}(undef, L)
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        n0 = 2^l
        x[l] = reshape((hcat((0:n0-1)./T(n0),(1:n0)./T(n0))' .* 2 .- 1), 2, n0, 1)
        N = 2*n0
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T, I, N, N)
        uniform[l] = sparse(ones(T, (N, 1)))
    end

    N_blocks = 2^L

    id_block = zeros(T, p, p, N_blocks)
    dx_block = zeros(T, p, p, N_blocks)
    scale = T(2^(L-1))
    for i in 1:N_blocks
        id_block[1,1,i] = one(T)
        id_block[2,2,i] = one(T)
        dx_block[1,1,i] = -scale
        dx_block[1,2,i] =  scale
        dx_block[2,1,i] = -scale
        dx_block[2,2,i] =  scale
    end
    id = BlockDiag(id_block)
    dx = BlockDiag(dx_block)

    w = fill(T(2) / T(2 * N_blocks), 2 * N_blocks)

    ref_sub1 = T[1 0; 0.5 0.5]
    ref_sub2 = T[0.5 0.5; 0 1]
    coar_sub1 = T[1 0; 0 0]
    coar_sub2 = T[0 0; 0 1]

    n0_1 = 2^1
    ref_data_1 = zeros(T, p, p, 2 * n0_1)
    coar_data_1 = zeros(T, p, p, 2 * n0_1)
    for i in 1:n0_1
        ref_data_1[:, :, 2*(i-1)+1] = ref_sub1
        ref_data_1[:, :, 2*(i-1)+2] = ref_sub2
        coar_data_1[:, :, 2*(i-1)+1] = coar_sub1
        coar_data_1[:, :, 2*(i-1)+2] = coar_sub2
    end
    ref1 = VBlockDiag(p, p, 2, n0_1, ref_data_1)
    coar1 = HBlockDiag(p, p, 2, n0_1, coar_data_1)

    refine = Vector{typeof(ref1)}(undef, L)
    coarsen = Vector{typeof(coar1)}(undef, L)
    refine[1] = ref1
    coarsen[1] = coar1

    for l in 2:L-1
        n0 = 2^l
        ref_data = zeros(T, p, p, 2 * n0)
        coar_data = zeros(T, p, p, 2 * n0)
        for i in 1:n0
            ref_data[:, :, 2*(i-1)+1] = ref_sub1
            ref_data[:, :, 2*(i-1)+2] = ref_sub2
            coar_data[:, :, 2*(i-1)+1] = coar_sub1
            coar_data[:, :, 2*(i-1)+2] = coar_sub2
        end
        refine[l] = VBlockDiag(p, p, 2, n0, ref_data)
        coarsen[l] = HBlockDiag(p, p, 2, n0, coar_data)
    end

    id_ref_data = zeros(T, p, p, N_blocks)
    for i in 1:N_blocks
        id_ref_data[1,1,i] = one(T)
        id_ref_data[2,2,i] = one(T)
    end
    refine[L] = VBlockDiag(p, p, 1, N_blocks, id_ref_data)
    coarsen[L] = HBlockDiag(p, p, 1, N_blocks, copy(id_ref_data))

    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict{Symbol, BlockDiag{T,Array{T,3}}}(:id => id, :dx => dx)
    disc = FEM1D{T}()
    geom = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, FEM1D{T}}(
        disc, x[end], w, operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# ============================================================================
# Interpolation and plotting (Geometry-level).
# ============================================================================

function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::T) where{T}
    b = length(x)
    if t<x[1]
        return y[1]
    elseif t>x[b]
        return y[b]
    end
    a = 1
    while b-a>1
        c = (a+b)÷2
        if x[c]<=t
            a=c
        else
            b=c
        end
    end
    w = (t-x[a])/(x[b]-x[a])
    return w*y[b]+(1-w)*y[a]
end

function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}
    [fem1d_interp(x,y,t[k]) for k=1:length(t)]
end

interpolate(M::Geometry{T,Array{T,3},Vector{T},<:Any,FEM1D{T}}, z::Vector{T}, t) where {T} =
    fem1d_interp(vec(M.x),z,t)

plot(M::Geometry{T,Array{T,3},Vector{T},<:Any,FEM1D{T}}, z::Vector{T}; kwargs...) where {T} =
    plot(_xflat(M.x),z; kwargs...)

# ============================================================================
# Helpers (AMG-on-corners)
# ============================================================================

# Continuous P1 Dirichlet stiffness on interior nodes (default boundary: 1 and n).
function _assemble_dirichlet_stiffness(h::Vector{T}) where {T}
    n_e   = length(h)
    n_int = n_e - 1
    if n_int == 0
        return spzeros(T, 0, 0)
    end
    if n_int == 1
        return sparse(reshape([T(1)/h[1] + T(1)/h[2]], 1, 1))
    end
    d = T[T(1)/h[k] + T(1)/h[k+1] for k in 1:n_int]
    e = T[-T(1)/h[k+1]            for k in 1:n_int-1]
    return sparse(SymTridiagonal(d, e))
end

# Full n x n continuous P1 stiffness; subset by `interior` to get the constrained system.
function _assemble_p1_stiffness_full(h::Vector{T}, ::Type{T_out}=T) where {T,T_out}
    n_e = length(h)
    n   = n_e + 1
    rows = Int[]; cols = Int[]; vals = T_out[]
    sizehint!(rows, 4*n_e); sizehint!(cols, 4*n_e); sizehint!(vals, 4*n_e)
    @inbounds for e in 1:n_e
        invh = T_out(1) / T_out(h[e])
        push!(rows, e);   push!(cols, e);   push!(vals,  invh)
        push!(rows, e);   push!(cols, e+1); push!(vals, -invh)
        push!(rows, e+1); push!(cols, e);   push!(vals, -invh)
        push!(rows, e+1); push!(cols, e+1); push!(vals,  invh)
    end
    return sparse(rows, cols, vals, n, n)
end

# RS-AMG on K_int. Returns prolongations P[1] (finest)...P[end] (coarsest interior step).
function _amg_prolongations(K_int::SparseMatrixCSC{T,Int}, ::Type{T_out};
                             max_coarse::Int=2) where {T, T_out}
    if size(K_int, 1) == 0
        return SparseMatrixCSC{T_out,Int}[]
    end
    K64 = SparseMatrixCSC{Float64,Int}(K_int)
    ml = AlgebraicMultigrid.ruge_stuben(K64; max_coarse=max_coarse)
    return [SparseMatrixCSC{T_out,Int}(ml.levels[i].P) for i in 1:length(ml.levels)]
end

# Build a sparse C-point-injection restriction R such that R * P = I exactly.
function _amg_injection(P::SparseMatrixCSC{T,Int}) where {T}
    n_fine, n_coarse = size(P)
    c_inds = Vector{Int}(undef, n_coarse)
    found  = falses(n_coarse)
    rows   = rowvals(P)
    vals   = nonzeros(P)
    nz_per_row = zeros(Int, n_fine)
    @inbounds for j in 1:n_coarse
        for k in nzrange(P, j)
            nz_per_row[rows[k]] += 1
        end
    end
    @inbounds for j in 1:n_coarse
        for k in nzrange(P, j)
            i = rows[k]
            if nz_per_row[i] == 1 && isapprox(vals[k], one(T); atol=128*eps(real(T)))
                if !found[j]
                    c_inds[j] = i
                    found[j]  = true
                    break
                end
            end
        end
        found[j] || error("could not identify C-point for coarse DOF $j " *
                          "(P column has no unit-vector row); P may not be RS classical")
    end
    return sparse(1:n_coarse, c_inds, ones(T, n_coarse), n_coarse, n_fine)
end

# Direct bridge: interior continuous P1 (dim length(interior)) -> doubled P1 (dim 2*(n-1)).
# Per element i (endpoints = nodes i, i+1): emit a 1 at the left/right doubled
# DOF iff that endpoint is in the `interior` set.
function _interior_continuous_to_doubled_p1(n::Int, interior::AbstractVector{<:Integer},
                                            ::Type{T}) where {T}
    n_e   = n - 1
    n_int = length(interior)
    # pos[c] = column index of node c in `interior`, or 0 if c is on the Dirichlet boundary.
    pos = zeros(Int, n)
    for (j, c) in enumerate(interior)
        pos[c] = j
    end
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, 2*n_e); sizehint!(cols, 2*n_e); sizehint!(vals, 2*n_e)
    @inbounds for i in 1:n_e
        # left endpoint of element i = node i
        if pos[i] != 0
            push!(rows, 2i-1); push!(cols, pos[i]); push!(vals, T(1))
        end
        # right endpoint of element i = node i+1
        if pos[i+1] != 0
            push!(rows, 2i); push!(cols, pos[i+1]); push!(vals, T(1))
        end
    end
    return sparse(rows, cols, vals, 2*n_e, n_int)
end

# For each interior node c, pick one representative doubled DOF that lives on c.
# Node c is the left endpoint of element c (DOF 2c-1) when c <= n_e, and the right
# endpoint of element c-1 (DOF 2(c-1) = 2n_e) when c = n.
function _doubled_p1_to_interior_continuous_pick(n::Int, interior::AbstractVector{<:Integer},
                                                 ::Type{T}) where {T}
    n_e   = n - 1
    n_int = length(interior)
    cols  = Vector{Int}(undef, n_int)
    @inbounds for (j, c) in enumerate(interior)
        cols[j] = c <= n_e ? 2c - 1 : 2*n_e
    end
    return sparse(1:n_int, cols, ones(T, n_int), n_int, 2*n_e)
end

# Doubled Dirichlet subspace (continuity + zero boundary on doubled basis).
function _doubled_dirichlet_subspace(n_e::Int, ::Type{T}) where {T}
    n_int = n_e - 1
    if n_int <= 0
        return spzeros(T, 2*n_e, 0)
    end
    rows = Vector{Int}(undef, 2*n_int)
    cols = Vector{Int}(undef, 2*n_int)
    @inbounds for k in 2:n_e
        j = k - 1
        rows[2j-1] = 2*(k-1); cols[2j-1] = j
        rows[2j]   = 2*k - 1; cols[2j]   = j
    end
    return sparse(rows, cols, ones(T, 2*n_int), 2*n_e, n_int)
end

# Block-diagonal dx with per-element 2×2 blocks (1/h_i) * [-1 1; -1 1].
function _dx_doubled(h::Vector{T}, ::Type{T_out}) where {T, T_out}
    n_e = length(h)
    rows = Vector{Int}(undef, 4*n_e)
    cols = Vector{Int}(undef, 4*n_e)
    vals = Vector{T_out}(undef, 4*n_e)
    @inbounds for i in 1:n_e
        s = T_out(1) / T_out(h[i])
        base = 4*(i-1)
        rows[base+1] = 2i-1; cols[base+1] = 2i-1; vals[base+1] = -s
        rows[base+2] = 2i-1; cols[base+2] = 2i;   vals[base+2] =  s
        rows[base+3] = 2i;   cols[base+3] = 2i-1; vals[base+3] = -s
        rows[base+4] = 2i;   cols[base+4] = 2i;   vals[base+4] =  s
    end
    return sparse(rows, cols, vals, 2*n_e, 2*n_e)
end

# Per-DOF trapezoidal weights = h_i / 2 for the two doubled DOFs of element i.
function _doubled_weights(h::Vector{T}) where {T}
    n_e = length(h)
    w   = Vector{T}(undef, 2*n_e)
    @inbounds for i in 1:n_e
        w[2i-1] = h[i] / 2
        w[2i]   = h[i] / 2
    end
    return w
end
