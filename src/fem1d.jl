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
Attach a multigrid hierarchy with either `amg(geom)` (algebraic multigrid on the continuous
P1 stiffness) or `geometric_mg(geom, L)` (uniform geometric subdivision on `[-1, 1]`).

# Arguments
- `nodes::Vector{T}`: strictly increasing fine-mesh vertices; seeds the default `K`.
- `K::Matrix{T}` (`2n_e × 1`): doubled per-element corner matrix; two rows per element
  giving each element's left and right endpoints. Defaults to the reshape of `nodes`.

The Geometry is intended for Dirichlet boundary conditions.
"""
function fem1d(::Type{T}=Float64;
                         nodes::Vector{T},
                         K::Matrix{T} = reshape([nodes[k] for i in 1:length(nodes)-1 for k in (i, i+1)], :, 1),
                         rest...) where {T}
    n_e   = size(K, 1) ÷ 2
    h     = [K[2k, 1] - K[2k-1, 1] for k in 1:n_e]
    n_doubled = 2 * n_e

    # Operators on fine doubled level
    id_op = sparse(one(T) * I, n_doubled, n_doubled)
    dx_op = _dx_doubled(h, T)

    # Quadrature
    x = K
    w = _doubled_weights(h)

    # Subspaces on the fine doubled level
    sub_dirichlet = _doubled_dirichlet_subspace(n_e, T)
    sub_full      = sparse(one(T) * I, n_doubled, n_doubled)
    sub_uniform   = sparse(ones(T, n_doubled, 1))

    subspaces = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :dirichlet => sub_dirichlet,
        :full      => sub_full,
        :uniform   => sub_uniform,
    )
    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => id_op,
        :dx => dx_op,
    )

    disc = FEM1D{T}()
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int}, FEM1D{T}}(
        disc, x, w, subspaces, operators)
end

# ============================================================================
# amg(::Geometry{FEM1D}) — algebraic-MG hierarchy on the fine doubled-DOF P1 mesh.
# ============================================================================

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM1D{T}};
             max_coarse::Int=2) where {T}
    K = geom.x
    n_e = size(K, 1) ÷ 2
    n   = n_e + 1
    n_int = n - 2
    h     = [K[2k, 1] - K[2k-1, 1] for k in 1:n_e]

    # 1. Continuous P1 Dirichlet stiffness on interior nodes (n_int × n_int).
    K_int = _assemble_dirichlet_stiffness(h)

    # 2. AMG hierarchy on K_int.
    P_amg = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)
    K_amg = n_amg_steps + 1
    L = K_amg + 1

    # 3. refine / coarsen.
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    for i in 1:n_amg_steps
        k = K_amg - i
        refine[k]  = P_amg[i]
        coarsen[k] = _amg_injection(P_amg[i])
    end

    refine[K_amg]  = _interior_continuous_to_doubled_p1(n, T)
    coarsen[K_amg] = _doubled_p1_to_interior_continuous_pick(n, T)

    refine[L]  = sparse(one(T) * I, 2*n_e, 2*n_e)
    coarsen[L] = sparse(one(T) * I, 2*n_e, 2*n_e)

    # 4. per-level sizes
    sizes = Vector{Int}(undef, L)
    sizes[K_amg] = n_int
    for k in K_amg-1:-1:1
        sizes[k] = size(refine[k], 2)
    end
    sizes[L] = 2*n_e

    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    for k in 1:K_amg
        sub_full[k]      = sparse(one(T) * I, sizes[k], sizes[k])
        sub_dirichlet[k] = sparse(one(T) * I, sizes[k], sizes[k])
        sub_uniform[k]   = sparse(ones(T, sizes[k], 1))
    end

    # Fine doubled level: reuse the Geometry's own subspaces.
    sub_full[L]      = SparseMatrixCSC{T,Int}(geom.subspaces[:full])
    sub_dirichlet[L] = SparseMatrixCSC{T,Int}(geom.subspaces[:dirichlet])
    sub_uniform[L]   = SparseMatrixCSC{T,Int}(geom.subspaces[:uniform])

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :full      => sub_full,
        :dirichlet => sub_dirichlet,
        :uniform   => sub_uniform,
    )
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# ============================================================================
# geometric_mg(::Geometry{FEM1D}, L) — geometric subdivision hierarchy on [-1,1] / 2^L
# (replaces the user's nodes with the canonical geometric mesh).
# ============================================================================

function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,<:Any,FEM1D{T}}, L::Int;
                      structured::Bool=true) where {T}
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))
    structured ? _geometric_fem1d_structured(T, L) : _geometric_fem1d_sparse(T, L)
end

function _geometric_fem1d_sparse(::Type{T}, L::Int) where {T}
    ls = [2^k for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))',(2*n0,1)) .* 2 .- 1
        N = size(x[l])[1]
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T,I,N,N)
        uniform[l] = sparse(ones(T,(N,1)))
    end
    N = size(x[L])[1]
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
    geom = Geometry{T,Matrix{T},Vector{T},SparseMatrixCSC{T,Int},SparseMatrixCSC{T,Int},FEM1D{T}}(
        disc, x[end], w,
        Dict{Symbol,SparseMatrixCSC{T,Int}}(:dirichlet => dirichlet[end],
                                            :full      => full[end],
                                            :uniform   => uniform[end]),
        operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

# Direct structured construction — builds block types without sparse intermediates
function _geometric_fem1d_structured(::Type{T}, L::Int) where {T}
    p = 2

    x = Array{Matrix{T},1}(undef, L)
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))', (2*n0, 1)) .* 2 .- 1
        N = size(x[l], 1)
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
    geom = Geometry{T, Matrix{T}, Vector{T}, BlockDiag{T,Array{T,3}},
                    SparseMatrixCSC{T,Int}, FEM1D{T}}(
        disc, x[end], w,
        Dict{Symbol,SparseMatrixCSC{T,Int}}(:dirichlet => dirichlet[end],
                                            :full      => full[end],
                                            :uniform   => uniform[end]),
        operators)
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

interpolate(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,FEM1D{T}}, z::Vector{T}, t) where {T} =
    fem1d_interp(reshape(M.x,(:,)),z,t)

plot(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,FEM1D{T}}, z::Vector{T}; kwargs...) where {T} =
    plot(M.x,z; kwargs...)

_default_block_size(::FEM1D) = 2

# ============================================================================
# Helpers (AMG-on-corners)
# ============================================================================

# Continuous P1 Dirichlet stiffness on interior nodes.
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

# Direct bridge: interior continuous P1 (dim n-2) -> doubled P1 (dim 2*(n-1)).
# Per element i (endpoints = nodes i, i+1): emit a 1 at the left/right doubled
# DOF iff that endpoint is an interior node (boundary endpoints contribute zero).
function _interior_continuous_to_doubled_p1(n::Int, ::Type{T}) where {T}
    n_int = n - 2
    n_e   = n - 1
    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, 2*n_e); sizehint!(cols, 2*n_e); sizehint!(vals, 2*n_e)
    @inbounds for i in 1:n_e
        # left endpoint = node i; interior iff 2 <= i <= n-1
        if 2 <= i <= n-1
            push!(rows, 2i-1); push!(cols, i-1); push!(vals, T(1))
        end
        # right endpoint = node i+1; interior iff 1 <= i <= n-2
        if 1 <= i <= n-2
            push!(rows, 2i); push!(cols, i); push!(vals, T(1))
        end
    end
    return sparse(rows, cols, vals, 2*n_e, n_int)
end

# Pick the left-of-element-(j+1) doubled DOF as the representative of interior node j+1.
function _doubled_p1_to_interior_continuous_pick(n::Int, ::Type{T}) where {T}
    n_int = n - 2
    n_e   = n - 1
    cols = Vector{Int}(undef, n_int)
    @inbounds for j in 1:n_int
        cols[j] = 2j + 1
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
