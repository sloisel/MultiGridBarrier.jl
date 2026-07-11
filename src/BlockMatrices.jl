# BlockMatrices.jl -- Generic structured block matrix types for CPU Hessian assembly
#
# These types mirror the CUDA extension's block types but are parametric over the
# underlying array storage type (AbstractArray{T,3}, AbstractVector{T}, etc.),
# so that CPU FEM benefits from structured assembly using Array{T,3}.

# ============================================================================
# Type Definitions — Parametric over Storage
# ============================================================================

"""
    BlockDiag{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}

Block-diagonal matrix stored as a 3D array (p × q × N), where each slice
`data[:,:,i]` is the i-th dense block. The full matrix is (p*N) × (q*N).
"""
struct BlockDiag{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}
    p::Int
    q::Int
    N::Int
    data::A3  # p × q × N
end

function BlockDiag(data::A3) where {T, A3<:AbstractArray{T,3}}
    p, q, N = size(data)
    BlockDiag{T, A3}(p, q, N, data)
end

Base.size(A::BlockDiag) = (A.p * A.N, A.q * A.N)


"""
    BlockColumn{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}

Represents D[L,k] = hcat(Z, ..., Op, ..., Z) where Op is a BlockDiag at
position `active_col` among `nu` column-blocks.
"""
struct BlockColumn{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}
    active_block::BlockDiag{T, A3}
    active_col::Int
    nu::Int
    col_sizes::Vector{Int}
    total_rows::Int
end

Base.size(A::BlockColumn) = (A.total_rows, sum(A.col_sizes))


"""
    BlockHessian{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}

nu×nu grid of BlockDiag blocks representing the accumulated Hessian
before restriction by R.
"""
struct BlockHessian{T, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}
    blocks::Matrix{Union{BlockDiag{T, A3}, Nothing}}
    p::Int
    N::Int
    block_sizes::Vector{Int}
end

Base.size(H::BlockHessian) = (sum(H.block_sizes), sum(H.block_sizes))

# AbstractMatrix interface: O(1) scalar getindex so the block types display at
# the REPL and convert (`Matrix`, `sparse`, `norm`, ...) like any matrix. The
# solver never goes through these — the hot operations all hit the dispatch
# chain below. (For GPU-backed data this is scalar indexing, as for any CuArray.)
function Base.getindex(A::BlockDiag{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    bi, r = divrem(i - 1, A.p)
    bj, c = divrem(j - 1, A.q)
    return bi == bj ? A.data[r+1, c+1, bi+1] : zero(T)
end

function Base.getindex(A::BlockColumn{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    lo = sum(A.col_sizes[1:A.active_col-1])
    hi = lo + A.col_sizes[A.active_col]
    return lo < j <= hi ? A.active_block[i, j-lo] : zero(T)
end

function Base.getindex(H::BlockHessian{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(H, i, j)
    bi = 1; ii = Int(i)
    while ii > H.block_sizes[bi]
        ii -= H.block_sizes[bi]; bi += 1
    end
    bj = 1; jj = Int(j)
    while jj > H.block_sizes[bj]
        jj -= H.block_sizes[bj]; bj += 1
    end
    blk = H.blocks[bi, bj]
    return blk === nothing ? zero(T) : blk[ii, jj]
end


# Multigrid refine/coarsen transfers built by the structured geometric_mg builders.
# They are composed into the (sparse) prolongations R at MultiGrid construction, so
# they are emitted directly as SparseMatrixCSC — `data` is the p×q×(M*K) block array.
#
# `_vblock_sparse` (refine-like): M outer blocks of (K*p)×q.
function _vblock_sparse(p::Int, q::Int, K::Int, M::Int, data::AbstractArray{T,3}) where {T}
    I_idx = Int[]; J_idx = Int[]; V_val = T[]
    for i in 1:M, j in 1:K
        sub_idx = (i - 1) * K + j
        for c in 1:q, r in 1:p
            v = data[r, c, sub_idx]
            if v != zero(T)
                push!(I_idx, (i - 1) * K * p + (j - 1) * p + r)
                push!(J_idx, (i - 1) * q + c)
                push!(V_val, v)
            end
        end
    end
    sparse(I_idx, J_idx, V_val, K * p * M, q * M)
end


"""
    ScaledAdjBlockCol{T, A3, V1}

Lazy type representing `adjoint(D[j]::BlockColumn) * Diagonal(v)`.
"""
struct ScaledAdjBlockCol{T, A3<:AbstractArray{T,3}, V1<:AbstractVector{T}}
    op::BlockColumn{T, A3}
    diag::V1
end

"""
    LazyBlockHessianProduct{T, A3, MatR}

Lazy type representing `R' * H` where R is a sparse matrix and H is BlockHessian.
"""
struct LazyBlockHessianProduct{T, A3<:AbstractArray{T,3}, MatR}
    R::MatR
    H::BlockHessian{T, A3}
end

# ============================================================================
# Overridable Kernel Functions
# ============================================================================

"""
    block_batched_gemm!(C, A, B, ::Val{transA})

Generic batched GEMM for AbstractArray{T,3}. Default CPU loop implementation.
"""
function block_batched_gemm!(C::AbstractArray{T,3}, A::AbstractArray{T,3},
                              B::AbstractArray{T,3}, ::Val{transA}) where {T, transA}
    N = size(C, 3)
    for i in 1:N
        Ci = @view C[:,:,i]
        Bi = @view B[:,:,i]
        if transA
            Ai = @view A[:,:,i]
            mul!(Ci, Ai', Bi)
        else
            Ai = @view A[:,:,i]
            mul!(Ci, Ai, Bi)
        end
    end
    C
end

"""
    block_fused_triple!(C, A, v, B, p)

Generic fused triple product: C[:,:,i] = A[:,:,i]' * diag(v_block_i) * B[:,:,i].
"""
function block_fused_triple!(C::AbstractArray{T,3}, A::AbstractArray{T,3},
                              v::AbstractVector{T}, B::AbstractArray{T,3}, p::Int) where T
    N = size(C, 3)
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    for blk in 1:N
        v_offset = (blk - 1) * p
        for j in 1:cols_C
            for i in 1:rows_C
                s = zero(T)
                for k in 1:p
                    @inbounds s += A[k, i, blk] * v[v_offset + k] * B[k, j, blk]
                end
                @inbounds C[i, j, blk] = s
            end
        end
    end
    C
end


"""
    block_alloc(::Type{T}, A::AbstractArray, dims...)

Allocate an array similar to A with element type T and given dimensions.
"""
block_alloc(::Type{T}, A::AbstractArray, dims...) where T = similar(A, T, dims...)

# ============================================================================
# Block-level operations
# ============================================================================


function block_triple_product(A::BlockDiag{T,A3}, v::AbstractVector{T}, B::BlockDiag{T,A3}) where {T, A3}
    @assert A.N == B.N
    @assert A.p == B.p
    @assert length(v) == A.p * A.N
    rows_C = A.q
    cols_C = B.q
    C_data = block_alloc(T, A.data, rows_C, cols_C, A.N)
    block_fused_triple!(C_data, A.data, v, B.data, A.p)
    BlockDiag{T, typeof(C_data)}(rows_C, cols_C, A.N, C_data)
end

# ============================================================================
# Dispatch chain for f2
# ============================================================================

# (1) mgb_diag(D[1]::BlockColumn, v) → Diagonal(v)
mgb_diag(::BlockColumn{T}, z::AbstractVector{T}, m=length(z), n=length(z)) where {T} =
    Diagonal(z)

# (2a) adjoint(D[j]::BlockColumn) * Diagonal(v) → ScaledAdjBlockCol
function Base.:*(A::Adjoint{T,BlockColumn{T,A3}}, D::Diagonal{T}) where {T, A3}
    ScaledAdjBlockCol{T, A3, typeof(D.diag)}(parent(A), D.diag)
end

# (2b) ScaledAdjBlockCol * D[k]::BlockColumn → BlockHessian
function Base.:*(A::ScaledAdjBlockCol{T,A3}, B::BlockColumn{T,A3}) where {T, A3}
    op_j = A.op
    op_k = B
    v = A.diag

    @assert op_j.nu == op_k.nu
    @assert op_j.active_block.N == op_k.active_block.N
    @assert op_j.active_block.p == op_k.active_block.p

    result_block = block_triple_product(op_j.active_block, v, op_k.active_block)

    nu = op_j.nu
    blocks = Matrix{Union{BlockDiag{T, A3}, Nothing}}(nothing, nu, nu)
    blocks[op_j.active_col, op_k.active_col] = result_block

    BlockHessian{T, A3}(blocks, result_block.p, result_block.N, op_j.col_sizes)
end

# (3) Hessian-term accumulation. `_hess_add!(a, b)` merges `b` into `a` and
# returns the result. For BlockHessian pairs this is IN PLACE: it mutates `a`'s
# blocks and aliases `b`'s blocks into the result, so both operands must be
# freshly-owned temporaries — exactly what the accumulation loop in `barrier`'s
# f2 produces. It is deliberately NOT spelled `Base.:+`, which callers may
# reasonably assume is pure (a former `+` method here corrupted its left
# operand when reused). The generic fallback covers the dense/sparse
# (spectral) operator path out of place.
_hess_add!(a, b) = a + b
function _hess_add!(A::BlockHessian{T,A3}, B::BlockHessian{T,A3}) where {T, A3}
    size(A.blocks) == size(B.blocks) && A.p == B.p && A.N == B.N ||
        throw(DimensionMismatch("_hess_add!: incompatible BlockHessian layouts"))
    nu = size(A.blocks, 1)
    blocks = Matrix{Union{BlockDiag{T, A3}, Nothing}}(nothing, nu, nu)
    for j in 1:nu, k in 1:nu
        a = A.blocks[j, k]
        b = B.blocks[j, k]
        if a === nothing && b === nothing
            # leave as nothing
        elseif a === nothing
            blocks[j, k] = b
        elseif b === nothing
            blocks[j, k] = a
        else
            a.data .+= b.data
            blocks[j, k] = a
        end
    end
    BlockHessian{T, A3}(blocks, A.p, A.N, A.block_sizes)
end

# ============================================================================
# Assembly Plan for R' * BlockHessian * R
# ============================================================================

struct BlockAssemblyPlan{T}
    # Output CSC sparsity
    out_colptr::Vector{Int}
    out_rowval::Vector{Int}
    out_m::Int
    out_n::Int
    out_nnz::Int

    # Per block k: dense R panels (p × c_max_k × N)
    panels::Vector{Array{T, 3}}

    # Per block k: column indices (c_max_k × N)
    col_indices::Vector{Matrix{Int}}

    # Per block k: actual column count per element
    c_counts::Vector{Vector{Int}}

    # Per block pair (i,j): scatter map (c_max_i × c_max_j × N)
    scatter_idx::Dict{Tuple{Int,Int}, Array{Int, 3}}

    # Block partitioning
    p::Int
    N::Int
    nu::Int
    c_max::Vector{Int}
end

# Plans are cached per R by identity: the IdDict holds R alive, so its heap slot
# (hence objectid) cannot be recycled while the entry exists. A previous scheme
# keyed a plain Dict on hash((objectid(R), H metadata...)). objectid of a
# mutable struct is address-derived, so once an entry outlived its R — e.g. a
# solve that threw before mgb_solve's cleanup ran — a later R allocated on the
# recycled objectid with matching metadata would silently receive the dead R's
# plan: a wrong Hessian, since the plan bakes in R's values. Using a raw UInt64
# hash as the key was also unguarded against hash collisions. The value is a
# short list of (metadata, plan) pairs compared with ==, because one R may be
# paired with more than one Hessian block pattern.
# (Kept structurally identical to _assembly_plan_cache in
# ext/MultiGridBarrierCUDAExt/block_ops.jl — change both together.)
const _block_assembly_plan_cache = IdDict{Any, Vector{Tuple{Any, Any}}}()

function _make_block_assembly_plan(R::SparseMatrixCSC{T,Int}, H::BlockHessian{T}) where T
    nu = size(H.blocks, 1)
    p = H.p
    N = H.N
    block_sizes = H.block_sizes
    nrows_R = size(R, 1)
    ncols_R = size(R, 2)

    row_offset = zeros(Int, nu)
    for k in 2:nu
        row_offset[k] = row_offset[k-1] + block_sizes[k-1]
    end

    # Step 1: Extract column indices per block per element
    col_indices_cpu = Vector{Matrix{Int}}(undef, nu)
    c_counts_cpu = Vector{Vector{Int}}(undef, nu)
    c_max = zeros(Int, nu)

    # Build transpose for fast row access
    Rt = sparse(R')  # Rt is CSC of R', so Rt.colptr gives R's row access

    row_offset[nu] + N * p <= nrows_R || throw(DimensionMismatch(
        "R has $nrows_R rows but the BlockHessian block layout spans $(row_offset[nu] + N * p)"))

    for k in 1:nu
        element_cols = [Int[] for _ in 1:N]
        for e in 1:N
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                for idx in Rt.colptr[global_row]:(Rt.colptr[global_row+1]-1)
                    push!(element_cols[e], Rt.rowval[idx])
                end
            end
        end
        for e in 1:N
            element_cols[e] = sort!(unique!(element_cols[e]))
        end
        c_max[k] = maximum(length(ec) for ec in element_cols; init=0)
        if c_max[k] == 0
            c_max[k] = 1  # avoid zero-size arrays
        end

        ci = zeros(Int, c_max[k], N)
        cc = zeros(Int, N)
        for e in 1:N
            cc[e] = length(element_cols[e])
            for a in 1:length(element_cols[e])
                ci[a, e] = element_cols[e][a]
            end
        end
        col_indices_cpu[k] = ci
        c_counts_cpu[k] = cc
    end

    # Step 2: Extract dense R panels per block (using Rt built above)
    panels_cpu = Vector{Array{T, 3}}(undef, nu)
    for k in 1:nu
        panel = zeros(T, p, c_max[k], N)
        ci = col_indices_cpu[k]
        for e in 1:N
            nc = c_counts_cpu[k][e]
            col_to_local = Dict{Int, Int}()
            for a in 1:nc
                col_to_local[ci[a, e]] = a
            end
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                # Iterate over columns in row global_row of R using Rt
                for idx in Rt.colptr[global_row]:(Rt.colptr[global_row+1]-1)
                    col = Rt.rowval[idx]
                    val = Rt.nzval[idx]
                    if haskey(col_to_local, col)
                        a = col_to_local[col]
                        panel[r, a, e] = val
                    end
                end
            end
        end
        panels_cpu[k] = panel
    end

    # Step 3: Build output CSC sparsity pattern
    out_rows = Int[]
    out_cols = Int[]
    for bi in 1:nu, bj in 1:nu
        if H.blocks[bi, bj] === nothing
            continue
        end
        ci_i = col_indices_cpu[bi]
        ci_j = col_indices_cpu[bj]
        cc_i = c_counts_cpu[bi]
        cc_j = c_counts_cpu[bj]
        for e in 1:N
            for a in 1:cc_i[e]
                for b in 1:cc_j[e]
                    push!(out_rows, ci_i[a, e])
                    push!(out_cols, ci_j[b, e])
                end
            end
        end
    end

    if isempty(out_rows)
        out_nnz = 0
        out_colptr = ones(Int, ncols_R + 1)
        out_rowval = Int[]
        scatter_idx = Dict{Tuple{Int,Int}, Array{Int, 3}}()
        return BlockAssemblyPlan{T}(
            out_colptr, out_rowval, ncols_R, ncols_R, out_nnz,
            panels_cpu, col_indices_cpu, c_counts_cpu,
            scatter_idx, p, N, nu, c_max)
    end

    # Build sparse pattern (deduplicate)
    indicator = sparse(out_rows, out_cols, ones(Float32, length(out_rows)), ncols_R, ncols_R)
    # indicator is CSC: colptr, rowval give us the output pattern
    out_colptr = Vector{Int}(indicator.colptr)
    out_rowval = Vector{Int}(indicator.rowval)
    out_nnz = length(out_rowval)

    # Step 4: Build scatter maps
    scatter_idx = Dict{Tuple{Int,Int}, Array{Int, 3}}()
    for bi in 1:nu, bj in 1:nu
        if H.blocks[bi, bj] === nothing
            continue
        end
        scatter = zeros(Int, c_max[bi], c_max[bj], N)
        ci_i = col_indices_cpu[bi]
        ci_j = col_indices_cpu[bj]
        cc_i = c_counts_cpu[bi]
        cc_j = c_counts_cpu[bj]
        for e in 1:N
            for b in 1:cc_j[e]
                col = ci_j[b, e]
                # Get the range in CSC for this column
                rs = out_colptr[col]
                re = out_colptr[col + 1] - 1
                col_rows = @view out_rowval[rs:re]
                for a in 1:cc_i[e]
                    row = ci_i[a, e]
                    # Binary search for row in col_rows
                    lo, hi = 1, length(col_rows)
                    while lo <= hi
                        mid = (lo + hi) ÷ 2
                        if col_rows[mid] < row
                            lo = mid + 1
                        elseif col_rows[mid] > row
                            hi = mid - 1
                        else
                            scatter[a, b, e] = rs + mid - 1
                            break
                        end
                    end
                end
            end
        end
        scatter_idx[(bi, bj)] = scatter
    end

    BlockAssemblyPlan{T}(
        out_colptr, out_rowval, ncols_R, ncols_R, out_nnz,
        panels_cpu, col_indices_cpu, c_counts_cpu,
        scatter_idx, p, N, nu, c_max)
end

function _get_block_assembly_plan(R::SparseMatrixCSC{T,Int}, H::BlockHessian{T}) where T
    nu = size(H.blocks, 1)
    meta = (H.p, H.N, H.block_sizes,
            [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing])
    entries = get!(Vector{Tuple{Any, Any}}, _block_assembly_plan_cache, R)
    for (m, plan) in entries
        m == meta && return plan::BlockAssemblyPlan{T}
    end
    plan = _make_block_assembly_plan(R, H)
    push!(entries, (meta, plan))
    return plan::BlockAssemblyPlan{T}
end

function _assemble_RtHR(R::SparseMatrixCSC{T,Int}, H::BlockHessian{T}) where T
    plan = _get_block_assembly_plan(R, H)

    if plan.out_nnz == 0
        return spzeros(T, plan.out_m, plan.out_n)
    end

    output_nzval = zeros(T, plan.out_nnz)
    nu = plan.nu
    p = plan.p
    N = plan.N

    cm_max = maximum(plan.c_max)
    tmp = Array{T}(undef, p, cm_max, N)

    for bi in 1:nu, bj in 1:nu
        blk = H.blocks[bi, bj]
        if blk === nothing
            continue
        end

        scatter = plan.scatter_idx[(bi, bj)]
        panel_i = plan.panels[bi]
        panel_j = plan.panels[bj]
        cmi = plan.c_max[bi]
        cmj = plan.c_max[bj]

        # Phase 1: tmp[r, b, e] = sum_s H[r, s, e] * panel_j[s, b, e]
        tmp_view = @view tmp[:, 1:cmj, :]
        block_batched_gemm!(tmp_view, blk.data, (@view panel_j[:, 1:cmj, :]), Val(false))

        # Phase 2: dot + scatter
        for e in 1:N
            for b in 1:cmj
                for a in 1:cmi
                    scatter_pos = scatter[a, b, e]
                    if scatter_pos > 0
                        val = zero(T)
                        for r in 1:p
                            @inbounds val += panel_i[r, a, e] * tmp[r, b, e]
                        end
                        output_nzval[scatter_pos] += val
                    end
                end
            end
        end
    end

    SparseMatrixCSC(plan.out_m, plan.out_n, plan.out_colptr, plan.out_rowval, output_nzval)
end

# (4) R' * BlockHessian → LazyBlockHessianProduct
function Base.:*(A::Adjoint{T, SparseMatrixCSC{T,Int}}, H::BlockHessian{T,A3}) where {T, A3}
    LazyBlockHessianProduct{T, A3, SparseMatrixCSC{T,Int}}(parent(A), H)
end

# (5) LazyBlockHessianProduct * R → SparseMatrixCSC
function Base.:*(lhp::LazyBlockHessianProduct{T,A3,SparseMatrixCSC{T,Int}}, R::SparseMatrixCSC{T,Int}) where {T, A3}
    # A real check, not an @assert: the structured assembly computes R'*H*R for
    # a single R, so silently accepting R1'*H*R2 would return a wrong matrix.
    lhp.R === R || throw(ArgumentError(
        "structured Hessian assembly computes R'*H*R for one matrix R; " *
        "got different matrices on the two sides of R'*H*R"))
    _assemble_RtHR(lhp.R, lhp.H)
end


# ============================================================================
# Matrix-vector products
# ============================================================================

# BlockDiag * Vector (CPU): block-local matvec, O(p·q·N). Without this method
# the AbstractMatrix fallback runs a full dense matvec through scalar getindex —
# O((pN)·(qN)) — which is what `value(deriv(u, :op))` in the JuMP front end
# would otherwise hit on every FEM geometry. Restricted to CPU storage: the
# live GPU solve never multiplies a bare CuArray-backed BlockDiag (see
# ext/MultiGridBarrierCUDAExt/block_ops.jl).
function Base.:*(A::BlockDiag{T,Array{T,3}}, z::AbstractVector{T}) where {T}
    p, q, N = A.p, A.q, A.N
    length(z) == q * N || throw(DimensionMismatch(
        "BlockDiag has $(q * N) columns, vector has length $(length(z))"))
    data = A.data
    out = Vector{T}(undef, p * N)
    @inbounds for blk in 1:N
        zoff = (blk - 1) * q
        ooff = (blk - 1) * p
        for r in 1:p
            s = zero(T)
            for c in 1:q
                s += data[r, c, blk] * z[zoff + c]
            end
            out[ooff + r] = s
        end
    end
    out
end

# BlockColumn * Vector
function Base.:*(A::BlockColumn{T,A3}, z::Vector{T}) where {T, A3}
    col_offset = sum(A.col_sizes[1:A.active_col-1])
    p = A.active_block.p
    q = A.active_block.q
    N = A.active_block.N
    out = zeros(T, A.total_rows)
    for blk in 1:N
        for r in 1:p
            s = zero(T)
            for c in 1:q
                @inbounds s += A.active_block.data[r, c, blk] * z[col_offset + (blk - 1) * q + c]
            end
            @inbounds out[(blk - 1) * p + r] = s
        end
    end
    out
end

# adjoint(BlockColumn) * Vector
function Base.:*(A::Adjoint{T,BlockColumn{T,A3}}, z::Vector{T}) where {T, A3}
    op = parent(A)
    col_offset = sum(op.col_sizes[1:op.active_col-1])
    p = op.active_block.p
    q = op.active_block.q
    N = op.active_block.N
    out = zeros(T, sum(op.col_sizes))
    for blk in 1:N
        for c in 1:q
            s = zero(T)
            for r in 1:p
                @inbounds s += op.active_block.data[r, c, blk] * z[(blk - 1) * p + r]
            end
            @inbounds out[col_offset + (blk - 1) * q + c] = s
        end
    end
    out
end

# ============================================================================
# mgb_* dispatch methods
# ============================================================================

mgb_zeros(::BlockColumn{T}, m, n) where {T} = spzeros(T, m, n)

function mgb_zeros(A::BlockDiag{T}, m, n) where T
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    data = similar(A.data, A.p, A.p, N)
    fill!(data, zero(T))
    BlockDiag{T, typeof(data)}(A.p, A.p, N, data)
end

# ============================================================================
# BlockColumn construction (the D_fine rows of amg_helper)
# ============================================================================

# One D_fine row: the operator `op` in column-block `active` of `nu` equal-width
# column blocks (the rest structurally zero). For BlockDiag operators this
# builds the BlockColumn wrapper directly — the caller knows the active slot, so
# no value inspection is needed and the batched-GEMM structure is preserved
# unconditionally. The generic method reproduces the historical hcat-of-zeros
# for sparse/dense (spectral) operators.
function _block_column(op::AbstractMatrix, active::Int, nu::Int)
    n = size(op, 1)
    Z = mgb_zeros(op, n, n)
    foo = [Z for j = 1:nu]
    foo[active] = op
    hcat(foo...)
end
_block_column(op::BlockDiag{T,A3}, active::Int, nu::Int) where {T,A3} =
    BlockColumn{T,A3}(op, active, nu, fill(size(op, 2), nu), size(op, 1))

# `hcat` of BlockDiags has no faithful structured result (BlockColumn holds a
# single active block), and silently degrading to SparseMatrixCSC would
# forfeit the batched-GEMM Hessian path downstream. Refuse, loudly. (Nothing in
# the package calls this — amg_helper builds BlockColumns via `_block_column`.)
Base.hcat(args::BlockDiag{T}...) where {T} = throw(ArgumentError(
    "hcat of BlockDiag operators has no structured representation; convert " *
    "explicitly with `hcat(map(MultiGridBarrier.to_sparse, ops)...)` if a " *
    "sparse result is intended"))


# ============================================================================
# Conversion to SparseMatrixCSC
# ============================================================================

function to_sparse(B::BlockDiag{T}) where T
    p, q, N = B.p, B.q, B.N
    m = p * N
    n = q * N
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    for blk in 1:N
        for c in 1:q
            for r in 1:p
                v = B.data[r, c, blk]
                if v != zero(T)
                    push!(I_idx, (blk - 1) * p + r)
                    push!(J_idx, (blk - 1) * q + c)
                    push!(V_val, v)
                end
            end
        end
    end
    sparse(I_idx, J_idx, V_val, m, n)
end


# ============================================================================
# Extraction: SparseMatrixCSC → Block types
# ============================================================================

function _extract_block_diag(A::SparseMatrixCSC{T,Int}, p::Int) where T
    m, n = size(A)
    @assert m % p == 0
    N = m ÷ p
    @assert n % N == 0
    q = n ÷ N
    data = zeros(T, p, q, N)
    for blk in 1:N
        for c in 1:q
            for r in 1:p
                global_row = (blk - 1) * p + r
                global_col = (blk - 1) * q + c
                data[r, c, blk] = A[global_row, global_col]
            end
        end
    end
    BlockDiag{T, Array{T,3}}(p, q, N, data)
end

# Cleanup: clear assembly plan cache
function _block_assembly_cleanup()
    empty!(_block_assembly_plan_cache)
end

# Throw-path variant (see device.jl): no MGBSOL exists, flush unconditionally.
mgb_cleanup(::Type{CPUDevice}) = (_block_assembly_cleanup(); nothing)

# Extend mgb_cleanup for block-structured solutions
function mgb_cleanup(sol::MGBSOL{T, <:Any, <:Vector{T}}) where T
    # Check if the geometry uses block types
    if sol.geometry isa Geometry && any(v -> v isa BlockDiag, values(sol.geometry.operators))
        _block_assembly_cleanup()
    end
    sol
end
