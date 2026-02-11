# BlockMatrices.jl -- Generic structured block matrix types for CPU Hessian assembly
#
# These types mirror the CUDA extension's block types but are parametric over the
# underlying array storage type (AbstractArray{T,3}, AbstractVector{T}, etc.),
# so that CPU FEM benefits from structured assembly using Array{T,3}.
#
# The CUDA extension is left completely untouched.

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

function Base.getindex(A::BlockDiag{T}, i::Int, j::Int) where T
    bi = (i - 1) ÷ A.p + 1
    bj = (j - 1) ÷ A.q + 1
    if bi != bj
        return zero(T)
    end
    li = (i - 1) % A.p + 1
    lj = (j - 1) % A.q + 1
    return A.data[li, lj, bi]
end

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

function Base.getindex(A::BlockColumn{T}, i::Int, j::Int) where T
    col_offset = sum(A.col_sizes[1:A.active_col-1])
    col_end = col_offset + A.col_sizes[A.active_col]
    if j <= col_offset || j > col_end
        return zero(T)
    end
    return A.active_block[i, j - col_offset]
end

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

function Base.size(A::BlockHessian)
    total = sum(A.block_sizes)
    (total, total)
end

function Base.getindex(A::BlockHessian{T}, i::Int, j::Int) where T
    bi = 0
    row_offset = 0
    for k in 1:length(A.block_sizes)
        if i <= row_offset + A.block_sizes[k]
            bi = k
            break
        end
        row_offset += A.block_sizes[k]
    end
    bj = 0
    col_offset = 0
    for k in 1:length(A.block_sizes)
        if j <= col_offset + A.block_sizes[k]
            bj = k
            break
        end
        col_offset += A.block_sizes[k]
    end
    blk = A.blocks[bi, bj]
    if blk === nothing
        return zero(T)
    end
    return blk[i - row_offset, j - col_offset]
end

"""
    SubBlockDiag{T, Orient, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}

Block-diagonal matrix with sub-block structure for multigrid coarsen/refine operators.

**VBlockDiag** (Orient=Val{:V}, refine-like): M outer blocks of (K*p)×q.
**HBlockDiag** (Orient=Val{:H}, coarsen-like): M outer blocks of p×(K*q).
"""
struct SubBlockDiag{T, Orient, A3<:AbstractArray{T,3}} <: AbstractMatrix{T}
    p::Int
    q::Int
    K::Int
    M::Int
    data::A3  # p × q × (M*K)
end

const VBlockDiag{T, A3} = SubBlockDiag{T, Val{:V}, A3}
const HBlockDiag{T, A3} = SubBlockDiag{T, Val{:H}, A3}

function VBlockDiag(p::Int, q::Int, K::Int, M::Int, data::A3) where {T, A3<:AbstractArray{T,3}}
    SubBlockDiag{T, Val{:V}, A3}(p, q, K, M, data)
end

function HBlockDiag(p::Int, q::Int, K::Int, M::Int, data::A3) where {T, A3<:AbstractArray{T,3}}
    SubBlockDiag{T, Val{:H}, A3}(p, q, K, M, data)
end

Base.size(A::VBlockDiag) = (A.K * A.p * A.M, A.q * A.M)
Base.size(A::HBlockDiag) = (A.p * A.M, A.K * A.q * A.M)

function Base.getindex(A::VBlockDiag{T}, i::Int, j::Int) where T
    bj = (j - 1) ÷ A.q + 1
    lj = (j - 1) % A.q + 1
    outer_row_size = A.K * A.p
    bi = (i - 1) ÷ outer_row_size + 1
    if bi != bj
        return zero(T)
    end
    local_row = (i - 1) % outer_row_size
    sub_idx = local_row ÷ A.p
    li = local_row % A.p + 1
    global_sub = (bi - 1) * A.K + sub_idx + 1
    return A.data[li, lj, global_sub]
end

function Base.getindex(A::HBlockDiag{T}, i::Int, j::Int) where T
    bi = (i - 1) ÷ A.p + 1
    li = (i - 1) % A.p + 1
    outer_col_size = A.K * A.q
    bj = (j - 1) ÷ outer_col_size + 1
    if bi != bj
        return zero(T)
    end
    local_col = (j - 1) % outer_col_size
    sub_idx = local_col ÷ A.q
    lj = local_col % A.q + 1
    global_sub = (bj - 1) * A.K + sub_idx + 1
    return A.data[li, lj, global_sub]
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
    block_batched_gemm_broadcast_B!(C, A, B, K)

C[:,:,s] = A[:,:,s] * B[:,:,(s-1)÷K+1]. Each B sub-block is reused K times.
"""
function block_batched_gemm_broadcast_B!(C::AbstractArray{T,3}, A::AbstractArray{T,3},
                                          B::AbstractArray{T,3}, K::Int) where T
    N = size(C, 3)
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    inner = size(A, 2)
    for blk in 1:N
        b_blk = (blk - 1) ÷ K + 1
        for j in 1:cols_C
            for i in 1:rows_C
                s = zero(T)
                for k in 1:inner
                    @inbounds s += A[i, k, blk] * B[k, j, b_blk]
                end
                @inbounds C[i, j, blk] = s
            end
        end
    end
    C
end

"""
    block_batched_gemm_broadcast_A!(C, A, B, K_B)

C[:,:,s] = A[:,:,(s-1)÷K_B+1] * B[:,:,s]. Each A sub-block is reused K_B times.
"""
function block_batched_gemm_broadcast_A!(C::AbstractArray{T,3}, A::AbstractArray{T,3},
                                          B::AbstractArray{T,3}, K_B::Int) where T
    N = size(C, 3)
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    inner = size(A, 2)
    for blk in 1:N
        a_blk = (blk - 1) ÷ K_B + 1
        for j in 1:cols_C
            for i in 1:rows_C
                s = zero(T)
                for k in 1:inner
                    @inbounds s += A[i, k, a_blk] * B[k, j, blk]
                end
                @inbounds C[i, j, blk] = s
            end
        end
    end
    C
end

"""
    block_segmented_sum!(out, data, K)

out[:,:,i] = sum(data[:,:,(i-1)*K+1:i*K]) for i = 1:M.
"""
function block_segmented_sum!(out::AbstractArray{T,3}, data::AbstractArray{T,3}, K::Int) where T
    p = size(out, 1)
    q = size(out, 2)
    M = size(out, 3)
    for outer in 1:M
        for c in 1:q
            for r in 1:p
                s = zero(T)
                for k in 1:K
                    @inbounds s += data[r, c, (outer - 1) * K + k]
                end
                @inbounds out[r, c, outer] = s
            end
        end
    end
    out
end

"""
    block_alloc(::Type{T}, A::AbstractArray, dims...)

Allocate an array similar to A with element type T and given dimensions.
"""
block_alloc(::Type{T}, A::AbstractArray, dims...) where T = similar(A, T, dims...)

# ============================================================================
# Block-level operations
# ============================================================================

function block_mul(A::BlockDiag{T,A3}, B::BlockDiag{T,A3}; transA::Bool=false) where {T, A3}
    @assert A.N == B.N
    if transA
        @assert A.p == B.p
        rows_C = A.q
    else
        @assert A.q == B.p
        rows_C = A.p
    end
    cols_C = B.q
    C_data = block_alloc(T, A.data, rows_C, cols_C, A.N)
    block_batched_gemm!(C_data, A.data, B.data, Val(transA))
    BlockDiag{T, typeof(C_data)}(rows_C, cols_C, A.N, C_data)
end

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

# (1) amgb_diag(D[1]::BlockColumn, v) → Diagonal(v)
amgb_diag(::BlockColumn{T}, z::AbstractVector{T}, m=length(z), n=length(z)) where {T} =
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

# (3) BlockHessian + BlockHessian → BlockHessian
function Base.:+(A::BlockHessian{T,A3}, B::BlockHessian{T,A3}) where {T, A3}
    @assert size(A.blocks) == size(B.blocks)
    @assert A.p == B.p && A.N == B.N
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

const _block_assembly_plan_cache = Dict{UInt64, Any}()

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

    for k in 1:nu
        element_cols = [Int[] for _ in 1:N]
        for e in 1:N
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                if global_row > nrows_R
                    continue
                end
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
                if global_row > nrows_R
                    continue
                end
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
    key = hash((objectid(R), H.p, H.N, H.block_sizes,
                [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing]))
    get!(_block_assembly_plan_cache, key) do
        _make_block_assembly_plan(R, H)
    end::BlockAssemblyPlan{T}
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
    @assert lhp.R === R "LazyBlockHessianProduct expects same R on both sides"
    _assemble_RtHR(lhp.R, lhp.H)
end

# Fallback: BlockHessian * SparseMatrixCSC
function Base.:*(H::BlockHessian{T}, B::SparseMatrixCSC{T,Int}) where T
    to_sparse(H) * B
end

# ============================================================================
# Matrix-vector products
# ============================================================================

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

# BlockDiag * Vector
function Base.:*(A::BlockDiag{T}, x::Vector{T}) where T
    p, q, N = A.p, A.q, A.N
    @assert length(x) == q * N
    out = Vector{T}(undef, p * N)
    for blk in 1:N
        for r in 1:p
            s = zero(T)
            for c in 1:q
                @inbounds s += A.data[r, c, blk] * x[(blk - 1) * q + c]
            end
            @inbounds out[(blk - 1) * p + r] = s
        end
    end
    out
end

# VBlockDiag * Vector
function Base.:*(A::VBlockDiag{T}, x::Vector{T}) where T
    @assert length(x) == A.q * A.M
    MK = A.M * A.K
    total = A.p * MK
    out = Vector{T}(undef, total)
    for blk in 1:MK
        outer = (blk - 1) ÷ A.K + 1
        for r in 1:A.p
            s = zero(T)
            for c in 1:A.q
                @inbounds s += A.data[r, c, blk] * x[(outer - 1) * A.q + c]
            end
            @inbounds out[(blk - 1) * A.p + r] = s
        end
    end
    out
end

# HBlockDiag * Vector
function Base.:*(A::HBlockDiag{T}, x::Vector{T}) where T
    @assert length(x) == A.K * A.q * A.M
    total = A.p * A.M
    out = zeros(T, total)
    for outer in 1:A.M
        for j in 1:A.K
            sub_idx = (outer - 1) * A.K + j
            for r in 1:A.p
                s = zero(T)
                for c in 1:A.q
                    @inbounds s += A.data[r, c, sub_idx] * x[(sub_idx - 1) * A.q + c]
                end
                @inbounds out[(outer - 1) * A.p + r] += s
            end
        end
    end
    out
end

# adjoint(VBlockDiag) * Vector
function Base.:*(A::Adjoint{T, <:VBlockDiag{T}}, x::Vector{T}) where T
    V = parent(A)
    @assert length(x) == V.K * V.p * V.M
    total = V.q * V.M
    out = zeros(T, total)
    for outer in 1:V.M
        for j in 1:V.K
            sub_idx = (outer - 1) * V.K + j
            for c in 1:V.q
                s = zero(T)
                for r in 1:V.p
                    @inbounds s += V.data[r, c, sub_idx] * x[(sub_idx - 1) * V.p + r]
                end
                @inbounds out[(outer - 1) * V.q + c] += s
            end
        end
    end
    out
end

# adjoint(HBlockDiag) * Vector
function Base.:*(A::Adjoint{T, <:HBlockDiag{T}}, x::Vector{T}) where T
    H = parent(A)
    MK = H.M * H.K
    @assert length(x) == H.p * H.M
    total = H.q * MK
    out = Vector{T}(undef, total)
    for sub_idx in 1:MK
        outer = (sub_idx - 1) ÷ H.K + 1
        for c in 1:H.q
            s = zero(T)
            for r in 1:H.p
                @inbounds s += H.data[r, c, sub_idx] * x[(outer - 1) * H.p + r]
            end
            @inbounds out[(sub_idx - 1) * H.q + c] = s
        end
    end
    out
end

# Matrix-matrix products (column-loop)
function Base.:*(A::BlockDiag{T}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

function Base.:*(A::VBlockDiag{T}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

function Base.:*(A::HBlockDiag{T}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

function Base.:*(A::Adjoint{T, <:VBlockDiag{T}}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

function Base.:*(A::Adjoint{T, <:HBlockDiag{T}}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

# BlockColumn * Matrix
function Base.:*(A::BlockColumn{T}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

# adjoint(BlockColumn) * Matrix
function Base.:*(A::Adjoint{T, <:BlockColumn{T}}, B::Matrix{T}) where T
    ncols = size(B, 2)
    out = Matrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * Vector{T}(B[:, col])
    end
    out
end

# ============================================================================
# Matrix-Matrix products (for amg_helper)
# ============================================================================

# BlockDiag * VBlockDiag → VBlockDiag
function Base.:*(A::BlockDiag{T,A3}, B::VBlockDiag{T,A3}) where {T, A3}
    N = A.N
    @assert N == B.M * B.K
    @assert A.q == B.p
    C_data = block_alloc(T, A.data, A.p, B.q, N)
    block_batched_gemm!(C_data, A.data, B.data, Val(false))
    VBlockDiag(A.p, B.q, B.K, B.M, C_data)
end

# HBlockDiag * BlockDiag → HBlockDiag
function Base.:*(A::HBlockDiag{T,A3}, B::BlockDiag{T,A3}) where {T, A3}
    N = B.N
    @assert A.M * A.K == N
    @assert A.q == B.p
    C_data = block_alloc(T, A.data, A.p, B.q, N)
    block_batched_gemm!(C_data, A.data, B.data, Val(false))
    HBlockDiag(A.p, B.q, A.K, A.M, C_data)
end

# HBlockDiag * VBlockDiag → BlockDiag
function Base.:*(A::HBlockDiag{T,A3}, B::VBlockDiag{T,A3}) where {T, A3}
    @assert A.K == B.K && A.M == B.M
    @assert A.q == B.p
    N = A.M * A.K
    products = block_alloc(T, A.data, A.p, B.q, N)
    block_batched_gemm!(products, A.data, B.data, Val(false))
    out = block_alloc(T, A.data, A.p, B.q, A.M)
    block_segmented_sum!(out, products, A.K)
    BlockDiag{T, typeof(out)}(A.p, B.q, A.M, out)
end

# VBlockDiag * VBlockDiag → VBlockDiag
function Base.:*(A::VBlockDiag{T,A3}, B::VBlockDiag{T,A3}) where {T, A3}
    @assert A.M == B.M * B.K
    @assert A.q == B.p
    K_result = A.K * B.K
    M_result = B.M
    N_total = M_result * K_result
    C_data = block_alloc(T, A.data, A.p, B.q, N_total)
    block_batched_gemm_broadcast_B!(C_data, A.data, B.data, A.K)
    VBlockDiag(A.p, B.q, K_result, M_result, C_data)
end

# HBlockDiag * HBlockDiag → HBlockDiag
function Base.:*(A::HBlockDiag{T,A3}, B::HBlockDiag{T,A3}) where {T, A3}
    @assert B.M == A.M * A.K
    @assert A.q == B.p
    K_result = A.K * B.K
    M_result = A.M
    N_total = M_result * K_result
    C_data = block_alloc(T, A.data, A.p, B.q, N_total)
    block_batched_gemm_broadcast_A!(C_data, A.data, B.data, B.K)
    HBlockDiag(A.p, B.q, K_result, M_result, C_data)
end

# ============================================================================
# Sparse fallbacks: BlockDiag * SparseMatrixCSC and vice versa
# ============================================================================

function Base.:*(A::SparseMatrixCSC{T,Int}, B::BlockDiag{T}) where T
    A * to_sparse(B)
end

function Base.:*(A::BlockDiag{T}, B::SparseMatrixCSC{T,Int}) where T
    to_sparse(A) * B
end

function Base.:*(A::Adjoint{T,<:BlockDiag{T}}, B::SparseMatrixCSC{T,Int}) where T
    to_sparse(parent(A))' * B
end

function Base.:*(A::SparseMatrixCSC{T,Int}, B::VBlockDiag{T}) where T
    A * to_sparse(B)
end

function Base.:*(A::SparseMatrixCSC{T,Int}, B::HBlockDiag{T}) where T
    A * to_sparse(B)
end

function Base.:*(A::VBlockDiag{T}, B::SparseMatrixCSC{T,Int}) where T
    to_sparse(A) * B
end

function Base.:*(A::HBlockDiag{T}, B::SparseMatrixCSC{T,Int}) where T
    to_sparse(A) * B
end

# ============================================================================
# amgb_* dispatch methods
# ============================================================================

amgb_zeros(::BlockColumn{T}, m, n) where {T} = spzeros(T, m, n)
amgb_zeros(::Adjoint{T, <:BlockColumn{T}}, m, n) where {T} = spzeros(T, m, n)

function amgb_zeros(A::BlockDiag{T}, m, n) where T
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    data = similar(A.data, A.p, A.p, N)
    fill!(data, zero(T))
    BlockDiag{T, typeof(data)}(A.p, A.p, N, data)
end

function amgb_zeros(A::SubBlockDiag{T}, m, n) where T
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    data = similar(A.data, A.p, A.p, N)
    fill!(data, zero(T))
    BlockDiag{T, typeof(data)}(A.p, A.p, N, data)
end

# amgb_blockdiag for V/HBlockDiag
function amgb_blockdiag(args::VBlockDiag{T}...) where T
    p = args[1].p
    q = args[1].q
    K = args[1].K
    for a in args
        @assert a.p == p && a.q == q && a.K == K
    end
    M_total = sum(a.M for a in args)
    data_new = cat([a.data for a in args]...; dims=3)
    VBlockDiag(p, q, K, M_total, data_new)
end

function amgb_blockdiag(args::HBlockDiag{T}...) where T
    p = args[1].p
    q = args[1].q
    K = args[1].K
    for a in args
        @assert a.p == p && a.q == q && a.K == K
    end
    M_total = sum(a.M for a in args)
    data_new = cat([a.data for a in args]...; dims=3)
    HBlockDiag(p, q, K, M_total, data_new)
end

# ============================================================================
# hcat for constructing BlockColumn from D0 in amg_helper
# ============================================================================

function Base.hcat(args::BlockDiag{T}...) where T
    block_idx = 0
    for (i, a) in enumerate(args)
        if sum(abs2, a.data) > zero(T)
            if block_idx != 0
                return hcat((to_sparse(a) for a in args)...)
            end
            block_idx = i
        end
    end
    if block_idx == 0
        return hcat((to_sparse(a) for a in args)...)
    end
    blk = args[block_idx]
    nu = length(args)
    col_sizes = [size(a, 2) for a in args]
    total_rows = size(blk, 1)
    A3 = typeof(blk.data)
    BlockColumn{T, A3}(blk, block_idx, nu, col_sizes, total_rows)
end

# Mixed hcat: BlockDiag + SparseMatrixCSC
function Base.hcat(A::BlockDiag{T}, B::SparseMatrixCSC{T,Int}) where T
    _hcat_mixed_cpu(T, Any[A, B])
end

function Base.hcat(A::SparseMatrixCSC{T,Int}, B::BlockDiag{T}) where T
    _hcat_mixed_cpu(T, Any[A, B])
end

function _hcat_block_column_cpu(args::Vector)
    block_idx = 0
    T_elem = nothing
    for (i, a) in enumerate(args)
        if a isa BlockDiag
            if block_idx != 0
                return nothing
            end
            block_idx = i
            T_elem = eltype(a.data)
        end
    end
    if block_idx == 0
        return nothing
    end
    for (i, a) in enumerate(args)
        if i != block_idx
            if a isa SparseMatrixCSC && nnz(a) == 0
                continue
            else
                return nothing
            end
        end
    end
    blk = args[block_idx]::BlockDiag
    nu = length(args)
    col_sizes = [size(a, 2) for a in args]
    total_rows = size(blk, 1)
    A3 = typeof(blk.data)
    BlockColumn{T_elem, A3}(blk, block_idx, nu, col_sizes, total_rows)
end

function _hcat_mixed_cpu(::Type{T}, args::Vector) where T
    result = _hcat_block_column_cpu(args)
    if result !== nothing
        return result
    end
    sparse_args = [a isa BlockDiag ? to_sparse(a) : a for a in args]
    hcat(sparse_args...)
end

# ============================================================================
# BlockDiag - UniformScaling and norm (for amg_helper sanity check)
# ============================================================================

function Base.:-(A::BlockDiag{T}, ::UniformScaling) where T
    @assert A.p == A.q
    p = A.p
    I_cpu = zeros(T, p, p, A.N)
    for k in 1:A.N, i in 1:p
        I_cpu[i, i, k] = one(T)
    end
    I_data = block_alloc(T, A.data, p, p, A.N)
    copyto!(I_data, I_cpu)
    new_data = A.data .- I_data
    BlockDiag{T, typeof(new_data)}(p, p, A.N, new_data)
end

LinearAlgebra.norm(A::BlockDiag) = norm(A.data)

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

function to_sparse(B::BlockHessian{T}) where T
    total = sum(B.block_sizes)
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    nu = size(B.blocks, 1)
    row_offset = zeros(Int, nu)
    for k in 2:nu
        row_offset[k] = row_offset[k-1] + B.block_sizes[k-1]
    end
    for bi in 1:nu, bj in 1:nu
        blk = B.blocks[bi, bj]
        if blk === nothing
            continue
        end
        S = to_sparse(blk)
        rows, cols, vals = findnz(S)
        append!(I_idx, rows .+ row_offset[bi])
        append!(J_idx, cols .+ row_offset[bj])
        append!(V_val, vals)
    end
    sparse(I_idx, J_idx, V_val, total, total)
end

function to_sparse(B::VBlockDiag{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = K * p * M
    total_cols = q * M
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    for i in 1:M
        for j in 1:K
            sub_idx = (i - 1) * K + j
            for c in 1:q
                for r in 1:p
                    v = B.data[r, c, sub_idx]
                    if v != zero(T)
                        global_row = (i - 1) * K * p + (j - 1) * p + r
                        global_col = (i - 1) * q + c
                        push!(I_idx, global_row)
                        push!(J_idx, global_col)
                        push!(V_val, v)
                    end
                end
            end
        end
    end
    sparse(I_idx, J_idx, V_val, total_rows, total_cols)
end

function to_sparse(B::HBlockDiag{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = p * M
    total_cols = K * q * M
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    for i in 1:M
        for j in 1:K
            sub_idx = (i - 1) * K + j
            for c in 1:q
                for r in 1:p
                    v = B.data[r, c, sub_idx]
                    if v != zero(T)
                        global_row = (i - 1) * p + r
                        global_col = (i - 1) * K * q + (j - 1) * q + c
                        push!(I_idx, global_row)
                        push!(J_idx, global_col)
                        push!(V_val, v)
                    end
                end
            end
        end
    end
    sparse(I_idx, J_idx, V_val, total_rows, total_cols)
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

function _extract_sub_block_diag(A::SparseMatrixCSC{T,Int}, p::Int, K::Int, orient::Symbol) where T
    m, n = size(A)
    if orient == :V
        @assert m % (K * p) == 0
        M = m ÷ (K * p)
        @assert n == p * M
    else
        @assert n % (K * p) == 0
        M = n ÷ (K * p)
        @assert m == p * M
    end

    data = zeros(T, p, p, M * K)

    if orient == :V
        for i in 1:M
            for j in 1:K
                sub_idx = (i - 1) * K + j
                for r in 1:p
                    global_row = (i - 1) * K * p + (j - 1) * p + r
                    for c in 1:p
                        global_col = (i - 1) * p + c
                        data[r, c, sub_idx] = A[global_row, global_col]
                    end
                end
            end
        end
        VBlockDiag(p, p, K, M, data)
    else
        for i in 1:M
            for j in 1:K
                sub_idx = (i - 1) * K + j
                for r in 1:p
                    global_row = (i - 1) * p + r
                    for c in 1:p
                        global_col = (i - 1) * K * p + (j - 1) * p + c
                        data[r, c, sub_idx] = A[global_row, global_col]
                    end
                end
            end
        end
        HBlockDiag(p, p, K, M, data)
    end
end

# ============================================================================
# _structurize_geometry: convert Geometry operators/refine/coarsen to block types
# ============================================================================

# _default_block_size: stub. Methods added in fem1d.jl, fem2d.jl, Mesh3d/Mesh3d.jl
function _default_block_size end

function _structurize_geometry(g::Geometry{T,X,W,SparseMatrixCSC{T,Int},M_ref,M_coar,M_sub,Disc},
                               p::Int) where {T,X,W,M_ref,M_coar,M_sub,Disc}
    L = length(g.refine)

    # Convert operators to BlockDiag
    operators_new = Dict(key => _extract_block_diag(op, p) for (key, op) in g.operators)

    # Convert refine/coarsen to V/HBlockDiag
    m1, n1 = size(g.refine[1])
    if m1 == n1
        ref1 = _extract_sub_block_diag(g.refine[1], p, 1, :V)
        coar1 = _extract_sub_block_diag(g.coarsen[1], p, 1, :H)
    else
        N_1 = n1 ÷ p
        K_1 = m1 ÷ (p * N_1)
        ref1 = _extract_sub_block_diag(g.refine[1], p, K_1, :V)
        coar1 = _extract_sub_block_diag(g.coarsen[1], p, K_1, :H)
    end
    refine_new = Vector{typeof(ref1)}(undef, L)
    coarsen_new = Vector{typeof(coar1)}(undef, L)
    refine_new[1] = ref1
    coarsen_new[1] = coar1
    for l in 2:L
        m, n = size(g.refine[l])
        if m == n
            refine_new[l] = _extract_sub_block_diag(g.refine[l], p, 1, :V)
            coarsen_new[l] = _extract_sub_block_diag(g.coarsen[l], p, 1, :H)
        else
            N_l = n ÷ p
            K_l = m ÷ (p * N_l)
            refine_new[l] = _extract_sub_block_diag(g.refine[l], p, K_l, :V)
            coarsen_new[l] = _extract_sub_block_diag(g.coarsen[l], p, K_l, :H)
        end
    end

    # Subspaces stay as their original type
    subspaces_new = Dict{Symbol, Vector{M_sub}}()
    for (key, vec) in g.subspaces
        subspaces_new[key] = Vector{M_sub}(vec)
    end

    M_op_type = valtype(operators_new)
    M_ref_type = eltype(refine_new)
    M_coar_type = eltype(coarsen_new)
    Geometry{T,X,W,M_op_type,M_ref_type,M_coar_type,M_sub,Disc}(
        g.discretization, g.x, g.w,
        subspaces_new, operators_new, refine_new, coarsen_new)
end

# Cleanup: clear assembly plan cache
function _block_assembly_cleanup()
    empty!(_block_assembly_plan_cache)
end

# Extend amgb_cleanup for block-structured solutions
function amgb_cleanup(sol::AMGBSOL{T, <:Any, <:Vector{T}}) where T
    # Check if the geometry uses block types
    if sol.geometry isa Geometry && any(v -> v isa BlockDiag, values(sol.geometry.operators))
        _block_assembly_cleanup()
    end
    sol
end
