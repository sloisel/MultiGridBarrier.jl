# block_ops.jl -- CUDA kernel specializations for structured block matrix operations
#
# The core block operations are defined in MultiGridBarrier/BlockMatrices.jl using
# overridable kernel functions (block_batched_gemm!, block_fused_triple!, etc.).
# This file provides:
#   1. CUDA kernel specializations for those kernel functions
#   2. CuVector/CuMatrix matvec/matmat operations (GPU kernels)
#   3. R' * BlockHessian * R assembly (GPU kernels + plans)
#   4. Sparse conversion and extraction for CuArray-backed block types

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

import MultiGridBarrier: amgb_diag, amgb_zeros, amgb_blockdiag, apply_D, amgb_cleanup,
                         block_batched_gemm!, block_fused_triple!, block_segmented_sum!,
                         block_batched_gemm_broadcast_B!, block_batched_gemm_broadcast_A!,
                         block_alloc,
                         BlockDiag, BlockColumn, BlockHessian, SubBlockDiag,
                         VBlockDiag, HBlockDiag, ScaledAdjBlockCol, LazyBlockHessianProduct

# ============================================================================
# CUDA Kernels (definitions)
# ============================================================================

"""
    _batched_gemm_kernel!(C, A, B, transA, rows_C, cols_C, inner, N)

CUDA kernel for batched matrix multiply: C[:,:,i] = op(A[:,:,i]) * B[:,:,i]
"""
function _batched_gemm_kernel!(C, A, B, ::Val{transA}, rows_C, cols_C, inner, N) where transA
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = rows_C * cols_C * N
    if idx <= total
        idx0 = idx - 1
        i = idx0 % rows_C + 1
        idx0 = idx0 ÷ rows_C
        j = idx0 % cols_C + 1
        blk = idx0 ÷ cols_C + 1
        s = zero(eltype(C))
        for k = 1:inner
            if transA
                @inbounds s += A[k, i, blk] * B[k, j, blk]
            else
                @inbounds s += A[i, k, blk] * B[k, j, blk]
            end
        end
        @inbounds C[i, j, blk] = s
    end
    return nothing
end

"""
    _fused_triple_kernel!(C, A, v, B, rows_C, cols_C, inner, N)

Fused kernel for A' * diag(v) * B.
"""
function _fused_triple_kernel!(C, A, v, B, rows_C, cols_C, inner, N)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = rows_C * cols_C * N
    if idx <= total
        idx0 = idx - 1
        i = idx0 % rows_C + 1
        idx0 = idx0 ÷ rows_C
        j = idx0 % cols_C + 1
        blk = idx0 ÷ cols_C + 1
        v_offset = (blk - 1) * inner
        s = zero(eltype(C))
        for k = 1:inner
            @inbounds s += A[k, i, blk] * v[v_offset + k] * B[k, j, blk]
        end
        @inbounds C[i, j, blk] = s
    end
    return nothing
end

"""
    _batched_gemm_broadcast_B_kernel!(C, A, B, rows_C, cols_C, inner, N, K)

CUDA kernel: C[:,:,s] = A[:,:,s] * B[:,:,(s-1)÷K+1]
"""
function _batched_gemm_broadcast_B_kernel!(C, A, B, rows_C, cols_C, inner, N, K)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = rows_C * cols_C * N
    if idx <= total
        idx0 = idx - 1
        i = idx0 % rows_C + 1
        idx0 = idx0 ÷ rows_C
        j = idx0 % cols_C + 1
        blk = idx0 ÷ cols_C + 1
        b_blk = (blk - 1) ÷ K + 1
        s = zero(eltype(C))
        for k = 1:inner
            @inbounds s += A[i, k, blk] * B[k, j, b_blk]
        end
        @inbounds C[i, j, blk] = s
    end
    return nothing
end

"""
    _batched_gemm_broadcast_A_kernel!(C, A, B, rows_C, cols_C, inner, N, K_B)

CUDA kernel: C[:,:,s] = A[:,:,(s-1)÷K_B+1] * B[:,:,s]
"""
function _batched_gemm_broadcast_A_kernel!(C, A, B, rows_C, cols_C, inner, N, K_B)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = rows_C * cols_C * N
    if idx <= total
        idx0 = idx - 1
        i = idx0 % rows_C + 1
        idx0 = idx0 ÷ rows_C
        j = idx0 % cols_C + 1
        blk = idx0 ÷ cols_C + 1
        a_blk = (blk - 1) ÷ K_B + 1
        s = zero(eltype(C))
        for k = 1:inner
            @inbounds s += A[i, k, a_blk] * B[k, j, blk]
        end
        @inbounds C[i, j, blk] = s
    end
    return nothing
end

"""
    _segmented_block_sum_kernel!(out, data, p, q, K, M)

CUDA kernel: out[:,:,i] = sum(data[:,:,(i-1)*K+1 : i*K]) for i = 1:M.
"""
function _segmented_block_sum_kernel!(out, data, p, q, K, M)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = p * q * M
    if idx <= total
        idx0 = idx - 1
        r = idx0 % p + 1
        idx0 = idx0 ÷ p
        c = idx0 % q + 1
        outer = idx0 ÷ q + 1
        s = zero(eltype(out))
        for k = 1:K
            @inbounds s += data[r, c, (outer - 1) * K + k]
        end
        @inbounds out[r, c, outer] = s
    end
    return nothing
end

# ============================================================================
# Kernel specializations: override core's generic kernels for CuArray
# ============================================================================

function _launch_cuda_kernel(kernel_func, args...; total::Int)
    kernel = @cuda launch=false kernel_func(args...)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(args...; threads=threads, blocks=blocks)
end

function MultiGridBarrier.block_batched_gemm!(C::CuArray{T,3}, A::CuArray{T,3},
                                               B::CuArray{T,3}, vt::Val{transA}) where {T, transA}
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    N = size(C, 3)
    inner = transA ? size(A, 1) : size(A, 2)
    total = rows_C * cols_C * N
    _launch_cuda_kernel(_batched_gemm_kernel!, C, A, B, vt, rows_C, cols_C, inner, N; total=total)
    C
end

function MultiGridBarrier.block_fused_triple!(C::CuArray{T,3}, A::CuArray{T,3},
                                               v::CuVector{T}, B::CuArray{T,3}, p::Int) where T
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    N = size(C, 3)
    total = rows_C * cols_C * N
    _launch_cuda_kernel(_fused_triple_kernel!, C, A, v, B, rows_C, cols_C, p, N; total=total)
    C
end

function MultiGridBarrier.block_batched_gemm_broadcast_B!(C::CuArray{T,3}, A::CuArray{T,3},
                                                           B::CuArray{T,3}, K::Int) where T
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    N = size(C, 3)
    inner = size(A, 2)
    total = rows_C * cols_C * N
    _launch_cuda_kernel(_batched_gemm_broadcast_B_kernel!, C, A, B, rows_C, cols_C, inner, N, K; total=total)
    C
end

function MultiGridBarrier.block_batched_gemm_broadcast_A!(C::CuArray{T,3}, A::CuArray{T,3},
                                                           B::CuArray{T,3}, K_B::Int) where T
    rows_C = size(C, 1)
    cols_C = size(C, 2)
    N = size(C, 3)
    inner = size(A, 2)
    total = rows_C * cols_C * N
    _launch_cuda_kernel(_batched_gemm_broadcast_A_kernel!, C, A, B, rows_C, cols_C, inner, N, K_B; total=total)
    C
end

function MultiGridBarrier.block_segmented_sum!(out::CuArray{T,3}, data::CuArray{T,3}, K::Int) where T
    p = size(out, 1)
    q = size(out, 2)
    M = size(out, 3)
    total = p * q * M
    _launch_cuda_kernel(_segmented_block_sum_kernel!, out, data, p, q, K, M; total=total)
    out
end

function MultiGridBarrier.block_alloc(::Type{T}, A::CuArray, dims...) where T
    CuArray{T}(undef, dims...)
end

# ============================================================================
# diag_scale: diag(v) * BlockDiag on GPU
# ============================================================================

function _block_diag_scale_kernel!(out, data, v, p, q, N)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = p * q * N
    if idx <= total
        idx0 = idx - 1
        r = idx0 % p + 1
        idx0 = idx0 ÷ p
        c = idx0 % q + 1
        blk = idx0 ÷ q + 1
        vi = v[(blk - 1) * p + r]
        @inbounds out[r, c, blk] = vi * data[r, c, blk]
    end
    return nothing
end

function diag_scale(v::CuVector{T}, B::CuBlockDiag{T}) where T
    @assert length(v) == B.p * B.N
    out = similar(B.data)
    total = B.p * B.q * B.N
    _launch_cuda_kernel(_block_diag_scale_kernel!, out, B.data, v, B.p, B.q, B.N; total=total)
    BlockDiag(out)
end

# ============================================================================
# Matrix-vector products (GPU kernels)
# ============================================================================

function _block_matvec_kernel!(out, data, v, p, q, N, col_offset)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = p * N
    if idx <= total
        idx0 = idx - 1
        r = idx0 % p + 1
        blk = idx0 ÷ p + 1
        s = zero(eltype(out))
        for c = 1:q
            @inbounds s += data[r, c, blk] * v[col_offset + (blk - 1) * q + c]
        end
        @inbounds out[(blk - 1) * p + r] = s
    end
    return nothing
end

function _block_adj_matvec_kernel!(out, data, v, p, q, N, col_offset)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = q * N
    if idx <= total
        idx0 = idx - 1
        c = idx0 % q + 1
        blk = idx0 ÷ q + 1
        s = zero(eltype(out))
        for r = 1:p
            @inbounds s += data[r, c, blk] * v[(blk - 1) * p + r]
        end
        @inbounds out[col_offset + (blk - 1) * q + c] = s
    end
    return nothing
end

# BlockColumn * CuVector
function Base.:*(A::CuBlockColumn{T}, z::CuVector{T}) where T
    col_offset = sum(A.col_sizes[1:A.active_col-1])
    p = A.active_block.p
    q = A.active_block.q
    N = A.active_block.N
    out = CUDA.zeros(T, A.total_rows)
    total = p * N
    _launch_cuda_kernel(_block_matvec_kernel!, out, A.active_block.data, z, p, q, N, col_offset; total=total)
    out
end

# adjoint(BlockColumn) * CuVector
function Base.:*(A::Adjoint{T,<:CuBlockColumn{T}}, z::CuVector{T}) where T
    op = parent(A)
    col_offset = sum(op.col_sizes[1:op.active_col-1])
    p = op.active_block.p
    q = op.active_block.q
    N = op.active_block.N
    out = CUDA.zeros(T, sum(op.col_sizes))
    total = q * N
    _launch_cuda_kernel(_block_adj_matvec_kernel!, out, op.active_block.data, z, p, q, N, col_offset; total=total)
    out
end

# BlockDiag * CuVector
function Base.:*(A::CuBlockDiag{T}, x::CuVector{T}) where T
    p, q, N = A.p, A.q, A.N
    @assert length(x) == q * N
    out = CuVector{T}(undef, p * N)
    total = p * N
    _launch_cuda_kernel(_block_matvec_kernel!, out, A.data, x, p, q, N, 0; total=total)
    out
end

# --- VBlockDiag * CuVector ---
function _vblock_matvec_kernel!(out, data, x, p, q, K, MK)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= p * MK
        idx0 = idx - 1
        r = idx0 % p + 1
        blk = idx0 ÷ p + 1
        outer = (blk - 1) ÷ K + 1
        s = zero(eltype(out))
        for c = 1:q
            @inbounds s += data[r, c, blk] * x[(outer - 1) * q + c]
        end
        @inbounds out[idx] = s
    end
    return nothing
end

function Base.:*(A::CuVBlockDiag{T}, x::CuVector{T}) where T
    @assert length(x) == A.q * A.M
    MK = A.M * A.K
    total = A.p * MK
    out = CuVector{T}(undef, total)
    _launch_cuda_kernel(_vblock_matvec_kernel!, out, A.data, x, A.p, A.q, A.K, MK; total=total)
    out
end

# --- HBlockDiag * CuVector ---
function _hblock_matvec_kernel!(out, data, x, p, q, K, M)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= p * M
        idx0 = idx - 1
        r = idx0 % p + 1
        outer = idx0 ÷ p + 1
        s = zero(eltype(out))
        for j = 1:K
            sub_idx = (outer - 1) * K + j
            for c = 1:q
                @inbounds s += data[r, c, sub_idx] * x[(sub_idx - 1) * q + c]
            end
        end
        @inbounds out[idx] = s
    end
    return nothing
end

function Base.:*(A::CuHBlockDiag{T}, x::CuVector{T}) where T
    @assert length(x) == A.K * A.q * A.M
    total = A.p * A.M
    out = CuVector{T}(undef, total)
    _launch_cuda_kernel(_hblock_matvec_kernel!, out, A.data, x, A.p, A.q, A.K, A.M; total=total)
    out
end

# --- adjoint(VBlockDiag) * CuVector ---
function _vblock_adj_matvec_kernel!(out, data, x, p, q, K, M)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= q * M
        idx0 = idx - 1
        c = idx0 % q + 1
        outer = idx0 ÷ q + 1
        s = zero(eltype(out))
        for j = 1:K
            sub_idx = (outer - 1) * K + j
            for r = 1:p
                @inbounds s += data[r, c, sub_idx] * x[(sub_idx - 1) * p + r]
            end
        end
        @inbounds out[idx] = s
    end
    return nothing
end

function Base.:*(A::Adjoint{T, <:CuVBlockDiag{T}}, x::CuVector{T}) where T
    V = parent(A)
    @assert length(x) == V.K * V.p * V.M
    total = V.q * V.M
    out = CuVector{T}(undef, total)
    _launch_cuda_kernel(_vblock_adj_matvec_kernel!, out, V.data, x, V.p, V.q, V.K, V.M; total=total)
    out
end

# --- adjoint(HBlockDiag) * CuVector ---
function _hblock_adj_matvec_kernel!(out, data, x, p, q, K, MK)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= q * MK
        idx0 = idx - 1
        c = idx0 % q + 1
        sub_idx = idx0 ÷ q + 1
        outer = (sub_idx - 1) ÷ K + 1
        s = zero(eltype(out))
        for r = 1:p
            @inbounds s += data[r, c, sub_idx] * x[(outer - 1) * p + r]
        end
        @inbounds out[idx] = s
    end
    return nothing
end

function Base.:*(A::Adjoint{T, <:CuHBlockDiag{T}}, x::CuVector{T}) where T
    H = parent(A)
    MK = H.M * H.K
    @assert length(x) == H.p * H.M
    total = H.q * MK
    out = CuVector{T}(undef, total)
    _launch_cuda_kernel(_hblock_adj_matvec_kernel!, out, H.data, x, H.p, H.q, H.K, MK; total=total)
    out
end

# --- Matrix-matrix products (column-loop over CuVector kernels) ---

function Base.:*(A::CuBlockDiag{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::CuVBlockDiag{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::CuHBlockDiag{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::Adjoint{T, <:CuVBlockDiag{T}}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::Adjoint{T, <:CuHBlockDiag{T}}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

# ============================================================================
# amgb_cleanup: flush GPU caches when solve completes
# ============================================================================

const _sparse_plan_cache = Dict{UInt64, Any}()
const _assembly_plan_cache = Dict{UInt64, Any}()

function MultiGridBarrier.amgb_cleanup(sol::MultiGridBarrier.AMGBSOL{T, <:Any, <:CuVector}) where T
    empty!(_sparse_plan_cache)
    empty!(_assembly_plan_cache)
    MultiGridBarrier.clear_cudss_cache!()
    sol
end

# ============================================================================
# Sparse conversion: BlockHessian (CuArray-backed) → CuSparseMatrixCSR
# ============================================================================

function _scatter_nzval_kernel!(nzval, combined, scatter_block, scatter_offset,
                                 scatter_element, p, N, total_nnz)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= total_nnz
        blk_idx = scatter_block[idx]
        offset = scatter_offset[idx]
        e = scatter_element[idx]
        r = ((offset - Int32(1)) % Int32(p)) + Int32(1)
        c = ((offset - Int32(1)) ÷ Int32(p)) + Int32(1)
        @inbounds nzval[idx] = combined[r, c, (blk_idx - Int32(1)) * Int32(N) + e]
    end
    return nothing
end

function make_sparse_plan(H::CuBlockHessian{T}; Ti=Int32) where T
    nu = size(H.blocks, 1)
    p = H.p
    N = H.N
    block_sizes = H.block_sizes

    nz_bi = Int[]
    nz_bj = Int[]
    for bi in 1:nu, bj in 1:nu
        if H.blocks[bi, bj] !== nothing
            push!(nz_bi, bi)
            push!(nz_bj, bj)
        end
    end

    total_rows = sum(block_sizes)
    total_cols = total_rows

    row_offset = zeros(Int, nu)
    for bi in 2:nu
        row_offset[bi] = row_offset[bi-1] + block_sizes[bi-1]
    end
    col_offset = row_offset

    rowptr_cpu = Vector{Ti}(undef, total_rows + 1)
    colval_list = Ti[]
    scatter_block_list = Int32[]
    scatter_offset_list = Int32[]

    rowptr_cpu[1] = Ti(1)
    for bi in 1:nu
        for e in 1:N
            for r in 1:p
                global_row = row_offset[bi] + (e - 1) * p + r
                entries = Tuple{Ti, Int, Int32}[]
                for (nz_idx, (nbi, nbj)) in enumerate(zip(nz_bi, nz_bj))
                    if nbi == bi
                        blk = H.blocks[nbi, nbj]
                        q_blk = blk.q
                        for c in 1:q_blk
                            global_col = Ti(col_offset[nbj] + (e - 1) * q_blk + c)
                            offset = Int32(r + (c - 1) * p)
                            push!(entries, (global_col, nz_idx, offset))
                        end
                    end
                end
                sort!(entries, by=x->x[1])
                for (col, nz_idx, offset) in entries
                    push!(colval_list, col)
                    push!(scatter_block_list, Int32(nz_idx))
                    push!(scatter_offset_list, offset)
                end
                rowptr_cpu[global_row + 1] = Ti(length(colval_list) + 1)
            end
        end
    end

    total_nnz = length(colval_list)

    rowptr_final = vcat(Int32[1], Ti.(cumsum([rowptr_cpu[i+1] - rowptr_cpu[i] for i in 1:total_rows]) .+ 1))
    rows_per_group = p * N
    scatter_element_list = Int32[]
    for row in 1:total_rows
        bi_row = (row - 1) ÷ rows_per_group + 1
        row_in_group = (row - 1) % rows_per_group
        e = Int32(row_in_group ÷ p + 1)
        nz_count = rowptr_cpu[row + 1] - rowptr_cpu[row]
        for _ in 1:nz_count
            push!(scatter_element_list, e)
        end
    end

    SparseConversionPlan{Ti}(
        CuVector{Ti}(rowptr_cpu),
        CuVector{Ti}(colval_list),
        total_nnz,
        total_rows,
        total_cols,
        CuVector{Int32}(scatter_block_list),
        CuVector{Int32}(scatter_offset_list),
        CuVector{Int32}(scatter_element_list),
        nz_bi,
        nz_bj
    )
end

function to_cusparse(H::CuBlockHessian{T}, plan::SparseConversionPlan{Ti}) where {T, Ti}
    p = H.p
    N = H.N
    n_nz_blocks = length(plan.block_bi)

    combined = CuArray{T}(undef, p, p, n_nz_blocks * N)
    for (nz_idx, (bi, bj)) in enumerate(zip(plan.block_bi, plan.block_bj))
        blk = H.blocks[bi, bj]
        combined[:, :, (nz_idx-1)*N+1 : nz_idx*N] = blk.data
    end

    total_nnz = plan.total_nnz
    nzval = CuVector{T}(undef, total_nnz)

    _launch_cuda_kernel(_scatter_nzval_kernel!,
        nzval, combined, plan.scatter_block, plan.scatter_offset,
        plan.scatter_element, Int32(p), Int32(N), Int32(total_nnz);
        total=total_nnz)

    CuSparseMatrixCSR{T}(plan.rowptr, plan.colval, nzval,
                          (plan.total_rows, plan.total_cols))
end

function _get_sparse_plan(H::CuBlockHessian{T}; Ti=Int32) where T
    nu = size(H.blocks, 1)
    key = hash((nu, H.p, H.N, H.block_sizes,
                [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing]))
    get!(_sparse_plan_cache, key) do
        make_sparse_plan(H; Ti=Ti)
    end::SparseConversionPlan{Ti}
end

# ============================================================================
# R' * BlockHessian * R dispatch (CuSparseMatrixCSR)
# ============================================================================

# R' * H → LazyBlockHessianProduct (lazy, no work)
function Base.:*(A::Adjoint{T, CuSparseMatrixCSR{T,Ti}}, H::CuBlockHessian{T}) where {T, Ti}
    A3 = typeof(first(b for b in H.blocks if b !== nothing).data)
    LazyBlockHessianProduct{T, A3, CuSparseMatrixCSR{T,Ti}}(parent(A), H)
end

# LazyBlockHessianProduct * R → CuSparseMatrixCSR (element-wise assembly)
function Base.:*(lhp::LazyBlockHessianProduct{T,<:CuArray,CuSparseMatrixCSR{T,Ti}},
                 R::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    @assert lhp.R === R "LazyBlockHessianProduct expects same R on both sides"
    _assemble_RtHR(lhp.R, lhp.H)
end

# Fallback: BlockHessian * CuSparse
function Base.:*(H::CuBlockHessian{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    plan = _get_sparse_plan(H; Ti=Ti)
    H_sparse = to_cusparse(H, plan)
    return H_sparse * B
end

# ============================================================================
# Element-wise assembly: R' * BlockHessian * R without SPGEMM
# ============================================================================

function _dot_scatter_kernel!(output_nzval, panel_i, tmp, scatter, c_max_i, c_max_j, p, N)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = c_max_i * c_max_j * N
    if idx <= total
        idx0 = idx - 1
        a = idx0 % c_max_i + 1
        idx0 = idx0 ÷ c_max_i
        b = idx0 % c_max_j + 1
        e = idx0 ÷ c_max_j + 1

        scatter_pos = scatter[a, b, e]
        if scatter_pos > Int32(0)
            val = zero(eltype(output_nzval))
            for r = 1:p
                @inbounds val += panel_i[r, a, e] * tmp[r, b, e]
            end
            CUDA.@atomic output_nzval[scatter_pos] += val
        end
    end
    return nothing
end

function _make_assembly_plan(R::CuSparseMatrixCSR{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
    nu = size(H.blocks, 1)
    p = H.p
    N = H.N
    block_sizes = H.block_sizes

    R_rowptr_cpu = Array(R.rowPtr)
    R_colval_cpu = Array(R.colVal)
    R_nzval_cpu = Array(R.nzVal)
    nrows_R = size(R, 1)
    ncols_R = size(R, 2)

    row_offset = zeros(Int, nu)
    for k in 2:nu
        row_offset[k] = row_offset[k-1] + block_sizes[k-1]
    end

    col_indices_cpu = Vector{Matrix{Ti}}(undef, nu)
    c_counts_cpu = Vector{Vector{Int32}}(undef, nu)
    c_max = zeros(Int, nu)

    for k in 1:nu
        element_cols = Vector{Vector{Ti}}(undef, N)
        for e in 1:N
            cols_set = Set{Ti}()
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                if global_row > nrows_R
                    continue
                end
                for idx in R_rowptr_cpu[global_row]:(R_rowptr_cpu[global_row + 1] - 1)
                    push!(cols_set, R_colval_cpu[idx])
                end
            end
            element_cols[e] = sort!(collect(cols_set))
        end
        c_max[k] = maximum(length(ec) for ec in element_cols)

        ci = zeros(Ti, c_max[k], N)
        cc = zeros(Int32, N)
        for e in 1:N
            cc[e] = Int32(length(element_cols[e]))
            for a in 1:length(element_cols[e])
                ci[a, e] = element_cols[e][a]
            end
        end
        col_indices_cpu[k] = ci
        c_counts_cpu[k] = cc
    end

    panels_cpu = Vector{Array{T, 3}}(undef, nu)
    for k in 1:nu
        panel = zeros(T, p, c_max[k], N)
        ci = col_indices_cpu[k]
        for e in 1:N
            nc = c_counts_cpu[k][e]
            col_to_local = Dict{Ti, Int}()
            for a in 1:nc
                col_to_local[ci[a, e]] = a
            end
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                if global_row > nrows_R
                    continue
                end
                for idx in R_rowptr_cpu[global_row]:(R_rowptr_cpu[global_row + 1] - 1)
                    col = R_colval_cpu[idx]
                    a = col_to_local[col]
                    panel[r, a, e] = R_nzval_cpu[idx]
                end
            end
        end
        panels_cpu[k] = panel
    end

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
                    push!(out_rows, Int(ci_i[a, e]))
                    push!(out_cols, Int(ci_j[b, e]))
                end
            end
        end
    end

    if isempty(out_rows)
        out_nnz = 0
        out_rowptr = CuVector{Ti}(ones(Ti, ncols_R + 1))
        out_colval = CuVector{Ti}(undef, 0)
        scatter_idx = Dict{Tuple{Int,Int}, CuArray{Int32, 3}}()
        panels_gpu = [CuArray{T}(panels_cpu[k]) for k in 1:nu]
        col_indices_gpu = [CuArray{Ti}(col_indices_cpu[k]) for k in 1:nu]
        c_counts_gpu = [CuVector{Int32}(c_counts_cpu[k]) for k in 1:nu]
        return AssemblyPlan{T, Ti}(
            out_rowptr, out_colval, ncols_R, ncols_R, out_nnz,
            panels_gpu, col_indices_gpu, c_counts_gpu,
            scatter_idx, p, N, nu, c_max)
    end

    indicator = sparse(out_rows, out_cols, ones(Float32, length(out_rows)), ncols_R, ncols_R)
    S_t = SparseMatrixCSC(indicator')
    out_rowptr_cpu = Vector{Ti}(S_t.colptr)
    out_colval_cpu = Vector{Ti}(S_t.rowval)
    out_nnz = length(out_colval_cpu)

    scatter_idx = Dict{Tuple{Int,Int}, CuArray{Int32, 3}}()
    for bi in 1:nu, bj in 1:nu
        if H.blocks[bi, bj] === nothing
            continue
        end
        scatter = zeros(Int32, c_max[bi], c_max[bj], N)
        ci_i = col_indices_cpu[bi]
        ci_j = col_indices_cpu[bj]
        cc_i = c_counts_cpu[bi]
        cc_j = c_counts_cpu[bj]
        for e in 1:N
            for a in 1:cc_i[e]
                row = Int(ci_i[a, e])
                rs = Int(out_rowptr_cpu[row])
                re = Int(out_rowptr_cpu[row + 1]) - 1
                row_cols = @view out_colval_cpu[rs:re]
                for b in 1:cc_j[e]
                    col = ci_j[b, e]
                    lo, hi = 1, length(row_cols)
                    while lo <= hi
                        mid = (lo + hi) ÷ 2
                        if row_cols[mid] < col
                            lo = mid + 1
                        elseif row_cols[mid] > col
                            hi = mid - 1
                        else
                            scatter[a, b, e] = Int32(rs + mid - 1)
                            break
                        end
                    end
                end
            end
        end
        scatter_idx[(bi, bj)] = CuArray{Int32}(scatter)
    end

    panels_gpu = [CuArray{T}(panels_cpu[k]) for k in 1:nu]
    col_indices_gpu = [CuArray{Ti}(col_indices_cpu[k]) for k in 1:nu]
    c_counts_gpu = [CuVector{Int32}(c_counts_cpu[k]) for k in 1:nu]

    AssemblyPlan{T, Ti}(
        CuVector{Ti}(out_rowptr_cpu), CuVector{Ti}(out_colval_cpu),
        ncols_R, ncols_R, out_nnz,
        panels_gpu, col_indices_gpu, c_counts_gpu,
        scatter_idx, p, N, nu, c_max)
end

function _get_assembly_plan(R::CuSparseMatrixCSR{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
    nu = size(H.blocks, 1)
    key = hash((objectid(R), H.p, H.N, H.block_sizes,
                [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing]))
    get!(_assembly_plan_cache, key) do
        _make_assembly_plan(R, H)
    end::AssemblyPlan{T, Ti}
end

function _assemble_RtHR(R::CuSparseMatrixCSR{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
    plan = _get_assembly_plan(R, H)

    if plan.out_nnz == 0
        return CuSparseMatrixCSR{T}(plan.out_rowptr, plan.out_colval,
                                     CuVector{T}(undef, 0), (plan.out_m, plan.out_n))
    end

    output_nzval = CUDA.zeros(T, plan.out_nnz)

    nu = plan.nu
    p = plan.p
    N = plan.N

    cm_max = maximum(plan.c_max)
    tmp = CuArray{T}(undef, p, cm_max, N)

    first_blk = nothing
    first_scatter_key = nothing
    for bi in 1:nu, bj in 1:nu
        if H.blocks[bi, bj] !== nothing
            first_blk = H.blocks[bi, bj]
            first_scatter_key = (bi, bj)
            break
        end
    end
    if first_blk === nothing
        return CuSparseMatrixCSR{T}(plan.out_rowptr, plan.out_colval, output_nzval,
                                     (plan.out_m, plan.out_n))
    end

    gemm_kern = @cuda launch=false _batched_gemm_kernel!(
        tmp, first_blk.data, plan.panels[1], Val(false),
        Int32(p), Int32(cm_max), Int32(p), Int32(N))
    gemm_max_threads = launch_configuration(gemm_kern.fun).threads

    dot_kern = @cuda launch=false _dot_scatter_kernel!(
        output_nzval, plan.panels[1], tmp, plan.scatter_idx[first_scatter_key],
        Int32(cm_max), Int32(cm_max), Int32(p), Int32(N))
    dot_max_threads = launch_configuration(dot_kern.fun).threads

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

        total_gemm = p * cmj * N
        gemm_threads = min(total_gemm, gemm_max_threads)
        gemm_blocks = cld(total_gemm, gemm_threads)
        gemm_kern(tmp, blk.data, panel_j, Val(false), Int32(p), Int32(cmj), Int32(p), Int32(N);
                  threads=gemm_threads, blocks=gemm_blocks)

        total_dot = cmi * cmj * N
        dot_threads = min(total_dot, dot_max_threads)
        dot_blocks = cld(total_dot, dot_threads)
        dot_kern(output_nzval, panel_i, tmp, scatter, Int32(cmi), Int32(cmj), Int32(p), Int32(N);
                 threads=dot_threads, blocks=dot_blocks)
    end

    CuSparseMatrixCSR{T}(plan.out_rowptr, plan.out_colval, output_nzval,
                          (plan.out_m, plan.out_n))
end

# ============================================================================
# amgb_zeros for BlockColumn (CuArray-backed)
# ============================================================================

MultiGridBarrier.amgb_zeros(::CuBlockColumn{T}, m, n) where {T} =
    _cu_spzeros(T, m, n)
MultiGridBarrier.amgb_zeros(::Adjoint{T, <:CuBlockColumn{T}}, m, n) where {T} =
    _cu_spzeros(T, m, n)

# ============================================================================
# hcat for constructing BlockColumn from D0 in amg_helper (CuArray-backed)
# ============================================================================

# 2-arg hcat methods for CuBlockDiag + CuSparseMatrixCSR
function Base.hcat(A::CuBlockDiag{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _hcat_mixed(T, Ti, Any[A, B])
end

function Base.hcat(A::CuSparseMatrixCSR{T,Ti}, B::CuBlockDiag{T}) where {T, Ti}
    _hcat_mixed(T, Ti, Any[A, B])
end

function _hcat_block_column(args::Vector)
    block_idx = 0
    T_elem = nothing
    for (i, a) in enumerate(args)
        if a isa BlockDiag{<:Any, <:CuArray}
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
            if a isa CuSparseMatrixCSR && nnz(a) == 0
                continue
            else
                return nothing
            end
        end
    end
    blk = args[block_idx]::BlockDiag{<:Any, <:CuArray}
    nu = length(args)
    col_sizes = [size(a, 2) for a in args]
    total_rows = size(blk, 1)
    A3 = typeof(blk.data)
    BlockColumn{T_elem, A3}(blk, block_idx, nu, col_sizes, total_rows)
end

function _hcat_mixed(::Type{T}, ::Type{Ti}, args::Vector) where {T, Ti}
    result = _hcat_block_column(args)
    if result !== nothing
        return result
    end
    sparse_args = [a isa BlockDiag{<:Any, <:CuArray} ? _to_cusparse(a) : a for a in args]
    hcat(sparse_args...)
end

# ============================================================================
# BlockDiag (CuArray) → CuSparseMatrixCSR conversion
# ============================================================================

function _to_cusparse(B::CuBlockDiag{T}) where T
    p, q, N = B.p, B.q, B.N
    m = p * N
    n = q * N

    data_cpu = Array(B.data)
    nnz_total = 0
    for blk in 1:N, r in 1:p, c in 1:q
        if data_cpu[r, c, blk] != 0
            nnz_total += 1
        end
    end

    rowptr = Vector{Int32}(undef, m + 1)
    colval = Vector{Int32}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    pos = 1
    for blk in 1:N
        for r in 1:p
            global_row = (blk - 1) * p + r
            rowptr[global_row] = Int32(pos)
            for c in 1:q
                v = data_cpu[r, c, blk]
                if v != 0
                    colval[pos] = Int32((blk - 1) * q + c)
                    nzval[pos] = v
                    pos += 1
                end
            end
        end
    end
    rowptr[m + 1] = Int32(pos)

    CuSparseMatrixCSR{T}(CuVector{Int32}(rowptr), CuVector{Int32}(colval),
                          CuVector{T}(nzval), (m, n))
end

function _to_cusparse(B::CuVBlockDiag{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = K * p * M
    total_cols = q * M
    data_cpu = Array(B.data)

    nnz_total = count(!=(zero(T)), data_cpu)
    rowptr = Vector{Int32}(undef, total_rows + 1)
    colval = Vector{Int32}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    pos = 1
    for i in 1:M
        for j in 1:K
            sub_idx = (i - 1) * K + j
            for r in 1:p
                global_row = (i - 1) * K * p + (j - 1) * p + r
                rowptr[global_row] = Int32(pos)
                for c in 1:q
                    v = data_cpu[r, c, sub_idx]
                    if v != zero(T)
                        colval[pos] = Int32((i - 1) * q + c)
                        nzval[pos] = v
                        pos += 1
                    end
                end
            end
        end
    end
    rowptr[total_rows + 1] = Int32(pos)

    CuSparseMatrixCSR{T}(CuVector{Int32}(rowptr), CuVector{Int32}(colval),
                          CuVector{T}(nzval), (total_rows, total_cols))
end

function _to_cusparse(B::CuHBlockDiag{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = p * M
    total_cols = K * q * M
    data_cpu = Array(B.data)

    nnz_total = count(!=(zero(T)), data_cpu)
    rowptr = Vector{Int32}(undef, total_rows + 1)
    colval = Vector{Int32}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    pos = 1
    for i in 1:M
        for r in 1:p
            global_row = (i - 1) * p + r
            rowptr[global_row] = Int32(pos)
            for j in 1:K
                sub_idx = (i - 1) * K + j
                for c in 1:q
                    v = data_cpu[r, c, sub_idx]
                    if v != zero(T)
                        colval[pos] = Int32((i - 1) * K * q + (j - 1) * q + c)
                        nzval[pos] = v
                        pos += 1
                    end
                end
            end
        end
    end
    rowptr[total_rows + 1] = Int32(pos)

    CuSparseMatrixCSR{T}(CuVector{Int32}(rowptr), CuVector{Int32}(colval),
                          CuVector{T}(nzval), (total_rows, total_cols))
end

# ============================================================================
# Sparse fallbacks: CuSparse * block types and vice versa
# ============================================================================

function Base.:*(A::CuSparseMatrixCSR{T,Ti}, B::CuBlockDiag{T}) where {T, Ti}
    A * _to_cusparse(B)
end

function Base.:*(A::CuBlockDiag{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _to_cusparse(A) * B
end

function Base.:*(A::Adjoint{T,<:CuBlockDiag{T}}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _to_cusparse(parent(A))' * B
end

function Base.:*(A::CuSparseMatrixCSR{T,Ti}, B::SubBlockDiag{T,<:Any,<:CuArray{T,3}}) where {T, Ti}
    A * _to_cusparse(B)
end

function Base.:*(A::SubBlockDiag{T,<:Any,<:CuArray{T,3}}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _to_cusparse(A) * B
end

# ============================================================================
# Extraction: CuSparseMatrixCSR → block types (with CuArray backing)
# ============================================================================

function extract_block_diag(A::CuSparseMatrixCSR{T,Ti}, p::Int) where {T, Ti}
    m, n = size(A)
    @assert m % p == 0 "Matrix rows ($m) not divisible by block size ($p)"
    N = m ÷ p
    @assert n % N == 0 "Matrix cols ($n) not divisible by block count ($N)"
    q = n ÷ N

    A_cpu = SparseMatrixCSC(A)
    data_cpu = zeros(T, p, q, N)

    for blk in 1:N
        for r in 1:p
            global_row = (blk - 1) * p + r
            for c in 1:q
                global_col = (blk - 1) * q + c
                data_cpu[r, c, blk] = A_cpu[global_row, global_col]
            end
        end
    end

    BlockDiag(CuArray{T}(data_cpu))
end

function extract_block_column(D_entry::CuSparseMatrixCSR{T,Ti}, p::Int, nu::Int,
                               col_sizes::Vector{Int}) where {T, Ti}
    m = size(D_entry, 1)
    total_cols = sum(col_sizes)
    @assert size(D_entry, 2) == total_cols

    D_cpu = SparseMatrixCSC(D_entry)
    col_offsets = cumsum([0; col_sizes])
    active_col = 0
    for k in 1:nu
        c_start = col_offsets[k] + 1
        c_end = col_offsets[k + 1]
        sub = D_cpu[:, c_start:c_end]
        if nnz(sub) > 0
            @assert active_col == 0 "Multiple active column blocks found"
            active_col = k
        end
    end
    @assert active_col > 0 "No active column block found"

    c_start = col_offsets[active_col] + 1
    c_end = col_offsets[active_col + 1]
    active_sparse = D_cpu[:, c_start:c_end]

    active_cu = CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(
        active_sparse.m, active_sparse.n,
        Int32.(active_sparse.colptr), Int32.(active_sparse.rowval),
        active_sparse.nzval))
    active_block = extract_block_diag(active_cu, p)

    A3 = typeof(active_block.data)
    BlockColumn{T, A3}(active_block, active_col, nu, col_sizes, m)
end

function extract_sub_block_diag(A::CuSparseMatrixCSR{T,Ti}, p::Int, K::Int, orient::Symbol) where {T, Ti}
    m, n = size(A)
    if orient == :V
        @assert m % (K * p) == 0 "Rows ($m) not divisible by K*p ($(K*p))"
        M = m ÷ (K * p)
        @assert n == p * M "Cols ($n) should be p*M ($(p*M))"
    else
        @assert n % (K * p) == 0 "Cols ($n) not divisible by K*p ($(K*p))"
        M = n ÷ (K * p)
        @assert m == p * M "Rows ($m) should be p*M ($(p*M))"
    end

    A_cpu = SparseMatrixCSC(A)
    data_cpu = zeros(T, p, p, M * K)

    if orient == :V
        for i in 1:M
            for j in 1:K
                sub_idx = (i - 1) * K + j
                for r in 1:p
                    global_row = (i - 1) * K * p + (j - 1) * p + r
                    for c in 1:p
                        global_col = (i - 1) * p + c
                        data_cpu[r, c, sub_idx] = A_cpu[global_row, global_col]
                    end
                end
            end
        end
        VBlockDiag(p, p, K, M, CuArray{T}(data_cpu))
    else
        for i in 1:M
            for j in 1:K
                sub_idx = (i - 1) * K + j
                for r in 1:p
                    global_row = (i - 1) * p + r
                    for c in 1:p
                        global_col = (i - 1) * K * p + (j - 1) * p + c
                        data_cpu[r, c, sub_idx] = A_cpu[global_row, global_col]
                    end
                end
            end
        end
        HBlockDiag(p, p, K, M, CuArray{T}(data_cpu))
    end
end

# ============================================================================
# _structurize_geometry: convert Geometry operators/refine/coarsen to block types
# ============================================================================

function _structurize_geometry(g::MultiGridBarrier.Geometry{T,X,W,<:Any,<:Any,<:Any,M_sub,Disc},
                               p::Int) where {T,X,W,M_sub,Disc}
    L = length(g.refine)

    operators_new = Dict(key => extract_block_diag(op, p) for (key, op) in g.operators)

    m1, n1 = size(g.refine[1])
    if m1 == n1
        ref1 = extract_sub_block_diag(g.refine[1], p, 1, :V)
        coar1 = extract_sub_block_diag(g.coarsen[1], p, 1, :H)
    else
        N_1 = n1 ÷ p; K_1 = m1 ÷ (p * N_1)
        ref1 = extract_sub_block_diag(g.refine[1], p, K_1, :V)
        coar1 = extract_sub_block_diag(g.coarsen[1], p, K_1, :H)
    end
    refine_new = Vector{typeof(ref1)}(undef, L)
    coarsen_new = Vector{typeof(coar1)}(undef, L)
    refine_new[1] = ref1
    coarsen_new[1] = coar1
    for l in 2:L
        m, n = size(g.refine[l])
        if m == n
            refine_new[l] = extract_sub_block_diag(g.refine[l], p, 1, :V)
            coarsen_new[l] = extract_sub_block_diag(g.coarsen[l], p, 1, :H)
        else
            N_l = n ÷ p
            K_l = m ÷ (p * N_l)
            refine_new[l] = extract_sub_block_diag(g.refine[l], p, K_l, :V)
            coarsen_new[l] = extract_sub_block_diag(g.coarsen[l], p, K_l, :H)
        end
    end

    subspaces_new = Dict{Symbol, Vector{M_sub}}()
    for (key, vec) in g.subspaces
        subspaces_new[key] = Vector{M_sub}(vec)
    end

    M_op_type = valtype(operators_new)
    M_ref_type = eltype(refine_new)
    M_coar_type = eltype(coarsen_new)
    MultiGridBarrier.Geometry{T,X,W,M_op_type,M_ref_type,M_coar_type,M_sub,Disc}(
        g.discretization, g.x, g.w,
        subspaces_new, operators_new, refine_new, coarsen_new)
end

# ============================================================================
# _detect_column_structure (for post-processing)
# ============================================================================

function _detect_column_structure(D_row, p::Int, N::Int)
    nD = length(D_row)
    total_cols = size(D_row[1], 2)

    active_ranges = Vector{Tuple{Int,Int}}(undef, nD)
    for k in 1:nD
        D_cpu = SparseMatrixCSC(D_row[k])
        col_start = 0
        col_end = 0
        for j in 1:D_cpu.n
            if D_cpu.colptr[j] != D_cpu.colptr[j+1]
                if col_start == 0
                    col_start = j
                end
                col_end = j
            end
        end
        active_ranges[k] = (col_start, col_end)
    end

    boundaries = sort(unique(vcat(
        [r[1] for r in active_ranges],
        [r[2] + 1 for r in active_ranges]
    )))

    if boundaries[1] != 1
        pushfirst!(boundaries, 1)
    end
    if boundaries[end] != total_cols + 1
        push!(boundaries, total_cols + 1)
    end

    col_sizes = Int[]
    for i in 1:length(boundaries)-1
        push!(col_sizes, boundaries[i+1] - boundaries[i])
    end
    nu = length(col_sizes)

    active_cols = Vector{Int}(undef, nD)
    for k in 1:nD
        cs = active_ranges[k][1]
        offset = 0
        for b in 1:nu
            if cs <= offset + col_sizes[b]
                active_cols[k] = b
                break
            end
            offset += col_sizes[b]
        end
    end

    return col_sizes, active_cols
end
