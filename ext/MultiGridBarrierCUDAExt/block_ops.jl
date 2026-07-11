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

import MultiGridBarrier: mgb_zeros, mgb_cleanup,
                         block_batched_gemm!, block_fused_triple!,
                         block_alloc,
                         BlockDiag, BlockColumn, BlockHessian,
                         ScaledAdjBlockCol, LazyBlockHessianProduct

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

function MultiGridBarrier.block_alloc(::Type{T}, A::CuArray, dims...) where T
    CuArray{T}(undef, dims...)
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

# ============================================================================
# mgb_cleanup: flush GPU caches when solve completes
# ============================================================================

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
const _assembly_plan_cache = IdDict{Any, Vector{Tuple{Any, Any}}}()

function MultiGridBarrier.mgb_cleanup(sol::MultiGridBarrier.MGBSOL{T, <:Any, <:CuVector}) where T
    empty!(_assembly_plan_cache)
    MultiGridBarrier.clear_cudss_cache!()
    sol
end

# Throw-path variant (see device.jl): no MGBSOL exists, flush unconditionally.
function MultiGridBarrier.mgb_cleanup(::Type{MultiGridBarrier.CUDADevice})
    empty!(_assembly_plan_cache)
    MultiGridBarrier.clear_cudss_cache!()
    nothing
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
    # A real check, not an @assert: the structured assembly computes R'*H*R for
    # a single R, so silently accepting R1'*H*R2 would return a wrong matrix.
    lhp.R === R || throw(ArgumentError(
        "structured Hessian assembly computes R'*H*R for one matrix R; " *
        "got different matrices on the two sides of R'*H*R"))
    _assemble_RtHR(lhp.R, lhp.H)
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

    # Same consistency check as the CPU plan builder (BlockMatrices.jl): a
    # mismatched R must throw here, not be silently skipped row by row.
    row_offset[nu] + N * p <= nrows_R || throw(DimensionMismatch(
        "R has $nrows_R rows but the BlockHessian block layout spans $(row_offset[nu] + N * p)"))

    col_indices_cpu = Vector{Matrix{Ti}}(undef, nu)
    c_counts_cpu = Vector{Vector{Int32}}(undef, nu)
    c_max = zeros(Int, nu)

    for k in 1:nu
        element_cols = Vector{Vector{Ti}}(undef, N)
        for e in 1:N
            cols_set = Set{Ti}()
            for r in 1:p
                global_row = row_offset[k] + (e - 1) * p + r
                for idx in R_rowptr_cpu[global_row]:(R_rowptr_cpu[global_row + 1] - 1)
                    push!(cols_set, R_colval_cpu[idx])
                end
            end
            element_cols[e] = sort!(collect(cols_set))
        end
        c_max[k] = maximum(length(ec) for ec in element_cols; init=0)
        if c_max[k] == 0
            c_max[k] = 1  # avoid zero-size arrays (mirrors the CPU builder)
        end

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
        return AssemblyPlan{T, Ti}(
            out_rowptr, out_colval, ncols_R, ncols_R, out_nnz,
            panels_gpu, scatter_idx, p, N, nu, c_max)
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

    AssemblyPlan{T, Ti}(
        CuVector{Ti}(out_rowptr_cpu), CuVector{Ti}(out_colval_cpu),
        ncols_R, ncols_R, out_nnz,
        panels_gpu, scatter_idx, p, N, nu, c_max)
end

function _get_assembly_plan(R::CuSparseMatrixCSR{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
    nu = size(H.blocks, 1)
    meta = (H.p, H.N, H.block_sizes,
            [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing])
    entries = get!(Vector{Tuple{Any, Any}}, _assembly_plan_cache, R)
    for (m, plan) in entries
        m == meta && return plan::AssemblyPlan{T, Ti}
    end
    plan = _make_assembly_plan(R, H)
    push!(entries, (meta, plan))
    return plan::AssemblyPlan{T, Ti}
end

function _assemble_RtHR(R::CuSparseMatrixCSR{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
    # Function barrier: the cache hands back an abstractly-typed plan; dispatch
    # on the concrete AssemblyPlan specializes the assembly body.
    _assemble_RtHR_impl(_get_assembly_plan(R, H), H)
end

function _assemble_RtHR_impl(plan::AssemblyPlan{T, Ti}, H::CuBlockHessian{T}) where {T, Ti}
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
    first_blk === nothing &&
        error("internal: assembly plan is non-empty but H has no blocks")

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
# mgb_zeros for BlockColumn (CuArray-backed)
# ============================================================================

MultiGridBarrier.mgb_zeros(::CuBlockColumn{T}, m, n) where {T} =
    _cu_spzeros(T, m, n)

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

# (Sparse-fallback `*` for bare CuBlockDiag operands, and the
# extract_block_diag / extract_block_column / _detect_column_structure
# extractors, were removed: the live GPU solve never constructs or multiplies a
# bare CuBlockDiag — structured operators arrive as CuBlockColumn (D_fine) or
# CuSparseMatrixCSR (R_fine), already built on the CPU before transfer.)
