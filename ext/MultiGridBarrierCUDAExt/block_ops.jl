# block_ops.jl -- Operations on structured block matrix types for GPU Hessian assembly
#
# Implements the dispatch chain for f2 in AlgebraicMultiGridBarrier.jl:
#   (1) amgb_diag(D[1]::BlockColumnOp, v) → Diagonal(v)
#   (2a) adjoint(D[j]::BlockColumnOp) * Diagonal(v) → ScaledAdjBlockCol
#   (2b) ScaledAdjBlockCol * D[k]::BlockColumnOp → BlockHessianGPU
#   (3) BlockHessianGPU + BlockHessianGPU → BlockHessianGPU
#   (5) R' * BlockHessianGPU * R → CuSparseMatrixCSR (via sparse conversion + SpGEMM)

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

import MultiGridBarrier: amgb_diag, amgb_zeros, amgb_blockdiag, apply_D, amgb_cleanup

# ============================================================================
# Block-level GPU operations
# ============================================================================

"""
    _block_diag_scale_kernel!(out, data, v, p, q, N)

CUDA kernel: scale rows of each block by diagonal entries.
out[:,:,i] = diag(v_block_i) * data[:,:,i]
where v_block_i = v[(i-1)*p+1 : i*p]
"""
function _block_diag_scale_kernel!(out, data, v, p, q, N)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = p * q * N
    if idx <= total
        # Decode: idx → (r, c, blk) in p × q × N
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

"""
    diag_scale(v::CuVector, B::BlockDiagGPU) → BlockDiagGPU

Compute diag(v) * B: scale row k of each block by v's corresponding entry.
v must have length p*N (one entry per row of the full block-diagonal matrix).
"""
function diag_scale(v::CuVector{T}, B::BlockDiagGPU{T}) where T
    @assert length(v) == B.p * B.N
    out = similar(B.data)
    total = B.p * B.q * B.N
    kernel = @cuda launch=false _block_diag_scale_kernel!(out, B.data, v, B.p, B.q, B.N)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, B.data, v, B.p, B.q, B.N; threads=threads, blocks=blocks)
    BlockDiagGPU{T}(B.p, B.q, B.N, out)
end

"""
    _batched_gemm_kernel!(C, A, B, transA, p, q, r, N)

CUDA kernel for batched matrix multiply: C[:,:,i] = op(A[:,:,i]) * B[:,:,i]
where op = transpose if transA, identity otherwise.

A is p×q×N (or q×p×N if transA), B is q×r×N (or p×r×N if transA), C is result.
When transA: C[i,j,blk] = sum_k A[k,i,blk] * B[k,j,blk]  (result is q×r)
When !transA: C[i,j,blk] = sum_k A[i,k,blk] * B[k,j,blk]  (result is p×r)
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
    block_mul(A::BlockDiagGPU, B::BlockDiagGPU; transA=false) → BlockDiagGPU

Batched block multiply: result[i] = op(A[i]) * B[i] for each block.
If transA: op = transpose, result is q_A × q_B × N.
If !transA: op = identity, result is p_A × q_B × N.
"""
function block_mul(A::BlockDiagGPU{T}, B::BlockDiagGPU{T}; transA::Bool=false) where T
    @assert A.N == B.N
    if transA
        @assert A.p == B.p  # inner dimension matches
        rows_C = A.q
        inner = A.p
    else
        @assert A.q == B.p
        rows_C = A.p
        inner = A.q
    end
    cols_C = B.q
    C_data = CuArray{T}(undef, rows_C, cols_C, A.N)
    total = rows_C * cols_C * A.N
    kernel = @cuda launch=false _batched_gemm_kernel!(C_data, A.data, B.data, Val(transA), rows_C, cols_C, inner, A.N)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(C_data, A.data, B.data, Val(transA), rows_C, cols_C, inner, A.N; threads=threads, blocks=blocks)
    BlockDiagGPU{T}(rows_C, cols_C, A.N, C_data)
end

"""
    _fused_triple_kernel!(C, A, v, B, rows_C, cols_C, inner, N)

Fused kernel for A' * diag(v) * B.
C[i,j,blk] = sum_k A[k,i,blk] * v[(blk-1)*inner + k] * B[k,j,blk]

Eliminates the intermediate diag(v)*B allocation and kernel launch.
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
    block_triple_product(A::BlockDiagGPU, v::CuVector, B::BlockDiagGPU) → BlockDiagGPU

Compute A' * diag(v) * B block-by-block via fused kernel (no intermediate allocation).
Result: q_A × q_B × N.
"""
function block_triple_product(A::BlockDiagGPU{T}, v::CuVector{T}, B::BlockDiagGPU{T}) where T
    @assert A.N == B.N
    @assert A.p == B.p
    @assert length(v) == A.p * A.N
    rows_C = A.q
    cols_C = B.q
    inner = A.p
    N = A.N
    C_data = CuArray{T}(undef, rows_C, cols_C, N)
    total = rows_C * cols_C * N
    kernel = @cuda launch=false _fused_triple_kernel!(C_data, A.data, v, B.data, rows_C, cols_C, inner, N)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(C_data, A.data, v, B.data, rows_C, cols_C, inner, N; threads=threads, blocks=blocks)
    BlockDiagGPU{T}(rows_C, cols_C, N, C_data)
end

# ============================================================================
# Dispatch chain for f2
# ============================================================================

# (1) amgb_diag(D[1]::BlockColumnOp, v::CuVector) → Diagonal(v)
# Returns LinearAlgebra.Diagonal wrapping the CuVector — no sparse matrix construction.
MultiGridBarrier.amgb_diag(::BlockColumnOp{T}, z::CuVector{T}, m=length(z), n=length(z)) where {T} =
    Diagonal(z)

# (2a) adjoint(D[j]::BlockColumnOp) * Diagonal(v) → ScaledAdjBlockCol
# Lazy: just captures operands for the next multiply.
# Note: CuVector{T} = CuArray{T,1} but actual runtime type includes memory parameter
# (CuArray{T,1,Mem}), so we need <:CuVector{T} to match.
function Base.:*(A::Adjoint{T,BlockColumnOp{T}}, D::Diagonal{T,<:CuVector{T}}) where T
    ScaledAdjBlockCol{T}(parent(A), CuVector{T}(D.diag))
end

# (2b) ScaledAdjBlockCol * D[k]::BlockColumnOp → BlockHessianGPU
# THIS is where the real computation happens.
function Base.:*(A::ScaledAdjBlockCol{T}, B::BlockColumnOp{T}) where T
    op_j = A.op       # D[j]
    op_k = B           # D[k]
    v = A.diag         # diagonal scaling vector

    @assert op_j.nu == op_k.nu
    @assert op_j.active_block.N == op_k.active_block.N
    @assert op_j.active_block.p == op_k.active_block.p  # same row count = same mesh

    # Compute the triple product for the active blocks
    result_block = block_triple_product(op_j.active_block, v, op_k.active_block)

    # Build BlockHessianGPU with this single block at position (active_col_j, active_col_k)
    nu = op_j.nu
    blocks = Matrix{Union{BlockDiagGPU{T}, Nothing}}(nothing, nu, nu)
    blocks[op_j.active_col, op_k.active_col] = result_block

    BlockHessianGPU{T}(blocks, result_block.p, result_block.N, op_j.col_sizes)
end

# (3) BlockHessianGPU + BlockHessianGPU → BlockHessianGPU
function Base.:+(A::BlockHessianGPU{T}, B::BlockHessianGPU{T}) where T
    @assert size(A.blocks) == size(B.blocks)
    @assert A.p == B.p && A.N == B.N
    nu = size(A.blocks, 1)
    blocks = Matrix{Union{BlockDiagGPU{T}, Nothing}}(nothing, nu, nu)
    for j in 1:nu, k in 1:nu
        a = A.blocks[j, k]
        b = B.blocks[j, k]
        if a === nothing && b === nothing
            # both nothing — leave as nothing
        elseif a === nothing
            blocks[j, k] = b
        elseif b === nothing
            blocks[j, k] = a
        else
            # In-place addition: accumulate into a's data to avoid allocation
            a.data .+= b.data
            blocks[j, k] = a
        end
    end
    BlockHessianGPU{T}(blocks, A.p, A.N, A.block_sizes)
end

# ============================================================================
# Sparse conversion: BlockHessianGPU → CuSparseMatrixCSR
# ============================================================================

"""
    SparseConversionPlan{Ti}

Cached sparse conversion plan for converting BlockHessianGPU to CuSparseMatrixCSR.
The CSR structure (rowptr, colval) depends only on block sizes and which blocks
are nonzero — fixed across iterations. Only nzval changes.
"""
struct SparseConversionPlan{Ti}
    rowptr::CuVector{Ti}
    colval::CuVector{Ti}
    total_nnz::Int
    total_rows::Int
    total_cols::Int
    # For each CSR nonzero: which block (bi, bj), local row, local col
    # Encoded as: source_block_idx (into list of nonzero blocks),
    #             source_offset within that block's data array
    scatter_block::CuVector{Int32}    # which nonzero-block index
    scatter_offset::CuVector{Int32}   # offset within block's data (column-major: r + (c-1)*p)
    scatter_element::CuVector{Int32}  # element index for each nonzero (for GPU scatter)
    # Block mapping: nonzero_block_list[i] → (bi, bj) position
    block_bi::Vector{Int}
    block_bj::Vector{Int}
end

"""
    make_sparse_plan(H::BlockHessianGPU{T}; Ti=Int32) → SparseConversionPlan

Build a cached conversion plan from the block structure. Called once per
sparsity pattern; reused for all subsequent conversions.
"""
function make_sparse_plan(H::BlockHessianGPU{T}; Ti=Int32) where T
    nu = size(H.blocks, 1)
    p = H.p
    N = H.N
    block_sizes = H.block_sizes

    # Enumerate nonzero blocks
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

    # Build CSR structure on CPU, then transfer to GPU
    # Each block (bi, bj) contributes: for each element e, a dense p×q sub-block
    # at rows [(bi-1)*p*N + (e-1)*p + 1 : ...], cols [(bj-1)*q*N + (e-1)*q + 1 : ...]
    # But the blocks are interleaved: row i in the full matrix corresponds to
    # block-row-group bi, element e, local row r such that:
    #   global_row = row_offset[bi] + (e-1)*p + r  where row_offset[bi] = sum(block_sizes[1:bi-1])

    # For BlockDiagGPU, all blocks are p×p (for Hessian), N elements
    # Row ordering: block-group bi=1 has rows 1..block_sizes[1],
    #   within that, element e=1 has rows 1..p, e=2 has rows p+1..2p, etc.

    row_offset = zeros(Int, nu)
    for bi in 2:nu
        row_offset[bi] = row_offset[bi-1] + block_sizes[bi-1]
    end
    col_offset = row_offset  # symmetric for Hessian

    # Build rowptr, colval, and scatter maps
    # For each row, collect all nonzero columns from all nonzero blocks
    rowptr_cpu = Vector{Ti}(undef, total_rows + 1)
    colval_list = Ti[]
    scatter_block_list = Int32[]
    scatter_offset_list = Int32[]

    rowptr_cpu[1] = Ti(1)
    for bi in 1:nu
        for e in 1:N
            for r in 1:p
                global_row = row_offset[bi] + (e - 1) * p + r
                # For this row, find all nonzero blocks in block-row bi
                entries = Tuple{Ti, Int, Int32}[]  # (col, nz_idx, offset)
                for (nz_idx, (nbi, nbj)) in enumerate(zip(nz_bi, nz_bj))
                    if nbi == bi
                        blk = H.blocks[nbi, nbj]
                        q_blk = blk.q
                        for c in 1:q_blk
                            global_col = Ti(col_offset[nbj] + (e - 1) * q_blk + c)
                            # offset in data array (column-major): r + (c-1)*p within element e
                            # But data is p × q × N, so data[r, c, e]
                            # For scatter: we need the linear index into data[:,:,e]
                            # = r + (c-1)*p (1-indexed, within the element)
                            offset = Int32(r + (c - 1) * p)
                            push!(entries, (global_col, nz_idx, offset))
                        end
                    end
                end
                # Sort by column for CSR
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

    # Pre-compute element index for each nonzero (for GPU scatter kernel)
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

"""
    _scatter_nzval_kernel!(nzval, combined, scatter_block, scatter_offset,
                           scatter_element, p, N, total_nnz)

CUDA kernel: fill CSR nzval from combined block data using the scatter map.
For each nonzero idx:
  - scatter_block[idx]: which nonzero-block (1-indexed)
  - scatter_offset[idx]: r + (c-1)*p within block
  - scatter_element[idx]: element index (1-indexed)
  - combined[r, c, (blk_idx-1)*N + e]
"""
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

"""
    to_cusparse(H::BlockHessianGPU, plan::SparseConversionPlan) → CuSparseMatrixCSR

Convert BlockHessianGPU to CuSparseMatrixCSR using cached plan.
Only fills nzval; rowptr and colval come from the plan.
Uses a GPU scatter kernel — no CPU round-trip.
"""
function to_cusparse(H::BlockHessianGPU{T}, plan::SparseConversionPlan{Ti}) where {T, Ti}
    p = H.p
    N = H.N
    n_nz_blocks = length(plan.block_bi)

    # Stack all nonzero block data into a single 3D array on GPU
    combined = CuArray{T}(undef, p, p, n_nz_blocks * N)
    for (nz_idx, (bi, bj)) in enumerate(zip(plan.block_bi, plan.block_bj))
        blk = H.blocks[bi, bj]
        combined[:, :, (nz_idx-1)*N+1 : nz_idx*N] = blk.data
    end

    # GPU scatter: fill nzval from combined using pre-computed maps
    total_nnz = plan.total_nnz
    nzval = CuVector{T}(undef, total_nnz)

    kernel = @cuda launch=false _scatter_nzval_kernel!(
        nzval, combined, plan.scatter_block, plan.scatter_offset,
        plan.scatter_element, Int32(p), Int32(N), Int32(total_nnz))
    config = launch_configuration(kernel.fun)
    threads = min(total_nnz, config.threads)
    blocks = cld(total_nnz, threads)
    kernel(nzval, combined, plan.scatter_block, plan.scatter_offset,
           plan.scatter_element, Int32(p), Int32(N), Int32(total_nnz);
           threads=threads, blocks=blocks)

    CuSparseMatrixCSR{T}(plan.rowptr, plan.colval, nzval,
                          (plan.total_rows, plan.total_cols))
end

# Cache for sparse conversion plans
const _sparse_plan_cache = Dict{UInt64, Any}()

function _get_sparse_plan(H::BlockHessianGPU{T}; Ti=Int32) where T
    # Cache key: block structure (which positions are nonzero, sizes)
    nu = size(H.blocks, 1)
    key = hash((nu, H.p, H.N, H.block_sizes,
                [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing]))
    get!(_sparse_plan_cache, key) do
        make_sparse_plan(H; Ti=Ti)
    end::SparseConversionPlan{Ti}
end

# (5a) Adjoint{CuSparse} * BlockHessianGPU → LazyHessianProduct (no work)
# R' * H where H is BlockHessianGPU — just capture for later assembly
function Base.:*(A::Adjoint{T, CuSparseMatrixCSR{T,Ti}}, H::BlockHessianGPU{T}) where {T, Ti}
    LazyHessianProduct{T,Ti}(parent(A), H)
end

# (5b) LazyHessianProduct * CuSparse → CuSparseMatrixCSR (element-wise assembly)
# (R' * H) * R computed via _assemble_RtHR
function Base.:*(lhp::LazyHessianProduct{T,Ti}, R::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    @assert lhp.R === R "LazyHessianProduct expects same R on both sides"
    _assemble_RtHR(lhp.R, lhp.H)
end

# Fallback: BlockHessianGPU * CuSparse → CuSparseMatrixCSR (for other usages)
function Base.:*(H::BlockHessianGPU{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    plan = _get_sparse_plan(H; Ti=Ti)
    H_sparse = to_cusparse(H, plan)
    return H_sparse * B
end

# ============================================================================
# Element-wise assembly: R' * BlockHessianGPU * R without SPGEMM
# ============================================================================

"""
    AssemblyPlan{T, Ti}

Cached plan for computing R' * H * R via element-wise assembly.
Precomputes R panels (dense sub-blocks of R per element), output CSR pattern,
and scatter maps. R is fixed across Newton iterations; only H changes.
"""
struct AssemblyPlan{T, Ti}
    # Output CSR structure
    out_rowptr::CuVector{Ti}
    out_colval::CuVector{Ti}
    out_m::Int          # output rows = ncols(R)
    out_n::Int          # output cols = ncols(R)
    out_nnz::Int

    # Per block k: dense panels of R, shape (p, c_max_k, N)
    # panels[k][r, a, e] = R[row_offset[k] + (e-1)*p + r, col_indices[k][a, e]]
    panels::Vector{CuArray{T, 3}}

    # Per block k: column indices, shape (c_max_k, N)
    # col_indices[k][a, e] = the a-th distinct column index for element e in block k
    col_indices::Vector{CuArray{Ti, 2}}

    # Per block k: actual column count per element, shape (N,)
    c_counts::Vector{CuVector{Int32}}

    # Per block pair (i,j): scatter map, shape (c_max_i, c_max_j, N)
    # scatter_idx[(i,j)][a, b, e] = 1-based index into output nzval
    scatter_idx::Dict{Tuple{Int,Int}, CuArray{Int32, 3}}

    # Block partitioning
    p::Int              # element block size
    N::Int              # number of elements
    nu::Int             # number of block groups
    c_max::Vector{Int}  # max columns per block
end

# Cache for assembly plans
const _assembly_plan_cache = Dict{UInt64, Any}()

# Flush all GPU caches when solve completes (dispatches on CuVector in AMGBSOL)
function MultiGridBarrier.amgb_cleanup(sol::MultiGridBarrier.AMGBSOL{T, <:Any, <:CuVector}) where T
    empty!(_sparse_plan_cache)
    empty!(_assembly_plan_cache)
    MultiGridBarrier.clear_cudss_cache!()
    sol
end

"""
    _make_assembly_plan(R::CuSparseMatrixCSR, H::BlockHessianGPU) → AssemblyPlan

Build a cached assembly plan from R's sparsity pattern and H's block structure.
Executed once per sparsity pattern; reused for all subsequent Newton iterations.
"""
function _make_assembly_plan(R::CuSparseMatrixCSR{T, Ti}, H::BlockHessianGPU{T}) where {T, Ti}
    nu = size(H.blocks, 1)
    p = H.p
    N = H.N
    block_sizes = H.block_sizes

    # Transfer R to CPU for plan construction
    R_rowptr_cpu = Array(R.rowPtr)
    R_colval_cpu = Array(R.colVal)
    R_nzval_cpu = Array(R.nzVal)
    nrows_R = size(R, 1)
    ncols_R = size(R, 2)

    # Row offsets for each block group
    row_offset = zeros(Int, nu)
    for k in 2:nu
        row_offset[k] = row_offset[k-1] + block_sizes[k-1]
    end

    # ---- Step 1: Extract column indices per block per element ----
    # For each block k and element e, find the distinct nonzero column indices in R
    col_indices_cpu = Vector{Matrix{Ti}}(undef, nu)
    c_counts_cpu = Vector{Vector{Int32}}(undef, nu)
    c_max = zeros(Int, nu)

    for k in 1:nu
        # First pass: find c_max for block k
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

        # Build padded column index matrix (c_max_k x N)
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

    # ---- Step 2: Extract dense R panels per block ----
    panels_cpu = Vector{Array{T, 3}}(undef, nu)
    for k in 1:nu
        panel = zeros(T, p, c_max[k], N)
        ci = col_indices_cpu[k]
        for e in 1:N
            nc = c_counts_cpu[k][e]
            # Build column→local_index map for this element
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

    # ---- Step 3: Build output CSR sparsity pattern ----
    # For each block pair (i,j), element e contributes a c_i x c_j dense block
    # at rows = col_indices[i][:,e], cols = col_indices[j][:,e]
    # Collect all (row, col) pairs, deduplicate, build CSR

    # Use SparseArrays to build the pattern
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

    # Build CSR via SparseArrays (deduplicate by constructing a sparse matrix)
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

    # Build CSR sparsity via SparseArrays: construct CSC, then get CSR via transpose
    # sparse() sums duplicates, which is fine — we only need the nonzero pattern
    indicator = sparse(out_rows, out_cols, ones(Float32, length(out_rows)), ncols_R, ncols_R)
    # CSC of A' has the same structure as CSR of A
    S_t = SparseMatrixCSC(indicator')
    out_rowptr_cpu = Vector{Ti}(S_t.colptr)
    out_colval_cpu = Vector{Ti}(S_t.rowval)
    out_nnz = length(out_colval_cpu)

    # ---- Step 4: Build scatter maps ----
    # For each block pair (i,j), element e, local entry (a, b):
    #   output_row = col_indices[i][a, e], output_col = col_indices[j][b, e]
    #   find the nzval index via binary search in output CSR

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
                # Get the range in CSR for this row
                rs = Int(out_rowptr_cpu[row])
                re = Int(out_rowptr_cpu[row + 1]) - 1
                row_cols = @view out_colval_cpu[rs:re]
                for b in 1:cc_j[e]
                    col = ci_j[b, e]
                    # Binary search for col in row_cols
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

    # Transfer to GPU
    panels_gpu = [CuArray{T}(panels_cpu[k]) for k in 1:nu]
    col_indices_gpu = [CuArray{Ti}(col_indices_cpu[k]) for k in 1:nu]
    c_counts_gpu = [CuVector{Int32}(c_counts_cpu[k]) for k in 1:nu]

    AssemblyPlan{T, Ti}(
        CuVector{Ti}(out_rowptr_cpu), CuVector{Ti}(out_colval_cpu),
        ncols_R, ncols_R, out_nnz,
        panels_gpu, col_indices_gpu, c_counts_gpu,
        scatter_idx, p, N, nu, c_max)
end

function _get_assembly_plan(R::CuSparseMatrixCSR{T, Ti}, H::BlockHessianGPU{T}) where {T, Ti}
    nu = size(H.blocks, 1)
    # Cache key uses R's object identity (pointer) + H's block structure.
    # R is fixed per multigrid level within a solve, so objectid is stable.
    # If R is reconstructed, the plan will be recomputed (correct, just slower).
    key = hash((objectid(R), H.p, H.N, H.block_sizes,
                [(i,j) for i in 1:nu for j in 1:nu if H.blocks[i,j] !== nothing]))
    get!(_assembly_plan_cache, key) do
        _make_assembly_plan(R, H)
    end::AssemblyPlan{T, Ti}
end

# ============================================================================
# GPU kernels for element-wise assembly (two-phase approach)
# ============================================================================

"""
    _dot_scatter_kernel!(output_nzval, panel_i, tmp, scatter, c_max_i, c_max_j, p, N)

Phase 2 kernel: for each (a, b, e), compute:
    val = sum_{r=1}^{p} panel_i[r, a, e] * tmp[r, b, e]
Then atomicAdd val to output_nzval[scatter[a, b, e]].

Each thread computes one (a, b, e) triple — just a p-length dot product.
"""
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

"""
    _assemble_RtHR(R, H) → CuSparseMatrixCSR

Compute R' * H * R via element-wise assembly. Uses cached assembly plan.
Two-phase approach per block pair (i,j):
  Phase 1: tmp = H.blocks[i,j] * panel_j  (batched GEMM, reuses existing kernel)
  Phase 2: panel_i' * tmp → scatter to output  (dot + atomicAdd)
"""
function _assemble_RtHR(R::CuSparseMatrixCSR{T, Ti}, H::BlockHessianGPU{T}) where {T, Ti}
    plan = _get_assembly_plan(R, H)

    if plan.out_nnz == 0
        return CuSparseMatrixCSR{T}(plan.out_rowptr, plan.out_colval,
                                     CuVector{T}(undef, 0), (plan.out_m, plan.out_n))
    end

    # Allocate and zero output nzval
    output_nzval = CUDA.zeros(T, plan.out_nnz)

    nu = plan.nu
    p = plan.p
    N = plan.N

    # Pre-allocate tmp buffer with max c_max to avoid repeated GPU malloc
    cm_max = maximum(plan.c_max)
    tmp = CuArray{T}(undef, p, cm_max, N)

    # Find first non-null block for kernel compilation
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

    # Cache kernel configs (one per unique kernel signature — avoid repeated
    # @cuda launch=false + launch_configuration in the inner loop)
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

        # Phase 1: tmp[r, b, e] = sum_s H[r, s, e] * panel_j[s, b, e]
        # tmp is pre-allocated as p × cm_max × N; kernel only accesses 1:cmj columns
        total_gemm = p * cmj * N
        gemm_threads = min(total_gemm, gemm_max_threads)
        gemm_blocks = cld(total_gemm, gemm_threads)
        gemm_kern(tmp, blk.data, panel_j, Val(false), Int32(p), Int32(cmj), Int32(p), Int32(N);
                  threads=gemm_threads, blocks=gemm_blocks)

        # Phase 2: dot + scatter: output[scatter[a,b,e]] += panel_i[:,a,e]' * tmp[:,b,e]
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
# BlockColumnOp * vector (for apply_D)
# ============================================================================

"""
    _block_matvec_kernel!(out, data, v, p, q, N, offset)

CUDA kernel: batched block matrix-vector multiply.
out[(blk-1)*p + r] = sum_c data[r, c, blk] * v[offset + (blk-1)*q + c]
"""
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

function Base.:*(A::BlockColumnOp{T}, z::CuVector{T}) where T
    # z has length sum(col_sizes) = total columns of the full D[L,k] operator
    # Extract the column block at active_col and multiply by active_block
    col_offset = sum(A.col_sizes[1:A.active_col-1])
    p = A.active_block.p
    q = A.active_block.q
    N = A.active_block.N
    out = CUDA.zeros(T, A.total_rows)
    total = p * N
    kernel = @cuda launch=false _block_matvec_kernel!(out, A.active_block.data, z, p, q, N, col_offset)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks_count = cld(total, threads)
    kernel(out, A.active_block.data, z, p, q, N, col_offset;
           threads=threads, blocks=blocks_count)
    out
end

"""
    _block_adj_matvec_kernel!(out, data, v, p, q, N, col_offset)

CUDA kernel: batched adjoint block matrix-vector multiply.
out[col_offset + (blk-1)*q + c] = sum_r data[r, c, blk] * v[(blk-1)*p + r]
This computes D[k]' * v where D[k] is BlockColumnOp.
"""
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

# adjoint(BlockColumnOp) * CuVector → CuVector
# Computes D[k]' * v. Result has length sum(col_sizes).
# Only the active_col block is nonzero, the rest are zero.
function Base.:*(A::Adjoint{T,BlockColumnOp{T}}, z::CuVector{T}) where T
    op = parent(A)
    col_offset = sum(op.col_sizes[1:op.active_col-1])
    p = op.active_block.p
    q = op.active_block.q
    N = op.active_block.N
    out = CUDA.zeros(T, sum(op.col_sizes))
    total = q * N
    kernel = @cuda launch=false _block_adj_matvec_kernel!(out, op.active_block.data, z, p, q, N, col_offset)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks_count = cld(total, threads)
    kernel(out, op.active_block.data, z, p, q, N, col_offset;
           threads=threads, blocks=blocks_count)
    out
end

# ============================================================================
# amgb_zeros for BlockColumnOp, BlockDiagGPU, SubBlockDiagGPU
# ============================================================================

MultiGridBarrier.amgb_zeros(::BlockColumnOp{T}, m, n) where {T} =
    _cu_spzeros(T, m, n)
MultiGridBarrier.amgb_zeros(::Adjoint{T, BlockColumnOp{T}}, m, n) where {T} =
    _cu_spzeros(T, m, n)

# For BlockDiagGPU/SubBlockDiagGPU: return a zero BlockDiagGPU with matching block size.
# This ensures [Z for j=1:nu] creates Vector{BlockDiagGPU{T}}, so assigning the
# triple-product result (also BlockDiagGPU{T}) works without type mismatch.
function MultiGridBarrier.amgb_zeros(A::BlockDiagGPU{T}, m, n) where {T}
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    BlockDiagGPU{T}(A.p, A.p, N, CUDA.zeros(T, A.p, A.p, N))
end

function MultiGridBarrier.amgb_zeros(A::SubBlockDiagGPU{T}, m, n) where {T}
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    BlockDiagGPU{T}(A.p, A.p, N, CUDA.zeros(T, A.p, A.p, N))
end

# ============================================================================
# hcat for constructing BlockColumnOp from D0 in amg_helper
# ============================================================================

# D0[l,k] = hcat(Z, ..., Op_k, ..., Z) where:
#   - Z is a zero BlockDiagGPU (from amgb_zeros on structured types)
#   - Op_k is a non-zero BlockDiagGPU (the coarsened operator)
#
# We detect the block-column pattern (exactly one non-zero BlockDiagGPU)
# and return a BlockColumnOp.

"""
    hcat(args::BlockDiagGPU{T}...) → BlockColumnOp

Variadic hcat for BlockDiagGPU. Detects the block-column pattern where
exactly one arg has non-zero data and builds a BlockColumnOp.
"""
function Base.hcat(args::BlockDiagGPU{T}...) where T
    # Find the active (non-zero) block via GPU reduction
    block_idx = 0
    for (i, a) in enumerate(args)
        if sum(abs2, a.data) > zero(T)
            if block_idx != 0
                # Multiple non-zero: can't form BlockColumnOp, fall back to sparse
                return hcat((_to_cusparse(a) for a in args)...)
            end
            block_idx = i
        end
    end
    if block_idx == 0
        # All zeros — fall back to sparse
        return hcat((_to_cusparse(a) for a in args)...)
    end
    blk = args[block_idx]
    nu = length(args)
    col_sizes = [size(a, 2) for a in args]
    total_rows = size(blk, 1)
    BlockColumnOp{T}(blk, block_idx, nu, col_sizes, total_rows)
end

# 2-arg hcat methods for BlockDiagGPU + CuSparseMatrixCSR (kept for compatibility)
function Base.hcat(A::BlockDiagGPU{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _hcat_mixed(T, Ti, Any[A, B])
end

function Base.hcat(A::CuSparseMatrixCSR{T,Ti}, B::BlockDiagGPU{T}) where {T, Ti}
    _hcat_mixed(T, Ti, Any[A, B])
end

function _hcat_block_column(args::Vector)
    block_idx = 0
    T_elem = nothing
    for (i, a) in enumerate(args)
        if a isa BlockDiagGPU
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
    blk = args[block_idx]::BlockDiagGPU
    nu = length(args)
    col_sizes = [size(a, 2) for a in args]
    total_rows = size(blk, 1)
    BlockColumnOp{T_elem}(blk, block_idx, nu, col_sizes, total_rows)
end

function _hcat_mixed(::Type{T}, ::Type{Ti}, args::Vector) where {T, Ti}
    result = _hcat_block_column(args)
    if result !== nothing
        return result
    end
    sparse_args = [a isa BlockDiagGPU ? _to_cusparse(a) : a for a in args]
    hcat(sparse_args...)
end

# ============================================================================
# BlockDiagGPU → CuSparseMatrixCSR conversion
# ============================================================================

"""
    _to_cusparse(B::BlockDiagGPU{T}) → CuSparseMatrixCSR{T,Int32}

Convert a BlockDiagGPU to a standard CuSparseMatrixCSR.
"""
function _to_cusparse(B::BlockDiagGPU{T}) where T
    p, q, N = B.p, B.q, B.N
    m = p * N
    n = q * N

    # Build CSR on CPU
    data_cpu = Array(B.data)
    nnz_total = 0
    # Count nonzeros
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

# ============================================================================
# CuSparseMatrixCSR * BlockDiagGPU → CuSparseMatrixCSR
# For coarser levels: coarsen_fine[l] * operators[k] * refine_fine[l]
# where operators[k] is BlockDiagGPU but coarsen/refine are CuSparse
# ============================================================================

function Base.:*(A::CuSparseMatrixCSR{T,Ti}, B::BlockDiagGPU{T}) where {T, Ti}
    B_sparse = _to_cusparse(B)
    A * B_sparse
end

function Base.:*(A::BlockDiagGPU{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    A_sparse = _to_cusparse(A)
    A_sparse * B
end

# adjoint * CuSparse for BlockDiagGPU (needed for D[j]' * foo when D[j] falls back)
function Base.:*(A::Adjoint{T,BlockDiagGPU{T}}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _to_cusparse(parent(A))' * B
end

# ============================================================================
# Extraction: CuSparseMatrixCSR → BlockDiagGPU (post-processing)
# ============================================================================

"""
    extract_block_diag(A::CuSparseMatrixCSR{T}, p::Int) → BlockDiagGPU{T}

Extract block-diagonal structure from a CuSparseMatrixCSR that is known to be
block-diagonal with p×p dense blocks. The matrix must be (p*N) × (p*N).

This is used to post-process the D[L,:] matrices after amg_helper builds them.
At the finest level, operators like dx, dy are block-diagonal with element blocks.
"""
function extract_block_diag(A::CuSparseMatrixCSR{T,Ti}, p::Int) where {T, Ti}
    m, n = size(A)
    @assert m % p == 0 "Matrix rows ($m) not divisible by block size ($p)"
    N = m ÷ p
    # For block-diagonal: each block is p×q where q may differ from p
    # But for operators, they're square: q = n ÷ N
    @assert n % N == 0 "Matrix cols ($n) not divisible by block count ($N)"
    q = n ÷ N

    # Transfer to CPU and extract blocks
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

    BlockDiagGPU{T}(p, q, N, CuArray{T}(data_cpu))
end

"""
    extract_block_column(D_entry::CuSparseMatrixCSR{T}, p::Int, nu::Int,
                         col_sizes::Vector{Int}) → BlockColumnOp{T}

Extract BlockColumnOp from a D[L,k] entry that is of the form hcat(Z,...,Op,...,Z).
Detects which column block is the active (non-zero) one and extracts its block-diagonal
structure.

Arguments:
- D_entry: The CuSparseMatrixCSR matrix for D[L,k]
- p: Block size (e.g., 7 for fem2d)
- nu: Number of column blocks (state variables)
- col_sizes: Size of each column block
"""
function extract_block_column(D_entry::CuSparseMatrixCSR{T,Ti}, p::Int, nu::Int,
                               col_sizes::Vector{Int}) where {T, Ti}
    m = size(D_entry, 1)  # total rows
    total_cols = sum(col_sizes)
    @assert size(D_entry, 2) == total_cols

    # Find the active column block by checking which has nonzeros
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

    # Extract the active block as BlockDiagGPU
    c_start = col_offsets[active_col] + 1
    c_end = col_offsets[active_col + 1]
    active_sparse = D_cpu[:, c_start:c_end]

    # Convert back to CuSparse for extract_block_diag
    active_cu = CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(
        active_sparse.m, active_sparse.n,
        Int32.(active_sparse.colptr), Int32.(active_sparse.rowval),
        active_sparse.nzval))
    active_block = extract_block_diag(active_cu, p)

    BlockColumnOp{T}(active_block, active_col, nu, col_sizes, m)
end

# Old structurize_amg has been replaced by _structurize_geometry (called by
# native_to_cuda when structured=true), which converts the Geometry's
# operators/refine/coarsen to block types before amg_helper runs.

"""
    _detect_column_structure(D_row, p, N) → (col_sizes, active_cols)

Analyze a row of D matrices to detect the column block structure.
Each D[L,k] has one active column block (non-zero). By scanning all k entries,
we determine the column block sizes and which column each k activates.
"""
function _detect_column_structure(D_row, p::Int, N::Int)
    nD = length(D_row)
    total_cols = size(D_row[1], 2)

    # For each D[L,k], find the column range with nonzeros
    active_ranges = Vector{Tuple{Int,Int}}(undef, nD)
    for k in 1:nD
        D_cpu = SparseMatrixCSC(D_row[k])
        # Find first and last nonzero column
        col_start = 0
        col_end = 0
        for j in 1:D_cpu.n
            if D_cpu.colptr[j] != D_cpu.colptr[j+1]  # column j has nonzeros
                if col_start == 0
                    col_start = j
                end
                col_end = j
            end
        end
        active_ranges[k] = (col_start, col_end)
    end

    # Collect all unique column block boundaries
    # Sort by start column to determine blocks
    boundaries = sort(unique(vcat(
        [r[1] for r in active_ranges],
        [r[2] + 1 for r in active_ranges]
    )))

    # Add 1 as start if not present and total_cols+1 as end
    if boundaries[1] != 1
        pushfirst!(boundaries, 1)
    end
    if boundaries[end] != total_cols + 1
        push!(boundaries, total_cols + 1)
    end

    # Build col_sizes from boundaries
    col_sizes = Int[]
    for i in 1:length(boundaries)-1
        push!(col_sizes, boundaries[i+1] - boundaries[i])
    end
    nu = length(col_sizes)

    # Determine which column block each D[L,k] activates
    active_cols = Vector{Int}(undef, nD)
    for k in 1:nD
        cs = active_ranges[k][1]
        # Find which block cs falls into
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

# ============================================================================
# SubBlockDiagGPU: Extraction, multiplication, and structurization
# ============================================================================

"""
    extract_sub_block_diag(A::CuSparseMatrixCSR, p::Int, K::Int, orient::Symbol) → SubBlockDiagGPU

Extract p×p sub-blocks from a rectangular block-diagonal CuSparse matrix.

For orient=:V (refine-like): A is (K*p*M) × (p*M) with M outer blocks of (K*p)×p,
  each containing K dense p×p sub-blocks stacked vertically.
For orient=:H (coarsen-like): A is (p*M) × (K*p*M) with M outer blocks of p×(K*p),
  each containing K dense p×p sub-blocks side by side.
"""
function extract_sub_block_diag(A::CuSparseMatrixCSR{T,Ti}, p::Int, K::Int, orient::Symbol) where {T, Ti}
    m, n = size(A)
    if orient == :V
        @assert m % (K * p) == 0 "Rows ($m) not divisible by K*p ($(K*p))"
        M = m ÷ (K * p)
        @assert n == p * M "Cols ($n) should be p*M ($(p*M))"
    else  # :H
        @assert n % (K * p) == 0 "Cols ($n) not divisible by K*p ($(K*p))"
        M = n ÷ (K * p)
        @assert m == p * M "Rows ($m) should be p*M ($(p*M))"
    end

    A_cpu = SparseMatrixCSC(A)
    data_cpu = zeros(T, p, p, M * K)

    if orient == :V
        # Each outer block i: rows [(i-1)*K*p+1 : i*K*p], cols [(i-1)*p+1 : i*p]
        # Sub-block j within outer block i: rows [(i-1)*K*p + (j-1)*p + 1 : ... + j*p]
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
        VBlockDiagGPU(p, p, K, M, CuArray{T}(data_cpu))
    else  # :H
        # Each outer block i: rows [(i-1)*p+1 : i*p], cols [(i-1)*K*p+1 : i*K*p]
        # Sub-block j within outer block i: cols [(i-1)*K*p + (j-1)*p + 1 : ... + j*p]
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
        HBlockDiagGPU(p, p, K, M, CuArray{T}(data_cpu))
    end
end

# ============================================================================
# SubBlockDiagGPU multiplication kernels
# ============================================================================

"""
    _batched_gemm_broadcast_kernel!(C, A, B, ::Val{broadcastB}, rows_C, cols_C, inner, N, K)

CUDA kernel for batched GEMM with optional broadcast on B operand.
When broadcastB=true: C[:,:,s] = A[:,:,s] * B[:,:, (s-1)÷K + 1]
  Each B sub-block is reused K times.
When broadcastB=false: standard batched GEMM C[:,:,i] = A[:,:,i] * B[:,:,i]
"""
function _batched_gemm_broadcast_kernel!(C, A, B, ::Val{broadcastB}, ::Val{transA},
                                          rows_C, cols_C, inner, N, K) where {broadcastB, transA}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = rows_C * cols_C * N
    if idx <= total
        idx0 = idx - 1
        i = idx0 % rows_C + 1
        idx0 = idx0 ÷ rows_C
        j = idx0 % cols_C + 1
        blk = idx0 ÷ cols_C + 1
        b_blk = broadcastB ? (blk - 1) ÷ K + 1 : blk
        s = zero(eltype(C))
        for k = 1:inner
            if transA
                @inbounds s += A[k, i, blk] * B[k, j, b_blk]
            else
                @inbounds s += A[i, k, blk] * B[k, j, b_blk]
            end
        end
        @inbounds C[i, j, blk] = s
    end
    return nothing
end

"""
    _segmented_block_sum_kernel!(out, data, p, q, K, M)

CUDA kernel for K-fold segmented sum of blocks.
out[:,:,i] = sum(data[:,:,(i-1)*K+1 : i*K]) for i = 1:M.
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

"""
    _launch_broadcast_gemm(A_data, B_data, rows_C, cols_C, inner, N; broadcastB=false, K=1, transA=false)

Helper to launch batched GEMM kernel with optional broadcast on B.
"""
function _launch_broadcast_gemm(::Type{T}, A_data, B_data, rows_C, cols_C, inner, N;
                                 broadcastB::Bool=false, K::Int=1, transA::Bool=false) where T
    C_data = CuArray{T}(undef, rows_C, cols_C, N)
    total = rows_C * cols_C * N
    kernel = @cuda launch=false _batched_gemm_broadcast_kernel!(
        C_data, A_data, B_data, Val(broadcastB), Val(transA), rows_C, cols_C, inner, N, K)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(C_data, A_data, B_data, Val(broadcastB), Val(transA), rows_C, cols_C, inner, N, K;
           threads=threads, blocks=blocks)
    C_data
end

"""
    _launch_segmented_sum(data, p, q, K, M)

Helper to launch segmented sum kernel. Returns p×q×M array.
"""
function _launch_segmented_sum(::Type{T}, data, p, q, K, M) where T
    out = CuArray{T}(undef, p, q, M)
    total = p * q * M
    kernel = @cuda launch=false _segmented_block_sum_kernel!(out, data, p, q, K, M)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, data, p, q, K, M; threads=threads, blocks=blocks)
    out
end

# --- BlockDiagGPU * VBlockDiagGPU → VBlockDiagGPU ---
# op has N=M*K blocks (one per sub-block of refine). Each sub-block of result:
# result[:,:,s] = op[:,:,s] * refine[:,:,s]
function Base.:*(A::BlockDiagGPU{T}, B::VBlockDiagGPU{T}) where T
    N = A.N
    @assert N == B.M * B.K "BlockDiagGPU has $N blocks, VBlockDiagGPU has $(B.M*B.K) sub-blocks"
    @assert A.q == B.p "Inner dimension mismatch: A.q=$(A.q) vs B.p=$(B.p)"
    C_data = _launch_broadcast_gemm(T, A.data, B.data, A.p, B.q, A.q, N)
    VBlockDiagGPU(A.p, B.q, B.K, B.M, C_data)
end

# --- HBlockDiagGPU * BlockDiagGPU → HBlockDiagGPU ---
# coarsen has M*K sub-blocks, op has N=M*K blocks.
# result[:,:,s] = coarsen[:,:,s] * op[:,:,s]
function Base.:*(A::HBlockDiagGPU{T}, B::BlockDiagGPU{T}) where T
    N = B.N
    @assert A.M * A.K == N "HBlockDiagGPU has $(A.M*A.K) sub-blocks, BlockDiagGPU has $N blocks"
    @assert A.q == B.p "Inner dimension mismatch: A.q=$(A.q) vs B.p=$(B.p)"
    C_data = _launch_broadcast_gemm(T, A.data, B.data, A.p, B.q, A.q, N)
    HBlockDiagGPU(A.p, B.q, A.K, A.M, C_data)
end

# --- HBlockDiagGPU * VBlockDiagGPU → BlockDiagGPU ---
# coarsen (H) has M outer blocks with K sub-blocks each, refine (V) has same K, M.
# Per outer block i: result[:,:,i] = sum_{j=1}^{K} coarsen_sub[j] * refine_sub[j]
# = N batched GEMM + K-fold segmented sum → M blocks.
function Base.:*(A::HBlockDiagGPU{T}, B::VBlockDiagGPU{T}) where T
    @assert A.K == B.K && A.M == B.M "SubBlockDiagGPU K/M mismatch"
    @assert A.q == B.p "Inner dimension mismatch: A.q=$(A.q) vs B.p=$(B.p)"
    N = A.M * A.K
    # Step 1: N batched GEMMs producing p × q × N products
    products = _launch_broadcast_gemm(T, A.data, B.data, A.p, B.q, A.q, N)
    # Step 2: Segmented sum: group K consecutive products into M outer blocks
    out = _launch_segmented_sum(T, products, A.p, B.q, A.K, A.M)
    BlockDiagGPU{T}(A.p, B.q, A.M, out)
end

# --- VBlockDiagGPU * VBlockDiagGPU → VBlockDiagGPU ---
# Building cumulative refine_fine: refine_fine[l] = refine_fine[l+1] * refine[l]
# A (refine_fine[l+1]) has K_A sub-blocks per outer block, M_A outer blocks.
# B (refine[l]) has K_B sub-blocks per outer block, M_B outer blocks.
# Requirement: M_A == M_B * K_B (A's outer blocks align with B's sub-blocks).
# Result has K_A * K_B sub-blocks per outer block, M_B outer blocks.
# result[:,:,s] = A[:,:,s] * B[:,:, (s-1)÷K_A + 1]
function Base.:*(A::VBlockDiagGPU{T}, B::VBlockDiagGPU{T}) where T
    @assert A.M == B.M * B.K "VBlockDiagGPU * VBlockDiagGPU: A.M=$(A.M) should equal B.M*B.K=$(B.M*B.K)"
    @assert A.q == B.p "Inner dimension mismatch: A.q=$(A.q) vs B.p=$(B.p)"
    K_result = A.K * B.K
    M_result = B.M
    N_total = M_result * K_result  # = A.M * A.K
    # Each B sub-block is reused K_A times
    C_data = _launch_broadcast_gemm(T, A.data, B.data, A.p, B.q, A.q, N_total;
                                     broadcastB=true, K=A.K)
    VBlockDiagGPU(A.p, B.q, K_result, M_result, C_data)
end

# --- HBlockDiagGPU * HBlockDiagGPU → HBlockDiagGPU ---
# Building cumulative coarsen_fine: coarsen_fine[l] = coarsen[l] * coarsen_fine[l+1]
# A (coarsen[l]) has K_A sub-blocks per outer block, M_A outer blocks.
# B (coarsen_fine[l+1]) has K_B sub-blocks per outer block, M_B outer blocks.
# Requirement: M_B == M_A * K_A (B's outer blocks align with A's sub-blocks).
# Result has K_A * K_B sub-blocks per outer block, M_A outer blocks.
# For sub-block s: result[:,:,s] = A[:,:, (s-1)÷K_B + 1] * B[:,:,s]
function Base.:*(A::HBlockDiagGPU{T}, B::HBlockDiagGPU{T}) where T
    @assert B.M == A.M * A.K "HBlockDiagGPU * HBlockDiagGPU: B.M=$(B.M) should equal A.M*A.K=$(A.M*A.K)"
    @assert A.q == B.p "Inner dimension mismatch: A.q=$(A.q) vs B.p=$(B.p)"
    K_result = A.K * B.K
    M_result = A.M
    N_total = M_result * K_result  # = B.M * B.K
    # Each A sub-block is reused K_B times: A[:,:, (s-1)÷K_B + 1]
    # We need to broadcast A. Swap operand order in the kernel perspective:
    # C[s] = A[(s-1)÷K_B + 1] * B[s]
    # This is a broadcast on A with stride K_B.
    # Use a dedicated kernel launch with transA=false and broadcast on A side.
    C_data = _launch_broadcast_gemm_A(T, A.data, B.data, A.p, B.q, A.q, N_total, B.K)
    HBlockDiagGPU(A.p, B.q, K_result, M_result, C_data)
end

"""
    _batched_gemm_broadcast_A_kernel!(C, A, B, rows_C, cols_C, inner, N, K_B)

CUDA kernel for batched GEMM with broadcast on A operand.
C[:,:,s] = A[:,:, (s-1)÷K_B + 1] * B[:,:,s]
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

function _launch_broadcast_gemm_A(::Type{T}, A_data, B_data, rows_C, cols_C, inner, N, K_B) where T
    C_data = CuArray{T}(undef, rows_C, cols_C, N)
    total = rows_C * cols_C * N
    kernel = @cuda launch=false _batched_gemm_broadcast_A_kernel!(
        C_data, A_data, B_data, rows_C, cols_C, inner, N, K_B)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(C_data, A_data, B_data, rows_C, cols_C, inner, N, K_B;
           threads=threads, blocks=blocks)
    C_data
end

# ============================================================================
# SubBlockDiagGPU → CuSparseMatrixCSR conversion (for debugging/fallback)
# ============================================================================

function _to_cusparse(B::VBlockDiagGPU{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = K * p * M
    total_cols = q * M
    data_cpu = Array(B.data)

    nnz_total = count(!=(zero(T)), data_cpu)
    rowptr = Vector{Int32}(undef, total_rows + 1)
    colval = Vector{Int32}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    pos = 1
    for i in 1:M          # outer block
        for j in 1:K      # sub-block within outer block
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

function _to_cusparse(B::HBlockDiagGPU{T}) where T
    p, q, K, M = B.p, B.q, B.K, B.M
    total_rows = p * M
    total_cols = K * q * M
    data_cpu = Array(B.data)

    nnz_total = count(!=(zero(T)), data_cpu)
    rowptr = Vector{Int32}(undef, total_rows + 1)
    colval = Vector{Int32}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    pos = 1
    for i in 1:M          # outer block
        for r in 1:p
            global_row = (i - 1) * p + r
            rowptr[global_row] = Int32(pos)
            for j in 1:K  # sub-block within outer block
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

# Fallback: CuSparse * SubBlockDiagGPU and SubBlockDiagGPU * CuSparse
function Base.:*(A::CuSparseMatrixCSR{T,Ti}, B::SubBlockDiagGPU{T}) where {T, Ti}
    A * _to_cusparse(B)
end

function Base.:*(A::SubBlockDiagGPU{T}, B::CuSparseMatrixCSR{T,Ti}) where {T, Ti}
    _to_cusparse(A) * B
end

# ============================================================================
# amgb_blockdiag for VBlockDiagGPU and HBlockDiagGPU
# ============================================================================

# amgb_blockdiag of nu VBlockDiagGPUs (e.g., refine_z = blockdiag of nu copies of refine[l]).
# Each arg has the same (p, q, K) but possibly different M (though typically same).
# Result: VBlockDiagGPU with M_total = sum of M values, data concatenated along dim 3.
function MultiGridBarrier.amgb_blockdiag(args::VBlockDiagGPU{T}...) where T
    p = args[1].p
    q = args[1].q
    K = args[1].K
    for a in args
        @assert a.p == p && a.q == q && a.K == K "amgb_blockdiag: VBlockDiagGPU p/q/K mismatch"
    end
    M_total = sum(a.M for a in args)
    data_new = cat([a.data for a in args]...; dims=3)
    VBlockDiagGPU(p, q, K, M_total, data_new)
end

function MultiGridBarrier.amgb_blockdiag(args::HBlockDiagGPU{T}...) where T
    p = args[1].p
    q = args[1].q
    K = args[1].K
    for a in args
        @assert a.p == p && a.q == q && a.K == K "amgb_blockdiag: HBlockDiagGPU p/q/K mismatch"
    end
    M_total = sum(a.M for a in args)
    data_new = cat([a.data for a in args]...; dims=3)
    HBlockDiagGPU(p, q, K, M_total, data_new)
end

# ============================================================================
# BlockDiagGPU - UniformScaling and norm (for sanity check in amg_helper)
# ============================================================================

function Base.:-(A::BlockDiagGPU{T}, ::UniformScaling) where T
    @assert A.p == A.q "BlockDiagGPU - I requires square blocks"
    p = A.p
    # Build identity blocks on CPU and transfer once
    I_cpu = zeros(T, p, p, A.N)
    for k = 1:A.N, i = 1:p
        I_cpu[i, i, k] = one(T)
    end
    BlockDiagGPU{T}(p, p, A.N, A.data .- CuArray{T}(I_cpu))
end

# Frobenius norm of a block-diagonal matrix = norm of the block data
# (off-diagonal zeros don't contribute).
LinearAlgebra.norm(A::BlockDiagGPU) = norm(A.data)

# ============================================================================
# Matrix-vector products for structured block types
# ============================================================================

# --- BlockDiagGPU * CuVector ---
# Reuses _block_matvec_kernel! with col_offset=0
function Base.:*(A::BlockDiagGPU{T}, x::CuVector{T}) where T
    p, q, N = A.p, A.q, A.N
    @assert length(x) == q * N
    out = CuVector{T}(undef, p * N)
    total = p * N
    kernel = @cuda launch=false _block_matvec_kernel!(out, A.data, x, p, q, N, 0)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, A.data, x, p, q, N, 0; threads=threads, blocks=blocks)
    out
end

# --- VBlockDiagGPU * CuVector ---
# y[(blk-1)*p + r] = sum_c data[r,c,blk] * x[(outer-1)*q + c]
# where outer = (blk-1) ÷ K + 1, blk goes 1..M*K
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

function Base.:*(A::VBlockDiagGPU{T}, x::CuVector{T}) where T
    @assert length(x) == A.q * A.M
    MK = A.M * A.K
    total = A.p * MK
    out = CuVector{T}(undef, total)
    kernel = @cuda launch=false _vblock_matvec_kernel!(out, A.data, x, A.p, A.q, A.K, MK)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, A.data, x, A.p, A.q, A.K, MK; threads=threads, blocks=blocks)
    out
end

# --- HBlockDiagGPU * CuVector ---
# y[(outer-1)*p + r] = sum_{j=1}^{K} sum_{c=1}^{q} data[r,c,(outer-1)*K+j] * x[((outer-1)*K+j-1)*q + c]
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

function Base.:*(A::HBlockDiagGPU{T}, x::CuVector{T}) where T
    @assert length(x) == A.K * A.q * A.M
    total = A.p * A.M
    out = CuVector{T}(undef, total)
    kernel = @cuda launch=false _hblock_matvec_kernel!(out, A.data, x, A.p, A.q, A.K, A.M)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, A.data, x, A.p, A.q, A.K, A.M; threads=threads, blocks=blocks)
    out
end

# --- adjoint(VBlockDiagGPU) * CuVector ---
# V' is (q*M) × (K*p*M). y[(outer-1)*q + c] = sum_{j,r} data[r,c,(outer-1)*K+j] * x[((outer-1)*K+j-1)*p + r]
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

function Base.:*(A::Adjoint{T, <:VBlockDiagGPU{T}}, x::CuVector{T}) where T
    V = parent(A)
    @assert length(x) == V.K * V.p * V.M
    total = V.q * V.M
    out = CuVector{T}(undef, total)
    kernel = @cuda launch=false _vblock_adj_matvec_kernel!(out, V.data, x, V.p, V.q, V.K, V.M)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, V.data, x, V.p, V.q, V.K, V.M; threads=threads, blocks=blocks)
    out
end

# --- adjoint(HBlockDiagGPU) * CuVector ---
# H' is (K*q*M) × (p*M). y[(sub-1)*q + c] = sum_r data[r,c,sub] * x[(outer-1)*p + r]
# where outer = (sub-1) ÷ K + 1
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

function Base.:*(A::Adjoint{T, <:HBlockDiagGPU{T}}, x::CuVector{T}) where T
    H = parent(A)
    MK = H.M * H.K
    @assert length(x) == H.p * H.M
    total = H.q * MK
    out = CuVector{T}(undef, total)
    kernel = @cuda launch=false _hblock_adj_matvec_kernel!(out, H.data, x, H.p, H.q, H.K, MK)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(out, H.data, x, H.p, H.q, H.K, MK; threads=threads, blocks=blocks)
    out
end

# --- Matrix-matrix products (column-loop over CuVector kernels) ---
# Used by multigrid_from_fine_grid (coarsen * f_grid) and V-cycle (coarsen_u * cm).

function Base.:*(A::HBlockDiagGPU{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::VBlockDiagGPU{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::BlockDiagGPU{T}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::Adjoint{T, <:VBlockDiagGPU{T}}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

function Base.:*(A::Adjoint{T, <:HBlockDiagGPU{T}}, B::CuMatrix{T}) where T
    ncols = size(B, 2)
    out = CuMatrix{T}(undef, size(A, 1), ncols)
    for col in 1:ncols
        out[:, col] = A * B[:, col]
    end
    out
end

# ============================================================================
# _structurize_geometry: convert Geometry operators/refine/coarsen to block types
# (internal; called by native_to_cuda when structured=true)
# ============================================================================

function _structurize_geometry(g::MultiGridBarrier.Geometry{T,X,W,<:Any,<:Any,<:Any,M_sub,Disc},
                               p::Int) where {T,X,W,M_sub,Disc}
    L = length(g.refine)

    # Convert operators to BlockDiagGPU
    operators_new = Dict(key => extract_block_diag(op, p) for (key, op) in g.operators)

    # Convert refine/coarsen to V/HBlockDiagGPU
    # Build first element to determine concrete types, then allocate typed vectors
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

    # Subspaces stay as their original type (CuSparse)
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
