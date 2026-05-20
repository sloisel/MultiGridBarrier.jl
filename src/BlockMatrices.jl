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


# Multigrid refine/coarsen transfers built by the structured geometric_mg builders.
# They are composed into the (sparse) prolongations R at MultiGrid construction, so
# they are emitted directly as SparseMatrixCSC — `data` is the p×q×(M*K) block array.
#
# `_vblock_sparse` (refine-like): M outer blocks of (K*p)×q.
# `_hblock_sparse` (coarsen-like): M outer blocks of p×(K*q).
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

function _hblock_sparse(p::Int, q::Int, K::Int, M::Int, data::AbstractArray{T,3}) where {T}
    I_idx = Int[]; J_idx = Int[]; V_val = T[]
    for i in 1:M, j in 1:K
        sub_idx = (i - 1) * K + j
        for c in 1:q, r in 1:p
            v = data[r, c, sub_idx]
            if v != zero(T)
                push!(I_idx, (i - 1) * p + r)
                push!(J_idx, (i - 1) * K * q + (j - 1) * q + c)
                push!(V_val, v)
            end
        end
    end
    sparse(I_idx, J_idx, V_val, p * M, K * q * M)
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

# Matrix-matrix products (column-loop)
function Base.:*(A::BlockDiag{T}, B::Matrix{T}) where T
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
# mgb_* dispatch methods
# ============================================================================

mgb_zeros(::BlockColumn{T}, m, n) where {T} = spzeros(T, m, n)
mgb_zeros(::Adjoint{T, <:BlockColumn{T}}, m, n) where {T} = spzeros(T, m, n)

function mgb_zeros(A::BlockDiag{T}, m, n) where T
    @assert m == n && m % A.p == 0
    N = m ÷ A.p
    data = similar(A.data, A.p, A.p, N)
    fill!(data, zero(T))
    BlockDiag{T, typeof(data)}(A.p, A.p, N, data)
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

# Extend mgb_cleanup for block-structured solutions
function mgb_cleanup(sol::MGBSOL{T, <:Any, <:Vector{T}}) where T
    # Check if the geometry uses block types
    if sol.geometry isa Geometry && any(v -> v isa BlockDiag, values(sol.geometry.operators))
        _block_assembly_cleanup()
    end
    sol
end
