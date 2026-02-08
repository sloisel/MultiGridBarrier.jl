# block_types.jl -- Structured matrix types for GPU Hessian assembly
#
# These types enable dispatch-based optimization of D[j]' * diag(v) * D[k]
# triple products without modifying the f2 code in AlgebraicMultiGridBarrier.jl.
#
# For fem2d: operators (dx, dy, id) are block-diagonal with 7×7 dense element blocks.
# At the finest multigrid level, D[L,k] = hcat(Z, ..., Op, ..., Z) where Op is
# block-diagonal. Representing these structurally avoids SpGEMM entirely for
# Hessian assembly, reducing ~34 SpGEMMs to ~2 per Newton step.

using CUDA
using LinearAlgebra

"""
    BlockDiagGPU{T} <: AbstractMatrix{T}

Block-diagonal matrix stored as a 3D CuArray (p × q × N), where each slice
`data[:,:,i]` is the i-th dense block. The full matrix is (p*N) × (q*N).

For fem2d: p=q=7, N = number of elements.
"""
struct BlockDiagGPU{T} <: AbstractMatrix{T}
    p::Int          # block rows
    q::Int          # block cols
    N::Int          # number of blocks
    data::CuArray{T,3}  # p × q × N
end

function BlockDiagGPU(data::CuArray{T,3}) where T
    p, q, N = size(data)
    BlockDiagGPU{T}(p, q, N, data)
end

Base.size(A::BlockDiagGPU) = (A.p * A.N, A.q * A.N)

function Base.getindex(A::BlockDiagGPU{T}, i::Int, j::Int) where T
    # For debugging only — triggers scalar indexing
    bi = (i - 1) ÷ A.p + 1  # block index for row
    bj = (j - 1) ÷ A.q + 1  # block index for col
    if bi != bj
        return zero(T)
    end
    li = (i - 1) % A.p + 1  # local row within block
    lj = (j - 1) % A.q + 1  # local col within block
    return CUDA.@allowscalar A.data[li, lj, bi]
end

"""
    BlockColumnOp{T} <: AbstractMatrix{T}

Represents D[L,k] = hcat(Z, ..., Op, ..., Z) where Op is a BlockDiagGPU at
position `active_col` among `nu` column-blocks. Rows are p*N, columns are
sum(col_sizes).

The only non-zero column block is `active_block` at column position `active_col`.
"""
struct BlockColumnOp{T} <: AbstractMatrix{T}
    active_block::BlockDiagGPU{T}
    active_col::Int         # which column block is active (1-indexed)
    nu::Int                 # total number of column blocks
    col_sizes::Vector{Int}  # size of each column block
    total_rows::Int         # = p * N
end

function Base.size(A::BlockColumnOp)
    (A.total_rows, sum(A.col_sizes))
end

function Base.getindex(A::BlockColumnOp{T}, i::Int, j::Int) where T
    # For debugging only
    col_offset = sum(A.col_sizes[1:A.active_col-1])
    col_end = col_offset + A.col_sizes[A.active_col]
    if j <= col_offset || j > col_end
        return zero(T)
    end
    return A.active_block[i, j - col_offset]
end

"""
    BlockHessianGPU{T} <: AbstractMatrix{T}

nu×nu grid of BlockDiagGPU blocks representing the accumulated Hessian
before restriction by R. Block (i,j) holds the contribution from
D[i]' * diag(v) * D[j].

Only positions that have been filled are non-nothing.
"""
struct BlockHessianGPU{T} <: AbstractMatrix{T}
    blocks::Matrix{Union{BlockDiagGPU{T}, Nothing}}  # nu × nu
    p::Int              # element block size (rows = cols for Hessian)
    N::Int              # number of elements
    block_sizes::Vector{Int}  # size of each row/col block
end

function Base.size(A::BlockHessianGPU)
    total = sum(A.block_sizes)
    (total, total)
end

function Base.getindex(A::BlockHessianGPU{T}, i::Int, j::Int) where T
    # For debugging — find which block (bi, bj) and local indices
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
    SubBlockDiagGPU{T, Orient} <: AbstractMatrix{T}

Block-diagonal matrix with sub-block structure for multigrid coarsen/refine operators.

Each "outer block" consists of K sub-blocks of size p×q stacked vertically (VBlockDiagGPU)
or horizontally (HBlockDiagGPU). There are M outer blocks, so M*K total sub-blocks.

Data is stored as a CuArray{T,3} of size p × q × (M*K), where each slice is one sub-block.

**VBlockDiagGPU** (refine-like, Orient=Val{:V}):
  M outer blocks of (K*p)×q. Full matrix: (K*p*M) × (q*M).
  Used for refine operators where each outer block stacks K sub-blocks vertically.

**HBlockDiagGPU** (coarsen-like, Orient=Val{:H}):
  M outer blocks of p×(K*q). Full matrix: (p*M) × (K*q*M).
  Used for coarsen operators where each outer block places K sub-blocks side by side.

For FEM2D: p=q=7, K=4 per level (4 child elements per parent), M = elements at coarser level.
"""
struct SubBlockDiagGPU{T, Orient} <: AbstractMatrix{T}
    p::Int       # sub-block rows
    q::Int       # sub-block cols
    K::Int       # sub-blocks per outer block
    M::Int       # number of outer blocks
    data::CuArray{T,3}  # p × q × (M*K)
end

const VBlockDiagGPU{T} = SubBlockDiagGPU{T, Val{:V}}
const HBlockDiagGPU{T} = SubBlockDiagGPU{T, Val{:H}}

function VBlockDiagGPU(p::Int, q::Int, K::Int, M::Int, data::CuArray{T,3}) where T
    SubBlockDiagGPU{T, Val{:V}}(p, q, K, M, data)
end

function HBlockDiagGPU(p::Int, q::Int, K::Int, M::Int, data::CuArray{T,3}) where T
    SubBlockDiagGPU{T, Val{:H}}(p, q, K, M, data)
end

function Base.size(A::VBlockDiagGPU)
    (A.K * A.p * A.M, A.q * A.M)
end

function Base.size(A::HBlockDiagGPU)
    (A.p * A.M, A.K * A.q * A.M)
end

function Base.getindex(A::VBlockDiagGPU{T}, i::Int, j::Int) where T
    # For debugging only — triggers scalar indexing
    # Outer block index from column: each outer block has q columns
    bj = (j - 1) ÷ A.q + 1
    lj = (j - 1) % A.q + 1
    # Outer block index from row: each outer block has K*p rows
    outer_row_size = A.K * A.p
    bi = (i - 1) ÷ outer_row_size + 1
    if bi != bj
        return zero(T)
    end
    # Local row within outer block → sub-block index and local row
    local_row = (i - 1) % outer_row_size
    sub_idx = local_row ÷ A.p  # 0-indexed sub-block within outer block
    li = local_row % A.p + 1
    # Global sub-block index in data array
    global_sub = (bi - 1) * A.K + sub_idx + 1
    return CUDA.@allowscalar A.data[li, lj, global_sub]
end

function Base.getindex(A::HBlockDiagGPU{T}, i::Int, j::Int) where T
    # For debugging only — triggers scalar indexing
    # Outer block index from row: each outer block has p rows
    bi = (i - 1) ÷ A.p + 1
    li = (i - 1) % A.p + 1
    # Outer block index from column: each outer block has K*q columns
    outer_col_size = A.K * A.q
    bj = (j - 1) ÷ outer_col_size + 1
    if bi != bj
        return zero(T)
    end
    # Local col within outer block → sub-block index and local col
    local_col = (j - 1) % outer_col_size
    sub_idx = local_col ÷ A.q  # 0-indexed sub-block within outer block
    lj = local_col % A.q + 1
    # Global sub-block index in data array
    global_sub = (bj - 1) * A.K + sub_idx + 1
    return CUDA.@allowscalar A.data[li, lj, global_sub]
end

"""
    ScaledAdjBlockCol{T}

Lazy type representing `adjoint(D[j]::BlockColumnOp) * Diagonal(v)`.
No computation — just captures operands for the next multiply.
"""
struct ScaledAdjBlockCol{T}
    op::BlockColumnOp{T}   # the D[j] (not transposed — we store the original)
    diag::CuVector{T}      # the diagonal scaling vector
end

"""
    LazyHessianProduct{T, Ti}

Lazy type representing `R' * H` where R is CuSparseMatrixCSR and H is BlockHessianGPU.
No computation — just captures operands so that `(R' * H) * R` can be computed via
element-wise assembly instead of SPGEMM.
"""
struct LazyHessianProduct{T, Ti}
    R::CuSparseMatrixCSR{T, Ti}
    H::BlockHessianGPU{T}
end
