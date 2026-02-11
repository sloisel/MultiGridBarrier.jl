# block_types.jl -- Type aliases and GPU-internal plan types for structured Hessian assembly
#
# The core parametric block types (BlockDiag, BlockColumn, BlockHessian, SubBlockDiag,
# ScaledAdjBlockCol, LazyBlockHessianProduct) are defined in MultiGridBarrier/BlockMatrices.jl.
# This file provides:
#   1. Convenience type aliases for CuArray-backed dispatch
#   2. GPU-internal plan types (SparseConversionPlan, AssemblyPlan) for R'*H*R assembly

using CUDA
using LinearAlgebra

import MultiGridBarrier: BlockDiag, BlockColumn, BlockHessian, SubBlockDiag,
                         VBlockDiag, HBlockDiag, ScaledAdjBlockCol, LazyBlockHessianProduct

# ============================================================================
# Type aliases for CuArray-backed block types (dispatch convenience)
# ============================================================================

const CuBlockDiag{T} = BlockDiag{T, <:CuArray{T,3}}
const CuBlockColumn{T} = BlockColumn{T, <:CuArray{T,3}}
const CuBlockHessian{T} = BlockHessian{T, <:CuArray{T,3}}
const CuVBlockDiag{T} = VBlockDiag{T, <:CuArray{T,3}}
const CuHBlockDiag{T} = HBlockDiag{T, <:CuArray{T,3}}

# ============================================================================
# SparseConversionPlan: cached CSR conversion for BlockHessian → CuSparseMatrixCSR
# ============================================================================

"""
    SparseConversionPlan{Ti}

Cached sparse conversion plan for converting BlockHessian to CuSparseMatrixCSR.
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

# ============================================================================
# AssemblyPlan: cached plan for R' * BlockHessian * R via element-wise assembly
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
    panels::Vector{CuArray{T, 3}}

    # Per block k: column indices, shape (c_max_k, N)
    col_indices::Vector{CuArray{Ti, 2}}

    # Per block k: actual column count per element, shape (N,)
    c_counts::Vector{CuVector{Int32}}

    # Per block pair (i,j): scatter map, shape (c_max_i, c_max_j, N)
    scatter_idx::Dict{Tuple{Int,Int}, CuArray{Int32, 3}}

    # Block partitioning
    p::Int              # element block size
    N::Int              # number of elements
    nu::Int             # number of block groups
    c_max::Vector{Int}  # max columns per block
end
