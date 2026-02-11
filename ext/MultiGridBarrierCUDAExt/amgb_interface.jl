# amgb_interface.jl -- Extend MultiGridBarrier API for CUDA types
#
# Follows the pattern from HPCMultiGridBarrier.jl/src/MultiGridBarrierMPI.jl lines 62-193.

using CUDA.CUSPARSE: CuSparseMatrixCSR

import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_diag, amgb_blockdiag,
                         map_rows, map_rows_gpu, vertex_indices, _raw_array,
                         _to_cpu_array, _rows_to_svectors

# ============================================================================
# amgb_zeros: Create zero matrices/vectors on GPU
# ============================================================================

MultiGridBarrier.amgb_zeros(::CuSparseMatrixCSR{T,Int32}, m, n) where {T} =
    _cu_spzeros(T, m, n)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:CuSparseMatrixCSR{T,Int32}}, m, n) where {T} =
    _cu_spzeros(T, m, n)

MultiGridBarrier.amgb_zeros(::CuMatrix{T}, m, n) where {T} = CUDA.zeros(T, m, n)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:CuMatrix{T}}, m, n) where {T} = CUDA.zeros(T, m, n)

MultiGridBarrier.amgb_zeros(::Type{<:CuVector{T}}, m) where {T} = CUDA.zeros(T, m)

# ============================================================================
# amgb_all_isfinite: Check if all elements are finite (GPU-friendly)
# ============================================================================

function MultiGridBarrier.amgb_all_isfinite(z::CuVector{T}) where {T}
    all(isfinite.(z))
end

function MultiGridBarrier.amgb_all_isfinite(z::CuMatrix{T}) where {T}
    all(isfinite.(z))
end

# ============================================================================
# amgb_diag: Create diagonal sparse matrix from vector
# ============================================================================

MultiGridBarrier.amgb_diag(::CuSparseMatrixCSR{T,Int32}, z::CuVector{T}, m=length(z), n=length(z)) where {T} =
    _cu_spdiag(z, m, n)

function MultiGridBarrier.amgb_diag(::CuMatrix{T}, z::CuVector{T}, m=length(z), n=length(z)) where {T}
    # Dense path (spectral): return a dense CuMatrix diagonal
    D = CUDA.zeros(T, m, n)
    len = min(length(z), m, n)
    if len > 0
        z_cpu = Array(z)
        D_cpu = zeros(T, m, n)
        for i in 1:len
            D_cpu[i,i] = z_cpu[i]
        end
        copyto!(D, D_cpu)
    end
    D
end

# Also handle plain Vector z with CUDA matrix types -- convert to CuVector first
MultiGridBarrier.amgb_diag(A::CuSparseMatrixCSR{T,Int32}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    _cu_spdiag(CuVector{T}(z), m, n)

function MultiGridBarrier.amgb_diag(A::CuMatrix{T}, z::Vector{T}, m=length(z), n=length(z)) where {T}
    D_cpu = zeros(T, m, n)
    len = min(length(z), m, n)
    for i in 1:len
        D_cpu[i,i] = z[i]
    end
    CuMatrix{T}(D_cpu)
end

# BlockColumnOp dispatches are in block_ops.jl:
#   amgb_diag(::BlockColumnOp, z::CuVector) → Diagonal(z)
#   amgb_zeros(::BlockColumnOp, m, n) → _cu_spzeros(T, m, n)

# Also handle plain Vector z with BlockColumnOp -- convert to CuVector first
MultiGridBarrier.amgb_diag(A::BlockColumnOp{T}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Diagonal(CuVector{T}(z))

# ============================================================================
# amgb_blockdiag: Block diagonal concatenation
# ============================================================================

function MultiGridBarrier.amgb_blockdiag(args::CuSparseMatrixCSR{T,Int32}...) where {T}
    blockdiag(args...)
end

function MultiGridBarrier.amgb_blockdiag(args::CuMatrix{T}...) where {T}
    total_rows = sum(size(a, 1) for a in args)
    total_cols = sum(size(a, 2) for a in args)
    result = CUDA.zeros(T, total_rows, total_cols)
    row_off = 0
    col_off = 0
    for a in args
        m, n = size(a)
        result[row_off+1:row_off+m, col_off+1:col_off+n] .= a
        row_off += m
        col_off += n
    end
    result
end

# ============================================================================
# map_rows and map_rows_gpu: GPU-accelerated row-wise map
# ============================================================================

# map_rows: CPU fallback for arbitrary closures (used in setup code)
# Transfers to CPU, runs default map_rows, transfers result back to GPU.
function MultiGridBarrier.map_rows(f, A::CuMatrix{T}, rest::CuMatrix...) where T
    A_cpu = Matrix{T}(Array(A))
    rest_cpu = map(m -> Matrix{T}(Array(m)), rest)
    # Call the default (non-CUDA) map_rows on CPU arrays
    result_cpu = MultiGridBarrier.map_rows(f, A_cpu, rest_cpu...)
    # Transfer result back to GPU
    if result_cpu isa AbstractMatrix
        return CuMatrix{T}(result_cpu)
    else
        return CuVector{T}(result_cpu)
    end
end

# map_rows_gpu: True GPU kernel execution for GPU-friendly barrier functions
function MultiGridBarrier.map_rows_gpu(f, A::CuMatrix{T}, rest::CuMatrix{T}...) where T
    output = _map_rows_gpu_cuda(f, A, rest...)
    if size(output, 2) == 1
        return vec(output)
    end
    return output
end

# CuMatrix first with non-CuMatrix rest args (CuVector etc.) — use base map_rows path
function MultiGridBarrier.map_rows_gpu(f, A::CuMatrix{T}, rest...) where T
    return MultiGridBarrier.map_rows(f, A, rest...)
end


# ============================================================================
# _raw_array: Extract raw array (identity for CUDA types, no MPI wrappers)
# ============================================================================

MultiGridBarrier._raw_array(x::CuVector) = x
MultiGridBarrier._raw_array(x::CuMatrix) = x

# ============================================================================
# _rows_to_svectors: For CUDA types, pass through to default
# ============================================================================

# CuMatrix and CuVector pass through to the default implementation
# which uses reinterpret+transpose (works for GPU arrays)

# ============================================================================
# _to_cpu_array: Convert GPU arrays to CPU for barrier scalar indexing
# ============================================================================

MultiGridBarrier._to_cpu_array(x::CuMatrix) = Array(x)
MultiGridBarrier._to_cpu_array(x::CuVector) = Array(x)

# ============================================================================
# vertex_indices: Create vertex index vector for GPU types
# ============================================================================

function MultiGridBarrier.vertex_indices(A::CuMatrix)
    return CuVector{Int}(1:size(A, 1))
end

function MultiGridBarrier.vertex_indices(A::CuVector)
    return CuVector{Int}(1:length(A))
end

