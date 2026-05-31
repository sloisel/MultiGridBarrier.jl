# mgb_interface.jl -- Extend MultiGridBarrier API for CUDA types
#
# Follows the pattern from HPCMultiGridBarrier.jl/src/MultiGridBarrierMPI.jl lines 62-193.

using CUDA.CUSPARSE: CuSparseMatrixCSR

import MultiGridBarrier: mgb_zeros, mgb_all_isfinite, mgb_diag,
                         map_rows, map_rows_gpu, _to_cpu_array, _rows_to_svectors

# ============================================================================
# mgb_zeros: Create zero matrices/vectors on GPU
# ============================================================================

MultiGridBarrier.mgb_zeros(::CuMatrix{T}, m, n) where {T} = CUDA.zeros(T, m, n)

MultiGridBarrier.mgb_zeros(::Type{<:CuVector{T}}, m) where {T} = CUDA.zeros(T, m)

# ============================================================================
# mgb_all_isfinite: Check if all elements are finite (GPU-friendly)
# ============================================================================

function MultiGridBarrier.mgb_all_isfinite(z::CuVector{T}) where {T}
    all(isfinite.(z))
end

# ============================================================================
# mgb_diag: Create diagonal sparse matrix from vector
# ============================================================================

function MultiGridBarrier.mgb_diag(::CuMatrix{T}, z::CuVector{T}, m=length(z), n=length(z)) where {T}
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

# The CuVector path above is what the solver actually hits. (The plain-`Vector`
# mgb_diag overloads and the GPU mgb_blockdiag were removed: the live GPU solve
# only passes CuVector/CuMatrix arguments, and block concatenation happens
# CPU-side before transfer.)

# ============================================================================
# map_rows and map_rows_gpu: GPU-accelerated row-wise map
# ============================================================================

# map_rows: CPU fallback for arbitrary closures, reached via the map_rows_gpu
# fallback below. Transfers to CPU, runs default map_rows, transfers back.
function MultiGridBarrier.map_rows(f, A::CuMatrix{T}, rest::CuMatrix...) where T
    A_cpu = Matrix{T}(Array(A))
    rest_cpu = map(m -> Matrix{T}(Array(m)), rest)
    result_cpu = MultiGridBarrier.map_rows(f, A_cpu, rest_cpu...)
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
# _to_cpu_array: Convert GPU arrays to CPU for barrier scalar indexing
# ============================================================================

MultiGridBarrier._to_cpu_array(x::CuMatrix) = Array(x)
MultiGridBarrier._to_cpu_array(x::CuVector) = Array(x)

