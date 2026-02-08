# sparse_ops.jl -- GPU-native sparse operations for CuSparseMatrixCSR

using CUDA.CUSPARSE: CuSparseMatrixCSR

"""
    _cu_spzeros(::Type{T}, m, n) where T

Create a zero CuSparseMatrixCSR with no nonzero entries.
"""
function _cu_spzeros(::Type{T}, m, n) where T
    rowPtr = CUDA.ones(Int32, m + 1)
    colVal = CuVector{Int32}(undef, 0)
    nzVal = CuVector{T}(undef, 0)
    CuSparseMatrixCSR{T}(rowPtr, colVal, nzVal, (m, n))
end

"""
    _cu_spdiag(v::CuVector{T}, m, n) where T

Create a diagonal CuSparseMatrixCSR from a CuVector.
The diagonal has length `length(v)`, embedded in an m×n matrix.
"""
function _cu_spdiag(v::CuVector{T}, m, n) where T
    k = length(v)
    rowPtr = CuVector{Int32}(vcat(Int32.(1:k), fill(Int32(k + 1), m - k + 1)))
    colVal = CuVector{Int32}(1:k)
    CuSparseMatrixCSR{T}(rowPtr, colVal, copy(v), (m, n))
end

"""
    blockdiag(A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}) where T

Block diagonal concatenation of two CuSparseMatrixCSR matrices on GPU.
Result has A in top-left and B in bottom-right.
"""
function blockdiag(A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}) where T
    Ti = Int32
    nnzA = nnz(A)
    new_rowPtr = vcat(A.rowPtr, B.rowPtr[2:end] .+ Ti(nnzA))
    new_colVal = vcat(A.colVal, B.colVal .+ Ti(size(A, 2)))
    new_nzVal = vcat(A.nzVal, B.nzVal)
    CuSparseMatrixCSR{T}(new_rowPtr, new_colVal, new_nzVal,
        (size(A, 1) + size(B, 1), size(A, 2) + size(B, 2)))
end

function blockdiag(args::CuSparseMatrixCSR{T}...) where T
    result = args[1]
    for i in 2:length(args)
        result = blockdiag(result, args[i])
    end
    result
end

"""
    _cu_hcat_kernel!(new_colVal, new_nzVal, new_rowPtr,
                     A_colVal, A_nzVal, A_rowPtr,
                     B_colVal, B_nzVal, B_rowPtr,
                     n1, nrows)

CUDA kernel for hcat: each thread processes one row,
copying A's entries then B's entries (with B's col indices offset by n1).
"""
function _cu_hcat_kernel!(new_colVal, new_nzVal, new_rowPtr,
                          A_colVal, A_nzVal, A_rowPtr,
                          B_colVal, B_nzVal, B_rowPtr,
                          n1, nrows)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= nrows
        dest = new_rowPtr[i] - Int32(1)  # 0-based dest offset

        # Copy A's entries for row i
        a_start = A_rowPtr[i]
        a_end = A_rowPtr[i + 1] - Int32(1)
        for j in a_start:a_end
            pos = dest + (j - a_start) + Int32(1)
            @inbounds new_colVal[pos] = A_colVal[j]
            @inbounds new_nzVal[pos] = A_nzVal[j]
        end

        # Copy B's entries for row i (offset columns by n1)
        b_start = B_rowPtr[i]
        b_end = B_rowPtr[i + 1] - Int32(1)
        a_count = a_end - a_start + Int32(1)
        for j in b_start:b_end
            pos = dest + a_count + (j - b_start) + Int32(1)
            @inbounds new_colVal[pos] = B_colVal[j] + n1
            @inbounds new_nzVal[pos] = B_nzVal[j]
        end
    end
    return nothing
end

"""
    Base.hcat(A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}) where T

Horizontal concatenation of two CuSparseMatrixCSR matrices on GPU.
Both matrices must have the same number of rows.
"""
function Base.hcat(A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}) where T
    Ti = Int32
    nrows = size(A, 1)
    @assert size(B, 1) == nrows "hcat requires same number of rows"

    n1 = Ti(size(A, 2))

    # Compute per-row nnz counts
    A_nnz_per_row = A.rowPtr[2:end] .- A.rowPtr[1:end-1]
    B_nnz_per_row = B.rowPtr[2:end] .- B.rowPtr[1:end-1]
    total_nnz_per_row = A_nnz_per_row .+ B_nnz_per_row

    # Build new rowPtr via cumsum (no scalar indexing)
    cs = cumsum(total_nnz_per_row)
    new_rowPtr = vcat(CuVector{Ti}([Ti(1)]), cs .+ Ti(1))

    total_nnz = Int(CUDA.@allowscalar new_rowPtr[end] - 1)

    new_colVal = CuVector{Ti}(undef, total_nnz)
    new_nzVal = CuVector{T}(undef, total_nnz)

    # Launch kernel
    kernel = @cuda launch=false _cu_hcat_kernel!(
        new_colVal, new_nzVal, new_rowPtr,
        A.colVal, A.nzVal, A.rowPtr,
        B.colVal, B.nzVal, B.rowPtr,
        n1, nrows)
    config = launch_configuration(kernel.fun)
    threads = min(nrows, config.threads)
    blocks = cld(nrows, threads)
    kernel(new_colVal, new_nzVal, new_rowPtr,
           A.colVal, A.nzVal, A.rowPtr,
           B.colVal, B.nzVal, B.rowPtr,
           n1, nrows; threads=threads, blocks=blocks)

    CuSparseMatrixCSR{T}(new_rowPtr, new_colVal, new_nzVal,
        (nrows, size(A, 2) + size(B, 2)))
end

function Base.hcat(args::CuSparseMatrixCSR{T}...) where T
    result = args[1]
    for i in 2:length(args)
        result = hcat(result, args[i])
    end
    result
end
