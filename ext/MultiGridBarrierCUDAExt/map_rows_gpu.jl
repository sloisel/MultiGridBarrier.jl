# map_rows_gpu.jl -- CUDA kernel implementations for map_rows on CuMatrix
#
# Adapted from HPCSparseArraysCUDAExt.jl lines 896-1000.
# Explicit CUDA kernels for 1-2 arg CuMatrix cases.

using StaticArrays

# ============================================================================
# CUDA kernels
# ============================================================================

function _cuda_map_rows_kernel_1arg(f, output, arg1, ::Val{NC1}, ::Val{OCols}) where {NC1, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        result = f(row1)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_2args(f, output, arg1, arg2, ::Val{NC1}, ::Val{NC2}, ::Val{OCols}) where {NC1, NC2, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        result = f(row1, row2)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_3args(f, output, arg1, arg2, arg3, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{OCols}) where {NC1, NC2, NC3, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        result = f(row1, row2, row3)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        result = f(row1, row2, row3, row4)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

function _cuda_map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{OCols}) where {NC1, NC2, NC3, NC4, NC5, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(arg1, 1)
    if i <= n
        T = eltype(arg1)
        row1 = SVector{NC1,T}(ntuple(j -> @inbounds(arg1[i,j]), Val(NC1)))
        row2 = SVector{NC2,T}(ntuple(j -> @inbounds(arg2[i,j]), Val(NC2)))
        row3 = SVector{NC3,T}(ntuple(j -> @inbounds(arg3[i,j]), Val(NC3)))
        row4 = SVector{NC4,T}(ntuple(j -> @inbounds(arg4[i,j]), Val(NC4)))
        row5 = SVector{NC5,T}(ntuple(j -> @inbounds(arg5[i,j]), Val(NC5)))
        result = f(row1, row2, row3, row4, row5)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

# ============================================================================
# Result writers
# ============================================================================

@inline function _cuda_write_result!(output, i, result::Number, ::Val{1})
    @inbounds output[i, 1] = result
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SVector{N,T}, ::Val{N}) where {N,T}
    for j in 1:N
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

@inline function _cuda_write_result!(output, i, result::SMatrix{M,N,T}, ::Val{MN}) where {M,N,T,MN}
    for j in 1:MN
        @inbounds output[i, j] = result[j]
    end
    return nothing
end

# ============================================================================
# Dispatch functions
# ============================================================================

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}) where T
    n = size(arg1, 1)
    ncols1 = size(arg1, 2)
    out_cols = size(output, 2)
    kernel = @cuda launch=false _cuda_map_rows_kernel_1arg(f, output, arg1, Val(ncols1), Val(out_cols))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, Val(ncols1), Val(out_cols); threads=threads, blocks=blocks)
end

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}) where T
    n = size(arg1, 1)
    kernel = @cuda launch=false _cuda_map_rows_kernel_2args(f, output, arg1, arg2,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(output,2)))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(output,2));
        threads=threads, blocks=blocks)
end

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}, arg3::CuMatrix{T}) where T
    n = size(arg1, 1)
    kernel = @cuda launch=false _cuda_map_rows_kernel_3args(f, output, arg1, arg2, arg3,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(output,2)))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(output,2));
        threads=threads, blocks=blocks)
end

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}, arg3::CuMatrix{T}, arg4::CuMatrix{T}) where T
    n = size(arg1, 1)
    kernel = @cuda launch=false _cuda_map_rows_kernel_4args(f, output, arg1, arg2, arg3, arg4,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(arg4,2)), Val(size(output,2)))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(arg4,2)), Val(size(output,2));
        threads=threads, blocks=blocks)
end

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, arg1::CuMatrix{T}, arg2::CuMatrix{T}, arg3::CuMatrix{T}, arg4::CuMatrix{T}, arg5::CuMatrix{T}) where T
    n = size(arg1, 1)
    kernel = @cuda launch=false _cuda_map_rows_kernel_5args(f, output, arg1, arg2, arg3, arg4, arg5,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(arg4,2)), Val(size(arg5,2)), Val(size(output,2)))
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, arg1, arg2, arg3, arg4, arg5,
        Val(size(arg1,2)), Val(size(arg2,2)), Val(size(arg3,2)), Val(size(arg4,2)), Val(size(arg5,2)), Val(size(output,2));
        threads=threads, blocks=blocks)
end

# ============================================================================
# Main entry point
# ============================================================================

"""
    _map_rows_gpu_cuda(f, arg1::CuMatrix, rest::CuMatrix...)

GPU-accelerated row-wise map for CUDA arrays.
Each thread processes one row, constructing SVector from row data,
calling f, and writing the result back.
"""
function _map_rows_gpu_cuda(f, arg1::CuMatrix{T}, rest::CuMatrix...) where T
    n = size(arg1, 1)

    # Determine output size by evaluating f on first row (on CPU)
    arg1_row1 = Array(view(arg1, 1:1, :))[1, :]
    first_rows = (SVector{size(arg1,2),T}(arg1_row1...),)
    for m in rest
        m_row1 = Array(view(m, 1:1, :))[1, :]
        first_rows = (first_rows..., SVector{size(m,2),T}(m_row1...))
    end
    sample_out = f(first_rows...)

    if sample_out isa SVector
        out_cols = length(sample_out)
    elseif sample_out isa SMatrix
        out_cols = length(sample_out)
    else
        out_cols = 1
    end

    output = CUDA.zeros(T, n, out_cols)
    _cuda_map_rows_dispatch(f, output, arg1, rest...)
    return output
end
