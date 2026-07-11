# map_rows_gpu.jl -- CUDA kernel implementation of map_rows on CuMatrix
#
# One variadic kernel handles any number of same-eltype CuMatrix data
# arguments (the piecewise barrier of `intersect` reaches 6+: select grid,
# per-piece A/b grids, Dz — and the phase-I slack map adds z0 and w).
# Each thread processes one row: it gathers an SVector per argument, calls f,
# and writes the SVector/SMatrix/scalar result into its output row. Grids are
# indexed directly — no per-call transpose copies (the generic broadcast path
# in src/utils.jl pays one per argument per evaluation; measured ~2-6x slower
# on the A40, 5-10% end-to-end).

using StaticArrays

# ============================================================================
# CUDA kernel
# ============================================================================

# `args` is a tuple of CuDeviceMatrix; NCs is the compile-time tuple of their
# column counts, OCols the output column count.
function _cuda_map_rows_kernel(f, output, args, ::Val{NCs}, ::Val{OCols}) where {NCs, OCols}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(output, 1)
        rows = _cuda_svector_rows(args, Val(NCs), i)
        result = f(rows...)
        _cuda_write_result!(output, i, result, Val(OCols))
    end
    return nothing
end

# One SVector row per data argument. @generated so every SVector width is a
# compile-time constant even though the arguments have heterogeneous widths.
# The emitted AST must stay closure-free (a @generated function may not return
# an expression containing closures), so the element reads are fully unrolled.
@generated function _cuda_svector_rows(args::Tuple, ::Val{NCs}, i) where {NCs}
    svecs = map(1:length(NCs)) do k
        elems = [:(@inbounds args[$k][i, $j]) for j in 1:NCs[k]]
        :(SVector{$(NCs[k]), eltype(args[$k])}($(elems...)))
    end
    :(tuple($(svecs...)))
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
# Dispatch
# ============================================================================

function _cuda_map_rows_dispatch(f, output::CuMatrix{T}, args::CuMatrix{T}...) where T
    n = size(output, 1)
    ncs = Val(map(a -> size(a, 2), args))
    ocols = Val(size(output, 2))
    kernel = @cuda launch=false _cuda_map_rows_kernel(f, output, args, ncs, ocols)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    kernel(f, output, args, ncs, ocols; threads=threads, blocks=blocks)
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
    out_cols = _map_rows_out_width(f, arg1, rest...)
    output = CUDA.zeros(T, n, out_cols)
    _cuda_map_rows_dispatch(f, output, arg1, rest...)
    return output
end

# Number of output columns of `f` over SVector rows: the length of its
# SVector/SMatrix result, or 1 for a scalar. Resolved by type inference so the
# hot path makes no device-to-host transfer; when inference does not yield a
# concrete static type, fall back to sampling row 1 on the CPU (one small
# transfer — the previous behavior).
function _map_rows_out_width(f, arg1::CuMatrix, rest::CuMatrix...)
    RT = Base.promote_op(f, SVector{size(arg1,2),eltype(arg1)},
                         (SVector{size(m,2),eltype(m)} for m in rest)...)
    if isconcretetype(RT)
        RT <: StaticArray && return length(RT)
        RT <: Number && return 1
    end
    row1(m) = SVector{size(m,2),eltype(m)}(Array(view(m, 1:1, :))[1, :]...)
    sample_out = f(row1(arg1), map(row1, rest)...)
    return sample_out isa Union{SVector,SMatrix} ? length(sample_out) : 1
end
