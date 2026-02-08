# cudss_solver.jl -- cuDSS single-GPU sparse direct solver via CUDSS_jll
#
# Caches cuDSS analysis (reordering + symbolic factorization) per sparsity pattern.
# Subsequent solves with the same pattern skip analysis and only re-factorize + solve.

using CUDSS_jll
using CUDA.CUSPARSE: CuSparseMatrixCSR

# ============================================================================
# Constants
# ============================================================================

const cudssHandle_t = Ptr{Cvoid}
const cudssConfig_t = Ptr{Cvoid}
const cudssData_t = Ptr{Cvoid}
const cudssMatrix_t = Ptr{Cvoid}

const CUDSS_STATUS_SUCCESS = UInt32(0)

# Phases (can be OR'd together)
const CUDSS_PHASE_ANALYSIS = Cint(3)       # REORDERING | SYMBOLIC
const CUDSS_PHASE_FACTORIZATION = Cint(4)
const CUDSS_PHASE_SOLVE = Cint(1008)

# Matrix types
const CUDSS_MTYPE_GENERAL = UInt32(0)
const CUDSS_MTYPE_SYMMETRIC = UInt32(1)

# Matrix view types
const CUDSS_MVIEW_FULL = UInt32(0)

# Index base
const CUDSS_BASE_ZERO = UInt32(0)
const CUDSS_BASE_ONE = UInt32(1)

# Layout
const CUDSS_LAYOUT_COL_MAJOR = UInt32(0)

# CUDA data type mapping
_cuda_data_type(::Type{Float32}) = UInt32(0)   # CUDA_R_32F
_cuda_data_type(::Type{Float64}) = UInt32(1)   # CUDA_R_64F
_cuda_data_type(::Type{Int32}) = UInt32(10)    # CUDA_R_32I
_cuda_data_type(::Type{Int64}) = UInt32(24)    # CUDA_R_64I

# ============================================================================
# ccall wrappers
# ============================================================================

function _cudss_create(handle_ref::Ref{cudssHandle_t})
    status = @ccall libcudss.cudssCreate(handle_ref::Ptr{cudssHandle_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssCreate failed with status $status")
    return nothing
end

function _cudss_destroy(handle::cudssHandle_t)
    status = @ccall libcudss.cudssDestroy(handle::cudssHandle_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDestroy failed with status $status")
    return nothing
end

function _cudss_config_create(config_ref::Ref{cudssConfig_t})
    status = @ccall libcudss.cudssConfigCreate(config_ref::Ptr{cudssConfig_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigCreate failed with status $status")
    return nothing
end

function _cudss_config_destroy(config::cudssConfig_t)
    status = @ccall libcudss.cudssConfigDestroy(config::cudssConfig_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssConfigDestroy failed with status $status")
    return nothing
end

function _cudss_data_create(handle::cudssHandle_t, data_ref::Ref{cudssData_t})
    status = @ccall libcudss.cudssDataCreate(handle::cudssHandle_t,
                                              data_ref::Ptr{cudssData_t})::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataCreate failed with status $status")
    return nothing
end

function _cudss_data_destroy(handle::cudssHandle_t, data::cudssData_t)
    status = @ccall libcudss.cudssDataDestroy(handle::cudssHandle_t,
                                               data::cudssData_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssDataDestroy failed with status $status")
    return nothing
end

function _cudss_matrix_create_csr(matrix_ref::Ref{cudssMatrix_t},
                                   nrows::Int64, ncols::Int64, nnz::Int64,
                                   row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
                                   col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
                                   index_type::UInt32, value_type::UInt32,
                                   mtype::UInt32, mview::UInt32, index_base::UInt32)
    status = @ccall libcudss.cudssMatrixCreateCsr(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, nnz::Int64,
        row_offsets::CuPtr{Cvoid}, row_end::CuPtr{Cvoid},
        col_indices::CuPtr{Cvoid}, values::CuPtr{Cvoid},
        index_type::UInt32, value_type::UInt32,
        mtype::UInt32, mview::UInt32, index_base::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateCsr failed with status $status")
    return nothing
end

function _cudss_matrix_create_dn(matrix_ref::Ref{cudssMatrix_t},
                                  nrows::Int64, ncols::Int64, ld::Int64,
                                  values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)
    status = @ccall libcudss.cudssMatrixCreateDn(
        matrix_ref::Ptr{cudssMatrix_t},
        nrows::Int64, ncols::Int64, ld::Int64,
        values::CuPtr{Cvoid}, value_type::UInt32, layout::UInt32)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixCreateDn failed with status $status")
    return nothing
end

function _cudss_matrix_destroy(matrix::cudssMatrix_t)
    status = @ccall libcudss.cudssMatrixDestroy(matrix::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssMatrixDestroy failed with status $status")
    return nothing
end

function _cudss_execute(handle::cudssHandle_t, phase::Cint,
                        config::cudssConfig_t, data::cudssData_t,
                        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)
    status = @ccall libcudss.cudssExecute(
        handle::cudssHandle_t, phase::Cint,
        config::cudssConfig_t, data::cudssData_t,
        matrix::cudssMatrix_t, solution::cudssMatrix_t, rhs::cudssMatrix_t)::UInt32
    status == CUDSS_STATUS_SUCCESS || error("cudssExecute (phase=$phase) failed with status $status")
    return nothing
end

# ============================================================================
# Factorization cache
# ============================================================================

"""
Cache key for cuDSS factorizations.  Hash is cheap O(1) on (m, n, nnz, mtype).
Identity is confirmed by deep GPU comparison of rowPtr and colVal.
"""
struct CuDSSCacheKey
    m::Int
    n::Int
    nnz_val::Int
    mtype::UInt32
    rowPtr::CuVector{Int32}
    colVal::CuVector{Int32}
end

Base.hash(k::CuDSSCacheKey, h::UInt) = hash((k.m, k.n, k.nnz_val, k.mtype), h)

function Base.:(==)(a::CuDSSCacheKey, b::CuDSSCacheKey)
    a.m == b.m && a.n == b.n && a.nnz_val == b.nnz_val && a.mtype == b.mtype || return false
    all(a.rowPtr .== b.rowPtr) && all(a.colVal .== b.colVal)
end

"""
Cached cuDSS state: handles, descriptors, and persistent GPU buffers.
The cuDSS descriptors hold pointers into the persistent buffers, so the
buffers must stay alive as long as the cache entry exists.
"""
mutable struct CuDSSCacheEntry{T}
    handle::cudssHandle_t
    config::cudssConfig_t
    data::cudssData_t
    matrix_desc::cudssMatrix_t
    solution_desc::cudssMatrix_t
    rhs_desc::cudssMatrix_t
    # Persistent GPU buffers (cuDSS descriptors point into these)
    rowPtr_0::CuVector{Int32}
    colVal_0::CuVector{Int32}
    nzVal_buf::CuVector{T}
    x_buf::CuVector{T}
    rhs_buf::CuVector{T}
end

const _cudss_cache = Dict{CuDSSCacheKey, Any}()

"""
    clear_cudss_cache!()

Destroy all cached cuDSS factorizations and free associated GPU memory.
"""
function MultiGridBarrier.clear_cudss_cache!()
    for (_, entry) in _cudss_cache
        _cudss_matrix_destroy(entry.rhs_desc)
        _cudss_matrix_destroy(entry.solution_desc)
        _cudss_matrix_destroy(entry.matrix_desc)
        _cudss_data_destroy(entry.handle, entry.data)
        _cudss_config_destroy(entry.config)
        _cudss_destroy(entry.handle)
    end
    empty!(_cudss_cache)
    nothing
end

# ============================================================================
# Solver: A \ b  (with cached analysis)
# ============================================================================

"""
    _cudss_solve(A::CuSparseMatrixCSR{T,Int32}, b::CuVector{T}, mtype::UInt32) where T

Solve A*x = b using cuDSS.  Caches the analysis phase (reordering + symbolic
factorization) per sparsity pattern.  Subsequent calls with the same pattern
only re-run numerical factorization + solve.
"""
function _cudss_solve(A::CuSparseMatrixCSR{T,Int32}, b::CuVector{T}, mtype::UInt32) where T
    m = size(A, 1)
    n = size(A, 2)
    nnz_val = nnz(A)

    # Lookup: temporary key (no copy — A is alive for duration of this call)
    lookup_key = CuDSSCacheKey(m, n, nnz_val, mtype, A.rowPtr, A.colVal)
    entry = get(_cudss_cache, lookup_key, nothing)::Union{CuDSSCacheEntry{T}, Nothing}

    if entry !== nothing
        # Cache hit: update values, re-factorize, solve
        copyto!(entry.nzVal_buf, A.nzVal)
        copyto!(entry.rhs_buf, b)

        _cudss_execute(entry.handle, CUDSS_PHASE_FACTORIZATION, entry.config, entry.data,
                       entry.matrix_desc, entry.solution_desc, entry.rhs_desc)
        _cudss_execute(entry.handle, CUDSS_PHASE_SOLVE, entry.config, entry.data,
                       entry.matrix_desc, entry.solution_desc, entry.rhs_desc)

        return copy(entry.x_buf)
    end

    # Cache miss: create everything, run full analysis + factorization + solve

    # Persistent GPU buffers — cuDSS descriptors will point into these
    rowPtr_0 = A.rowPtr .- Int32(1)
    colVal_0 = A.colVal .- Int32(1)
    nzVal_buf = copy(A.nzVal)
    x_buf = CUDA.zeros(T, m)
    rhs_buf = copy(b)

    # Create handle
    handle_ref = Ref{cudssHandle_t}(C_NULL)
    _cudss_create(handle_ref)
    handle = handle_ref[]

    # Create config
    config_ref = Ref{cudssConfig_t}(C_NULL)
    _cudss_config_create(config_ref)
    config = config_ref[]

    # Create data
    data_ref = Ref{cudssData_t}(C_NULL)
    _cudss_data_create(handle, data_ref)
    data = data_ref[]

    # Create CSR matrix descriptor (points to persistent buffers)
    matrix_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_csr(matrix_ref,
        Int64(m), Int64(n), Int64(nnz_val),
        reinterpret(CuPtr{Cvoid}, pointer(rowPtr_0)),
        CuPtr{Cvoid}(0),
        reinterpret(CuPtr{Cvoid}, pointer(colVal_0)),
        reinterpret(CuPtr{Cvoid}, pointer(nzVal_buf)),
        _cuda_data_type(Int32), _cuda_data_type(T),
        mtype, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO)
    matrix_desc = matrix_ref[]

    # Create dense solution descriptor (points to persistent x_buf)
    solution_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(solution_ref,
        Int64(m), Int64(1), Int64(m),
        reinterpret(CuPtr{Cvoid}, pointer(x_buf)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    solution_desc = solution_ref[]

    # Create dense RHS descriptor (points to persistent rhs_buf)
    rhs_ref = Ref{cudssMatrix_t}(C_NULL)
    _cudss_matrix_create_dn(rhs_ref,
        Int64(m), Int64(1), Int64(m),
        reinterpret(CuPtr{Cvoid}, pointer(rhs_buf)),
        _cuda_data_type(T), CUDSS_LAYOUT_COL_MAJOR)
    rhs_desc = rhs_ref[]

    # Execute: analysis, factorization, solve
    _cudss_execute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix_desc, solution_desc, rhs_desc)
    _cudss_execute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix_desc, solution_desc, rhs_desc)
    _cudss_execute(handle, CUDSS_PHASE_SOLVE, config, data, matrix_desc, solution_desc, rhs_desc)

    # Store in cache (copy rowPtr/colVal so key survives after A is GC'd)
    store_key = CuDSSCacheKey(m, n, nnz_val, mtype, copy(A.rowPtr), copy(A.colVal))
    new_entry = CuDSSCacheEntry{T}(handle, config, data, matrix_desc, solution_desc, rhs_desc,
                                    rowPtr_0, colVal_0, nzVal_buf, x_buf, rhs_buf)
    _cudss_cache[store_key] = new_entry

    return copy(x_buf)
end

"""
    Base.:\\(A::CuSparseMatrixCSR{T,Int32}, b::CuVector{T}) where T

Solve A*x = b using cuDSS for a general (non-symmetric) sparse matrix on GPU.
"""
function Base.:\(A::CuSparseMatrixCSR{T,Int32}, b::CuVector{T}) where T
    _cudss_solve(A, b, CUDSS_MTYPE_GENERAL)
end

"""
    Base.:\\(A::Symmetric{T, <:CuSparseMatrixCSR{T,Int32}}, b::CuVector{T}) where T

Solve A*x = b using cuDSS for a symmetric matrix on GPU.
Uses symmetric mode with LDLT factorization.
"""
function Base.:\(A::Symmetric{T, <:CuSparseMatrixCSR{T,Int32}}, b::CuVector{T}) where T
    _cudss_solve(parent(A), b, CUDSS_MTYPE_SYMMETRIC)
end
