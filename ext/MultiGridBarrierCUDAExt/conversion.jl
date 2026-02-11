# conversion.jl -- native_to_cuda / cuda_to_native type conversion + convenience functions

using CUDA.CUSPARSE: CuSparseMatrixCSR
import MultiGridBarrier: AMGBSOL
using MultiGridBarrier.Mesh3d: fem3d

# Device-agnostic CuSparseMatrixCSR → SparseMatrixCSC conversion.
# Array() works from any device context, so we transfer CSR components to CPU
# and reconstruct CSC on CPU. Avoids CUSPARSE calls that require the correct device.
function _cusparse_to_cpu(A::CuSparseMatrixCSR{T,Ti}) where {T,Ti}
    m, n = size(A)
    CUDA.synchronize()
    rp = Vector{Ti}(Array(A.rowPtr))
    cv = Vector{Ti}(Array(A.colVal))
    nz = Vector{T}(Array(A.nzVal))
    if length(nz) == 0
        return spzeros(T, m, n)
    end
    # CSR of A ≡ CSC of Aᵀ
    At_csc = SparseMatrixCSC{T,Ti}(n, m, rp, cv, nz)
    return SparseMatrixCSC{T,Ti}(sparse(At_csc'))
end

# Default element block sizes per discretization type
_default_block_size(::FEM1D) = 2
_default_block_size(::FEM2D) = 7
_default_block_size(d::FEM3D) = (d.k + 1)^3

"""
    native_to_cuda(g_native::Geometry; Ti=Int32, structured=true, block_size=auto)

Convert a native Geometry object (with Julia CPU arrays) to use CUDA GPU types.

# Type mapping
- `Matrix{T}` → `CuMatrix{T}`
- `Vector{T}` → `CuVector{T}`
- `SparseMatrixCSC{T,Int}` → `CuSparseMatrixCSR{T,Int32}`

When `structured=true` (default), operators are further converted to `BlockDiagGPU`
and refine/coarsen to `VBlockDiagGPU`/`HBlockDiagGPU`, enabling batched block
operations in the solver and eliminating most SpGEMMs.

# Arguments
- `g_native`: Native Geometry with Julia arrays
- `Ti`: Index type for sparse matrices (default: `Int32`)
- `structured`: Convert to block-structured types (default: `true`)
- `block_size`: Element block size (default: auto-detected from discretization — 2 for fem1d, 7 for fem2d)
"""
function MultiGridBarrier.native_to_cuda(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization};
                        Ti::Type{<:Integer}=Int32,
                        structured::Bool=true,
                        block_size::Int=_default_block_size(g_native.discretization)) where {T, Discretization}
    # Convert x (geometry coordinates) to CuMatrix
    x_cuda = CuMatrix{T}(g_native.x)

    # Convert w (weights) to CuVector
    w_cuda = CuVector{T}(g_native.w)

    # Helper to convert sparse matrices: CSC -> CSR with Int32 indices
    convert_sparse = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    MType = CuSparseMatrixCSR{T,Ti}

    # Convert all operators
    operators_cuda = Dict{Symbol, MType}()
    for key in sort(collect(keys(g_native.operators)))
        operators_cuda[key] = convert_sparse(g_native.operators[key])
    end

    # Convert all subspace matrices
    subspaces_cuda = Dict{Symbol, Vector{MType}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspace_vec = g_native.subspaces[key]
        cuda_vec = Vector{MType}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            cuda_vec[i] = convert_sparse(subspace_vec[i])
        end
        subspaces_cuda[key] = cuda_vec
    end

    # Convert refine and coarsen
    refine_cuda = Vector{MType}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        refine_cuda[i] = convert_sparse(g_native.refine[i])
    end

    coarsen_cuda = Vector{MType}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        coarsen_cuda[i] = convert_sparse(g_native.coarsen[i])
    end

    XType = CuMatrix{T}
    WType = CuVector{T}
    DType = typeof(g_native.discretization)

    g_cuda = Geometry{T, XType, WType, MType, DType}(
        g_native.discretization,
        x_cuda,
        w_cuda,
        subspaces_cuda,
        operators_cuda,
        refine_cuda,
        coarsen_cuda
    )

    if structured
        g_cuda = _structurize_geometry(g_cuda, block_size)
    end

    return g_cuda
end

"""
    native_to_cuda(g_native::Geometry{T, Matrix{T}, Vector{T}, Matrix{T}, Discretization})

Convert a native dense Geometry (spectral discretizations) to CUDA GPU types.
All matrices become CuMatrix, vectors become CuVector.
"""
function MultiGridBarrier.native_to_cuda(g_native::Geometry{T, Matrix{T}, Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Discretization};
                        kwargs...) where {T, Discretization}
    x_cuda = CuMatrix{T}(g_native.x)
    w_cuda = CuVector{T}(g_native.w)

    operators_cuda = Dict{Symbol, CuMatrix{T}}()
    for key in sort(collect(keys(g_native.operators)))
        operators_cuda[key] = CuMatrix{T}(g_native.operators[key])
    end

    subspaces_cuda = Dict{Symbol, Vector{CuMatrix{T}}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspaces_cuda[key] = [CuMatrix{T}(s) for s in g_native.subspaces[key]]
    end

    refine_cuda = [CuMatrix{T}(r) for r in g_native.refine]
    coarsen_cuda = [CuMatrix{T}(c) for c in g_native.coarsen]

    Geometry{T, CuMatrix{T}, CuVector{T}, CuMatrix{T}, CuMatrix{T}, CuMatrix{T}, CuMatrix{T}, Discretization}(
        g_native.discretization, x_cuda, w_cuda,
        subspaces_cuda, operators_cuda, refine_cuda, coarsen_cuda)
end

"""
    cuda_to_native(g_cuda::Geometry) — sparse FEM variant

Convert a CUDA Geometry object with sparse operators back to native Julia CPU types.
"""
function MultiGridBarrier.cuda_to_native(g_cuda::Geometry{T, <:CuMatrix{T}, <:CuVector{T}, <:CuSparseMatrixCSR{T}, <:CuSparseMatrixCSR{T}, <:CuSparseMatrixCSR{T}, <:CuSparseMatrixCSR{T}, Discretization}) where {T, Discretization}
    x_native = Matrix{T}(Array(g_cuda.x))
    w_native = Vector{T}(Array(g_cuda.w))

    Ti = Int  # Use Int for native
    convert_sparse = function(op)
        A = _cusparse_to_cpu(op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    operators_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g_cuda.operators)))
        operators_native[key] = convert_sparse(g_cuda.operators[key])
    end

    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(g_cuda.subspaces)))
        subspace_vec = g_cuda.subspaces[key]
        native_vec = Vector{SparseMatrixCSC{T,Ti}}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            native_vec[i] = convert_sparse(subspace_vec[i])
        end
        subspaces_native[key] = native_vec
    end

    refine_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_cuda.refine))
    for i in 1:length(g_cuda.refine)
        refine_native[i] = convert_sparse(g_cuda.refine[i])
    end

    coarsen_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_cuda.coarsen))
    for i in 1:length(g_cuda.coarsen)
        coarsen_native[i] = convert_sparse(g_cuda.coarsen[i])
    end

    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Ti}, Discretization}(
        g_cuda.discretization,
        x_native,
        w_native,
        subspaces_native,
        operators_native,
        refine_native,
        coarsen_native
    )
end

"""
    cuda_to_native(g_cuda::Geometry) — dense spectral variant

Convert a CUDA Geometry object with dense operators back to native Julia CPU types.
"""
function MultiGridBarrier.cuda_to_native(g_cuda::Geometry{T, <:CuMatrix{T}, <:CuVector{T}, <:CuMatrix{T}, <:CuMatrix{T}, <:CuMatrix{T}, <:CuMatrix{T}, Discretization}) where {T, Discretization}
    x_native = Matrix{T}(Array(g_cuda.x))
    w_native = Vector{T}(Array(g_cuda.w))

    operators_native = Dict{Symbol, Matrix{T}}()
    for key in sort(collect(keys(g_cuda.operators)))
        operators_native[key] = Matrix{T}(Array(g_cuda.operators[key]))
    end

    subspaces_native = Dict{Symbol, Vector{Matrix{T}}}()
    for key in sort(collect(keys(g_cuda.subspaces)))
        subspaces_native[key] = [Matrix{T}(Array(s)) for s in g_cuda.subspaces[key]]
    end

    refine_native = [Matrix{T}(Array(r)) for r in g_cuda.refine]
    coarsen_native = [Matrix{T}(Array(c)) for c in g_cuda.coarsen]

    Geometry{T, Matrix{T}, Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Discretization}(
        g_cuda.discretization, x_native, w_native,
        subspaces_native, operators_native, refine_native, coarsen_native)
end

"""
    cuda_to_native(g_cuda::Geometry) — structured FEM variant (BlockDiagGPU etc.)

Convert a structured CUDA Geometry back to native Julia CPU types.
"""
function MultiGridBarrier.cuda_to_native(g_cuda::Geometry{T, <:CuMatrix{T}, <:CuVector{T}, <:Any, <:Any, <:Any, <:Any, Discretization}) where {T, Discretization}
    # Structured geometry: operators are BlockDiagGPU etc., convert via _to_cusparse first
    x_native = Matrix{T}(Array(g_cuda.x))
    w_native = Vector{T}(Array(g_cuda.w))

    Ti = Int
    convert_to_native = function(op)
        sparse_op = op isa CuSparseMatrixCSR ? op : _to_cusparse(op)
        A = _cusparse_to_cpu(sparse_op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    operators_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g_cuda.operators)))
        operators_native[key] = convert_to_native(g_cuda.operators[key])
    end

    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(g_cuda.subspaces)))
        subspace_vec = g_cuda.subspaces[key]
        native_vec = Vector{SparseMatrixCSC{T,Ti}}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            native_vec[i] = convert_to_native(subspace_vec[i])
        end
        subspaces_native[key] = native_vec
    end

    refine_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_cuda.refine))
    for i in 1:length(g_cuda.refine)
        refine_native[i] = convert_to_native(g_cuda.refine[i])
    end

    coarsen_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_cuda.coarsen))
    for i in 1:length(g_cuda.coarsen)
        coarsen_native[i] = convert_to_native(g_cuda.coarsen[i])
    end

    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Ti}, Discretization}(
        g_cuda.discretization,
        x_native,
        w_native,
        subspaces_native,
        operators_native,
        refine_native,
        coarsen_native
    )
end

_convert_cuda_to_native(x::CuMatrix) = Matrix(Array(x))
_convert_cuda_to_native(x::CuVector) = Vector(Array(x))
_convert_cuda_to_native(x::CuSparseMatrixCSR) = _cusparse_to_cpu(x)
_convert_cuda_to_native(x) = x

function _convert_cuda_value(value)
    if isa(value, CuMatrix) || isa(value, CuVector) || isa(value, CuSparseMatrixCSR)
        return _convert_cuda_to_native(value)
    elseif isa(value, Array)
        return map(_convert_cuda_value, value)
    else
        return value
    end
end

# ============================================================================
# Convenience functions
# ============================================================================

"""
    fem1d_cuda(::Type{T}=Float64; kwargs...)

Create a CUDA-based Geometry from fem1d parameters.
"""
function MultiGridBarrier.fem1d_cuda(::Type{T}=Float64; kwargs...) where {T}
    g_native = fem1d(T; kwargs...)
    return native_to_cuda(g_native)
end

"""
    fem1d_cuda_solve(::Type{T}=Float64; kwargs...)

Solve a fem1d problem using amgb with CUDA GPU types.
"""
function MultiGridBarrier.fem1d_cuda_solve(::Type{T}=Float64; kwargs...) where {T}
    g = fem1d_cuda(T; kwargs...)
    return amgb(g; kwargs...)
end

"""
    fem2d_cuda(::Type{T}=Float64; structured=true, kwargs...)

Create a CUDA-based Geometry from fem2d parameters.

When `structured=true` (default), uses BlockDiagGPU structured types for
operators and V/HBlockDiagGPU for refine/coarsen, enabling batched block
operations in the solver.
"""
function MultiGridBarrier.fem2d_cuda(::Type{T}=Float64; structured::Bool=true, kwargs...) where {T}
    g_native = fem2d(T; kwargs...)
    return native_to_cuda(g_native; structured=structured)
end

"""
    fem2d_cuda_solve(::Type{T}=Float64; structured=true, kwargs...)

Solve a fem2d problem using amgb with CUDA GPU types.

When `structured=true` (default), uses BlockDiagGPU structured types for
operators and V/HBlockDiagGPU for refine/coarsen in the Geometry, so that
amg_helper produces structured AMG objects via dispatch. This replaces
~34 SpGEMMs per Newton step with batched block operations, keeping only ~2
SpGEMMs for the R'*H*R restriction.
"""
function MultiGridBarrier.fem2d_cuda_solve(::Type{T}=Float64; structured::Bool=true, kwargs...) where {T}
    g = fem2d_cuda(T; structured=structured, kwargs...)
    return amgb(g; kwargs...)
end

"""
    fem3d_cuda(::Type{T}=Float64; structured=true, kwargs...)

Create a CUDA-based Geometry from fem3d parameters.

When `structured=true` (default), uses BlockDiagGPU structured types for
operators and V/HBlockDiagGPU for refine/coarsen, enabling batched block
operations in the solver. Block size is `(k+1)^3` where `k` is the
polynomial order (default 3, giving block size 64).
"""
function MultiGridBarrier.fem3d_cuda(::Type{T}=Float64; structured::Bool=true, kwargs...) where {T}
    g_native = fem3d(T; kwargs...)
    return native_to_cuda(g_native; structured=structured)
end

"""
    fem3d_cuda_solve(::Type{T}=Float64; structured=true, kwargs...)

Solve a fem3d problem using amgb with CUDA GPU types.
"""
function MultiGridBarrier.fem3d_cuda_solve(::Type{T}=Float64; structured::Bool=true, kwargs...) where {T}
    g = fem3d_cuda(T; structured=structured, kwargs...)
    return amgb(g; kwargs...)
end

"""
    spectral1d_cuda(::Type{T}=Float64; kwargs...)

Create a CUDA-based Geometry from spectral1d parameters (dense matrices).
"""
function MultiGridBarrier.spectral1d_cuda(::Type{T}=Float64; kwargs...) where {T}
    g_native = spectral1d(T; kwargs...)
    return native_to_cuda(g_native)
end

"""
    spectral1d_cuda_solve(::Type{T}=Float64; kwargs...)

Solve a spectral1d problem using amgb with CUDA GPU types.
"""
function MultiGridBarrier.spectral1d_cuda_solve(::Type{T}=Float64; kwargs...) where {T}
    g = spectral1d_cuda(T; kwargs...)
    return amgb(g; kwargs...)
end

"""
    spectral2d_cuda(::Type{T}=Float64; kwargs...)

Create a CUDA-based Geometry from spectral2d parameters (dense matrices).
"""
function MultiGridBarrier.spectral2d_cuda(::Type{T}=Float64; kwargs...) where {T}
    g_native = spectral2d(T; kwargs...)
    return native_to_cuda(g_native)
end

"""
    spectral2d_cuda_solve(::Type{T}=Float64; kwargs...)

Solve a spectral2d problem using amgb with CUDA GPU types.
"""
function MultiGridBarrier.spectral2d_cuda_solve(::Type{T}=Float64; kwargs...) where {T}
    g = spectral2d_cuda(T; kwargs...)
    return amgb(g; kwargs...)
end

# cuda_to_native for AMGBSOL
function MultiGridBarrier.cuda_to_native(sol::AMGBSOL{T, <:Any, <:Any, Discretization}) where {T, Discretization}
    z_native = _convert_cuda_to_native(sol.z)

    function convert_namedtuple(nt)
        nt === nothing && return nothing
        converted_fields = []
        for (name, value) in pairs(nt)
            push!(converted_fields, name => _convert_cuda_value(value))
        end
        return NamedTuple(converted_fields)
    end

    SOL_feasibility_native = convert_namedtuple(sol.SOL_feasibility)
    SOL_main_native = convert_namedtuple(sol.SOL_main)
    geometry_native = cuda_to_native(sol.geometry)

    return AMGBSOL(
        z_native,
        SOL_feasibility_native,
        SOL_main_native,
        sol.log,
        geometry_native
    )
end
