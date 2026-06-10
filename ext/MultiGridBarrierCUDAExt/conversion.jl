# conversion.jl -- native_to_cuda / cuda_to_native type conversion

using CUDA.CUSPARSE: CuSparseMatrixCSR
using StaticArrays: SVector
import MultiGridBarrier: MGBSOL, Geometry, MultiGrid, FEM1D, FEM2D_P1, FEM2D_P2, FEM3D,
                         Convex, map_rows, AMG, MGBProblem,
                         native_to_device, device_to_native, CPUDevice, CUDADevice

# Device-agnostic CuSparseMatrixCSR → SparseMatrixCSC conversion.
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

# ============================================================================
# native_to_cuda(::Geometry)
# ============================================================================

"""
    native_to_cuda(g::Geometry; Ti=Int32) -> Geometry

Convert a native (CPU) `Geometry` to CUDA GPU types.

# Type mapping
- `Array{T,3}` mesh tensor → `CuArray{T,3}`
- `Vector{T}` → `CuVector{T}`
- `BlockDiag` operators → CuArray-backed `BlockDiag` (structured FEM)
- Dense `Matrix{T}` operators → `CuMatrix{T}` (spectral)
"""
# BlockDiag-operator variant: FEM Geometry whose `operators` came in via
# `subdivide` (or `geometric_mg(...; structured=true).geometry`). Carry the
# BlockDiag to GPU as a CuArray-backed BlockDiag; subspaces stay sparse CSR.
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Array{T,3}, Vector{T},
                                                      MultiGridBarrier.BlockDiag{T, A3},
                                                      Discretization};
                                         Ti::Type{<:Integer}=Int32) where {T, A3<:Array{T,3}, Discretization}
    x_cuda = CuArray{T,3}(g.x)
    w_cuda = CuVector{T}(g.w)

    function op_to_gpu(op::MultiGridBarrier.BlockDiag{T})
        gpu_data = CuArray{T,3}(op.data)
        MultiGridBarrier.BlockDiag{T, typeof(gpu_data)}(op.p, op.q, op.N, gpu_data)
    end

    sample_op = op_to_gpu(first(values(g.operators)))
    OpType = typeof(sample_op)

    operators_cuda = Dict{Symbol, OpType}(
        key => op_to_gpu(g.operators[key]) for key in keys(g.operators))

    Geometry{T, CuArray{T,3}, CuVector{T}, OpType, Discretization}(
        g.discretization, g.t, x_cuda, w_cuda, operators_cuda)
end

# Dense spectral variant.
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Array{T,3}, Vector{T}, Matrix{T}, Discretization};
                                         kwargs...) where {T, Discretization}
    x_cuda = CuArray{T,3}(g.x)
    w_cuda = CuVector{T}(g.w)

    operators_cuda = Dict{Symbol, CuMatrix{T}}()
    for key in sort(collect(keys(g.operators)))
        operators_cuda[key] = CuMatrix{T}(g.operators[key])
    end

    Geometry{T, CuArray{T,3}, CuVector{T}, CuMatrix{T}, Discretization}(
        g.discretization, g.t, x_cuda, w_cuda, operators_cuda)
end

# ============================================================================
# cuda_to_native: Geometry/MGBSOL → CPU
# ============================================================================

# Dense spectral Geometry.
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuArray{T,3}, <:CuVector{T},
                                                      <:CuMatrix{T},
                                                      Discretization}) where {T, Discretization}
    x_native = Array{T,3}(Array(g.x))
    w_native = Vector{T}(Array(g.w))

    operators_native = Dict{Symbol, Matrix{T}}()
    for key in sort(collect(keys(g.operators)))
        operators_native[key] = Matrix{T}(Array(g.operators[key]))
    end

    Geometry{T, Array{T,3}, Vector{T}, Matrix{T}, Discretization}(
        g.discretization, g.t, x_native, w_native, operators_native)
end

# Structured FEM Geometry (block ops).
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuArray{T,3}, <:CuVector{T},
                                                      <:Any, Discretization}) where {T, Discretization}
    x_native = Array{T,3}(Array(g.x))
    w_native = Vector{T}(Array(g.w))

    Ti = Int
    convert_to_native = function(op)
        sparse_op = op isa CuSparseMatrixCSR ? op : _to_cusparse(op)
        A = _cusparse_to_cpu(sparse_op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    operators_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g.operators)))
        operators_native[key] = convert_to_native(g.operators[key])
    end

    Geometry{T, Array{T,3}, Vector{T}, SparseMatrixCSC{T,Ti}, Discretization}(
        g.discretization, g.t, x_native, w_native, operators_native)
end

# ============================================================================
# Convex-set value lifting (native → CUDA).
#
# A `Convex` set carries per-vertex parameter arrays in its `args` (e.g. the
# `A`/`b`/`p` grids built by `convex_Euclidian_power`/`convex_linear`). These
# must move to the GPU when the enclosing `MGBProblem` does; the barrier /
# cobarrier / slack functors are isbits and travel unchanged. Used by
# `native_to_cuda(::Convex)` below, which the `MGBProblem.Q` conversion calls.
# (A former `native_to_cuda(::NamedTuple)` zoo-problem converter was removed:
# zoo constructors now return an assembled `MGBProblem`, lifted by
# `native_to_cuda(::MGBProblem)`.)
# ============================================================================

# Recursively lift values in `Convex.args` to GPU. CPU arrays → CuArray;
# already-GPU arrays pass through unchanged (idempotent); tuples recurse; anything
# else (isbits scalars, UniformScaling sentinels, isbits barrier functors) passes
# through unchanged.
_zoo_value_to_cuda(x::CuArray)       = x
_zoo_value_to_cuda(x::AbstractArray) = CuArray(x)
_zoo_value_to_cuda(x::Tuple)         = map(_zoo_value_to_cuda, x)
_zoo_value_to_cuda(x)                = x

function _zoo_convex_to_cuda(q::Convex{T}) where {T}
    Convex{T}(q.barrier, q.cobarrier, q.slack, map(_zoo_value_to_cuda, q.args))
end

# ============================================================================
# Device dispatch (native_to_device / device_to_native) for CUDADevice.
#
# These let `mgb_solve(prob; device=CUDADevice)` move the assembled `MGBProblem`
# to the GPU without the caller touching `native_to_cuda` directly, and serve as
# the per-piece converters used while lifting the problem's fields.
# ============================================================================

MultiGridBarrier.native_to_device(::Type{CUDADevice}, x) = native_to_cuda(x)
MultiGridBarrier.device_to_native(::Type{CUDADevice}, x) = cuda_to_native(x)

# Per-piece native→cuda conversions used by the device layer (the Geometry / AMG /
# MGBProblem / MGBSOL methods above cover the structured cases).
MultiGridBarrier.native_to_cuda(x::CuArray)        = x                  # idempotent
MultiGridBarrier.native_to_cuda(x::AbstractArray)  = CuArray(x)
MultiGridBarrier.native_to_cuda(q::Convex)         = _zoo_convex_to_cuda(q)

# Idempotency for an already-GPU Geometry, so re-requesting CUDADevice on data
# that is already on the device is a no-op rather than a double conversion.
MultiGridBarrier.native_to_cuda(g::Geometry{T,<:CuArray}) where {T} = g

# ---- AMG / MGBProblem (the CPU-canonical assembled problem) → CUDA -------------
# `mgb_solve(prob::MGBProblem; device=CUDADevice)` lifts the whole closure-free
# problem to the GPU field-by-field: operators in the AMG hierarchy convert by their
# matrix type, the f/g grids and the convex set Q reuse the per-piece converters
# above, and the Geometry reuses its own method.

# Per-operator conversion, used by the AMG / MGBProblem lifts below.
_op_to_cuda(op::SparseMatrixCSC{T,Ti}) where {T,Ti} =
    CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(op.m, op.n, Int32.(op.colptr), Int32.(op.rowval), op.nzval))
function _op_to_cuda(op::MultiGridBarrier.BlockDiag{T}) where {T}
    gpu_data = CuArray{T,3}(op.data)
    MultiGridBarrier.BlockDiag{T, typeof(gpu_data)}(op.p, op.q, op.N, gpu_data)
end
_op_to_cuda(op::Matrix{T}) where {T} = CuMatrix{T}(op)
# A structured D[L,k] operator: lift its single active BlockDiag to the GPU and
# rebuild the wrapper (the col-partition metadata is plain Ints/Vector{Int}).
# Delegating to `_op_to_cuda(active_block)` makes this idempotent on GPU input too.
function _op_to_cuda(op::MultiGridBarrier.BlockColumn{T}) where {T}
    gpu_block = _op_to_cuda(op.active_block)
    A3 = typeof(gpu_block.data)
    MultiGridBarrier.BlockColumn{T, A3}(gpu_block, op.active_col, op.nu, op.col_sizes, op.total_rows)
end
# Idempotent on already-GPU operators.
_op_to_cuda(op::CuSparseMatrixCSR) = op
_op_to_cuda(op::CuMatrix) = op
_op_to_cuda(op::MultiGridBarrier.BlockDiag{T,<:CuArray}) where {T} = op

function MultiGridBarrier.native_to_cuda(a::AMG)
    AMG(; geometry = native_to_cuda(a.geometry),
          x        = native_to_cuda(a.x),
          w        = native_to_cuda(a.w),
          R_fine   = [_op_to_cuda(op) for op in a.R_fine],
          D_fine   = [_op_to_cuda(op) for op in a.D_fine])
end

MultiGridBarrier.native_to_cuda(prob::MGBProblem{T}) where {T} =
    MGBProblem{T}(map(native_to_cuda, prob.M),
                  native_to_cuda(prob.f),
                  native_to_cuda(prob.g),
                  native_to_cuda(prob.Q),
                  native_to_cuda(prob.geometry))

# MGBSOL cuda → native.
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

function MultiGridBarrier.cuda_to_native(sol::MGBSOL{T, <:Any, <:Any, Discretization}) where {T, Discretization}
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

    return MGBSOL(z_native, SOL_feasibility_native, SOL_main_native, sol.log, geometry_native)
end
