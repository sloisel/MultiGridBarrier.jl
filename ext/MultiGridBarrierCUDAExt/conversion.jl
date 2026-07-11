# conversion.jl -- native_to_cuda / cuda_to_native type conversion
#
# All conversions run through an identity-memoized context (`_Memo`, an IdDict
# scoped to ONE top-level conversion): every source array or structured object
# is converted exactly once, and the result is reused everywhere the source was
# shared. An assembled `MGBProblem` shares aggressively on the CPU — one
# `Geometry` referenced by the problem and both AMG hierarchies, one quadrature
# weight vector, and every `D_fine` block wrapping the same arrays as
# `geometry.operators` — so unmemoized conversion would make an independent
# persistent device copy per reference. The memo is created per call and
# discarded with it: no global state, no stale device allocations.

using CUDA.CUSPARSE: CuSparseMatrixCSR
import MultiGridBarrier: MGBSOL, Geometry, Convex, AMG, MGBProblem,
                         native_to_device, device_to_native, CUDADevice

const _Memo = IdDict{Any,Any}

# Device-agnostic CuSparseMatrixCSR → SparseMatrixCSC conversion.
function _cusparse_to_cpu(A::CuSparseMatrixCSR{T,Ti}) where {T,Ti}
    m, n = size(A)
    CUDA.synchronize()
    rp = Array(A.rowPtr)
    cv = Array(A.colVal)
    nz = Array(A.nzVal)
    if length(nz) == 0
        return spzeros(T, m, n)
    end
    # CSR of A ≡ CSC of Aᵀ
    At_csc = SparseMatrixCSC{T,Ti}(n, m, rp, cv, nz)
    return sparse(At_csc')
end

# ============================================================================
# native → CUDA: memoized per-node converters
#
# Only the value types the old field-by-field conversion moved to the GPU are
# converted here; wrapper metadata (BlockColumn col_sizes, Geometry.t, plain
# Ints) passes through untouched. Every method ends in a typeassert: the memo
# is an IdDict{Any,Any}, so the assert restores type stability at each barrier.
# ============================================================================

_cu(ctx::_Memo, x::CuArray) = x                                   # idempotent
_cu(ctx::_Memo, x::Array{T,3}) where {T} =
    get!(() -> CuArray{T,3}(x), ctx, x)::CuArray{T,3}
_cu(ctx::_Memo, x::Vector{T}) where {T} =
    get!(() -> CuVector{T}(x), ctx, x)::CuVector{T}
_cu(ctx::_Memo, x::Matrix{T}) where {T} =
    get!(() -> CuMatrix{T}(x), ctx, x)::CuMatrix{T}
_cu(ctx::_Memo, x::AbstractArray) = get!(() -> CuArray(x), ctx, x)
_cu(ctx::_Memo, x::Tuple) = map(v -> _cu(ctx, v), x)
_cu(ctx::_Memo, x) = x            # isbits scalars, UniformScaling, functors, …

# --- operators ---------------------------------------------------------------

_op_to_cuda(ctx::_Memo, op::SparseMatrixCSC{T,Ti}) where {T,Ti} =
    get!(ctx, op) do
        CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(op.m, op.n, Int32.(op.colptr),
                                                   Int32.(op.rowval), op.nzval))
    end::CuSparseMatrixCSR{T,Int32}

function _op_to_cuda(ctx::_Memo, op::MultiGridBarrier.BlockDiag{T}) where {T}
    get!(ctx, op) do
        # Memoized on the wrapper AND on the data array: two distinct BlockDiag
        # wrappers around the same data still share one device array.
        gpu_data = _cu(ctx, op.data)
        MultiGridBarrier.BlockDiag{T, typeof(gpu_data)}(op.p, op.q, op.N, gpu_data)
    end::MultiGridBarrier.BlockDiag{T, CuArray{T,3,CUDA.DeviceMemory}}
end

_op_to_cuda(ctx::_Memo, op::Matrix{T}) where {T} = _cu(ctx, op)

# A structured D[L,k] operator: lift its single active BlockDiag to the GPU and
# rebuild the wrapper (the col-partition metadata is plain Ints/Vector{Int}).
function _op_to_cuda(ctx::_Memo, op::MultiGridBarrier.BlockColumn{T}) where {T}
    get!(ctx, op) do
        gpu_block = _op_to_cuda(ctx, op.active_block)
        A3 = typeof(gpu_block.data)
        MultiGridBarrier.BlockColumn{T, A3}(gpu_block, op.active_col, op.nu,
                                            op.col_sizes, op.total_rows)
    end::MultiGridBarrier.BlockColumn{T, CuArray{T,3,CUDA.DeviceMemory}}
end

# Idempotent on already-GPU operators.
_op_to_cuda(ctx::_Memo, op::CuSparseMatrixCSR) = op
_op_to_cuda(ctx::_Memo, op::CuMatrix) = op
_op_to_cuda(ctx::_Memo, op::MultiGridBarrier.BlockDiag{T,<:CuArray}) where {T} = op

# --- Geometry ----------------------------------------------------------------

# BlockDiag-operator variant: FEM Geometry whose `operators` came in via
# `subdivide` (or `geometric_mg(...).geometry`). Carry the BlockDiag to GPU as
# a CuArray-backed BlockDiag. Dense `Matrix` operators (spectral) become
# CuMatrix. The connectivity `t` stays on the CPU.
function _geometry_to_cuda(ctx::_Memo, g::Geometry{T, Array{T,3}, Vector{T}, M_op,
                                                   Discretization}) where {T, M_op, Discretization}
    get!(ctx, g) do
        x_cuda = _cu(ctx, g.x)
        w_cuda = _cu(ctx, g.w)
        sample_op = _op_to_cuda(ctx, first(values(g.operators)))
        OpType = typeof(sample_op)
        operators_cuda = Dict{Symbol, OpType}(
            key => _op_to_cuda(ctx, g.operators[key]) for key in keys(g.operators))
        Geometry{T, CuArray{T,3}, CuVector{T}, OpType, Discretization}(
            g.discretization, g.t, x_cuda, w_cuda, operators_cuda)
    end::Geometry
end
_geometry_to_cuda(ctx::_Memo, g::Geometry{T,<:CuArray}) where {T} = g  # idempotent

# --- Convex ------------------------------------------------------------------

# A `Convex` set carries per-vertex parameter arrays in its `args` (e.g. the
# `A`/`b`/`p` grids built by `convex_Euclidian_power`/`convex_linear`). These
# must move to the GPU when the enclosing `MGBProblem` does; the barrier /
# cobarrier / slack functors are isbits and travel unchanged.
_convex_to_cuda(ctx::_Memo, q::Convex{T}) where {T} =
    Convex{T}(q.barrier, q.cobarrier, q.slack, _cu(ctx, q.args), q.input_spec)

# --- AMG / MGBProblem ----------------------------------------------------------

function _amg_to_cuda(ctx::_Memo, a::AMG)
    get!(ctx, a) do
        AMG(; geometry = _geometry_to_cuda(ctx, a.geometry),
              x        = _cu(ctx, a.x),
              w        = _cu(ctx, a.w),
              R_fine   = [_op_to_cuda(ctx, op) for op in a.R_fine],
              D_fine   = [_op_to_cuda(ctx, op) for op in a.D_fine])
    end::AMG
end

# ============================================================================
# native → CUDA: public entry points. Each creates one fresh memo scoped to
# the call; the MGBProblem method threads a single memo through every field,
# which is what preserves the problem-wide sharing graph on the device.
# ============================================================================

"""
    native_to_cuda(x) -> GPU value

Convert a native (CPU) value — array, `Geometry`, `AMG`, `Convex`, or a whole
`MGBProblem` — to CUDA GPU types. Conversion is identity-memoized per call:
objects shared (`===`) on the CPU are shared on the GPU. Structured operators
stay structured (`BlockDiag` → CuArray-backed `BlockDiag`); sparse matrices
become `CuSparseMatrixCSR`; dense spectral operators become `CuMatrix`.
"""
MultiGridBarrier.native_to_cuda(x::CuArray)       = x                  # idempotent
MultiGridBarrier.native_to_cuda(x::AbstractArray) = _cu(_Memo(), x)
MultiGridBarrier.native_to_cuda(q::Convex)        = _convex_to_cuda(_Memo(), q)
MultiGridBarrier.native_to_cuda(g::Geometry)      = _geometry_to_cuda(_Memo(), g)
MultiGridBarrier.native_to_cuda(a::AMG)           = _amg_to_cuda(_Memo(), a)

function MultiGridBarrier.native_to_cuda(prob::MGBProblem{T}) where {T}
    ctx = _Memo()
    MGBProblem{T}(map(a -> _amg_to_cuda(ctx, a), prob.M),
                  _cu(ctx, prob.f),
                  _cu(ctx, prob.g),
                  _convex_to_cuda(ctx, prob.Q),
                  _geometry_to_cuda(ctx, prob.geometry))
end

# ============================================================================
# CUDA → native: memoized, structure-preserving return path
# ============================================================================

_cpu(ctx::_Memo, x::CuArray{T,3}) where {T} = get!(() -> Array(x), ctx, x)::Array{T,3}
_cpu(ctx::_Memo, x::CuVector{T}) where {T} = get!(() -> Array(x), ctx, x)::Vector{T}
_cpu(ctx::_Memo, x::CuMatrix{T}) where {T} = get!(() -> Array(x), ctx, x)::Matrix{T}

# Structured operators return structured: a CuArray-backed BlockDiag comes back
# as BlockDiag(Array(data)), NOT as a degraded SparseMatrixCSC — downstream
# consumers (re-assembly, batched-GEMM Hessians) depend on the block type.
function _op_to_native(ctx::_Memo, op::MultiGridBarrier.BlockDiag{T,<:CuArray}) where {T}
    get!(ctx, op) do
        data = _cpu(ctx, op.data)
        MultiGridBarrier.BlockDiag{T, typeof(data)}(op.p, op.q, op.N, data)
    end::MultiGridBarrier.BlockDiag{T, Array{T,3}}
end

function _op_to_native(ctx::_Memo, op::MultiGridBarrier.BlockColumn{T,<:CuArray}) where {T}
    get!(ctx, op) do
        blk = _op_to_native(ctx, op.active_block)
        MultiGridBarrier.BlockColumn{T, Array{T,3}}(blk, op.active_col, op.nu,
                                                    op.col_sizes, op.total_rows)
    end::MultiGridBarrier.BlockColumn{T, Array{T,3}}
end

function _op_to_native(ctx::_Memo, op::CuSparseMatrixCSR{T}) where {T}
    get!(ctx, op) do
        A = _cusparse_to_cpu(op)
        SparseMatrixCSC{T,Int}(A.m, A.n, Int.(A.colptr), Int.(A.rowval), A.nzval)
    end::SparseMatrixCSC{T,Int}
end

_op_to_native(ctx::_Memo, op::CuMatrix) = _cpu(ctx, op)

function _geometry_to_native(ctx::_Memo, g::Geometry{T, <:CuArray{T,3}, <:CuVector{T},
                                                     <:Any, Discretization}) where {T, Discretization}
    get!(ctx, g) do
        x_native = _cpu(ctx, g.x)
        w_native = _cpu(ctx, g.w)
        sample_op = _op_to_native(ctx, first(values(g.operators)))
        OpType = typeof(sample_op)
        operators_native = Dict{Symbol, OpType}(
            key => _op_to_native(ctx, g.operators[key]) for key in keys(g.operators))
        Geometry{T, Array{T,3}, Vector{T}, OpType, Discretization}(
            g.discretization, g.t, x_native, w_native, operators_native)
    end::Geometry
end

"""
    cuda_to_native(x) -> CPU value

Convert a CUDA GPU value (`Geometry`, `MGBSOL`, arrays) back to native CPU
types. Identity-memoized per call, and structure-preserving: CuArray-backed
`BlockDiag`/`BlockColumn` operators return as their CPU block types, not as
generic sparse matrices.
"""
MultiGridBarrier.cuda_to_native(g::Geometry{T,<:CuArray}) where {T} =
    _geometry_to_native(_Memo(), g)

# MGBSOL cuda → native.
_convert_cuda_to_native(ctx::_Memo, x::CuMatrix) = _cpu(ctx, x)
_convert_cuda_to_native(ctx::_Memo, x::CuVector) = _cpu(ctx, x)
_convert_cuda_to_native(ctx::_Memo, x::CuSparseMatrixCSR) = _op_to_native(ctx, x)

function _convert_cuda_value(ctx::_Memo, value)
    if isa(value, CuMatrix) || isa(value, CuVector) || isa(value, CuSparseMatrixCSR)
        return _convert_cuda_to_native(ctx, value)
    elseif isa(value, Array)
        return map(v -> _convert_cuda_value(ctx, v), value)
    else
        return value
    end
end

function MultiGridBarrier.cuda_to_native(sol::MGBSOL{T, <:Any, <:Any, Discretization}) where {T, Discretization}
    ctx = _Memo()
    z_native = _convert_cuda_value(ctx, sol.z)

    function convert_namedtuple(nt)
        nt === nothing && return nothing
        converted_fields = []
        for (name, value) in pairs(nt)
            push!(converted_fields, name => _convert_cuda_value(ctx, value))
        end
        return NamedTuple(converted_fields)
    end

    SOL_feasibility_native = convert_namedtuple(sol.SOL_feasibility)
    SOL_main_native = convert_namedtuple(sol.SOL_main)
    geometry_native = _geometry_to_native(ctx, sol.geometry)

    return MGBSOL(z_native, SOL_feasibility_native, SOL_main_native, sol.log, geometry_native)
end

# ============================================================================
# Device dispatch (native_to_device / device_to_native) for CUDADevice.
#
# These let `mgb_solve(prob; device=CUDADevice)` move the assembled `MGBProblem`
# to the GPU without the caller touching `native_to_cuda` directly.
# ============================================================================

MultiGridBarrier.native_to_device(::Type{CUDADevice}, x) = native_to_cuda(x)
MultiGridBarrier.device_to_native(::Type{CUDADevice}, x) = cuda_to_native(x)
