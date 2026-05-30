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
- `SparseMatrixCSC{T,Int}` → `CuSparseMatrixCSR{T,Int32}`
- Dense `Matrix{T}` operators → `CuMatrix{T}` (spectral)
"""
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Array{T,3}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization};
                                         Ti::Type{<:Integer}=Int32) where {T, Discretization}
    x_cuda = CuArray{T,3}(g.x)
    w_cuda = CuVector{T}(g.w)

    convert_sparse = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    MType = CuSparseMatrixCSR{T,Ti}

    operators_cuda = Dict{Symbol, MType}()
    for key in sort(collect(keys(g.operators)))
        operators_cuda[key] = convert_sparse(g.operators[key])
    end

    Geometry{T, CuArray{T,3}, CuVector{T}, MType, Discretization}(
        g.discretization, x_cuda, w_cuda, operators_cuda)
end

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
        g.discretization, x_cuda, w_cuda, operators_cuda)
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
        g.discretization, x_cuda, w_cuda, operators_cuda)
end

# ============================================================================
# native_to_cuda(::MultiGrid)
# ============================================================================

"""
    native_to_cuda(mg::MultiGrid; Ti=Int32) -> MultiGrid

Convert a native `MultiGrid` to CUDA, faithfully preserving operator and prolongation
types: `BlockDiag` operators (built by `geometric_mg(...; structured=true)` / `subdivide`)
become GPU `BlockDiag` — driving the structured batched-GEMM Hessian assembly — while
sparse operators and the prolongations `R[X]` become `CuSparseMatrixCSR`. Whether the
solve is structured is therefore decided once, at geometry construction, not here.
"""
function MultiGridBarrier.native_to_cuda(mg::MultiGrid{T, SparseMatrixCSC{T,Int}};
                                         Ti::Type{<:Integer}=Int32) where {T}
    geom_cuda = native_to_cuda(mg.geometry; Ti=Ti)

    convert_sparse = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    MType = CuSparseMatrixCSR{T,Ti}

    R_cuda = Dict{Symbol, Vector{MType}}()
    for key in sort(collect(keys(mg.R)))
        rv = mg.R[key]
        cv = Vector{MType}(undef, length(rv))
        for i in 1:length(rv)
            cv[i] = convert_sparse(rv[i])
        end
        R_cuda[key] = cv
    end

    return MultiGrid(geom_cuda, R_cuda)
end

# Dense spectral MultiGrid.
function MultiGridBarrier.native_to_cuda(mg::MultiGrid{T, Matrix{T}};
                                         kwargs...) where {T}
    geom_cuda = native_to_cuda(mg.geometry)

    R_cuda = Dict{Symbol, Vector{CuMatrix{T}}}()
    for key in sort(collect(keys(mg.R)))
        R_cuda[key] = [CuMatrix{T}(r) for r in mg.R[key]]
    end

    return MultiGrid(geom_cuda, R_cuda)
end

# ============================================================================
# cuda_to_native: Geometry/MultiGrid/MGBSOL → CPU
# ============================================================================

# Sparse FEM Geometry (CSR operators).
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuArray{T,3}, <:CuVector{T},
                                                      <:CuSparseMatrixCSR{T},
                                                      Discretization}) where {T, Discretization}
    x_native = Array{T,3}(Array(g.x))
    w_native = Vector{T}(Array(g.w))
    Ti = Int
    convert_back = op -> begin
        A = _cusparse_to_cpu(op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    operators_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g.operators)))
        operators_native[key] = convert_back(g.operators[key])
    end

    Geometry{T, Array{T,3}, Vector{T}, SparseMatrixCSC{T,Ti}, Discretization}(
        g.discretization, x_native, w_native, operators_native)
end

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
        g.discretization, x_native, w_native, operators_native)
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
        g.discretization, x_native, w_native, operators_native)
end

# MultiGrid cuda → native (sparse FEM).
function MultiGridBarrier.cuda_to_native(mg::MultiGrid{T, <:CuSparseMatrixCSR{T}}) where {T}
    geom_native = cuda_to_native(mg.geometry)
    Ti = Int
    convert_back = op -> begin
        A = _cusparse_to_cpu(op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    R_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.R)))
        rv = mg.R[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(rv))
        for i in 1:length(rv)
            nv[i] = convert_back(rv[i])
        end
        R_native[key] = nv
    end

    return MultiGrid(geom_native, R_native)
end

# MultiGrid cuda → native (dense spectral).
function MultiGridBarrier.cuda_to_native(mg::MultiGrid{T, <:CuMatrix{T}}) where {T}
    geom_native = cuda_to_native(mg.geometry)

    R_native = Dict{Symbol, Vector{Matrix{T}}}()
    for key in sort(collect(keys(mg.R)))
        R_native[key] = [Matrix{T}(Array(r)) for r in mg.R[key]]
    end

    return MultiGrid(geom_native, R_native)
end

# Generic structured MultiGrid (block types) — convert all to sparse.
function MultiGridBarrier.cuda_to_native(mg::MultiGrid{T}) where {T}
    geom_native = cuda_to_native(mg.geometry)
    Ti = Int
    convert_back = function(op)
        sparse_op = op isa CuSparseMatrixCSR ? op : _to_cusparse(op)
        A = _cusparse_to_cpu(sparse_op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    R_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.R)))
        rv = mg.R[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(rv))
        for i in 1:length(rv)
            nv[i] = convert_back(rv[i])
        end
        R_native[key] = nv
    end

    return MultiGrid(geom_native, R_native)
end

# ============================================================================
# native_to_cuda(::NamedTuple)  — zoo problem CPU → GPU conversion.
#
# Zoo problem constructors (Zoo.elastoplastic_torsion, …) return a NamedTuple
# of `mgb_solve` kwargs. Five of the six pass non-trivial `A`/`b`/`p` closures
# to `convex_Euclidian_power`/`convex_linear`, plus user-supplied `f`/`g`
# closures — all of which evaluate fine on CPU but trip CUDA.jl's broadcast
# JIT on non-isbits captures when shipped to GPU verbatim. The fix is to
# pre-evaluate everything closure-shaped on CPU and ship the resulting
# per-vertex arrays to GPU; the `Convex` barrier/cobarrier/slack functors
# (e.g. EuclidianPowerBarrier, structured at convex_Euclidian_power
# construction time) are already isbits and travel unchanged.
# ============================================================================

"""
    native_to_cuda(problem::NamedTuple; structured::Bool=false) -> NamedTuple

Convert a CPU zoo-problem NamedTuple to GPU. Pre-evaluates `f` / `g`
closures on CPU and ships the resulting per-vertex grids to GPU; converts
the MultiGrid and each `Convex` in `Q` so all per-vertex parameter arrays
live on the device. The returned NamedTuple can be splatted directly into
`mgb_solve(; problem_gpu...)`.

If the input has `f` / `g` closure fields, the output replaces them with
`f_grid` / `g_grid` arrays. If the input already has `f_grid` / `g_grid`,
those are converted in place. All other fields (`state_variables`, `D`,
`p`, …) are passed through unchanged.
"""
function MultiGridBarrier.native_to_cuda(problem::NamedTuple)
    haskey(problem, :mg) || error("native_to_cuda(problem::NamedTuple): missing :mg field")
    haskey(problem, :Q)  || error("native_to_cuda(problem::NamedTuple): missing :Q field")

    mg_gpu = native_to_cuda(problem.mg)

    # Pre-evaluate f, g on CPU at the mesh nodes; transfer the grids to GPU.
    x_cpu = MultiGridBarrier._xflat(problem.mg)
    f_grid_gpu = if haskey(problem, :f_grid)
        _zoo_value_to_cuda(problem.f_grid)
    elseif haskey(problem, :f)
        _zoo_value_to_cuda(map_rows(xi -> SVector(Tuple(problem.f(xi))), x_cpu))
    else
        nothing
    end
    g_grid_gpu = if haskey(problem, :g_grid)
        _zoo_value_to_cuda(problem.g_grid)
    elseif haskey(problem, :g)
        _zoo_value_to_cuda(map_rows(xi -> SVector(Tuple(problem.g(xi))), x_cpu))
    else
        nothing
    end

    Q_gpu = _zoo_convex_to_cuda(problem.Q)

    # Pass through any remaining fields (state_variables, D, p, tol, …)
    # untouched. Drop the slots we've replaced.
    drop = (:mg, :f, :g, :f_grid, :g_grid, :Q)
    kept = NamedTuple{filter(k -> !(k in drop), keys(problem))}(problem)

    out = (; mg=mg_gpu, kept..., Q=Q_gpu)
    f_grid_gpu === nothing || (out = merge(out, (; f_grid=f_grid_gpu)))
    g_grid_gpu === nothing || (out = merge(out, (; g_grid=g_grid_gpu)))
    return out
end

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
# These let `mgb_solve(mg; device=CUDADevice)` move data without the caller
# touching `native_to_cuda` directly. They also serve as the per-piece converters
# `assemble` applies to any explicitly supplied problem data (e.g. a CPU `Zoo`
# NamedTuple's `Q`/grids), so a CPU problem assembles onto the GPU uniformly.
# ============================================================================

MultiGridBarrier.native_to_device(::Type{CUDADevice}, x) = native_to_cuda(x)
MultiGridBarrier.device_to_native(::Type{CUDADevice}, x) = cuda_to_native(x)

# Per-piece native→cuda conversions used by the device layer (the existing
# Geometry/MultiGrid/MGBSOL/NamedTuple methods above cover the structured cases).
MultiGridBarrier.native_to_cuda(x::CuArray)        = x                  # idempotent
MultiGridBarrier.native_to_cuda(x::AbstractArray)  = CuArray(x)
MultiGridBarrier.native_to_cuda(q::Convex)         = _zoo_convex_to_cuda(q)

# Idempotency for already-GPU hierarchies, so re-requesting CUDADevice on data
# that is already on the device is a no-op rather than a double conversion.
MultiGridBarrier.native_to_cuda(g::Geometry{T,<:CuArray}) where {T} = g
MultiGridBarrier.native_to_cuda(mg::MultiGrid{T,<:CuSparseMatrixCSR}) where {T} = mg
MultiGridBarrier.native_to_cuda(mg::MultiGrid{T,<:CuMatrix}) where {T} = mg

# ---- AMG / MGBProblem (the CPU-canonical assembled problem) → CUDA -------------
# `mgb_solve(prob::MGBProblem; device=CUDADevice)` lifts the whole closure-free
# problem to the GPU field-by-field: operators in the AMG hierarchy convert by their
# matrix type, the f/g grids and the convex set Q reuse the per-piece converters
# above, and the Geometry reuses its own method.

# Per-operator conversion, mirroring the inline conversions in native_to_cuda(::MultiGrid).
_op_to_cuda(op::SparseMatrixCSC{T,Ti}) where {T,Ti} =
    CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(op.m, op.n, Int32.(op.colptr), Int32.(op.rowval), op.nzval))
function _op_to_cuda(op::MultiGridBarrier.BlockDiag{T}) where {T}
    gpu_data = CuArray{T,3}(op.data)
    MultiGridBarrier.BlockDiag{T, typeof(gpu_data)}(op.p, op.q, op.N, gpu_data)
end
_op_to_cuda(op::Matrix{T}) where {T} = CuMatrix{T}(op)
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
