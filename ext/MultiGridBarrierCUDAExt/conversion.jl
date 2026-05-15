# conversion.jl -- native_to_cuda / cuda_to_native type conversion

using CUDA.CUSPARSE: CuSparseMatrixCSR
import MultiGridBarrier: AMGBSOL, Geometry, MultiGrid, FEM1D, FEM2D_P1, FEM2D_P2, FEM3D,
                         _structurize_multigrid, _default_block_size

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
- `Matrix{T}` → `CuMatrix{T}`
- `Vector{T}` → `CuVector{T}`
- `SparseMatrixCSC{T,Int}` → `CuSparseMatrixCSR{T,Int32}`
- Dense `Matrix{T}` operators → `CuMatrix{T}` (spectral)
"""
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int}, Discretization};
                                         Ti::Type{<:Integer}=Int32) where {T, Discretization}
    x_cuda = CuMatrix{T}(g.x)
    w_cuda = CuVector{T}(g.w)

    convert_sparse = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    MType = CuSparseMatrixCSR{T,Ti}

    operators_cuda = Dict{Symbol, MType}()
    for key in sort(collect(keys(g.operators)))
        operators_cuda[key] = convert_sparse(g.operators[key])
    end

    subspaces_cuda = Dict{Symbol, MType}()
    for key in sort(collect(keys(g.subspaces)))
        subspaces_cuda[key] = convert_sparse(g.subspaces[key])
    end

    Geometry{T, CuMatrix{T}, CuVector{T}, MType, MType, Discretization}(
        g.discretization, x_cuda, w_cuda, subspaces_cuda, operators_cuda)
end

# BlockDiag-operator variant: FEM Geometry whose `operators` came in via
# `subdivide` (or `geometric_mg(...; structured=true).geometry`). Carry the
# BlockDiag to GPU as a CuArray-backed BlockDiag; subspaces stay sparse CSR.
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Matrix{T}, Vector{T},
                                                      MultiGridBarrier.BlockDiag{T, A3},
                                                      SparseMatrixCSC{T,Int},
                                                      Discretization};
                                         Ti::Type{<:Integer}=Int32) where {T, A3<:Array{T,3}, Discretization}
    x_cuda = CuMatrix{T}(g.x)
    w_cuda = CuVector{T}(g.w)

    function op_to_gpu(op::MultiGridBarrier.BlockDiag{T})
        gpu_data = CuArray{T,3}(op.data)
        MultiGridBarrier.BlockDiag{T, typeof(gpu_data)}(op.p, op.q, op.N, gpu_data)
    end

    sparse_to_gpu = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    sample_op = op_to_gpu(first(values(g.operators)))
    OpType = typeof(sample_op)
    SubType = CuSparseMatrixCSR{T,Ti}

    operators_cuda = Dict{Symbol, OpType}(
        key => op_to_gpu(g.operators[key]) for key in keys(g.operators))
    subspaces_cuda = Dict{Symbol, SubType}(
        key => sparse_to_gpu(g.subspaces[key]) for key in keys(g.subspaces))

    Geometry{T, CuMatrix{T}, CuVector{T}, OpType, SubType, Discretization}(
        g.discretization, x_cuda, w_cuda, subspaces_cuda, operators_cuda)
end

# Dense spectral variant.
function MultiGridBarrier.native_to_cuda(g::Geometry{T, Matrix{T}, Vector{T}, Matrix{T}, Matrix{T}, Discretization};
                                         kwargs...) where {T, Discretization}
    x_cuda = CuMatrix{T}(g.x)
    w_cuda = CuVector{T}(g.w)

    operators_cuda = Dict{Symbol, CuMatrix{T}}()
    for key in sort(collect(keys(g.operators)))
        operators_cuda[key] = CuMatrix{T}(g.operators[key])
    end

    subspaces_cuda = Dict{Symbol, CuMatrix{T}}()
    for key in sort(collect(keys(g.subspaces)))
        subspaces_cuda[key] = CuMatrix{T}(g.subspaces[key])
    end

    Geometry{T, CuMatrix{T}, CuVector{T}, CuMatrix{T}, CuMatrix{T}, Discretization}(
        g.discretization, x_cuda, w_cuda, subspaces_cuda, operators_cuda)
end

# ============================================================================
# native_to_cuda(::MultiGrid)
# ============================================================================

"""
    native_to_cuda(mg::MultiGrid; Ti=Int32, structured=true, block_size=auto) -> MultiGrid

Convert a native `MultiGrid` to CUDA. Converts the inner `Geometry` and the per-level
subspaces, refine, coarsen vectors. When `structured=true` and the discretization supports
it (FEM*), applies `_structurize_multigrid` afterwards.
"""
function MultiGridBarrier.native_to_cuda(mg::MultiGrid{T, SparseMatrixCSC{T,Int},
                                                        SparseMatrixCSC{T,Int},
                                                        SparseMatrixCSC{T,Int}};
                                         Ti::Type{<:Integer}=Int32,
                                         structured::Bool=true,
                                         block_size::Int=_default_block_size(mg.discretization)) where {T}
    geom_cuda = native_to_cuda(mg.geometry; Ti=Ti)

    convert_sparse = op -> CuSparseMatrixCSR(
        SparseMatrixCSC{T,Ti}(op.m, op.n, Ti.(op.colptr), Ti.(op.rowval), op.nzval))

    MType = CuSparseMatrixCSR{T,Ti}

    subspaces_cuda = Dict{Symbol, Vector{MType}}()
    for key in sort(collect(keys(mg.subspaces)))
        sv = mg.subspaces[key]
        cv = Vector{MType}(undef, length(sv))
        for i in 1:length(sv)
            cv[i] = convert_sparse(sv[i])
        end
        subspaces_cuda[key] = cv
    end

    refine_cuda = Dict{Symbol, Vector{MType}}()
    for key in sort(collect(keys(mg.refine)))
        rv = mg.refine[key]
        cv = Vector{MType}(undef, length(rv))
        for i in 1:length(rv)
            cv[i] = convert_sparse(rv[i])
        end
        refine_cuda[key] = cv
    end

    coarsen_cuda = Dict{Symbol, Vector{MType}}()
    for key in sort(collect(keys(mg.coarsen)))
        cvec = mg.coarsen[key]
        cuv = Vector{MType}(undef, length(cvec))
        for i in 1:length(cvec)
            cuv[i] = convert_sparse(cvec[i])
        end
        coarsen_cuda[key] = cuv
    end

    mg_cuda = MultiGrid(geom_cuda, subspaces_cuda, refine_cuda, coarsen_cuda)

    if structured
        mg_cuda = _structurize_multigrid(mg_cuda, block_size)
    end

    return mg_cuda
end

# Dense spectral MultiGrid.
function MultiGridBarrier.native_to_cuda(mg::MultiGrid{T, Matrix{T}, Matrix{T}, Matrix{T}};
                                         kwargs...) where {T}
    geom_cuda = native_to_cuda(mg.geometry)

    subspaces_cuda = Dict{Symbol, Vector{CuMatrix{T}}}()
    for key in sort(collect(keys(mg.subspaces)))
        subspaces_cuda[key] = [CuMatrix{T}(s) for s in mg.subspaces[key]]
    end

    refine_cuda = Dict{Symbol, Vector{CuMatrix{T}}}()
    for key in sort(collect(keys(mg.refine)))
        refine_cuda[key] = [CuMatrix{T}(r) for r in mg.refine[key]]
    end

    coarsen_cuda = Dict{Symbol, Vector{CuMatrix{T}}}()
    for key in sort(collect(keys(mg.coarsen)))
        coarsen_cuda[key] = [CuMatrix{T}(c) for c in mg.coarsen[key]]
    end

    return MultiGrid(geom_cuda, subspaces_cuda, refine_cuda, coarsen_cuda)
end

# ============================================================================
# cuda_to_native: Geometry/MultiGrid/AMGBSOL → CPU
# ============================================================================

# Sparse FEM Geometry (CSR operators).
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuMatrix{T}, <:CuVector{T},
                                                      <:CuSparseMatrixCSR{T}, <:CuSparseMatrixCSR{T},
                                                      Discretization}) where {T, Discretization}
    x_native = Matrix{T}(Array(g.x))
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

    subspaces_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g.subspaces)))
        subspaces_native[key] = convert_back(g.subspaces[key])
    end

    Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Ti}, SparseMatrixCSC{T,Ti}, Discretization}(
        g.discretization, x_native, w_native, subspaces_native, operators_native)
end

# Dense spectral Geometry.
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuMatrix{T}, <:CuVector{T},
                                                      <:CuMatrix{T}, <:CuMatrix{T},
                                                      Discretization}) where {T, Discretization}
    x_native = Matrix{T}(Array(g.x))
    w_native = Vector{T}(Array(g.w))

    operators_native = Dict{Symbol, Matrix{T}}()
    for key in sort(collect(keys(g.operators)))
        operators_native[key] = Matrix{T}(Array(g.operators[key]))
    end

    subspaces_native = Dict{Symbol, Matrix{T}}()
    for key in sort(collect(keys(g.subspaces)))
        subspaces_native[key] = Matrix{T}(Array(g.subspaces[key]))
    end

    Geometry{T, Matrix{T}, Vector{T}, Matrix{T}, Matrix{T}, Discretization}(
        g.discretization, x_native, w_native, subspaces_native, operators_native)
end

# Structured FEM Geometry (block ops).
function MultiGridBarrier.cuda_to_native(g::Geometry{T, <:CuMatrix{T}, <:CuVector{T},
                                                      <:Any, <:Any, Discretization}) where {T, Discretization}
    x_native = Matrix{T}(Array(g.x))
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

    subspaces_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g.subspaces)))
        subspaces_native[key] = convert_to_native(g.subspaces[key])
    end

    Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Ti}, SparseMatrixCSC{T,Ti}, Discretization}(
        g.discretization, x_native, w_native, subspaces_native, operators_native)
end

# MultiGrid cuda → native (sparse FEM).
function MultiGridBarrier.cuda_to_native(mg::MultiGrid{T, <:CuSparseMatrixCSR{T},
                                                       <:CuSparseMatrixCSR{T},
                                                       <:CuSparseMatrixCSR{T}}) where {T}
    geom_native = cuda_to_native(mg.geometry)
    Ti = Int
    convert_back = op -> begin
        A = _cusparse_to_cpu(op)
        SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
    end

    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.subspaces)))
        sv = mg.subspaces[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(sv))
        for i in 1:length(sv)
            nv[i] = convert_back(sv[i])
        end
        subspaces_native[key] = nv
    end

    refine_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.refine)))
        rv = mg.refine[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(rv))
        for i in 1:length(rv)
            nv[i] = convert_back(rv[i])
        end
        refine_native[key] = nv
    end

    coarsen_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.coarsen)))
        cv = mg.coarsen[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(cv))
        for i in 1:length(cv)
            nv[i] = convert_back(cv[i])
        end
        coarsen_native[key] = nv
    end

    return MultiGrid(geom_native, subspaces_native, refine_native, coarsen_native)
end

# MultiGrid cuda → native (dense spectral).
function MultiGridBarrier.cuda_to_native(mg::MultiGrid{T, <:CuMatrix{T}, <:CuMatrix{T}, <:CuMatrix{T}}) where {T}
    geom_native = cuda_to_native(mg.geometry)

    subspaces_native = Dict{Symbol, Vector{Matrix{T}}}()
    for key in sort(collect(keys(mg.subspaces)))
        subspaces_native[key] = [Matrix{T}(Array(s)) for s in mg.subspaces[key]]
    end

    refine_native = Dict{Symbol, Vector{Matrix{T}}}()
    for key in sort(collect(keys(mg.refine)))
        refine_native[key] = [Matrix{T}(Array(r)) for r in mg.refine[key]]
    end

    coarsen_native = Dict{Symbol, Vector{Matrix{T}}}()
    for key in sort(collect(keys(mg.coarsen)))
        coarsen_native[key] = [Matrix{T}(Array(c)) for c in mg.coarsen[key]]
    end

    return MultiGrid(geom_native, subspaces_native, refine_native, coarsen_native)
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

    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.subspaces)))
        sv = mg.subspaces[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(sv))
        for i in 1:length(sv)
            nv[i] = convert_back(sv[i])
        end
        subspaces_native[key] = nv
    end

    refine_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.refine)))
        rv = mg.refine[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(rv))
        for i in 1:length(rv)
            nv[i] = convert_back(rv[i])
        end
        refine_native[key] = nv
    end

    coarsen_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(mg.coarsen)))
        cv = mg.coarsen[key]
        nv = Vector{SparseMatrixCSC{T,Ti}}(undef, length(cv))
        for i in 1:length(cv)
            nv[i] = convert_back(cv[i])
        end
        coarsen_native[key] = nv
    end

    return MultiGrid(geom_native, subspaces_native, refine_native, coarsen_native)
end

# AMGBSOL cuda → native.
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

    return AMGBSOL(z_native, SOL_feasibility_native, SOL_main_native, sol.log, geometry_native)
end
