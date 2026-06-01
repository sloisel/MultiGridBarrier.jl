# amg_prolongators.jl — pluggable AMG prolongator factories shared by every FEM
# `amg(geom)` method (TensorFEM, FEM2D_P1, FEM2D_P2, FEM3D).
#
# A "prolongator" is a callable
#     K64::SparseMatrixCSC{Float64,Int} -> Vector{SparseMatrixCSC{Float64,Int}}
# returning the level prolongations finest -> coarsest. The factories below
# capture the underlying AMG parameters and return such a callable.

"""
    amg_ruge_stuben(; kwargs...) -> prolongator

Build a prolongator that runs classical Ruge–Stüben AMG via
`AlgebraicMultigrid.ruge_stuben` (the package default). Any `kwargs`
(e.g. `max_coarse=4`) are forwarded. Pass the result to `amg(geom; prolongator=...)`.
"""
amg_ruge_stuben(; kwargs...) =
    K64::SparseMatrixCSC{Float64,Int} ->
        [lvl.P for lvl in AlgebraicMultigrid.ruge_stuben(K64; kwargs...).levels]

"""
    amg_smoothed_aggregation(; kwargs...) -> prolongator

Build a prolongator that runs smoothed-aggregation AMG via
`AlgebraicMultigrid.smoothed_aggregation`. Any `kwargs` (e.g. `max_coarse=4`)
are forwarded. Pass the result to `amg(geom; prolongator=...)`.
"""
amg_smoothed_aggregation(; kwargs...) =
    K64::SparseMatrixCSC{Float64,Int} ->
        [lvl.P for lvl in AlgebraicMultigrid.smoothed_aggregation(K64; kwargs...).levels]

# Convert a scipy sparse matrix (PyObject) to a Julia SparseMatrixCSC{Float64,Int}.
function _scipy_to_julia(Pyobj)
    csc     = Pyobj.tocsc()
    indptr  = Vector{Int}(csc.indptr)      # 0-based CSC column pointers
    indices = Vector{Int}(csc.indices)     # 0-based row indices
    data    = Vector{Float64}(csc.data)
    shape   = csc.shape
    m, n    = Int(shape[1]), Int(shape[2])
    colptr  = indptr .+ 1
    rowval  = indices .+ 1
    return SparseMatrixCSC{Float64,Int}(m, n, colptr, rowval, data)
end

"""
    amg_pyamg(; solver::Symbol=:rootnode, kwargs...) -> prolongator

Build a prolongator backed by the Python `pyamg` package (imported lazily via
PyCall, installing from conda-forge if necessary). `solver` selects the pyamg
solver: `:rootnode` (rootnode energy-minimization, the default), `:smoothed_aggregation`,
or `:ruge_stuben`. Any `kwargs` are forwarded to the pyamg solver constructor.
Pass the result to `amg(geom; prolongator=...)`.
"""
function amg_pyamg(; solver::Symbol=:rootnode, kwargs...)
    return function (K64::SparseMatrixCSC{Float64,Int})
        pyamg = pyimport_conda("pyamg", "pyamg", "conda-forge")
        scipy_sparse = pyimport_conda("scipy.sparse", "scipy", "conda-forge")
        # Build a scipy CSC matrix from the Julia CSC arrays (0-based indices).
        A = scipy_sparse.csc_matrix(
            (K64.nzval, K64.rowval .- 1, K64.colptr .- 1),
            shape = (size(K64, 1), size(K64, 2)))
        ml = if solver === :rootnode
            pyamg.rootnode_solver(A; kwargs...)
        elseif solver === :smoothed_aggregation
            pyamg.smoothed_aggregation_solver(A; kwargs...)
        elseif solver === :ruge_stuben
            pyamg.ruge_stuben_solver(A; kwargs...)
        else
            throw(ArgumentError("amg_pyamg: unknown solver $(solver); " *
                "use :rootnode, :smoothed_aggregation, or :ruge_stuben"))
        end
        # The coarsest level carries no P; take all levels but the last.
        levels = collect(ml.levels)
        return [_scipy_to_julia(lvl.P) for lvl in levels[1:end-1]]
    end
end

# Apply a prolongator to K_int. Returns prolongations P[1] (finest)...P[end]
# (coarsest interior step), converted to T_out.
function _amg_prolongations(K_int::SparseMatrixCSC{T,Int}, ::Type{T_out},
                            prolongator) where {T, T_out}
    if size(K_int, 1) == 0
        return SparseMatrixCSC{T_out,Int}[]
    end
    K64 = SparseMatrixCSC{Float64,Int}(K_int)
    Ps  = prolongator(K64)
    return [SparseMatrixCSC{T_out,Int}(P) for P in Ps]
end
