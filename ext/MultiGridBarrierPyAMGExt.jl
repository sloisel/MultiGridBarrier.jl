# MultiGridBarrierPyAMGExt — the pyamg-backed AMG prolongator, as a package
# extension. Loads automatically when both MultiGridBarrier and PyCall are
# imported (`using MultiGridBarrier, PyCall`); supplies the method of the
# parent stub `amg_pyamg` (src/amg_prolongators.jl). The Python pyamg and scipy
# packages are imported lazily on the first call, installing from conda-forge
# if necessary.
module MultiGridBarrierPyAMGExt

using PyCall
using SparseArrays
import MultiGridBarrier
import MultiGridBarrier: amg_pyamg

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

function MultiGridBarrier.amg_pyamg(; solver::Symbol=:rootnode, kwargs...)
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

end # module
