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

"""
    amg_pyamg(; solver::Symbol=:rootnode, kwargs...) -> prolongator

Build a prolongator backed by the Python `pyamg` package. Provided by the
`MultiGridBarrierPyAMGExt` extension: load PyCall first
(`using MultiGridBarrier, PyCall`); pyamg and scipy are then imported lazily,
installing from conda-forge if necessary. `solver` selects the pyamg
solver: `:rootnode` (rootnode energy-minimization, the default), `:smoothed_aggregation`,
or `:ruge_stuben`. Any `kwargs` are forwarded to the pyamg solver constructor.
Pass the result to `amg(geom; prolongator=...)`.
"""
function amg_pyamg end

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
