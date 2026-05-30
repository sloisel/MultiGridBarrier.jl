"""
    Zoo

A library of convex variational test problems. Each constructor takes a `MultiGrid`
and returns an assembled, closure-free [`MGBProblem`](@ref); solve it with
`mgb_solve(problem; kwargs...)`. Problem-defining parameters (`p`, forcing `f`,
boundary data `g_u`, …) are keyword arguments of the constructor; solver-control
parameters (`tol`, `device`, …) are passed to `mgb_solve`.

# GPU support
Solve on a GPU by loading the CUDA extension (`using CUDA, CUDSS_jll`) and passing
`device = CUDADevice` to `mgb_solve`, e.g.
`mgb_solve(Zoo.p_harmonic(mg); device = CUDADevice)`. The problem is assembled on the
CPU and moved to the device (and the solution back) by `mgb_solve`.
"""
module Zoo

using ..MultiGridBarrier
using StaticArrays
import ..MultiGridBarrier: MultiGrid, default_D, default_idx, amg_dim,
        convex_Euclidian_power, convex_linear, intersect, assemble

export elastoplastic_torsion, minimal_surface, p_harmonic, norton_hoff,
        rof, two_sided_obstacle

# Spatial dimension d, extracted from a MultiGrid.
_dim(mg) = amg_dim(mg.geometry.discretization)

include("elastoplastic_torsion.jl")
include("minimal_surface.jl")
include("p_harmonic.jl")
include("norton_hoff.jl")
include("rof.jl")
include("two_sided_obstacle.jl")

end  # module Zoo
