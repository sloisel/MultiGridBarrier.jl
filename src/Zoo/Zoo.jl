"""
    Zoo

A library of convex variational test problems. Each constructor takes a
`MultiGrid` and returns a `NamedTuple` of `mgb_solve` keyword arguments;
splat with `mgb_solve(; problem...)`.

# GPU support (planned)
The current implementations are **CPU-only**. The non-trivial `A`/`b`
arguments to `convex_Euclidian_power` and `convex_linear` are passed as
closures, which is the friendly path on CPU but is unfriendly on GPU
(non-isbits captures in broadcast kernels). The intended escape route is
a future `native_to_cuda(problem::NamedTuple)` method that pre-evaluates
`f`, `g`, and the constraint-data closures on CPU, ships the resulting
grids to GPU, and returns a structurally-identical `NamedTuple` whose
`mg`, `f_grid`, `g_grid`, and `Q.args` arrays all live on the device.
Until that lands, build a CPU `mg`, build the `Zoo` problem, and solve
on CPU.
"""
module Zoo

using ..MultiGridBarrier
using StaticArrays
import ..MultiGridBarrier: MultiGrid, default_D, default_idx, amg_dim,
        convex_Euclidian_power, convex_linear, intersect

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
