export mgb_solve, amg, geometric_mg, subdivide, MultiGrid,
       Geometry, Convex, convex_linear, convex_Euclidian_power, convex_piecewise,
       MGBConvergenceFailure, linesearch_illinois, linesearch_backtracking,
       stopping_exact, stopping_inexact, interpolate, intersect, plot,
       find_boundary

# The algebraic multigrid barrier solver, split by theme into the files below.
# These are not standalone modules — each is included directly into module
# MultiGridBarrier. Include order matters only where a type is referenced in a
# later file's struct fields or method signatures (e.g. `Geometry`/`MultiGrid`
# before the convex sets that annotate `mg::MultiGrid`), so it is kept here.
include("utils.jl")                   # logging, interpolation, mesh helpers
include("multigrid.jl")               # Geometry, MultiGrid, AMG hierarchy, amg()
include("convex.jl")                  # Barrier/Convex types, intersect, barrier(Q)
include("convex_linear.jl")           # linear inequality constraints
include("convex_euclidian_power.jl")  # Euclidian-power constraints
include("convex_piecewise.jl")        # piecewise-active constraints
include("newton.jl")                  # Newton iteration + line searches
include("mgb.jl")                     # the MGB V-cycle and mgb_solve
