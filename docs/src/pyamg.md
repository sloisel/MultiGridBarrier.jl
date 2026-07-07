```@meta
CurrentModule = MultiGridBarrier
```

# PyAMG prolongators

The multigrid hierarchy that `amg(geom)` builds is driven by a *prolongator* —
a callable mapping a stiffness matrix to its level prolongations. Two
pure-Julia factories ship in the core (`amg_ruge_stuben`, the default, and
`amg_smoothed_aggregation`, both via
[AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)).
A third is backed by the Python [pyamg](https://github.com/pyamg/pyamg)
package and lives in the `MultiGridBarrierPyAMGExt` extension:

!!! note "Requires PyCall"
    Add PyCall to your environment (`pkg> add PyCall`) and load both packages:
    `using MultiGridBarrier, PyCall`. The Python `pyamg` and `scipy` packages
    are imported lazily on the first call, installing from conda-forge if
    necessary. (Loading PyPlot for [plotting](plotting.md) also loads PyCall,
    so the plotting setup enables this extension too.)

```julia
using MultiGridBarrier, PyCall
geom = subdivide(fem2d_P1(), 4)
mg   = amg(geom; prolongator = amg_pyamg(solver = :rootnode))
sol  = mgb_solve(assemble(mg; p = 1.0))
```

`solver` selects pyamg's `:rootnode` (energy-minimization, the default),
`:smoothed_aggregation`, or `:ruge_stuben`; remaining keyword arguments are
forwarded to the pyamg solver constructor.

When is this worth reaching for? The pure-Julia Ruge–Stüben default is robust
across the package's benchmarks. Aggregation-based coarsening —
`amg_pyamg(solver = :rootnode)` in particular — is an escape hatch for highly
anisotropic problems near `p = 1`, especially combined with the
`auxiliary_postprocess` keyword of `amg` (see the [`amg`](@ref) docstring),
where classical coarsening can blow up Newton iteration counts on the central
path.

## API reference

```@docs
amg_pyamg
```
