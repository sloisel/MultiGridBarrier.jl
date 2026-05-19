```@meta
CurrentModule = MultiGridBarrier
```

```@eval
using Markdown
using Pkg
using MultiGridBarrier
v = string(pkgversion(MultiGridBarrier))
md"# MultiGridBarrier $v"
```

[MultiGridBarrier](https://github.com/sloisel/MultiGridBarrier.jl) is a Julia module for solving nonlinear convex optimization problems in function spaces, such as p-Laplace problems. When regularity conditions are satisfied, the solvers are quasi-optimal.

The package features finite element and spectral discretizations in 1d, 2d, and 3d.

## Citation

If you use this package in your research, please cite:

> S. Loisel, "The spectral barrier method to solve analytic convex optimization problems in function spaces," *Numerische Mathematik*, vol. 158, no. 1, pp. 281–302, 2026. DOI: [10.1007/s00211-025-01508-0](https://doi.org/10.1007/s00211-025-01508-0)

BibTeX:
```bibtex
@article{loisel2026spectral,
  author    = {Loisel, Sébastien},
  title     = {The spectral barrier method to solve analytic convex optimization problems in function spaces},
  journal   = {Numerische Mathematik},
  volume    = {158},
  number    = {1},
  pages     = {281--302},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/s00211-025-01508-0}
}
```

## Workflow at a glance

Every problem is solved with the same three-step pattern:

1. **Build a single-level `Geometry`** with one of the mesh constructors (`fem1d`,
   `fem2d_P1`, `fem2d_P2`, `fem3d`, `spectral1d`, `spectral2d`). These take only mesh
   inputs — no `L`, no hierarchy.
2. **Attach a multigrid hierarchy** with `amg(geom)` (algebraic multigrid). Returns a
   `MultiGrid`.
3. **Solve** with `mgb_solve(mg; kwargs...)`.

If you want a finer mesh than the single-level `geom` provides, refine it first with
`subdivide(geom, L)` and then attach AMG: `amg(subdivide(geom, L))`. The legacy
`geometric_mg(geom, L)` builds a geometric-subdivision hierarchy instead of AMG; it is
still available for callers that specifically want geometric transfers.

## Finite elements

A 1d p-Laplace problem:
```@example 1
using PyPlot # hide
using MultiGridBarrier
nodes = collect(range(-1.0, 1.0, length=33))
geom = fem1d(; nodes)
plot(mgb_solve(amg(geom); p=1.0, verbose=false));
savefig("fem1d.svg"); nothing # hide
close() #hide
```

![](fem1d.svg)

A 2d p-Laplace problem on a uniformly subdivided unit-square mesh:
```@example 1
plot(mgb_solve(amg(subdivide(fem2d_P2(), 3)); p=1.0, verbose=false));
savefig("fem2d_P2.svg"); nothing # hide
close() #hide
```

![](fem2d_P2.svg)

## Spectral elements

1d p-Laplace via spectral elements:
```@example 1
plot(mgb_solve(amg(spectral1d(n=40)); p=1.0, verbose=false));
savefig("spectral1d.svg"); nothing # hide
close() #hide
```

![](spectral1d.svg)

A 2d p-Laplace problem:
```@example 1
plot(mgb_solve(amg(spectral2d(n=5)); p=1.5, verbose=false));
savefig("spectral2d.svg"); nothing # hide
close() #hide
```

![](spectral2d.svg)

## Solving $\infty$-Laplacians

For $p \geq 1$ on a domain $\Omega$, the $p$-Laplace problem minimises
$$J(u) = \|\nabla u\|_{L^p(\Omega)}^p + \int_{\Omega} fu,$$
where $u$ lies in a suitable function space satisfying, e.g., Dirichlet conditions and $f$ is a forcing.
The $\infty$-Laplace problem replaces the inner norm by $L^\infty$:
$$J(u) = \|\nabla u\|_{L^\infty(\Omega)}^p + \int_{\Omega} fu.$$
We take $p=1$ below for simplicity.

```@example 1
plot(mgb_solve(amg(fem1d(; nodes)); p=1.0, state_variables=[:u :dirichlet; :s :uniform], verbose=false));
savefig("fem1dinfty.svg"); nothing # hide
close() #hide
```

![](fem1dinfty.svg)

## Parabolic problems

A time-dependent problem:

```@example 1
plot(parabolic_solve(amg(subdivide(fem2d_P1(), 3)); h=0.1, verbose=false))
```

## 3D Finite Elements

The `Mesh3d` submodule provides 3D hexahedral finite elements using PyVista for visualization.

```@example 1
sol = mgb_solve(amg(subdivide(fem3d(; k=1), 2)); verbose=false)
fig = plot(sol)
savefig(fig, "fem3d_demo.png"); nothing # hide
```

![](fem3d_demo.png)

A time-dependent 3D problem:

```@example 1
plot(parabolic_solve(amg(subdivide(fem3d(; k=1), 2)); h=0.1, verbose=false))
```

## Front-end summary

The mesh constructors below all return a single-level `Geometry`. They are intended for
Dirichlet boundary conditions.

| Function     | Element                     | Dim | Required kwargs |
| ---          | ---                         | --- | ---             |
| `fem1d`      | P1                          | 1D  | `nodes`         |
| `fem2d_P1`   | P1 triangles                | 2D  | `K` (defaulted) |
| `fem2d_P2`   | P2 + cubic bubble triangles | 2D  | `K` (defaulted) |
| `fem3d`      | Q_k hexahedra               | 3D  | `K`, `k` (defaulted) |
| `spectral1d` | spectral (Chebyshev)        | 1D  | `n`             |
| `spectral2d` | spectral (tensor Chebyshev) | 2D  | `n`             |

Pass the resulting `Geometry` to `amg(geom)` to obtain a `MultiGrid` — an algebraic-
multigrid hierarchy on the fine mesh. To refine the mesh first, compose with
`subdivide(geom, L)`: `amg(subdivide(geom, L))`.

The legacy `geometric_mg(geom, L)` builds a geometric-subdivision hierarchy instead of
AMG; it remains available for callers that specifically want geometric transfers.

### Inputs: the `K` mesh

All FEM constructors take their fine mesh as a `K` keyword argument
(`fem1d(; nodes)` derives `K` from `nodes` by default). `K` is a 3-tensor
`K::Array{T,3}` of shape `(V, N, D)`:

- `V` is the number of local nodes per element (2 for `fem1d`, 3 for
  `fem2d_P1`, 7 for `fem2d_P2`, `(k+1)^3` for `fem3d`);
- `N` is the number of elements;
- `D` is the spatial dimension (1, 2, or 3).

`K[v, e, d]` is the `d`-th coordinate of the `v`-th local node of the
`e`-th element. The format is the *broken* / discontinuous-Galerkin
convention — each element carries its own copy of every local node, and
shared degrees of freedom are deduplicated by coincident coordinates at
assembly time. So in 1D, nodes `x[1] < x[2] < … < x[m]` defining elements
`[x[1],x[2]], [x[2],x[3]], …, [x[m-1],x[m]]` give a `(2, m-1, 1)` tensor

```julia
K = reshape(T[x[1], x[2], x[2], x[3], x[3], x[4], …, x[m-1], x[m]], 2, m-1, 1)
```

with each interior node appearing twice (once per neighbouring element).
The stored `geom.x` carries the same shape as `K`; the flat
`(V·N, D)` view used by sparse operators is recovered via
`reshape(geom.x, :, size(geom.x, 3))`.

Spectral discretizations (`spectral1d`, `spectral2d`) have no element
structure and use `N = 1` — `geom.x` has shape `(n, 1, 1)` in 1D and
`(n², 1, 2)` in 2D.

### `amg` vs `subdivide` vs `geometric_mg`

- `amg(geom)` is the recommended hierarchy: it builds AMG from the fine mesh's
  continuous-P1 stiffness. Works whether `geom` came from your own mesher or from
  uniform subdivision via `subdivide`.
- `subdivide(geom, L)` is mesh refinement only: it returns a fine `Geometry` obtained
  by subdividing each input element `L−1` times. Compose with `amg(subdivide(geom, L))`
  to get AMG on the refined mesh.
- `geometric_mg(geom, L)` is the legacy alternative: it produces a hierarchy of `L`
  levels using purely geometric subdivision transfers (no AMG coarsening). Kept for
  cases that need geometric transfers specifically.

# Module reference

```@autodocs
Modules = [MultiGridBarrier, MultiGridBarrier.Mesh3d]
Order   = [:module]
Private = false
```

# Types reference

```@autodocs
Modules = [MultiGridBarrier, MultiGridBarrier.Mesh3d]
Order   = [:type]
Private = false
```

# Functions reference

```@autodocs
Modules = [MultiGridBarrier, MultiGridBarrier.Mesh3d]
Order   = [:function]
Private = false
```

## PyPlot extensions

```@docs
PyPlot.savefig(::MultiGridBarrier.Mesh3d.Plotting.MGB3DFigure, ::String)
```

# Index

```@index
```
