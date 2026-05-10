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
2. **Attach a multigrid hierarchy** with either `amg(geom)` (algebraic multigrid) or
   `geometric_mg(geom, L)` (geometric subdivision with `L` levels). Both return a
   `MultiGrid`.
3. **Solve** with `mgb_solve(mg; kwargs...)`.

For a separate fine-mesh refinement step without keeping the transfer operators, use
`subdivide(geom, L)`.

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
plot(mgb_solve(geometric_mg(fem2d_P2(), 3); p=1.0, verbose=false));
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

For $p \geq 1$ and domain $\Omega$, the solution $u$ of the $p$-Laplace problem is the minimizer of
$$J(u) = \|\nabla u\|_{L^p(\Omega)}^p + \int_{\Omega} fu,$$
where $u$ is in a suitable space of function satisfying, e.g. Dirichlet conditions, and $f$ is a forcing.
This definition must be modified for the $\infty$-Laplace problem. Here we show how to minimize:
$$J(u) = \|\nabla u\|_{L^\infty(\Omega)}^p + \int_{\Omega} fu.$$
We put $p=1$ for simplicity.

```@example 1
plot(mgb_solve(amg(fem1d(; nodes)); p=1.0, state_variables=[:u :dirichlet; :s :uniform], verbose=false));
savefig("fem1dinfty.svg"); nothing # hide
close() #hide
```

![](fem1dinfty.svg)

## Parabolic problems

A time-dependent problem:

```@example 1
plot(parabolic_solve(geometric_mg(fem2d_P1(), 3); h=0.1, verbose=false))
```

## 3D Finite Elements

The `Mesh3d` submodule provides 3D hexahedral finite elements using PyVista for visualization.

```@example 1
sol = mgb_solve(geometric_mg(fem3d(; k=1), 2); verbose=false)
fig = plot(sol)
savefig(fig, "fem3d_demo.png"); nothing # hide
```

![](fem3d_demo.png)

A time-dependent 3D problem:

```@example 1
plot(parabolic_solve(geometric_mg(fem3d(; k=1), 2); h=0.1, verbose=false))
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

Pass the resulting `Geometry` to one of these to obtain a `MultiGrid`:

- `amg(geom)` — algebraic-multigrid hierarchy on the fine mesh.
- `geometric_mg(geom, L)` — geometric subdivision hierarchy with `L` levels (the finest
  mesh has each input element subdivided `L−1` times).
- `subdivide(geom, L)` — geometric mesh refinement only (returns a `Geometry`, drops
  the transfer operators). Combine with `amg(subdivide(geom, L))` for AMG on a finer
  mesh.

### Inputs: the `K` mesh

All FEM constructors take their fine mesh as a `K` keyword argument
(`fem1d(; nodes)` derives `K` from `nodes` by default). `K` is always a **matrix** with
one row per local node and one column per spatial dimension. The format is the *broken*
/ discontinuous-Galerkin convention: each element carries its own copy of every local
node, and shared degrees of freedom are deduplicated by coincident coordinates at
assembly time. So in 1D, nodes `x[1] < x[2] < … < x[m]` defining elements
`[x[1],x[2]], [x[2],x[3]], …, [x[m-1],x[m]]` are encoded as the `2(m−1) × 1` matrix

```julia
K = [x[1]; x[2]; x[2]; x[3]; x[3]; x[4]; … ; x[m-1]; x[m-1]; x[m];;]
```

with each interior node appearing twice (once per neighbouring element). The same
convention applies in 2D and 3D: `fem2d_P1`'s `K` has 3 rows per triangle and 2
columns, `fem2d_P2`'s `K` has 7 rows per triangle (corners + edge midpoints + centroid)
and 2 columns, and `fem3d`'s `K` has `(k+1)^3` rows per hex and 3 columns.

### `geometric_mg` vs `subdivide` vs `amg`

- `amg(geom)` builds the hierarchy from the fine mesh's continuous-P1 stiffness via
  algebraic multigrid. It's the right choice when you've already produced a fine `K`
  (e.g. via your own mesher) and just want a hierarchy on it.
- `geometric_mg(geom, L)` subdivides geometrically `L−1` times and produces a
  hierarchy of `L` levels by uniform subdivision — useful for demos and quick
  experiments on the reference domain.
- `subdivide(geom, L)` is mesh refinement only: it returns the fine `Geometry` from
  `geometric_mg(geom, L)`, discarding the transfer operators. Compose with
  `amg(subdivide(geom, L))` if you want AMG on a finer mesh than `geom` directly.

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
