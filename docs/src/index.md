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

The `MultiGridBarrier` module features finite element and spectral discretizations in 1d, 2d, and 3d.

## Citation

If you use this package in your research, please cite:

> S. Loisel, "The spectral barrier method to solve analytic convex optimization problems in function spaces," *Numerische Mathematik*, 2025. DOI: [10.1007/s00211-025-01508-0](https://doi.org/10.1007/s00211-025-01508-0)

BibTeX:
```bibtex
@article{Loisel2025,
  author = {Loisel, Sébastien},
  title = {The spectral barrier method to solve analytic convex optimization problems in function spaces},
  journal = {Numerische Mathematik},
  year = {2025},
  doi = {10.1007/s00211-025-01508-0}
}
```

## Finite elements

After installing `MultiGridBarrier` with the Julia package manager, in a Jupyter notebook, one solves a 1d p-Laplace problem as follows:
```@example 1
using PyPlot # hide
using MultiGridBarrier
nodes = collect(range(-1.0, 1.0, length=33))
plot(fem1d_solve(; nodes, p=1.0, verbose=false));
savefig("fem1d.svg"); nothing # hide
close() #hide
```

![](fem1d.svg)

A 2d p-Laplace problem:
```@example 1
plot(fem2d_P2_solve(L=3, p=1.0, verbose=false));
savefig("fem2d_P2.svg"); nothing # hide
close() #hide
```

![](fem2d_P2.svg)

## Spectral elements

Solve a 1d p-Laplace problem using spectral methods as follows:
```@example 1
plot(spectral1d_solve(n=40,p=1.0,verbose=false));
savefig("spectral1d.svg"); nothing # hide
close() #hide
```

![](spectral1d.svg)

A 2d p-Laplace problem:
```@example 1
plot(spectral2d_solve(n=5,p=1.5,verbose=false));
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
plot(fem1d_solve(; nodes, p=1.0, state_variables=[:u :dirichlet; :s :uniform], verbose=false));
savefig("fem1dinfty.svg"); nothing # hide
close() #hide
```

![](fem1dinfty.svg)

## Parabolic problems

A time-dependent problem:

```@example 1
plot(parabolic_solve(fem2d_P1(L=3); h=0.1, verbose=false))
```

## 3D Finite Elements

The `Mesh3d` submodule provides 3D hexahedral finite elements using PyVista for visualization.

```@example 1
sol = fem3d_solve(L=2, k=1, verbose=false)
fig = plot(sol)
savefig(fig, "fem3d_demo.png"); nothing # hide
```

![](fem3d_demo.png)

A time-dependent 3D problem:

```@example 1
plot(parabolic_solve(fem3d(L=2, k=1); h=0.1, verbose=false))
```

## Front-end summary

The default FEM front-ends (`fem1d`, `fem2d_P1`, `fem2d_P2`, `fem3d`) build
the multigrid hierarchy via algebraic multigrid (AMG). They accept either a
fine mesh `K` directly *or* an `L` kwarg (for `fem2d_P1`, `fem2d_P2`, `fem3d`)
that subdivides a default coarse mesh geometrically `L−1` times before AMG.
AMG coarsening below the fine mesh is set by `max_coarse`. They are the
recommended entry points for FEM problems and are intended for Dirichlet
boundary conditions. Each has a matching `*_solve` one-liner that constructs
the geometry and runs `amgb` in a single call.

The `geometric_*` variants build the hierarchy by uniform geometric
subdivision instead of AMG; otherwise the API is the same.

| Function              | Element                     | Dim | Hierarchy |
| ---                   | ---                         | --- | ---       |
| `fem1d`               | P1                          | 1D  | AMG       |
| `geometric_fem1d`     | P1                          | 1D  | geometric |
| `fem2d_P1`            | P1 triangles                | 2D  | AMG       |
| `geometric_fem2d_P1`  | P1 triangles                | 2D  | geometric |
| `fem2d_P2`            | P2 + cubic bubble triangles | 2D  | AMG       |
| `geometric_fem2d_P2`  | P2 + cubic bubble triangles | 2D  | geometric |
| `fem3d`               | Q_k hexahedra               | 3D  | AMG       |
| `geometric_fem3d`     | Q_k hexahedra               | 3D  | geometric |

### Inputs: the `K` mesh and the `L` kwarg

All FEM front-ends take their fine mesh as a `K` keyword argument
(`fem1d(K=…)`, `fem2d_P2(K=…)`, `fem3d(K=…)`, …). The format is the
*broken* / discontinuous-Galerkin convention: each element carries its
own copy of every local node, and shared degrees of freedom are
deduplicated by coincident coordinates at assembly time. So in 1D,
nodes `x[1] < x[2] < … < x[m]` defining elements
`[x[1],x[2]], [x[2],x[3]], …, [x[m-1],x[m]]` are encoded as the
`2(m−1) × 1` matrix

```
K = [x[1], x[2], x[2], x[3], x[3], x[4], …, x[m-1], x[m-1], x[m]]
```

with each interior node appearing twice (once per neighbouring element).
The same convention applies in 2D and 3D: `fem2d_P1`'s `K` has 3 rows
per triangle, `fem2d_P2`'s `K` has 7 rows per triangle (corners + edge
midpoints + centroid), and `fem3d`'s `K` has `(k+1)^3` rows per hex.

In addition to `K`, the 2D/3D AMG front-ends accept an `L` kwarg
(default `L=1`). **`L` controls mesh generation, not solver depth.** When
`L>1`, the solver silently replaces your `K` with a finer mesh obtained
by subdividing each element `L−1` times geometrically. The AMG hierarchy
is then built from that finer mesh. `L=1` is the right default for real
work — most users should pass their own fine mesh as `K` and leave `L=1`.

This matters for performance comparisons. `fem2d_P2_solve(L=3)` is *not*
the same problem as `fem2d_P2_solve(L=1)`: the former runs on a mesh
with 16× the elements and is correspondingly more expensive. If you
benchmark this solver against another and they're solving meshes of
different sizes, the comparison is meaningless.

`L>1` is mainly a convenience for quick experiments and demos on the
default reference domain. For real work, generate the mesh you want once
(with your mesher of choice, or with a `geometric_*` call), pass it as
`K`, and leave `L=1`.

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

