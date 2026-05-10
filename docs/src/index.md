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
K = geometric_fem2d_P1(L=3).x  # fine triangulation of the unit square
plot(fem2d_P2_solve(; K, p=1.0, verbose=false));
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
K_2d = geometric_fem2d_P1(L=3).x
plot(parabolic_solve(fem2d_P1(; K=K_2d); h=0.1, verbose=false))
```

## 3D Finite Elements

The `Mesh3d` submodule provides 3D hexahedral finite elements using PyVista for visualization.

```@example 1
K_3d = MultiGridBarrier.geometric_fem3d(L=2, k=1).x  # 8-corner-per-hex Q1 mesh
sol = fem3d_solve(; K=K_3d, k=1, verbose=false)
fig = plot(sol)
savefig(fig, "fem3d_demo.png"); nothing # hide
```

![](fem3d_demo.png)

A time-dependent 3D problem:

```@example 1
plot(parabolic_solve(fem3d(; K=K_3d, k=1); h=0.1, verbose=false))
```

## Front-end summary

The default FEM front-ends (`fem1d`, `fem2d_P1`, `fem2d_P2`, `fem3d`) build the
multigrid hierarchy via algebraic multigrid (AMG) from a fine mesh you supply.
There is no `L` parameter — coarsening depth is determined by `max_coarse`.
They are the recommended entry points for FEM problems on user-provided meshes
and are intended for Dirichlet boundary conditions. Each has a matching
`*_solve` one-liner that constructs the geometry and runs `amgb` in a single
call.

For meshes built by repeated geometric subdivision, the `geometric_*` variants
take an integer `L` (number of refinement levels) and a small coarse mesh `K`,
and build the hierarchy by uniform subdivision instead of AMG.

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

