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
plot(fem1d_solve(L=5,p=1.0,verbose=false));
savefig("fem1d.svg"); nothing # hide
close() #hide
```

![](fem1d.svg)

A 2d p-Laplace problem:
```@example 1
plot(fem2d_P2_solve(L=3,p=1.0,verbose=false));
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
plot(fem1d_solve(L=5,p=1.0,state_variables=[:u :dirichlet; :s :uniform],verbose=false));
savefig("fem1dinfty.svg"); nothing # hide
close() #hide
```

![](fem1dinfty.svg)

## Parabolic problems

A time-dependent problem:

```@example 1
plot(parabolic_solve(fem2d_P2(L=3);h=0.1,verbose=false))
```

## 3D Finite Elements

The `Mesh3d` submodule provides 3D hexahedral finite elements using PyVista for visualization.

```@example 1
sol = fem3d_solve(L=2, verbose=false)
fig = plot(sol)
savefig(fig, "fem3d_demo.png"); nothing # hide
```

![](fem3d_demo.png)

A time-dependent 3D problem:

```@example 1
plot(parabolic_solve(fem3d(L=2);h=0.1,verbose=false))
```

## Algebraic multigrid front-ends

For unstructured fine meshes, you can build the multigrid hierarchy via algebraic
multigrid (AMG) instead of geometric subdivision. You hand over a fine mesh; AMG
figures out the coarse levels (no `L` parameter — coarsening depth is determined
by `max_coarse`). The returned `Geometry` plugs into `amgb` exactly like the
geometric-MG variants. These front-ends are intended for Dirichlet boundary
conditions.

| Function | Element | Dimension | Hierarchy |
| --- | --- | --- | --- |
| `algebraic_fem1d`     | P1                          | 1D | AMG |
| `fem2d_P1`            | P1 triangles                | 2D | geometric |
| `algebraic_fem2d_P1`  | P1 triangles                | 2D | AMG |
| `algebraic_fem2d_P2`     | P2 + cubic bubble triangles | 2D | AMG |
| `algebraic_fem3d`     | Q_k hexahedra               | 3D | AMG |

Each `algebraic_*` function has a matching `algebraic_*_solve` one-liner that
constructs the geometry and runs `amgb` in a single call.

```@example 1
nodes = collect(range(-1.0, 1.0, length=33))
plot(algebraic_fem1d_solve(; nodes, p=1.5, verbose=false));
savefig("algebraic_fem1d.svg"); nothing # hide
close() #hide
```

![](algebraic_fem1d.svg)

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

