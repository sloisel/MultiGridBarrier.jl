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
   `fem2d`, `fem2d_P1`, `fem2d_P2`, `fem3d`, `spectral1d`, `spectral2d`). These take only
   mesh inputs — no `L`, no hierarchy.
2. **Attach a multigrid hierarchy** with `amg(geom)` (algebraic multigrid). Returns a
   `MultiGrid`.
3. **Assemble** the problem with `assemble(mg; kwargs...)`, returning an `MGBProblem`.
4. **Solve** with `mgb_solve(prob; kwargs...)`.

To solve on a GPU, load the CUDA extension (`using CUDA, CUDSS_jll`) and pass
`device = CUDADevice` to `mgb_solve`; the assembled problem is moved to the device, solved
there, and the returned `MGBSOL` is moved back to native CPU types. With a functional GPU
present this becomes the default device — pass `device = CPUDevice` to force the CPU.

If you want a finer mesh than the single-level `geom` provides, refine it first with
`subdivide(geom, L)` and then attach AMG: `amg(subdivide(geom, L))`. The legacy
`geometric_mg(geom, L)` builds a geometric-subdivision hierarchy instead of AMG; it is
still available for callers that specifically want geometric transfers.

### Choosing the AMG coarsening

For the FEM discretizations, `amg(geom; prolongator=...)` selects how the multigrid
hierarchy is built. A *prolongator* is a callable mapping a stiffness matrix to its
level prolongations; three factories are provided (each forwards keyword arguments
such as `max_coarse` to the underlying builder):

- `amg_ruge_stuben(; kwargs...)` — classical Ruge–Stüben (**the default**), via
  [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).
- `amg_smoothed_aggregation(; kwargs...)` — smoothed aggregation, via the same package.
- `amg_pyamg(; solver=:rootnode, kwargs...)` — the Python
  [`pyamg`](https://github.com/pyamg/pyamg) package (`:rootnode` energy-minimization,
  `:smoothed_aggregation`, or `:ruge_stuben`), imported lazily through `PyCall`.

```julia
mgb_solve(assemble(amg(fem2d_P1(); prolongator = amg_ruge_stuben(max_coarse=4)); p=1.5); verbose=false);
```

## Finite elements

A 1d p-Laplace problem:
```@example 1
using PyPlot # hide
using MultiGridBarrier
nodes = collect(range(-1.0, 1.0, length=33))
geom = fem1d(; nodes)
plot(mgb_solve(assemble(amg(geom); p=1.0); verbose=false));
savefig("fem1d.svg"); nothing # hide
close() #hide
```

![](fem1d.svg)

A 2d p-Laplace problem on tensor-product Q_k quadrilaterals (here biquadratic, `k=2`):
```@example 1
plot(mgb_solve(assemble(amg(subdivide(fem2d(; k=2), 3)); p=1.0); verbose=false));
savefig("fem2d.svg"); nothing # hide
close() #hide
```

![](fem2d.svg)

The same problem on P2+cubic-bubble triangles (`fem2d_P2`):
```@example 1
plot(mgb_solve(assemble(amg(subdivide(fem2d_P2(), 3)); p=1.0); verbose=false));
savefig("fem2d_P2.svg"); nothing # hide
close() #hide
```

![](fem2d_P2.svg)

## Spectral elements

1d p-Laplace via spectral elements:
```@example 1
plot(mgb_solve(assemble(amg(spectral1d(n=40)); p=1.0); verbose=false));
savefig("spectral1d.svg"); nothing # hide
close() #hide
```

![](spectral1d.svg)

A 2d p-Laplace problem:
```@example 1
plot(mgb_solve(assemble(amg(spectral2d(n=5)); p=1.5); verbose=false));
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
plot(mgb_solve(assemble(amg(fem1d(; nodes)); p=1.0, state_variables=[:u :dirichlet; :s :uniform]); verbose=false));
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

`fem3d` provides 3D hexahedral Q_k finite elements — the 3D case of the same
dimension-generic tensor-product family as `fem1d`/`fem2d` — visualized with PyVista.

```@example 1
sol = mgb_solve(assemble(amg(subdivide(fem3d(; k=1), 2))); verbose=false)
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

| Function     | Element                       | Dim | Required kwargs |
| ---          | ---                           | --- | ---             |
| `fem1d`      | Q_k interval (P1 at `k=1`)    | 1D  | `nodes`, `k` (defaulted) |
| `fem2d`      | Q_k quadrilaterals            | 2D  | `K`, `k` (defaulted) |
| `fem2d_P1`   | P1 triangles                  | 2D  | `K` (defaulted) |
| `fem2d_P2`   | P2 + cubic bubble triangles   | 2D  | `K` (defaulted) |
| `fem3d`      | Q_k hexahedra                 | 3D  | `K`, `k` (defaulted) |
| `spectral1d` | spectral (Chebyshev)          | 1D  | `n`             |
| `spectral2d` | spectral (tensor Chebyshev)   | 2D  | `n`             |

`fem1d`/`fem2d`/`fem3d` are the tensor-product Q_k family (the map is
isoparametric, so curved elements are supported); `fem2d_P1`/`fem2d_P2` are the
simplicial P_k family on triangles.

Pass the resulting `Geometry` to `amg(geom)` to obtain a `MultiGrid` — an algebraic-
multigrid hierarchy on the fine mesh. To refine the mesh first, compose with
`subdivide(geom, L)`: `amg(subdivide(geom, L))`.

The legacy `geometric_mg(geom, L)` builds a geometric-subdivision hierarchy instead of
AMG; it remains available for callers that specifically want geometric transfers.

### Inputs: the `K` mesh

All FEM constructors take their fine mesh as a `K` keyword argument
(`fem1d(; nodes)` derives `K` from `nodes` by default). `K` is a 3-tensor
`K::Array{T,3}` of shape `(V, N, D)`:

- `V` is the number of local nodes per element. For the tensor-product Q_k
  family it is `(k+1)^d`: `k+1` for `fem1d`, `(k+1)^2` for `fem2d`, `(k+1)^3`
  for `fem3d` (you may instead pass the `2^d`-corner tensor — 2, 4 or 8 — for
  straight elements, which is promoted internally). For the simplicial family
  `V` is 3 for `fem2d_P1` and 7 for `fem2d_P2`;
- `N` is the number of elements;
- `D` is the spatial dimension (1, 2, or 3).

`K[v, e, d]` is the `d`-th coordinate of the `v`-th local node of the
`e`-th element. The format is the *broken* / discontinuous-Galerkin
convention — each element carries its own copy of every local node, and
shared degrees of freedom are deduplicated by coincident coordinates when the
`Geometry` is built. This determines `geom.t`, the full-node connectivity
(`t[v,e]` = global id of local node `v` in element `e`) that the AMG hierarchy
and boundary detection consult thereafter. The tensor-product constructors
(`fem1d`/`fem2d`/`fem3d`) also accept a `t=` keyword to supply that connectivity
explicitly — bypassing the coordinate dedup — so that geometrically-coincident
nodes can stay topologically distinct on slit domains, branch cuts, and glued
manifolds; [`tensor_dofmap`](@ref) builds such a `t` from corner connectivity.
So in 1D, nodes `x[1] < x[2] < … < x[m]` defining elements
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
Modules = [MultiGridBarrier]
Order   = [:module]
Private = false
```

# Types reference

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:type]
Private = false
```

# Functions reference

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:function]
Private = false
```

# Index

```@index
```
