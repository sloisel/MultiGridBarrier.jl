@doc raw"""
    module MultiGridBarrier

MultiGridBarrier solves nonlinear convex optimization problems in function spaces using
a barrier (interior-point) method accelerated by a multigrid hierarchy constructed from
your chosen discretization (FEM or spectral). The package provides simple, high-level
entry points as well as a general solver that accept a "geometry" and optional keywords.

## A gentle introduction via the p-Laplacian
For a domain Ω ⊂ ℝᵈ and p ≥ 1, consider the variational problem
```math
\min_{u} \; J(u) = \int_{\Omega} \|\nabla u\|_2^p + f\,u \, dx
```
subject to appropriate boundary conditions (e.g., homogeneous Dirichlet). The Euler–Lagrange
equation gives the p-Laplace PDE:
```math
\nabla \cdot \big(\|\nabla u\|_2^{p-2}\,\nabla u\big) = \tfrac{1}{p}\,f \quad \text{in } \Omega,
```
with the specified boundary conditions. This connection is obtained by integration by parts
applied to the first variation of J(u).

## Constrained linear reformulation with a slack variable
Introduce a slack $s(x) \geq \|\nabla u(x)\|_2^p$ and rewrite the objective using s:
```math
\min_{u,\,s} \; \int_{\Omega} s + f\,u \, dx
\quad \text{subject to}\quad s \ge \|\nabla u\|_2^p.
```
This is a convex optimization problem with a linear objective and convex constraints.
In discrete form, we bundle the state into z, and apply a block "differential" operator D
so that
```math
D z = \begin{bmatrix} u \\ \nabla u \\ s \end{bmatrix},
\qquad
c^\top = \begin{bmatrix} f & 0 & 1 \end{bmatrix}.
```
The problem becomes
```math
\min_{z} \int_{\Omega} c(x)^\top \, (D z)(x) \, dx
\quad \text{subject to}\quad (u,q,s) \in \mathcal{Q} := \{ s \ge \|q\|_2^p \}\ \text{pointwise},
```
which MultiGridBarrier solves by a barrier method. An illustrative (simple) barrier for
\mathcal{Q} is
```math
\mathcal{F}(q,s) = -\log\!\big(s^{2/p} - \|q\|_2^2\big) - 2\log s,
```
and the method minimizes the barrier-augmented functional
```math
\int_{\Omega} t\, c(x)^\top (D z)(x) + \mathcal{F}\!\big((D z)(x)\big) \, dx
```
for increasing barrier parameter t. Internally, the solve proceeds on a hierarchy of grids
with damped Newton steps and line search, but these details are abstracted away.

## How to use it (discretizations and solvers)
- Choose a geometry (discretization and multilevel structure):
  - `fem1d(; L=4)`         → 1D FEM on [-1, 1] with 2^L elements
  - `fem2d(; L=2, K=...)`  → 2D FEM (quadratic + bubble triangles)
  - `spectral1d(; n=16)`   → 1D spectral (Chebyshev/Clenshaw–Curtis)
  - `spectral2d(; n=4)`    → 2D spectral (tensor Chebyshev)
- Solve with a convenience wrapper (recommended to start):
  - `fem1d_solve(; kwargs...)`
  - `fem2d_solve(; kwargs...)`
  - `spectral1d_solve(; kwargs...)`
  - `spectral2d_solve(; kwargs...)`
- Or call the general solver directly:
  - `amgb(geometry; kwargs...)` → `NamedTuple`

## Quick examples
```julia
# 1D FEM p-Laplace
z = fem1d_solve(L=5, p=1.0).z

# 2D spectral p-Laplace
z = spectral2d_solve(n=8, p=2.0).z

# 2D FEM with custom boundary data
g_custom(x) = [sin(π*x[1])*sin(π*x[2]), 10.0]
z = fem2d_solve(L=3; p=1.0, g=g_custom).z

# Time-dependent (implicit Euler, returns nodes × components × timesteps)
U = parabolic_solve(h=0.1, L=3)
```

## Inputs and defaults (high level)
- `p::Real` = 1.0: exponent in the p-Laplace term
- `g`, `f`: boundary/initial data and forcing; either as functions `g(x)`, `f(x)` or as grids `g_grid`, `f_grid`
- `D` and `state_variables`: symbolic specifications of which operators act on which variables
  (sane defaults provided based on the geometry’s dimension)
- `Q`: convex set (by default, a p-Laplace-compatible set via `convex_Euclidian_power`)
- `verbose`, `logfile`: visualization and logging
- Advanced control: `tol`, `t`, `t_feasibility`, `line_search`, `stopping_criterion`, `finalize`

## What you get back
All top-level solvers return a solution object with fields:
- `z::Matrix`: solution on the finest grid (nodes × components)
- `SOL_main`, `SOL_feasibility`: per-phase diagnostics
- `log::String`: textual log for debugging
- `geometry`: the `Geometry` used to construct the multilevel operators
The solution object supports `plot(sol)` to visualize the first component.

## Utilities
- `interpolate(geometry, z, points)`: evaluate the discrete solution at arbitrary points
- `plot(geometry, z)`: plot 1D curves or 2D surfaces; `plot(geometry, U; interval=...)` animates in time
- Convex set helpers: `convex_Euclidian_power`, `convex_linear`, `convex_piecewise`, `intersect`

## Errors and diagnostics
- Throws `AMGBConvergenceFailure` if the feasibility subproblem or the main solve cannot converge
- Set `verbose=true` for a progress bar; inspect `SOL_main`/feasibility and `log` for details

## See also
- Discretizations: `fem1d`, `fem2d`, `spectral1d`, `spectral2d`
- Solvers: `amgb`, `fem1d_solve`, `fem2d_solve`, `spectral1d_solve`, `spectral2d_solve`, `parabolic_solve`
- Convex: `convex_Euclidian_power`, `convex_linear`, `convex_piecewise`, `intersect`
- Visualization & sampling: `plot`, `interpolate`
"""
module MultiGridBarrier

using SparseArrays
using LinearAlgebra
using PyPlot
import PyPlot: plot
using PyCall
using ForwardDiff
using ProgressMeter
using QuadratureRules
import Base: intersect

include("AlgebraicMultiGridBarrier.jl")
include("fem1d.jl")
include("fem2d.jl")
include("spectral1d.jl")
include("spectral2d.jl")
include("Parabolic.jl")

end
