@doc raw"""
    module MultiGridBarrier

Module `MultiGridBarrier` solves convex optimization problems in function spaces, for example, solving p-Laplace problems. We recommend to start with the functions `fem1d_solve()`, `fem2d_solve()`, `spectral1d_solve()`, `spectral2d_solve()`. These functions are sufficient to solve p-Laplace problems in 1d or 2d, using finite or spectral elements.

For more general use, the user will need to familiarize themselves with the basic ideas of convex optimization.

* Overview of convex optimization in function spaces by MultiGrid Barrier method.

The general idea is to build a multigrid hierarchy, represented by an `AMG` object, and barrier for a convex set, represented by a `Barrier` object, and then solve a convex optimization problem using the `amgb()` solver.

To generate the multigrid hierarchy represented by the `AMG` object, use either `fem1d()`, `fem2d()`, `spectral1d()` or `spectral2d()` functions. These constructors will assemble suitable `AMG` objects for either FEM or spectral discretizations, in 1d or 2d. One should think of these four constructors as being specialized in constructing some specific function spaces. A user can use the `amg()` constructor directly if custom function spaces are required, but this is more difficult.

We now describe the barrier function.

Assume that ``\Omega \subset \mathbb{R}^d`` is some open set. Consider the example of the p-Laplace problem on ``\Omega``. Let ``f(x)`` be a "forcing" (a function) on ``\Omega``, and ``1 \leq p < \infty``. One wishes to solve the minimization problem
```math
\begin{equation}
\inf_u \int_{\Omega} fu + \|\nabla u\|_2^p \, dx. \tag{1}
\end{equation}
```
Generally speaking, ``u`` will range in some function space, e.g. a space of differentiable functions satisfying homogeneous Dirichlet conditions. Under some conditions, minimizing (1) is equivalent to solving the p-Laplace PDE:
```math
\nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = {1 \over p} f.
```
We introduce the "slack function" ``s(x)`` and replace (1) with the following equivalent problem:
```math
\begin{equation}
\inf_{s(x) \geq \|\nabla u(x)\|_2^p} \int_{\Omega} fu + s \, dx. \tag{2}
\end{equation}
```
Define the convex set ``\mathcal{Q} = \{ (u(x),q(x),s(x)) \; : \; s(x) \geq \|q(x)\|_2^p \}``, and
```math
z = \begin{bmatrix} u \\ s \end{bmatrix}, \qquad
c^T = [f,0,1], \qquad
Dz = \begin{bmatrix} u \\ \nabla u \\ s \end{bmatrix}.
```
Then, (2) can be rewritten as
```math
\begin{equation}
\inf_{Dz \in \mathcal{Q}} \int_{\Omega} c^T(x)Dz(x) \, dx. \tag{3}
\end{equation}
```
Recall that a barrier for ``\mathcal{Q}`` is a convex function ``\mathcal{F}`` on ``\mathcal{Q}`` such that ``\mathcal{F} < \infty`` in the interior of ``\mathcal{Q}`` and ``\mathcal{F} = \infty`` on the boundary of ``\mathcal{Q}``. A barrier for the p-Laplace problem is:
```math
\mathcal{F}(u,q,s) = \int_{\Omega} -\log(s^{2 \over p} - \|q\|_2^2) - 2\log s \, dx = \int_{\Omega} F(Dz(x)) \, dx.
```

The central path ``z^*(t)`` minimizes, for each fixed ``t>0``, the quantity
```math
\int_{\Omega} tc^TDz + F(Dz) \, dx.
```
As ``t \to \infty``, ``z^*(t)`` forms a minimizing sequence (or filter) for (3). We think of the function ``c(x)`` as the "functional" that we seek to minimize.

The `Convex{T}` type describes various convex sets (denoted ``Q`` above) by way of functions `barrier()`, `cobarrier()` and `slack()`. `barrier` is indeed a barrier for ``Q``, `cobarrier()` is a barrier for a related feasibility problems, and `slack()` is used in solving the feasibility problem. `Convex{T}` objects can be created using the various `convex_...()` constructors, e.g. `convex_Euclidian_power()` for the p-Laplace problem.

Once one has `AMG` and `Convex` objects, and a suitable "functional" `c`, one uses the `amgb()` function to solve the optimization problem by the MultiGrid Barrier method, a variant of the barrier method (or interior point method) that is quasi-optimal for sufficiently regular problems.
"""
module MultiGridBarrier

using SparseArrays
using LinearAlgebra
using PyPlot
using PyCall
using ForwardDiff
using ProgressMeter
using QuadratureRules

include("AlgebraicMultiGridBarrier.jl")
include("fem1d.jl")
include("fem2d.jl")
include("spectral1d.jl")
include("spectral2d.jl")
include("Parabolic.jl")

end
