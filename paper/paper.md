---
title: 'MultiGridBarrier.jl: quasi-optimal solvers for convex variational problems'
tags:
  - Julia
  - partial differential equations
  - finite element method
  - spectral method
  - multigrid
  - interior-point method
  - convex optimization
  - p-Laplacian
authors:
  - name: Sébastien Loisel
    orcid: 0009-0001-6009-5289
    affiliation: 1
affiliations:
  - name: Heriot-Watt University, Edinburgh, United Kingdom
    index: 1
date: 9 June 2026
bibliography: paper.bib
---

# Summary

`MultiGridBarrier.jl` is a Julia [@bezanson2017julia] package for solving convex variational
problems in function spaces. These are the nonlinear partial differential equations (PDEs) and boundary-value
problems that arise from minimizing a convex functional. Representative examples include the
$p$-Laplacian for any $p \in [1, \infty]$, total-variation problems, and obstacle problems.
The most useful of these are *nonsmooth*: the energy is convex but not differentiable (for
example $p = 1$, or total variation), a regime in which Newton-type solvers applied naively
either fail or require a number of iterations that grows rapidly with the mesh resolution.

The package implements the **multigrid barrier method** [@loiselmgb; @loisel2020efficient;
@loisel2026spectral], which couples an interior-point (barrier) method with a multigrid
hierarchy. For the problem classes covered by the supporting theory the method is
*quasi-optimal*: the number of interior-point/Newton iterations grows only mildly with the
number of degrees of freedom $n$. For instance, this count is $O(\sqrt{n}\,\log n)$ for the
$p$-Laplacian [@loisel2020efficient], and polylogarithmic in the analytic, spectral setting
[@loisel2026spectral].

`MultiGridBarrier.jl` provides finite-element discretizations in one, two, and three
dimensions (simplicial $P_1$/$P_2$ elements and tensor-product $Q_k$ elements), as well as
Chebyshev spectral discretizations, all with isoparametric element maps. It builds an
algebraic-multigrid hierarchy automatically, supports user-specified mesh connectivity
(enabling slit domains, branch cuts, and glued manifolds), solves time-dependent problems,
and offers optional GPU acceleration through CUDA. A typical solve is three lines:

```julia
using MultiGridBarrier
geom = fem2d_P2()
sol  = mgb_solve(assemble(amg(geom); p = 1.0))   # a nonsmooth p = 1 problem
```

# Statement of need

Convex variational problems are ubiquitous in computational science: nonlinear elasticity
and plasticity, image denoising and segmentation (total variation), contact and obstacle
problems, and non-Newtonian flow (the $p$-Laplacian). The difficulty is that the most
interesting cases are nonsmooth (the energy is convex but not differentiable), so Newton-type
methods applied naively either stagnate or require an iteration count that grows rapidly as
the mesh is refined.

Interior-point (barrier) methods handle nonsmoothness robustly by following a smooth central
path, but a single barrier solve still requires solving a sequence of large, increasingly
ill-conditioned linear systems. The multigrid barrier method addresses both issues at once: a
multigrid hierarchy preconditions the central-path subproblems so that the *total* cost stays
close to linear in the number of unknowns, with rigorous bounds for the covered problem
classes [@loiselmgb; @loisel2020efficient; @loisel2026spectral].

# State of the field

General-purpose finite-element libraries in Julia, such as `Gridap.jl` [@gridap] and
`Ferrite.jl` [@ferrite], and in the wider ecosystem, such as the FEniCS project [@fenics],
provide flexible tools for discretizing PDEs; algebraic-multigrid libraries such as
`AlgebraicMultigrid.jl` [@AlgebraicMultigrid] and `PyAMG` [@pyamg] solve the resulting linear
systems. These are general building blocks, but none provides an out-of-the-box,
theoretically grounded solver for *nonsmooth convex variational* problems.
`MultiGridBarrier.jl` fills this gap. It packages a discretization, a multigrid hierarchy, and
a barrier solver behind a small high-level interface, and builds on (rather than reinvents)
the Julia ecosystem: it uses `AlgebraicMultigrid.jl`, or optionally `PyAMG`, to coarsen its
auxiliary problems, and runs on both CPU and GPU.

# Software design

A problem is solved with four composable steps. A *mesh constructor* (`fem1d`, `fem2d`,
`fem2d_P1`, `fem2d_P2`, `fem3d`, `spectral1d`, `spectral2d`) returns a `Geometry` describing
the discretization. `amg(geom)` attaches an algebraic-multigrid hierarchy, `assemble(mg; ...)`
builds the convex problem (the functional, its barrier, and any constraints), and
`mgb_solve(prob)` runs the barrier method.

Internally, the multigrid barrier method tracks the central path of an interior-point method
while using a multigrid hierarchy to solve the Newton systems along that path
[@loiselmgb; @loisel2026spectral]. The finite-element discretizations are isoparametric
simplicial $P_k$ and tensor-product $Q_k$ elements. For these, the multigrid hierarchy is built
algebraically: the package coarsens an auxiliary $P_1$/$Q_1$ problem on the element corners and
lifts the resulting transfer operators to the full high-order basis, so the same machinery
serves every finite-element family. Mesh topology is represented by an explicit node
connectivity array, which decouples geometry from topology and lets geometrically coincident
nodes remain distinct, supporting slit domains, branch cuts, and glued manifolds. The package
also provides a structured, batched-GEMM assembly of the Newton Hessians that maps efficiently
onto GPUs [@loiselhpc], a variant for nonuniform grids [@loiselnonuniform], and the algebraic
formulation used here [@loiseldd29].

The Chebyshev spectral discretizations (`spectral1d`, `spectral2d`) use an intrinsically
spectral hierarchy instead. The multigrid levels are a sequence of polynomial approximation
spaces of increasing degree, with exact polynomial interpolation as the inter-level transfer
and zero-trace boundary conditions imposed by basis construction rather than node masking. This
is the setting of the spectral barrier method [@loisel2026spectral]: when the solution is
analytic, the spectral discretization converges geometrically and the overall method is
quasi-optimal, with the iteration count growing only polylogarithmically in the number of
degrees of freedom.

Convex constraints (p-norm / Euclidian-power, linear, and piecewise) are built in and
composable for any discretization, and `parabolic_solve` extends the method to time-dependent
problems.

![A nonsmooth ($p = 1$) solution of a two-dimensional $p$-Laplace problem, computed with
`MultiGridBarrier.jl` on a refined $Q_2$ mesh and rendered with the package's plotting
front-end.](figure.png)

# Research impact

The methods implemented in `MultiGridBarrier.jl` underpin the numerical experiments in the
papers that introduce them [@loisel2020efficient; @loiselmgb; @loisel2026spectral], and the
solver and its earlier implementations have been used by other researchers. For example, Zhang
and Jiang use this algorithm within a convolutional-neural-network reduced-order modeling
method for multiscale problems [@zhangjiang2025cnn]. More broadly, the underlying $p$-Laplacian
algorithm [@loisel2020efficient] has been cited around 26 times (Google Scholar, as of June
2026) across the numerical-PDE and optimization literature, in areas such as computational
$p$-Laplacian numerics [@balci2023kacanov] and $p$-harmonic shape optimization
[@muller2021pharmonic].

# AI usage disclosure

Generative AI assistance (Anthropic Claude, via the Claude Code command-line tool) was used in
preparing this submission: drafting and editing this paper and parts of the documentation and
README, generating the illustrative figure, and assisting with some software changes
(refactoring, a correctness fix, and a mesh-connectivity feature). All AI-assisted output was
reviewed, validated, and edited by the author, who made all core design and research decisions
and takes full responsibility for the software and the manuscript.

# Acknowledgements

This work received no specific external funding. The author declares no conflicts of interest.

# References
