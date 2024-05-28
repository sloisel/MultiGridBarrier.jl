# MultiGridBarrier

# Author: Sébastien Loisel

[![Build Status](https://github.com/sloisel/MultiGridBarrier.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/MultiGridBarrier.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/MultiGridBarrier.jl/dev/)

This package solves convex variational problems, e.g. nonlinear PDEs and BVPs, using the MultiGrid Barrier method (with either finite elements or spectral elements), which is theoretically optimal for some problem classes.

# Finite elements

A 1d example for solving the p-Laplacian:
```
julia> using MultiGridBarrier; fem_solve_1d(Float64,L=3,p=1.0);
```
This uses piecewise linear finite elements.

A 2d example for solving the p-Laplacian:
```
julia> using MultiGridBarrier; fem_solve_2d(Float64,L=3,p=1.0);
```
This uses piecewise quadratic elements on triangles. As is typical, a "bubble" function is added to the function space of each triangle, so that the quadrature weights are all positive.

# Spectral Barrier Method

To solve a p-Laplace equation in 1d, do:

```
julia> using MultiGridBarrier; SOL=spectral_solve_1d(Float64,n=80,p=1.1);
```

To solve a p-Laplace equation in 2d, do:

```
julia> using MultiGridBarrier; SOL=spectral_solve_2d(Float64,n=8,p=1.0);
```

To see some more examples, look at the documentation or source code for `spectral_solve_1d` and `spectral_solve_2d`.

