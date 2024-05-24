# MultiGridBarrier

# Author: Sébastien Loisel

This package solved convex variational problems, e.g. nonlinear PDEs and BVPs, using the MultiGrid Barrier method (and the finite element method), which is theoretically optimal for some problem classes.

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