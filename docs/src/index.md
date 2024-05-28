```@meta
CurrentModule = MultiGridBarrier
```

# MultiGridBarrier

[MultiGridBarrier](https://github.com/sloisel/MultiGridBarrier.jl) is a Julia module for solving nonlinear convex optimization problems in function spaces, such as p-Laplace problems. When regularity conditions are satisfied, the solvers are quasi-optimal.

The `MultiGridBarrier` module features finite element and spectral discretizations in 1d and 2d.

## Finite elements

After installing `MultiGridBarrier` with the Julia package manager, in a Jupyter notebook, one solves a 1d p-Laplace problem as follows:
```@example 1
using PyPlot # hide
using MultiGridBarrier
fem_solve_1d(Float64,L=5,p=1.0,verbose=false);
savefig("fem1d.svg"); nothing # hide
close() #hide
```

![](fem1d.svg)

A 2d p-Laplace problem:
```@example 1
fem_solve_2d(Float64,L=3,p=1.0,verbose=false);
savefig("fem2d.svg"); nothing # hide
close() #hide
```

![](fem2d.svg)

## Spectral elements

Solve a 1d p-Laplace problem as follows:
```@example 1
spectral_solve_1d(Float64,n=40,p=1.0,verbose=false);
savefig("spectral1d.svg"); nothing # hide
close() #hide
```

![](spectral1d.svg)

A 2d p-Laplace problem:
```@example 1
spectral_solve_2d(Float64,n=5,p=1.5,verbose=false);
savefig("spectral2d.svg"); nothing # hide
close() #hide
```

![](spectral2d.svg)

# Module reference

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:module]
```

# Types reference

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:type]
```

# Functions reference

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:function]
```

# Index

```@index
```

