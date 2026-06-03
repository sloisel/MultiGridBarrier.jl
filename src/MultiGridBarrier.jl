@doc raw"""
    module MultiGridBarrier

MultiGridBarrier solves nonlinear convex optimization problems in function spaces using a
barrier (interior-point) method accelerated by a multigrid hierarchy. The package exposes:

- Single-level mesh constructors: `fem1d`, `fem2d`, `fem2d_P1`, `fem2d_P2`, `fem3d`, `spectral1d`,
  `spectral2d`. Each returns a `Geometry`.
- Topological connectivity for slit domains / glued manifolds: `tensor_dofmap` builds
  full-node connectivity from corner connectivity (no coordinates); pass the result as the
  `t=` keyword of `fem1d`/`fem2d`/`fem3d` so geometrically-coincident nodes stay distinct.
- The hierarchy builder: `amg(geom)` wraps a `Geometry` and returns a `MultiGrid`
  (algebraic-multigrid hierarchy on the fine mesh).
- A mesh-refinement utility: `subdivide(geom, L)` returns a refined `Geometry`. Compose
  with AMG via `amg(subdivide(geom, L))`.
- A legacy geometric-subdivision hierarchy: `geometric_mg(geom, L)`. Still available for
  callers that specifically want geometric transfers; new code should prefer `amg`.
- Problem assembly: `assemble(mg::MultiGrid; kwargs...) -> MGBProblem`.
- The main solver: `mgb_solve(prob::MGBProblem; kwargs...) -> MGBSOL`.
- A time-dependent solver: `parabolic_solve(mg::MultiGrid; kwargs...)`.

## A gentle introduction via the p-Laplacian
For a domain Ω ⊂ ℝᵈ and p ≥ 1, consider the variational problem
```math
\min_{u} \; J(u) = \int_{\Omega} \|\nabla u\|_2^p + f\,u \, dx
```
subject to appropriate boundary conditions (e.g., homogeneous Dirichlet). The Euler–Lagrange
equation gives the p-Laplace PDE.

## Typical workflow
```julia
geom = fem2d_P2()                       # single-level mesh
mg   = amg(geom)                        # build AMG hierarchy
prob = assemble(mg; p = 1.5)            # assemble the (native) MGBProblem
sol  = mgb_solve(prob)
plot(sol)
```

## CUDA GPU acceleration (optional extension)
CUDA support is provided as a Julia package extension (`MultiGridBarrierCUDAExt`). Load
`CUDA` and `CUDSS_jll` to enable it, then select the GPU with `device = CUDADevice`:
```julia
using CUDA, CUDSS_jll
using MultiGridBarrier
sol = mgb_solve(assemble(amg(fem2d_P2()); p = 1.5); device = CUDADevice)
```
`mgb_solve` moves the assembled problem to the GPU, solves there, and moves the solution
back, so the returned `MGBSOL` is always in native CPU types. When a functional GPU is
present the default device becomes `CUDADevice`; pass `device = CPUDevice` to force the CPU.
The lower-level `native_to_cuda` / `cuda_to_native` converters remain available.

## See also
- Mesh constructors: `fem1d`, `fem2d`, `fem2d_P1`, `fem2d_P2`, `fem3d`, `spectral1d`, `spectral2d`;
  `tensor_dofmap` (+ the `t=` keyword) for slit-domain / glued-manifold connectivity.
- Hierarchy: `amg`, `subdivide` (and the legacy `geometric_mg`).
- Solvers: `mgb_solve`, `parabolic_solve`.
- Convex: `convex_Euclidian_power`, `convex_linear`, `convex_piecewise`, `intersect`.
- Visualization & sampling: `plot`, `interpolate`.
- Backend selection: `device` kwarg of `mgb_solve`, `CPUDevice`, `CUDADevice`,
  `native_to_device`, `device_to_native`, `default_device`.
- Problem assembly: `assemble`, `MGBProblem`.
- CUDA (lower level): `native_to_cuda`, `cuda_to_native`, `clear_cudss_cache!`.
"""
module MultiGridBarrier

using SparseArrays
using LinearAlgebra
using StaticArrays
using PyPlot
import PyPlot: plot
using PyCall
using ProgressMeter
using QuadratureRules
using PrecompileTools
using AlgebraicMultigrid
import Base: intersect

include("AlgebraicMultiGridBarrier.jl")
include("BlockMatrices.jl")
include("spectral1d.jl")
include("spectral2d.jl")
include("Parabolic.jl")

# Single-level mesh constructors. `amg(geom)` attaches the AMG hierarchy (preferred);
# `geometric_mg(geom, L)` is the legacy geometric-subdivision alternative.
# `fem1d`/`fem2d`/`fem3d` are the dimension-generic tensor-product Q_k elements
# (TensorFEM.jl); `fem2d_P1`/`fem2d_P2` are the simplicial P_k family.
include("amg_prolongators.jl")
include("fem2d_P2.jl")
include("fem2d_P1.jl")
include("TensorFEM.jl")
include("plot3d.jl")          # 3D (FEM3D = TensorFEM{3}) PyVista plotting + animation
export FEM2D_P1, FEM2D_P2
export amg_ruge_stuben, amg_smoothed_aggregation, amg_pyamg
export plot, savefig, HTML5anim, MGB3DFigure

# CUDA extension stubs -- methods added by MultiGridBarrierCUDAExt

"""
    native_to_cuda(geom::Geometry) -> Geometry
    native_to_cuda(mg::MultiGrid) -> MultiGrid

Convert a native (CPU) `Geometry` or `MultiGrid` to CUDA GPU types, faithfully
preserving matrix types: `BlockDiag` operators stay block (driving the structured
batched-GEMM Hessian assembly), sparse matrices become `CuSparseMatrixCSR`. FEM
geometries always carry `BlockDiag` operators, so the solve is structured; only the
spectral discretizations use dense operators. Requires `using CUDA, CUDSS_jll`.
"""
function native_to_cuda end

"""
    cuda_to_native(x)

Convert a CUDA `Geometry`, `MultiGrid`, or `MGBSOL` back to native CPU types.
"""
function cuda_to_native end

"""
    clear_cudss_cache!()

Destroy all cached cuDSS factorizations and free associated GPU memory.
"""
function clear_cudss_cache! end

export native_to_cuda, cuda_to_native, clear_cudss_cache!

# Zoo: a small library of convex variational test problems.
include("Zoo/Zoo.jl")
using .Zoo
export Zoo

function amg_precompile()
    mgb_solve(assemble(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3))))); verbose=false, tol=0.1)
    mgb_solve(assemble(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3)))));
              line_search=linesearch_illinois(Float64), verbose=false, tol=0.1)
    mgb_solve(assemble(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3)))));
              line_search=linesearch_illinois(Float64), stopping_criterion=stopping_exact(0.1),
              finalize=false, verbose=false, tol=0.1)
    mgb_solve(assemble(amg(fem2d_P2())); verbose=false, tol=0.1)
    mgb_solve(assemble(amg(spectral1d(n=2))); verbose=false, tol=0.1)
    mgb_solve(assemble(amg(spectral2d(n=2))); verbose=false, tol=0.1)
    # Sparse solves in Float32 broken in tested Julia versions.
    mgb_solve(assemble(amg(spectral1d(Float32; n=4)); p=1.0f0); verbose=false)
    mgb_solve(assemble(amg(spectral2d(Float32; n=4)); p=1.0f0); verbose=false)
    mgb_solve(assemble(amg(fem3d(; k=1))); verbose=false, tol=0.1)
end

function parabolic_precompile()
    parabolic_solve(amg(fem1d(; nodes=collect(range(-1.0, 1.0, length=3)))); h=0.5, verbose=false, tol=0.1)
    parabolic_solve(amg(fem2d_P2()); h=0.5, verbose=false, tol=0.1)
    parabolic_solve(amg(spectral1d(n=2)); h=0.5, verbose=false, tol=0.1)
    parabolic_solve(amg(spectral2d(n=2)); h=0.5, verbose=false, tol=0.1)
end

@compile_workload begin
    amg_precompile()
    parabolic_precompile()
end

end
