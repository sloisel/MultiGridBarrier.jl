```@meta
CurrentModule = MultiGridBarrier
```

# GPU solves with CUDA

MultiGridBarrier solves on NVIDIA GPUs through the `MultiGridBarrierCUDAExt`
extension — and there is no GPU code path to opt into and no keyword to
remember. Load the two CUDA packages alongside MultiGridBarrier, and every
solve runs on the GPU whenever a working device is present:

```julia
using MultiGridBarrier, CUDA, CUDSS_jll

sol = mgb_solve(assemble(amg(subdivide(fem3d(; k = 2), 3)); p = 1.5))  # solved on the GPU
plot(sol)                                  # sol is native CPU data, as always
```

At load time the extension checks `CUDA.functional()` and, if a working GPU is
found, silently makes `CUDADevice` the package-wide default device. The same
script runs unchanged on a machine without a GPU — the default simply stays
`CPUDevice`. Nothing is ever printed to the console; each solve records the
backend it actually used in its log, so `sol.log` above begins with
`mgb_solve: device = CUDADevice`.

!!! note "Requires CUDA and CUDSS_jll"
    Add both packages (`pkg> add CUDA CUDSS_jll`) and load them
    (`using CUDA, CUDSS_jll`) before or alongside MultiGridBarrier. An NVIDIA
    GPU and driver are required at run time; the Newton linear systems are
    factored on the device by NVIDIA's
    [cuDSS](https://developer.nvidia.com/cudss) direct sparse solver, which
    ships as the `CUDSS_jll` binary artifact.

## Overriding the autodetection

The `device` keyword of `mgb_solve` overrides the autodetected default for
that solve:

```julia
mgb_solve(prob; device = CPUDevice)    # force this solve onto the CPU
mgb_solve(prob; device = CUDADevice)   # force this solve onto the GPU
```

## How it works

Assembly always happens on the CPU: `assemble` lowers every problem closure to
per-node grids, so an `MGBProblem` is pure data. `mgb_solve` moves that data to
the device ([`native_to_device`](@ref)), runs the entire interior-point
iteration there — the structured `BlockDiag` operators keep the batched-GEMM
Hessian assembly on the GPU, and the Newton systems are factored by cuDSS —
and moves the result back ([`device_to_native`](@ref)), so the returned
`MGBSOL` is always native CPU data regardless of the device, and plotting and
post-processing work unchanged. Both `Float64` and `Float32` are supported,
and the two backends produce matching solutions: the test suite solves the
same assembled problems on CPU and GPU across the FEM and spectral
discretizations and compares the results to `1e-8`.

## API reference

```@docs
Device
CPUDevice
CUDADevice
native_to_device
device_to_native
```
