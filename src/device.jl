# device.jl -- Device markers and the native<->device transfer interface.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.
#
# The solver core is backend-generic: it dispatches on the array types carried by
# the problem data, never on CUDA directly. A `Device` selects which backend the
# problem is moved to before solving. `CPUDevice` is the identity; `CUDADevice` is
# supplied by the MultiGridBarrierCUDAExt extension (load `CUDA` and `CUDSS_jll`).
# Because every closure (f, g, A, b, p, select) is lowered to a per-vertex grid
# during CPU assembly, the device transfer is pure, dumb data movement.

"""
    abstract type Device

Marker selecting the compute backend for `mgb_solve`. Concrete devices are
`CPUDevice` (always available) and `CUDADevice` (requires the CUDA extension:
`using CUDA, CUDSS_jll`).
"""
abstract type Device end

"""    CPUDevice <: Device

The native CPU backend; `native_to_device`/`device_to_native` are the identity.
"""
struct CPUDevice <: Device end

"""    CUDADevice <: Device

The CUDA GPU backend. Requires the `MultiGridBarrierCUDAExt` extension, enabled by
`using CUDA, CUDSS_jll` before (or alongside) `using MultiGridBarrier`.
"""
struct CUDADevice <: Device end

"""
    native_to_device(D::Type{<:Device}, x)

Move `x` from native (CPU) types onto device `D`. `CPUDevice` is the identity; the CUDA
extension supplies the `CUDADevice` method, delegating to `native_to_cuda`. The inverse is
[`device_to_native`](@ref).
"""
function native_to_device end

"""
    device_to_native(D::Type{<:Device}, x)

Move `x` from device `D` back to native (CPU) types. `CPUDevice` is the identity; the CUDA
extension supplies the `CUDADevice` method, delegating to `cuda_to_native`. The inverse is
[`native_to_device`](@ref).
"""
function device_to_native end

# CPU is the identity.
native_to_device(::Type{CPUDevice}, x) = x
device_to_native(::Type{CPUDevice}, x) = x

# Friendly error for any device whose extension is not loaded (less specific than
# the concrete-device methods the extension adds, so it never shadows them).
native_to_device(::Type{D}, x) where {D<:Device} = error(
    "native_to_device: device $D is unavailable. For CUDADevice, run `using CUDA, CUDSS_jll`.")
device_to_native(::Type{D}, x) where {D<:Device} = error(
    "device_to_native: device $D is unavailable. For CUDADevice, run `using CUDA, CUDSS_jll`.")

# Default device used when `mgb_solve` is called without `device=...`. The CUDA
# extension's `__init__` flips this to `CUDADevice` when `CUDA.functional()`.
const _DEFAULT_DEVICE = Ref{Type{<:Device}}(CPUDevice)

"""
    default_device() -> Type{<:Device}

The device `mgb_solve` uses when `device` is not given. `CPUDevice`, unless the
CUDA extension is loaded and `CUDA.functional()`, in which case `CUDADevice`.
Pass `device=CPUDevice`/`device=CUDADevice` to `mgb_solve` to override per call.
"""
default_device() = _DEFAULT_DEVICE[]

"""
    default_device!(D::Type{<:Device}) -> Type{<:Device}

Set the device `mgb_solve` uses when `device` is not given, and return it. Use this to
force a backend regardless of what the CUDA extension auto-selected, e.g.
`default_device!(CPUDevice)` to keep a test suite on the CPU even when a GPU is present.
A per-call `device=` keyword still overrides this default.
"""
default_device!(D::Type{<:Device}) = (_DEFAULT_DEVICE[] = D)

# Device-dispatched cache cleanup, used by the throw path of `mgb_solve`: when
# the solve throws there is no MGBSOL to dispatch `mgb_cleanup(sol)` on, but
# the backend caches must still be flushed — otherwise assembly plans and
# factorizations (and, with the identity-keyed plan caches, the R matrices
# keying them) stay resident until the next successful solve on that backend.
# The fallback is a no-op; `BlockMatrices.jl` adds the `CPUDevice` method and
# the CUDA extension adds the `CUDADevice` method.
mgb_cleanup(::Type{<:Device}) = nothing
