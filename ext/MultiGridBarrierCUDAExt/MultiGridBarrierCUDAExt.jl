module MultiGridBarrierCUDAExt

using CUDA
using CUDA.CUSPARSE
using CUDSS_jll
using LinearAlgebra
using SparseArrays
using StaticArrays
using MultiGridBarrier

include("sparse_ops.jl")
include("cudss_solver.jl")
include("map_rows_gpu.jl")
include("block_types.jl")
include("block_ops.jl")
include("mgb_interface.jl")
include("conversion.jl")

# Make CUDADevice the default for mgb_solve when a working GPU is present. Users
# override per call with `device=CPUDevice`. Gated on `CUDA.functional()` so that
# loading the extension on a GPU-less machine still defaults to the CPU.
# Deliberately silent: a load-time @info lands on stderr through a logger the
# user would have to reconfigure *before* `using` to suppress. The device each
# solve actually uses is recorded in its log instead (`mgb_solve: device = ...`
# in `sol.log`).
function __init__()
    if CUDA.functional()
        MultiGridBarrier._DEFAULT_DEVICE[] = MultiGridBarrier.CUDADevice
    end
end

end
