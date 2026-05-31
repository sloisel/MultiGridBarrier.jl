# Sets up a dedicated environment (this folder) that dev-links the local
# MultiGridBarrier package and adds the CUDA-path weakdeps, so we can iterate on
# the GPU extension without touching the package's own Manifest or pulling the
# whole plotting test suite. Run with: julia --project=tools/cudatest tools/cudatest/setup.jl
using Pkg
pkgroot = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop(path=pkgroot)
Pkg.add(["CUDA", "CUDSS_jll", "Test"])
Pkg.instantiate()
println("CUDATEST_SETUP_DONE")
