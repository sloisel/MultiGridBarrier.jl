"""
    module Mesh3d

A submodule of MultiGridBarrier providing 3D hexahedral finite element discretization.

Exports the `FEM3D` discretization descriptor and 3D plotting helpers.
"""
module Mesh3d

using LinearAlgebra
using SparseArrays
using ..MultiGridBarrier: AMGBSOL, ParabolicSOL, mgb_solve, parabolic_solve, HTML5anim,
    BlockDiag, VBlockDiag, HBlockDiag, MultiGrid, amg, geometric_mg
import ..MultiGridBarrier: Geometry, default_f, default_g, default_D, default_D_parabolic,
    default_f_parabolic, default_g_parabolic, amg_dim,
    _default_block_size

include("MeshGen.jl")
include("ReferenceElement.jl")
include("Geometry.jl")
include("Operators.jl")
include("Plotting.jl")
using .Plotting

export FEM3D, plot, savefig, _geometric_fem3d_mg, parabolic_solve, HTML5anim

_default_block_size(d::FEM3D) = (d.k + 1)^3

# Extend defaults for 3D (static solver)
default_f(::Type{T}, ::Val{3}) where {T} = (x)->T[0.5, 0.0, 0.0, 0.0, 1.0]
default_g(::Type{T}, ::Val{3}) where {T} = (x)->T[x[1]^2 + x[2]^2 + x[3]^2, 100.0]
# Note: default_D(::Val{3}) is defined in AlgebraicMultiGridBarrier.jl

# Extend defaults for 3D (parabolic solver)
default_D_parabolic(::Val{3}) = [:u  :id
     :u  :dx
     :u  :dy
     :u  :dz
     :s1 :id
     :s2 :id]
default_f_parabolic(::Val{3}) = (f1,w1,w2)->[f1,0,0,0,w1,w2]
default_g_parabolic(::Val{3}) = (t,x)->[x[1]^2+x[2]^2+x[3]^2,0,0]

end # module
