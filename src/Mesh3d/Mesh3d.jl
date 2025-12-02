"""
    module Mesh3d

A submodule of MultiGridBarrier providing 3D hexahedral finite element discretization.

Exports:
- `FEM3D`: Discretization type for 3D hexahedral finite elements
- `fem3d`: Create a 3D FEM geometry
- `fem3d_solve`: Solve a 3D PDE using the Spectral Barrier Method
- `plot`, `savefig`: Plotting functions for 3D solutions
- `HTML5anim`: Animation type for time-dependent solutions
"""
module Mesh3d

using LinearAlgebra
using SparseArrays
using ..MultiGridBarrier: AMGBSOL, ParabolicSOL, amgb, parabolic_solve, HTML5anim
import ..MultiGridBarrier: Geometry, default_f, default_g, default_D, default_D_parabolic, default_f_parabolic, default_g_parabolic, amg_dim

include("MeshGen.jl")
include("ReferenceElement.jl")
include("Geometry.jl")
include("Operators.jl")
include("Plotting.jl")
using .Plotting

export FEM3D, plot, savefig, fem3d, fem3d_solve, parabolic_solve, HTML5anim

# Extend defaults for 3D (static solver)
default_f(::Type{T}, ::Val{3}) where {T} = (x)->T[0.5, 0.0, 0.0, 0.0, 1.0]
default_g(::Type{T}, ::Val{3}) where {T} = (x)->T[x[1]^2 + x[2]^2 + x[3]^2, 100.0]
default_D(::Val{3}) = [:u :id; :u :dx; :u :dy; :u :dz; :s :id]

# Extend defaults for 3D (parabolic solver)
default_D_parabolic(::Val{3}) = [:u  :id
     :u  :dx
     :u  :dy
     :u  :dz
     :s1 :id
     :s2 :id]
default_f_parabolic(::Val{3}) = (f1,w1,w2)->[f1,0,0,0,w1,w2]
default_g_parabolic(::Val{3}) = (t,x)->[x[1]^2+x[2]^2+x[3]^2,0,0]

"""
    fem3d_solve(::Type{T}=Float64; rest...) where {T}

Solve a 3D PDE using the Spectral Barrier Method.

# Arguments
- `T`: Floating-point type for computations (default `Float64`).
- `rest...`: Keyword arguments passed to `fem3d` (e.g., `L`, `k`) and `amgb` (e.g., `D`, `f`, `g`, `maxiter`, `verbose`).

# Returns
An `AMGBSOL` object containing the solution field `z` and convergence history.

See `amgb` for the full list of keyword arguments and their defaults.
"""
function fem3d_solve(::Type{T}=Float64; rest...) where {T}
    # Create geometry
    geo = fem3d(T; rest...)

    # Call amgb (uses default_D, default_f, default_g for dim=3)
    return amgb(geo; rest...)
end

end # module
