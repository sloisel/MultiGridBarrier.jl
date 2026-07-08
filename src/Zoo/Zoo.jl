"""
    Zoo

A library of convex variational test problems. Each constructor takes a `MultiGrid`
and returns an assembled, closure-free [`MGBProblem`](@ref); solve it with
`mgb_solve(problem; kwargs...)`. Problem-defining parameters (`p`, forcing `f`,
boundary data `g_u`, …) are keyword arguments of the constructor; solver-control
parameters (`tol`, `device`, …) are passed to `mgb_solve`.

# GPU support
Solve on a GPU by loading the CUDA extension (`using CUDA, CUDSS_jll`) and passing
`device = CUDADevice` to `mgb_solve`, e.g.
`mgb_solve(Zoo.p_harmonic(mg); device = CUDADevice)`. The problem is assembled on the
CPU and moved to the device (and the solution back) by `mgb_solve`.
"""
module Zoo

using ..MultiGridBarrier
using StaticArrays
import ..MultiGridBarrier: MultiGrid, default_D, default_idx, amg_dim,
        convex_Euclidian_power, convex_linear, intersect, assemble

export elastoplastic_torsion, minimal_surface, p_harmonic, norton_hoff,
        rof, two_sided_obstacle

# Spatial dimension d, extracted from a MultiGrid.
_dim(mg) = amg_dim(mg.geometry.discretization)

# Shared setup for the vector-valued problems (p_harmonic, norton_hoff):
# state (u_1, …, u_d, s); the D table with, per component, an :id row plus d
# partials, and a trailing s:id row; the linear functional (f_i on each
# u_i:id row, 1 on the slack, 0 on partials); the Dirichlet/start closure;
# and idx = the y-positions of the d² partials plus the slack.
function _vector_state_setup(::Type{T}, d::Int, f::Function, g_u::Function,
                             s_init::Real) where {T}
    state_variables = vcat([[Symbol("u$i") :dirichlet] for i in 1:d]..., [:s :full])
    op_syms = (:dx, :dy, :dz)
    rows = Matrix{Symbol}[]
    for i in 1:d
        push!(rows, [Symbol("u$i") :id])
        for j in 1:d
            push!(rows, [Symbol("u$i") op_syms[j]])
        end
    end
    push!(rows, [:s :id])
    D = vcat(rows...)
    nrows = size(D, 1)                  # = d*(1 + d) + 1

    f_kw = let f0 = f, d = d
        x -> begin
            fv = f0(x)
            SVector{nrows,T}(ntuple(k -> begin
                if k == nrows
                    one(T)
                else
                    pos = k - 1
                    i = pos ÷ (d + 1) + 1
                    off = pos - (i - 1) * (d + 1)
                    (1 <= i <= d && off == 0) ? T(fv[i]) : zero(T)
                end
            end, Val(nrows)))
        end
    end
    g_kw = let gu = g_u, d = d
        x -> begin
            gv = gu(x)
            SVector{d + 1,T}(ntuple(k -> k <= d ? T(gv[k]) : T(s_init), Val(d + 1)))
        end
    end

    # ∂u_i/∂x_j sits at y-position (i-1)(d+1)+1+j; the slack at nrows.
    nz = d * d + 1
    partial_positions = Int[]
    for i in 1:d, j in 1:d
        push!(partial_positions, (i - 1) * (d + 1) + 1 + j)
    end
    push!(partial_positions, nrows)
    idx = SVector{nz,Int}(partial_positions)

    return (; state_variables, D, f_kw, g_kw, idx, nz, nrows)
end

# Shared linear functional / boundary closures for the scalar problems
# (elastoplastic_torsion, two_sided_obstacle): f on the u:id row, 1/2 on the
# slack row, 0 on the partials; g = (g_u, s_init).
function _scalar_fg(::Type{T}, nrows::Int, f::Function, g_u::Function,
                    s_init::Real) where {T}
    f_kw = let f0 = f
        x -> SVector{nrows,T}(ntuple(i -> i == 1 ? T(f0(x)) :
                                          i == nrows ? T(0.5) : T(0), Val(nrows)))
    end
    g_kw = let gu = g_u
        x -> SVector{2,T}(T(gu(x)), T(s_init))
    end
    return f_kw, g_kw
end

include("elastoplastic_torsion.jl")
include("minimal_surface.jl")
include("p_harmonic.jl")
include("norton_hoff.jl")
include("rof.jl")
include("two_sided_obstacle.jl")

end  # module Zoo
