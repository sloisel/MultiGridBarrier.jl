# Logging, interpolation, mesh helpers, and the convergence-failure exception.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

@doc raw"""
    Log(x::T) where {T} = x<=0 ? T(-Inf) : Base.log(x)     # "Convex programmer's log"
"""
Log(x::T) where {T} = x<=0 ? T(-Inf) : Base.log(x)
const log = Log

@doc raw"""
    interpolate(M::Geometry, z::Vector, t)

Interpolate a solution vector at specified points.

Given a solution `z` on the mesh `M`, evaluates the solution at new points `t`
using the appropriate interpolation method for the discretization.

Supported discretizations
- 1D FEM (`FEM1D`): exact per-element degree-`k` Lagrange (Q_k) interpolation
  (piecewise linear at `k = 1`); evaluation points outside the mesh are clamped
  to the boundary values
- 1D spectral (`SPECTRAL1D`): spectral polynomial interpolation
- 2D spectral (`SPECTRAL2D`): tensor-product spectral interpolation

Note: 2D/3D FEM interpolation is not currently provided.

# Arguments
- `M::Geometry`: The geometry containing grid and basis information
- `z::Vector`: Solution vector on the finest grid (length must match number of DOFs)
- `t`: Evaluation points. Format depends on dimension:
  - 1D: scalar or `Vector{T}` of x-coordinates
  - 2D spectral: `Matrix{T}` where each row is `[x, y]`

# Returns
Interpolated values at the specified points. Shape matches input `t`.

# Examples
```julia
# 1D interpolation (FEM)
geom = subdivide(fem1d(; nodes=collect(range(-1.0, 1.0, length=3))), 3)
z = sin.(π .* vec(geom.x))
y = interpolate(geom, z, 0.5)
y_vec = interpolate(geom, z, [-0.5, 0.0, 0.5])

# 2D interpolation (spectral)
geom = spectral2d(n=4)
xf = reshape(geom.x, :, size(geom.x, 3))   # flat (n_nodes, 2) view
z = exp.(-xf[:,1].^2 .- xf[:,2].^2)
points = [0.0 0.0; 0.5 0.5; -0.5 0.5]
vals = interpolate(geom, z, points)
```
""" interpolate

# The `plot` methods (and their docstring) live in MultiGridBarrierPyPlotExt:
# they extend `PyPlot.plot`, so `using MultiGridBarrier, PyPlot` provides them.

"""
    MGB3DFigure

A rendered 3D plot returned by the FEM3D `plot` methods of the
`MultiGridBarrierPyPlotExt` extension (load PyPlot to enable them); the `png`
field holds the PNG bytes. Displays inline as `image/png` (Jupyter, Documenter,
…); write it to a file with `savefig(fig, "out.png")`.
"""
struct MGB3DFigure
    png::Vector{UInt8}
end

function Base.show(io::IO, ::MIME"image/png", fig::MGB3DFigure)
    write(io, fig.png)
end

mgb_zeros(::Matrix{T}, m,n) where {T} = zeros(T,m,n)
mgb_zeros(::Type{Vector{T}}, m) where {T} = zeros(T, m)
mgb_all_isfinite(z::Vector{T}) where {T} = all(isfinite, z)

mgb_diag(::Matrix{T}, z::Vector{T},m=length(z),n=length(z)) where {T} = diagm(m,n,0=>z)

mgb_blockdiag(args::SparseMatrixCSC{T,Int}...) where {T} = blockdiag(args...)
mgb_blockdiag(args::Matrix{T}...) where {T} = Matrix{T}(blockdiag((sparse(args[k]) for k=1:length(args))...))

mgb_cleanup(sol) = sol

# Convert matrix rows to Vector{SVector} via transpose + reinterpret
# copy(transpose(M)) materializes to contiguous memory (required for reinterpret)
# and preserves the array type (stays on GPU for GPU arrays)
# vec() ensures result is 1D (needed when K=1 since SVector{1} has same size as element)
# Vectors pass through unchanged (no SVector wrapping)
_rows_to_svectors(::Val{K}, M::AbstractMatrix{T}) where {K, T} =
    vec(reinterpret(reshape, SVector{K, T}, copy(transpose(M))))
_rows_to_svectors(M::AbstractMatrix{T}) where {T} = _rows_to_svectors(Val(size(M, 2)), M)
_rows_to_svectors(v::AbstractVector) = v  # vectors pass through unchanged

# Convert output: Vector{SVector} -> Matrix, Vector{scalar} -> Vector
# Use copy(transpose(...)) to preserve array type (stays on GPU for GPU arrays)
# K = 1: `reinterpret(reshape, T, v)` drops the singleton dimension and returns a
# 1D `Vector{T}`; `transpose` of a vector is a 1×N row, which then makes
# `coarsen[l] * f_grid[l+1]` fail with a DimensionMismatch. Force a 1×N 2D view
# before the transpose so the result is the expected N×1 Matrix.
_svectors_to_rows(v::AbstractVector{SVector{1,T}}) where {T} =
    copy(transpose(reshape(reinterpret(reshape, T, v), 1, :)))
_svectors_to_rows(v::AbstractVector{SVector{K,T}}) where {K,T} =
    copy(transpose(reinterpret(reshape, T, v)))
_svectors_to_rows(v::AbstractVector{<:Number}) = v

"""
    map_rows(f, A::AbstractMatrix...) -> AbstractMatrix or AbstractVector

Apply `f` row-wise to matrices. Each row of each argument matrix is converted to
an `SVector`, `f` is broadcast over rows, and results are collected back into
a matrix (if `f` returns vectors) or a vector (if `f` returns scalars).

Used to evaluate functions at mesh nodes, e.g. `map_rows(g, geometry.x)` evaluates
`g` at each node.
"""
function map_rows(f, A...)
    processed = map(_rows_to_svectors, A)
    results = f.(processed...)
    _svectors_to_rows(results)
end

# GPU-compatible code path (for non-MPI, just calls map_rows)
# Defined as a function (not const alias) so it can be specialized in MultiGridBarrierMPI
map_rows_gpu(f, args...) = map_rows(f, args...)

"""
    _to_cpu_array(x)

Convert array to CPU if it's a GPU array, otherwise return unchanged.
Barrier functions need CPU arrays for scalar indexing (A[j,:]).
For non-GPU arrays, this is a no-op.
MultiGridBarrierMPI specializes this to handle GPU arrays.
"""
_to_cpu_array(x) = x

symmetric(A) = Symmetric(A)

# Type-stable linear solver
solve(A, b) = A \ b

macro debug(args...)
    escargs = map(esc, args)
    return :($(esc(:printlog))(nameof($(esc(:(var"#self#")))), ":", $(escargs...)))
end

"""
    MGBConvergenceFailure <: Exception

Thrown when the MGB solver fails to converge (feasibility or main phase).
Carries a descriptive `message` and a machine-readable `code`:

- `:infeasible`: the feasibility subproblem converged to a minimizer with
  positive constraint violation strictly inside the bounding box, certifying
  that the problem is infeasible (see `mgb_solve`, section "Feasibility phase").
- `:feasibility_Rmax`: no strictly feasible point was found with nodal values
  bounded by `feasibility_Rmax`; the problem is infeasible, or its feasible
  points have values exceeding the cap.
- `:stall`: the `t`-ramp of the barrier method stopped making progress before
  reaching the target tolerance (the step refinement collapsed).
- `:iteration_limit`: the `t`-ramp hit the `maxit` iteration cap.
- `:failure`: any other convergence failure (the default of the one-argument
  constructor).

Front ends can dispatch on `code`; the JuMP extension maps it to the
corresponding `MOI.TerminationStatusCode`.
"""
struct MGBConvergenceFailure <: Exception
    message::String
    code::Symbol
end
MGBConvergenceFailure(message::String) = MGBConvergenceFailure(message, :failure)

Base.showerror(io::IO, e::MGBConvergenceFailure) = print(io, "MGBConvergenceFailure:\n", e.message)

