export mgb_solve, amg, geometric_mg, subdivide, MultiGrid,
       Geometry, Convex, convex_linear, convex_Euclidian_power, convex_piecewise,
       AMGBConvergenceFailure, linesearch_illinois, linesearch_backtracking,
       stopping_exact, stopping_inexact, interpolate, intersect, plot,
       multigrid_from_fine_grid, find_boundary

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
- 1D FEM (`FEM1D`): piecewise-linear interpolation
- 1D spectral (`SPECTRAL1D`): spectral polynomial interpolation
- 2D spectral (`SPECTRAL2D`): tensor-product spectral interpolation

Note: 2D FEM interpolation is not currently provided.

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
z = exp.(-geom.x[:,1].^2 .- geom.x[:,2].^2)
points = [0.0 0.0; 0.5 0.5; -0.5 0.5]
vals = interpolate(geom, z, points)
```
""" interpolate

@doc raw"""
    plot(sol::AMGBSOL, k::Int=1; kwargs...)
    plot(sol::ParabolicSOL, k::Int=1; kwargs...)
    plot(M::Geometry, z::Vector; kwargs...)
    plot(M::Geometry, ts::AbstractVector, U::Matrix; frame_time=..., embed_limit=..., printer=...)

Visualize solutions and time sequences on meshes.

- 1D problems: Line plot. For spectral methods, you can specify evaluation points with `x=-1:0.01:1`.
- 2D FEM: Triangulated surface plot using the mesh structure.
- 2D spectral: 3D surface plot. You can specify evaluation grids with `x=-1:0.01:1, y=-1:0.01:1`.

Time sequences (animation):
- Call `plot(M, ts, U; frame_time=1/30, printer=anim->nothing)` where `U` has columns as frames and `ts` are absolute times in seconds (non-uniform allowed).
- Or simply call `plot(sol)` where `sol` is a `ParabolicSOL` returned by `parabolic_solve` (uses `sol.ts`).
- Animation advances at a fixed frame rate given by `frame_time` (seconds per video frame). For irregular `ts`, each video frame shows the latest data frame with timestamp ≤ current video time.
- The `printer` callback receives the Matplotlib animation object; use it to display or save (e.g., `anim.save("out.mp4")`).
- `embed_limit` controls the maximum embedded HTML5 video size in megabytes.

When `sol` is a solution object returned by `mgb_solve`, `plot(sol,k)` plots the kth
component `sol.z[:, k]` using `sol.geometry`. `plot(sol)` uses the default k=1.

All other keyword arguments are passed to the underlying `PyPlot` functions.
""" plot

amgb_zeros(::SparseMatrixCSC{T,Int}, m,n) where {T} = spzeros(T,m,n)
amgb_zeros(::LinearAlgebra.Adjoint{T, SparseArrays.SparseMatrixCSC{T, Int64}},m,n) where {T} = spzeros(T,m,n)
amgb_zeros(::Matrix{T}, m,n) where {T} = zeros(T,m,n)
amgb_zeros(::LinearAlgebra.Adjoint{T, Matrix{T}},m,n) where {T} = zeros(T,m,n)
amgb_zeros(::Type{Vector{T}}, m) where {T} = zeros(T, m)
amgb_all_isfinite(z::Vector{T}) where {T} = all(isfinite.(z))

amgb_diag(::SparseMatrixCSC{T,Int}, z::Vector{T},m=length(z),n=length(z)) where {T} = spdiagm(m,n,0=>z)
amgb_diag(::Matrix{T}, z::Vector{T},m=length(z),n=length(z)) where {T} = diagm(m,n,0=>z)

amgb_blockdiag(args::SparseMatrixCSC{T,Int}...) where {T} = blockdiag(args...)
amgb_blockdiag(args::Matrix{T}...) where {T} = Matrix{T}(blockdiag((sparse(args[k]) for k=1:length(args))...))

mgb_cleanup(sol) = sol

# makes a matrix (if f returns adjoint vectors) or a vector (if f returns scalars)
_maybevec(x::AbstractArray) = vec(x)
_maybevec(x) = x

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
    _raw_array(x)

Extract the raw array from an MPI wrapper, or return x unchanged for non-MPI types.
This is needed for GPU kernels: barriers must capture only isbits-compatible arrays
(like MtlVector/MtlMatrix), not MPI wrappers that contain non-isbits partition data.

For non-MPI types, returns `x` unchanged.
MultiGridBarrierMPI specializes this to extract `.v` from VectorMPI and `.A` from MatrixMPI.
"""
_raw_array(x) = x

"""
    _to_cpu_array(x)

Convert array to CPU if it's a GPU array, otherwise return unchanged.
Barrier functions need CPU arrays for scalar indexing (A[j,:]).
For non-GPU arrays, this is a no-op.
MultiGridBarrierMPI specializes this to handle GPU arrays.
"""
_to_cpu_array(x) = x

"""
    vertex_indices(A::AbstractMatrix)

Create a vector of vertex indices (1:n) for use with barrier functions.
For non-MPI arrays, returns a simple Vector{Int}.
"""
function vertex_indices(A::AbstractMatrix)
    return collect(1:size(A, 1))
end

function vertex_indices(A::AbstractVector)
    return collect(1:length(A))
end

symmetric(A) = Symmetric(A)

# Type-stable linear solver
solve(A, b) = A \ b

macro debug(args...)
    escargs = map(esc, args)
    return :($(esc(:printlog))(nameof($(esc(:(var"#self#")))), ":", $(escargs...)))
end

"""
    AMGBConvergenceFailure <: Exception

Thrown when the AMGB solver fails to converge (feasibility or main phase).
Includes a descriptive message about the failure.
"""
struct AMGBConvergenceFailure <: Exception
    message
end

Base.showerror(io::IO, e::AMGBConvergenceFailure) = print(io, "AMGBConvergenceFailure:\n", e.message)

@kwdef struct Barrier
    f0::Function
    f1::Function
    f2::Function
end

"""
    Geometry{T,X,W,M_op,M_sub,Discretization}

Single-level container for discretization geometry. Holds only the fine-level mesh and
operators — no multigrid hierarchy. Use `amg(geom)` to attach an algebraic-multigrid
hierarchy and return a `MultiGrid`. (The legacy `geometric_mg(geom, L)` builds a
geometric-subdivision hierarchy instead; new code should prefer `amg`.)

Type parameters
- `T`: scalar numeric type (e.g. `Float64`)
- `X`: type of the point storage `x` (typically `Matrix{T}`)
- `W`: type of the weight storage `w` (typically `Vector{T}`)
- `M_op`: matrix type for operators (e.g. `SparseMatrixCSC{T,Int}`, `BlockDiag{T}`)
- `M_sub`: matrix type for subspace embeddings (e.g. `SparseMatrixCSC{T,Int}`)
- `Discretization`: front-end descriptor (e.g. `FEM1D{T}`, `FEM2D_P2{T}`, `SPECTRAL1D{T}`)

Fields
- `discretization::Discretization`: discretization descriptor encoding dimension and grid info.
- `x::X`: mesh/sample points; size `(n_nodes, dim)`.
- `w::W`: quadrature weights matching `x` (length `n_nodes`).
- `subspaces::Dict{Symbol,M_sub}`: fine-level selection/embedding matrices (one per symbol).
- `operators::Dict{Symbol,M_op}`: fine-level discrete operators (e.g. `:id`, `:dx`, `:dy`).
"""
struct Geometry{T,X,W,M_op,M_sub,Discretization}
    discretization::Discretization
    x::X
    w::W
    subspaces::Dict{Symbol,M_sub}
    operators::Dict{Symbol,M_op}
end

"""
    MultiGrid{T,M_sub,M_ref,M_coar,G<:Geometry{T}}

A `Geometry` plus a multigrid hierarchy with **per-subspace** transfer operators.
Each state-variable subspace symbol (`:dirichlet`, `:full`, `:uniform`) gets its
own AMG hierarchy; the per-level subspace embedding matrices and the
level → level+1 transfer matrices are all keyed by the subspace symbol.

Fields
- `geometry::G`: the fine-level `Geometry`.
- `subspaces::Dict{Symbol,Vector{M_sub}}`: `subspaces[X][l]` is the embedding of
  level-`l`-`X`-subspace coefficients in the level-`l` broken basis.
- `refine::Dict{Symbol,Vector{M_ref}}`: `refine[X][l]` is the level-`l` →
  level-`l+1` prolongation in subspace `X`. `refine[X][end]` is identity.
- `coarsen::Dict{Symbol,Vector{M_coar}}`: matching restriction. `coarsen[X][end]`
  is identity.

A two-argument constructor `MultiGrid(geometry, subspaces, refine, coarsen)`
accepts plain `Vector` inputs for `refine`/`coarsen` and wraps them into Dicts
that map every key of `subspaces` to the same Vector — preserving the
single-shared-hierarchy semantics of the pre-refactor `MultiGrid` for callers
that aren't yet aware of per-subspace transfers.

Property forwarding: `mg.x`, `mg.w`, `mg.operators`, `mg.discretization` return
the corresponding fields of `mg.geometry`.
"""
struct MultiGrid{T,M_sub,M_ref,M_coar,G<:Geometry{T}}
    geometry::G
    subspaces::Dict{Symbol,Vector{M_sub}}
    refine::Dict{Symbol,Vector{M_ref}}
    coarsen::Dict{Symbol,Vector{M_coar}}

    function MultiGrid(geometry::G,
                       subspaces::Dict{Symbol,Vector{M_sub}},
                       refine::Dict{Symbol,Vector{M_ref}},
                       coarsen::Dict{Symbol,Vector{M_coar}}) where {T,M_sub,M_ref,M_coar,G<:Geometry{T}}
        refine_s, coarsen_s, subspaces_s =
            _stretch_per_subspace(T, refine, coarsen, subspaces)
        subspaces_s, refine_s, coarsen_s =
            _normalize_uniform_subspace(T, subspaces_s, refine_s, coarsen_s)
        new{T,M_sub,M_ref,M_coar,G}(geometry, subspaces_s, refine_s, coarsen_s)
    end
end

# Stretch each subspace's hierarchy to the common depth L_max = max(L_X) via
# ceil-interpolation. Subspace X with natural depth L_X < L_max gets its
# synthetic level i mapped to natural level n_i = ceil(L_X*i/L_max); transitions
# with n_{i+1} == n_i are identities (no-op refinement) at the level-l-X-broken-
# basis row dim; transitions with n_{i+1} > n_i reuse X's natural AMG step.
# Returns the stretched (refine, coarsen, subspaces) dicts. If every subspace
# already has depth L_max, the originals are returned unchanged.
function _stretch_per_subspace(::Type{T},
        refine::Dict{Symbol,Vector{M_ref}},
        coarsen::Dict{Symbol,Vector{M_coar}},
        subspaces::Dict{Symbol,Vector{M_sub}}) where {T,M_ref,M_coar,M_sub}
    L_X = Dict(X => length(refine[X]) for X in keys(refine))
    L_max = maximum(values(L_X))
    if all(==(L_max), values(L_X))
        return refine, coarsen, subspaces
    end
    refine_s   = Dict{Symbol,Vector{M_ref}}()
    coarsen_s  = Dict{Symbol,Vector{M_coar}}()
    subspaces_s = Dict{Symbol,Vector{M_sub}}()
    for X in keys(refine)
        Lx = L_X[X]
        if Lx == L_max
            refine_s[X]    = refine[X]
            coarsen_s[X]   = coarsen[X]
            subspaces_s[X] = subspaces[X]
            continue
        end
        Lx <= L_max ||
            error("Subspace `$X` has L_X = $Lx > L_max = $L_max; truncation not supported")
        synth2nat = [ceil(Int, Lx * i / L_max) for i in 1:L_max]
        rfX = Vector{M_ref}(undef, L_max)
        crX = Vector{M_coar}(undef, L_max)
        ssX = Vector{M_sub}(undef, L_max)
        for i in 1:L_max
            ni = synth2nat[i]
            ssX[i] = subspaces[X][ni]
            if i == L_max
                rfX[i] = refine[X][Lx]                # identity at fine
                crX[i] = coarsen[X][Lx]
            elseif synth2nat[i+1] > ni
                rfX[i] = refine[X][ni]                # real AMG step
                crX[i] = coarsen[X][ni]
            else
                # Identity transition at the level-l-X-broken-basis (rows of
                # subspaces[X][l]). `refine[l]` maps level-l-broken-basis to
                # level-(l+1)-broken-basis; both are the same here.
                m = size(ssX[i], 1)
                rfX[i] = sparse(one(T)*I, m, m)
                crX[i] = sparse(one(T)*I, m, m)
            end
        end
        refine_s[X]    = rfX
        coarsen_s[X]   = crX
        subspaces_s[X] = ssX
    end
    return refine_s, coarsen_s, subspaces_s
end

# Rewrite the `:uniform` subspace to use its *intrinsic* one-dimensional
# representation at every level except the finest. Mesh constructors set up
# `subspaces[:uniform][l] = ones(n_l, 1)` (broken-basis embedding of the constant
# function) and alias `refine[:uniform] === refine[:dirichlet]`. That makes
# `refine_fine_per[:uniform][l]` a dense rank-1 averaging matrix of shape
# n_doubled × n_l per level — which OOMs on GPU at large L.
#
# Here we collapse the broken-basis intermediate for `:uniform`:
#   • At the fine level L, keep `subspaces[:uniform][L] = ones(n_doubled, 1)`
#     so R_coarse[L] still lifts the scalar :uniform coefficient to the
#     broken-basis fine iterate (length n_doubled). refines/coarsens at L are
#     the identity at fine.
#   • At every coarser level l < L, `subspaces[:uniform][l] = [1]` (1×1
#     identity); the level-l :uniform iterate is just a scalar.
#   • The level-(L-1) → fine transition `refines_s[:uniform][L-1] =
#     ones(n_doubled, 1)` lifts the scalar to a constant fine vector, and
#     `coarsens_s[:uniform][L-1] = ones(1, n_doubled) / n_doubled` averages back.
#   • Earlier levels' refines/coarsens are 1×1 identities (no-op).
#
# The composed `refine_fine_per[:uniform][l]` then collapses (via the chain in
# amg_helper) to `(n_doubled × 1) ones` at every l < L — a sparse column, no
# outer product. `R_coarse[l]`/`D_levels[l,k]`/`refine_z[l]`/`coarsen_z[l]` all
# acquire heterogeneous block sizes for the :uniform variable, but each
# downstream consumer indexes through the actual matrix shapes (e.g.
# `size(R_coarse[J], 2)`) rather than a hardcoded `n_l`, so the math threads
# through.
function _normalize_uniform_subspace(::Type{T},
        subspaces::Dict{Symbol,Vector{M_sub}},
        refine::Dict{Symbol,Vector{M_ref}},
        coarsen::Dict{Symbol,Vector{M_coar}}) where {T,M_sub,M_ref,M_coar}
    haskey(subspaces, :uniform) || return subspaces, refine, coarsen
    # Only the SparseMatrixCSC path is supported here. The structured
    # geometric_mg path uses V/HBlockDiag refines/coarsens and would need its
    # own normalization (not currently a known OOM source — the bench uses
    # `amg(geom)` which is fully sparse).
    (M_sub <: SparseMatrixCSC && M_ref <: SparseMatrixCSC && M_coar <: SparseMatrixCSC) ||
        return subspaces, refine, coarsen

    L_max = length(refine[:uniform])
    n_doubled = size(subspaces[:uniform][L_max], 1)

    sub_new  = Vector{M_sub}(undef, L_max)
    ref_new  = Vector{M_ref}(undef, L_max)
    coar_new = Vector{M_coar}(undef, L_max)

    sub_new[L_max]  = subspaces[:uniform][L_max]
    ref_new[L_max]  = sparse(one(T)*I, n_doubled, n_doubled)
    coar_new[L_max] = sparse(one(T)*I, n_doubled, n_doubled)

    one_x_one = sparse(reshape([one(T)], (1, 1)))
    for l in 1:L_max-1
        sub_new[l]  = one_x_one
        if l == L_max - 1
            ref_new[l]  = sparse(ones(T, n_doubled, 1))
            coar_new[l] = sparse(ones(T, 1, n_doubled) ./ n_doubled)
        else
            ref_new[l]  = one_x_one
            coar_new[l] = one_x_one
        end
    end

    subspaces_new = copy(subspaces)
    refine_new    = copy(refine)
    coarsen_new   = copy(coarsen)
    subspaces_new[:uniform] = sub_new
    refine_new[:uniform]    = ref_new
    coarsen_new[:uniform]   = coar_new

    return subspaces_new, refine_new, coarsen_new
end

# Backward-compat constructor: accept plain Vector refine/coarsen and replicate them
# across every subspace key. Lets per-mesh `amg()` keep building a single shared
# hierarchy while we incrementally add per-subspace hierarchies.
function MultiGrid(geometry::G,
                   subspaces::Dict{Symbol,Vector{M_sub}},
                   refine::Vector{M_ref},
                   coarsen::Vector{M_coar}) where {T,M_sub,M_ref,M_coar,G<:Geometry{T}}
    refine_dict  = Dict{Symbol,Vector{M_ref}}(k  => refine  for k in keys(subspaces))
    coarsen_dict = Dict{Symbol,Vector{M_coar}}(k => coarsen for k in keys(subspaces))
    MultiGrid(geometry, subspaces, refine_dict, coarsen_dict)
end

# Forward common Geometry fields onto MultiGrid.
function Base.getproperty(mg::MultiGrid, sym::Symbol)
    if sym === :x || sym === :w || sym === :operators || sym === :discretization
        return getproperty(getfield(mg, :geometry), sym)
    end
    return getfield(mg, sym)
end

Base.propertynames(mg::MultiGrid, private::Bool=false) =
    (:geometry, :subspaces, :refine, :coarsen, :x, :w, :operators, :discretization)

@kwdef struct AMG{X,W,M_sub,M_D_lev,M_D_fine,M_ref,M_coar,M_refz,M_coarz,G}
    geometry::G
    x::X
    w::W
    R_fine::Vector{M_sub}
    R_coarse::Vector{M_sub}
    # D_levels[l, k]: discrete operator k at multigrid level l, kept sparse so the
    # Galerkin triple product `coarsen_fine[l] * op * refine_fine[l]` is well-typed
    # at every level. Used by `mgb_phase1` only.
    D_levels::Matrix{M_D_lev}
    # D_fine[k]: discrete operator k at the finest level, preserving whatever block
    # structure `geometry.operators[k]` has. Used by `mgb_step` / `mgb_core` /
    # `mgb_driver` — i.e. the main phase, where every Newton step in the V-cycle
    # benefits from batched-gemm Hessian assembly when this is a `BlockDiag`.
    D_fine::Vector{M_D_fine}
    refine_u::Vector{M_ref}
    coarsen_u::Vector{M_coar}
    refine_z::Vector{M_refz}
    coarsen_z::Vector{M_coarz}
end

"""
    multigrid_from_fine_grid(mg::MultiGrid, f_grid_fine; subspace::Symbol=:dirichlet)

Coarsen a fine-level grid value to all multigrid levels using the `MultiGrid`'s
coarsening operators in `subspace` (defaults to `:dirichlet`, which is the
canonical geometric coarsening). Returns a `Vector` of length `L` where index `L`
is the fine level and index 1 is the coarsest.
"""
function multigrid_from_fine_grid(mg::MultiGrid, f_grid_fine; subspace::Symbol=:dirichlet)
    coarsen = mg.coarsen[subspace]
    L = length(coarsen)
    f_grid = Vector{typeof(f_grid_fine)}(undef, L)
    f_grid[L] = f_grid_fine
    for l = L-1:-1:1
        f_grid[l] = coarsen[l] * f_grid[l+1]
    end
    return f_grid
end

"""
    amg(geom::Geometry) -> MultiGrid

Build an algebraic-multigrid hierarchy on top of `geom`, returning a `MultiGrid`.
Dispatched per discretization; the hierarchy's fine level matches `geom`.
"""
function amg end

"""
    geometric_mg(geom::Geometry, L::Int) -> MultiGrid

Build a geometric-subdivision multigrid hierarchy of `L` levels on top of `geom`. The
returned `MultiGrid`'s `geometry` is the finest mesh (after `L-1` levels of subdivision).
"""
function geometric_mg end

"""
    find_boundary(geom::Geometry) -> Vector{Int}

Return the **broken-basis row indices** (indices into `geom.x`) of the mesh
nodes that lie on `∂Ω`, in the user's native per-element layout. Duplicates
are present: a corner vertex shared by `k` triangles contributes its `k`
broken-basis rows; an edge midpoint shared by two triangles contributes both.

`amg(geom; dirichlet_nodes=…)` accepts the same broken-basis indexing. A
geometric position is treated as Dirichlet iff **any** broken-basis row at
that position is in `dirichlet_nodes`, so the user can pass either the full
set returned by `find_boundary` (or a subset of it) or a sparser
representative-only set — either works.

Defined for each FEM discretization (`FEM1D`, `FEM2D_P1`, `FEM2D_P2`,
`FEM3D`). For spectral discretizations the zero-trace subspace is built by
basis truncation rather than by node masking; the spectral `amg` does not
accept `dirichlet_nodes` and `find_boundary` returns the perimeter Chebyshev
indices for reference only.

Empty or singleton `dirichlet_nodes` are allowed; the resulting AMG problem
may still be well-posed if the variational form carries a mass term.
"""
function find_boundary end

"""
    subdivide(geom::Geometry, L::Int) -> Geometry

Refine `geom`'s mesh by `L-1` levels of geometric subdivision and return a new fine-mesh
`Geometry`, discarding the transfer operators that the geometric MG construction would
otherwise produce. The returned `Geometry` preserves whatever block structure
`geometric_mg(...; structured=true)` produces in its fine operators (BlockDiag for FEM),
so that a subsequent `amg(subdivide(geom, L))` benefits from batched-gemm Hessian
assembly at the finest level via the D_fine path.
"""
subdivide(geom::Geometry, L::Int) = geometric_mg(geom, L; structured=true).geometry

# Galerkin triple product coarsen * op * refine, used by amg_helper to build the
# per-level D_levels matrix. Default: just do the natural matmul. The specialization
# in BlockMatrices.jl handles the awkward case of *sparse* AMG transfers wrapped
# around a *structured* (BlockDiag) operator by converting the operator to sparse
# first (otherwise the generic sparse-times-BlockDiag fallback scalar-indexes the
# BlockDiag and crashes). When both transfers and op are structured (geometric MG),
# the natural matmul kicks in and the structured form is preserved.
_galerkin(coar, op, ref) = coar * op * ref

function amg_helper(mg::MultiGrid{T,M_sub,M_ref,M_coar,G},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol}) where {T,M_sub,M_ref,M_coar,G}
    geometry = mg.geometry
    x = geometry.x
    w = geometry.w
    operators = geometry.operators

    # Per-subspace hierarchies are pre-stretched to a common depth L_max at
    # `MultiGrid` construction time (see `_stretch_per_subspace`), so every
    # `mg.refine[X]` / `mg.coarsen[X]` / `mg.subspaces[X]` already has the
    # same length here.
    #
    # The CANONICAL hierarchy below (used for the level-l integration grid via
    # `refine_fine` / `coarsen_fine` / `ncanon`, and for quadrature-weight
    # restriction via `refine_u`) is `:full`'s all-corners Neumann chain — so
    # the level-l grid includes boundary corners and the level-l integration
    # covers the whole domain. Cross-hierarchy interpolation from each
    # per-variable level-l basis to this canonical grid happens automatically
    # in the Galerkin triple product `coarsen_fine[l] · op · refine_fine_per[X][l]`
    # via the fine basis as a bridge. Per-variable iterate transfers
    # (`refine_z`/`coarsen_z`) still consult each variable's own subspace.
    refines_s   = mg.refine
    coarsens_s  = mg.coarsen
    subspaces_s = mg.subspaces
    subspaces = subspaces_s

    refine = refines_s[:full]
    coarsen = coarsens_s[:full]
    L = length(refine)
    @assert size(w) == (size(x)[1],) && length(refine)==L && length(coarsen)==L
    for l=1:L
        @assert norm(coarsen[l]*refine[l]-I)<sqrt(eps(T))
    end
    refine_fine = Vector{M_ref}(undef,(L,))
    refine_fine[L] = refine[L]
    coarsen_fine = Vector{M_coar}(undef,(L,))
    coarsen_fine[L] = coarsen[L]
    for l=L-1:-1:1
        refine_fine[l] = refine_fine[l+1]*refine[l]
        coarsen_fine[l] = coarsen[l]*coarsen_fine[l+1]
    end
    nu = size(state_variables)[1]
    @assert size(state_variables)[2] == 2
    # Per-subspace composed fine prolongations: `refine_fine_per[X][l] *
    # subspaces[X][l]` is the level-l-X-coords → fine-broken-basis lift for
    # subspace X. For most subspaces this is the canonical (:dirichlet) AMG's
    # `refine_fine[l]`, which goes through the interior-corner P1 bridge. For
    # `:uniform`, however, that bridge is *wrong*: the AMG's interior space
    # excludes boundary corners, so a "constant on interior corners" coarse
    # representation lifts to the interior-corner indicator at fine (zero at
    # boundary), not the true `ones(n_doubled)` constant. We special-case
    # `:uniform` to produce its `refine_fine` block as the rank-1 averaging
    # operator `(1/n_l) * ones(n_doubled, n_l)`, which sends the all-ones
    # `subspaces[:uniform][l] = ones(n_l, 1)` representation directly to
    # `ones(n_doubled, 1)` — preserving the global constant at every level.
    n_doubled = size(refine_fine[L], 1)
    refine_fine_per = Dict{Symbol, Vector}()
    # Only build refine_fine_per for subspaces that the current solve actually
    # consumes (state_variables[:,2]).
    needed_subspaces = unique(state_variables[:, 2])
    for X in needed_subspaces
        # :full short-circuit: refine_fine was already built above for the
        # canonical (:full) hierarchy; don't recompute the chain.
        if X === :full
            refine_fine_per[X] = refine_fine
        else
            # Compose X's per-subspace chain up to fine. For :dirichlet this is
            # the AMG-tuned interior-corner P1 chain (preserves zero trace at
            # every level). For :uniform with the heterogeneous (intrinsic dim
            # 1) representation, the chain collapses to a (n_doubled × 1)
            # column ones(n_doubled) at l<L plus identity at l=L. Cross-mapping
            # to the canonical :full level-l grid happens implicitly in the
            # Galerkin product `coarsen_fine[l] · op · refine_fine_per[X][l]`
            # via the fine basis bridge.
            rfp = Vector{eltype(refines_s[X])}(undef, L)
            rfp[L] = refines_s[X][L]
            for l = L-1:-1:1
                rfp[l] = rfp[l+1] * refines_s[X][l]
            end
            refine_fine_per[X] = rfp
        end
    end
    R_coarse = [amgb_blockdiag((subspaces[state_variables[k,2]][l] for k=1:nu)...) for l=1:L]
    R_fine = [amgb_blockdiag((refine_fine_per[state_variables[k,2]][l]*subspaces[state_variables[k,2]][l] for k=1:nu)...) for l=1:L]
    nD = size(D)[1]
    @assert size(D)[2]==2
    bar = Dict{Symbol,Int}()
    for k=1:nu
        bar[state_variables[k,1]] = k
    end
    # D_levels[l, k]: Galerkin projection of operator k to multigrid level l, for
    # the coarser levels only (l = 1 .. L-1). Stays structured if transfers are
    # structured (geometric MG); falls back to sparse if transfers are sparse (AMG)
    # via the specialized `_galerkin` overload that converts a BlockDiag op to
    # sparse before multiplying. The fine-level (l = L) operator is *not* stored
    # here — phase 1 picks it up from `D_fine` so its level-L Newton solves get
    # the structure-preserving batched-gemm path. With L = 1, D_levels is empty
    # (0 × nD) and phase 1 never indexes it.
    #
    # Per-variable block widths: at level l, variable v's column block has
    # width `size(subspaces[X_v][l], 1)` (rows of X_v's level-l broken-basis
    # embedding). When every X shares :dirichlet's hierarchy this equals
    # `sizes[l]` for all v, giving the legacy homogeneous nu*sizes[l] cols;
    # once a subspace exposes a different level-l-broken-basis row count
    # (e.g. a depth-1 :uniform with rows = n_doubled), the block widths
    # become heterogeneous, but the canonical output dim (`sizes[l]` rows)
    # is preserved.
    ncanon = [size(coarsen_fine[l], 1) for l in 1:L]
    D_levels = [let
            foo = [v == bar[D[k,1]] ?
                       _galerkin(coarsen_fine[l], operators[D[k,2]],
                                 refine_fine_per[state_variables[v,2]][l]) :
                       amgb_zeros(coarsen_fine[l],
                                  ncanon[l],
                                  size(subspaces[state_variables[v,2]][l], 1))
                   for v in 1:nu]
            hcat(foo...)
        end for l=1:L-1, k=1:nD]
    # D_fine[k]: finest-level operator k with its original structure preserved.
    # At l = L, `coarsen_fine[L]` and `refine_fine[L]` are identity, so we skip
    # the triple product entirely and slot `operators[k]` straight into the
    # nu-state-variable hcat. `amgb_zeros` returns BlockDiag zeros when given a
    # BlockDiag, so `hcat` returns a BlockColumn — exactly the structured form
    # the f2 barrier closure exploits for batched-gemm Hessian assembly.
    D_fine = [let
            op = operators[D[k,2]]
            n = size(op, 1)
            Z = amgb_zeros(op, n, n)
            foo = [Z for j=1:nu]
            foo[bar[D[k,1]]] = op
            hcat(foo...)
        end for k=1:nD]
    # Per-variable iterate transfers: each state variable's iterate at level l
    # gets prolonged using *its own* subspace hierarchy. Today, every subspace
    # symbol points at the same shared `Vector` (the canonical :dirichlet
    # hierarchy), so the blockdiag is equivalent to repeating one matrix. Once
    # per-subspace hierarchies land (:uniform with depth-1, :full with continuous
    # all-corners AMG, etc.), each block here will differ in shape.
    # Per-variable iterate transfers: blockdiag of each state variable's
    # *stretched* per-subspace hierarchy step at level l.
    refine_z  = [amgb_blockdiag([refines_s[state_variables[k,2]][l]  for k=1:nu]...) for l=1:L]
    coarsen_z = [amgb_blockdiag([coarsens_s[state_variables[k,2]][l] for k=1:nu]...) for l=1:L]
    AMG(geometry=geometry,x=x,w=w,R_fine=R_fine,R_coarse=R_coarse,
        D_levels=D_levels,D_fine=D_fine,
        refine_u=refine,coarsen_u=coarsen,refine_z=refine_z,coarsen_z=coarsen_z)
end

# Internal: build the (M1, M2) AMG pair from a MultiGrid.
function _prepare_amg(mg::MultiGrid{T};
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol},
        full_space=:full,
        id_operator=:id,
        feasibility_slack=:feasibility_slack
        ) where {T}
    M1 = amg_helper(mg,state_variables,D)
    s1 = vcat(state_variables,[feasibility_slack full_space])
    D1 = vcat(D,[feasibility_slack id_operator])
    M2 = amg_helper(mg,s1,D1)
    return M1,M2
end

# Helper for convex_piecewise: extract args slice and call piece barrier
# Uses @generated to create compile-time efficient code for tuple slicing
@generated function _call_piece_barrier(f::F, all_rows_and_y::Tuple, ::Val{a}, ::Val{b}) where {F, a, b}
    M = fieldcount(all_rows_and_y)
    # Extract args at positions a to b, plus y at position M
    arg_exprs = [:(all_rows_and_y[$i]) for i in a:b]
    y_expr = :(all_rows_and_y[$M])
    quote
        @inline f($(arg_exprs...), $y_expr)
    end
end

@doc raw"""
    Convex{T}

Container for a convex constraint set used by AMGB.

Fields:
- barrier: (F0, F1, F2) - value, gradient, Hessian functions
- cobarrier: (F0, F1, F2) - with slack for feasibility
- slack: initial slack function
- args: tuple of parameter arrays, splatted to map_rows_gpu

Barrier functions receive `(args_rows..., y)` where args_rows are per-vertex
parameter data (via broadcasting), and y is the solution SVector.
This enables true GPU execution without scalar indexing.

Construct via helpers like `convex_linear`, `convex_Euclidian_power`, `convex_piecewise`, or `intersect`.
These helpers return `Vector{Convex{T}}` with one Convex per multigrid level.
"""
struct Convex{T, Args<:Tuple, B<:Tuple, CB<:Tuple, S}
    barrier::B      # (F0, F1, F2) - value, gradient, Hessian (any callable)
    cobarrier::CB   # (F0, F1, F2) - value, gradient, Hessian (any callable)
    slack::S        # slack function (any callable)
    args::Args      # Tuple of parameter arrays for this level
end

# Outer constructor: infer all type parameters
function Convex{T}(barrier::B, cobarrier::CB, slack::S, args::Args) where {T, B<:Tuple, CB<:Tuple, S, Args<:Tuple}
    Convex{T, Args, B, CB, S}(barrier, cobarrier, slack, args)
end

# Helper: A' * Diagonal(d) * A for SMatrix or UniformScaling, returns flattened SVector
@inline function _At_diag_A(::UniformScaling, d::SVector{N,T}) where {N,T}
    # I' * Diag(d) * I = Diag(d), flattened column-major
    SVector(ntuple(i -> (i - 1) ÷ N + 1 == (i - 1) % N + 1 ? d[(i - 1) % N + 1] : zero(T), Val(N * N)))
end

@inline function _At_diag_A(A::SMatrix{M,N,T}, d::SVector{M,T}) where {M,N,T}
    # (A'DA)[i,j] = sum_k A[k,i] * d[k] * A[k,j]
    H = A' * Diagonal(d) * A
    SVector(H)
end

# Helper: A' * v for SMatrix or UniformScaling
@inline _At_mul(::UniformScaling, v::SVector) = v
@inline _At_mul(A::SMatrix, v::SVector) = A' * v

# Helper: A * v + b for SMatrix or UniformScaling
@inline _A_mul_plus_b(::UniformScaling, y::SVector, b::SVector) = y .+ b
@inline _A_mul_plus_b(::UniformScaling, y::SVector, b::T) where {T<:Number} = y .+ b
@inline _A_mul_plus_b(A::SMatrix, y::SVector, b::SVector) = A * y .+ b
@inline _A_mul_plus_b(A::SMatrix, y::SVector, b::T) where {T<:Number} = A * y .+ b

# GPU-compatible index types: Colon (all) or SVector of Int (static indices)
const GPUIndex = Union{Colon, SVector{<:Any, Int}}

"""
    convex_linear(::Type{T}=Float64; geometry, idx=Colon(), A=(x)->I, b=(x)->T(0), A_grid=nothing, b_grid=nothing)

Create a convex set defined by linear inequality constraints, with GPU support.

Constructs `Vector{Convex{T}}` (one per multigrid level) representing linear constraints.
Defines `F(y) = A * y[idx] + b` where A, b are pre-computed per vertex.
The interior is `F > 0` (logarithmic barrier applied to each component).

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `mg::MultiGrid`: Required. The multigrid hierarchy (provides grid and coarsen operators).
- `idx=Colon()`: Indices of y to which constraints apply (default: all)
- `A::Function`: Matrix function `x -> A(x)` for constraint coefficients
- `b::Function`: Vector function `x -> b(x)` for constraint bounds
- `A_grid`, `b_grid`: Optional pre-computed grids (computed from A,b if not provided)

# Returns
`Vector{Convex{T}}` with one Convex per multigrid level. Each level's barriers
capture their pre-computed grids and receive `(j::Integer, y)`.

# Examples
```julia
mg = amg(fem1d(Float32; nodes=collect(range(-1f0, 1f0, length=33))))

# Box constraints: -1 ≤ y ≤ 1
A_box(x) = SMatrix{4,2,Float32}(1,0,-1,0, 0,1,0,-1)
b_box(x) = SVector{4,Float32}(1, 1, 1, 1)
Q = convex_linear(Float32; mg=mg, A=A_box, b=b_box, idx=SVector(1, 2))
```
"""
function convex_linear(::Type{T}=Float64;
        mg::MultiGrid,
        idx::GPUIndex=Colon(),
        A::Function=(x)->I,
        b::Function=(x)->T(0),
        A_grid = nothing,
        b_grid = nothing) where {T}

    L = length(mg.coarsen[:dirichlet])
    x_fine = mg.x

    # Determine constraint dimension from sample evaluation
    # Use _to_cpu_array to avoid scalar indexing on GPU arrays
    x_cpu = _to_cpu_array(x_fine)
    A_sample = A(x_cpu isa AbstractMatrix ? x_cpu[1,:] : x_cpu[1])
    nconstraints = A_sample isa UniformScaling ? nothing : size(A_sample, 1)
    nidx = idx isa Colon ? (A_sample isa UniformScaling ? nothing : size(A_sample, 2)) : length(idx)

    # Pre-compute grids at fine level if not provided
    if A_grid === nothing
        A_grid = multigrid_from_fine_grid(mg,
            map_rows(xi -> begin
                Ax = A(xi)
                if Ax isa UniformScaling
                    Ax  # Keep UniformScaling as-is (will be handled specially)
                else
                    SVector(vec(Ax))  # Flatten matrix to SVector
                end
            end, x_fine))
    end

    if b_grid === nothing
        b_grid = multigrid_from_fine_grid(mg,
            map_rows(xi -> begin
                bx = b(xi)
                if bx isa Number
                    SVector(bx)
                else
                    SVector(bx)
                end
            end, x_fine))
    end

    # Build Convex for each level
    Q = Vector{Convex{T}}(undef, L)

    for l = 1:L
        # Capture this level's grids - keep MPI wrappers for proper dispatch
        A_l = A_grid[l]
        b_l = b_grid[l]

        # Barrier functions receive row data via broadcasting: (A_row, b_row, y)
        # No index lookup - GPU compatible
        function barrier_f0_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            # Reconstruct A from flattened form if needed
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx
            end
            -sum(log.(Fval))
        end

        function barrier_f1_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx
                inv_F = one(TT) ./ Fval
                grad_idx = -inv_F
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx
                inv_F = one(TT) ./ Fval
                grad_idx = -_At_mul(Ax, inv_F)
            end
            _scatter_gradient(idx, grad_idx, Val(N))
        end

        function barrier_f2_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx
                inv_F2 = one(TT) ./ (Fval .^ 2)
                H_idx_flat = _At_diag_A(I, inv_F2)
                M = length(inv_F2)
                H_idx = reshape(H_idx_flat, Size(M, M))
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx
                inv_F2 = one(TT) ./ (Fval .^ 2)
                H_idx_flat = _At_diag_A(Ax, inv_F2)
                M = nc
                H_idx = reshape(H_idx_flat, Size(ni, ni))
            end
            _scatter_hessian(idx, H_idx, Val(N))
        end

        # Cobarrier functions receive row data via broadcasting: (A_row, b_row, yhat)
        function cobarrier_f0_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx .+ slack
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx .+ slack
            end
            -sum(log.(Fval))
        end

        function cobarrier_f1_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx .+ slack
                inv_F = one(TT) ./ Fval
                grad_idx = -inv_F
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx .+ slack
                inv_F = one(TT) ./ Fval
                grad_idx = -_At_mul(Ax, inv_F)
            end
            g_slack = -sum(inv_F)
            _scatter_cobarrier_gradient(idx, grad_idx, g_slack, Val(NP1))
        end

        function cobarrier_f2_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx .+ slack
                inv_F2 = one(TT) ./ (Fval .^ 2)
                H_idx_flat = _At_diag_A(I, inv_F2)
                cross = inv_F2
                M = length(inv_F2)
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx .+ slack
                inv_F2 = one(TT) ./ (Fval .^ 2)
                H_idx_flat = _At_diag_A(Ax, inv_F2)
                cross = _At_mul(Ax, inv_F2)
                M = nc
            end
            H_ss = sum(inv_F2)
            _scatter_cobarrier_hessian(idx, H_idx_flat, cross, H_ss, Val(M), Val(NP1))
        end

        function slack_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            if Ax_flat isa UniformScaling
                Fval = yidx .+ bx
            else
                nc = length(bx)
                ni = length(yidx)
                Ax = SMatrix{nc,ni,TT}(Ax_flat)
                Fval = Ax * yidx .+ bx
            end
            -minimum(Fval)
        end

        Q[l] = Convex{T}(
            (barrier_f0_l, barrier_f1_l, barrier_f2_l),
            (cobarrier_f0_l, cobarrier_f1_l, cobarrier_f2_l),
            slack_l,
            (A_l, b_l)  # args tuple - splatted to map_rows_gpu
        )
    end

    return Q
end

normsquared(z) = dot(z,z)

# GPU-compatible helpers for scattering gradients and Hessians
# Must be top-level functions (not closures) to avoid Core.Box capture on GPU

"""
    _scatter_gradient(idx, grad, ::Val{N})

GPU-compatible helper: scatter a gradient vector to full-size SVector.
When idx is Colon, returns grad unchanged.
When idx is SVector{M,Int}, scatters grad into a zero vector of size N.
"""
@inline _scatter_gradient(::Colon, grad::SVector{N,T}, ::Val{N}) where {N,T} = grad
@inline function _scatter_gradient(idx_sv::SVector{M,Int}, grad::SVector{M,T}, ::Val{N}) where {M,N,T}
    # GPU-compatible: avoid return inside loop (use assignment instead)
    SVector{N,T}(ntuple(Val(N)) do i
        result = zero(T)
        @inbounds for k in 1:M
            if idx_sv[k] == i
                result = grad[k]
            end
        end
        result
    end)
end

"""
    _scatter_hessian(idx, H, ::Val{N})

GPU-compatible helper: scatter a Hessian matrix to full-size matrix (returned as SVector).
When idx is Colon, returns SVector(H).
When idx is SVector{M,Int}, scatters H into a zero matrix of size N×N.
"""
@inline _scatter_hessian(::Colon, H::SMatrix{N,N,T}, ::Val{N}) where {N,T} = SVector(H)
@inline function _scatter_hessian(idx_sv::SVector{M,Int}, H::SMatrix{M,M,T}, ::Val{N}) where {M,N,T}
    # GPU-compatible: avoid return inside loop (use assignment instead)
    SVector(SMatrix{N,N,T}(ntuple(Val(N*N)) do k
        i = (k - 1) % N + 1
        j = (k - 1) ÷ N + 1
        ki = 0
        kj = 0
        @inbounds for l in 1:M
            if idx_sv[l] == i
                ki = l
            end
            if idx_sv[l] == j
                kj = l
            end
        end
        result = zero(T)
        if ki > 0 && kj > 0
            result = H[ki, kj]
        end
        result
    end))
end

"""
    _scatter_cobarrier_gradient(idx, grad, g_slack, ::Val{NP1})

GPU-compatible helper: scatter cobarrier gradient with slack term.
Builds full gradient of size NP1 where positions idx get grad and position NP1 gets g_slack.
"""
@inline function _scatter_cobarrier_gradient(::Colon, grad::SVector{N,T}, g_slack::T, ::Val{NP1}) where {N,T,NP1}
    push(grad, g_slack)
end
@inline function _scatter_cobarrier_gradient(idx_sv::SVector{M,Int}, grad::SVector{M,T}, g_slack::T, ::Val{NP1}) where {M,T,NP1}
    SVector{NP1,T}(ntuple(Val(NP1)) do i
        result = zero(T)
        if i == NP1
            result = g_slack
        else
            @inbounds for k in 1:M
                if idx_sv[k] == i
                    result = grad[k]
                end
            end
        end
        result
    end)
end

"""
    _scatter_cobarrier_hessian(idx, H_idx_flat, cross, H_ss, ::Val{M}, ::Val{NP1})

GPU-compatible helper: scatter cobarrier Hessian with slack cross terms.
H_idx_flat is flattened column-major (from _At_diag_A), M is size of H_idx.
Builds full Hessian of size NP1×NP1 with:
- H_idx at positions (idx, idx)
- cross at positions (idx, NP1) and (NP1, idx)
- H_ss at position (NP1, NP1)
"""
@inline function _scatter_cobarrier_hessian(::Colon, H_idx_flat::SVector{MM,T}, cross::SVector{M,T}, H_ss::T, ::Val{M}, ::Val{NP1}) where {MM,M,T,NP1}
    # For Colon idx, NP1 = M+1, so we can build directly
    # H_idx_flat is M×M flattened column-major
    SVector(SMatrix{NP1,NP1,T}(ntuple(Val(NP1*NP1)) do k
        i = (k - 1) % NP1 + 1
        j = (k - 1) ÷ NP1 + 1
        result = zero(T)
        if i <= M && j <= M
            # H_idx_flat is column-major: element (i,j) is at index (j-1)*M + i
            result = H_idx_flat[(j - 1) * M + i]
        elseif i <= M && j == NP1
            result = cross[i]
        elseif i == NP1 && j <= M
            result = cross[j]
        else  # i == NP1 && j == NP1
            result = H_ss
        end
        result
    end))
end
@inline function _scatter_cobarrier_hessian(idx_sv::SVector{M,Int}, H_idx_flat::SVector{MM,T}, cross::SVector{M,T}, H_ss::T, ::Val{M2}, ::Val{NP1}) where {MM,M,T,M2,NP1}
    # H_idx_flat is M×M flattened column-major
    SVector(SMatrix{NP1,NP1,T}(ntuple(Val(NP1*NP1)) do k
        i = (k - 1) % NP1 + 1
        j = (k - 1) ÷ NP1 + 1
        # Find indices in idx_sv
        ki = 0
        kj = 0
        @inbounds for l in 1:M
            if idx_sv[l] == i
                ki = l
            end
            if idx_sv[l] == j
                kj = l
            end
        end
        result = zero(T)
        if i == NP1 && j == NP1
            result = H_ss
        elseif i == NP1 && kj > 0
            result = cross[kj]
        elseif j == NP1 && ki > 0
            result = cross[ki]
        elseif ki > 0 && kj > 0
            # H_idx_flat is column-major: element (ki,kj) is at index (kj-1)*M + ki
            result = H_idx_flat[(kj - 1) * M + ki]
        end
        result
    end))
end

"""
    _static_pop(z::SVector{NZ,T}, ::Val{NZM1}) where {NZ,T,NZM1}

GPU-compatible helper: pop the last element from an SVector.
Uses compile-time constant NZM1 to avoid dynamic dispatch.
Returns an SVector of size NZM1 = NZ-1.
"""
@inline function _static_pop(z::SVector{NZ,T}, ::Val{NZM1}) where {NZ,T,NZM1}
    SVector{NZM1,T}(ntuple(i -> @inbounds(z[i]), Val(NZM1)))
end

"""
    _safe_pow(s::T, α::T) where T

GPU-compatible power function: compute s^α using exp(α * log(s)).
Avoids boxing issues with non-integer exponents on GPU.
"""
@inline function _safe_pow(s::T, α::T) where T
    exp(α * log(s))
end

# =============================================================================
# GPU-Compatible Barrier Functors for Euclidian Power Constraints
# =============================================================================
#
# These functor structs encode compile-time constants (NZ, IDX) as type parameters
# so the GPU compiler can resolve all dimensions without heap allocation.

"""
    _ep_get_z_and_parts(Ax, bx, idx, y, nz_m1_val)

GPU-compatible helper to compute z = Ax * y[idx] + bx and split into (q, s).
Dispatches on idx type for optimal GPU code generation.
"""
@inline function _ep_get_z_and_parts(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                      ::Colon, y::SVector{N,TT},
                                      ::Val{NZM1}) where {NZ,NZM1,N,TT}
    z = Ax * y + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ]
    return z, q, s
end

@inline function _ep_get_z_and_parts(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                      idx::SVector{M,Int}, y::SVector{N,TT},
                                      ::Val{NZM1}) where {NZ,NZM1,M,N,TT}
    # Extract y[idx] using static indexing
    y_idx = SVector{NZ,TT}(ntuple(i -> @inbounds(y[idx[i]]), Val(NZ)))
    z = Ax * y_idx + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ]
    return z, q, s
end

# Cobarrier version that pops yhat first
@inline function _ep_get_z_and_parts_cobarrier(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                                ::Colon, yhat::SVector{NP1,TT},
                                                ::Val{NZM1}) where {NZ,NZM1,NP1,TT}
    # Pop slack from yhat
    y = _static_pop(yhat, Val(NP1 - 1))
    slack = yhat[NP1]
    z = Ax * y + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ] + slack
    return z, q, s, slack
end

@inline function _ep_get_z_and_parts_cobarrier(Ax::SMatrix{NZ,NZ,TT}, bx::SVector{NZ,TT},
                                                idx::SVector{M,Int}, yhat::SVector{NP1,TT},
                                                ::Val{NZM1}) where {NZ,NZM1,M,NP1,TT}
    # Pop slack from yhat
    y = _static_pop(yhat, Val(NP1 - 1))
    slack = yhat[NP1]
    # Extract y[idx] using static indexing
    y_idx = SVector{NZ,TT}(ntuple(i -> @inbounds(y[idx[i]]), Val(NZ)))
    z = Ax * y_idx + bx
    q = _static_pop(z, Val(NZM1))
    s = z[NZ] + slack
    return z, q, s, slack
end

"""
    EuclidianPowerBarrier{NZ,NZM1,IDX}

GPU-compatible functor for barrier function evaluation.
Encodes dimension NZ and index type IDX as type parameters.
"""
struct EuclidianPowerBarrier{NZ,NZM1,IDX}
    idx::IDX  # Store the actual index value
end

EuclidianPowerBarrier(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerBarrier{NZ,NZM1,IDX}(idx)

# Barrier f0 (value)
@inline function (b::EuclidianPowerBarrier{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    α = TT(2) / p0
    -log(_safe_pow(s, α) - normsquared(q)) - μ * log(s)
end

"""
    EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}

GPU-compatible functor for barrier gradient evaluation.
"""
struct EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}
    idx::IDX
    core_grad::CoreGrad
end

EuclidianPowerBarrierGrad(::Val{NZ}, ::Val{NZM1}, idx::IDX, cg::CoreGrad) where {NZ,NZM1,IDX,CoreGrad} =
    EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad}(idx, cg)

@inline function (b::EuclidianPowerBarrierGrad{NZ,NZM1,IDX,CoreGrad})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX,CoreGrad}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    grad_z = b.core_grad(q, s, p0, μ)
    grad_idx = Ax' * grad_z
    return _scatter_gradient(b.idx, grad_idx, Val(N))
end

"""
    EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}

GPU-compatible functor for barrier Hessian evaluation.
"""
struct EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}
    idx::IDX
    core_hess::CoreHess
end

EuclidianPowerBarrierHess(::Val{NZ}, ::Val{NZM1}, idx::IDX, ch::CoreHess) where {NZ,NZM1,IDX,CoreHess} =
    EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess}(idx, ch)

@inline function (b::EuclidianPowerBarrierHess{NZ,NZM1,IDX,CoreHess})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX,CoreHess}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    H_z_flat = b.core_hess(q, s, p0, μ)
    H_z = reshape(SVector(H_z_flat), Size(NZ, NZ))
    H_idx = Ax' * H_z * Ax
    return _scatter_hessian(b.idx, H_idx, Val(N))
end

"""
    EuclidianPowerCobarrier{NZ,NZM1,IDX}

GPU-compatible functor for cobarrier function evaluation.
"""
struct EuclidianPowerCobarrier{NZ,NZM1,IDX}
    idx::IDX
end

EuclidianPowerCobarrier(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerCobarrier{NZ,NZM1,IDX}(idx)

@inline function (b::EuclidianPowerCobarrier{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s, _ = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    α = TT(2) / p0
    -log(_safe_pow(s, α) - normsquared(q)) - μ * log(s)
end

"""
    EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}

GPU-compatible functor for cobarrier gradient evaluation.
"""
struct EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}
    idx::IDX
    core_grad::CoreGrad
end

EuclidianPowerCobarrierGrad(::Val{NZ}, ::Val{NZM1}, idx::IDX, cg::CoreGrad) where {NZ,NZM1,IDX,CoreGrad} =
    EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad}(idx, cg)

@inline function (b::EuclidianPowerCobarrierGrad{NZ,NZM1,IDX,CoreGrad})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX,CoreGrad}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s, _ = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    grad_z = b.core_grad(q, s, p0, μ)

    # Build gradient using ntuple (GPU-compatible)
    grad_idx = Ax' * grad_z
    g = if b.idx isa Colon
        SVector{NP1,TT}(ntuple(Val(NP1)) do i
            i == NP1 ? grad_z[NZ] : grad_idx[i]
        end)
    else
        SVector{NP1,TT}(ntuple(Val(NP1)) do i
            if i == NP1
                grad_z[NZ]
            else
                # Find if i is in idx
                found = zero(TT)
                for (k, idx_k) in enumerate(b.idx)
                    if idx_k == i
                        found = grad_idx[k]
                    end
                end
                found
            end
        end)
    end
    return g
end

"""
    EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}

GPU-compatible functor for cobarrier Hessian evaluation.
"""
struct EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}
    idx::IDX
    core_hess::CoreHess
end

EuclidianPowerCobarrierHess(::Val{NZ}, ::Val{NZM1}, idx::IDX, ch::CoreHess) where {NZ,NZM1,IDX,CoreHess} =
    EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess}(idx, ch)

@inline function (b::EuclidianPowerCobarrierHess{NZ,NZM1,IDX,CoreHess})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, yhat::SVector{NP1,TT}) where {NZ,NZ2,NZM1,NP1,TT,IDX,CoreHess}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)
    μ = TT(mu_val)

    _, q, s, _ = _ep_get_z_and_parts_cobarrier(Ax, bx, b.idx, yhat, Val(NZM1))
    H_z_flat = b.core_hess(q, s, p0, μ)
    H_z = reshape(SVector(H_z_flat), Size(NZ, NZ))

    H_idx = Ax' * H_z * Ax
    cross = Ax' * H_z[:, NZ]
    H_ss = H_z[NZ, NZ]

    return _scatter_cobarrier_hessian(b.idx, SVector(H_idx), cross, H_ss, Val(NZ), Val(NP1))
end

"""
    EuclidianPowerSlack{NZ,NZM1,IDX}

GPU-compatible functor for slack computation.
"""
struct EuclidianPowerSlack{NZ,NZM1,IDX}
    idx::IDX
end

EuclidianPowerSlack(::Val{NZ}, ::Val{NZM1}, idx::IDX) where {NZ,NZM1,IDX} =
    EuclidianPowerSlack{NZ,NZM1,IDX}(idx)

@inline function (b::EuclidianPowerSlack{NZ,NZM1,IDX})(
        A_row::SVector{NZ2,TT}, b_row::SVector{NZ,TT},
        p_val, mu_val, y::SVector{N,TT}) where {NZ,NZ2,NZM1,N,TT,IDX}
    Ax = reshape(A_row, Size(NZ, NZ))
    bx = b_row
    p0 = TT(p_val)

    _, q, s = _ep_get_z_and_parts(Ax, bx, b.idx, y, Val(NZM1))
    q_sq = normsquared(q)
    -min(s - _safe_pow(q_sq, p0 / TT(2)), s)
end

@doc raw"""
    convex_Euclidian_power(::Type{T}=Float64; geometry, idx=Colon(), A=(x)->I, b=(x)->T(0), p=x->T(2), ...)

Create a convex set defined by Euclidean norm power constraints, with GPU support.

Constructs a `Vector{Convex{T}}` (one per multigrid level) representing the power cone:
`{y : s ≥ ‖q‖₂^p}` where `[q; s] = A(x)*y[idx] + b(x)`

This is the fundamental constraint for p-Laplace problems where we need
`s ≥ ‖∇u‖^p` for some scalar field u.

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `mg::MultiGrid`: Required. The multigrid hierarchy (provides grid and coarsen operators).
- `idx=Colon()`: Indices of y to which transformation applies
- `A::Function`: Matrix function `x -> A(x)` for linear transformation
- `b::Function`: Vector function `x -> b(x)` for affine shift
- `p::Function`: Exponent function `x -> p(x)` where p(x) ≥ 1
- `A_grid`, `b_grid`, `p_grid`: Optional pre-computed grids (computed from A,b,p if not provided)

# Returns
`Vector{Convex{T}}` with one Convex per multigrid level. Each level's barriers
capture their level's pre-computed parameter grids (`A_l`, `b_l`, `p_l`) and
receive `(j::Integer, y::SVector)` where `j` is the vertex index.

# Mathematical Details
The barrier function is:
- For p = 2: `-log(s² - ‖q‖²)`
- For p ≠ 2: `-log(s^(2/p) - ‖q‖²) - μ(p)*log(s)`
  where μ(p) = 0 if p∈{1,2}, 1 if p<2, 2 if p>2

# Examples
```julia
# Standard p-Laplace constraint with GPU support
mg = amg(fem1d(Float32; nodes=collect(range(-1f0, 1f0, length=33))))
Q = convex_Euclidian_power(Float32; mg=mg, idx=default_idx(1), p=x->1.5f0)

# Q is now Vector{Convex{Float32}} with one per level
# Each Q[l] has barriers that capture level-l pre-computed arrays
```
"""
function convex_Euclidian_power(::Type{T}=Float64;
        mg::MultiGrid,
        idx::GPUIndex=Colon(),
        A::Function=(x)->I,
        b::Function=(x)->T(0),
        p::Function=x->T(2),
        # Pre-computed grids (computed from functions if not provided)
        A_grid = nothing,
        b_grid = nothing,
        p_grid = nothing) where {T}

    L = length(mg.coarsen[:dirichlet])
    x_fine = mg.x

    # Helper to determine dimensions from idx
    # For idx=Colon(), we need the full dimension which we get from first evaluation
    # For idx=SVector{M,Int}, M is the dimension
    nz = if idx isa Colon
        # Evaluate A once to get dimensions
        # Use _to_cpu_array to avoid scalar indexing on GPU arrays
        x_cpu = _to_cpu_array(x_fine)
        A_sample = A(x_cpu isa AbstractMatrix ? x_cpu[1,:] : x_cpu[1])
        if A_sample isa UniformScaling
            error("For idx=Colon() with UniformScaling A, cannot determine dimensions. Use explicit SVector idx.")
        end
        size(A_sample, 1)
    else
        length(idx)
    end

    # Pre-compute grids at fine level if not provided
    # These use CPU map_rows to handle arbitrary closures
    if A_grid === nothing
        A_grid = multigrid_from_fine_grid(mg,
            map_rows(xi -> begin
                Ax = A(xi)
                if Ax isa UniformScaling
                    SVector{nz*nz,T}(vec(one(SMatrix{nz,nz,T})))
                else
                    SVector{nz*nz,T}(vec(Ax))
                end
            end, x_fine))
    end

    if b_grid === nothing
        b_grid = multigrid_from_fine_grid(mg,
            map_rows(xi -> begin
                bx = b(xi)
                if bx isa Number
                    SVector{nz,T}(ntuple(i -> i == nz ? T(bx) : zero(T), Val(nz)))
                else
                    SVector{nz,T}(bx)
                end
            end, x_fine))
    end

    if p_grid === nothing
        p_grid = multigrid_from_fine_grid(mg,
            map_rows(xi -> T(p(xi)), x_fine))
    end

    # Pre-compute mu grid on CPU (eliminates conditional in GPU barrier)
    # mu = 0 for p=1 or p=2, mu = 1 for p<2, mu = 2 for p>2
    mu_func(p0) = (p0 == 2 || p0 == 1) ? T(0) : (p0 < 2 ? T(1) : T(2))
    mu_grid = multigrid_from_fine_grid(mg,
        map_rows(p_val -> mu_func(T(p_val)), p_grid[L]))

    # Compile-time constant for static pop operations
    nz_m1_val = Val(nz - 1)

    # Core gradient w.r.t. (q, s) - GPU compatible, receives μ as parameter
    @inline function core_grad(q::SVector{NQ,TT}, s::TT, p0::TT, μ::TT) where {NQ,TT}
        α = TT(2) / p0
        q_sq = normsquared(q)
        s_α = _safe_pow(s, α)
        r = s_α - q_sq
        inv_r = one(TT) / r
        grad_q = (TT(2) * inv_r) .* q
        s_α_m1 = _safe_pow(s, α - one(TT))
        grad_s = -α * s_α_m1 * inv_r - μ / s
        return push(grad_q, grad_s)
    end

    # Core Hessian - GPU compatible, receives μ as parameter
    @inline function core_hess(q::SVector{NQ,TT}, s::TT, p0::TT, μ::TT) where {NQ,TT}
        α = TT(2) / p0
        q_sq = normsquared(q)
        s_α = _safe_pow(s, α)
        r = s_α - q_sq
        inv_r = one(TT) / r
        inv_r2 = inv_r * inv_r
        s_α_m1 = _safe_pow(s, α - one(TT))
        coef_qs = -TT(2) * α * s_α_m1 * inv_r2
        s_α_m2 = _safe_pow(s, α - TT(2))
        s_2α_m2 = _safe_pow(s, TT(2) * α - TT(2))
        H_ss = -α * (α - one(TT)) * s_α_m2 * inv_r + α * α * s_2α_m2 * inv_r2 + μ / (s * s)

        nz_local = NQ + 1
        H = SMatrix{nz_local, nz_local, TT}(ntuple(Val(nz_local * nz_local)) do k
            i = (k - 1) % nz_local + 1
            j = (k - 1) ÷ nz_local + 1
            result = zero(TT)
            if i <= NQ && j <= NQ
                result = TT(4) * q[i] * q[j] * inv_r2
                if i == j
                    result += TT(2) * inv_r
                end
            elseif i <= NQ && j == nz_local
                result = coef_qs * q[i]
            elseif i == nz_local && j <= NQ
                result = coef_qs * q[j]
            else
                result = H_ss
            end
            result
        end)
        return SVector(H)
    end

    # Build Convex for each level using GPU-compatible functors
    # Functors encode nz and idx as type parameters for GPU compilation
    Q = Vector{Convex{T}}(undef, L)

    # Create functor instances (shared across all levels)
    nz_val = Val(nz)
    barrier_f0 = EuclidianPowerBarrier(nz_val, nz_m1_val, idx)
    barrier_f1 = EuclidianPowerBarrierGrad(nz_val, nz_m1_val, idx, core_grad)
    barrier_f2 = EuclidianPowerBarrierHess(nz_val, nz_m1_val, idx, core_hess)
    cobarrier_f0 = EuclidianPowerCobarrier(nz_val, nz_m1_val, idx)
    cobarrier_f1 = EuclidianPowerCobarrierGrad(nz_val, nz_m1_val, idx, core_grad)
    cobarrier_f2 = EuclidianPowerCobarrierHess(nz_val, nz_m1_val, idx, core_hess)
    slack_f = EuclidianPowerSlack(nz_val, nz_m1_val, idx)

    for l = 1:L
        # Get this level's grids - stored in Convex.args, passed to map_rows_gpu
        # Keep as MPI wrappers so MPI dispatch fires; _rows_to_svectors extracts raw arrays
        A_l = A_grid[l]
        b_l = b_grid[l]
        p_l = p_grid[l]
        mu_l = mu_grid[l]  # Pre-computed μ values (no conditionals needed on GPU)

        Q[l] = Convex{T}(
            (barrier_f0, barrier_f1, barrier_f2),
            (cobarrier_f0, cobarrier_f1, cobarrier_f2),
            slack_f,
            (A_l, b_l, p_l, mu_l)  # args tuple - includes pre-computed μ values
        )
    end

    return Q
end

@doc raw"""
    convex_piecewise(::Type{T}=Float64; Q::Tuple{Vararg{Vector{Convex{T}}}}, geometry, select::Function=x->(true,...)) where {T}

Build a `Vector{Convex{T}}` (one per level) that combines multiple convex domains with spatial selectivity.

# Arguments
- `Q::Tuple{Vararg{Vector{Convex{T}}}}`: tuple of convex piece vectors. Each element is a `Vector{Convex{T}}` of length L (one per level).
- `mg::MultiGrid`: multigrid hierarchy with coarsen operators (determines L levels).
- `select::Function`: a function `x -> Tuple{Bool,...}` indicating which pieces are active at spatial position `x`.
- `select_grid`: (optional) pre-computed selection grid for each level. If not provided, computed from `select` function.

# Semantics
For each level l, the resulting `Convex` has:
- `barrier(j, y) = ∑(Q[k][l].barrier(j, y) for k where sel_l[j][k])`
- `cobarrier(j, yhat) = ∑(Q[k][l].cobarrier(j, yhat) for k where sel_l[j][k])`
- `slack(j, y) = max(Q[k][l].slack(j, y) for k where sel_l[j][k])`

The slack is the maximum over active pieces, ensuring a single slack value suffices for feasibility.

# Examples
```julia
# Intersection of two convex domains
U = convex_Euclidian_power(Float64; mg=mg, idx=SVector(1, 3), p=x->2)
V = convex_linear(Float64; mg=mg, A=x->A_matrix, b=x->b_vector)
select_both(x) = (true, true)
Qint = convex_piecewise(Float64; Q=(U, V), mg=mg, select=select_both)

# Region-dependent constraints
Q_left = convex_Euclidian_power(Float64; mg=mg, p=x->1.5)
Q_right = convex_Euclidian_power(Float64; mg=mg, p=x->2.0)
select(x) = (x[1] < 0, x[1] >= 0)
Qreg = convex_piecewise(Float64; Q=(Q_left, Q_right), mg=mg, select=select)
```

See also: [`intersect`](@ref), [`convex_linear`](@ref), [`convex_Euclidian_power`](@ref).
"""
function convex_piecewise(::Type{T}=Float64;
        Q::Tuple{Vararg{Vector{Convex{T}}}},
        mg::MultiGrid,
        select::Function = x -> ntuple(_ -> true, length(Q)),
        select_grid = nothing) where {T}

    n = length(Q)  # Number of pieces
    L = length(mg.coarsen[:dirichlet])  # Number of levels
    x_fine = mg.x

    # Pre-compute select_grid if not provided
    if select_grid === nothing
        # select_grid[l] is an N_l × n matrix indicating which pieces are active
        # Use T instead of Bool for MPI compatibility (coarsen matrices expect matching element types)
        select_grid_fine = map_rows(xi -> SVector{n,T}(T.(select(xi))), x_fine)
        select_grid = multigrid_from_fine_grid(mg, select_grid_fine)
    end

    # Build combined Convex for each level
    Q_combined = Vector{Convex{T}}(undef, L)

    for l = 1:L
        # Get selection grid for this level - keep MPI wrapper for proper dispatch
        sel_l = select_grid[l]

        # Extract all barrier functions at level l into tuples
        barrier_f0s = ntuple(k -> Q[k][l].barrier[1], Val(n))
        barrier_f1s = ntuple(k -> Q[k][l].barrier[2], Val(n))
        barrier_f2s = ntuple(k -> Q[k][l].barrier[3], Val(n))
        cobarrier_f0s = ntuple(k -> Q[k][l].cobarrier[1], Val(n))
        cobarrier_f1s = ntuple(k -> Q[k][l].cobarrier[2], Val(n))
        cobarrier_f2s = ntuple(k -> Q[k][l].cobarrier[3], Val(n))
        slack_fns = ntuple(k -> Q[k][l].slack, Val(n))

        # Collect args from all pieces at this level
        # Each piece's args is a tuple; concatenate them all
        piece_args = ntuple(k -> Q[k][l].args, Val(n))

        # Compute cumulative arg lengths for slicing
        # arg_lengths[k] = number of args for piece k
        arg_lengths = map(length, piece_args)

        # Compute start indices for each piece's args (1-based, after sel)
        # sel is at position 1, so piece args start at position 2
        # Piece 1: starts at 2
        # Piece 2: starts at 2 + arg_lengths[1]
        # Piece k: starts at 2 + sum(arg_lengths[1:k-1])
        arg_starts = ntuple(Val(n)) do k
            2 + sum(arg_lengths[1:k-1]; init=0)
        end
        arg_ends = ntuple(Val(n)) do k
            arg_starts[k] + arg_lengths[k] - 1
        end

        # Store ranges as tuples of Val for compile-time slicing
        arg_ranges_val = ntuple(Val(n)) do k
            (Val(arg_starts[k]), Val(arg_ends[k]))
        end

        # Flatten all args into combined_args tuple
        all_args_flat = reduce((a, b) -> (a..., b...), piece_args; init=())
        combined_args = (sel_l, all_args_flat...)

        # Combined barrier functions receive row data via broadcasting
        # Signature: (sel_row, piece1_args_rows..., piece2_args_rows..., ..., y)
        # Note: sel_row contains T values (not Bool) for MPI compatibility; use !iszero for tests
        function barrier_f0_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            y = all_rows_and_y[M]
            TT = eltype(y)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(barrier_f0s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    zero(TT)
                end
            end
            sum(vals)
        end

        function barrier_f1_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            y = all_rows_and_y[M]
            NY = length(y)
            TT = eltype(y)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(barrier_f1s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    SVector(ntuple(i -> zero(TT), Val(NY)))
                end
            end
            reduce(+, vals)
        end

        function barrier_f2_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            y = all_rows_and_y[M]
            NY = length(y)
            TT = eltype(y)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(barrier_f2s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    SVector(ntuple(i -> zero(TT), Val(NY*NY)))
                end
            end
            reduce(+, vals)
        end

        function cobarrier_f0_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            yhat = all_rows_and_y[M]
            TT = eltype(yhat)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(cobarrier_f0s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    zero(TT)
                end
            end
            sum(vals)
        end

        function cobarrier_f1_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            yhat = all_rows_and_y[M]
            NY = length(yhat)
            TT = eltype(yhat)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(cobarrier_f1s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    SVector(ntuple(i -> zero(TT), Val(NY)))
                end
            end
            reduce(+, vals)
        end

        function cobarrier_f2_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            yhat = all_rows_and_y[M]
            NY = length(yhat)
            TT = eltype(yhat)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(cobarrier_f2s[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    SVector(ntuple(i -> zero(TT), Val(NY*NY)))
                end
            end
            reduce(+, vals)
        end

        function slack_l(all_rows_and_y::Vararg{Any,M}) where M
            sel_row = all_rows_and_y[1]
            y = all_rows_and_y[M]
            TT = eltype(y)
            vals = ntuple(Val(n)) do k
                if !iszero(sel_row[k])
                    _call_piece_barrier(slack_fns[k], all_rows_and_y, arg_ranges_val[k]...)
                else
                    typemin(TT)
                end
            end
            maximum(vals)
        end

        Q_combined[l] = Convex{T}(
            (barrier_f0_l, barrier_f1_l, barrier_f2_l),
            (cobarrier_f0_l, cobarrier_f1_l, cobarrier_f2_l),
            slack_l,
            combined_args  # Combined args tuple - splatted to map_rows_gpu
        )
    end

    return Q_combined
end

@doc raw"""
    intersect(mg::MultiGrid, U::Vector{Convex{T}}, rest...) where {T}

Return the intersection of convex domain vectors as a single `Vector{Convex{T}}`.
Equivalent to `convex_piecewise` with all pieces active at all vertices.
"""
function intersect(mg::MultiGrid, U::Vector{<:Convex{T}}, rest::Vector{<:Convex{T}}...) where {T}
    pieces = (U, rest...)
    n = length(pieces)
    # All pieces always active
    select_all(x) = ntuple(_ -> true, Val(n))
    convex_piecewise(T; Q=pieces, mg=mg, select=select_all)
end

@doc raw"""    apply_D(D,z) = hcat([D[k]*z for k in 1:length(D)]...)"""
apply_D(D,z) = hcat([D[k]*z for k in 1:length(D)]...)

"""
    barrier(Q::Convex{T}) -> Barrier

Create a Barrier from a Convex constraint specification.

The Convex's barrier functions receive row data via broadcasting:
`F0(args_rows..., y)` where `args_rows` are per-vertex parameter data
(from Q.args) and `y` is the solution SVector at that vertex.

This enables true GPU execution without scalar indexing - Q.args
are splatted to map_rows_gpu which broadcasts them together.
"""
function barrier(Q::Convex{T})::Barrier where T
    (F0, F1, F2) = Q.barrier
    args = Q.args

    function f0(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        # Splat Q.args to map_rows_gpu - barriers receive (args_rows..., y)
        y = map_rows_gpu(F0, args..., Dz)
        # GPU-compatible: avoid inline closure by splitting computation
        result = dot(w, y) + sum(w .* map_rows_gpu(dot, c, Dz))
        result
    end

    function f1(z::W, w::W, c, R, D, z0) where {W}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        # Splat Q.args to map_rows_gpu
        grad_barrier = map_rows_gpu(F1, args..., Dz)
        y = map_rows_gpu(+, grad_barrier, c)  # + is a pure function
        ret = 0
        for k = 1:n
            foo = D[k]' * (w .* y[:, k])
            if k > 1
                ret += foo
            else
                ret = foo
            end
        end
        R' * ret
    end

    function f2(z::W, w::W, c, R::Mat, D, z0) where {W, Mat}
        Dz = apply_D(D, z0 + R * z)
        n = length(D)
        # Splat Q.args to map_rows_gpu
        y = map_rows_gpu(F2, args..., Dz)
        ret = D[1]
        for j = 1:n
            foo = amgb_diag(D[1], w .* y[:, (j - 1) * n + j])
            bar = (D[j])' * foo * D[j]
            if j > 1
                ret += bar
            else
                ret = bar
            end
            for k = 1:j-1
                foo = amgb_diag(D[1], w .* y[:, (j - 1) * n + k])
                ret += D[j]' * foo * D[k] + D[k]' * foo * D[j]
            end
        end
        R' * ret * R
    end

    Barrier(; f0, f1, f2)
end
function divide_and_conquer(eta,j,J)
    if eta(j,J) return true end
    jmid = (j+J)÷2
    if jmid==j || jmid==J return false end
    return divide_and_conquer(eta,j,jmid) && divide_and_conquer(eta,jmid,J)
end
function amgb_coarsen_levels(M::AMG, z, c)
    L = length(M.R_fine)
    zm = Vector{typeof(z)}(undef,L); zm[L] = z
    cm = Vector{typeof(c)}(undef,L); cm[L] = c
    wm = Vector{typeof(M.w)}(undef,L); wm[L] = M.w
    for l=L-1:-1:1
        cm[l] = M.coarsen_u[l]*cm[l+1]
        zm[l] = M.coarsen_z[l]*zm[l+1]
        wm[l] = M.refine_u[l]'*wm[l+1]
    end
    (zm,cm,wm)
end
function mgb_phase1(Q::Vector{<:Convex{T}},
        M::AMG{X,W,M_sub,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
        z::W,
        c::X;
        maxit,
        max_newton,
        stopping_criterion,
        line_search,
        printlog,
        args...
        ) where {T,X,W,M_sub}
    @debug("start")
    L = length(M.R_fine)
    zm, cm, wm = amgb_coarsen_levels(M, z, c)
    passed = falses((L,))
    its = zeros(Int,(L,))
    function zeta(j,J)
        @debug("j=",j," J=",J)
        # Create barrier for level J from Q[J]
        B_J = barrier(Q[J])
        (f0,f1,f2) = (B_J.f0, B_J.f1, B_J.f2)
        w = wm[J]
        R = M.R_coarse[J]
        D = J == L ? M.D_fine : M.D_levels[J,:]
        z0 = zm[J]
        c0 = cm[J]
        s0 = amgb_zeros(W, size(R, 2))
        # AMG coarsening does not preserve feasibility; if z0 is infeasible
        # at level J, skip the level (phase 1 is best-effort).
        y_check = try f0(s0,w,c0,R,D,z0)::T catch; T(NaN) end
        isfinite(y_check) || return false
        mi = J-j==1 ? maxit : max_newton
        SOL = newton(M_sub,T,
                s->f0(s,w,c0,R,D,z0),
                s->f1(s,w,c0,R,D,z0),
                s->f2(s,w,c0,R,D,z0),
                s0,
                maxit=mi,
                stopping_criterion=stopping_criterion,
                ;line_search,printlog)
        SOL.converged || return false
        znext = copy(zm)
        s = R*SOL.x
        znext[J] = zm[J]+s
        try
            for k=J+1:L
                s = M.refine_z[k-1]*s
                znext[k] = zm[k]+s
                # Create barrier for level k
                B_k = barrier(Q[k])
                s0 = amgb_zeros(W,size(M.R_coarse[k])[2])
                D_k = k == L ? M.D_fine : M.D_levels[k,:]
                y0 = B_k.f0(s0,wm[k],cm[k],M.R_coarse[k],D_k,znext[k])::T
                y1 = B_k.f1(s0,wm[k],cm[k],M.R_coarse[k],D_k,znext[k])
                @assert isfinite(y0) && amgb_all_isfinite(y1)
            end
            zm = znext
            passed[J] = true
        catch
        end
        return true
    end
    divide_and_conquer(zeta,0,L)
    # Best-effort: zm[L] is at worst the input z (still fine-grid feasible);
    # the main phase will absorb any lack of centring.
    (;z=zm[L],its,passed)
end
function mgb_step(Q::Vector{<:Convex{T}},
        M::AMG{X,W,M_sub,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
        z::W,
        c::X;
        early_stop,
        maxit,
        max_newton,
        line_search,
        stopping_criterion,
        finalize,
        printlog,
        args...
        ) where {T,X,W,M_sub}
    L = length(M.R_fine)
    # Use finest level barrier for main optimization
    B = barrier(Q[L])
    (f0,f1,f2) = (B.f0,B.f1,B.f2)
    its = zeros(Int,(L,))
    w = M.w
    D = M.D_fine
    function eta(j,J,sc,maxit,ls)
        @debug("j=",j," J=",J)
        if early_stop(z) return true end
        R = M.R_fine[J]
        s0 = amgb_zeros(W, size(R, 2))
        SOL = newton(M_sub,T,
            s->f0(s,w,c,R,D,z),
            s->f1(s,w,c,R,D,z),
            s->f2(s,w,c,R,D,z),
            s0,
            ;maxit,
            stopping_criterion=sc,
            line_search=ls,
            printlog)
        its[J] += SOL.k
        if SOL.converged
            z = z + R*SOL.x
        end
        return SOL.converged
    end
    converged = divide_and_conquer((j,J)->eta(j,J,stopping_criterion,max_newton,line_search),0,L)
    z_unfinalized = z
    if finalize!=false
        @debug("finalize")
        foo = eta(L-1,L,finalize,maxit,line_search)
        converged = converged && foo
    end
    @debug("converged=",converged)
    return (;z,z_unfinalized,its,converged)
end

function illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}
    @assert isfinite(fa) && isfinite(fb)
    if fa==0
        return a
    end
    if fa*fb>=0
        return b
    end
    for k=1:maxit
        c = (a*fb-b*fa)/(fb-fa)
        fc = f(c)
        @assert isfinite(fc)
        if c<=min(a,b) || c>=max(a,b) || fc*fa==0 || fc*fb==0
            return c
        end
        if fb*fc<0
            a,fa = b,fb
        else
            fa /= 2
        end
        b,fb = c,fc
    end
    throw("Illinois solver failed to converge.")
end

raw"""
    linesearch_illinois(::Type{T}=Float64; beta=T(0.5)) where {T}

Create an Illinois-based line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction when Illinois fails (default: 0.5).

# Returns
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : Newton direction (typically H\g where H is the Hessian).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
The Illinois algorithm finds a root of `φ(s) = ⟨∇F(x - s*n), n⟩`, which corresponds to
the exact line search condition. If the Illinois solver fails or encounters numerical
issues, the step size is reduced by factor `beta` and the process repeats.

# Notes
This line search strategy aims for the exact minimizer along the search direction,
making it potentially more aggressive than backtracking but also more expensive per iteration.
"""
function linesearch_illinois(::Type{T}=Float64;beta=T(0.5)) where {T}
    function ls_illinois(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
        s = T(1)
        test_s = true
        xnext = x
        ynext = y
        gnext = g
        inc = dot(g,n)
        while s>T(0) && test_s
            @debug("s=",s)
            try
                function phi(s)
                    xn = x-s*n
                    @assert(isfinite(F0(xn)))
                    return dot(F1(xn),n)
                end
                s = illinois(phi,T(0),s,fa=inc)
                xnext = x-s*n
                # GPU-compatible: use norm to check if step made any difference
                test_s = norm(xnext - x) > 0
                ynext,gnext = F0(xnext)::T, F1(xnext)
                @assert isfinite(ynext) && amgb_all_isfinite(gnext)
                break
            catch e
                @debug(e.msg)
            end
            s = s*beta
        end
        return (xnext,ynext,gnext)
    end
    return ls_illinois
end

raw"""
    linesearch_backtracking(::Type{T}=Float64; beta=T(0.5)) where {T}

Create a backtracking line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction (default: 0.5).

# Returns
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : search direction (typically Newton direction H\g).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
Implements the Armijo backtracking line search with sufficient decrease condition:
`F(x - s*n) ≤ F(x) - c₁ * s * ⟨∇F(x), n⟩` where `c₁ = 0.1`.
The step size starts at `s = 1` and is reduced by factor `beta` until the condition
is satisfied or numerical limits are reached.

# Notes
This is a robust and commonly used line search that guarantees sufficient decrease
in the objective function, making it suitable for general nonlinear optimization.
"""
function linesearch_backtracking(::Type{T}=Float64;beta = T(0.5)) where {T}
    function ls_backtracking(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
        s = T(1)
        test_s = true
        xnext = x
        ynext = y
        gnext = g
        inc = dot(g,n)
        while s>T(0) && test_s
            @debug("s=",s)
            try
                xnext = x-s*n
                # GPU-compatible: use norm to check if step made any difference
                test_s = norm(xnext - x) > 0
                ynext,gnext = F0(xnext)::T, F1(xnext)
                @assert isfinite(ynext) && amgb_all_isfinite(gnext)
                if ynext <= y - T(0.1)*inc*s
                    break
                end
            catch e
                @debug(e.msg)
            end
            s = s*beta
        end
        return (xnext,ynext,gnext)
    end
    return ls_backtracking
end

"""
    stopping_exact(theta::T) where {T}

Create an exact stopping criterion for Newton methods.

# Arguments
* `theta` : tolerance parameter for gradient norm relative decrease (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement.

# Algorithm
Returns `true` (stop) if both conditions hold:
1. No objective improvement: `ynext ≥ ymin`
2. Gradient norm stagnation: `‖gnext‖ ≥ theta * gmin`

# Notes
This criterion is "exact" in the sense that it requires both objective and gradient
stagnation before stopping, making it suitable for high-precision optimization.
Typical values of `theta` are in the range [0.1, 0.9].
"""
stopping_exact(theta::T) where {T} = (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->ynext>=ymin && norm(gnext)>=theta*gmin
"""
    stopping_inexact(lambda_tol::T, theta::T) where {T}

Create an inexact stopping criterion for Newton methods that combines Newton decrement
and exact stopping conditions.

# Arguments
* `lambda_tol` : tolerance for the Newton decrement (type T).
* `theta` : tolerance parameter for the exact stopping criterion (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement (√(gᵀH⁻¹g)).

# Algorithm
Returns `true` (stop) if either condition holds:
1. Newton decrement condition: `ndec < lambda_tol`
2. Exact stopping condition: `stopping_exact(theta)` is satisfied

# Notes
This criterion is "inexact" because it allows early termination based on the Newton
decrement, which provides a quadratic convergence estimate. The Newton decrement
`λ = √(gᵀH⁻¹g)` approximates the distance to the optimum in the Newton metric.
Typical values: `lambda_tol ∈ [1e-6, 1e-3]`, `theta ∈ [0.1, 0.9]`.
"""
function stopping_inexact(lambda_tol::T,theta::T) where {T} 
    exact_stop = stopping_exact(theta)
    (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->((ndec<lambda_tol || exact_stop(ymin,ynext,gmin,gnext,n,ndecmin,ndec)))
end

function newton(::Type{Mat}, ::Type{T},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::V;
                       maxit=10000,
                       stopping_criterion=nothing,
                       printlog,
                       line_search=nothing,
        ) where {T,Mat,V}
    stopping_criterion = stopping_criterion === nothing ? stopping_exact(T(0.1)) : stopping_criterion
    line_search = line_search === nothing ? linesearch_illinois(T) : line_search
    ss = T[]
    ys = T[]
    @assert amgb_all_isfinite(x)
    y = F0(x) ::T
    @assert isfinite(y)
    ymin = y
    push!(ys,y)
    converged = false
    k = 0
    g = F1(x)
    @assert amgb_all_isfinite(g)
    ynext,xnext,gnext=y,x,g
    gmin = norm(g)
    incmin = T(Inf)
    while k<maxit && !converged
        k+=1
        H = F2(x) ::Mat
        n = solve(symmetric(H), g)
        @assert amgb_all_isfinite(n)
        inc = dot(g,n)
        @debug("k=",k," y=",y," ‖g‖=",norm(g), " λ^2=",inc)
        if inc<=0
            converged = true
            break
        end
        (xnext,ynext,gnext) = line_search(x,y,g,n,F0,F1;printlog)
        if stopping_criterion(ymin,ynext,gmin,gnext,n,sqrt(incmin),sqrt(inc)) #ynext>=ymin && norm(gnext)>=theta*norm(g)
            @debug("converged: ymin=",ymin," ynext=",ynext," ‖gnext‖=",norm(gnext)," λ=",sqrt(inc)," λmin=",sqrt(incmin))
            converged = true
        end
        x,y,g = xnext,ynext,gnext
        gmin = min(gmin,norm(g))
        ymin = min(ymin,y)
        incmin = min(inc,incmin)
        push!(ys,y)
    end
    if !converged
        @debug("diverge")
    end
    return (;x,y,k,converged,ys)
end

function mgb_core(Q::Vector{<:Convex{T}},
        M::AMG{X,W,M_sub,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
        z::W,
        c::X;
        tol=sqrt(eps(T)),
        t=T(0.1),
        maxit=10000,
        kappa=T(10.0),
        early_stop=z->false,
        progress=x->nothing,
#        c0=T(0),
        max_newton= Int(ceil((log2(-log2(eps(T))))+2)),
        printlog,
        finalize,
        args...) where {T,X,W,M_sub}
    t_begin = time()
    tinit = t
    kappa0 = kappa
    L = length(M.R_fine)
    its = zeros(Int,(L,maxit))
    ts = zeros(T,(maxit,))
    kappas = zeros(T,(maxit,))
    times = zeros(Float64,(maxit,))
    c_dot_Dz = zeros(T,(maxit,))
    k = 1
    times[k] = time()
    SOL = mgb_phase1(Q,M,z,t*c;maxit,max_newton,printlog,args...)
    @debug("phase 1 success")
    passed = SOL.passed
    its[:,k] = SOL.its
    kappas[k] = kappa
    ts[k] = t
    z = SOL.z
    z_unfinalized = z
    Dz = apply_D(M.D_fine, z)
    c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:length(M.D_fine)])
    while t<=1/tol && kappa > 1 && k<maxit && !early_stop(z)
        k = k+1
        its[:,k] .= 0
        times[k] = time()
        prog = ((log(t)-log(tinit))/(log(1/tol)-log(tinit)))
        progress(prog)
        while kappa > 1
            t1 = kappa*t
            @debug("k=",k," t=",t," kappa=",kappa," t1=",t1)
            fin = (t1>1/tol) ? finalize : false
            SOL = mgb_step(Q,M,z,t1*c;
                max_newton,early_stop,maxit,printlog,finalize=fin,args...)
            its[:,k] += SOL.its
            if SOL.converged
                if maximum(SOL.its)<=max_newton*0.5
                    @debug("increasing t step size?")
                    kappa = min(kappa0,kappa^2)
                end
                z = SOL.z
                z_unfinalized = SOL.z_unfinalized
                t = t1
                break
            end
            @debug("t refinement failed, shrinking kappa")
            kappa = sqrt(kappa)
        end
        ts[k] = t
        kappas[k] = kappa
        #c_dot_Dz[k] = dot(M.w .* c, apply_D(M.D_fine, z))
        Dz = apply_D(M.D_fine, z)
        c_dot_Dz[k] = sum([dot(M.w .* c[:,k], Dz[:,k]) for k=1:length(M.D_fine)])
    end
    converged = (t>1/tol) || early_stop(z)
    if !converged
        throw(AMGBConvergenceFailure("Convergence failure in mgb_solve at t=$t, k=$k, kappa=$kappa, tol=$tol, maxit=$maxit."))
    end
    t_end = time()
    t_elapsed = t_end-t_begin
    progress(1.0)
    @debug("success. t=",t," tol=",tol)
    return (;z,z_unfinalized,c,its=its[:,1:k],ts=ts[1:k],kappas=kappas[1:k],
            t_begin,t_end,t_elapsed,times=times[1:k],
            passed,c_dot_Dz=c_dot_Dz[1:k])
end

function mgb_driver(M::Tuple{AMG{X,W,M_sub,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},AMG{X,W,M_sub,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}},
              f::X,
              g::X,
              Q::Vector{<:Convex{T}};
              t=T(0.1),
              t_feasibility=t,
              progress = x->nothing,
              stopping_criterion=stopping_inexact(sqrt(minimum(M[1].w))/2,T(0.5)),
              printlog = (args...)->nothing,
              line_search=linesearch_backtracking(T),
              finalize=stopping_exact(T(0.5)),
              rest...) where {T,X,W,M_sub}
    D0 = M[1].D_fine[1]
    m = size(M[1].x,1)
    ns = Int(size(D0,2)/m)
    nD = length(M[1].D_fine)
    L = length(Q)  # Number of multigrid levels
    c0 = f
    z0 = g
    Z = amgb_zeros(M[1].D_fine[1],m,m)
    # GPU-compatible: use `one` instead of closure capturing T
    ONES = map_rows_gpu(one, M[1].w)
    II = amgb_diag(M[1].D_fine[1],ONES)
    RR2 = []
    for k = 1:size(z0,2)
        foo = fill(Z,size(z0,2))
        foo[k] = II
        push!(RR2,hcat(foo...))
    end
    z2 = vcat([z0[:,k] for k=1:size(z0,2)]...)
    w = hcat([M[1].D_fine[k]*z2 for k=1:nD]...)
    pbarfeas = 0.0
    SOL_feasibility=nothing
    # Use finest level (L) for feasibility check
    Q_L = Q[L]
    barrier_f0_fn = Q_L.barrier[1]
    slack_fn = Q_L.slack
    cobar_f0 = Q_L.cobarrier[1]
    cobar_f1 = Q_L.cobarrier[2]
    cobar_f2 = Q_L.cobarrier[3]
    args_L = Q_L.args  # Args for finest level, splatted to map_rows_gpu
    try
        # GPU-compatible: splat Q.args to map_rows_gpu
        foo = map_rows_gpu(barrier_f0_fn, args_L..., w)
        @assert amgb_all_isfinite(foo)
    catch
        pbarfeas = 0.1
        # GPU-compatible: use slack_fn with args splatted
        z1 = map_rows_gpu((args_and_vecs...)->begin
            # Last two are z0_j and w_j, rest are args
            w_j = args_and_vecs[end]
            z0_j = args_and_vecs[end-1]
            args_j = args_and_vecs[1:end-2]
            push(z0_j, 2*max(slack_fn(args_j..., w_j), 1))
        end, args_L..., z0, w)
        b = 2*max(1,maximum(z1[:,size(z1,2)]))
        foo = zeros(T,(nD+1,)); foo[end] = 1
        foo_sv = SVector(Tuple(foo))
        c1 = map_rows_gpu(k->foo_sv, M[1].w)

        # Feasibility barrier: dot(y,y) + Q.cobarrier(args...,y) - log(b² - y[end]²)
        # GPU-compatible: barriers receive (args_rows..., yy)
        function feas_f0(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            u = yy[end]
            dot(yy, yy) + cobar_f0(args_j..., yy) - log(b^2 - u^2)
        end
        function feas_f1(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            N = length(yy)
            TT = eltype(yy)
            u = yy[end]
            denom = b^2 - u^2
            # GPU-compatible: use ntuple instead of zeros(MVector)
            g_extra = SVector(ntuple(i -> i == N ? TT(2) * u / denom : zero(TT), Val(N)))
            TT(2) .* yy .+ cobar_f1(args_j..., yy) .+ g_extra
        end
        function feas_f2(args_and_y::Vararg{Any,M}) where M
            yy = args_and_y[M]
            args_j = args_and_y[1:M-1]
            N = length(yy)
            TT = eltype(yy)
            u = yy[end]
            denom = b^2 - u^2
            # Get cobarrier Hessian (flattened)
            H_cobar_flat = cobar_f2(args_j..., yy)
            # H_extra has only (N,N) entry = 2*(b² + u²)/denom²
            H_extra_nn = TT(2) * (b^2 + u^2) / denom^2
            # GPU-compatible: build H directly with ntuple instead of zeros(MMatrix)
            H = SMatrix{N,N,TT}(ntuple(Val(N*N)) do k
                i = (k - 1) % N + 1
                jj = (k - 1) ÷ N + 1
                val = H_cobar_flat[k]  # cobarrier contribution
                if i == jj
                    val += TT(2)  # 2*I diagonal
                end
                if i == N && jj == N
                    val += H_extra_nn
                end
                val
            end)
            SVector(H)  # Flatten
        end
        # Wrap feasibility barrier in Vector{Convex} for compatibility
        # Use args from each level of Q
        Q_feas = [Convex{T}((feas_f0, feas_f1, feas_f2), (feas_f0, feas_f1, feas_f2), slack_fn, Q[l].args) for l in 1:L]

        z1 = vcat([z1[:,k] for k=1:size(z1,2)]...)
        foo = fill(Z,size(z0,2)+1)
        foo[end] = II
        WW = hcat(foo...)
        early_stop(z) = (maximum(WW*z)<0)
        try
            SOL_feasibility = mgb_core(Q_feas,M[2],z1,c1;t=t_feasibility,
                progress=x->progress(pbarfeas*x),
                early_stop,
                printlog,
                stopping_criterion,
                line_search,
                finalize,
                rest...)
            @assert early_stop(SOL_feasibility.z)
        catch e
            if isa(e,AMGBConvergenceFailure)
                throw(AMGBConvergenceFailure("Could not solve the feasibility subproblem, probem may be infeasible. Failure was: "*e.message))
            end
            throw(e)
        end
        # Extract main-problem components, dropping the feasibility slack
        z2 = SOL_feasibility.z[1:length(z2)]
    end
    SOL_main = mgb_core(Q,M[1],z2,c0;
        t,
        progress=x->progress((1-pbarfeas)*x+pbarfeas),
        printlog,
        stopping_criterion,
        line_search,
        finalize,
        rest...)
    z = hcat([RR2[k]*SOL_main.z for k=1:size(z0,2)]...)
    return (;z,SOL_feasibility,SOL_main)
end

# GPU-compatible default functions - return SVector, infer type from input x
default_f(T,::Val{1}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,::Val{2}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,::Val{3}) = (x)->SVector(oftype(x[1],0.5), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],0.0), oftype(x[1],1.0))
default_f(T,k::Int) = default_f(T,Val(k))
default_g(T,::Val{1}) = (x)->SVector(x[1], oftype(x[1],2))
default_g(T,::Val{2}) = (x)->SVector(x[1]^2+x[2]^2, oftype(x[1],100))
default_g(T,::Val{3}) = (x)->SVector(x[1]^2+x[2]^2+x[3]^2, oftype(x[1],100))
default_g(T,k::Int) = default_g(T,Val(k))
default_D(::Val{1}) = [:u :id
              :u :dx
              :s :id]
default_D(::Val{2}) = [:u :id
              :u :dx
              :u :dy
              :s :id]
default_D(::Val{3}) = [:u :id
              :u :dx
              :u :dy
              :u :dz
              :s :id]
default_D(k::Int) = default_D(Val(k))

# Static indices for GPU-compatible indexing: idx = 2:dim+2 as SVector
default_idx(::Val{1}) = SVector(2, 3)
default_idx(::Val{2}) = SVector(2, 3, 4)
default_idx(::Val{3}) = SVector(2, 3, 4, 5)
default_idx(k::Int) = default_idx(Val(k))

"""
    AMGBSOL{T,X,W,Discretization,G}

Solution object returned by `amgb` and the `*_solve` convenience functions.

# Type Parameters
- `T`: scalar numeric type
- `X`: solution/point matrix type (e.g. `Matrix{T}`, `CuMatrix{T}`)
- `W`: weight vector type
- `Discretization`: geometry descriptor (e.g. `FEM2D_P2{T}`, `SPECTRAL1D{T}`)
- `G`: full `Geometry` type

# Fields
- `z::X`: solution matrix of size `(n_nodes, n_components)`
- `SOL_feasibility`: feasibility phase diagnostics (`nothing` if initial point was feasible)
- `SOL_main`: main optimization phase diagnostics (NamedTuple)
- `log::String`: detailed iteration log
- `geometry::G`: the input `Geometry`

Supports `plot(sol)` to visualize the first solution component.
"""
struct AMGBSOL{T,X,W,Discretization,G}
    z::X
    SOL_feasibility
    SOL_main
    log::String
    geometry::G
end
function AMGBSOL(z::X, sf, sm, log::String, geometry::Geometry{T,<:Any,W,<:Any,<:Any,Discretization}) where {T,X,W,Discretization}
    AMGBSOL{T,X,W,Discretization,typeof(geometry)}(z, sf, sm, log, geometry)
end
plot(sol::AMGBSOL,k::Int=1;kwargs...) = plot(sol.geometry,sol.z[:,k];kwargs...)

"""
    mgb_solve(mg::MultiGrid; kwargs...) -> AMGBSOL

MultiGrid Barrier (MGB) solver for nonlinear convex optimization problems on a multigrid
hierarchy. Operates in a feasibility phase followed by a main optimization phase, with
damped Newton inner solves and line search.

# Keyword Arguments

## Problem Specification
- `dim::Integer = amg_dim(mg.discretization)`: spatial dimension; auto-detected.
- `state_variables = [:u :dirichlet; :s :full]`: solution components and their function spaces.
- `D = default_D(dim)`: differential operators to apply to state variables.
- `x = mg.x`: sample points where `f`/`g` are evaluated when given as functions.

## Problem Data
- `p::T = T(1.0)`: exponent for the p-Laplace term.
- `g`/`g_grid`, `f`/`f_grid`: boundary/initial data and forcing.
- `Q::Vector{Convex{T}}`: convex constraint specification (one per level); defaults to a
  p-Laplace power-cone barrier.

## Output Control
- `verbose::Bool = true`: progress bar.
- `logfile = devnull`: optional log stream.

## Solver Control (forwarded internally)
- `tol`, `t`, `t_feasibility`, `maxit`, `kappa`, `early_stop`, `max_newton`,
  `stopping_criterion`, `line_search`, `finalize`, `progress`, `printlog`.

# Returns
An `AMGBSOL` whose `z` is the fine-level solution matrix and whose `geometry` is the
`MultiGrid`'s fine-level `Geometry` (the `MultiGrid` itself is not stored).

# Examples
```julia
sol = mgb_solve(amg(fem1d(; nodes = collect(range(-1.0, 1.0, length=33)))); p = 1.5)
sol = mgb_solve(amg(subdivide(fem2d_P2(), 3)); p = 1.5)
sol = mgb_solve(amg(spectral2d(n = 8)); p = 2.0)
```
"""
function mgb_solve(mg::MultiGrid{T};
        dim::Integer = amg_dim(mg.discretization),
        state_variables = [:u :dirichlet ; :s :full],
        D = default_D(dim),
        M = _prepare_amg(mg;state_variables,D),
        x = mg.x,
        p::T = T(1.0),
        g::Function = default_g(T,dim),
        f::Function = default_f(T,dim),
        g_grid = map_rows(xi->SVector(Tuple(g(xi))),x),
        f_grid = map_rows(xi->SVector(Tuple(f(xi))),x),
        Q::Vector{Convex{T}} = convex_Euclidian_power(T; mg=mg, idx=default_idx(dim), p=xi->p),
        verbose=true,
        logfile=devnull,
        rest...) where {T}
    progress = x->nothing
    pbar = 0
    if verbose
        pbar = Progress(1000000; dt=1.0)
        finished = false
        function _progress(x)
            if !finished
                fooz = Int(floor(1000000*x))
                update!(pbar,fooz)
                if fooz==1000000
                    finished = true
                end
            end
        end
        progress = _progress
    end
    log_buffer = IOBuffer()
    function printlog(args...)
        println(log_buffer,args...)
        println(logfile,args...)
    end
    SOL = mgb_driver(M, f_grid, g_grid, Q; progress, printlog, rest...)
    return mgb_cleanup(AMGBSOL(SOL.z, SOL.SOL_feasibility, SOL.SOL_main, String(take!(log_buffer)), mg.geometry))
end

"""
    mgb_solve(; mg::MultiGrid, kwargs...) -> AMGBSOL

Keyword-only convenience method. Lets callers splat a `NamedTuple` produced by
`Zoo` problem constructors: `mgb_solve(; problem...)`.
"""
function mgb_solve(; mg::MultiGrid, kwargs...)
    mgb_solve(mg; kwargs...)
end

