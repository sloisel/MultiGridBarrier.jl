# Geometry, MultiGrid, the AMG hierarchy, and amg()/geometric_mg() assembly.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

"""
    Geometry{T,X<:AbstractArray{T,3},W,M_op,Discretization}

Single-level container for discretization geometry. Holds only the fine-level mesh and
operators — no multigrid hierarchy. Use `amg(geom)` to attach an algebraic-multigrid
hierarchy and return a `MultiGrid`. (The legacy `geometric_mg(geom, L)` builds a
geometric-subdivision hierarchy instead; new code should prefer `amg`.)

Type parameters
- `T`: scalar numeric type (e.g. `Float64`)
- `X<:AbstractArray{T,3}`: type of the mesh tensor `x` (typically `Array{T,3}`).
- `W`: type of the weight storage `w` (typically `Vector{T}`).
- `M_op`: matrix type for operators (e.g. `SparseMatrixCSC{T,Int}`, `BlockDiag{T}`).
- `Discretization`: front-end descriptor (e.g. `FEM1D{e,T}`, `FEM2D_P2{T}`, `SPECTRAL1D{T}`).

Fields
- `discretization::Discretization`: discretization descriptor encoding dimension and grid info.
- `t::Matrix{Int}`: cached element connectivity of shape `(V, N)` — `t[v, e]` is the
  global node id of local node `v` in element `e`, so broken nodes that coincide
  across elements share an id and `maximum(t)` is the number of distinct nodes.
  The 4-arg constructor derives `t` from `x` by geometric dedup; the inner 5-arg
  constructor (and `fem1d`/`fem2d`/`fem3d` with `t=`) take it verbatim, letting
  coincident-but-distinct nodes (slits, branch cuts, glued manifolds) stay
  topologically separate. The amg path (`find_boundary`, `amg`) reads `t` directly;
  coordinate dedup builds `t` only in the 4-arg constructor and the legacy
  `geometric_mg` path. Always a CPU `Matrix{Int}`, even for GPU geometries.
- `x::X`: mesh as a 3-tensor of shape `(V, N, D)` — `V` vertices per element,
  `N` elements, `D` spatial dimensions. Reshape-compatible with the legacy
  flat layout via `reshape(x, V*N, D)` (zero-copy). Spectral discretizations
  use `N = 1` (a single notional "element" comprising every Chebyshev node).
- `w::W`: quadrature weights matching the flattened node order (length `V*N`).
- `operators::Dict{Symbol,M_op}`: fine-level discrete operators (e.g. `:id`, `:dx`, `:dy`).
"""
struct Geometry{T,X<:AbstractArray{T,3},W,M_op,Discretization}
    discretization::Discretization
    t::Matrix{Int}
    x::X
    w::W
    operators::Dict{Symbol,M_op}
end

# Backward-compatible 4-arg constructor: derive the cached connectivity `t` from the
# broken mesh coordinates by geometric dedup, so existing call sites that pass
# `(discretization, x, w, operators)` keep working unchanged. `t[v,e]` is the global
# id of local node `v` in element `e` (`reshape(_dedupe(_xflat(x))[2], V, N)`). `t`
# is intentionally not a type parameter — it is always a CPU `Matrix{Int}`, even for
# GPU geometries (small, read only during setup). The 5-arg form (the auto-generated
# inner constructor) takes `t` explicitly and is used by the GPU converters.
function Geometry{T,X,W,M_op,D}(discretization::D, x::X, w::W,
        operators::Dict{Symbol,M_op}) where {T,X<:AbstractArray{T,3},W,M_op,D}
    V, N = size(x, 1), size(x, 2)
    _, labels = _dedupe(reshape(x, V * N, size(x, 3)))
    return Geometry{T,X,W,M_op,D}(discretization, reshape(labels, V, N), x, w, operators)
end

"""
    _xflat(x::AbstractArray{T,3}) -> AbstractMatrix{T}
    _xflat(geom::Geometry)        -> AbstractMatrix{T}
    _xflat(mg::MultiGrid)         -> AbstractMatrix{T}

Zero-copy view of the mesh tensor as a flat `(V*N, D)` matrix, matching the
legacy `geom.x` layout used by `map_rows`, sparse operators, etc.
"""
_xflat(x::AbstractArray{T,3}) where {T} = reshape(x, :, size(x, 3))
_xflat(g::Geometry)                     = _xflat(g.x)
_xflat(mg)                              = _xflat(mg.geometry)  # MultiGrid: forwards through .geometry

"""
    _pairs_to_linear(pairs::AbstractVector{<:Tuple{Int,Int}}, V::Int) -> Vector{Int}

Translate `(v, e)` index pairs (vertex within element, element index) into linear
indices into the flat `(V*N, D)` view of `geom.x` — i.e. `v + (e - 1) * V`.
Used by AMG hierarchy builders that work on the flat broken-basis layout.
"""
_pairs_to_linear(pairs::AbstractVector{<:Tuple{Int,Int}}, V::Int) =
    Int[v + (e - 1) * V for (v, e) in pairs]

"""
    _mask_dirichlet_rows(B, labels, dd_set) -> SparseMatrixCSC

Zero every row of the fine-level bridge/prolongation `B` (rows indexed in the
broken Q_k / P2+bubble basis) whose dedup label `labels[row]` is in `dd_set`.

The AMG hierarchy of a zero-trace subspace coarsens a P1/Q1 problem on the
element *corners* only; the bridge `B` then lifts interior-corner values to the
broken Q_k basis via the multilinear corner weights. That lift writes nonzero
values onto Dirichlet edge/face/centroid nodes whenever such a node lies on an
element facet with a free (non-Dirichlet) corner — and the fine level, whose
search space has no support at those nodes, cannot remove the leak. Masking the
bridge rows for the *full* Dirichlet set forces every coarse search-space
function to vanish at *all* Dirichlet DOFs, restoring nestedness within the
fine constrained space. `dd_set` is the full Dirichlet dedup set (corners and
higher-order nodes alike), in the same indexing as `labels`.
"""
function _mask_dirichlet_rows(B::SparseMatrixCSC{T,Int},
                              labels::AbstractVector{Int}, dd_set) where {T}
    keep = T[labels[i] in dd_set ? zero(T) : one(T) for i in 1:size(B, 1)]
    return dropzeros!(spdiagm(0 => keep) * B)
end

"""
    _unique_coords(labels, x) -> Matrix

First-occurrence coordinate of each connectivity id: `out[labels[i], :] = x[i, :]`
for the first `i` carrying that id (natural scan order). Reproduces *exactly* the
`unique_xy` table that `_dedupe(x)` returns, but driven by the cached `labels`
(`vec(geom.t)`) instead of a fresh coordinate dedup. Used where an auxiliary
problem needs node coordinates (e.g. the affine P1 stiffness).
"""
function _unique_coords(labels::AbstractVector{Int}, x::AbstractMatrix{T}) where {T}
    n = isempty(labels) ? 0 : maximum(labels)
    out  = Matrix{T}(undef, n, size(x, 2))
    seen = falses(n)
    @inbounds for i in eachindex(labels)
        l = labels[i]
        if !seen[l]
            out[l, :] = @view x[i, :]
            seen[l] = true
        end
    end
    return out
end

"""
    _corner_labels_from_t(t, corner_local) -> (labels, n_v)

Compact corner connectivity from the cached full-node connectivity `t` (shape
`(V, N)`). For each element `e` and corner slot `ci` (local node `corner_local[ci]`)
emit a compact id in `1:n_v` for the full-node id `t[corner_local[ci], e]`, in
`(corner, element)` flat order (index `(e-1)*nc + ci`), assigning ids by first
occurrence. The corner numbering is arbitrary: the auxiliary problem is
relabel-invariant.
"""
function _corner_labels_from_t(t::AbstractMatrix{Int}, corner_local::NTuple{nc,Int}) where {nc}
    N = size(t, 2)
    remap = Dict{Int,Int}()
    out   = Vector{Int}(undef, nc * N)
    n_v   = 0
    @inbounds for e in 1:N, ci in 1:nc
        fid = t[corner_local[ci], e]
        cc  = get(remap, fid, 0)
        if cc == 0
            n_v += 1; cc = n_v; remap[fid] = cc
        end
        out[(e-1)*nc + ci] = cc
    end
    return out, n_v
end

"""
    MultiGrid{T,M_R,G<:Geometry{T}}

A `Geometry` plus a multigrid hierarchy. For each state-variable subspace symbol
(`:dirichlet`, `:full`, `:uniform`, or a user-named Dirichlet class) the hierarchy
stores a single family of **fine-level prolongations** `R[X][l]`: the matrix that
lifts level-`l`-`X`-subspace coefficients directly to the fine broken basis (the
`R_{ℓ,X}` of the paper). `R[X][end]` is the fine-level subspace embedding itself
(identity for `:full`, the constant column for `:uniform`, the continuous
zero-trace embedding for Dirichlet classes).

The per-level (level → level+1) transfer/restriction operators are an
*implementation detail* of `amg`/`geometric_mg`: they are composed into the
`R[X][l]` at construction time and not retained, since the solver only ever
evaluates the barrier at the fine level and restricts to a coarse search space
via `R[X][l]`.

Fields
- `geometry::G`: the fine-level `Geometry`.
- `R::Dict{Symbol,Vector{M_R}}`: `R[X][l]` is the level-`l` → fine prolongation in
  subspace `X`.

Constructors
- `MultiGrid(geometry, subspaces, refine)` — stretches the per-subspace
  hierarchies to a common depth and composes
  `R[X][l] = (refine[X] chain l→L) · subspaces[X][l]`. Accepts either per-subspace
  `Dict` transfers or plain `Vector`s (replicated across every subspace key).
- `MultiGrid(geometry, R)` — store precomposed prolongations directly (e.g. the
  Kronecker construction in `spectral2d`).
"""
struct MultiGrid{T,M_R,G<:Geometry{T}}
    geometry::G
    R::Dict{Symbol,Vector{M_R}}
end

# Compose per-subspace level-l → fine prolongations:
#   R[X][l] = (refine[X][l] · refine[X][l+1] · … · refine[X][L-1]) · subspaces[X][l].
function _compose_R(subspaces::Dict{Symbol,Vector{M_sub}},
                    refine::Dict{Symbol,Vector{M_ref}}) where {M_sub,M_ref}
    function compose(X)
        rX = refine[X]; sX = subspaces[X]; L = length(rX)
        rfp = Vector{M_ref}(undef, L)
        rfp[L] = rX[L]
        for l = L-1:-1:1
            rfp[l] = rfp[l+1] * rX[l]
        end
        [rfp[l] * sX[l] for l = 1:L]
    end
    Dict(X => compose(X) for X in keys(subspaces))
end

function MultiGrid(geometry::G,
                   subspaces::Dict{Symbol,Vector{M_sub}},
                   refine::Dict{Symbol,Vector{M_ref}}) where {T,M_sub,M_ref,G<:Geometry{T}}
    refine_s, subspaces_s =
        _stretch_per_subspace(T, refine, subspaces)
    # `:uniform` is not special-cased here: it aliases `:full`'s refine chain (set in
    # `_assemble_amg_dicts`), so it never assembles its own (degenerate, zero) auxiliary
    # problem, and composes through `_compose_R` like every other subspace — its level-l
    # prolongation is the fine constant column `ones(n_doubled, 1)` (the AMG chain on the
    # all-corners Neumann stiffness preserves constants exactly).
    MultiGrid(geometry, _compose_R(subspaces_s, refine_s))
end

# Stretch each subspace's hierarchy to the common depth L_max = max(L_X) via
# ceil-interpolation. Subspace X with natural depth L_X < L_max gets its
# synthetic level i mapped to natural level n_i = ceil(L_X*i/L_max); transitions
# with n_{i+1} == n_i are identities (no-op refinement) at the level-l-X-broken-
# basis row dim; transitions with n_{i+1} > n_i reuse X's natural AMG step.
# Returns the stretched (refine, subspaces) dicts. If every subspace
# already has depth L_max, the originals are returned unchanged.
function _stretch_per_subspace(::Type{T},
        refine::Dict{Symbol,Vector{M_ref}},
        subspaces::Dict{Symbol,Vector{M_sub}}) where {T,M_ref,M_sub}
    L_X = Dict(X => length(refine[X]) for X in keys(refine))
    L_max = maximum(values(L_X))
    if all(==(L_max), values(L_X))
        return refine, subspaces
    end
    refine_s   = Dict{Symbol,Vector{M_ref}}()
    subspaces_s = Dict{Symbol,Vector{M_sub}}()
    for X in keys(refine)
        Lx = L_X[X]
        if Lx == L_max
            refine_s[X]    = refine[X]
            subspaces_s[X] = subspaces[X]
            continue
        end
        synth2nat = [ceil(Int, Lx * i / L_max) for i in 1:L_max]
        rfX = Vector{M_ref}(undef, L_max)
        ssX = Vector{M_sub}(undef, L_max)
        for i in 1:L_max
            ni = synth2nat[i]
            ssX[i] = subspaces[X][ni]
            if i == L_max
                rfX[i] = refine[X][Lx]                # identity at fine
            elseif synth2nat[i+1] > ni
                rfX[i] = refine[X][ni]                # real AMG step
            else
                # Identity transition at the level-l-X-broken-basis (rows of
                # subspaces[X][l]). `refine[l]` maps level-l-broken-basis to
                # level-(l+1)-broken-basis; both are the same here.
                m = size(ssX[i], 1)
                rfX[i] = sparse(one(T)*I, m, m)
            end
        end
        refine_s[X]    = rfX
        subspaces_s[X] = ssX
    end
    return refine_s, subspaces_s
end

# Shared-hierarchy constructor: accept a plain Vector refine — a single
# hierarchy shared by every subspace — and replicate it across the subspace
# keys. Used by the `geometric_mg` mesh builders, which construct one hierarchy
# for the whole mesh; `amg()` instead builds the per-subspace `Dict` form.
function MultiGrid(geometry::G,
                   subspaces::Dict{Symbol,Vector{M_sub}},
                   refine::Vector{M_ref}) where {T,M_sub,M_ref,G<:Geometry{T}}
    refine_dict  = Dict{Symbol,Vector{M_ref}}(k  => refine  for k in keys(subspaces))
    MultiGrid(geometry, subspaces, refine_dict)
end

@kwdef struct AMG{X,W,M_sub,M_D_fine,G}
    geometry::G
    x::X
    w::W
    R_fine::Vector{M_sub}
    # D_fine[k]: discrete operator k at the finest level, preserving whatever block
    # structure `geometry.operators[k]` has. Used by `mgb_step` / `mgb_core` /
    # `mgb_driver` — every Newton step in the V-cycle benefits from batched-gemm
    # Hessian assembly when this is a `BlockDiag`.
    D_fine::Vector{M_D_fine}
end

"""
    amg(geom::Geometry; prolongator=amg_ruge_stuben(max_coarse=2), dirichlet_nodes=Dict(:dirichlet => find_boundary(geom))) -> MultiGrid   # FEM
    amg(geom::Geometry) -> MultiGrid                                                                          # spectral

Build an algebraic-multigrid hierarchy on top of `geom`, returning a `MultiGrid`.
Dispatched per discretization; the hierarchy's fine level matches `geom`.

# Keyword arguments (FEM discretizations: `FEM1D`, `FEM2D_P1`, `FEM2D_P2`, `FEM3D`)
- `prolongator = amg_ruge_stuben(max_coarse=2)`: the algebraic-multigrid hierarchy
  builder used to coarsen the auxiliary P1 problem. A prolongator is a callable
  mapping a `SparseMatrixCSC{Float64,Int}` stiffness to the vector of level
  prolongation matrices (finest → coarsest); these set the depth of the hierarchy.
  Three factory functions construct one while capturing the underlying AMG
  parameters:
    - `amg_ruge_stuben(; kwargs...)` — classical Ruge–Stüben (the default), via
      `AlgebraicMultigrid.ruge_stuben`. The depth parameter `max_coarse` is set
      through the factory, e.g. `amg(geom; prolongator=amg_ruge_stuben(max_coarse=4))`.
    - `amg_smoothed_aggregation(; kwargs...)` — smoothed aggregation, via
      `AlgebraicMultigrid.smoothed_aggregation`.
    - `amg_pyamg(; solver=:rootnode, kwargs...)` — the Python `pyamg` package
      (`:rootnode` energy-minimization, `:smoothed_aggregation`, or `:ruge_stuben`);
      provided by the `MultiGridBarrierPyAMGExt` extension (load PyCall).
  All `kwargs` are forwarded to the underlying solver.
- `dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} = Dict(:dirichlet => find_boundary(geom))`:
  one entry per zero-trace ("dirichlet-style") subspace. Each key is a subspace
  symbol you may assign to a state-variable component via `state_variables`, and
  its value is the set of mesh nodes constrained to zero trace in that subspace,
  given as `(vertex, element)` index pairs (the format `find_boundary` returns).

  The default builds a single `:dirichlet` subspace constraining the whole
  boundary `∂Ω`. To impose **different Dirichlet boundaries on different
  components**, name them and supply a node set for each, e.g. for `z = (u, s)`
  with `u` clamped on the north edge and `s` on the south:
  ```julia
  state_variables = [:u :dirichlet_north
                     :s :dirichlet_south]
  mg = amg(geom; dirichlet_nodes = Dict(:dirichlet_north => north_pairs,
                                        :dirichlet_south => south_pairs))
  ```
  Per entry, pass a subset of `∂Ω` for mixed Dirichlet/Neumann conditions,
  `Tuple{Int,Int}[]` for pure Neumann, or a single pinned node to break the
  constant nullspace. The reserved subspaces `:full` (all-corners Neumann) and
  `:uniform` (global constants) are always available and must not appear as keys.
  These select *which* nodes are constrained; the boundary *values* (the
  Dirichlet lift `g`) are supplied separately to `mgb_solve`.
- `auxiliary_postprocess::Function = identity`  *(opt-in; the tensor-product family
  `FEM1D`/`FEM2D`/`FEM3D`, and `FEM2D_P1` — not `FEM2D_P2`)*:
  a unary function applied to the all-corners (Neumann) auxiliary stiffness before
  it is fed to `prolongator`. Use to swap the geometric Galerkin matrix for a
  graph-Laplacian-style operator that AMG coarsens on graph topology alone.
  Example: pass the combinatorial Laplacian of the same sparsity (off-diag = -1,
  diag = degree) when running aggregation-based prolongators (`amg_smoothed_aggregation`,
  `amg_pyamg(:rootnode)`) on highly anisotropic problems near p=1 — that regime
  can otherwise blow up Newton iteration counts on the central path. The default
  `identity` is robust across the bench (the geometric stiffness encodes useful
  per-row scaling that RS uses), so this is an opt-in escape hatch rather than
  a recommended default.

# Spectral discretizations (`SPECTRAL1D`, `SPECTRAL2D`)
`amg(geom)` takes no keyword arguments. The zero-trace subspace is built by basis
truncation rather than node masking, so `dirichlet_nodes` does not apply.

See also [`find_boundary`](@ref), [`geometric_mg`](@ref), [`subdivide`](@ref).
"""
function amg end

# Assemble the `MultiGrid` subspace/refine dicts shared by every FEM
# `amg` method. The reserved `:full` (all-corners Neumann) hierarchy is always
# built; `:uniform` (global constants) aliases `:full`'s refine chain — it assembles
# no auxiliary problem of its own (the constant has zero energy, so that problem would
# be singular), and composes through `:full`'s chain like any other subspace, yielding
# the fine constant column `ones(n_doubled, 1)` at every level (the AMG chain on the
# all-corners Neumann stiffness preserves constants exactly). Each `(sym, nodes)` entry
# of `dirichlet_nodes` adds one zero-trace continuous subspace built by the
# discretization-specific `build_dirichlet(nodes) -> (refine, sub)` closure.
function _assemble_amg_dicts(::Type{T}, geom, n_doubled::Int,
        dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}},
        refine_full::Vector{SparseMatrixCSC{T,Int}},
        sizes_full::AbstractVector{Int}, L_full::Int, K_amg_full::Int,
        build_dirichlet) where {T}
    sub_full    = Vector{SparseMatrixCSC{T,Int}}(undef, L_full)
    sub_uniform = Vector{SparseMatrixCSC{T,Int}}(undef, L_full)
    for kk in 1:K_amg_full
        sub_full[kk]    = sparse(one(T) * I, sizes_full[kk], sizes_full[kk])
        sub_uniform[kk] = sparse(ones(T, sizes_full[kk], 1))
    end
    # Fine-level (level-L) embeddings: `:full` is the entire broken space
    # (identity) and `:uniform` is the constant column. Both depend only on the
    # broken-DOF count `n_doubled`.
    sub_full[L_full]    = sparse(one(T) * I, n_doubled, n_doubled)
    sub_uniform[L_full] = sparse(ones(T, n_doubled, 1))

    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(:full => sub_full, :uniform => sub_uniform)
    refine_d  = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(:full => refine_full, :uniform => refine_full)

    for (sym, nodes) in dirichlet_nodes
        (sym === :full || sym === :uniform) &&
            throw(ArgumentError("dirichlet_nodes key :$sym is reserved; choose another symbol"))
        r, s = build_dirichlet(nodes)
        subspaces[sym] = s
        refine_d[sym]  = r
    end
    return MultiGrid(geom, subspaces, refine_d)
end

"""
    geometric_mg(geom::Geometry, L::Int) -> MultiGrid

Build a geometric-subdivision multigrid hierarchy of `L` levels on top of `geom`. The
returned `MultiGrid`'s `geometry` is the finest mesh (after `L-1` levels of subdivision).

The spectral discretizations have no geometric subdivision: their `geometric_mg`
ignores `L` and returns the same hierarchy as `amg` (so [`subdivide`](@ref) is a
no-op for them).
"""
function geometric_mg end

"""
    find_boundary(geom::Geometry) -> Vector{Tuple{Int,Int}}

Return the `(v, e)` index pairs of the mesh nodes on `∂Ω`, in the
per-element layout of `geom.x` (the 3-tensor of shape `(V, N, D)`): `v`
is the local vertex index within element `e`. Duplicates are present —
a corner shared by `k` elements contributes its `k` pairs (one per
element that owns it).

`amg(geom; dirichlet_nodes=Dict(:sym => set, …))` consumes these `(v, e)`
pairs: each value `set` is a `Vector{Tuple{Int,Int}}` of the nodes
constrained to zero trace in subspace `:sym`. A geometric position is
treated as Dirichlet for that subspace iff **any** pair at that position
is in `set`, so you can pass either the full set returned by
`find_boundary` (or a subset) or a sparser representative-only set.

Defined for each FEM discretization (the tensor-product family
`FEM1D`/`FEM2D`/`FEM3D`, plus `FEM2D_P1` and `FEM2D_P2`). For spectral
discretizations the zero-trace subspace is built
by basis truncation rather than by node masking; the spectral `amg` does
not accept `dirichlet_nodes` and `find_boundary` returns the perimeter
Chebyshev nodes (paired with the single notional element index `1`) for
reference only.

Empty or singleton node sets are allowed; the resulting AMG problem may
still be well-posed if the variational form carries a mass term.
"""
function find_boundary end

"""
    subdivide(geom::Geometry, L::Int) -> Geometry

Refine `geom`'s mesh by `L-1` levels of geometric subdivision and return a new fine-mesh
`Geometry`, discarding the transfer operators that the geometric MG construction would
otherwise produce. For FEM discretizations the fine operators are `BlockDiag`, so a
subsequent `amg(subdivide(geom, L))` runs the batched-GEMM (structured) Hessian assembly
via the `D_fine` path.
"""
subdivide(geom::Geometry, L::Int) = geometric_mg(geom, L).geometry

function amg_helper(mg::MultiGrid{T,M_R,G},
        state_variables::Matrix{Symbol},
        D::Matrix{Symbol}) where {T,M_R,G}
    geometry = mg.geometry
    x = _xflat(geometry.x)   # flat (V*N, D) matrix view of the mesh tensor
    w = geometry.w
    operators = geometry.operators

    nu = size(state_variables, 1)
    size(state_variables, 2) == 2 || throw(ArgumentError(
        "state_variables must be an (n × 2) Matrix{Symbol} of (name, subspace) rows"))
    # `mg.R[X][l]` is the level-l → fine prolongation for subspace X (pre-stretched
    # to a common depth at MultiGrid construction time). The level-l search space
    # is the block-diagonal join of each state variable's level-l prolongation.
    L = length(mg.R[state_variables[1, 2]])
    size(w) == (size(x, 1),) || throw(DimensionMismatch(
        "quadrature weights have length $(length(w)) but the mesh has $(size(x, 1)) nodes"))
    R_fine = [mgb_blockdiag((mg.R[state_variables[k,2]][l] for k=1:nu)...) for l=1:L]
    nD = size(D, 1)
    size(D, 2) == 2 || throw(ArgumentError(
        "D must be an (n × 2) Matrix{Symbol} of (state variable, operator) rows"))
    bar = Dict{Symbol,Int}()
    for k=1:nu
        bar[state_variables[k,1]] = k
    end
    # D_fine[k]: finest-level operator k with its original structure preserved.
    # `operators[k]` is slotted into column-block `bar[D[k,1]]` of nu equal
    # blocks; for BlockDiag operators `_block_column` builds the BlockColumn
    # wrapper directly — exactly the structured form the f2 barrier closure
    # exploits for batched-gemm Hessian assembly.
    D_fine = [let
            haskey(bar, D[k,1]) || throw(ArgumentError(
                "D row $k references state variable :$(D[k,1]), which is not in state_variables"))
            haskey(operators, D[k,2]) || throw(ArgumentError(
                "D row $k references operator :$(D[k,2]); available: $(collect(keys(operators)))"))
            _block_column(operators[D[k,2]], bar[D[k,1]], nu)
        end for k=1:nD]
    AMG(geometry=geometry,x=x,w=w,R_fine=R_fine,D_fine=D_fine)
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
    # Feasibility operator layout: the user's D rows, then the slack id row (so
    # the problem cobarrier's `(D rows..., slack)` input is the leading nD+1
    # entries of each per-node argument), then one id row per state component.
    # The trailing id rows expose every component's nodal values to the
    # feasibility barrier, which boxes them (see `_feasibility_convex`): this
    # bounds the phase-I domain, as the self-concordant path-following theory
    # requires. On an unbounded feasibility domain the barrier is unbounded
    # below and the t-ramp chases the false minimum at infinity.
    nu = size(state_variables,1)
    D1 = vcat(D,
        [feasibility_slack id_operator],
        hcat(state_variables[:,1:1],fill(id_operator,(nu,1))))
    M2 = amg_helper(mg,s1,D1)
    return M1,M2
end

