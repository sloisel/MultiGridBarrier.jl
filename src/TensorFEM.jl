# TensorFEM.jl — dimension-generic tensor-product Q_k Lagrange finite elements.
#
# One discretization type `TensorFEM{d,e,T}` covers 1D/2D/3D structured Q_k elements
# (intrinsic dim `d`, ambient/embedding dim `e ≥ d`): both dimensions are *type
# parameters* (so the generic assembly compiles per-dimension and plotting /
# interpolation dispatch on intrinsic *and* ambient dim), while the polynomial order
# `k` is a *field* (matching FEM3D and the batched-GEMM perf model). When `e > d` the
# element is an embedded manifold (curve in ℝ²/ℝ³, surface in ℝ³); `e = d` is the
# ordinary codimension-0 case. The user-facing constructors are `fem1d`, `fem2d`,
# `fem3d` (aliases FEM1D/FEM2D/FEM3D = TensorFEM{1/2/3}, fixing only `d`). At k=1 in
# 1D this reproduces the legacy P1 `fem1d` exactly.
#
# The AMG / continuity / boundary plumbing mirrors the FEM3D pattern (Q1-corner
# auxiliary stiffness derived from the broken operators, a 2^d-corner multilinear
# lift, the continuous-Q_k embedding `_p2_continuous_subspace`, and face-use-count
# boundary detection) — lifted to arbitrary `d`. Reused helpers `_dedupe`,
# `_p2_continuous_subspace`, `_amg_prolongations`, `_assemble_amg_dicts`,
# `_pairs_to_linear`, `_xflat`, `BlockDiag` are resolved at call time from their
# defining files (every file is included into the one `MultiGridBarrier` module).

export TensorFEM, FEM1D, FEM2D, FEM3D, fem1d, fem2d, fem3d, tensor_dofmap

using Random: Xoshiro

# ============================================================================
# Discretization type + aliases
# ============================================================================

"""
    TensorFEM{d, e, T}

Discretization descriptor for a `d`-dimensional tensor-product Q_k Lagrange FEM
embedded in ambient dimension `e ≥ d` (an embedded manifold when `e > d`). Here
`d` is the intrinsic dimension (number of reference axes) and `e` the ambient /
embedding dimension (`= size(geom.x, 3)`); both are integers, `T` the scalar type.

Fields
- `k::Int`: polynomial order (each element carries `(k+1)^d` Lagrange-Chebyshev nodes).
- `K::Array{T,3}`: the Q1 corner mesh tensor of shape `(2^d, N, e)` —
  `K[v, j, c]` is ambient coordinate `c` of corner `v` of element `j`. Corners are
  in tensor-product order over `{-1,+1}^d` (axis-1 fastest). Informational — it
  records the corner input; the hierarchy builders work from the stored full
  node tensor `geom.x`.

`FEM1D = TensorFEM{1}`, `FEM2D = TensorFEM{2}`, `FEM3D = TensorFEM{3}` are aliases
fixing only the intrinsic dimension (leaving `e`, `T` free): e.g. `TensorFEM{2,2}`
is a planar quad mesh, `TensorFEM{2,3}` a surface in ℝ³, `TensorFEM{1,3}` a curve
in ℝ³.
"""
struct TensorFEM{d, e, T}
    k::Int
    K::Array{T,3}
end

# Convenience constructor that infers the ambient dim `e` and scalar `T` from the
# corner tensor. Used by setup-time internal construction (geometric_mg, etc.); the
# public `fem1d/fem2d/fem3d` instead thread a `Val(e)` so their return type stays
# inferrable (e is a compile-time constant rather than `size(K,3)`).
TensorFEM{d}(k::Int, K::Array{T,3}) where {d,T} = TensorFEM{d, size(K, 3), T}(k, K)

const FEM1D = TensorFEM{1}
const FEM2D = TensorFEM{2}
const FEM3D = TensorFEM{3}

amg_dim(::TensorFEM{d}) where {d} = d              # intrinsic dimension
ambient_dim(::TensorFEM{d,e}) where {d,e} = e      # ambient / embedding dimension

# Operator symbols by axis (1=x, 2=y, 3=z).
const _TF_AXIS_SYMS = (:dx, :dy, :dz)

# Generic d-dimensional vertex dedup (random-projection sort + tol bucket).
# Returns (unique_coords, labels) where labels[i] is the unique-id of row i.
# Shared by the TensorFEM core and `fem2d_P2`'s amg/find_boundary.
function _dedupe(x::Matrix{T}) where {T}
    n, d = size(x)
    rng  = Xoshiro(hash(x))
    u    = randn(rng, T, d); u ./= norm(u)
    p    = x * u
    P    = sortperm(p)
    tol  = max(maximum(abs, x), one(T)) * 100 * eps(real(T))
    labels = zeros(Int, n)
    count  = 0
    a      = 1
    while a <= n
        if labels[P[a]] == 0
            count += 1
            labels[P[a]] = count
            b = a + 1
            while b <= n && p[P[b]] <= p[P[a]] + tol
                b += 1
            end
            for kk in a+1:b-1
                if norm(@view(x[P[a], :]) .- @view(x[P[kk], :])) <= tol
                    labels[P[kk]] = count
                end
            end
        end
        a += 1
    end
    unique_xy = zeros(T, count, d)
    seen = falses(count)
    @inbounds for kk in 1:n
        l = labels[kk]
        if !seen[l]
            unique_xy[l, :] .= @view x[kk, :]
            seen[l] = true
        end
    end
    return unique_xy, labels
end

# ============================================================================
# 1D reference primitives (self-contained; mirror Mesh3d's, ascending nodes)
# ============================================================================

# Chebyshev-Lobatto nodes on [-1,1], ascending: k=1 -> [-1, +1].
_tf_nodes(k::Int) = [-cos(i * π / k) for i in 0:k]

# Clenshaw-Curtis weights for the k+1 nodes above (sum to 2 on [-1,1]).
function _tf_weights(k::Int)
    k == 0 && return [2.0]
    N = k
    θ = [i * π / N for i in 0:N]
    w = zeros(N + 1)
    for i in 0:N
        val = 1.0
        for j in 1:div(N, 2)
            c = (2j == N) ? 1.0 : 2.0
            val += c / (1.0 - 4.0 * j^2) * cos(2.0 * j * θ[i+1])
        end
        w[i+1] = (i == 0 || i == N) ? val / N : 2.0 * val / N
    end
    return w
end

# Dense 1D differentiation matrix D[i,j] = L_j'(x_i) on the given nodes.
function _tf_dmat(nodes::AbstractVector{T}) where {T}
    k = length(nodes) - 1
    D = zeros(T, k+1, k+1)
    for i in 1:k+1, j in 1:k+1
        if i == j
            s = zero(T)
            for m in 1:k+1
                m != i && (s += one(T) / (nodes[i] - nodes[m]))
            end
            D[i, j] = s
        else
            num = one(T)
            for m in 1:k+1
                (m != j && m != i) && (num *= (nodes[i] - nodes[m]))
            end
            den = one(T)
            for m in 1:k+1
                m != j && (den *= (nodes[j] - nodes[m]))
            end
            D[i, j] = num / den
        end
    end
    return D
end

# 1D Lagrange basis values at x_val on `nodes`.
function _tf_lagrange(nodes::AbstractVector{T}, x_val) where {T}
    k = length(nodes) - 1
    vals = zeros(T, k+1)
    for i in 1:k+1
        num = one(T); den = one(T)
        for j in 1:k+1
            if i != j
                num *= (x_val - nodes[j])
                den *= (nodes[i] - nodes[j])
            end
        end
        vals[i] = num / den
    end
    return vals
end

# ============================================================================
# Dimension-generic reference element
# ============================================================================

# kron over axes b=d..1 of (D_1d if b==axis else I_1d). Axis 1 = fastest index.
function _tf_kron_axis(D1::Matrix{T}, I1::Matrix{T}, ::Val{d}, axis::Int) where {d,T}
    facs = ntuple(j -> ((d - j + 1) == axis ? D1 : I1), Val(d))
    foldl(kron, facs)
end

struct _TFRef{T,d}
    s::Int
    nodes1::Vector{T}
    w1::Vector{T}
    D1::Matrix{T}
    Daxis::NTuple{d, Matrix{T}}    # per-axis reference derivative operators (s^d × s^d)
    nodesref::Matrix{T}           # (s^d × d) reference node coordinates
    wref::Vector{T}               # (s^d) tensor quadrature weights
    n::Int                        # s^d
end

function _tf_reference(::Val{d}, k::Int, ::Type{T}) where {d,T}
    s = k + 1
    nodes1 = T.(_tf_nodes(k))
    w1     = T.(_tf_weights(k))
    D1     = T.(_tf_dmat(nodes1))
    I1     = Matrix{T}(I, s, s)
    Daxis  = ntuple(a -> _tf_kron_axis(D1, I1, Val(d), a), Val(d))
    cart   = CartesianIndices(ntuple(_ -> s, Val(d)))
    n      = s^d
    nodesref = Matrix{T}(undef, n, d)
    wref     = Vector{T}(undef, n)
    for (lin, c) in enumerate(cart)
        ww = one(T)
        @inbounds for a in 1:d
            nodesref[lin, a] = nodes1[c[a]]
            ww *= w1[c[a]]
        end
        wref[lin] = ww
    end
    return _TFRef{T,d}(s, nodes1, w1, D1, Daxis, nodesref, wref, n)
end

# Multilinear (Q1) corner lift L (s^d × 2^d): L[i,c] = Π_a basis(corner c bit a, ξ_a).
# Corner c-1 bit (a-1): 0 -> the -1 end (weight (1-ξ_a)/2), 1 -> the +1 end ((1+ξ_a)/2).
function _tf_q1_lift(nodesref::Matrix{T}, ::Val{d}, ::Type{T}) where {d,T}
    n  = size(nodesref, 1)
    nc = 1 << d
    L  = zeros(T, n, nc)
    half = one(T) / 2
    @inbounds for i in 1:n, c in 0:nc-1
        wv = one(T)
        for a in 1:d
            ξa = nodesref[i, a]
            bit = (c >> (a-1)) & 1
            wv *= bit == 0 ? (one(T) - ξa) * half : (one(T) + ξa) * half
        end
        L[i, c+1] = wv
    end
    return L
end

# Local node linear-index (within an element's s^d block) of corner `c` (1-based).
function _tf_corner_local(c::Int, s::Int, ::Val{d}) where {d}
    lin = 1; stride = 1
    @inbounds for a in 1:d
        i_a = (((c-1) >> (a-1)) & 1) == 0 ? 1 : s
        lin += (i_a - 1) * stride
        stride *= s
    end
    return lin
end

# Extract the (2^d, N, D) Q1 corner tensor from the (s^d, N, D) Q_k node tensor
# (D = ambient dimension, ≥ d for embedded manifolds).
function _tf_extract_corners(x::Array{T,3}, k::Int, ::Val{d}) where {T,d}
    s = k + 1; nc = 1 << d; N = size(x, 2); D = size(x, 3)
    out = Array{T,3}(undef, nc, N, D)
    @inbounds for e in 1:N, c in 1:nc
        lin = _tf_corner_local(c, s, Val(d))
        for cc in 1:D
            out[c, e, cc] = x[lin, e, cc]
        end
    end
    return out
end

# ============================================================================
# Topological DOF numbering (corner connectivity -> full-node connectivity)
# ============================================================================

# Global ids of the corners spanning the minimal entity that contains a local node
# with multi-index `mi` (1..s per axis) and interior-axis list `inter`. Iterates the
# 2^|inter| corners obtained by holding the boundary axes at `mi`'s low/high value and
# ranging the interior axes over {low, high}; `combo` bit j ↦ axis `inter[j]` (bit 0 =
# low end i=1, bit 1 = high end i=s), in the package corner-bit order of `_tf_corner_local`.
function _tf_entity_corner_ids(cor, mi::NTuple{d,Int}, inter::Vector{Int}, s::Int,
                               ::Val{d}) where {d}
    nint = length(inter)
    out  = Vector{Int}(undef, 1 << nint)
    @inbounds for combo in 0:(1 << nint)-1
        cbits = 0
        for a in 1:d
            j = findfirst(==(a), inter)
            bit = j === nothing ? (mi[a] == s ? 1 : 0) : ((combo >> (j-1)) & 1)
            cbits |= bit << (a-1)
        end
        out[combo+1] = Int(cor[cbits + 1])
    end
    return out
end

"""
    tensor_dofmap(t_corner, k, ::Val{d}) -> t

Topological full-node connectivity for a `d`-dimensional Qₖ tensor mesh, built from
**corner connectivity alone** — no coordinates — so coincident-but-distinct nodes
(slits, branch cuts, glued manifolds) are preserved exactly. This is the import path
that the coordinate-dedup constructor cannot express.

`t_corner` is the `(2^d, N)` corner connectivity: `t_corner[c, e]` is the global id
of corner `c` of element `e`, in the package corner order (bit `a-1` of `c-1` selects
the low/high end of axis `a`, matching `_tf_corner_local`). The returned `t`
is `((k+1)^d, N)` in element-local node order (axis-1 fastest), suitable as the `t=`
argument to [`fem1d`](@ref)/[`fem2d`](@ref)/[`fem3d`](@ref).

Numbering: corner ids carry through unchanged; shared edges/faces are matched by their
corner-id set; cell-interior nodes get fresh ids. Shared **edge**-interior nodes (`k≥3`)
are oriented by the global ids of the two endpoints; shared **face**-interior grids
(`d≥3`, `k≥3`) are canonicalized by the quad face's eight symmetries (anchored at the
minimum-id corner, axes ordered by the neighbour ids), so two elements meeting at a
shared edge or face in any relative orientation agree on every node.

Supported: **any `k` for `d≤3`** (all of `fem1d`/`fem2d`/`fem3d`). The only case still
unimplemented is interior grids on shared entities of dimension `≥3` (possible only at
`d≥4`), which `MultiGridBarrier` does not use.
"""
# Canonical position of a face-interior node, invariant under the shared quad
# face's 8 symmetries, so the two elements sharing the face agree on its global
# id. `ids` are the 4 face-corner global ids indexed `g(i,j) = ids[1 + i + 2j]`
# (i = end along the first interior axis, j = end along the second); `(pi, pj)`
# is the node's interior index (1..k-1) along those two axes. The face corners
# are distinct in a valid mesh, so: anchor the frame at the minimum-id corner
# (unique), measure the node from it along each axis, then order the two axes so
# axis 1 points toward the smaller-id neighbour. The result is an integer
# encoding of the canonical `(r1, r2)`, base `k+1`.
@inline function _tf_face_pos(ids, pi::Int, pj::Int, k::Int)
    g(i, j) = @inbounds ids[1 + i + 2j]
    i0 = 0; j0 = 0; best = g(0, 0)
    for j in 0:1, i in 0:1
        if g(i, j) < best
            best = g(i, j); i0 = i; j0 = j
        end
    end
    ri = i0 == 0 ? pi : k - pi                  # measure from the min-id corner
    rj = j0 == 0 ? pj : k - pj
    g(1 - i0, j0) > g(i0, 1 - j0) && ((ri, rj) = (rj, ri))   # axis 1 toward smaller-id neighbour
    return ri + rj * (k + 1)
end

function tensor_dofmap(t_corner::AbstractMatrix{<:Integer}, k::Int, ::Val{d}) where {d}
    s = k + 1; n = s^d; nc = 1 << d
    size(t_corner, 1) == nc || throw(ArgumentError(
        "tensor_dofmap: t_corner must have 2^$d = $nc rows (got $(size(t_corner,1)))"))
    N = size(t_corner, 2)
    t = Matrix{Int}(undef, n, N)
    next_id = isempty(t_corner) ? 0 : Int(maximum(t_corner))
    reg  = Dict{Tuple{Vector{Int},Int},Int}()
    cart = CartesianIndices(ntuple(_ -> s, Val(d)))
    @inbounds for e in 1:N
        cor = view(t_corner, :, e)
        for v in 1:n
            mi = Tuple(cart[v])
            inter = Int[a for a in 1:d if mi[a] != 1 && mi[a] != s]
            nint = length(inter)
            if nint == d
                next_id += 1; t[v, e] = next_id; continue   # cell interior: unshared
            end
            ids = _tf_entity_corner_ids(cor, mi, inter, s, Val(d))
            if nint == 0
                t[v, e] = ids[1]                            # corner: carry id through
                continue
            end
            if nint == 1
                p   = mi[inter[1]] - 1                       # 1..k-1 from the i=1 end
                pos = ids[1] <= ids[2] ? p : (k - p)        # orient by endpoint ids
                key = (sort!([ids[1], ids[2]]), pos)
            elseif nint == 2
                # shared 2D face carrying a (k-1)² interior grid: canonicalize by
                # the face's 8 symmetries (subsumes the single-node k=2 case).
                pos = _tf_face_pos(ids, mi[inter[1]] - 1, mi[inter[2]] - 1, k)
                key = (sort(ids), pos)
            else
                throw(ArgumentError("tensor_dofmap: interior grids on shared " *
                    "entities of dimension ≥ 3 (only possible at d ≥ 4, k ≥ 3) " *
                    "are not supported"))
            end
            id = get(reg, key, 0)
            if id == 0
                next_id += 1; id = next_id; reg[key] = id
            end
            t[v, e] = id
        end
    end
    return t
end

# ============================================================================
# Geometry construction (isoparametric Q_k, BlockDiag operators)
# ============================================================================

# Promote a (2^d, N, D) Q1 corner tensor to the (s^d, N, D) Q_k node tensor via
# the multilinear corner map (straight elements). fem1d/fem2d build their mesh
# this way; fem3d supplies the full Q_k node tensor directly (curved hexes).
# D = ambient dimension (≥ d): the multilinear map acts componentwise, so an
# embedded straight element (e.g. a chord/secant of a curve or a planar quad in
# ℝ³) is promoted exactly like a codim-0 one.
function _tf_promote(K::Array{T,3}, k::Int, ::Val{d}) where {T,d}
    s = k + 1; n = s^d; N = size(K, 2); D = size(K, 3)
    ref = _tf_reference(Val(d), k, T)
    Lq1 = _tf_q1_lift(ref.nodesref, Val(d), T)        # n × 2^d
    x = Array{T,3}(undef, n, N, D)
    @inbounds for e in 1:N
        x[:, e, :] = Lq1 * @view K[:, e, :]
    end
    return x
end

# Resolve a user mesh `K` to the full (s^d, N, e) Q_k node tensor. Accepts either
# the full (k+1)^d-node tensor (isoparametric / curved elements) or the 2^d-corner
# shorthand (straight element via the multilinear map). At k=1 the two coincide.
# `e = size(K,3)` is the ambient dimension (≥ d for embedded manifolds).
function _tf_resolve_mesh(K::Array{T,3}, k::Int, ::Val{d}) where {T,d}
    s = k + 1; n = s^d; nc = 1 << d
    d <= size(K, 3) <= 3 || throw(ArgumentError("fem$(d)d: K ambient dim e must satisfy $d ≤ e ≤ 3 (got e=$(size(K,3)))"))
    if size(K, 1) == n
        return K                              # full Q_k node tensor (isoparametric)
    elseif size(K, 1) == nc
        return _tf_promote(K, k, Val(d))      # Q1 corners -> straight Q_k nodes
    else
        throw(ArgumentError("fem$(d)d: K needs $nc corners or (k+1)^$d=$n nodes per element (got $(size(K,1)))"))
    end
end

# Build the single-level Geometry directly from the full Q_k node tensor `x`
# (s^d, N, e). Isoparametric: the node-varying tangent Jacobian honours curved
# elements (user-displaced edge/face/interior nodes), reducing to the affine map
# when the nodes lie on a straight element. `e` (= size(x,3)) is the ambient
# dimension; for e > d the element is an embedded manifold and the operators
# :dx,:dy[,:dz] are the ambient components of the intrinsic (tangential) gradient.
function _tf_build_geometry(disc::TensorFEM{d,e,T}, x::Array{T,3};
                            t::Union{Nothing,AbstractMatrix{<:Integer}}=nothing) where {d,e,T}
    k = disc.k
    s = k + 1; n = s^d; N = size(x, 2)
    size(x, 1) == n || throw(ArgumentError("fem$(d)d: mesh needs (k+1)^$d = $n nodes/element (got $(size(x,1)))"))
    size(x, 3) == e || throw(ArgumentError("fem$(d)d: ambient dim e=$e but mesh has $(size(x,3)) coordinate columns"))
    d <= e <= 3 || throw(ArgumentError("fem$(d)d: ambient dim e must satisfy $d ≤ e ≤ 3 (got e=$e)"))
    ref = _tf_reference(Val(d), k, T)
    Daxis = ref.Daxis

    id_block = zeros(T, n, n, N)
    deriv_blocks = ntuple(_ -> zeros(T, n, n, N), Val(e))   # one per ambient axis: components of ∇_Γ
    w = Vector{T}(undef, n * N)

    Jm = Matrix{T}(undef, e, d)                    # tangent Jacobian ∂x/∂ξ (e×d, tall when e>d)
    for el in 1:N                                  # element loop (`el`: `e` is the ambient-dim type param)
        Xe = @view x[:, el, :]                         # n × e node coords (may be curved)
        grefs = ntuple(b -> Daxis[b] * Xe, Val(d))     # each n × e: ∂x/∂ξ_b
        @inbounds for i in 1:n
            for b in 1:d, dim in 1:e
                Jm[dim, b] = grefs[b][i, dim]
            end
            g = Jm' * Jm                               # first fundamental form (d×d, SPD)
            detg = det(g)
            P = g \ Jm'                                # left pseudo-inverse (d×e); P[b,dim]=∂ξ_b/∂x_dim, == inv(J) when e==d
            for dim in 1:e
                blk = deriv_blocks[dim]
                for m in 1:n
                    acc = zero(T)
                    for b in 1:d
                        acc += P[b, dim] * Daxis[b][i, m]
                    end
                    blk[i, m, el] = acc
                end
            end
            id_block[i, i, el] = one(T)
            w[(el-1)*n + i] = ref.wref[i] * sqrt(max(detg, zero(T)))  # surface measure √det g (= |det J| when e==d)
        end
    end

    if !all(>(zero(T)), w)
        bad = findall(<=(zero(T)), w)
        badelems = sort!(unique((bad .- 1) .÷ n .+ 1))
        error("fem$(d)d: non-positive quadrature weight at $(length(bad)) node(s) across " *
              "$(length(badelems)) element(s) (first few: $(first(badelems, 5))). The metric " *
              "det(JᵀJ) ≤ 0 there — the element map is rank-deficient (degenerate / zero-measure); " *
              "supply non-degenerate, non-self-intersecting elements.")
    end

    ops = Dict{Symbol, BlockDiag{T,Array{T,3}}}(:id => BlockDiag(id_block))
    for a in 1:e
        ops[_TF_AXIS_SYMS[a]] = BlockDiag(deriv_blocks[a])
    end
    G = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, TensorFEM{d,e,T}}
    if t === nothing
        return G(disc, x, w, ops)                 # connectivity recovered by coordinate dedup
    end
    size(t) == (n, N) || throw(ArgumentError(
        "fem$(d)d: supplied `t` must be ($n, $N) full-node connectivity (got $(size(t)))"))
    return G(disc, Matrix{Int}(t), x, w, ops)     # user-supplied connectivity (slits/manifolds)
end

# ============================================================================
# Constructors: fem1d, fem2d
# ============================================================================

# Default unit-square single-quad Q1 corners (4, 1, 2), tensor order over {-1,1}^2.
function _tf_default_square(::Type{T}) where {T}
    K = Array{T,3}(undef, 4, 1, 2)
    corners = T[-1 -1; 1 -1; -1 1; 1 1]
    @inbounds for c in 1:4, dim in 1:2
        K[c, 1, dim] = corners[c, dim]
    end
    return K
end

# Shared constructor core for fem1d/fem2d/fem3d: resolve the user mesh `K` to the
# full Q_k node tensor, check it against the requested ambient dimension `e`, and
# build the Geometry. `e` arrives as a `Val` (default `Val(d)` in the public
# constructors) so the returned Geometry type stays inferrable — `e` is a
# compile-time constant rather than the runtime `size(K,3)`.
function _tf_construct(::Type{T}, k::Int, K::Array{T,3},
                       t::Union{Nothing,AbstractMatrix{<:Integer}},
                       ::Val{d}, ::Val{e}) where {T,d,e}
    x = _tf_resolve_mesh(K, k, Val(d))
    size(x, 3) == e || throw(ArgumentError(
        "fem$(d)d: ambient=Val($e) but the mesh has $(size(x,3)) coordinate column(s); " *
        "pass `ambient=Val($(size(x,3)))` (or a mesh with $e columns)"))
    return _tf_build_geometry(TensorFEM{d,e,T}(k, _tf_extract_corners(x, k, Val(d))), x; t=t)
end

"""
    fem1d(::Type{T}=Float64; nodes, k=1, K=<from nodes>, ambient=Val(1), t=nothing) -> Geometry

Construct a single-level 1D Q_k FEM `Geometry`. Each element carries `k+1`
Lagrange-Chebyshev nodes; `k=1` reproduces the legacy P1 discretization exactly.
`nodes` is the strictly increasing vector of element endpoints (the default mesh
is the straight Q_k element on each `[nodes[i], nodes[i+1]]`).

The map is **isoparametric**: pass `K` as the full `(k+1, n_e, 1)` Lagrange-node
tensor to give each element a nontrivial 1D parametrization (interior nodes off
the affine positions), or as the `(2, n_e, 1)` endpoint tensor for straight
elements. `K` may be passed on its own — `nodes` is only needed when `K` is not
given.

The curve may be **embedded in a higher ambient dimension** `e ∈ {2,3}`: pass
`ambient=Val(2)` or `ambient=Val(3)` (default `Val(1)`) together with a mesh `K`
of that many coordinate columns, and `fem1d` builds a 1-manifold (curve) in ℝ^e
of type `TensorFEM{1,e,T}`. The operators `:dx, :dy[, :dz]` are then the `e`
ambient components of the intrinsic (arc-length) gradient — each tangent to the
curve — and the quadrature weights are the arc-length measure `√det(JᵀJ)`. Closed
curves (circles, loops) glue automatically by coordinate dedup; this reduces
exactly to the `e=1` case on the diagonal.

Pass `t` (a `(k+1, n_e)` `Integer` matrix; `t[v,e]` is the global id of local node
`v` in element `e`) to supply full-node connectivity explicitly instead of
recovering it by coordinate dedup — needed when geometrically-coincident nodes must
stay topologically distinct (slit domains, branch cuts, glued manifolds). Build one
from corner connectivity with [`tensor_dofmap`](@ref).

Attach a hierarchy with `amg(geom)`.
"""
function fem1d(::Type{T}=Float64;
               nodes::Union{Nothing,Vector{T}} = nothing,
               k::Int = 1,
               K::Union{Nothing,Array{T,3}} = nothing,
               ambient::Val = Val(1),
               t::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
               rest...) where {T}
    if K === nothing
        nodes === nothing && throw(ArgumentError(
            "fem1d: pass the element endpoints `nodes`, or the mesh tensor `K` directly"))
        K = reshape(T[nodes[j] for i in 1:length(nodes)-1 for j in (i, i+1)],
                    2, length(nodes)-1, 1)
    end
    return _tf_construct(T, k, K, t, Val(1), ambient)
end

"""
    fem2d(::Type{T}=Float64; k=1, K=<unit square>, ambient=Val(2), t=nothing) -> Geometry

Construct a single-level 2D Q_k FEM `Geometry` on quadrilaterals (`k=1` is
bilinear Q1). The map is **isoparametric**: pass `K` as the full `((k+1)^2, N, 2)`
Lagrange-node tensor (tensor order, axis-1 fastest) for curved quads, or as the
`(4, N, 2)` Q1-corner tensor — order `(-1,-1), (+1,-1), (-1,+1), (+1,+1)` — for
straight quads.

The quads may be **embedded in ℝ³**: pass `ambient=Val(3)` (default `Val(2)`)
together with a mesh `K` of `3` coordinate columns, and `fem2d` builds a 2-manifold
(surface) of type `TensorFEM{2,3,T}`. The operators `:dx, :dy, :dz` are then the
three ambient components of the intrinsic tangential gradient `∇_Γ` — tangent to the
surface, so `n·∇_Γ = 0` holds by construction — and the weights are the
surface-area measure `√det(JᵀJ)`. Closed surfaces (spheres, tori) glue by
coordinate dedup. This reduces exactly to the planar discretization when `e=2`.

Pass `t` (a `((k+1)^2, N)` `Integer` matrix; `t[v,e]` is the global id of local node
`v` in element `e`) to supply full-node connectivity explicitly instead of recovering
it by coordinate dedup — needed for slit domains, branch cuts, and glued manifolds,
where geometrically-coincident nodes must stay topologically distinct. Build one from
corner connectivity with [`tensor_dofmap`](@ref).

Attach a hierarchy with `amg(geom)`.
"""
function fem2d(::Type{T}=Float64;
               k::Int = 1,
               K::Array{T,3} = _tf_default_square(T),
               ambient::Val = Val(2),
               t::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
               rest...) where {T}
    return _tf_construct(T, k, K, t, Val(2), ambient)
end

# Default unit-cube single-hex Q1 corners (8, 1, 3), tensor order over {-1,1}^3.
function _tf_default_cube(::Type{T}) where {T}
    K = Array{T,3}(undef, 8, 1, 3)
    corners = T[-1 -1 -1; 1 -1 -1; -1 1 -1; 1 1 -1; -1 -1 1; 1 -1 1; -1 1 1; 1 1 1]
    @inbounds for c in 1:8, dim in 1:3
        K[c, 1, dim] = corners[c, dim]
    end
    return K
end

"""
    fem3d(::Type{T}=Float64; k=3, K=<unit cube>, t=nothing) -> Geometry

Construct a single-level 3D Q_k FEM `Geometry` on hexahedra. The map is
**isoparametric**: pass `K` as the full `((k+1)^3, N, 3)` Lagrange-node tensor
(tensor order, axis-1 fastest) so displacing edge/face/interior nodes curves the
hex (node-varying Jacobian), or as the `(8, N, 3)` Q1-corner tensor for straight
hexes. The default is a single straight unit cube.

Pass `t` (a `((k+1)^3, N)` `Integer` matrix; `t[v,e]` is the global id of local node
`v` in element `e`) to supply full-node connectivity explicitly instead of recovering
it by coordinate dedup (slit domains / glued manifolds). [`tensor_dofmap`](@ref) builds
one from corner connectivity, though for `k≥3` in 3D it does not yet number shared
face-interior grids (it throws) — there, supply `t` by hand or use the dedup default.

Attach a hierarchy with `amg(geom)`.
"""
function fem3d(::Type{T}=Float64;
               k::Int = 3,
               K::Array{T,3} = _tf_default_cube(T),
               t::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
               rest...) where {T}
    # No `ambient` kwarg: a 3-manifold can only live in ℝ³ (ambient e = 3 = d).
    return _tf_construct(T, k, K, t, Val(3), Val(3))
end

# ============================================================================
# find_boundary (face-use-count, dimension-generic)
# ============================================================================

"""
    find_boundary(geom::Geometry{...,TensorFEM{d,e,T}}) -> Vector{Tuple{Int,Int}}

`(v, e)` index pairs into `geom.x` for every Q_k DOF on `∂Ω`. A (d-1)-face used
by exactly one element is on the boundary; every DOF on such a face is returned
(corner / edge / face-interior, including the `k≥2` interior-of-face nodes).
"""
function find_boundary(geom::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{d,E,T}}) where {T,d,E}
    k = geom.discretization.k
    s = k + 1; n = s^d; N = size(geom.x, 2)
    labels = vec(geom.t)                 # cached connectivity (== _dedupe(_xflat(x))[2])

    cart = CartesianIndices(ntuple(_ -> s, Val(d)))
    faces_local = Vector{Vector{Int}}()
    for a in 1:d, layer in (1, s)
        idxs = Int[]
        for (lin, c) in enumerate(cart)
            c[a] == layer && push!(idxs, lin)
        end
        push!(faces_local, idxs)
    end

    facecount = Dict{Vector{Int}, Int}()
    for e in 1:N
        base = (e-1) * n
        for fl in faces_local
            sig = sort!([labels[base + li] for li in fl])
            facecount[sig] = get(facecount, sig, 0) + 1
        end
    end
    bdry = Set{Int}()
    for (sig, c) in facecount
        if c == 1
            for u in sig; push!(bdry, u); end
        end
    end

    pairs = Tuple{Int,Int}[]
    for e in 1:N, v in 1:n
        labels[(e-1)*n + v] in bdry && push!(pairs, (v, e))
    end
    return pairs
end

# ============================================================================
# AMG hierarchy (dimension-generic mirror of the FEM3D pattern)
# ============================================================================

# Lift interior-Q1 corners -> broken Q_k basis via the per-element 2^d-corner
# multilinear weights (drops boundary-corner pushes, remaps column to interior).
function _tf_interior_q1_lift(node_map_q1::Vector{Int}, k::Int, ::Val{d},
                              n_v::Int, interior_corners::Vector{Int},
                              ::Type{T}) where {d,T}
    s = k + 1; n = s^d; nc = 1 << d
    interior_idx = zeros(Int, n_v)
    @inbounds for (i, c) in enumerate(interior_corners)
        interior_idx[c] = i
    end
    n_int = length(interior_corners)
    ref = _tf_reference(Val(d), k, T)
    Lq1 = _tf_q1_lift(ref.nodesref, Val(d), T)        # n × nc
    N = length(node_map_q1) ÷ nc

    rows = Int[]; cols = Int[]; vals = T[]
    sizehint!(rows, N * n * nc); sizehint!(cols, N * n * nc); sizehint!(vals, N * n * nc)
    @inbounds for e in 1:N
        offset = (e - 1) * n
        cui = ntuple(c -> interior_idx[node_map_q1[nc*(e-1) + c]], nc)
        for r in 1:n, c in 1:nc
            v = Lq1[r, c]
            if v != 0 && cui[c] != 0
                push!(rows, offset + r); push!(cols, cui[c]); push!(vals, v)
            end
        end
    end
    return sparse(rows, cols, vals, N * n, n_int)
end

# Build the AMG hierarchy on the Q1-corner Galerkin stiffness restricted to
# `interior_set`, with the level-K_amg bridge mapping interior corners -> broken
# Q_k. Mirrors `_fem3d_hierarchy`.
function _tf_hierarchy(node_map_q1::Vector{Int}, k::Int, ::Val{d},
                       A_doubled::SparseMatrixCSC{Float64,Int},
                       interior_set::AbstractVector{<:Integer},
                       n_v::Int, n_doubled::Int, prolongator, ::Type{T};
                       amg_input::Union{Nothing,SparseMatrixCSC{T,Int}}=nothing) where {d,T}
    interior_vec = collect(interior_set)
    n_loc = length(interior_vec)

    S_lift = _tf_interior_q1_lift(node_map_q1, k, Val(d), n_v, interior_vec, T)
    S64    = SparseMatrixCSC{Float64,Int}(S_lift)
    K_loc  = SparseMatrixCSC{T,Int}(S64' * A_doubled * S64)

    K_for_amg   = amg_input === nothing ? K_loc : amg_input
    P_amg       = _amg_prolongations(K_for_amg, T, prolongator)
    n_amg_steps = length(P_amg)
    K_amg       = n_amg_steps + 1
    L_total     = K_amg + 1

    refine = Vector{SparseMatrixCSC{T,Int}}(undef, L_total)
    for i in 1:n_amg_steps
        refine[K_amg - i] = P_amg[i]
    end
    refine[K_amg]   = S_lift
    refine[L_total] = sparse(one(T) * I, n_doubled, n_doubled)

    sizes = Vector{Int}(undef, L_total)
    sizes[K_amg] = n_loc
    for kk in K_amg-1:-1:1
        sizes[kk] = size(refine[kk], 2)
    end
    sizes[L_total] = n_doubled
    return refine, sizes, L_total, K_amg
end

function amg(geom::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{d,E,T}};
             prolongator = amg_ruge_stuben(max_coarse=2),
             dirichlet_nodes::Dict{Symbol,Vector{Tuple{Int,Int}}} =
                 Dict(:dirichlet => find_boundary(geom)),
             auxiliary_postprocess::Function = identity) where {T,d,E}
    k = geom.discretization.k
    s = k + 1; n = s^d; N = size(geom.x, 2); n_doubled = n * N
    nc = 1 << d

    full_labels    = vec(geom.t)         # cached connectivity (== _dedupe(_xflat(x))[2])
    n_full_unique  = maximum(full_labels)

    # Corner connectivity from `t` (no separate corner-coordinate dedup). The auxiliary
    # stiffness below is the Galerkin restriction Sᵀ A S of the true operator and uses
    # no corner coordinates, so only the labelling is needed. (Reorders corners vs the
    # old dedup — solve-equivalent.)
    cornerlocal = ntuple(c -> _tf_corner_local(c, s, Val(d)), nc)
    node_map_q1, n_v = _corner_labels_from_t(geom.t, cornerlocal)

    # All-corners auxiliary stiffness, derived from the broken Q_k operators. The
    # Dirichlet energy ∫|∇u|² (∫_Γ|∇_Γu|² for an embedded manifold) sums over all
    # D ambient gradient components — D = ambient dim ≥ d, so this is `1:D`, not `1:d`.
    Da_dim = size(geom.x, 3)
    W = spdiagm(0 => Float64.(geom.w))
    A_doubled = spzeros(Float64, n_doubled, n_doubled)
    for a in 1:Da_dim
        Da = SparseMatrixCSC{Float64,Int}(to_sparse(geom.operators[_TF_AXIS_SYMS[a]]))
        A_doubled = A_doubled + Da' * W * Da
    end

    full_to_corner = Dict{Int,Int}()
    @inbounds for e in 1:N, c in 1:nc
        broken_full   = n  * (e-1) + cornerlocal[c]
        broken_corner = nc * (e-1) + c
        full_to_corner[full_labels[broken_full]] = node_map_q1[broken_corner]
    end

    S_full   = _tf_interior_q1_lift(node_map_q1, k, Val(d), n_v, collect(1:n_v), T)
    S64_full = SparseMatrixCSC{Float64,Int}(S_full)
    K_loc_full = SparseMatrixCSC{T,Int}(S64_full' * A_doubled * S64_full)
    M_full     = auxiliary_postprocess(K_loc_full)::SparseMatrixCSC{T,Int}

    refine_full, sizes_full, L_full, K_amg_full =
        _tf_hierarchy(node_map_q1, k, Val(d), A_doubled,
                      collect(1:n_v), n_v, n_doubled, prolongator, T; amg_input=M_full)

    build_dirichlet = function (nodes::Vector{Tuple{Int,Int}})
        dd_set = Set{Int}(full_labels[r] for r in _pairs_to_linear(nodes, n))
        dc_set = Set{Int}(
            full_to_corner[fid] for fid in dd_set if haskey(full_to_corner, fid))
        interior = sort!(collect(setdiff(1:n_v, dc_set)))
        refine_dir, sizes_dir, L_dir, K_amg_dir =
            _tf_hierarchy(node_map_q1, k, Val(d), A_doubled,
                          interior, n_v, n_doubled, prolongator, T;
                          amg_input=M_full[interior, interior])
        # Force the corner-only coarse search space to vanish at *every* Dirichlet
        # node (not just the corner DOFs the auxiliary problem represents): mask the
        # bridge so its multilinear lift cannot leak nonzero values onto Dirichlet
        # edge/face/centroid nodes hosted on a facet with a free corner.
        refine_dir[K_amg_dir] =
            _mask_dirichlet_rows(refine_dir[K_amg_dir], full_labels, dd_set)
        sub = Vector{SparseMatrixCSC{T,Int}}(undef, L_dir)
        for kk in 1:K_amg_dir
            sub[kk] = sparse(one(T) * I, sizes_dir[kk], sizes_dir[kk])
        end
        sub[L_dir] = _p2_continuous_subspace(full_labels, n_full_unique, dd_set, T)
        return refine_dir, sub
    end

    return _assemble_amg_dicts(T, geom, n_doubled, dirichlet_nodes,
        refine_full, sizes_full, L_full, K_amg_full, build_dirichlet)
end

# ============================================================================
# geometric_mg (dimension-generic geometric subdivision, structured BlockDiag)
# ============================================================================

# Continuous-Q_k zero-trace subspace (n × n_interior) for a single mesh level,
# built from the broken node coordinates via dedup + face-count boundary removal.
function _tf_continuous_subspace(x::Array{T,3}, k::Int, ::Val{d}) where {T,d}
    disc = TensorFEM{d}(k, Array{T,3}(undef, 1<<d, 0, size(x,3)))   # outer ctor infers e = size(x,3)
    geomlike = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}}, typeof(disc)}(
        disc, x, T[], Dict{Symbol,BlockDiag{T,Array{T,3}}}())
    labels = vec(geomlike.t)             # cached connectivity (== _dedupe(_xflat(x))[2])
    n_unique = maximum(labels)
    bdry_pairs = find_boundary(geomlike)
    s = k + 1; n = s^d
    bset = Set{Int}(labels[v + (e-1)*n] for (v, e) in bdry_pairs)
    return _p2_continuous_subspace(labels, n_unique, bset, T)
end

# Per-child broken-basis interpolation matrix P_local (nc*n × n): block `ch`
# interpolates the parent Q_k element at child `ch`'s node positions. Child `ch`
# occupies the parent sub-box with each axis in [-1,0] (bit 0) or [0,1] (bit 1).
function _tf_refine_local(k::Int, ::Val{d}, ::Type{T}) where {d,T}
    s = k + 1; n = s^d; nc = 1 << d
    nodes1 = T.(_tf_nodes(k))
    cart = CartesianIndices(ntuple(_ -> s, Val(d)))
    half = one(T) / 2
    P_local = zeros(T, nc * n, n)
    for ch in 1:nc
        childnodes = ntuple(a -> begin
            shift = ((ch-1) >> (a-1)) & 1 == 0 ? -half : half
            nodes1 .* half .+ shift
        end, Val(d))
        for (i, ci) in enumerate(cart), (j, cj) in enumerate(cart)
            wv = one(T)
            for a in 1:d
                la = _tf_lagrange(nodes1, childnodes[a][ci[a]])
                wv *= la[cj[a]]
            end
            P_local[(ch-1)*n + i, j] = wv
        end
    end
    return P_local
end

function geometric_mg(geom::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{d,E,T}}, L::Int) where {T,d,E}
    L >= 1 || throw(ArgumentError("L must be ≥ 1"))
    k = geom.discretization.k
    s = k + 1; n = s^d; nc = 1 << d
    P_local = _tf_refine_local(k, Val(d), T)

    # Refine the *full* Q_k node mesh L-1 times (curvature-preserving): each
    # element's node coords are interpolated to its 2^d children via P_local, so
    # curved (isoparametric) elements stay curved under subdivision.
    node_meshes = Vector{Array{T,3}}(undef, L)
    node_meshes[1] = Array{T,3}(geom.x)
    for l in 1:L-1
        Xc = node_meshes[l]
        Nl = size(Xc, 2)
        Xf = Array{T,3}(undef, n, Nl * nc, size(Xc, 3))
        @inbounds for e in 1:Nl
            Xe = @view Xc[:, e, :]                          # n × D
            for ch in 1:nc
                Xf[:, (e-1)*nc + ch, :] = @view(P_local[(ch-1)*n+1 : ch*n, :]) * Xe
            end
        end
        node_meshes[l+1] = Xf
    end

    K_fine = _tf_extract_corners(node_meshes[L], k, Val(d))
    geomL  = _tf_build_geometry(TensorFEM{d}(k, K_fine), node_meshes[L])
    N_fine = size(node_meshes[L], 2)

    id_data = zeros(T, n, n, N_fine)
    @inbounds for i in 1:N_fine, j in 1:n
        id_data[j, j, i] = one(T)
    end
    id_vbd = _vblock_sparse(n, n, 1, N_fine, id_data)

    refine = Vector{typeof(id_vbd)}(undef, L)
    for l in 1:L-1
        n_elems_l = size(node_meshes[l], 2)
        ref_data = zeros(T, n, n, nc * n_elems_l)
        @inbounds for e in 1:n_elems_l, ch in 1:nc
            ref_data[:, :, (e-1)*nc + ch] = P_local[(ch-1)*n+1 : ch*n, :]
        end
        refine[l] = _vblock_sparse(n, n, nc, n_elems_l, ref_data)
    end
    refine[L] = id_vbd

    # Per-level subspaces (continuous Q_k dirichlet, full identity, uniform column).
    subspaces = Dict{Symbol,Vector{SparseMatrixCSC{T,Int}}}(
        :dirichlet => Vector{SparseMatrixCSC{T,Int}}(undef, L),
        :full      => Vector{SparseMatrixCSC{T,Int}}(undef, L),
        :uniform   => Vector{SparseMatrixCSC{T,Int}}(undef, L))
    for l in 1:L
        nl = n * size(node_meshes[l], 2)
        subspaces[:dirichlet][l] = _tf_continuous_subspace(node_meshes[l], k, Val(d))
        subspaces[:full][l]      = sparse(one(T) * I, nl, nl)
        subspaces[:uniform][l]   = sparse(ones(T, nl, 1))
    end

    return MultiGrid(geomL, subspaces, refine)
end

# ============================================================================
# interpolate + plot (per-dimension)
# ============================================================================

# 1D: per-element Lagrange interpolation; clamp outside the domain.
# Element location is a binary search (`searchsortedlast`, O(log N) per query) over
# the element left-endpoints, which are sorted ascending for an ordered mesh — the
# layout `fem1d`'s `nodes` constructor produces. (Falls back to a linear scan if the
# left-endpoints are not sorted, e.g. a hand-built out-of-order `K`.) Each element's
# values are then evaluated with the genuine degree-k Lagrange basis, so this is the
# exact Q_k interpolant, not a piecewise-linear-between-nodes approximation.
function interpolate(M::Geometry{T,<:Any,<:Any,<:Any,TensorFEM{1,1,T}}, z::Vector{T}, t) where {T}
    k = M.discretization.k
    s = k + 1
    x = M.x                                  # (s, N, 1)
    N = size(x, 2)
    nodes1 = T.(_tf_nodes(k))
    lefts  = @view x[1, :, 1]                # element left endpoints
    sorted = issorted(lefts)
    x_lo = x[1, 1, 1]; x_hi = x[s, N, 1]
    locate(tq) = if sorted
        clamp(searchsortedlast(lefts, tq), 1, N)
    else
        e = 1; while e < N && tq > x[s, e, 1]; e += 1; end; e
    end
    _interp1(tq) = begin
        tq <= x_lo && return z[1]
        tq >= x_hi && return z[s*N]
        e = locate(tq)
        a = x[1, e, 1]; b = x[s, e, 1]
        ξ = 2 * (tq - a) / (b - a) - 1
        L = _tf_lagrange(nodes1, T(ξ))
        v = zero(T)
        @inbounds for j in 1:s
            v += L[j] * z[(e-1)*s + j]
        end
        return v
    end
    return t isa AbstractVector ? [_interp1(tt) for tt in t] : _interp1(t)
end

# plot(::Geometry{...TensorFEM{1,1}}, z) and plot(::Geometry{...TensorFEM{2,2}}, z)
# live in MultiGridBarrierPyPlotExt (as do the embedded-manifold and 3D variants).
