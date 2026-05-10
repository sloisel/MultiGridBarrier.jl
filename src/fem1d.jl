"""
    fem1d(::Type{T}=Float64; nodes, K=..., max_coarse=2) -> Geometry

Construct a 1D FEM geometry with piecewise-linear (P1) elements on the
mesh you supply. The result is passed to `amgb` to solve a Dirichlet
variational problem. The multigrid hierarchy is generated automatically;
there is no `L` parameter.

# Arguments
- `nodes`: strictly increasing fine-mesh vertices. Defines the default
  `K`; not otherwise used.
- `K`: doubled per-element corner matrix (`2N × 1`), two rows per element
  giving each element's left and right endpoints. Defaults to the reshape
  of `nodes`. Provided for API symmetry with `fem2d_P1`/`fem2d_P2`/`fem3d`,
  where `K` is the doubled per-element corner mesh.
- `max_coarse`: AMG keeps coarsening until the coarsest level has at most
  this many DOFs. The default `2` produces a hierarchy that bottoms out at
  a single global mode — matching `MultiGridBarrier.geometric_fem1d`'s geometric
  depth and what `amgb`'s slash cycle expects.

# Example
```julia
geom = fem1d(; nodes = collect(range(-1.0, 1.0, length=33)))
sol  = amgb(geom; p=1.5)
```

# Caveat — Dirichlet only
The returned Geometry is intended for Dirichlet boundary conditions. The
`:full` and `:uniform` subspaces are populated for API compatibility but
their semantics at coarse levels do not include boundary DOFs.
"""
function fem1d(::Type{T}=Float64;
                         nodes::Vector{T},
                         K::Matrix{T} = reshape([nodes[k] for i in 1:length(nodes)-1 for k in (i, i+1)], :, 1),
                         max_coarse::Int=2, rest...) where {T}
    # `nodes` exists only to seed K's default; K is the canonical mesh.
    n_e   = size(K, 1) ÷ 2
    n     = n_e + 1
    n_int = n - 2
    h     = [K[2k, 1] - K[2k-1, 1] for k in 1:n_e]

    # 1. Continuous P1 Dirichlet stiffness on the interior nodes (n_int × n_int).
    K_int = _assemble_dirichlet_stiffness(h)

    # 2. AMG hierarchy on K_int (returns prolongations finest→coarsest).
    P_amg = _amg_prolongations(K_int, T; max_coarse=max_coarse)
    n_amg_steps = length(P_amg)         # number of prolongation matrices
    K_amg = n_amg_steps + 1             # total interior grid levels (≥ 1)

    # 3. Total levels = AMG interior levels + continuous fine + doubled fine.
    L = K_amg + 2

    # ---------- refine[ℓ] (level ℓ → level ℓ+1) ----------
    refine  = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    coarsen = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    # AMG steps. Our convention: level 1 = coarsest, level K_amg = finest interior.
    # AMG returns P[i] for i=1..n_amg_steps with P[1] = finest interpolation.
    # Mapping: refine[K_amg - i] = P_amg[i].
    for i in 1:n_amg_steps
        k = K_amg - i
        refine[k]  = P_amg[i]
        coarsen[k] = _amg_injection(P_amg[i])
    end

    # Bridge 1: interior (n_int) → continuous fine (n).
    refine[K_amg]  = _interior_to_continuous(n, T)
    coarsen[K_amg] = _continuous_to_interior(n, T)

    # Bridge 2: continuous fine (n) → doubled fine (2·n_e).
    refine[K_amg + 1]  = _doubling_map(n, T)
    coarsen[K_amg + 1] = _undoubling_map(n, T)

    # Top: identity at finest.
    refine[L]  = sparse(one(T) * I, 2*n_e, 2*n_e)
    coarsen[L] = sparse(one(T) * I, 2*n_e, 2*n_e)

    # ---------- per-level native sizes ----------
    sizes = Vector{Int}(undef, L)
    sizes[K_amg]     = n_int
    for k in K_amg-1:-1:1
        sizes[k] = size(refine[k], 2)
    end
    sizes[K_amg + 1] = n
    sizes[L]         = 2*n_e

    # ---------- subspaces ----------
    sub_full      = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_dirichlet = Vector{SparseMatrixCSC{T,Int}}(undef, L)
    sub_uniform   = Vector{SparseMatrixCSC{T,Int}}(undef, L)

    # AMG interior levels: identity (boundary DOFs do not exist here).
    for k in 1:K_amg
        sub_full[k]      = sparse(one(T) * I, sizes[k], sizes[k])
        sub_dirichlet[k] = sparse(one(T) * I, sizes[k], sizes[k])
        sub_uniform[k]   = sparse(ones(T, sizes[k], 1))
    end

    # Continuous fine: :dirichlet zeros boundary continuous rows.
    sub_full[K_amg + 1]      = sparse(one(T) * I, n, n)
    sub_dirichlet[K_amg + 1] = _interior_to_continuous(n, T)
    sub_uniform[K_amg + 1]   = sparse(ones(T, n, 1))

    # Doubled fine: :dirichlet enforces continuity *and* zero boundary.
    sub_full[L]      = sparse(one(T) * I, 2*n_e, 2*n_e)
    sub_dirichlet[L] = _doubled_dirichlet_subspace(n_e, T)
    sub_uniform[L]   = sparse(ones(T, 2*n_e, 1))

    # ---------- operators (defined at fine doubled level only) ----------
    id_op = sparse(one(T) * I, 2*n_e, 2*n_e)
    dx_op = _dx_doubled(h, T)

    # ---------- quadrature ----------
    # K is the canonical doubled-DOF coordinate matrix; populates geometry.x.
    x = K                                 # (2·n_e, 1) Matrix{T}
    w = _doubled_weights(h)               # (2·n_e,) Vector

    subspaces = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}(
        :full      => sub_full,
        :dirichlet => sub_dirichlet,
        :uniform   => sub_uniform,
    )
    operators = Dict{Symbol, SparseMatrixCSC{T,Int}}(
        :id => id_op,
        :dx => dx_op,
    )

    disc = FEM1D{T}(L)
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, FEM1D{T}}(
        disc, x, w, subspaces, operators, refine, coarsen
    )
end

"""
    fem1d_solve(::Type{T}=Float64; rest...) -> AMGBSOL

Solve a 1D Dirichlet variational problem with P1 elements on the mesh you
supply. Equivalent to `amgb(fem1d(T; rest...); rest...)`: keyword
arguments are forwarded to both `fem1d` (mesh kwargs `nodes`,
`max_coarse`) and `amgb` (solver kwargs `p`, `f`, `g`, `verbose`, …).

# Example
```julia
sol = fem1d_solve(nodes = collect(range(-1.0, 1.0, length=33)), p = 1.5)
```

See `amgb` for the full set of solver kwargs.
"""
fem1d_solve(::Type{T}=Float64; rest...) where {T} =
    amgb(fem1d(T; rest...); rest...)

# ============================================================================
# Helpers
# ============================================================================

# Continuous P1 Dirichlet stiffness on interior nodes.
# Element i has length h[i] and connects v_i, v_{i+1}.
# Interior node v_{k+1} (matrix index k=1..n_int) gets contributions from elements k, k+1.
function _assemble_dirichlet_stiffness(h::Vector{T}) where {T}
    n_e   = length(h)
    n_int = n_e - 1
    if n_int == 0
        return spzeros(T, 0, 0)
    end
    if n_int == 1
        return sparse(reshape([T(1)/h[1] + T(1)/h[2]], 1, 1))
    end
    d = T[T(1)/h[k] + T(1)/h[k+1] for k in 1:n_int]
    e = T[-T(1)/h[k+1]            for k in 1:n_int-1]
    return sparse(SymTridiagonal(d, e))
end

# RS-AMG on K_int. Returns vector of prolongations [P_1, P_2, ...] where P_i is
# the i-th prolongation produced by AlgebraicMultigrid (finest first).
# AMG works in Float64; cast back to T. Empty vector if AMG produces no coarsening.
function _amg_prolongations(K_int::SparseMatrixCSC{T,Int}, ::Type{T_out};
                             max_coarse::Int=2) where {T, T_out}
    if size(K_int, 1) == 0
        return SparseMatrixCSC{T_out,Int}[]
    end
    K64 = SparseMatrixCSC{Float64,Int}(K_int)
    ml = AlgebraicMultigrid.ruge_stuben(K64; max_coarse=max_coarse)
    return [SparseMatrixCSC{T_out,Int}(ml.levels[i].P) for i in 1:length(ml.levels)]
end

# Build a sparse C-point-injection restriction R such that R * P = I exactly.
# RS-AMG places a unit row of P at every C-point; we locate them by scanning.
function _amg_injection(P::SparseMatrixCSC{T,Int}) where {T}
    n_fine, n_coarse = size(P)
    c_inds = Vector{Int}(undef, n_coarse)
    found  = falses(n_coarse)
    rows   = rowvals(P)
    vals   = nonzeros(P)
    # Rows where exactly one nonzero of value ≈ 1.
    nz_per_row = zeros(Int, n_fine)
    @inbounds for j in 1:n_coarse
        for k in nzrange(P, j)
            nz_per_row[rows[k]] += 1
        end
    end
    @inbounds for j in 1:n_coarse
        for k in nzrange(P, j)
            i = rows[k]
            if nz_per_row[i] == 1 && isapprox(vals[k], one(T); atol=128*eps(real(T)))
                if !found[j]
                    c_inds[j] = i
                    found[j]  = true
                    break
                end
            end
        end
        found[j] || error("could not identify C-point for coarse DOF $j " *
                          "(P column has no unit-vector row); P may not be RS classical")
    end
    return sparse(1:n_coarse, c_inds, ones(T, n_coarse), n_coarse, n_fine)
end

# n × (n-2): column k (k=1..n-2) is e_{k+1}. Embeds interior into continuous,
# zero-padding at the boundary entries.
function _interior_to_continuous(n::Int, ::Type{T}) where {T}
    n_int = n - 2
    return sparse(2:n-1, 1:n_int, ones(T, n_int), n, n_int)
end

# (n-2) × n: row k = e_{k+1}^T. Drops first and last continuous rows.
function _continuous_to_interior(n::Int, ::Type{T}) where {T}
    n_int = n - 2
    return sparse(1:n_int, 2:n-1, ones(T, n_int), n_int, n)
end

# 2(n-1) × n: doubled DOF j ← continuous DOF i where j is the appropriate copy.
# Element i covers v_i, v_{i+1}. Doubled DOFs (2i-1, 2i) = (left of e_i, right of e_i).
function _doubling_map(n::Int, ::Type{T}) where {T}
    n_e  = n - 1
    rows = Vector{Int}(undef, 2*n_e)
    cols = Vector{Int}(undef, 2*n_e)
    @inbounds for i in 1:n_e
        rows[2i-1] = 2i-1; cols[2i-1] = i      # left of element i  ← v_i
        rows[2i]   = 2i;   cols[2i]   = i + 1  # right of element i ← v_{i+1}
    end
    return sparse(rows, cols, ones(T, 2*n_e), 2*n_e, n)
end

# n × 2(n-1): selects one doubled copy per continuous DOF (left inverse of doubling).
function _undoubling_map(n::Int, ::Type{T}) where {T}
    n_e  = n - 1
    cols = Vector{Int}(undef, n)
    cols[1] = 1                     # v_1 ← d_1
    @inbounds for k in 2:n-1
        cols[k] = 2k - 1            # v_k ← d_{2k-1} (left of element k)
    end
    cols[n] = 2*n_e                 # v_n ← d_{2 n_e}
    return sparse(1:n, cols, ones(T, n), n, 2*n_e)
end

# 2 n_e × (n_e - 1): geometric_fem1d's Dirichlet subspace on the doubled basis.
# For each interior continuous node v_k (k=2..n_e), put 1's at d_{2(k-1)} and d_{2k-1}.
function _doubled_dirichlet_subspace(n_e::Int, ::Type{T}) where {T}
    n_int = n_e - 1
    if n_int <= 0
        return spzeros(T, 2*n_e, 0)
    end
    rows = Vector{Int}(undef, 2*n_int)
    cols = Vector{Int}(undef, 2*n_int)
    @inbounds for k in 2:n_e
        j = k - 1
        rows[2j-1] = 2*(k-1); cols[2j-1] = j   # right of element k-1
        rows[2j]   = 2*k - 1; cols[2j]   = j   # left  of element k
    end
    return sparse(rows, cols, ones(T, 2*n_int), 2*n_e, n_int)
end

# Block-diagonal dx with per-element 2×2 blocks (1/h_i) * [-1 1; -1 1].
function _dx_doubled(h::Vector{T}, ::Type{T_out}) where {T, T_out}
    n_e = length(h)
    rows = Vector{Int}(undef, 4*n_e)
    cols = Vector{Int}(undef, 4*n_e)
    vals = Vector{T_out}(undef, 4*n_e)
    @inbounds for i in 1:n_e
        s = T_out(1) / T_out(h[i])
        base = 4*(i-1)
        # row 2i-1: -s, +s; row 2i: -s, +s — at cols (2i-1, 2i)
        rows[base+1] = 2i-1; cols[base+1] = 2i-1; vals[base+1] = -s
        rows[base+2] = 2i-1; cols[base+2] = 2i;   vals[base+2] =  s
        rows[base+3] = 2i;   cols[base+3] = 2i-1; vals[base+3] = -s
        rows[base+4] = 2i;   cols[base+4] = 2i;   vals[base+4] =  s
    end
    return sparse(rows, cols, vals, 2*n_e, 2*n_e)
end

# Per-DOF trapezoidal weights = h_i / 2 for the two doubled DOFs of element i.
function _doubled_weights(h::Vector{T}) where {T}
    n_e = length(h)
    w   = Vector{T}(undef, 2*n_e)
    @inbounds for i in 1:n_e
        w[2i-1] = h[i] / 2
        w[2i]   = h[i] / 2
    end
    return w
end

