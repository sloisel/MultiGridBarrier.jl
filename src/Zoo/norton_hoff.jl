@doc raw"""
    norton_hoff(mg; p, f, g_u, s_init) -> MGBProblem

Norton–Hoff power-law (visco-)elasticity for a vector displacement $u : \Omega
\subset \mathbb{R}^d \to \mathbb{R}^d$:
```math
\min \int |\varepsilon(u)|_F^p + f \cdot u \, dx,
\qquad \varepsilon(u) = \tfrac{1}{2}(\nabla u + \nabla u^T).
```
The symmetric gradient $\varepsilon$ kills rigid rotations, so this is the
physically-correct power-law model: linear elasticity at $p=2$, Norton–Hoff
creep / glacier ice / non-Newtonian power-law flow for $1 \leq p < 2$.

State $z = (u_1, \ldots, u_d, s)$ with slack $s \geq |\varepsilon(u)|_F^p$.

Implemented in 2D ($d=2$) and 3D ($d=3$); 1D throws an error (in 1D the
symmetric gradient reduces to the scalar gradient, so use scalar p-Poisson
or [`elastoplastic_torsion`](@ref)).

# Keyword arguments
- `p::Real = 1.5`: power-law exponent. Default 1.5 puts us in the
  non-Newtonian / Norton–Hoff creep regime ($1 \leq p < 2$); `p = 2`
  collapses to linear elasticity and is the least interesting case.
- `f::Function = x -> ntuple(_->T(0.5), d)`: vector linear forcing.
- `g_u::Function`: Dirichlet boundary lift. The dim-aware default is a
  saddle-shaped trace on the first component (`(x_1 x_2, 0)` in 2D,
  `(x_1 x_2 x_3, 0, 0)` in 3D) — non-affine, so the interior deforms in
  response without needing strong forcing.
- `s_init::Real = 100`: feasible initial slack.
"""
function norton_hoff(mg::MultiGrid{T};
        p::Real = T(1.5),
        f::Function = x -> ntuple(_ -> T(0.5), _dim(mg)),
        g_u::Function = (d = _dim(mg);
            x -> ntuple(i -> i == 1 ? T(prod(x)) : T(0), d)),
        s_init::Real = T(100)) where {T}
    d = _dim(mg)
    if d == 1
        error("norton_hoff: 1D not supported (symmetric gradient = scalar gradient; " *
              "use scalar p-Poisson or elastoplastic_torsion).")
    end
    p_val = T(p)
    S = _vector_state_setup(T, d, f, g_u, s_init)
    nz = S.nz

    # Build A so that Ay[idx] packs (ε_diag..., √2 · ε_offdiag..., 0..., s)
    # of length nz, with q = first nz-1 entries and s last. Then |q|² = |ε|_F²:
    #   |ε|_F² = Σ_i ε_ii² + 2 Σ_{i<j} ε_ij²,  ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i).
    # The "√2 · ε_offdiag" = (∂u_i/∂x_j + ∂u_j/∂x_i) / √2 has square
    # ½(∂u_i/∂x_j + ∂u_j/∂x_i)² = 2 ε_ij², matching the symmetric Frobenius.
    A_nh = let nz = nz, d = d
        x -> SMatrix{nz,nz,T}(ntuple(k -> begin
            row = (k - 1) % nz + 1
            col = (k - 1) ÷ nz + 1
            # Within y[idx], the partial ∂u_i/∂x_j sits at position (i-1)*d + j (1-based).
            partial_col(i, j) = (i - 1) * d + j
            slack_col = nz
            # Diagonal strains ε_ii at rows 1..d
            if row <= d
                # row r: ε_rr = ∂u_r/∂x_r, at partial_col(r, r).
                col == partial_col(row, row) ? one(T) : zero(T)
            elseif row <= d * (d + 1) ÷ 2
                # Off-diagonal rows: pairs (i, j) with i < j.
                # Enumerate: d+1 → (1,2); d+2 → (1,3); ...; d+(d-1) → (1,d);
                # then d+d → (2,3); etc. Build list once.
                pair_idx = row - d
                # Find (i, j) with i<j corresponding to pair_idx via traversal.
                ii = 0
                jj = 0
                count = 0
                for i in 1:d, j in (i+1):d
                    count += 1
                    if count == pair_idx
                        ii = i
                        jj = j
                    end
                end
                if col == partial_col(ii, jj) || col == partial_col(jj, ii)
                    one(T) / sqrt(T(2))
                else
                    zero(T)
                end
            elseif row == nz
                col == slack_col ? one(T) : zero(T)
            else
                zero(T)                  # padding rows
            end
        end, Val(nz * nz)))
    end
    b_nh = x -> SVector{nz,T}(ntuple(_ -> zero(T), Val(nz)))

    Q = convex_Euclidian_power(T; mg=mg, idx=S.idx, A=A_nh, b=b_nh, p=x->p_val)

    return assemble(mg; state_variables=S.state_variables, D=S.D, f=S.f_kw, g=S.g_kw, Q)
end
