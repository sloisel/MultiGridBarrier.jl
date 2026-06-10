# convex_linear: linear inequality constraints and their barrier functions.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

"""
    convex_linear(::Type{T}=Float64; mg, idx=Colon(), A=(x)->I, b=(x)->T(0), A_grid=nothing, b_grid=nothing)

Create a convex set defined by linear inequality constraints, with GPU support.

Constructs a `Convex{T}` representing linear constraints.
Defines `F(y) = A * y[idx] + b` where A, b are pre-computed per vertex.
The interior is `F > 0` (logarithmic barrier applied to each component).

# Arguments
- `T::Type=Float64`: Numeric type for computations

# Keyword Arguments
- `mg::MultiGrid`: Required. The multigrid hierarchy (provides the fine grid).
- `idx=Colon()`: Indices of y to which constraints apply (default: all)
- `A::Function`: Matrix function `x -> A(x)` for constraint coefficients
- `b::Function`: Vector function `x -> b(x)` for constraint bounds
- `A_grid`, `b_grid`: Optional pre-computed fine grids (computed from A,b if not provided)

# Returns
A `Convex{T}` whose barriers capture the pre-computed fine grids.

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

    x_fine = _xflat(mg)

    # Sample A once to detect a UniformScaling (identity) A, which is materialized to a
    # concrete matrix below. `_to_cpu_array` avoids scalar indexing on GPU arrays.
    x_cpu = _to_cpu_array(x_fine)
    A_sample = A(x_cpu isa AbstractMatrix ? x_cpu[1,:] : x_cpu[1])

    # Pre-compute the fine-level A grid if not provided. A constraint `A x[idx] + b ≥ 0`
    # is stored as the flattened per-vertex matrix `A`; a `UniformScaling` (e.g. the
    # default `A = x -> I`) is materialized to a concrete `m×m` (scaled-)identity so it
    # takes the same matrix path as any other `A`.
    if A_grid === nothing
        if A_sample isa UniformScaling
            idx isa Colon && error("convex_linear: a UniformScaling A (e.g. the default A = x -> I) with idx = Colon() cannot determine the constraint size; pass an explicit SVector idx, or a matrix-valued A.")
            m = length(idx)
            A_grid = map_rows(xi -> SVector{m*m,T}(vec(Matrix{T}(A(xi), m, m))), x_fine)
        else
            A_grid = map_rows(xi -> SVector(vec(A(xi))), x_fine)
        end
    end

    if b_grid === nothing
        b_grid = map_rows(xi -> SVector(b(xi)), x_fine)
    end

    let
        A_l = A_grid
        b_l = b_grid

        # Barrier functions receive row data via broadcasting: (A_row, b_row, y)
        # No index lookup - GPU compatible
        function barrier_f0_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            # Reconstruct A from flattened form if needed
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx
            -sum(log.(Fval))
        end

        function barrier_f1_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx
            inv_F = one(TT) ./ Fval
            grad_idx = -_At_mul(Ax, inv_F)
            _scatter_gradient(idx, grad_idx, Val(N))
        end

        function barrier_f2_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx
            inv_F2 = one(TT) ./ (Fval .^ 2)
            H_idx_flat = _At_diag_A(Ax, inv_F2)
            H_idx = reshape(H_idx_flat, Size(ni, ni))
            _scatter_hessian(idx, H_idx, Val(N))
        end

        # Cobarrier functions receive row data via broadcasting: (A_row, b_row, yhat)
        function cobarrier_f0_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx .+ slack
            -sum(log.(Fval))
        end

        function cobarrier_f1_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx .+ slack
            inv_F = one(TT) ./ Fval
            grad_idx = -_At_mul(Ax, inv_F)
            g_slack = -sum(inv_F)
            _scatter_cobarrier_gradient(idx, grad_idx, g_slack, Val(NP1))
        end

        function cobarrier_f2_l(A_row, b_row, yhat::SVector{NP1,TT}) where {NP1,TT}
            y = pop(yhat)
            slack = yhat[NP1]
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx .+ slack
            inv_F2 = one(TT) ./ (Fval .^ 2)
            H_idx_flat = _At_diag_A(Ax, inv_F2)
            cross = _At_mul(Ax, inv_F2)
            M = nc
            H_ss = sum(inv_F2)
            _scatter_cobarrier_hessian(idx, H_idx_flat, cross, H_ss, Val(M), Val(NP1))
        end

        function slack_l(A_row, b_row, y::SVector{N,TT}) where {N,TT}
            yidx = y[idx]
            Ax_flat = SVector(A_row)
            bx = SVector(b_row)
            nc = length(bx)
            ni = length(yidx)
            Ax = SMatrix{nc,ni,TT}(Ax_flat)
            Fval = Ax * yidx .+ bx
            -minimum(Fval)
        end

        return Convex{T}(
            (barrier_f0_l, barrier_f1_l, barrier_f2_l),
            (cobarrier_f0_l, cobarrier_f1_l, cobarrier_f2_l),
            slack_l,
            (A_l, b_l)  # args tuple - splatted to map_rows_gpu
        )
    end
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

