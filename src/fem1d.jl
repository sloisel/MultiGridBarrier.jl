export fem1d, FEM1D, fem1d_solve

"""
    FEM1D{T}

1D FEM geometry descriptor. Field: `L::Int` (levels). Use with `amgb`.
"""
struct FEM1D{T}
    L::Int
end

amg_dim(::FEM1D) = 1

"""
    fem1d_solve(::Type{T}=Float64; kwargs...) -> AMGBSOL

Solve a 1D FEM problem. Keyword arguments are passed to both `fem1d` (e.g. `L`)
and `amgb` (e.g. `p`, `verbose`). See `amgb` for the full list.
"""
fem1d_solve(::Type{T}=Float64;rest...) where {T} = amgb(fem1d(T;rest...);rest...)

"""
    fem1d(::Type{T}=Float64; L=4, kwargs...)

Construct 1D FEM geometry (piecewise linear) on [-1, 1].
Returns a Geometry suitable for use with `amgb`. Keyword `L` sets 2^L elements.
"""
function fem1d(::Type{T}=Float64;L=4,structured::Bool=true,rest...) where {T}
    structured ? subdivide_structured(FEM1D{T}(L)) : subdivide(FEM1D{T}(L))
end

# subdivide method for FEM1D - generates the multigrid hierarchy
function subdivide(discretization::FEM1D{T}) where {T}
    L = discretization.L
    ls = [2^k for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))',(2*n0,1)) .* 2 .- 1
        N = size(x[l])[1]
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T,I,N,N)
        uniform[l] = sparse(ones(T,(N,1)))
    end
    N = size(x[L])[1]
    w = repeat([T(2)/N],outer=(N,))
    id = sparse(T,I,N,N)
    dx = blockdiag(repeat([sparse(T[-2^(L-1) 2^(L-1)
                                    -2^(L-1) 2^(L-1)])],outer=(2^L,))...)
    refine[L] = id
    coarsen[L] = id
    for l=1:L-1
        n0 = 2^l
        refine[l] = blockdiag(
            repeat([sparse(T[1.0 0.0
                     0.5 0.5
                     0.5 0.5
                     0.0 1.0])],outer=(n0,))...)
        coarsen[l] = blockdiag(
            repeat([sparse(T[1 0 0 0
                     0 0 0 1])],outer=(n0,))...)
    end
    subspaces = Dict(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict(:id => id, :dx => dx)
    return Geometry{T,Matrix{T},Vector{T},SparseMatrixCSC{T,Int},FEM1D{T}}(discretization,
        x[end],w,subspaces,operators,refine,coarsen)
end

# Direct structured construction — builds block types without sparse intermediates
function subdivide_structured(discretization::FEM1D{T}) where {T}
    L = discretization.L
    p = 2  # block size

    # Node coordinates (same as unstructured)
    x = Array{Matrix{T},1}(undef, L)
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))', (2*n0, 1)) .* 2 .- 1
        N = size(x[l], 1)
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T, I, N, N)
        uniform[l] = sparse(ones(T, (N, 1)))
    end

    N_blocks = 2^L  # number of elements

    # Operators as BlockDiag
    id_block = zeros(T, p, p, N_blocks)
    dx_block = zeros(T, p, p, N_blocks)
    scale = T(2^(L-1))
    for i in 1:N_blocks
        id_block[1,1,i] = one(T)
        id_block[2,2,i] = one(T)
        dx_block[1,1,i] = -scale
        dx_block[1,2,i] =  scale
        dx_block[2,1,i] = -scale
        dx_block[2,2,i] =  scale
    end
    id = BlockDiag(id_block)
    dx = BlockDiag(dx_block)

    # Quadrature weights
    w = fill(T(2) / T(2 * N_blocks), 2 * N_blocks)

    # Refine/coarsen as VBlockDiag/HBlockDiag
    # Reference blocks: refine = [1 0; 0.5 0.5; 0.5 0.5; 0 1] (4×2, K=2 sub-blocks of 2×2)
    #                   coarsen = [1 0 0 0; 0 0 0 1] (2×4, K=2 sub-blocks of 2×2)
    ref_sub1 = T[1 0; 0.5 0.5]   # first 2×2 sub-block of refine
    ref_sub2 = T[0.5 0.5; 0 1]   # second 2×2 sub-block of refine
    coar_sub1 = T[1 0; 0 0]      # first 2×2 sub-block of coarsen
    coar_sub2 = T[0 0; 0 1]      # second 2×2 sub-block of coarsen

    # Determine type from level 1 for vector element type
    n0_1 = 2^1
    ref_data_1 = zeros(T, p, p, 2 * n0_1)
    coar_data_1 = zeros(T, p, p, 2 * n0_1)
    for i in 1:n0_1
        ref_data_1[:, :, 2*(i-1)+1] = ref_sub1
        ref_data_1[:, :, 2*(i-1)+2] = ref_sub2
        coar_data_1[:, :, 2*(i-1)+1] = coar_sub1
        coar_data_1[:, :, 2*(i-1)+2] = coar_sub2
    end
    ref1 = VBlockDiag(p, p, 2, n0_1, ref_data_1)
    coar1 = HBlockDiag(p, p, 2, n0_1, coar_data_1)

    refine = Vector{typeof(ref1)}(undef, L)
    coarsen = Vector{typeof(coar1)}(undef, L)
    refine[1] = ref1
    coarsen[1] = coar1

    for l in 2:L-1
        n0 = 2^l
        ref_data = zeros(T, p, p, 2 * n0)
        coar_data = zeros(T, p, p, 2 * n0)
        for i in 1:n0
            ref_data[:, :, 2*(i-1)+1] = ref_sub1
            ref_data[:, :, 2*(i-1)+2] = ref_sub2
            coar_data[:, :, 2*(i-1)+1] = coar_sub1
            coar_data[:, :, 2*(i-1)+2] = coar_sub2
        end
        refine[l] = VBlockDiag(p, p, 2, n0, ref_data)
        coarsen[l] = HBlockDiag(p, p, 2, n0, coar_data)
    end

    # Level L: identity as VBlockDiag/HBlockDiag with K=1
    id_ref_data = zeros(T, p, p, N_blocks)
    for i in 1:N_blocks
        id_ref_data[1,1,i] = one(T)
        id_ref_data[2,2,i] = one(T)
    end
    refine[L] = VBlockDiag(p, p, 1, N_blocks, id_ref_data)
    coarsen[L] = HBlockDiag(p, p, 1, N_blocks, copy(id_ref_data))

    subspaces = Dict(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict(:id => id, :dx => dx)
    return Geometry{T, Matrix{T}, Vector{T}, BlockDiag{T,Array{T,3}},
                    VBlockDiag{T,Array{T,3}}, HBlockDiag{T,Array{T,3}},
                    SparseMatrixCSC{T,Int}, FEM1D{T}}(
        discretization, x[end], w, subspaces, operators, refine, coarsen)
end

# Internal interpolation function for piecewise linear functions
function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::T) where{T}
    b = length(x)
    if t<x[1]
        return y[1]
    elseif t>x[b]
        return y[b]
    end
    a = 1
    while b-a>1
        c = (a+b)÷2
        if x[c]<=t
            a=c
        else
            b=c
        end
    end
    w = (t-x[a])/(x[b]-x[a])
    return w*y[b]+(1-w)*y[a]
end

# Vector version of fem1d_interp
function fem1d_interp(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}
    [fem1d_interp(x,y,t[k]) for k=1:length(t)]
end

# Implementation of interpolate for FEM1D
interpolate(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,<:Any,<:Any,FEM1D{T}}, z::Vector{T}, t) where {T} = fem1d_interp(reshape(M.x,(:,)),z,t)

plot(M::Geometry{T,Matrix{T},Vector{T},<:Any,<:Any,<:Any,<:Any,FEM1D{T}}, z::Vector{T}; kwargs...) where {T} = plot(M.x,z; kwargs...)

_default_block_size(::FEM1D) = 2
