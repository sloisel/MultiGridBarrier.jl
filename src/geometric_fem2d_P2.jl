export geometric_fem2d_P2, FEM2D_P2, geometric_fem2d_P2_solve
using Random

"""
    FEM2D_P2{T}

2D FEM geometry descriptor for quadratic+bubble triangles.
Fields: `K::Matrix{T}` (3n×2 corner triangulation), `L::Int` (levels),
`K7::Matrix{T}` (7n×2 P2+bubble mesh used as the coarsest level's coordinates;
defaults to the canonical expansion of `K`). Use with `amgb`.
"""
struct FEM2D_P2{T}
    K::Matrix{T}
    L::Int
    K7::Matrix{T}
end

"""
    geometric_fem2d_P2_solve(::Type{T}=Float64; kwargs...) -> AMGBSOL

Solve a 2D FEM problem. Keyword arguments are passed to both `geometric_fem2d_P2` (e.g. `L`, `K`)
and `amgb` (e.g. `p`, `verbose`). See `amgb` for the full list.
"""
geometric_fem2d_P2_solve(::Type{T}=Float64;rest...) where {T} = amgb(geometric_fem2d_P2(T;rest...);rest...)

amg_dim(::FEM2D_P2{T}) where {T} = 2


function reference_triangle(::Type{T}) where {T}
    K = sparse(T[6 0 0
      3 3 0
      0 6 0
      0 3 3
      0 0 6
      3 0 3
      2 2 2]./6)
    w = T[3,8,3,8,3,8,27]./60
    dx =  sparse(T[  36    0   0    0   12  -48    0
   3   60  -9   12    3   12  -81
 -12   48   0  -48   12    0    0
  -3  -12   9  -60   -3  -12   81
 -12    0   0    0  -36   48    0
  12    0   0    0  -12    0    0
   4   16   0  -16   -4    0    0]./12)
    dy = sparse(T[  0   48  -12    0   12  -48    0
 -9   60    3   12    3   12  -81
  0    0   36  -48   12    0    0
  0    0   12    0  -12    0    0
  0    0  -12   48  -36    0    0
  9  -12   -3  -12   -3  -60   81
  0   16    4    0   -4  -16    0]./12)
    coarsen = sparse([6, 1, 2, 2, 3, 4, 4, 5, 6, 2, 4, 6, 7], [1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 28], T[1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3]./3, 7, 28)
    refine = sparse([2, 3, 4, 6, 7, 9, 13, 14, 18, 20, 21, 23, 25, 27, 4, 5, 6, 7, 8, 9, 13, 14, 20, 21, 22, 23, 25, 27, 4, 6, 7, 9, 10, 11, 13, 14, 16, 20, 21, 23, 25, 27, 6, 7, 11, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 27, 2, 6, 7, 11, 13, 14, 16, 17, 18, 20, 21, 23, 25, 27, 1, 2, 6, 7, 13, 14, 18, 19, 20, 21, 23, 25, 26, 27, 6, 7, 13, 14, 20, 21, 23, 25, 27, 28], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], T[243, 648, 243, 61, 180, -81, -20, -36, -81, -20, -36, -20, -20, 61, 486, 648, 80, 144, 648, 486, 80, 144, -82, -72, 648, 80, -82, 80, -81, -20, -36, 243, 648, 243, 61, 180, -81, -20, -36, 61, -20, -20, -82, -72, 486, 648, 80, 144, 648, 486, 80, 144, 80, 648, 80, -82, -81, -20, -36, -81, -20, -36, 243, 648, 243, 61, 180, -20, 61, -20, 648, 486, 80, 144, -82, -72, 486, 648, 80, 144, -82, 80, 648, 80, 549, 324, 549, 324, 549, 324, 549, 549, 549, 648]./648, 28, 7)
    return (K=K,w=w,dx=dx,dy=dy,coarsen=coarsen,refine=refine)
end

# 2-arg constructor: derive the canonical 7-DOF mesh K7 from the corner mesh K.
function FEM2D_P2{T}(K::Matrix{T}, L::Int) where {T}
    R = reference_triangle(T)
    nn = size(K, 1) ÷ 3
    K7 = Matrix{T}(blockdiag([R.K for _ in 1:nn]...) * K)
    FEM2D_P2{T}(K, L, K7)
end

function continuous(x::Matrix{T};
                    tol=maximum(abs.(x))*10*eps(T)) where {T}
    n = size(x)[1]
    a = 1
    seed = hash(x)
    rng  = Xoshiro(seed)
    u = randn(rng,T,2)
    u = u/norm(u)
    p = x*u
    P = sortperm(p)
    labels = zeros(Int,n)
    count = 0
    while a<=n
        if labels[P[a]]==0
            count += 1
            labels[P[a]] = count
            b = a+1
            while b<=n && p[P[b]]<=p[P[a]]+tol
                b+=1
            end
            for k=a+1:b-1
                if norm(x[P[a],:]-x[P[k],:])<=tol
                    labels[P[k]] = count
                    x[P[k],:] = x[P[a],:]
                end
            end
        end
        a+=1
    end
    t = reshape(labels,(7,:))
    e = hcat(t[1:2,:],t[2:3,:],t[3:4,:],t[4:5,:],t[5:6,:],t[[6,1],:])'
    e = sort(e,dims=2)
    P = sortperm(1:size(e,1),lt=(j,k)->e[j,:]<e[k,:])
    w = e[P,:]
    J = cumsum(vcat(1,(w[1:end-1,1].!=w[2:end,1]) .|| (w[1:end-1,2].!=w[2:end,2])))
    J = J[invperm(P)]
    ne = maximum(J)
    ec = zeros(Int,ne)
    for k=1:length(J)
        ec[J[k]] += 1
    end
    idx = findall(ec[J] .== 1)
    e = e[idx,:]
    boundary = unique(reshape(e,(length(e),)))
    interior = setdiff(1:count,boundary)

    C = sparse(1:n,labels,ones(T,n),n,count)
    C[:,interior]
end


"""
    geometric_fem2d_P2(::Type{T}=Float64; L=2, K=T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],
                       K7=<derived from K>, kwargs...)

Construct 2D FEM geometry (quadratic + bubble) on a triangular mesh.
Returns a Geometry suitable for use with `amgb`.

`K` is the corner triangulation (3n × 2). `K7` is the P2+bubble mesh
(7n × 2) used as the coarsest level's coordinates; by default it is the
canonical expansion of `K` (corners at local rows 1/3/5; edge midpoints at
2/4/6; centroid bubble at 7). Pass `K7` explicitly to bypass the default
expansion — useful when the caller already holds a canonical 7-DOF mesh
and wants `geometric_fem2d_P2` to reuse it verbatim instead of regenerating
the intermediate vertices.
"""
function geometric_fem2d_P2(::Type{T}=Float64; L::Int=2,
                    K::Matrix{T}=T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],
                    K7::Matrix{T} = let R = reference_triangle(T)
                                       Matrix{T}(blockdiag([R.K for _ in 1:(size(K,1)÷3)]...) * K)
                                    end,
                    structured::Bool=true,rest...) where {T}
    disc = FEM2D_P2{T}(K, L, K7)
    structured ? subdivide_structured(disc) : subdivide(disc)
end
# subdivide method for FEM2D_P2 - generates the multigrid hierarchy
function subdivide(discretization::FEM2D_P2{T}) where {T}
    L=discretization.L
    K7=discretization.K7
    R = reference_triangle(T)
    x = Array{Array{T,2},1}(undef,(L,))
    nn = Int(size(K7,1)/7)
    x[1] = K7
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L-1
        refine[l] = blockdiag([R.refine for k=1:nn*4^(l-1)]...)
        coarsen[l] = blockdiag([R.coarsen for k=1:nn*4^(l-1)]...)
        x[l+1] = refine[l]*x[l]
    end
    n = size(x[L])[1]
    id = spdiagm(0=>ones(T,n))
    N = Int(n/7)
    dx = Array{SparseMatrixCSC{T,Int},1}(undef,(N,))
    dy = Array{SparseMatrixCSC{T,Int},1}(undef,(N,))
    w = Array{Vector{T},1}(undef,(N,))
    xL = reshape(x[L]',(2,7,N))
    for k=1:N
        u = xL[:,1,k]-xL[:,5,k]
        v = xL[:,3,k]-xL[:,5,k]
        A = hcat(u,v)
        invA = inv(A)'
        dx[k] = invA[1,1]*R.dx+invA[1,2]*R.dy
        dy[k] = invA[2,1]*R.dx+invA[2,2]*R.dy
        w[k] = abs(det(A))*R.w
    end
    dx = blockdiag(dx...)
    dy = blockdiag(dy...)
    w = vcat(w...)
    refine[L] = id
    coarsen[L] = id
    for l=1:L
        dirichlet[l] = continuous(x[l])
        full[l] = spdiagm(0=>ones(T,size(x[l],1)))
        N = size(x[l])[1]
        uniform[l] = sparse(ones(T,(N,1)))
    end
    subspaces = Dict(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict(:id => id, :dx => dx, :dy => dy)
    return Geometry{T,Matrix{T},Vector{T},SparseMatrixCSC{T,Int},FEM2D_P2{T}}(discretization,
        x[end],w,subspaces,operators,refine,coarsen)
end

# Direct structured construction — builds block types without sparse intermediates
function subdivide_structured(discretization::FEM2D_P2{T}) where {T}
    L = discretization.L
    K7 = discretization.K7
    R = reference_triangle(T)
    p = 7  # block size

    # Use the (possibly user-supplied) 7-DOF mesh directly as the coarsest level.
    nn = Int(size(K7, 1) / 7)
    x = Array{Matrix{T}, 1}(undef, L)
    x[1] = K7

    # Reference refine/coarsen as dense matrices
    ref_dense = Matrix(R.refine)   # 28×7
    coar_dense = Matrix(R.coarsen) # 7×28

    # K_refine = 28/7 = 4 sub-blocks of 7×7 for VBlockDiag
    K_refine = 4

    # Build refine/coarsen and propagate coordinates
    N_blocks = nn * 4^(L-1)

    # Create identity V/HBlockDiag first to determine type
    id_data = zeros(T, p, p, N_blocks)
    for i in 1:N_blocks
        for j in 1:p
            id_data[j, j, i] = one(T)
        end
    end
    id_vbd = VBlockDiag(p, p, 1, N_blocks, id_data)
    id_hbd = HBlockDiag(p, p, 1, N_blocks, copy(id_data))

    refine = Vector{typeof(id_vbd)}(undef, L)
    coarsen = Vector{typeof(id_hbd)}(undef, L)

    for l in 1:L-1
        n_l = nn * 4^(l-1)
        ref_data = zeros(T, p, p, K_refine * n_l)
        coar_data = zeros(T, p, p, K_refine * n_l)
        for i in 1:n_l
            for s in 1:K_refine
                ref_data[:, :, (i-1)*K_refine + s] = ref_dense[(s-1)*p+1:s*p, :]
                coar_data[:, :, (i-1)*K_refine + s] = coar_dense[:, (s-1)*p+1:s*p]
            end
        end
        refine[l] = VBlockDiag(p, p, K_refine, n_l, ref_data)
        coarsen[l] = HBlockDiag(p, p, K_refine, n_l, coar_data)
        x[l+1] = refine[l] * x[l]
    end

    # Level L: identity
    refine[L] = id_vbd
    coarsen[L] = id_hbd

    # Build operators directly as BlockDiag
    n = size(x[L], 1)
    N = Int(n / p)
    xL = reshape(x[L]', (2, p, N))

    # Reference derivative matrices as dense
    R_dx = Matrix(R.dx)  # 7×7
    R_dy = Matrix(R.dy)  # 7×7

    id_block = zeros(T, p, p, N)
    dx_block = zeros(T, p, p, N)
    dy_block = zeros(T, p, p, N)
    w_vec = zeros(T, n)

    for k in 1:N
        u = xL[:, 1, k] - xL[:, 5, k]
        v = xL[:, 3, k] - xL[:, 5, k]
        A = hcat(u, v)
        invA = inv(A)'
        dx_block[:, :, k] = invA[1, 1] * R_dx + invA[1, 2] * R_dy
        dy_block[:, :, k] = invA[2, 1] * R_dx + invA[2, 2] * R_dy
        for j in 1:p
            id_block[j, j, k] = one(T)
        end
        w_blk = abs(det(A)) * R.w
        w_vec[(k-1)*p+1:k*p] = w_blk
    end

    id = BlockDiag(id_block)
    dx = BlockDiag(dx_block)
    dy = BlockDiag(dy_block)

    # Subspaces stay sparse
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    full = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    uniform = Array{SparseMatrixCSC{T,Int},1}(undef, L)
    for l in 1:L
        dirichlet[l] = continuous(x[l])
        full[l] = spdiagm(0 => ones(T, size(x[l], 1)))
        uniform[l] = sparse(ones(T, (size(x[l], 1), 1)))
    end

    subspaces = Dict(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict(:id => id, :dx => dx, :dy => dy)
    return Geometry{T, Matrix{T}, Vector{T}, BlockDiag{T,Array{T,3}},
                    VBlockDiag{T,Array{T,3}}, HBlockDiag{T,Array{T,3}},
                    SparseMatrixCSC{T,Int}, FEM2D_P2{T}}(
        discretization, x[end], w_vec, subspaces, operators, refine, coarsen)
end

function plot(M::Geometry{T, Matrix{T}, Vector{T}, <:Any, <:Any, <:Any, <:Any, FEM2D_P2{T}}, z::Vector{T}; kwargs...) where {T}
    x = M.x[:,1]
    y = M.x[:,2]
    S = [1 2 7
         2 3 7
         3 4 7
         4 5 7
         5 6 7
         6 1 7]
    N = Int(size(x,1)/7)
    S = vcat([S.+(7*k) for k=0:N-1]...)
    plot_trisurf(x,y,z,triangles=S .- 1; kwargs...)
end

_default_block_size(::FEM2D_P2) = 7
