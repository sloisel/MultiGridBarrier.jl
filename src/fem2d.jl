export fem2d, FEM2D, fem2d_solve

"    abstract type FEM2D end"
abstract type FEM2D end

"    fem2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM2D,rest...)"
fem2d_solve(::Type{T}=Float64;rest...) where {T} = amg_solve(T;method=FEM2D,rest...)
"    amg_dim(::Type{FEM2D}) = 2"
amg_dim(::Type{FEM2D}) = 2
"    amg_construct(::Type{T},::Type{FEM2D};rest...) where {T} = fem2d(T;rest...)"
amg_construct(::Type{T},::Type{FEM2D};rest...) where {T} = fem2d(T;rest...)


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

function continuous(x::Matrix{T};
                    tol=maximum(abs.(x))*10*eps(T)) where {T}
    n = size(x)[1]
    a = 1
    u = randn(T,2)
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
#    W = spdiagm(0=>1 ./ reshape(sum(C,dims=(1,)),(count,)))
#    C = C*W
    C[:,interior]
end


"""
    function fem2d(::Type{T}, L::Int, K::Matrix{T};
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :u :dy
                         :s :id],
                    generate_feasibility=true) where {T}

Construct an `AMG` object for a 2d finite element grid on the domain `K` with piecewise quadratic elements.
Parameters are:
* `K`: a triangular mesh. If there are `n` triangles, then `K` should be a 3n by 2 matrix of vertices. The first column of `K` represents `x` coordinates, the second column represents `y` coordinates.
* `L`: number of refinement levels (L for Levels).
* `state_variables`: the "state vector" consists of functions, by default this is `u(x)` and `s(x)`, on the finite element grid.
* `D`: the set of differential operators. The barrier function `F` will eventually be called with the parameters `F(x,y,Dz)`, where `z` is the state vector. By default, this results in `F(x,y,u,ux,uy,s)`, where `(ux,uy)` is the gradient of `u`.
* `generate_feasibility`: if `true`, returns a pair `M` of `AMG` objects. `M[1]` is the `AMG` object for the main problem, and `M[2]` is for the feasibility subproblem.
"""
function fem2d(::Type{T}=Float64; L::Int=2, n=nothing,
                    K=T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :u :dy
                         :s :id],
                    generate_feasibility=true) where {T}
    K = if isnothing(K) T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1] else K end
    R = reference_triangle(T)
    x = Array{Array{T,2},1}(undef,(L,))
    nn = Int(size(K,1)/3)
    x[1] = blockdiag([R.K for k=1:nn]...)*K
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
    return amg(FEM2D,x=x[L],w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen,
        generate_feasibility=generate_feasibility)
end

"""
    function amg_plot(M::AMG{T, Mat,FEM2D}, z::Vector{T}) where {T,Mat}

Plot a piecewise quadratic (plus cubic "bubble") solution `z` on the given mesh. Note that the solution is drawn as (linear) triangles, even though the underlying solution is piecewise cubic. To obtain a more accurate depiction, especially when the mesh is coarse, it would be preferable to apply a few levels of additional subdivision, so as to capture the curve of the quadratic basis functions.
"""
function amg_plot(M::AMG{T, Mat,FEM2D}, z::Array{T}) where {T,Mat}
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
    plot_trisurf(x,y,z,triangles=S .- 1)
end
