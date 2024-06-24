export fem1d, fem2d, fem_solve1d, fem_interp1d, fem_solve2d, fem_plot2d

"""
    function fem1d(::Type{T}, L::Int;
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id]) where {T}

Construct an `AMG` object for a 1d piecewise linear finite element grid. The interval is [-1,1]. Parameters are:
* `L`: divide the interval into 2^L subintervals (L for Levels).
* `state_variables`: the "state vector" consists of functions, by default this is `u(x)` and `s(x)`, on the finite element grid.
* `D`: the set of differential operator. The barrier function `F` will eventually be called with the parameters `F(x,Dz)`, where `z` is the state vector. By default, this results in `F(x,u,ux,s)`, where `ux` is the derivative of `u`.
"""
function fem1d(::Type{T}, L::Int;
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id]) where {T}
    ls = [2^k for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    refine = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    coarsen = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    for l=1:L
        n0 = 2^l
        x[l] = reshape(hcat((0:n0-1)./T(n0),(1:n0)./T(n0))',(2*n0,1)) .* 2 .- 1
        N = size(x[l])[1]
        dirichlet[l] = vcat(spzeros(T,1,n0-1),blockdiag(repeat([sparse(T[1 ; 1 ;;])],outer=(n0-1,))...),spzeros(T,1,n0-1))
        full[l] = sparse(T,I,N,N)
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
    subspaces = Dict(:dirichlet => dirichlet, :full => full)
    operators = Dict(:id => id, :dx => dx)
    return amg(x=x[L],w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen)
end

"""
    function fem_solve1d(::Type{T}; g = x->x,
        f = x->T(0.5), maxit=10000, L=2, p=T(1.0),
        verbose=true, show=true, tol=sqrt(eps(T)),
        F = (x,u,ux,s) -> -log(s^(2/p)-ux^2)-2*log(s),
        slack = x->T(2)) where {T}

Solve a 1d variational problem on the interval [-1,1] with piecewise linear elements. Parameters are:
* `g` the boundary conditions.
* `f` the forcing function.
* `maxit` a maximum number of iterations used in the solver.
* `L` the number of Levels of grid subdivisions, so that the grid consists of 2^L intervals.
* `p` the parameter of the p-Laplace problem, if that's what we're solving.
* `verbose`: set to `true` to get a progress bar.
* `tol`: a stopping criterion, the barrier method stops when `t>1/tol`.
* `F`: the barrier. The default barrier solves a p-Laplacian.
* `slack`: an initializer for the slack function `s(x)`.

This function returns `SOL,B`, where `SOL` is from `amgb`, and `B` is the `Barrier` object obtained from `F`.
"""
function fem_solve1d(::Type{T}; g = x->x,
        f = x->T(0.5), maxit=10000, L=2, p=T(1.0),
        verbose=true, show=true, tol=sqrt(eps(T)),
        F = (x,u,ux,s) -> -log(s^(2/p)-ux^2)-2*log(s),
        slack = x->T(2)) where {T}
    M = fem1d(T,L)
    u0 = g.(M.x[end][:,1])
    fh = f.(M.x[end][:,1])
    c = hcat(fh,zeros(T,size(fh)),ones(T,size(fh)))
    B = barrier((x,y)->F(x...,y...))
    x0 = vcat(u0,slack.(M.x[end][:,1]))
    SOL = amgb(B,M,x0,c,
        kappa=T(10),maxit=maxit,verbose=verbose,tol=tol)
    if show
        xs = Array(-1:T(0.01):1)
        plot(M.x[end],M.D[end,1]*SOL.z)
    end
    SOL
end

"""
    function fem_interp1d(x::Vector{T},
                      y::Vector{T},
                      t::T) where{T}

Interpolate a 1d piecewise linear function at the given `t` value. If `u(xi)` is the piecewise linear function such that `u(x[k])=y[k]` then this function returns `u(t)`.
"""
function fem_interp1d(x::Vector{T},
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

"""
function fem_interp1d(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}

Returns `[fem_interp1d(x,y,t[k]) for k=1:length(t)]`.
"""
function fem_interp1d(x::Vector{T},
                      y::Vector{T},
                      t::Vector{T}) where{T}
    [fem_interp1d(x,y,t[k]) for k=1:length(t)]
end

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
                         :s :id]) where {T}

Construct an `AMG` object for a 2d finite element grid on the domain `K` with piecewise quadratic elements.
Parameters are:
* `K`: a triangular mesh. If there are `n` triangles, then `K` should be a 3n by 2 matrix of vertices. The first column of `K` represents `x` coordinates, the second column represents `y` coordinates.
* `L`: divide the interval into 2^L subintervals (L for Levels).
* `state_variables`: the "state vector" consists of functions, by default this is `u(x)` and `s(x)`, on the finite element grid.
* `D`: the set of differential operator. The barrier function `F` will eventually be called with the parameters `F(x,y,Dz)`, where `z` is the state vector. By default, this results in `F(x,y,u,ux,uy,s)`, where `(ux,uy)` is the gradient of `u`.
"""
function fem2d(::Type{T}, L::Int, K::Matrix{T};
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :u :dy
                         :s :id]) where {T}
    R = reference_triangle(T)
    x = Array{Array{T,2},1}(undef,(L,))
    nn = Int(size(K,1)/3)
    x[1] = blockdiag([R.K for k=1:nn]...)*K
    dirichlet = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
    full = Array{SparseMatrixCSC{T,Int},1}(undef,(L,))
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
#    show(stdout, "text/plain",x[L])
#    show(stdout, "text/plain",xL)
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
    end
    subspaces = Dict(:dirichlet => dirichlet, :full => full)
    operators = Dict(:id => id, :dx => dx, :dy => dy)
    return amg(x=x[L],w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen)
end

"""
    function fem_plot2d(M::AMG{T, Mat}, z::Array{T}) where {T,Mat}

Plot a piecewise quadratic solution `z` on the given mesh. Note that the solution is drawn as (linear) triangles, even though the underlying solution is piecewise quadratic. To obtain a more accurate depiction, especially when the mesh is coarse, it would be preferable to apply a few levels of additional subdivision, so as to capture the curve of the quadratic basis functions.
"""
function fem_plot2d(M::AMG{T, Mat}, z::Array{T}) where {T,Mat}
    x = M.x[end][:,1]
    y = M.x[end][:,2]
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

"""
    function fem_solve2d(::Type{T}; 
        K = T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],
        g = (x,y)->x^2+y^2, 
        f = (x,y)->T(0.5), maxit=10000, L=2, p=T(1.0),
        verbose=true, show=true, tol=sqrt(eps(T)),
        F = (x,y,u,ux,uy,s) -> -log(s^(2/p)-ux^2-uy^2)-2*log(s),
        slack = (x,y)->T(100)) where {T}

Solve a 2d variational problem on the domain `K`, which defaults to the square [-1,1]x[-1,1], with piecewise quadratic elements. Parameters are:
* `K` a triangulation of the domain. For `n` triangles, `K` should be a 3n by 2 matrix of vertices.
* `g` the boundary conditions.
* `f` the forcing function.
* `maxit` a maximum number of iterations used in the solver.
* `L` the number of Levels of grid subdivisions, so that the grid consists of `N = n*4^L` quadratic triangular elements. Each elements is quadratic, plus a bump function, so each element consists of 7 vertices, i.e. there are `7*N` vertices in total.
* `p` the parameter of the p-Laplace problem, if that's what we're solving.
* `verbose`: set to `true` to get a progress bar.
* `tol`: a stopping criterion, the barrier method stops when `t>1/tol`.
* `F`: the barrier. The default barrier solves a p-Laplacian.
* `slack`: an initializer for the slack function `s(x)`.

This function returns `SOL,B`, where `SOL` is from `amgb`, and `B` is the `Barrier` object obtained from `F`.
"""
function fem_solve2d(::Type{T}; 
        K = T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],
        g = (x,y)->x^2+y^2, 
        f = (x,y)->T(0.5), maxit=10000, L=2, p=T(1.0),
        verbose=true, show=true, tol=sqrt(eps(T)),
        F = (x,y,u,ux,uy,s) -> -log(s^(2/p)-ux^2-uy^2)-2*log(s),
        slack = (x,y)->T(100)) where {T}
    M = fem2d(T,L,K)
    u0 = g.(M.x[end][:,1],M.x[end][:,2])
    fh = f.(M.x[end][:,1],M.x[end][:,2])
    c = hcat(fh,zeros(T,size(fh)),zeros(T,size(fh)),ones(T,size(fh)))
    B = barrier((x,y)->F(x...,y...))
    x0 = vcat(u0,slack.(M.x[end][:,1],M.x[end][:,2]))
    SOL = amgb(B,M,x0,c,
        kappa=T(10),maxit=maxit,verbose=verbose,tol=tol)
    if show
        x = M.x[end]
        z = M.D[end,1]*SOL.z
        fem_plot2d(M,z)
    end
    SOL
end

function fem_precompile()
    fem_solve1d(Float64,L=1)
    fem_solve2d(Float64,L=1)
end

precompile(fem_precompile,())
