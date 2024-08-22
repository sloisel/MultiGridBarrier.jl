export spectral1d_interp, spectral2d_interp, spectral1d_plot, spectral2d_plot, spectral1d, spectral1d_, spectral2d, spectral1d_solve, spectral2d_solve

function chebfun(c::Array{T,2}, x::T) where {T}
    n = size(c,1)-1
    m = size(c,2)
    if x>1
        return c'*cosh.((0:n).*acosh(x))
    elseif x>=-1
        return c'*cos.((0:n).*acos(x))
    end
    s = ones(T,n)
    s[2:2:n] .= T(-1)
    return c'*(s.*cosh.((0:n).*acosh(-x)))
end
function chebfun(c::Array{T}, x::Array{T}) where {T}
    sc = size(c)
    sx = size(x)
    c = reshape(c,(sc[1],:))
    m = size(c,2)
    n = prod(sx)
    x = reshape(x,n)
    y = zeros(T,n,m)
    for k=1:n
        y[k,:] = chebfun(c,x[k])
    end
    if length(sc)==1
        return reshape(y,sx)
    end
    return reshape(y,(sx...,sc[2:end]...))
end
chebfun(c::Array{T,1}, x::T) where {T} = chebfun(c,[x])[1]

function derivative(::Type{T},n::Integer) where {T}
    D = zeros(T,(n,n))
    for j=1:n-1
        for k=j+1:2:n
            D[j,k] = 2*(k-1)
        end
    end
    D[1,:]/=2
    D
end
derivative(n::Integer) = derivative(Float64,n)

function evaluation(xs::Array{T},n::Integer) where {T}
    m = size(xs,1)
    n = n-1
    M = zeros(T,(m,n+1))
    for j=1:m
        x = xs[j]
        if x>1
            M[j,:] = cosh.((0:n).*acosh(x))
        elseif x>=-1
            M[j,:] = cos.((0:n).*acos(x))
        else
            s = ones(T,n+1)
            s[2:2:n+1] .= T(-1)
            M[j,:] = s.*cosh.((0:n).*acosh(-x))
        end
    end
    M
end

function spectral1d_(::Type{T}, n::Integer;
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}
    L = Int(ceil(log2(n)))
    ls = [min(n,2^k) for k=1:L]
    x = Array{Array{T,2},1}(undef,(L,))
    w = 0
    dirichlet = Array{Array{T,2},1}(undef,(L,))
    full = Array{Array{T,2},1}(undef,(L,))
    uniform = Array{Array{T,2},1}(undef,(L,))
    refine = Array{Array{T,2},1}(undef,(L,))
    coarsen = Array{Array{T,2},1}(undef,(L,))
    M = "hi"
    for l=1:L
        Q = ClenshawCurtisQuadrature(T,ls[l])
        nodes,weights = Q.nodes,Q.weights
#        nodes,weights = gausslegendre(T,ls[l])
        w = 2 .* weights
        x[l] = reshape(2 .* nodes .- 1,(length(w),1))
        M = evaluation(x[l],ls[l])
        @assert size(M,1)==size(M,2)
        CI = M[:,3:end]
        for k=1:2:size(CI,2)
            CI[:,k] -= M[:,1]
        end
        for k=2:2:size(CI,2)
            CI[:,k] -= M[:,2]
        end
        dirichlet[l] = CI
        full[l] = M
        uniform[l] = ones(T,(size(x[l],1),1))
    end
    D0 = derivative(T,ls[L])
    @assert size(D0,1)==size(D0,2)
    dx = M*D0/M
    id = Matrix{T}(I,ls[L],ls[L])
    refine[L] = id
    coarsen[L] = id
    for l=1:L-1
        refine[l] = evaluation(x[l+1],ls[l])/full[l]
        coarsen[l] = evaluation(x[l],ls[l+1])/full[l+1]
    end
    subspaces = Dict{Symbol,Array{Array{T,2},1}}(:dirichlet => dirichlet, :full => full, :uniform => uniform)
    operators = Dict{Symbol,Array{T,2}}(:id => id, :dx => dx)
    
    return (x=x[L],w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen, 
        generate_feasibility=generate_feasibility)
end
"""
    function spectral1d(::Type{T}=Float64; n::Integer=5,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}

Construct an `AlgebraicMultiGridBarrier.AMG` object for a 1d spectral grid of polynomials of degree `n-1`. See also `fem1d` for a description of the parameters `state_variables` and `D`.
"""
function spectral1d(::Type{T}=Float64; n=nothing, L::Integer=2,
                    K=nothing,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}
    n = if isnothing(n) 2^L else n end
    return amg(;spectral1d_(T,n,state_variables=state_variables,D=D,generate_feasibility=generate_feasibility)...)
end

"""
    function spectral1d_interp(MM::AMG{T,Mat}, y::Array{T,1},x) where {T,Mat}

A function to interpolate a solution `y` at some point(s) `x`.

* `MM` the mesh of the solution.
* `y` the solution.
* `x` point(s) at which the solution should be evaluated.
"""
function spectral1d_interp(MM::AMG{T,Mat}, y::Array{T,1},x) where {T,Mat}
    n = length(MM.w)
    M = evaluation(MM.x,n)
    m1 = size(M,1)
    @assert m1==size(M,2)
    sz = size(y)
    y1 = reshape(y,(m1,:))
    z = chebfun(M\y1,x)
    if length(sz)==1
        ret = z        
    else
        ret = reshape(z,(size(x)...,sz[2:end]...))
    end
    ret
end

"""
    function spectral1d_plot(M::AMG{T,Mat},x,y,rest...) where {T,Mat}

Plot a solution using `pyplot`.

* `M`: a mesh.
* `x`: x values where the solution should be evaluated and plotted.
* `y`: the solution, to be interpolated at the given `x` values via `spectral1d_interp`.
* `rest...` parameters are passed directly to `pyplot.plot`.
"""
function spectral1d_plot(M::AMG{T,Mat},x,y,rest...) where {T,Mat}
    plot(Float64.(x),Float64.(spectral1d_interp(M,y,x)),rest...)
end

"""
    function spectral2d(::Type{T}=Float64; n=5::Integer,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :u :dy
                         :s :id],
                    generate_feasibility=true) where {T}

Construct an `AMG` object for a 2d spectral grid of degree `n-1`. See also `fem2d` for a description of `state_variables` and `D`.
"""
function spectral2d(::Type{T}=Float64; n=nothing,
                    L::Integer=2,
                    K=nothing,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :u :dy
                         :s :id],
                    generate_feasibility=true) where {T}
    if isnothing(n)
        n = 2^L
    end
    M = spectral1d_(T,n,state_variables=state_variables,D=D)
    L = Int(ceil(log2(n)))
    ls = [min(n,2^k) for k=1:L]
    w = M.w
    N = length(w)
    w = reshape(w,(N,1))
    w = w*(w')
    w = reshape(w,(N*N,))
    dirichlet = Array{Array{T,2},1}(undef,(L,))
    full = Array{Array{T,2},1}(undef,(L,))
    uniform = Array{Array{T,2},1}(undef,(L,))
    refine = Array{Array{T,2},1}(undef,(L,))
    coarsen = Array{Array{T,2},1}(undef,(L,))
    for l=1:L
        S = M.subspaces
        dirichlet[l] = kron(S[:dirichlet][l],S[:dirichlet][l])
        full[l] = kron(S[:full][l],S[:full][l])
        uniform[l] = kron(S[:uniform][l],S[:uniform][l])
        refine[l] = kron(M.refine[l],M.refine[l])
        coarsen[l] = kron(M.coarsen[l],M.coarsen[l])        
    end
    xl = M.x
    N = size(xl)[1]
    y = reshape(repeat(xl,outer=(1,N)),(N*N,1))
    z = reshape(repeat(xl,outer=(1,N))',(N*N,1))
    x = hcat(y,z)
    ID = M.operators[:id]
    DX = M.operators[:dx]
    id = kron(ID,ID)
    dx = kron(DX,ID)
    dy = kron(ID,DX)
    subspaces = Dict{Symbol,Array{Array{T,2},1}}(:dirichlet => dirichlet, :full => full, :uniform=>uniform)
    operators = Dict{Symbol,Array{T,2}}(:id => id, :dx => dx, :dy => dy)
    return amg(x=x,w=w,state_variables=state_variables,
        D=D,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen,
        generate_feasibility=generate_feasibility)
end


"""
    function spectral2d_interp(MM::AMG{T,Mat},z::Array{T,1},x::Array{T,2}) where {T,Mat}

Interpolate a solution `z` at point(s) `x`, given the mesh `MM`. See also
`spectral1d_interp`.
"""
function spectral2d_interp(MM::AMG{T,Mat},z::Array{T,1},x::Array{T,2}) where {T,Mat}
#    n = MM.n
#    M = spectralmesh(T,n)
    m1 = Int(sqrt(size(MM.x,1)))
    M = spectral1d(T, n=m1)
    Z0 = zeros(T,m1)
    function interp0(z::Array{T,1},x::T,y::T)
        ZW = reshape(z,(m1,m1))
        for k=1:m1
            Z0[k] = spectral1d_interp(M[1],ZW[:,k],x)[1]
        end
        spectral1d_interp(M[1],Z0,y)[1]
    end
    function interp1(z::Array{T,1},x::T,y::T)
        ZZ = reshape(z,(m1*m1,:))
        ret1 = zeros(T,size(ZZ,2))
        for k1=1:size(ZZ,2)
            ret1[k1] = interp0(ZZ[:,k1],x,y)
        end
        ret1
    end
    function interp(z::Array{T,1},x::Array{T,2})
        m = Int(size(z,1)/(m1*m1))
        ret2 = zeros(T,(size(x,1),m))
        for k2=1:size(x,1)
            foo = interp1(z,x[k2,1],x[k2,2])
            ret2[k2,:] = foo
        end
        ret2[:]
    end
    interp(z,x)
end

"""
    function spectral2d_plot(M::Mesh{T},x,y,z::Array{T,1};rest...) where {T}

Plot a 2d solution.

* `M` a 2d mesh.
* `x`, `y` should be ranges like -1:0.01:1.
* `z` the solution to plot.
"""
function spectral2d_plot(M::AMG{T,Mat},x,y,z::Array{T,1};rest...) where {T,Mat}
    X = repeat(x,1,length(y))
    Y = repeat(y,1,length(x))'
    sz = (length(x),length(y))
    Z = reshape(spectral2d_interp(M,z,hcat(X[:],Y[:])),(length(x),length(y)))
    gcf().add_subplot(projection="3d")
    dx = maximum(x)-minimum(x)
    dy = maximum(y)-minimum(y)
    lw = max(dx,dy)*0.002
    plot_surface(Float64.(x), Float64.(y), Float64.(Z); rcount=50, ccount=50, antialiased=false, edgecolor=:black, linewidth=Float64(lw), rest...)
#        plot_wireframe(x,y,Z; rcount=10, ccount=10, color=:white, edgecolor=:black)
end
