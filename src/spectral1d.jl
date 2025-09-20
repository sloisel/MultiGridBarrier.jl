export spectral1d, SPECTRAL1D, spectral1d_solve

"    abstract type SPECTRAL1D end"
struct SPECTRAL1D 
    n::Int
end

"    amg_dim(::Type{SPECTRAL1D}) = 1"
amg_dim(::SPECTRAL1D) = 1
"    spectral1d_solve(::Type{T}=Float64;rest...) where {T} = amgb_solve(T;method=SPECTRAL1D,rest...)"
spectral1d_solve(::Type{T}=Float64;rest...) where {T} = amgb(T,spectral1d(T;rest...);rest...)

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

function spectral1d_(::Type{T}, n::Integer) where {T}
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
    
    return (x=x[L],w=w,subspaces=subspaces,operators=operators,refine=refine,coarsen=coarsen)
end
"""
    spectral1d(::Type{T}=Float64; n=nothing, L::Integer=2,
                    K=nothing,
                    state_variables = [:u :dirichlet
                                       :s :full],
                    D = [:u :id
                         :u :dx
                         :s :id],
                    generate_feasibility=true) where {T}

Construct an `AMG` object for a 1d spectral grid of polynomials of degree `n-1`. See also `fem1d` for a description of the parameters `state_variables` and `D`.
"""
spectral1d(::Type{T}=Float64;n=16,rest...) where {T} = SPECTRAL1D(n)

function subdivide(::Type{T}, geometry::SPECTRAL1D;state_variables=[:u :dirichlet ; :s :full],D=[:u :id;:u :dx;:s :id], generate_feasibility=true) where {T}
    return amg(geometry;state_variables,D,generate_feasibility,spectral1d_(T,geometry.n)...)
end

"""
    spectral1d_interp(MM::AMG{T,Mat,SPECTRAL1D}, y::Array{T,1},x) where {T,Mat}

A function to interpolate a solution `y` at some point(s) `x`.

* `MM` the mesh of the solution.
* `y` the solution.
* `x` point(s) at which the solution should be evaluated.
"""
function spectral1d_interp(MM::AMG{T,Mat,SPECTRAL1D}, y::Array{T,1},x) where {T,Mat}
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

interpolate(M::AMG{T,Mat,SPECTRAL1D}, z::Vector{T}, t) where {T,Mat} = spectral1d_interp(M,z,t)

"""
    amg_plot(M::AMG{T,Mat,SPECTRAL1D},y;x=Array(-1:T(0.01):1),rest...) where {T,Mat}

Plot a solution using `pyplot`.

* `M`: a mesh.
* `x`: x values where the solution should be evaluated and plotted.
* `y`: the solution, to be interpolated at the given `x` values via `spectral1d_interp`.
* `rest...` parameters are passed directly to `pyplot.plot`.
"""
function PyPlot.plot(M::AMG{T,Mat,SPECTRAL1D},y;x=Array(-1:T(0.01):1),rest...) where {T,Mat}
    plot(Float64.(x),Float64.(interpolate(M,y,x)),rest...)
end

