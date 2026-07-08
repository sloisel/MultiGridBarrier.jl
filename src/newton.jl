# Newton iteration, line searches (Illinois/backtracking), and stopping criteria.
# Included into module MultiGridBarrier from AlgebraicMultiGridBarrier.jl.

function illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}
    @assert isfinite(fa) && isfinite(fb)
    if fa==0
        return a
    end
    if fa*fb>=0
        return b
    end
    for k=1:maxit
        c = (a*fb-b*fa)/(fb-fa)
        fc = f(c)
        @assert isfinite(fc)
        if c<=min(a,b) || c>=max(a,b) || fc*fa==0 || fc*fb==0
            return c
        end
        if fb*fc<0
            a,fa = b,fb
        else
            fa /= 2
        end
        b,fb = c,fc
    end
    error("Illinois solver failed to converge.")
end

# Shared trial-step loop for the line searches. `attempt(s)` computes a trial
# at scale `s` and returns `(xnext, ynext, gnext, done)`; it throws to reject
# the trial. The catch is broad on purpose: there is no fixed protocol for a
# barrier to signal domain escape (a DomainError from log, non-finite values,
# an InexactError, complex results, ...), so any failure rejects the step and
# shrinks `s` by `beta` — except an InterruptException, which aborts the solve.
function _linesearch_loop(attempt, x::V, y::T, g::V, beta::T; printlog) where {V,T}
    s = T(1)
    xnext, ynext, gnext = x, y, g
    while s > T(0)
        @debug("s=",s)
        try
            xnext, ynext, gnext, done = attempt(s)
            done && break
        catch e
            e isa InterruptException && rethrow()
            @debug("line search: trial step rejected", exception=e)
        end
        s = s*beta
    end
    return (xnext,ynext,gnext)
end

@doc raw"""
    linesearch_illinois(::Type{T}=Float64; beta=T(0.5)) where {T}

Create an Illinois-based line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction when Illinois fails (default: 0.5).

# Returns
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : Newton direction (typically H\g where H is the Hessian).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
The Illinois algorithm finds a root of `φ(s) = ⟨∇F(x - s*n), n⟩`, which corresponds to
the exact line search condition. If the Illinois solver fails or encounters numerical
issues, the step size is reduced by factor `beta` and the process repeats.

# Notes
This line search strategy aims for the exact minimizer along the search direction,
making it potentially more aggressive than backtracking but also more expensive per iteration.
"""
function linesearch_illinois(::Type{T}=Float64;beta=T(0.5)) where {T}
    function ls_illinois(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
        inc = dot(g,n)
        function attempt(s)
            function phi(sigma)
                xn = x-sigma*n
                isfinite(F0(xn)) || error("line search: non-finite barrier value")
                return dot(F1(xn),n)
            end
            s = illinois(phi,T(0),s,fa=inc)
            xnext = x-s*n
            ynext,gnext = F0(xnext)::T, F1(xnext)
            isfinite(ynext) && mgb_all_isfinite(gnext) || error("line search: non-finite step")
            (xnext, ynext, gnext, true)
        end
        _linesearch_loop(attempt, x, y, g, beta; printlog)
    end
    return ls_illinois
end

@doc raw"""
    linesearch_backtracking(::Type{T}=Float64; beta=T(0.5)) where {T}

Create a backtracking line search function for Newton methods.

# Arguments
* `T` : numeric type for computations (default: Float64).

# Keyword arguments
* `beta` : backtracking parameter for step size reduction (default: 0.5).
* `c1` : Armijo sufficient-decrease constant `c₁` (default: 0.1).

# Returns
A line search function `ls(x, y, g, n, F0, F1; printlog)` where:
* `x` : current point (vector of type T).
* `y` : current objective value F0(x).
* `g` : current gradient F1(x).
* `n` : search direction (typically Newton direction H\g).
* `F0` : objective function.
* `F1` : gradient function.
* `printlog` : logging function.

Returns `(xnext, ynext, gnext)` where `xnext = x - s*n` for some step size `s`.

# Algorithm
Implements the Armijo backtracking line search with sufficient decrease condition:
`F(x - s*n) ≤ F(x) - c₁ * s * ⟨∇F(x), n⟩`, where `c₁` is the keyword `c1`
(default 0.1). The step size starts at `s = 1` and is reduced by factor `beta`
until the condition is satisfied or numerical limits are reached.

# Notes
This is a robust and commonly used line search that guarantees sufficient decrease
in the objective function, making it suitable for general nonlinear optimization.
"""
function linesearch_backtracking(::Type{T}=Float64;beta = T(0.5), c1 = T(0.1)) where {T}
    function ls_backtracking(x::V,y::T,g::V,
        n::V,F0,F1;printlog) where {V}
        inc = dot(g,n)
        function attempt(s)
            xnext = x-s*n
            # GPU-compatible: use norm to check if the step made any difference
            stalled = norm(xnext - x) == 0
            ynext,gnext = F0(xnext)::T, F1(xnext)
            isfinite(ynext) && mgb_all_isfinite(gnext) || error("line search: non-finite step")
            (xnext, ynext, gnext, stalled || ynext <= y - c1*inc*s)
        end
        _linesearch_loop(attempt, x, y, g, beta; printlog)
    end
    return ls_backtracking
end

"""
    stopping_exact(theta::T) where {T}

Create an exact stopping criterion for Newton methods.

# Arguments
* `theta` : tolerance parameter for gradient norm relative decrease (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement.

# Algorithm
Returns `true` (stop) if both conditions hold:
1. No objective improvement: `ynext ≥ ymin`
2. Gradient norm stagnation: `‖gnext‖ ≥ theta * gmin`

# Notes
This criterion is "exact" in the sense that it requires both objective and gradient
stagnation before stopping, making it suitable for high-precision optimization.
Typical values of `theta` are in the range [0.1, 0.9].
"""
stopping_exact(theta::T) where {T} = (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->ynext>=ymin && norm(gnext)>=theta*gmin
"""
    stopping_inexact(lambda_tol::T, theta::T) where {T}

Create an inexact stopping criterion for Newton methods that combines Newton decrement
and exact stopping conditions.

# Arguments
* `lambda_tol` : tolerance for the Newton decrement (type T).
* `theta` : tolerance parameter for the exact stopping criterion (type T).

# Returns
A stopping criterion function with signature:
`stop(ymin, ynext, gmin, gnext, n, ndecmin, ndec) -> Bool`

where:
* `ymin` : minimum objective value seen so far.
* `ynext` : current objective value.
* `gmin` : minimum gradient norm seen so far.
* `gnext` : current gradient vector.
* `n` : current Newton direction.
* `ndecmin` : square root of minimum Newton decrement seen so far.
* `ndec` : square root of current Newton decrement (√(gᵀH⁻¹g)).

# Algorithm
Returns `true` (stop) if either condition holds:
1. Newton decrement condition: `ndec < lambda_tol`
2. Exact stopping condition: `stopping_exact(theta)` is satisfied

# Notes
This criterion is "inexact" because it allows early termination based on the Newton
decrement, which provides a quadratic convergence estimate. The Newton decrement
`λ = √(gᵀH⁻¹g)` approximates the distance to the optimum in the Newton metric.
Typical values: `lambda_tol ∈ [1e-6, 1e-3]`, `theta ∈ [0.1, 0.9]`.
"""
function stopping_inexact(lambda_tol::T,theta::T) where {T} 
    exact_stop = stopping_exact(theta)
    (ymin,ynext,gmin,gnext,n,ndecmin,ndec)->((ndec<lambda_tol || exact_stop(ymin,ynext,gmin,gnext,n,ndecmin,ndec)))
end

function newton(::Type{Mat}, ::Type{T},
                       F0::Function,
                       F1::Function,
                       F2::Function,
                       x::V;
                       maxit=10000,
                       stopping_criterion=stopping_exact(T(0.1)),
                       printlog,
                       line_search=linesearch_illinois(T),
        ) where {T,Mat,V}
    ys = T[]
    mgb_all_isfinite(x) || error("newton: initial point has non-finite entries")
    y = F0(x) ::T
    isfinite(y) || error("newton: initial objective value is not finite")
    ymin = y
    push!(ys,y)
    converged = false
    k = 0
    g = F1(x)
    mgb_all_isfinite(g) || error("newton: initial gradient has non-finite entries")
    ynext,xnext,gnext=y,x,g
    gmin = norm(g)
    incmin = T(Inf)
    while k<maxit && !converged
        k+=1
        H = F2(x) ::Mat
        n = solve(symmetric(H), g)
        mgb_all_isfinite(n) || error("newton: Newton direction has non-finite entries")
        inc = dot(g,n)
        @debug("k=",k," y=",y," ‖g‖=",norm(g), " λ^2=",inc)
        if inc<=0
            converged = true
            break
        end
        (xnext,ynext,gnext) = line_search(x,y,g,n,F0,F1;printlog)
        if stopping_criterion(ymin,ynext,gmin,gnext,n,sqrt(incmin),sqrt(inc))
            @debug("converged: ymin=",ymin," ynext=",ynext," ‖gnext‖=",norm(gnext)," λ=",sqrt(inc)," λmin=",sqrt(incmin))
            converged = true
        end
        x,y,g = xnext,ynext,gnext
        gmin = min(gmin,norm(g))
        ymin = min(ymin,y)
        incmin = min(inc,incmin)
        push!(ys,y)
    end
    if !converged
        @debug("diverge")
    end
    return (;x,y,k,converged,ys)
end

