var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MultiGridBarrier","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Markdown\nusing Pkg\nusing MultiGridBarrier\nv = string(pkgversion(MultiGridBarrier))\nmd\"# MultiGridBarrier $v\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"MultiGridBarrier is a Julia module for solving nonlinear convex optimization problems in function spaces, such as p-Laplace problems. When regularity conditions are satisfied, the solvers are quasi-optimal.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The MultiGridBarrier module features finite element and spectral discretizations in 1d and 2d.","category":"page"},{"location":"#Finite-elements","page":"Home","title":"Finite elements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"After installing MultiGridBarrier with the Julia package manager, in a Jupyter notebook, one solves a 1d p-Laplace problem as follows:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using PyPlot # hide\nusing MultiGridBarrier\nfem1d_solve(L=5,p=1.0,verbose=false);\nsavefig(\"fem1d.svg\"); nothing # hide\nclose() #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"A 2d p-Laplace problem:","category":"page"},{"location":"","page":"Home","title":"Home","text":"fem2d_solve(L=3,p=1.0,verbose=false);\nsavefig(\"fem2d.svg\"); nothing # hide\nclose() #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#Spectral-elements","page":"Home","title":"Spectral elements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Solve a 1d p-Laplace problem as follows:","category":"page"},{"location":"","page":"Home","title":"Home","text":"spectral1d_solve(n=40,p=1.0,verbose=false);\nsavefig(\"spectral1d.svg\"); nothing # hide\nclose() #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"A 2d p-Laplace problem:","category":"page"},{"location":"","page":"Home","title":"Home","text":"spectral2d_solve(n=5,p=1.5,verbose=false);\nsavefig(\"spectral2d.svg\"); nothing # hide\nclose() #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#Module-reference","page":"Home","title":"Module reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [MultiGridBarrier]\nOrder   = [:module]","category":"page"},{"location":"#MultiGridBarrier.MultiGridBarrier","page":"Home","title":"MultiGridBarrier.MultiGridBarrier","text":"module MultiGridBarrier\n\nModule MultiGridBarrier solves convex optimization problems in function spaces, for example, solving p-Laplace problems. We recommend to start with the functions fem1d_solve(), fem2d_solve(), spectral1d_solve(), spectral2d_solve(). These functions are sufficient to solve p-Laplace problems in 1d or 2d, using finite or spectral elements.\n\nFor more general use, the user will need to familiarize themselves with the basic ideas of convex optimization.\n\nOverview of convex optimization in function spaces by MultiGrid Barrier method.\n\nThe general idea is to build a multigrid hierarchy, represented by an AMG object, and barrier for a convex set, represented by a Barrier object, and then solve a convex optimization problem using the amgb() solver.\n\nTo generate the multigrid hierarchy represented by the AMG object, use either fem1d(), fem2d(), spectral1d() or spectral2d() functions. These constructors will assemble suitable AMG objects for either FEM or spectral discretizations, in 1d or 2d. One should think of these four constructors as being specialized in constructing some specific function spaces. A user can use the amg() constructor directly if custom function spaces are required, but this is more difficult.\n\nWe now describe the barrier function.\n\nAssume that Omega subset mathbbR^d is some open set. Consider the example of the p-Laplace problem on Omega. Let f(x) be a \"forcing\" (a function) on Omega, and 1 leq p  infty. One wishes to solve the minimization problem\n\nbeginequation\ninf_u int_Omega fu + nabla u_2^p  dx tag1\nendequation\n\nGenerally speaking, u will range in some function space, e.g. a space of differentiable functions satisfying homogeneous Dirichlet conditions. Under some conditions, minimizing (1) is equivalent to solving the p-Laplace PDE:\n\nnabla cdot (nabla u_2^p-2nabla u) = 1 over p f\n\nWe introduct the \"slack function\" s(x) and replace (1) with the following equivalent problem:\n\nbeginequation\ninf_s(x) geq nabla u(x)_2^p int_Omega fu + s  dx tag2\nendequation\n\nDefine the convex set mathcalQ =  (u(x)q(x)s(x))    s(x) geq q(x)_2^p , and\n\nz = beginbmatrix u  s endbmatrix qquad\nc^T = f01 qquad\nDz = beginbmatrix u  nabla u  s endbmatrix\n\nThen, (2) can be rewritten as\n\nbeginequation\ninf_Dz in mathcalQ int_Omega c^T(x)Dz(x)  dx tag3\nendequation\n\nRecall that a barrier for mathcalQ is a convex function mathcalF on mathcalQ such that mathcalF  infty in the interior of mathcalQ and mathcalF = infty on the boundary of mathcalQ. A barrier for the p-Laplace problem is:\n\nmathcalF(uqs) = int_Omega -log(s^2 over p - q_2^2) - 2log s  dx = int_Omega F(Dz(x))  dx\n\nThe central path z^*(t) minimizes, for each fixed t0, the quantity\n\nint_Omega tc^TDz + F(Dz)  dx\n\nAs t to infty, z^*(t) forms a minimizing sequence (or filter) for (3). We think of the function c(x) as the \"functional\" that we seek to minimize.\n\nThe Convex{T} type describes various convex sets (denoted Q above) by way of functions barrier(), cobarrier() and slack(). barrier is indeed a barrier for Q, cobarrier() is a barrier for a related feasibility problems, and slack() is used in solving the feasibility problem. Convex{T} objects can be created using the various convex_...() constructors, e.g. convex_Euclidian_power() for the p-Laplace problem.\n\nOnce one has AMG and Convex objects, and a suitable \"functional\" c, one uses the amgb() function to solve the optimization problem by the MultiGrid Barrier method, a variant of the barrier method (or interior point method) that is quasi-optimal for sufficiently regular problems.\n\n\n\n\n\n","category":"module"},{"location":"#Types-reference","page":"Home","title":"Types reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [MultiGridBarrier]\nOrder   = [:type]","category":"page"},{"location":"#MultiGridBarrier.AMG","page":"Home","title":"MultiGridBarrier.AMG","text":"@kwdef struct AMG{T,M}\n    ...\nend\n\nObjects of this type should probably be assembled by the constructor amg().\n\nA multigrid with L level. Denote by l between 1 and L, a grid level. Fields are:\n\nx::Array{Array{T,2},1} an array of L matrices. x[l] stores the vertices of the grid at multigrid level l.\nw::Array{Array{T,1},1} an array of L quadrature weights. w[l] corresponds to x[l].\nR_fine::Array{M,1} an array of L matrices. The columns of R_fine[l] are basis functions for the function space on grid level l, interpolated to the fine grid.\nR_coarse::Array{M,1} an array of L matrices. The columns of R_coarse[l] are basis functions for the function space on grid level l. Unlike R_fine[l], these basis functions are on grid level l, not interpolated to the fine grid.\nD::Array{M,2} an array of differential operators. For example, if the barrier parameters are to be u,ux,s, with ux the derivative of u, then D[l,:] = [I,Dx,I], where Dx is a numerical differentiation operator on grid level l.  \nrefine_u::Array{M,1} an array of L grid refinement matrices. If x[l] has n[l] vertices, then refine_u[l] is n[l+1] by n[l].\ncoarsen_u::Array{M,1} an array of L grid coarsening matrices. coarsen_u[l] is n[l] by n[l+1].\nrefine_z::Array{M,1} an array of L grid refining matrices for the \"state vector\" z. For example, if z contains the state functions u and s, then there are k=2 state functions, and refine_z[l] is k*n[l+1] by k*n[l].\ncoarsen_z::Array{M,1} an array of L grid coarsening matrices for the \"state vector\" z. coarsen_z[l] is k*n[l] by k*n[l+1].\n\nThese various matrices must satisfy a wide variety of algebraic relations. For this reason, it is recommended to use the constructor amg().\n\n\n\n\n\n","category":"type"},{"location":"#MultiGridBarrier.Barrier","page":"Home","title":"MultiGridBarrier.Barrier","text":"Barrier\n\nA type for holding barrier functions. Fields are:\n\nf0::Function\nf1::Function\nf2::Function\n\nf0 is the barrier function itself, while f1 is its gradient and f2 is the Hessian.\n\n\n\n\n\n","category":"type"},{"location":"#MultiGridBarrier.Convex","page":"Home","title":"MultiGridBarrier.Convex","text":"struct Convex\n    barrier::Function\n    cobarrier::Function\n    slack::Function\nend\n\nThe Convex data structure represents a convex domain Q implicitly by way of three functions. The barrier function is a barrier for Q. cobarrier is a barrier for the feasibility subproblem, and slack is a function that initializes a valid slack value for the feasibility subproblem. The various convex_ functions can be used to generate various convex domains.\n\nThese function are called as follows: barrier(x,y). x is a vertex in a grid, as per the AMG object. y is some vector. For each fixed x variable, y -> barrier(x,y) defines a barrier for a convex set in y.\n\n\n\n\n\n","category":"type"},{"location":"#Functions-reference","page":"Home","title":"Functions reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [MultiGridBarrier]\nOrder   = [:function]","category":"page"},{"location":"#MultiGridBarrier.amg-Union{Tuple{}, Tuple{M}, Tuple{T}} where {T, M}","page":"Home","title":"MultiGridBarrier.amg","text":"function amg(;\n    x::Matrix{T},\n    w::Vector{T},\n    state_variables::Matrix{Symbol},\n    D::Matrix{Symbol},\n    subspaces::Dict{Symbol,Vector{M}},\n    operators::Dict{Symbol,M},\n    refine::Vector{M},\n    coarsen::Vector{M},\n    full_space=:full,\n    id_operator=:id,\n    feasibility_slack=:feasibility_slack,\n    generate_feasibility=true) where {T,M}\n\nConstruct an AMG object for use with the amgb solver. In many cases, this constructor is not called directly by the user. For 1d and 2d finite elements, use the fem1d() or fem2d(). For 1d and 2d spectral elements, use  spectral1d() or spectral2d(). You use amg() directly if you are implementing your own function spaces.\n\nThe AMG object shall represent all L grid levels of the multigrid hierarchy. Parameters are:\n\nx: the vertices of the fine grid.\nw: the quadrature weights for the fine grid.\nstate_variables: a matrix of symbols. The first column indicates the names of the state vectors or functions, and the second column indicates the names of the corresponding subspaces. A typical example is: state_variables = [:u :dirichlet; :s :full]. This would define the solution as being functions named u(x) and s(x). The u function would lie in the space :dirichlet, presumably consisting of functions with homogeneous Dirichlet conditions. The s function would lie in the space :full, presumably being the full function space, without boundary conditions.\nD: a matrix of symbols. The first column indicates the names of various state variables, and the second column indicates the corresponding differentiation operator(s). For example: D = [:u :id ; :u :dx ; :s :id]. This would indicate that the barrier should be called as F(x,y) with y = [u,ux,s], where ux denotes the derivative of u with respect to the space variable x.\nsubspaces: a Dict mapping each subspace symbol to an array of L matrices, e.g. for each l, subspaces[:dirichlet][l] is a matrix whose columns span the homogeneous Dirichlet subspace of grid level l.\noperators: a Dict mapping each differential operator symbol to a matrix, e.g. operators[:id] is an identity matrix, while operators[:dx] is a numerical differentiation matrix, on the fine grid level L.\nrefine: an array of length L of matrices. For each l, refine[l] interpolates from grid level l to grid level l+1. refine[L] should be the identity, and coarsen[l]*refine[l] should be the identity.\ncoarsen: an array of length L of matrices. For each l, coarsen[l] interpolates or projects from grid level l+1 to grid level l. coarsen[L] should be the identity.\ngenerate_feasibility: if true, amg() returns a pair M of AMG objects. M[1] is an AMG object to be used for the main optimization problem, while M[2] is an AMG object for the preliminary feasibility sub problem. In this case, amg() also needs to be provided with the following additional information: feasibility_slack is the name of a special slack variable that must be unique to the feasibility subproblem (default: :feasibility_slack); full_space is the name of the \"full\" vector space (i.e. no boundary conditions, default: :full); and id_operator is the name of the identity operator (default: :id).\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.amgb-Union{Tuple{Mat}, Tuple{T}, Tuple{Tuple{AMG{T, Mat}, AMG{T, Mat}}, Function, Function, Convex}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.amgb","text":"function amgb(M::Tuple{AMG{T,Mat},AMG{T,Mat}},\n          f::Function, g::Function, Q::Convex;\n          tol=sqrt(eps(T)),\n          t=T(0.1),\n          t_feasibility=t,\n          maxit=10000,\n          kappa=T(10.0),\n          verbose=true,\n          return_details=false) where {T,Mat}\n\nA thin wrapper around amgb_core(). Parameters are:\n\nM: obtained from the amg constructor, a pair of AMG structures. M[1] is the main problem while M[2] is the feasibility problem.\nf: the functional to minimize.\ng: the \"boundary conditions\".\nQ: a Convex domain for the convex optimization problem.\n\nThe initial z0 guess, and the cost functional c0, are computed as follows:\n\nm = size(M[1].x[end],1)\nfor k=1:m\n    z0[k,:] .= g(M[1].x[end][k,:])\n    c0[k,:] .= f(M[1].x[end][k,:])\nend\n\nBy default, the return value z is an m×n matrix, where n is the number of state_variables, see either fem1d(), fem2d(), spectral1d() or spectral2d(). If return_details=true then the return value is a named tuple with fields z, SOL_feasibility and SOL_main; the latter two fields are named tuples with detailed information regarding the various solves.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.amgb_core-Union{Tuple{Mat}, Tuple{T}, Tuple{Barrier, AMG{T, Mat}, Vector{T}, Matrix{T}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.amgb_core","text":"function amgb_core(B::Barrier,\n    M::AMG{T,Mat},\n    z::Array{T,1},\n    c::Array{T,2};\n    tol=(eps(T)),\n    t=T(0.1),\n    maxit=10000,\n    kappa=T(10.0),\n    early_stop=z->false,\n    verbose=true) where {T,Mat}\n\nThe \"Algebraic MultiGrid Barrier\" method.\n\nB a Barrier object.\nM an AMG object.\nz a starting point for the minimization, which should be admissible, i.e. B.f0(z)<∞.\nc an objective functional to minimize. Concretely, we minimize the integral of c.*(D*z), as computed by the finest quadrature in M, subject to B.f0(z)<∞. Here, D is the differential operator provided in M.\n\nOptional parameters:\n\nt: the initial value of t\ntol: we stop when 1/t<tol.\nmaxit: the maximum number of t steps.\nkappa: the initial size of the t-step. Stepsize adaptation is used in the AMGB algorithm, where the t-step size may be made smaller or large, but it will never exceed the initial size provided here.\nverbose: set to true to see a progress bar.\nearly_stop: if early_stop(z) is true then the minimization is stopped early. This is used when solving the preliminary feasibility problem.\n\nReturn value is a named tuple SOL with the following fields:\n\nSOL.converged is true if convergence was obtained, else it is false.\nSOL.z the computed solution.\n\nFurther SOL fields contain various statistics about the solve process.\n\nThe following \"example usage\" is an extremely convoluted way of minimizing x in the interval [-1,1]:\n\nusing MultiGridBarrier\nM = amg(x = [-1.0 ; 1.0 ;;],\n        w = [1.0,1.0],\n        state_variables = [:u :space],\n        D = [:u :id],\n        subspaces = Dict(:space => [[1.0 ; -1.0 ;;]]),\n        operators = Dict(:id => [1.0 0.0;0.0 1.0]),\n        refine = [[1.0 0.0 ; 0.0 1.0]],\n        coarsen = [[1.0 0.0 ; 0.0 1.0]],\n        generate_feasibility=false)\nB = barrier((x,y)->-log(1-x[1]*y[1]))\namgb_core(B,M,[0.0,0.0],[1.0 ; 0.0 ;;]).z[1]\n\n# output\n\n-0.9999999999999998\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.barrier-Tuple{Any}","page":"Home","title":"MultiGridBarrier.barrier","text":"function barrier(F;\n    F1=(x,y)->ForwardDiff.gradient(z->F(x,z),y),\n    F2=(x,y)->ForwardDiff.hessian(z->F(x,z),y))::Barrier\n\nConstructor for barriers.\n\nF is the actual barrier function. It should take parameters (x,y).\nF1 is the gradient of F with respect to y.\nF2 is the Hessian of F with  respect to y.\n\nBy default, F1 and F2 are automatically generated by the module ForwardDiff.\n\nA more specific description of the Barrier object is as follows. The function Barrier.f0 has parameters:\n\nfunction Barrier.f0(z,x,w,c,R,D,z0)\n\nHere, R is a matrix and D is an array of matrices; x is a matrix of quadrature nodes with weights w, and c is a matrix describing the functional we seek to minimize. The value of Barrier.f0 is given by:\n\n        p = length(w)\n        n = length(D)\n        Rz = z0+R*z\n        Dz = hcat([D[k]*Rz for k=1:n]...)\n        y = [F(x[k,:],Dz[k,:]) for k=1:p]\n        dot(w,y)+sum([dot(w.*c[:,k],Dz[:,k]) for k=1:n])\n\nThus, Barrier.f0 can be regarded as a quadrature approximation of the integral\n\nint_Omega left(sum_k=1^nc_k(x)v_k(x)right) + F(xv_1(x)ldotsv_n(x))  dx text where  v_k = D_k(z_0 + Rz)\n\nFunctions Barrier.f1 and Barrier.f2 are the gradient and Hessian, respectively, of Barrier.f0, with respect to the z parameter. If the underlying matrices are sparse, then sparse arithmetic is used for Barrier.f2.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.convex_Euclidian_power-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.convex_Euclidian_power","text":"function convex_Euclidian_power(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0),p::Function=x->T(2)) where {T}\n\nGenerate a Convex object corresponding to the convex set defined by zend geq z1end-1_2^p where z = A(x)*yidx + b(x).\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.convex_linear-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.convex_linear","text":"function convex_linear(::Type{T}=Float64;idx=Colon(),A::Function=(x)->I,b::Function=(x)->T(0)) where {T}\n\nGenerate a Convex structure corresponding to the convex domain A(x,k)*y[idx] .+ b(x,k) ≤ 0.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem1d-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.fem1d","text":"function fem1d(::Type{T}=Float64; L::Int=4,\n                state_variables = [:u :dirichlet\n                                   :s :full],\n                D = [:u :id\n                     :u :dx\n                     :s :id],\n                generate_feasibility=true) where {T}\n\nConstruct an AMG object for a 1d piecewise linear finite element grid. The interval is [-1,1]. Parameters are:\n\nL: divide the interval into 2^L subintervals (L for Levels).\nstate_variables: the \"state vector\" consists of functions, by default this is u(x) and s(x), on the finite element grid.\nD: the set of differential operator. The barrier function F will eventually be called with the parameters F(x,Dz), where z is the state vector. By default, this results in F(x,u,ux,s), where ux is the derivative of u.\ngenerate_feasibility: if true, returns a pair M of AMG objects. M[1] is the AMG object for the main problem, and M[2] is for the feasibility subproblem.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem1d_interp-Union{Tuple{T}, Tuple{Vector{T}, Vector{T}, T}} where T","page":"Home","title":"MultiGridBarrier.fem1d_interp","text":"function fem1d_interp(x::Vector{T},\n                  y::Vector{T},\n                  t::T) where{T}\n\nInterpolate a 1d piecewise linear function at the given t value. If u(xi) is the piecewise linear function such that u(x[k])=y[k] then this function returns u(t).\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem1d_interp-Union{Tuple{T}, Tuple{Vector{T}, Vector{T}, Vector{T}}} where T","page":"Home","title":"MultiGridBarrier.fem1d_interp","text":"function fem1d_interp(x::Vector{T},\n                  y::Vector{T},\n                  t::Vector{T}) where{T}\n\nReturns [fem1d_interp(x,y,t[k]) for k=1:length(t)].\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem1d_solve-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.fem1d_solve","text":"function fem1d_solve(::Type{T}=Float64;\n    p = T(1.0),\n    g = (x)->T[x[1],2],\n    f = (x)->T[0.5,0.0,1.0],\n    Q = convex_Euclidian_power(idx=[2,3],p=x->p),\n    show=true, tol=sqrt(eps(T)),\n    t=T(0.1), kappa=T(10), maxit=10000, L=2,\n    state_variables = [:u :dirichlet\n                       :s :full],\n    D = [:u :id\n         :u :dx\n         :s :id],\n    verbose=true,\n    M = fem1d(T,L=L,state_variables=state_variables,D=D),\n    return_details=false) where {T}\n\nSolve a 1d variational problem on the interval [-1,1] with piecewise linear elements. L is the number of Levels of grid subdivisions, so that the grid consists of 2^L intervals. The solution is computed via:\n\n    SOL=amgb(M,f, g, Q,\n             tol=tol,t=t,maxit=maxit,kappa=kappa,verbose=verbose)\n\nIf show is true, the solution is also plotted.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem2d-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.fem2d","text":"function fem2d(::Type{T}, L::Int, K::Matrix{T};\n                state_variables = [:u :dirichlet\n                                   :s :full],\n                D = [:u :id\n                     :u :dx\n                     :u :dy\n                     :s :id],\n                generate_feasibility=true) where {T}\n\nConstruct an AMG object for a 2d finite element grid on the domain K with piecewise quadratic elements. Parameters are:\n\nK: a triangular mesh. If there are n triangles, then K should be a 3n by 2 matrix of vertices. The first column of K represents x coordinates, the second column represents y coordinates.\nL: divide the interval into 2^L subintervals (L for Levels).\nstate_variables: the \"state vector\" consists of functions, by default this is u(x) and s(x), on the finite element grid.\nD: the set of differential operator. The barrier function F will eventually be called with the parameters F(x,y,Dz), where z is the state vector. By default, this results in F(x,y,u,ux,uy,s), where (ux,uy) is the gradient of u.\ngenerate_feasibility: if true, returns a pair M of AMG objects. M[1] is the AMG object for the main problem, and M[2] is for the feasibility subproblem.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem2d_plot-Union{Tuple{Mat}, Tuple{T}, Tuple{AMG{T, Mat}, Array{T}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.fem2d_plot","text":"function fem2d_plot(M::AMG{T, Mat}, z::Array{T}) where {T,Mat}\n\nPlot a piecewise quadratic solution z on the given mesh. Note that the solution is drawn as (linear) triangles, even though the underlying solution is piecewise quadratic. To obtain a more accurate depiction, especially when the mesh is coarse, it would be preferable to apply a few levels of additional subdivision, so as to capture the curve of the quadratic basis functions.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.fem2d_solve-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.fem2d_solve","text":"function fem2dsolve(::Type{T}=Float64;          p = T(1.0),         K = T[-1 -1;1 -1;-1 1;1 -1;1 1;-1 1],         g = (x)->T[x[1]^2+x[2]^2,100.0],         f = (x)->T[0.5,0.0,0.0,1.0],         Q = convexEuclidianpower(idx=[2,3,4],p=x->p),         show=true, tol=sqrt(eps(T)),         t=T(0.1), kappa=T(10), maxit=10000, L=2,         statevariables = [:u :dirichlet                            :s :full],         D = [:u :id              :u :dx              :u :dy              :s :id],         verbose=true,         M = fem2d(T,L=L,K=K),         return_details=false) where {T}\n\nSolve a 2d variational problem on the domain K, which defaults to the square [-1,1]x[-1,1], with piecewise quadratic elements. K is a triangulation of the domain. For n triangles, K should be a 3n by 2 matrix of vertices. L the number of Levels of grid subdivisions, so that the grid consists of N = n*4^L quadratic triangular elements. Each elements is quadratic, plus a bump function, so each element consists of 7 vertices, i.e. there are 7*N vertices in total. The solution is computed via:\n\n    SOL=amgb(M,f, g, Q,\n            tol=tol,t=t,maxit=maxit,kappa=kappa,verbose=verbose)\n\nIf show is true then the solution is plotted.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.illinois-Union{Tuple{T}, Tuple{Any, T, T}} where T","page":"Home","title":"MultiGridBarrier.illinois","text":"function illinois(f,a::T,b::T;fa=f(a),fb=f(b),maxit=10000) where {T}\n\nFind a root of f between a and b using the Illinois algorithm. If f(a)*f(b)>=0, returns b.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.newton-Union{Tuple{Mat}, Tuple{T}, Tuple{Type{Mat}, Function, Function, Function, Vector{T}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.newton","text":"function newton(::Type{Mat},\n                   F0::Function,\n                   F1::Function,\n                   F2::Function,\n                   x::Array{T,1};\n                   maxit=10000,\n                   theta=T(0.1),\n                   beta=T(0.1),\n                   tol=eps(T)) where {T,Mat}\n\nDamped Newton iteration for minimizing a function.\n\nF0 the objective function\nF1 and F2 are the gradient and Hessian of F0, respectively.\nx the starting point of the minimization procedure.\n\nThe Hessian F2 return value should be of type Mat.\n\nThe optional parameters are:\n\nmaxit, the iteration aborts with a failure message if convergence is not achieved within maxit iterations.\ntol is used as a stopping criterion. We stop when the decrement in the objective is sufficiently small.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral1d-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.spectral1d","text":"function spectral1d(::Type{T}=Float64; n::Integer=5,\n                state_variables = [:u :dirichlet\n                                   :s :full],\n                D = [:u :id\n                     :u :dx\n                     :s :id],\n                generate_feasibility=true) where {T}\n\nConstruct an AlgebraicMultiGridBarrier.AMG object for a 1d spectral grid of polynomials of degree n-1. See also fem1d for a description of the parameters state_variables and D.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral1d_interp-Union{Tuple{Mat}, Tuple{T}, Tuple{AMG{T, Mat}, Vector{T}, Any}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.spectral1d_interp","text":"function spectral1d_interp(MM::AMG{T,Mat}, y::Array{T,1},x) where {T,Mat}\n\nA function to interpolate a solution y at some point(s) x.\n\nMM the mesh of the solution.\ny the solution.\nx point(s) at which the solution should be evaluated.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral1d_plot-Union{Tuple{Mat}, Tuple{T}, Tuple{AMG{T, Mat}, Any, Any, Vararg{Any}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.spectral1d_plot","text":"function spectral1d_plot(M::AMG{T,Mat},x,y,rest...) where {T,Mat}\n\nPlot a solution using pyplot.\n\nM: a mesh.\nx: x values where the solution should be evaluated and plotted.\ny: the solution, to be interpolated at the given x values via spectral1d_interp.\nrest... parameters are passed directly to pyplot.plot.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral1d_solve-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.spectral1d_solve","text":"function spectral1d_solve(::Type{T}=Float64;\n    p = T(1.0),\n    g = (x)->T[x[1],2],\n    f = (x)->T[0.5,0.0,1.0],\n    Q = convex_Euclidian_power(idx=[2,3],p=x->p),\n    show=true, tol=sqrt(eps(T)),\n    t=T(0.1), kappa=T(10), maxit=10000, n=4,\n    state_variables = [:u :dirichlet\n                       :s :full],\n    D = [:u :id\n         :u :dx\n         :s :id],\n    verbose=true,\n    M = spectral1d(T,n=n),\n    return_details=false) where {T}\n\nSolves a p-Laplace problem in d=1 dimension with the given value of p, by spectral elements (i.e. high degree polynomials). The solution is obtained via:\n\n    SOL=amgb(M,f, g, Q,\n             tol=tol,t=t,maxit=maxit,kappa=kappa,verbose=verbose)\n\nIf show is true, the solution is also plotted.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral2d-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.spectral2d","text":"function spectral2d(::Type{T}=Float64; n=5::Integer,\n                state_variables = [:u :dirichlet\n                                   :s :full],\n                D = [:u :id\n                     :u :dx\n                     :u :dy\n                     :s :id],\n                generate_feasibility=true) where {T}\n\nConstruct an AMG object for a 2d spectral grid of degree n-1. See also fem2d for a description of state_variables and D.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral2d_interp-Union{Tuple{Mat}, Tuple{T}, Tuple{AMG{T, Mat}, Vector{T}, Matrix{T}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.spectral2d_interp","text":"function spectral2d_interp(MM::AMG{T,Mat},z::Array{T,1},x::Array{T,2}) where {T,Mat}\n\nInterpolate a solution z at point(s) x, given the mesh MM. See also spectral1d_interp.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral2d_plot-Union{Tuple{Mat}, Tuple{T}, Tuple{AMG{T, Mat}, Any, Any, Vector{T}}} where {T, Mat}","page":"Home","title":"MultiGridBarrier.spectral2d_plot","text":"function spectral2d_plot(M::Mesh{T},x,y,z::Array{T,1};rest...) where {T}\n\nPlot a 2d solution.\n\nM a 2d mesh.\nx, y should be ranges like -1:0.01:1.\nz the solution to plot.\n\n\n\n\n\n","category":"method"},{"location":"#MultiGridBarrier.spectral2d_solve-Union{Tuple{}, Tuple{Type{T}}, Tuple{T}} where T","page":"Home","title":"MultiGridBarrier.spectral2d_solve","text":"function spectral2d_solve(::Type{T}=Float64;\n    p = T(1.0),\n    g = (x)->T[x[1]^2+x[2]^2,100],\n    f = (x)->T[0.5,0.0,0.0,1.0],\n    Q = convex_Euclidian_power(idx=[2,3,4],p=x->p),\n    show=true, tol=sqrt(eps(T)),\n    t=T(0.1), kappa=T(10), maxit=10000, n=4,\n    state_variables = [:u :dirichlet\n                       :s :full],\n    D = [:u :id\n         :u :dx\n         :u :dy\n         :s :id],\n    verbose=true,\n    M = spectral2d(T,n=n),\n    return_details=false) where {T}\n\nSolves a p-Laplace problem in d=2 dimensions using spectral elements (i.e. high degree polynomials). The domain is [0,1]x[0,1], and the solution is computed via:\n\n    SOL=amgb(;\n              M=M,f=f, g=g, F=F,\n              tol=tol,t=t,maxit=maxit,kappa=kappa,verbose=verbose)\n\nIf show is true, then the solution is also plotted.\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
