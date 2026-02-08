# Note: Geometry is imported in Mesh3d.jl module via:
# using ..MultiGridBarrier: Geometry, ...
# import ..MultiGridBarrier: default_f, default_g, default_D, ..., amg_dim

abstract type AbstractDiscretization end

"""
    FEM3D{T}

Discretization type for 3D hexahedral finite elements.

# Fields
- `k::Int`: Polynomial order of elements.
- `K::Matrix{T}`: Coarse mesh vertices (N x 3).
- `L::Int`: Number of multigrid levels.
"""
struct FEM3D{T} <: AbstractDiscretization
    k::Int
    K::Matrix{T}
    L::Int
end

amg_dim(::FEM3D) = 3

function default_D(::FEM3D)
    return [:u :id; :u :dx; :u :dy; :u :dz; :s :id]
end

function default_f(::FEM3D)
    return [:u, :u, :u, :u, :s]
end

function default_g(::FEM3D)
    return [:u, :u, :u, :u, :s]
end

# Geometry struct is now imported from MultiGridBarrier


function Geometry(discretization::D, x::X, w::W, subspaces::Dict{Symbol, Vector{M}}, operators::Dict{Symbol, M}, refine::Vector{M}, coarsen::Vector{M}) where {D<:FEM3D, X, W, M}
    T = eltype(x)
    return Geometry{T, X, W, M, D}(discretization, x, w, subspaces, operators, refine, coarsen)
end



"""
    fem3d(::Type{T}=Float64; L::Int=2, K=nothing, k::Int=3, rest...)

Create a `Geometry` object for Q_k hexahedral elements with `L` multigrid levels.

# Arguments
- `T`: Floating-point type (default `Float64`).
- `L`: Number of multigrid levels.
- `K`: Coarse Q1 mesh as an N x 3 matrix, where N is a multiple of 8 (8 vertices per hexahedron).
       If `nothing`, defaults to a single cube [-1,1]^3.
- `k`: Polynomial order of elements (default 3).
"""
function fem3d(::Type{T}=Float64; L::Int=2, K=T[-1.0 -1.0 -1.0; 1.0 -1.0 -1.0; -1.0 1.0 -1.0; 1.0 1.0 -1.0; -1.0 -1.0 1.0; 1.0 -1.0 1.0; -1.0 1.0 1.0; 1.0 1.0 1.0], k::Int=3, rest...) where T
    # Coarse grid (Level 1)
    K_q1 = K

    # Promote Q1 mesh to Qk mesh for Level 1
    x = promote_to_Qk(K_q1, k)

    # Initial weights (reference weights for now, will be updated/overwritten if L>1 loop runs?)
    # Actually, for Level 1, we also need physical weights.
    # If K is the standard cube, weights are just reference weights.
    # If K is distorted, we need to compute weights.

    # Let's reuse the logic.
    # We can treat Level 1 as "refined from nothing" or just compute it.

    meshes = Vector{Matrix{T}}(undef, L)
    meshes[1] = x
    weights = Vector{Vector{T}}(undef, L)

    # Compute weights for Level 1
    # We need to do this for Level 1 specifically because the loop starts at l=1 (refining 1 to 2).

    # Helper to compute weights for a mesh
    function compute_weights(mesh_x, ref_el::ReferenceElement)
        n_nodes_per_elem = (ref_el.k+1)^3
        n_elems = div(size(mesh_x, 1), n_nodes_per_elem)

        w_physical = zeros(T, size(mesh_x, 1))

        for e in 1:n_elems
            start_idx = (e-1) * n_nodes_per_elem + 1
            end_idx = e * n_nodes_per_elem

            x_elem = mesh_x[start_idx:end_idx, 1]
            y_elem = mesh_x[start_idx:end_idx, 2]
            z_elem = mesh_x[start_idx:end_idx, 3]

            x_xi = ref_el.D_xi_local * x_elem
            x_eta = ref_el.D_eta_local * x_elem
            x_zeta = ref_el.D_zeta_local * x_elem

            y_xi = ref_el.D_xi_local * y_elem
            y_eta = ref_el.D_eta_local * y_elem
            y_zeta = ref_el.D_zeta_local * y_elem

            z_xi = ref_el.D_xi_local * z_elem
            z_eta = ref_el.D_eta_local * z_elem
            z_zeta = ref_el.D_zeta_local * z_elem

            for i in 1:n_nodes_per_elem
                J = [x_xi[i] x_eta[i] x_zeta[i];
                     y_xi[i] y_eta[i] y_zeta[i];
                     z_xi[i] z_eta[i] z_zeta[i]]

                detJ = abs(det(J))
                w_physical[start_idx + i - 1] = ref_el.weights_ref[i] * detJ
            end
        end
        return w_physical
    end

    ref_el = ReferenceElement(k, T)
    weights[1] = compute_weights(meshes[1], ref_el)

    # Refine to create hierarchy
    for l in 1:L-1
        meshes[l+1] = refine_mesh(meshes[l], k)
        weights[l+1] = compute_weights(meshes[l+1], ref_el)
    end

    # Build subspaces and operators
    subspaces = Dict(
        :full => Vector{SparseMatrixCSC{T, Int}}(undef, L),
        :uniform => Vector{SparseMatrixCSC{T, Int}}(undef, L),
        :dirichlet => Vector{SparseMatrixCSC{T, Int}}(undef, L)
    )

    refine_ops = Vector{SparseMatrixCSC{T, Int}}(undef, L)
    coarsen_ops = Vector{SparseMatrixCSC{T, Int}}(undef, L)

    for l in 1:L
        subs = build_subspaces(meshes[l], k)
        subspaces[:full][l] = subs[:full][1]
        subspaces[:uniform][l] = subs[:uniform][1]
        subspaces[:dirichlet][l] = subs[:dirichlet][1]

        if l < L
            P, R = build_transfer_operators(meshes[l], k)
            refine_ops[l] = P
            coarsen_ops[l] = R
        else
            # Finest level: Identity
            n_fine = size(meshes[L], 1)
            refine_ops[l] = sparse(I, n_fine, n_fine)
            coarsen_ops[l] = sparse(I, n_fine, n_fine)
        end
    end


    # Fine grid operators

    ops::Dict{Symbol, SparseMatrixCSC{T, Int}} = build_operators(meshes[L], ref_el)

    # Discretization object
    disc = FEM3D{T}(k, K_q1, L)


    return Geometry(
        disc,
        meshes[L], # Fine grid vertices
        weights[L],
        subspaces,
        ops,
        refine_ops,
        coarsen_ops
    )
end

function create_geometry(k::Int, L::Int)
    Base.depwarn("create_geometry is deprecated, use fem3d instead", :create_geometry)
    return fem3d(Float64; L=L, k=k)
end

"""
    inverse_map_element(nodes_1d, x_elem, x_target; tol=1e-10, max_iter=20)

Find reference coordinates `xi` such that the mapping of `x_elem` at `xi` equals `x_target`.
Uses Newton-Raphson method.
Returns `(xi, success)`.
"""
function inverse_map_element(nodes_1d, x_elem, x_target; tol=1e-10, max_iter=20)
    k = length(nodes_1d) - 1
    n_nodes = (k+1)^3

    # Initial guess: center of element
    xi = zeros(3)

    # Pre-compute 1D derivative matrix for Jacobian calculation
    D_1d = derivative_matrix_1d(nodes_1d)
    I_1d = I(k+1)

    # We need to compute x(xi) and J(xi) at each step.
    # x(xi) = sum N_i(xi) * x_i
    # J(xi) = [dx/dxi dx/deta dx/dzeta; ...]

    # Since we are evaluating at a specific point xi, we don't need the full Kronecker matrices.
    # We can evaluate the basis functions and their derivatives at xi.

    for iter in 1:max_iter
        # Evaluate basis and derivatives at current xi
        Lx = lagrange_basis(nodes_1d, xi[1])
        Ly = lagrange_basis(nodes_1d, xi[2])
        Lz = lagrange_basis(nodes_1d, xi[3])

        # We also need derivatives of Lagrange basis at xi
        # We can compute them using the D_1d matrix logic or a helper
        # D_1d is for nodes, here we are at arbitrary xi.
        # Let's implement a helper for derivative of lagrange basis at point.

        dLx = lagrange_basis_derivative(nodes_1d, xi[1])
        dLy = lagrange_basis_derivative(nodes_1d, xi[2])
        dLz = lagrange_basis_derivative(nodes_1d, xi[3])

        # Compute x_curr and Jacobian J
        x_curr = zeros(3)
        J = zeros(3, 3) # [dx/dxi dx/deta dx/dzeta; ...]

        idx = 1
        for iz in 1:k+1
            for iy in 1:k+1
                for ix in 1:k+1
                    # Basis value
                    N = Lx[ix] * Ly[iy] * Lz[iz]

                    # Derivatives of Basis
                    dN_dxi   = dLx[ix] * Ly[iy] * Lz[iz]
                    dN_deta  = Lx[ix] * dLy[iy] * Lz[iz]
                    dN_dzeta = Lx[ix] * Ly[iy] * dLz[iz]

                    node_pos = x_elem[idx, :]

                    x_curr += N * node_pos

                    J[:, 1] += dN_dxi * node_pos
                    J[:, 2] += dN_deta * node_pos
                    J[:, 3] += dN_dzeta * node_pos

                    idx += 1
                end
            end
        end

        # Residual
        r = x_target - x_curr

        if norm(r) < tol
            return xi, true
        end

        # Newton step: J * delta = r
        try
            delta = J \ r
            xi += delta
        catch
            return xi, false # Singular Jacobian
        end

        # Check if we are way outside reference element
        # (Optional: clamp or fail early)
        if any(abs.(xi) .> 2.0)
            return xi, false
        end
    end

    return xi, false
end

function lagrange_basis_derivative(nodes, x_val)
    k = length(nodes) - 1
    vals = zeros(eltype(nodes), k+1)

    for i in 1:k+1
        # L_i'(x) = sum_{j!=i} (1/(x-x_j)) * L_i(x)
        # But L_i(x) might be zero if x is a node.
        # Robust way:
        # L_i(x) = l(x) / (l'(x_i) * (x - x_i)) where l(x) = prod(x - x_j)

        # Let's use the product rule directly
        # L_i(x) = num_i(x) / den_i
        # num_i(x) = prod_{m!=i} (x - x_m)
        # num_i'(x) = sum_{j!=i} prod_{m!=i,j} (x - x_m)

        num_prime = 0.0
        for j in 1:k+1
            if j != i
                prod_term = 1.0
                for m in 1:k+1
                    if m != i && m != j
                        prod_term *= (x_val - nodes[m])
                    end
                end
                num_prime += prod_term
            end
        end

        den = 1.0
        for m in 1:k+1
            if m != i
                den *= (nodes[i] - nodes[m])
            end
        end

        vals[i] = num_prime / den
    end
    return vals
end

"""
    evaluate_field(g::Geometry, u::Vector{T}, x_eval::Vector{T})

Evaluate the finite element field `u` at point `x_eval`.
Returns the value and a flag indicating if the point was found.
"""
function evaluate_field(g::Geometry{T,X,W,<:Any,<:Any,<:Any,<:Any,FEM3D{T}}, u::Vector{T}, x_eval::Vector{T}) where {T,X,W}
    k = g.discretization.k
    n_nodes_per_elem = (k+1)^3
    n_elems = div(size(g.x, 1), n_nodes_per_elem)

    nodes_1d = chebyshev_nodes(k)

    for e in 1:n_elems
        start_idx = (e-1) * n_nodes_per_elem + 1
        end_idx = e * n_nodes_per_elem
        elem_nodes = g.x[start_idx:end_idx, :]

        # Check AABB first for efficiency
        min_x = minimum(elem_nodes[:, 1])
        max_x = maximum(elem_nodes[:, 1])
        min_y = minimum(elem_nodes[:, 2])
        max_y = maximum(elem_nodes[:, 2])
        min_z = minimum(elem_nodes[:, 3])
        max_z = maximum(elem_nodes[:, 3])

        # Add small tolerance for AABB check
        if x_eval[1] >= min_x - 1e-10 && x_eval[1] <= max_x + 1e-10 &&
           x_eval[2] >= min_y - 1e-10 && x_eval[2] <= max_y + 1e-10 &&
           x_eval[3] >= min_z - 1e-10 && x_eval[3] <= max_z + 1e-10

            # Candidate element. Try inverse mapping.
            xi, success = inverse_map_element(nodes_1d, elem_nodes, x_eval)

            if success && all(abs.(xi) .<= 1.0 + 1e-8)
                # Point is inside (or very close to) this element

                # Interpolate u at xi
                Lx = lagrange_basis(nodes_1d, xi[1])
                Ly = lagrange_basis(nodes_1d, xi[2])
                Lz = lagrange_basis(nodes_1d, xi[3])

                u_elem = u[start_idx:end_idx]
                val = 0.0
                idx = 1
                for iz in 1:k+1
                    for iy in 1:k+1
                        for ix in 1:k+1
                            w = Lx[ix] * Ly[iy] * Lz[iz]
                            val += w * u_elem[idx]
                            idx += 1
                        end
                    end
                end

                return val, true
            end
        end
    end

    return 0.0, false
end

