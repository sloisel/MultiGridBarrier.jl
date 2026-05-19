# Note: Geometry is imported in Mesh3d.jl module via:
# using ..MultiGridBarrier: Geometry, ...

abstract type AbstractDiscretization end

"""
    FEM3D{T}

Discretization type for 3D hexahedral finite elements.

# Fields
- `k::Int`: Polynomial order of elements.
- `K::Array{T,3}`: Q1 corner mesh tensor of shape `(8, N, 3)`; informational.
"""
struct FEM3D{T} <: AbstractDiscretization
    k::Int
    K::Array{T,3}
end

amg_dim(::FEM3D) = 3

function default_D(::FEM3D)
    return [:u :id; :u :dx; :u :dy; :u :dz; :s :id]
end


"""
    _geometric_fem3d_mg(::Type{T}=Float64; L=2, K=<unit cube>, k=3, K_qk=…, structured=true)

Internal: build the geometric L-level multigrid for Q_k hexahedra. Returns a MultiGrid.
"""
function _geometric_fem3d_mg(::Type{T}=Float64; L::Int=2,
                         K::Matrix{T}=T[-1.0 -1.0 -1.0; 1.0 -1.0 -1.0; -1.0 1.0 -1.0; 1.0 1.0 -1.0; -1.0 -1.0 1.0; 1.0 -1.0 1.0; -1.0 1.0 1.0; 1.0 1.0 1.0],
                         k::Int=3,
                         K_qk::Matrix{T} = promote_to_Qk(K, k),
                         structured::Bool=true, rest...) where T
    K_q1 = K
    x = K_qk

    meshes = Vector{Matrix{T}}(undef, L)
    meshes[1] = x
    weights = Vector{Vector{T}}(undef, L)

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

    for l in 1:L-1
        meshes[l+1] = refine_mesh(meshes[l], k)
        weights[l+1] = compute_weights(meshes[l+1], ref_el)
    end

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
            n_fine = size(meshes[L], 1)
            refine_ops[l] = sparse(I, n_fine, n_fine)
            coarsen_ops[l] = sparse(I, n_fine, n_fine)
        end
    end

    ops::Dict{Symbol, SparseMatrixCSC{T, Int}} = build_operators(meshes[L], ref_el)

    s     = k + 1
    sk3   = s^3
    N_fine = div(size(meshes[L], 1), sk3)
    K_q1_tensor = reshape(K_q1, 8, div(size(K_q1, 1), 8), 3)
    disc        = FEM3D{T}(k, K_q1_tensor)

    if structured
        return _fem3d_structured(disc, meshes, weights, L, k, ref_el)
    end

    x_fine = reshape(meshes[L], sk3, N_fine, 3)
    geom = Geometry{T, Array{T,3}, Vector{T}, SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int}, FEM3D{T}}(
        disc,
        x_fine,
        weights[L],
        Dict{Symbol,SparseMatrixCSC{T,Int}}(
            :full      => subspaces[:full][end],
            :uniform   => subspaces[:uniform][end],
            :dirichlet => subspaces[:dirichlet][end]),
        ops,
    )
    return MultiGrid(geom, subspaces, refine_ops, coarsen_ops)
end

# Direct structured construction for FEM3D — builds block types without sparse intermediates
function _fem3d_structured(disc::FEM3D{T}, meshes, weights, L, k, ref_el) where {T}
    p = (k + 1)^3  # block size (nodes per element)
    K_refine = 8    # children per element in 3D

    D_xi_dense = Matrix(ref_el.D_xi_local)
    D_eta_dense = Matrix(ref_el.D_eta_local)
    D_zeta_dense = Matrix(ref_el.D_zeta_local)

    nodes_coarse = chebyshev_nodes(k)
    shifts = [-0.5, 0.5]
    scales = 0.5
    P_local = zeros(T, K_refine * p, p)
    child_idx = 1
    for sz in shifts, sy in shifts, sx in shifts
        child_nodes_x = nodes_coarse .* scales .+ sx
        child_nodes_y = nodes_coarse .* scales .+ sy
        child_nodes_z = nodes_coarse .* scales .+ sz
        Px = interpolation_matrix_1d(nodes_coarse, child_nodes_x)
        Py = interpolation_matrix_1d(nodes_coarse, child_nodes_y)
        Pz = interpolation_matrix_1d(nodes_coarse, child_nodes_z)
        P_child = kron(Pz, kron(Py, Px))
        row_start = (child_idx - 1) * p + 1
        P_local[row_start:row_start+p-1, :] = P_child
        child_idx += 1
    end
    R_local = pinv(P_local)

    N_blocks = div(size(meshes[L], 1), p)
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
        n_elems_l = div(size(meshes[l], 1), p)
        ref_data = zeros(T, p, p, K_refine * n_elems_l)
        coar_data = zeros(T, p, p, K_refine * n_elems_l)
        for i in 1:n_elems_l
            for s in 1:K_refine
                ref_data[:, :, (i-1)*K_refine + s] = P_local[(s-1)*p+1:s*p, :]
                coar_data[:, :, (i-1)*K_refine + s] = R_local[:, (s-1)*p+1:s*p]
            end
        end
        refine[l] = VBlockDiag(p, p, K_refine, n_elems_l, ref_data)
        coarsen[l] = HBlockDiag(p, p, K_refine, n_elems_l, coar_data)
    end

    refine[L] = id_vbd
    coarsen[L] = id_hbd

    id_block = zeros(T, p, p, N_blocks)
    dx_block = zeros(T, p, p, N_blocks)
    dy_block = zeros(T, p, p, N_blocks)
    dz_block = zeros(T, p, p, N_blocks)

    for e in 1:N_blocks
        start_idx = (e - 1) * p + 1
        x_elem = meshes[L][start_idx:start_idx+p-1, 1]
        y_elem = meshes[L][start_idx:start_idx+p-1, 2]
        z_elem = meshes[L][start_idx:start_idx+p-1, 3]

        x_xi = D_xi_dense * x_elem
        x_eta = D_eta_dense * x_elem
        x_zeta = D_zeta_dense * x_elem
        y_xi = D_xi_dense * y_elem
        y_eta = D_eta_dense * y_elem
        y_zeta = D_zeta_dense * y_elem
        z_xi = D_zeta_dense * z_elem  # NOTE: keep parity with original code (uses D_zeta?). See below.
        z_eta = D_eta_dense * z_elem
        z_zeta = D_zeta_dense * z_elem

        # The original uses D_xi/eta/zeta_dense for z components. Re-implement faithfully.
        z_xi = D_xi_dense * z_elem
        z_eta = D_eta_dense * z_elem
        z_zeta = D_zeta_dense * z_elem

        Dx_e = zeros(T, p, p)
        Dy_e = zeros(T, p, p)
        Dz_e = zeros(T, p, p)

        for i in 1:p
            J = [x_xi[i] x_eta[i] x_zeta[i];
                 y_xi[i] y_eta[i] y_zeta[i];
                 z_xi[i] z_eta[i] z_zeta[i]]
            invJ = inv(J)
            for jj in 1:p
                Dx_e[i, jj] = invJ[1,1]*D_xi_dense[i,jj] + invJ[2,1]*D_eta_dense[i,jj] + invJ[3,1]*D_zeta_dense[i,jj]
                Dy_e[i, jj] = invJ[1,2]*D_xi_dense[i,jj] + invJ[2,2]*D_eta_dense[i,jj] + invJ[3,2]*D_zeta_dense[i,jj]
                Dz_e[i, jj] = invJ[1,3]*D_xi_dense[i,jj] + invJ[2,3]*D_eta_dense[i,jj] + invJ[3,3]*D_zeta_dense[i,jj]
            end
        end

        id_block[:, :, e] .= zero(T)
        for j in 1:p
            id_block[j, j, e] = one(T)
        end
        dx_block[:, :, e] = Dx_e
        dy_block[:, :, e] = Dy_e
        dz_block[:, :, e] = Dz_e
    end

    id_op = BlockDiag(id_block)
    dx_op = BlockDiag(dx_block)
    dy_op = BlockDiag(dy_block)
    dz_op = BlockDiag(dz_block)

    subspaces = Dict(
        :full => Vector{SparseMatrixCSC{T, Int}}(undef, L),
        :uniform => Vector{SparseMatrixCSC{T, Int}}(undef, L),
        :dirichlet => Vector{SparseMatrixCSC{T, Int}}(undef, L)
    )
    for l in 1:L
        subs = build_subspaces(meshes[l], k)
        subspaces[:full][l] = subs[:full][1]
        subspaces[:uniform][l] = subs[:uniform][1]
        subspaces[:dirichlet][l] = subs[:dirichlet][1]
    end

    operators = Dict{Symbol, BlockDiag{T,Array{T,3}}}(
        :id => id_op, :dx => dx_op, :dy => dy_op, :dz => dz_op)
    s_   = k + 1
    sk3_ = s_^3
    x_fine = reshape(meshes[L], sk3_, div(size(meshes[L], 1), sk3_), 3)
    geom = Geometry{T, Array{T,3}, Vector{T}, BlockDiag{T,Array{T,3}},
                    SparseMatrixCSC{T,Int}, FEM3D{T}}(
        disc, x_fine, weights[L],
        Dict{Symbol,SparseMatrixCSC{T,Int}}(
            :full      => subspaces[:full][end],
            :uniform   => subspaces[:uniform][end],
            :dirichlet => subspaces[:dirichlet][end]),
        operators)
    return MultiGrid(geom, subspaces, refine, coarsen)
end

"""
    inverse_map_element(nodes_1d, x_elem, x_target; tol=1e-10, max_iter=20)

Find reference coordinates `xi` such that the mapping of `x_elem` at `xi` equals `x_target`.
Uses Newton-Raphson method. Returns `(xi, success)`.
"""
function inverse_map_element(nodes_1d, x_elem, x_target; tol=1e-10, max_iter=20)
    k = length(nodes_1d) - 1
    n_nodes = (k+1)^3

    xi = zeros(3)

    D_1d = derivative_matrix_1d(nodes_1d)
    I_1d = I(k+1)

    for iter in 1:max_iter
        Lx = lagrange_basis(nodes_1d, xi[1])
        Ly = lagrange_basis(nodes_1d, xi[2])
        Lz = lagrange_basis(nodes_1d, xi[3])

        dLx = lagrange_basis_derivative(nodes_1d, xi[1])
        dLy = lagrange_basis_derivative(nodes_1d, xi[2])
        dLz = lagrange_basis_derivative(nodes_1d, xi[3])

        x_curr = zeros(3)
        J = zeros(3, 3)

        idx = 1
        for iz in 1:k+1
            for iy in 1:k+1
                for ix in 1:k+1
                    N = Lx[ix] * Ly[iy] * Lz[iz]

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

        r = x_target - x_curr

        if norm(r) < tol
            return xi, true
        end

        try
            delta = J \ r
            xi += delta
        catch
            return xi, false
        end

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

Evaluate the finite element field `u` at point `x_eval`. Returns the value and a flag
indicating whether the point was found.
"""
function evaluate_field(g::Geometry{T,X,W,<:Any,<:Any,FEM3D{T}}, u::Vector{T}, x_eval::Vector{T}) where {T,X,W}
    k = g.discretization.k
    n_nodes_per_elem = (k+1)^3
    x_flat = _xflat(g.x)
    n_elems = div(size(x_flat, 1), n_nodes_per_elem)

    nodes_1d = chebyshev_nodes(k)

    for e in 1:n_elems
        start_idx = (e-1) * n_nodes_per_elem + 1
        end_idx = e * n_nodes_per_elem
        elem_nodes = x_flat[start_idx:end_idx, :]

        min_x = minimum(elem_nodes[:, 1])
        max_x = maximum(elem_nodes[:, 1])
        min_y = minimum(elem_nodes[:, 2])
        max_y = maximum(elem_nodes[:, 2])
        min_z = minimum(elem_nodes[:, 3])
        max_z = maximum(elem_nodes[:, 3])

        if x_eval[1] >= min_x - 1e-10 && x_eval[1] <= max_x + 1e-10 &&
           x_eval[2] >= min_y - 1e-10 && x_eval[2] <= max_y + 1e-10 &&
           x_eval[3] >= min_z - 1e-10 && x_eval[3] <= max_z + 1e-10

            xi, success = inverse_map_element(nodes_1d, elem_nodes, x_eval)

            if success && all(abs.(xi) .<= 1.0 + 1e-8)
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
