
function derivative_matrix_1d(nodes)
    k = length(nodes) - 1
    D = zeros(k+1, k+1)
    
    # For each node i, compute derivative of Lagrange basis L_j at node i
    for i in 1:k+1
        xi = nodes[i]
        for j in 1:k+1
            # D[i, j] = L_j'(xi)
            if i == j
                # Sum_{m!=i} 1/(xi - xm)
                val = 0.0
                for m in 1:k+1
                    if m != i
                        val += 1.0 / (xi - nodes[m])
                    end
                end
                D[i, j] = val
            else
                # (Product_{m!=i,j} (xi - xm)) / (Product_{m!=j} (xj - xm))
                # = (Product_{m!=i,j} (xi - xm)) / ((xj - xi) * Product_{m!=i,j} (xj - xm))
                # = (1/(xj - xi)) * (Product_{m!=i,j} (xi - xm) / (xj - xm))
                
                # Easier: L_j(x) = w_j / (x - x_j) / sum(w_k / (x - x_k))? No, that's barycentric.
                # Standard formula:
                # L_j'(x_i) = (w_j / w_i) * (1 / (x_i - x_j))
                # where w_j = 1 / Product_{k!=j} (x_j - x_k)
                
                # Let's compute weights first
                wi = 1.0
                wj = 1.0
                for m in 1:k+1
                    if m != i; wi *= (nodes[i] - nodes[m]); end
                    if m != j; wj *= (nodes[j] - nodes[m]); end
                end
                # wi is denominator of L_i
                # wj is denominator of L_j
                
                # Actually, L_j(x) = Product_{m!=j} (x - xm) / Product_{m!=j} (xj - xm)
                # L_j'(xi) = [Product_{m!=j} (x - xm)]' at xi / Denom
                # The derivative of product at a root xi (since i!=j, xi is a root of numerator)
                # is Product_{m!=j, i} (xi - xm).
                
                num = 1.0
                for m in 1:k+1
                    if m != j && m != i
                        num *= (nodes[i] - nodes[m])
                    end
                end
                
                den = 1.0
                for m in 1:k+1
                    if m != j
                        den *= (nodes[j] - nodes[m])
                    end
                end
                
                D[i, j] = num / den
            end
        end
    end
    return D
end

function interpolation_matrix_1d(coarse_nodes, fine_nodes)
    nc = length(coarse_nodes)
    nf = length(fine_nodes)
    P = zeros(nf, nc)
    
    for i in 1:nf
        vals = lagrange_basis(coarse_nodes, fine_nodes[i])
        P[i, :] = vals
    end
    return P
end

"""
    build_operators(x::Matrix{T}, ref_el::ReferenceElement{T})

Build differential operators for the mesh `x` using the reference element `ref_el`.

Returns a `Dict{Symbol, SparseMatrixCSC}` with keys `:dx`, `:dy`, `:dz`, and `:id`.
"""
function build_operators(x::Matrix{T}, ref_el::ReferenceElement{T}) where T
    n_nodes_per_elem = (ref_el.k+1)^3
    n_elems = div(size(x, 1), n_nodes_per_elem)
    n_total = size(x, 1)
    
    # Use cached operators
    D_xi_local = ref_el.D_xi_local
    D_eta_local = ref_el.D_eta_local
    D_zeta_local = ref_el.D_zeta_local
    
    # Pre-allocate element operators
    Dx_elems = Vector{SparseMatrixCSC{T, Int}}(undef, n_elems)
    Dy_elems = Vector{SparseMatrixCSC{T, Int}}(undef, n_elems)
    Dz_elems = Vector{SparseMatrixCSC{T, Int}}(undef, n_elems)
    
    for e in 1:n_elems
        start_idx = (e-1) * n_nodes_per_elem + 1
        end_idx = e * n_nodes_per_elem
        
        # Element coordinates
        x_elem = x[start_idx:end_idx, 1]
        y_elem = x[start_idx:end_idx, 2]
        z_elem = x[start_idx:end_idx, 3]
        
        # Compute geometric factors
        # x_xi = D_xi * x, etc.
        x_xi = D_xi_local * x_elem
        x_eta = D_eta_local * x_elem
        x_zeta = D_zeta_local * x_elem
        
        y_xi = D_xi_local * y_elem
        y_eta = D_eta_local * y_elem
        y_zeta = D_zeta_local * y_elem
        
        z_xi = D_xi_local * z_elem
        z_eta = D_eta_local * z_elem
        z_zeta = D_zeta_local * z_elem
        
        # Jacobian determinant and inverse metrics
        # J = [x_xi x_eta x_zeta; y_xi y_eta y_zeta; z_xi z_eta z_zeta]
        # We need xi_x, xi_y, xi_z, etc.
        
        # Allocate metric arrays
        xi_x = zeros(T, n_nodes_per_elem)
        xi_y = zeros(T, n_nodes_per_elem)
        xi_z = zeros(T, n_nodes_per_elem)
        eta_x = zeros(T, n_nodes_per_elem)
        eta_y = zeros(T, n_nodes_per_elem)
        eta_z = zeros(T, n_nodes_per_elem)
        zeta_x = zeros(T, n_nodes_per_elem)
        zeta_y = zeros(T, n_nodes_per_elem)
        zeta_z = zeros(T, n_nodes_per_elem)
        
        for i in 1:n_nodes_per_elem
            J = [x_xi[i] x_eta[i] x_zeta[i];
                 y_xi[i] y_eta[i] y_zeta[i];
                 z_xi[i] z_eta[i] z_zeta[i]]
            
            invJ = inv(J)
            
            # invJ = [xi_x xi_y xi_z;
            #         eta_x eta_y eta_z;
            #         zeta_x zeta_y zeta_z]
            # Wait, J_ji = dx_j / dxi_i
            # invJ_ij = dxi_i / dx_j
            # Row 1: dxi/dx, dxi/dy, dxi/dz
            # Row 2: deta/dx, deta/dy, deta/dz
            # Row 3: dzeta/dx, dzeta/dy, dzeta/dz
            
            xi_x[i] = invJ[1, 1]
            xi_y[i] = invJ[1, 2]
            xi_z[i] = invJ[1, 3]
            
            eta_x[i] = invJ[2, 1]
            eta_y[i] = invJ[2, 2]
            eta_z[i] = invJ[2, 3]
            
            zeta_x[i] = invJ[3, 1]
            zeta_y[i] = invJ[3, 2]
            zeta_z[i] = invJ[3, 3]
        end
        
        # Construct physical derivatives
        # Dx = diag(xi_x)*D_xi + diag(eta_x)*D_eta + diag(zeta_x)*D_zeta
        
        Dx_e = spdiagm(xi_x) * D_xi_local + spdiagm(eta_x) * D_eta_local + spdiagm(zeta_x) * D_zeta_local
        Dy_e = spdiagm(xi_y) * D_xi_local + spdiagm(eta_y) * D_eta_local + spdiagm(zeta_y) * D_zeta_local
        Dz_e = spdiagm(xi_z) * D_xi_local + spdiagm(eta_z) * D_eta_local + spdiagm(zeta_z) * D_zeta_local
        
        Dx_elems[e] = Dx_e
        Dy_elems[e] = Dy_e
        Dz_elems[e] = Dz_e
    end
    
    # Assemble block diagonal
    Dx = blockdiag(Dx_elems...)
    Dy = blockdiag(Dy_elems...)
    Dz = blockdiag(Dz_elems...)
    Id = sparse(I, n_total, n_total)
    
    return Dict(:dx => Dx, :dy => Dy, :dz => Dz, :id => Id)
end

function build_subspaces(x::Matrix{T}, k::Int) where T
    boundary_indices, unique_x, node_map = get_boundary_nodes(x, k)
    n_total = size(x, 1)
    n_unique = size(unique_x, 1)
    
    # :full -> Identity
    full = sparse(I, n_total, n_total)
    
    # :uniform -> Ones
    uniform = sparse(ones(T, n_total, 1))
    
    # :dirichlet
    # Basis for continuous functions with 0 on boundary.
    # Columns correspond to unique nodes NOT on boundary.
    
    interior_unique_indices = setdiff(1:n_unique, boundary_indices)
    sort!(interior_unique_indices)
    
    n_interior = length(interior_unique_indices)
    
    # Map unique index to column index in D
    unique_to_col = Dict{Int, Int}()
    for (j, ui) in enumerate(interior_unique_indices)
        unique_to_col[ui] = j
    end
    
    # Build D matrix
    # Rows: 1..n_total
    # Cols: 1..n_interior
    # D[i, j] = 1 if node_map[i] == interior_unique_indices[j]
    
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    
    for i in 1:n_total
        ui = node_map[i]
        if haskey(unique_to_col, ui)
            push!(I_idx, i)
            push!(J_idx, unique_to_col[ui])
            push!(V_val, 1.0)
        end
    end
    
    dirichlet = sparse(I_idx, J_idx, V_val, n_total, n_interior)
    
    return Dict(:full => [full], :uniform => [uniform], :dirichlet => [dirichlet])
end

function build_transfer_operators(x_coarse::Matrix{T}, k::Int) where T
    # Refine x_coarse to get x_fine structure (we don't need coordinates, just topology)
    # Actually, refinement is local.
    # P_local maps (k+1)^3 coarse nodes to 8*(k+1)^3 fine nodes.
    
    nodes_coarse = chebyshev_nodes(k)
    
    # Child nodes in reference frame
    shifts = [-0.5, 0.5]
    scales = 0.5
    
    # We need to order the 8 children correctly to match refine_mesh
    # refine_mesh order: z, y, x loops for shifts.
    
    # Construct P_local
    # Rows: 8*(k+1)^3
    # Cols: (k+1)^3
    
    P_local = zeros(T, 8*(k+1)^3, (k+1)^3)
    
    child_idx = 1
    for sz in shifts
        for sy in shifts
            for sx in shifts
                # For this child, compute interpolation matrix
                # Child nodes in parent frame
                child_nodes_x = nodes_coarse .* scales .+ sx
                child_nodes_y = nodes_coarse .* scales .+ sy
                child_nodes_z = nodes_coarse .* scales .+ sz
                
                # Tensor product interpolation
                Px = interpolation_matrix_1d(nodes_coarse, child_nodes_x)
                Py = interpolation_matrix_1d(nodes_coarse, child_nodes_y)
                Pz = interpolation_matrix_1d(nodes_coarse, child_nodes_z)
                
                # P_child = Pz (x) Py (x) Px
                P_child = kron(Pz, kron(Py, Px))
                
                # Place in P_local
                row_start = (child_idx-1)*(k+1)^3 + 1
                row_end = child_idx*(k+1)^3
                P_local[row_start:row_end, :] = P_child
                
                child_idx += 1
            end
        end
    end
    
    n_nodes_per_elem = (k+1)^3
    n_elems_coarse = div(size(x_coarse, 1), n_nodes_per_elem)
    
    # Global P is block diagonal
    P = blockdiag([sparse(P_local) for _ in 1:n_elems_coarse]...)
    
    # R is pseudoinverse of P.
    # Since P_local has full column rank (interpolation), R_local = (P'P)^-1 P'
    # Or just P' if orthogonal? Not orthogonal.
    # User said: "g.coarsen[l] should consist of blocks that are a pseudoinverse of g.refine[l]"
    
    R_local = pinv(P_local)
    R = blockdiag([sparse(R_local) for _ in 1:n_elems_coarse]...)
    
    return P, R
end
