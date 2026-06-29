function [C_all, R_i] = build_local_global_measurement_matrix_loop( ...
    i, n, local, sigma_abs, sigma_rel, Hpos)

    idx_x = @(q) (2*q-1):(2*q);

    deg_i = numel(local.neighbors);
    m_i = 1 + deg_i;

    C_all = zeros(m_i, 2*n);

    % Absolute measurement
    C_all(1, idx_x(i)) = Hpos;

    for r = 1:deg_i

        row = 1 + r;

        neigh = local.neighbors(r);
        selfSign = local.selfSigns(r);

        % neighborErrorSign = - original neighbor coefficient
        neighCoeffSign = -local.neighborErrorSigns(r);

        C_all(row, idx_x(i)) = selfSign * Hpos;
        C_all(row, idx_x(neigh)) = neighCoeffSign * Hpos;

    end

    R_i = diag([sigma_abs^2, sigma_rel^2*ones(1,deg_i)]);

end