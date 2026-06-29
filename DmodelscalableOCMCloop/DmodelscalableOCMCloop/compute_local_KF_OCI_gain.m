function [K_i, Ppost_i] = compute_local_KF_OCI_gain( ...
    i, local, Ppred, sigma_abs, sigma_rel, eps_prior, gamma_i, Hpos)

    deg_i = numel(local.neighbors);
    m_i = 1 + deg_i;

    Cself_i = local.Cself;

    Hoci = [eye(2);
            Cself_i];

    Rmeas = diag([sigma_abs^2, sigma_rel^2*ones(1,deg_i)]);

    Roci = blkdiag(eps_prior*eye(2), Rmeas);

    % Error stack:
    % eta_i = [delta_i; delta_neighbor1; delta_neighbor2; ...]
    stackAgents = [i, local.neighbors];

    q = numel(stackAgents);
    eta_dim = 2*q;

    Coci = zeros(2+m_i, eta_dim);

    % First block: prior prediction error of the agent itself
    Coci(1:2, 1:2) = eye(2);

    % Absolute measurement row has no prediction-error contribution.
    % Relative measurement rows contain neighbor prediction errors.
    for r = 1:deg_i

        row = 2 + 1 + r;        % first two are prior block, row 3 is abs
        block = (2*(r+1)-1):(2*(r+1));

        Coci(row, block) = local.neighborErrorSigns(r)*Hpos;

    end

    Yb = cell(q,1);

    for a = 1:q

        Y = zeros(eta_dim, eta_dim);

        agent = stackAgents(a);
        block = (2*a-1):(2*a);

        Y(block, block) = info_mat(Ppred{agent});

        Yb{a,1} = Y;

    end

    D = eye(eta_dim);
    E = zeros(eta_dim);

    [Xii, ~, K_i, ~, ~] = KF_OCI( ...
        Hoci, Roci, Coci, Yb, D, E, gamma_i);

    Ppost_i = clean_cov(Xii);

end