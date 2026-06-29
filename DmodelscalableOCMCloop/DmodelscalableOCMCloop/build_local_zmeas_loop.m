function zmeas = build_local_zmeas_loop(i, n, zcol, xpred, local, Hpos)

    idx_x = @(q) (2*q-1):(2*q);

    deg_i = numel(local.neighbors);

    zmeas = zeros(1+deg_i,1);

    % Absolute measurement
    zmeas(1) = zcol(i);

    for r = 1:deg_i

        neigh = local.neighbors(r);
        edgeId = local.edgeIds(r);
        neighborErrorSign = local.neighborErrorSigns(r);

        % Relative measurement index in global z:
        % absolute measurements occupy 1:n
        % relative edge e occupies n+e
        zrel = zcol(n + edgeId);

        zmeas(1+r) = zrel + neighborErrorSign * Hpos*xpred(idx_x(neigh));

    end

end
