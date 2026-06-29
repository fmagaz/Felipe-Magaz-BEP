function local = build_local_structure_chain(i, n, Hpos)

    neighbors = [];
    edgeIds = [];
    selfSigns = [];
    neighborErrorSigns = [];
    Cself = Hpos;   % absolute measurement

    % Left neighbor: edge (i-1, i)
    % z = p_i - p_{i-1}
    if i > 1
        left = i - 1;
        neighbors = [neighbors, left];
        edgeIds = [edgeIds, i-1];

        selfSigns = [selfSigns, +1];
        neighborErrorSigns = [neighborErrorSigns, +1];

        Cself = [Cself;
                 +Hpos];
    end

    % Right neighbor: edge (i, i+1)
    % z = p_{i+1} - p_i
    if i < n
        right = i + 1;
        neighbors = [neighbors, right];
        edgeIds = [edgeIds, i];

        selfSigns = [selfSigns, -1];
        neighborErrorSigns = [neighborErrorSigns, -1];

        Cself = [Cself;
                 -Hpos];
    end

    local.Cself = Cself;
    local.neighbors = neighbors;
    local.edgeIds = edgeIds;
    local.selfSigns = selfSigns;
    local.neighborErrorSigns = neighborErrorSigns;

end