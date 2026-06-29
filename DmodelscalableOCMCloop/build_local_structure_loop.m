function local = build_local_structure_loop(i, n, Hpos)

    % Loop/ring topology:
    %
    % edges are:
    % 1: (1,2)
    % 2: (2,3)
    % ...
    % n-1: (n-1,n)
    % n: (n,1)
    %
    % Each agent has:
    % - left neighbor  i-1, with wrap-around
    % - right neighbor i+1, with wrap-around
    %
    % Measurement order:
    % 1) absolute measurement of satellite i
    % 2) left relative measurement:  p_i - p_left
    % 3) right relative measurement: p_right - p_i

    left = i - 1;
    if left < 1
        left = n;
    end

    right = i + 1;
    if right > n
        right = 1;
    end

    % Edge IDs follow the global edge list:
    % edge e = (e,e+1), for e = 1,...,n-1
    % edge n = (n,1)
    leftEdge  = left;
    rightEdge = i;

    neighbors = [left, right];
    edgeIds   = [leftEdge, rightEdge];

    % Left edge:
    % z_left = p_i - p_left
    % local self coefficient = +Hpos
    % residualization uses +Hpos*xpred_left
    %
    % Right edge:
    % z_right = p_right - p_i
    % local self coefficient = -Hpos
    % residualization uses -Hpos*xpred_right
    selfSigns = [+1, -1];
    neighborErrorSigns = [+1, -1];

    Cself = [Hpos;
             +Hpos;
             -Hpos];

    local.Cself = Cself;
    local.neighbors = neighbors;
    local.edgeIds = edgeIds;
    local.selfSigns = selfSigns;
    local.neighborErrorSigns = neighborErrorSigns;

end