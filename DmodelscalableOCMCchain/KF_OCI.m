% Description: Implementation of Kahan-family-optimal solution to the CL 
% problem described in [1].
% Inputs: 
%   - H: matrix H
%   - R: matrix R
%   - C: matrix C
%   - Yb: Mx1 cell of matrices Y^i_\Ical
%   - D: matrix D
%   - E: matrix E
%   - gamma: regularization weight gamma
% Outputs: Matrices Xii, X, K, Y, U
% Assumptions and limitations: none
% Other m-files required: none
% MAT-files required: none
% Toolboxes required: YALMIP [2], MOSEK [3]
% Authors: Leonardo Pedroso, W.P.M.H. (Maurice) Heemels, Pedro Batista
% Revision history:
%   - 20/01/2026 - First draft of full implementation (Leonardo Pedroso)
% References:
% [1] L. Pedroso, W.P.M.H. Heemels, and P. Batista, "Consistent Distributed
%     Cooperative Localization for Ultra Large-Scale Multi-agent Systems",
%     2026. (submitted)
% [2] J. Lofberg, "YALMIP: A toolbox for modeling and optimization in 
%     MATLAB." In 2004 IEEE international conference on robotics and
%     automation, pp. 284-289, 2004.
% [3] MOSEK, MOSEK Modeling Cookbook, Release 3.3.0. 2024.

function [Xii,X,K,Y,U] = KF_OCI(H,R,C,Yb,D,E,gamma)
    % Sizes
    n = size(H,2);
    m = size(Yb{1,1},1);
    M = size(Yb,1);
    o = size(E,1);
    % Build SDP
    opts = sdpsettings('solver','mosek','verbose',0,'cachesolvers',1,...
        'mosek.MSK_IPAR_NUM_THREADS', 1);
    omega = sdpvar(M,1);
    Y = zeros(m,m);
    for i = 1:M
        Y = Y + omega(i,1)*Yb{i,1};
    end
    U = sdpvar(n,n);
    Xii = sdpvar(n,n);
    X = sdpvar(o,o);
    cntr = [U>=0, Xii>=0];
    cntr = [cntr, omega>=0, omega<=1, ones(1,M)*omega == 1];
    cntr = [cntr, [U (H'/R)*C; ((H'/R)*C)' Y+(C'/R)*C]>=0];
    cntr = [cntr, [Xii eye(n); eye(n) (H'/R)*H-U]>=0];
    cntr = [cntr, [X D; D' Y]>=0];
    % Solve SDP
    sol = optimize(cntr,trace(Xii)+(n/o)*gamma*trace(X+E),opts);
    if sol.problem ~= 0
        sol.info
        yalmiperror(sol.problem)
        error("OCI's numerical optimization failed.")
    end
    % Get solution
    Y = value(Y);
    U = value(U);
    Xii = value(Xii);
    X = value(X)+E;
    % Compute K
    aux = (R\(R-C*pinv(Y+(C'/R)*C)*C'))/R;
    K = (H'*aux*H)\(H'*aux);   
    K = K(:,n+1:end);
end