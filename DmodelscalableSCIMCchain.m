%% Scalable n-satellite 1D distributed SCI with Monte Carlo
%
% Chain topology:
% satellite 1 -- satellite 2 -- ... -- satellite n
%
% Global state:
% x = [p1; v1; p2; v2; ...; pn; vn]
%
% Global measurements:
% z_abs_i = p_i
% z_rel_e = p_{e+1} - p_e, e = 1,...,n-1
%
% Each agent i estimates only:
% x_i = [p_i; v_i]
%
% Each agent uses:
% - its own absolute measurement
% - relative measurements connected to its local neighbors
%
% SCI gains are precomputed once because the split covariance recursion is
% deterministic. Monte Carlo then reuses these gains.

clear; clc; close all;

%% Settings

n  = 7;
dt = 0.1;
N  = 1000;
M  = 5000;

t = (0:N)*dt;

alpha = 1.00;
eps_sci = 1e-6;

sigma_abs = 10;
sigma_rel = 1;

sigma_a_true   = 0.1  * ones(n,1);
sigma_a_filter = 0.1 * ones(n,1);

P0_local = diag([10^2, 1^2]);

make_plots = true;
save_figures = false;
save_data = true;

figDir = fullfile(pwd, 'SCI_chain');

assert(n >= 2, 'A chain topology needs n >= 2.');

%% Single-satellite model

Hpos = [1 0];

A1 = [1 dt;
      0  1];

B1 = [0.5*dt^2;
      dt];

Q_base = [dt^4/4, dt^3/2;
          dt^3/2, dt^2];

idx_p = @(i) 2*i - 1;
idx_v = @(i) 2*i;
idx_x = @(i) (2*i-1):(2*i);

%% Global model

A = kron(eye(n), A1);
B = kron(eye(n), B1);

x0 = zeros(2*n,1);

for i = 1:n
    x0(idx_p(i)) = 50*(i-1);
    x0(idx_v(i)) = 0;
end

u = zeros(n,N);

for i = 1:n
    u(i,:) = 0.25 * i * sawtooth( ...
        2*pi*(3+(i-1))*(1:N)/N - 0.5*pi*(i-1), 0.5);
end

Q_true = zeros(2*n);
Q_filter_cell = cell(n,1);

for i = 1:n
    idx = idx_x(i);

    Q_i_true   = sigma_a_true(i)^2   * Q_base;
    Q_i_filter = sigma_a_filter(i)^2 * Q_base;

    Q_true(idx,idx) = Q_i_true;
    Q_filter_cell{i} = Q_i_filter;
end

%% Global measurement model

Cpos = kron(eye(n), Hpos);

% Chain relative measurement edges:
% edge e = [e, e+1] gives p_{e+1} - p_e for e = 1,...,n-1
edges = [(1:n-1)' (2:n)'];

num_edges = size(edges,1);

D = zeros(num_edges,n);

for e = 1:num_edges
    i = edges(e,1);
    j = edges(e,2);

    D(e,i) = -1;
    D(e,j) =  1;
end

C_abs = Cpos;
C_rel = D*Cpos;

Cglob = [C_abs;
         C_rel];

R_abs = sigma_abs^2 * eye(n);
R_rel = sigma_rel^2 * eye(num_edges);

Rglob = blkdiag(R_abs, R_rel);

num_meas = n + num_edges;

%% Building local measurement structures

local = cell(n,1);

for i = 1:n
    local{i} = build_local_structure(i, edges, Hpos);
end

%% Precomputation of deterministic SCI gain and covariance histories

fprintf('Precomputing deterministic SCI gain schedule...\n');

P_SCI_d = cell(n,1);
P_SCI_i = cell(n,1);

P_SCI_d_pred = cell(n,1);
P_SCI_i_pred = cell(n,1);
P_SCI_tot_pred = cell(n,1);

for i = 1:n
    P_SCI_d{i} = zeros(2);
    P_SCI_i{i} = P0_local;
end

P_hist = zeros(2,2,N+1,n);

P_SCI_d_hist = zeros(2,2,N+1,n);
P_SCI_i_hist = zeros(2,2,N+1,n);

for i = 1:n
    P_hist(:,:,1,i) = clean_cov(P_SCI_d{i} + P_SCI_i{i});
    P_SCI_d_hist(:,:,1,i) = P_SCI_d{i};
    P_SCI_i_hist(:,:,1,i) = P_SCI_i{i};
end

K_hist = cell(n,N);
omega_hist = cell(n,N);

for k = 1:N

    P_SCI_d_old = P_SCI_d;
    P_SCI_i_old = P_SCI_i;

    for i = 1:n

        P_SCI_d_pred{i} = alpha*(A1*P_SCI_d_old{i}*A1');
        P_SCI_d_pred{i} = clean_cov(P_SCI_d_pred{i});

        P_SCI_i_pred{i} = alpha*(A1*P_SCI_i_old{i}*A1') + Q_filter_cell{i};
        P_SCI_i_pred{i} = clean_cov(P_SCI_i_pred{i});

        P_SCI_tot_pred{i} = clean_cov(P_SCI_d_pred{i} + P_SCI_i_pred{i});

    end

    for i = 1:n

        [Kmat_i, omega_i, P_d_post_i, P_i_post_i] = ...
            compute_local_SCI_gain_sequence( ...
                i, local{i}, ...
                P_SCI_d_pred{i}, P_SCI_i_pred{i}, ...
                P_SCI_tot_pred, ...
                sigma_abs, sigma_rel, Hpos, eps_sci);

        K_hist{i,k} = Kmat_i;
        omega_hist{i,k} = omega_i;

        P_SCI_d{i} = clean_cov(P_d_post_i);
        P_SCI_i{i} = clean_cov(P_i_post_i);

        P_hist(:,:,k+1,i) = clean_cov(P_SCI_d{i} + P_SCI_i{i});

        P_SCI_d_hist(:,:,k+1,i) = P_SCI_d{i};
        P_SCI_i_hist(:,:,k+1,i) = P_SCI_i{i};

    end

    if mod(k,10) == 0
        fprintf('SCI step %d / %d\n', k, N);
    end

end

fprintf('SCI gain schedule complete.\n');

%% Monte Carlo simulation loop using precomputed SCI gains

fprintf('Running Monte Carlo simulations...\n');

err_hist = zeros(2*n, N+1, M);
xhat_hist = zeros(2*n, N+1, M);

rough_v_mc = zeros(n,M);

x_true_last = [];
z_last = [];
xhat_last = [];

for m = 1:M

    x_true = zeros(2*n, N+1);
    x_true(:,1) = x0;

    w = mvnrnd(zeros(1,2*n), Q_true, N)';

    for k = 1:N
        x_true(:,k+1) = A*x_true(:,k) + B*u(:,k) + w(:,k);
    end

    meas_noise = mvnrnd(zeros(1,num_meas), Rglob, N+1)';
    z = Cglob*x_true + meas_noise;

    xhat = zeros(2*n, N+1);

    for i = 1:n
        xhat(idx_x(i),1) = x0(idx_x(i)) + mvnrnd(zeros(1,2), P0_local)';
    end

    for k = 1:N

        xpred = A*xhat(:,k) + B*u(:,k);

        xhat_next = zeros(2*n,1);

        for i = 1:n

            idx_i = idx_x(i);

            x_i_filt = xpred(idx_i);

            Kmat_i = K_hist{i,k};

            %% Absolute position measurement

            K_abs = Kmat_i(:,1);

            z_abs_i = z(i,k+1);

            innov_abs = z_abs_i - Hpos*x_i_filt;

            x_i_filt = x_i_filt + K_abs*innov_abs;

            %% Relative measurements

            deg_i = numel(local{i}.neighbors);

            for r = 1:deg_i

                K_rel = Kmat_i(:,1+r);

                neigh = local{i}.neighbors(r);
                edgeId = local{i}.edgeIds(r);

                zrel = z(n + edgeId, k+1);

                edge_from = edges(edgeId,1);
                edge_to   = edges(edgeId,2);

                if i == edge_from && neigh == edge_to

                    % zrel = p_neigh - p_i
                    % therefore p_i = p_neigh - zrel
                    z_self = Hpos*xpred(idx_x(neigh)) - zrel;

                elseif i == edge_to && neigh == edge_from

                    % zrel = p_i - p_neigh
                    % therefore p_i = p_neigh + zrel
                    z_self = Hpos*xpred(idx_x(neigh)) + zrel;

                else

                    error('Inconsistent local edge structure for agent %d, edge %d.', ...
                        i, edgeId);

                end

                innov_rel = z_self - Hpos*x_i_filt;

                x_i_filt = x_i_filt + K_rel*innov_rel;

            end

            xhat_next(idx_i) = x_i_filt;

        end

        xhat(:,k+1) = xhat_next;

    end

    err_hist(:,:,m) = x_true - xhat;
    xhat_hist(:,:,m) = xhat;

    for i = 1:n
        dv = diff(xhat(idx_v(i),:));
        rough_v_mc(i,m) = sqrt(mean(dv.^2));
    end

    if m == M
        x_true_last = x_true;
        z_last = z;
        xhat_last = xhat;
    end

    fprintf('MC run %d / %d complete\n', m, M);

end

fprintf('Monte Carlo complete.\n');

%% Monte Carlo empirical covariance

P_mc_diag = zeros(2*n, N+1);

for k = 1:N+1
    E = squeeze(err_hist(:,k,:));
    P_mc_diag(:,k) = var(E, 0, 2);
end

sigma_p = zeros(n,N+1);
sigma_v = zeros(n,N+1);

sigma_p_mc = zeros(n,N+1);
sigma_v_mc = zeros(n,N+1);

for i = 1:n

    sigma_p(i,:) = sqrt(squeeze(P_hist(1,1,:,i)))';
    sigma_v(i,:) = sqrt(squeeze(P_hist(2,2,:,i)))';

    sigma_p_mc(i,:) = sqrt(P_mc_diag(idx_p(i),:));
    sigma_v_mc(i,:) = sqrt(P_mc_diag(idx_v(i),:));

end

%% Monte Carlo summary metrics

err_all = reshape(err_hist, 2*n, []);

rmse_state = sqrt(mean(err_all.^2, 2));

rmse_p = rmse_state(1:2:end);
rmse_v = rmse_state(2:2:end);

rough_v_mean = mean(rough_v_mc, 2);

within_p = zeros(n,1);
within_v = zeros(n,1);

for i = 1:n

    E_p = squeeze(err_hist(idx_p(i),:,:));
    E_v = squeeze(err_hist(idx_v(i),:,:));

    bound_p = repmat(3*sigma_p(i,:)', 1, M);
    bound_v = repmat(3*sigma_v(i,:)', 1, M);

    mask_p = abs(E_p) <= bound_p;
    mask_v = abs(E_v) <= bound_v;

    within_p(i) = mean(mask_p(:));
    within_v(i) = mean(mask_v(:));

end

fprintf('\nMonte Carlo results over %d runs:\n', M);

fprintf('RMSE position [m]:\n');
disp(rmse_p.');

fprintf('RMSE velocity [m/s]:\n');
disp(rmse_v.');

fprintf('Mean velocity roughness [m/s]:\n');
disp(rough_v_mean.');

fprintf('Position inside SCI 3sigma [%%]:\n');
disp(100*within_p.');

fprintf('Velocity inside SCI 3sigma [%%]:\n');
disp(100*within_v.');

%% Plots

if make_plots

    reset(groot);
    set(groot, 'DefaultFigureWindowStyle', 'docked');

    for i = 1:n

        figure('Name',sprintf('SCI_chain_Satellite_%d_position',i));

        plot(t, x_true_last(idx_p(i),:), 'LineWidth', 1.5); hold on;
        plot(t, z_last(i,:), '.', 'MarkerSize', 5);
        plot(t, xhat_last(idx_p(i),:), 'LineWidth', 1.5);

        xlabel('Time [s]');
        ylabel(sprintf('Position p_%d [m]', i));
        legend('True','Measured abs','SCI estimate','Location','best');
        grid on;
        title(sprintf('Satellite %d position', i));

    end

    for e = 1:num_edges

        i = edges(e,1);
        j = edges(e,2);

        p_rel_true = x_true_last(idx_p(j),:) - x_true_last(idx_p(i),:);
        p_rel_est  = xhat_last(idx_p(j),:) - xhat_last(idx_p(i),:);

        figure('Name',sprintf('SCI_chain_Relative_position_satellites_%d_and_%d',i,j));

        plot(t, p_rel_true, 'LineWidth', 1.5); hold on;
        plot(t, z_last(n+e,:), '.', 'MarkerSize', 5);
        plot(t, p_rel_est, 'LineWidth', 1.5);

        xlabel('Time [s]');
        ylabel(sprintf('Relative position p_%d - p_%d [m]', j, i));
        legend('True','Measured rel','SCI estimate','Location','best');
        grid on;
        title(sprintf('Relative position: satellites %d and %d', i, j));

    end

    figure('Name','SCI_chain_Velocities_of_all_satellites');
    hold on;

    for i = 1:n

        plot(t, x_true_last(idx_v(i),:), ...
            'LineWidth', 1.5, ...
            'DisplayName', sprintf('True v_%d', i));

        plot(t, xhat_last(idx_v(i),:), '--', ...
            'LineWidth', 1.5, ...
            'DisplayName', sprintf('SCI estimate v_%d', i));

    end

    xlabel('Time [s]');
    ylabel('Velocity [m/s]');
    legend('Location','bestoutside');
    grid on;
    title('Velocities of all satellites');

    figure('Name','SCI_chain_Control_inputs');
    hold on;

    for i = 1:n
        plot(t(1:N), u(i,:), ...
            'LineWidth', 1.5, ...
            'DisplayName', sprintf('u_%d', i));
    end

    xlabel('Time [s]');
    ylabel('Acceleration [m/s^2]');
    legend('Location','bestoutside');
    grid on;
    title('Control inputs');

    err_last = x_true_last - xhat_last;

    for i = 1:n

        figure('Name',sprintf('SCI_chain_Position_error_satellite_%d',i));

        plot(t, err_last(idx_p(i),:), 'LineWidth', 1.5);
        hold on;

        plot(t,  3*sigma_p(i,:), '--r', 'LineWidth', 1.5);
        plot(t, -3*sigma_p(i,:), '--r', 'LineWidth', 1.5);

        plot(t,  3*sigma_p_mc(i,:), '--g', 'LineWidth', 1.5);
        plot(t, -3*sigma_p_mc(i,:), '--g', 'LineWidth', 1.5);

        xlabel('Time [s]');
        ylabel('Position error [m]');

        legend('Error', ...
               '+3\sigma SCI','-3\sigma SCI', ...
               '+3\sigma empirical MC','-3\sigma empirical MC', ...
               'Location','best');

        grid on;
        title(sprintf('Position error satellite %d', i));

    end

    for i = 1:n

        figure('Name',sprintf('SCI_chain_Velocity_error_satellite_%d',i));

        plot(t, err_last(idx_v(i),:), 'LineWidth', 1.5);
        hold on;

        plot(t,  3*sigma_v(i,:), '--r', 'LineWidth', 1.5);
        plot(t, -3*sigma_v(i,:), '--r', 'LineWidth', 1.5);

        plot(t,  3*sigma_v_mc(i,:), '--g', 'LineWidth', 1.5);
        plot(t, -3*sigma_v_mc(i,:), '--g', 'LineWidth', 1.5);

        xlabel('Time [s]');
        ylabel('Velocity error [m/s]');

        legend('Error', ...
               '+3\sigma SCI','-3\sigma SCI', ...
               '+3\sigma empirical MC','-3\sigma empirical MC', ...
               'Location','best');

        grid on;
        title(sprintf('Velocity error satellite %d', i));

    end

    pos_err_last = zeros(n,N+1);

    for i = 1:n
        pos_err_last(i,:) = err_last(idx_p(i),:);
    end

    e_common = mean(pos_err_last,1);

    figure('Name','SCI_chain_Common_mode_position_error');

    plot(t, e_common, 'LineWidth', 1.5);
    xlabel('Time [s]');
    ylabel('Common position error [m]');
    grid on;
    title('Common-mode position error');

    figure('Name','SCI_chain_Relative_error_components');
    hold on;

    for e = 1:num_edges

        i = edges(e,1);
        j = edges(e,2);

        e_rel = err_last(idx_p(j),:) - err_last(idx_p(i),:);

        plot(t, e_rel, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('e_%d - e_%d', j, i));

    end

    xlabel('Time [s]');
    ylabel('Relative error difference [m]');
    legend('Location','bestoutside');
    grid on;
    title('Relative error components');

    Kabs_pos = zeros(n,N);

    for i = 1:n
        for k = 1:N
            Kabs_pos(i,k) = K_hist{i,k}(1,1);
        end
    end

    figure('Name','SCI_chain_Absolute_measurement_gain');
    hold on;

    for i = 1:n
        plot(t(1:N), Kabs_pos(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Agent %d', i));
    end

    xlabel('Time [s]');
    ylabel('Position gain on absolute measurement');
    legend('Location','bestoutside');
    grid on;
    title('SCI absolute measurement gain');

    figure('Name','SCI_chain_Relative_measurement_gains');
    hold on;

    for i = 1:n

        deg_i = numel(local{i}.neighbors);

        for r = 1:deg_i

            gain_hist = zeros(1,N);

            col = 1 + r;

            for k = 1:N
                gain_hist(k) = K_hist{i,k}(1,col);
            end

            neigh = local{i}.neighbors(r);

            plot(t(1:N), gain_hist, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Agent %d rel with %d', i, neigh));

        end
    end

    xlabel('Time [s]');
    ylabel('Position gain on relative measurements');
    legend('Location','bestoutside');
    grid on;
    title('SCI relative measurement gains');

    for i = 1:n

        figure('Name',sprintf('SCI_chain_Agent_%d_velocity_gain_components',i));
        hold on;

        gain_abs = zeros(1,N);

        for k = 1:N
            gain_abs(k) = K_hist{i,k}(2,1);
        end

        plot(t(1:N), gain_abs, 'LineWidth', 1.5, ...
            'DisplayName','abs');

        deg_i = numel(local{i}.neighbors);

        for r = 1:deg_i

            col = 1 + r;
            gain_rel = zeros(1,N);

            for k = 1:N
                gain_rel(k) = K_hist{i,k}(2,col);
            end

            neigh = local{i}.neighbors(r);

            plot(t(1:N), gain_rel, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('rel with %d', neigh));

        end

        xlabel('Time [s]');
        ylabel(sprintf('Agent %d velocity gain', i));
        legend('Location','best');
        grid on;
        title(sprintf('SCI agent %d velocity gain components', i));

    end

end

%% Saving figures

if make_plots && save_figures

    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end

    sigmaTag = sprintf('sigmaaf_%g', sigma_a_filter(1));
    epsTag   = sprintf('epsSCI_%g', eps_sci);
    nTag     = sprintf('n_%d', n);
    MTag     = sprintf('MC_%d', M);

    paramTag = [nTag, '_', sigmaTag, '_', epsTag, '_', MTag];

    figs = findall(groot, 'Type', 'figure');

    figNums = zeros(numel(figs),1);

    for i = 1:numel(figs)
        figNums(i) = figs(i).Number;
    end

    [~, order] = sort(figNums);
    figs = figs(order);

    for i = 1:numel(figs)

        fig = figs(i);

        set(fig, 'Color', 'w');
        set(fig, 'InvertHardcopy', 'off');

        ax = findall(fig, 'Type', 'axes');

        for a = 1:numel(ax)

            set(ax(a), ...
                'Color', 'w', ...
                'XColor', 'k', ...
                'YColor', 'k', ...
                'FontSize', 12, ...
                'FontName', 'Times New Roman', ...
                'LineWidth', 1.0, ...
                'Box', 'on');

            ax(a).Title.Color = 'k';
            ax(a).Title.FontName = 'Times New Roman';
            ax(a).Title.FontSize = 12;
            ax(a).Title.FontWeight = 'bold';

            ax(a).XLabel.Color = 'k';
            ax(a).YLabel.Color = 'k';
            ax(a).XLabel.FontName = 'Times New Roman';
            ax(a).YLabel.FontName = 'Times New Roman';
            ax(a).XLabel.FontSize = 12;
            ax(a).YLabel.FontSize = 12;

        end

        lgd = findall(fig, 'Type', 'Legend');

        for l = 1:numel(lgd)
            set(lgd(l), ...
                'FontName', 'Times New Roman', ...
                'FontSize', 10, ...
                'TextColor', 'k', ...
                'Color', 'w', ...
                'EdgeColor', [0.3 0.3 0.3]);
        end

        name = fig.Name;

        if isempty(name)

            if ~isempty(ax)

                titleText = ax(1).Title.String;

                if iscell(titleText)
                    titleText = strjoin(titleText, '_');
                elseif isstring(titleText)
                    titleText = char(titleText);
                end

                name = titleText;

            end

        end

        if isempty(name)
            name = sprintf('figure_%02d', fig.Number);
        end

        name = char(name);
        name = regexprep(name, '[^\w\d-]', '_');
        name = regexprep(name, '_+', '_');

        fileBase = sprintf('%02d_%s_%s', fig.Number, name, paramTag);

        pngFile = fullfile(figDir, [fileBase, '.png']);
        figFile = fullfile(figDir, [fileBase, '.fig']);

        try
            exportgraphics(fig, pngFile, 'Resolution', 600);
        catch
            saveas(fig, pngFile);
        end

        savefig(fig, figFile);

    end

    fprintf('Saved %d SCI chain figures to:\n%s\n', numel(figs), figDir);

end

%% Save simulation data for plot-only reruns

if save_data

    dataDir = figDir;

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    sim_info = struct();
    sim_info.estimator = 'SCI';
    sim_info.topology = 'chain';
    sim_info.n = n;
    sim_info.dt = dt;
    sim_info.N = N;
    sim_info.M = M;
    sim_info.sigma_abs = sigma_abs;
    sim_info.sigma_rel = sigma_rel;
    sim_info.sigma_a_true = sigma_a_true;
    sim_info.sigma_a_filter = sigma_a_filter;
    sim_info.eps_sci = eps_sci;
    sim_info.alpha = alpha;

    dataFile = fullfile(dataDir, ...
        sprintf('SCI_chain_sim_data_n_%d_sigmaaf_%g_epsSCI_%g_MC_%d.mat', ...
        n, sigma_a_filter(1), eps_sci, M));

    save(dataFile, ...
        'sim_info', ...
        'n', 'dt', 'N', 'M', 't', ...
        'u', ...
        'edges', 'D', 'local', ...
        'x0', ...
        'x_true_last', 'z_last', 'xhat_last', ...
        'err_hist', 'xhat_hist', ...
        'P_hist', 'P_mc_diag', ...
        'P_SCI_d_hist', 'P_SCI_i_hist', ...
        'K_hist', 'omega_hist', ...
        'sigma_p', 'sigma_v', ...
        'sigma_p_mc', 'sigma_v_mc', ...
        'rmse_p', 'rmse_v', ...
        'rough_v_mean', ...
        'within_p', 'within_v', ...
        'sigma_abs', 'sigma_rel', ...
        'sigma_a_true', 'sigma_a_filter', ...
        'eps_sci', 'alpha', ...
        'C_abs', 'C_rel', 'Cglob', 'R_abs', 'R_rel', 'Rglob', ...
        'Q_true', 'Q_filter_cell', ...
        '-v7.3');

    fprintf('Saved SCI chain simulation data to:\n%s\n', dataFile);

end

%% Auxiliary functions

function local = build_local_structure(i, edges, Hpos)

    num_edges = size(edges,1);

    neighbors = [];
    edgeIds = [];
    selfSigns = [];
    neighborErrorSigns = [];

    for e = 1:num_edges

        edge_from = edges(e,1);
        edge_to   = edges(e,2);

        if i == edge_from

            neighbors(end+1) = edge_to; %#ok<AGROW>
            edgeIds(end+1) = e; %#ok<AGROW>

            % z = p_neighbor - p_i
            selfSigns(end+1) = -1; %#ok<AGROW>
            neighborErrorSigns(end+1) = +1; %#ok<AGROW>

        elseif i == edge_to

            neighbors(end+1) = edge_from; %#ok<AGROW>
            edgeIds(end+1) = e; %#ok<AGROW>

            % z = p_i - p_neighbor
            selfSigns(end+1) = +1; %#ok<AGROW>
            neighborErrorSigns(end+1) = -1; %#ok<AGROW>

        end

    end

    deg_i = numel(neighbors);

    Cself = zeros(1 + deg_i, numel(Hpos));
    Cself(1,:) = Hpos;

    for r = 1:deg_i
        Cself(1+r,:) = selfSigns(r)*Hpos;
    end

    local.Cself = Cself;
    local.neighbors = neighbors;
    local.edgeIds = edgeIds;
    local.selfSigns = selfSigns;
    local.neighborErrorSigns = neighborErrorSigns;

end

function [Kmat, omega_vec, P_d_post, P_i_post] = compute_local_SCI_gain_sequence( ...
    agentId, local_i, P_d_pred_i, P_i_pred_i, Ptot_pred, ...
    sigma_abs, sigma_rel, Hpos, eps_sci)

    %#ok<INUSD>

    deg_i = numel(local_i.neighbors);
    m_i = 1 + deg_i;

    nx = size(P_d_pred_i,1);

    Kmat = zeros(nx,m_i);
    omega_vec = zeros(m_i,1);

    P_d_cur = clean_cov(P_d_pred_i);
    P_i_cur = clean_cov(P_i_pred_i);

    for r = 1:m_i

        Pz1_d = P_d_cur;
        Pz1_i = P_i_cur;

        if r == 1

            Pz2_d = 0;
            Pz2_i = sigma_abs^2;

        else

            neigh = local_i.neighbors(r-1);

            Pz2_d = Hpos*Ptot_pred{neigh}*Hpos';
            Pz2_i = sigma_rel^2;

        end

        [K_sci, w_star, P_d_next, P_i_next] = sci_scalar_measurement_gain( ...
            Pz1_d, Pz1_i, Pz2_d, Pz2_i, Hpos, eps_sci);

        Kmat(:,r) = K_sci;
        omega_vec(r) = w_star;

        P_d_cur = clean_cov(P_d_next);
        P_i_cur = clean_cov(P_i_next);

    end

    P_d_post = clean_cov(P_d_cur);
    P_i_post = clean_cov(P_i_cur);

end

function [K_sci, w_star, P_d_post, P_i_post] = sci_scalar_measurement_gain( ...
    Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, eps_sci)

    nx = size(Pz1_d,1);

    Pz1_d = clean_cov(Pz1_d);
    Pz1_i = clean_cov(Pz1_i);

    Pz2_d = max(Pz2_d, 0);
    Pz2_i = max(Pz2_i, eps_sci);

    trace_bound = @(w) sci_trace_objective( ...
        w, Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, nx, eps_sci);

    opts = optimset('Display','off');

    [w_star,~] = fminbnd(trace_bound, eps_sci, 1-eps_sci, opts);

    [K_sci, P_d_post, P_i_post] = sci_update_for_weight( ...
        w_star, Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, nx, eps_sci);

end

function cost = sci_trace_objective( ...
    w, Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, nx, eps_sci)

    [~, P_d, P_i] = sci_update_for_weight( ...
        w, Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, nx, eps_sci);

    P = clean_cov(P_d + P_i);

    cost = trace(P);

end

function [K_sci, P_d_post, P_i_post] = sci_update_for_weight( ...
    w, Pz1_d, Pz1_i, Pz2_d, Pz2_i, H, nx, eps_sci)

    P1 = clean_cov(Pz1_d/w + Pz1_i);
    P2 = Pz2_d/(1-w) + Pz2_i;

    P2 = max(P2, eps_sci);

    Info = (P1 \ eye(nx)) + (H'/P2)*H;

    P_total_post = clean_cov(Info \ eye(nx));

    K_sci = Info \ (H'/P2);

    middle_i = (P1\Pz1_i)/P1 + H'*(Pz2_i/(P2^2))*H;

    P_i_post = P_total_post * middle_i * P_total_post;
    P_i_post = clean_cov(P_i_post);

    P_d_post = P_total_post - P_i_post;
    P_d_post = clean_cov(P_d_post);

end

function P = clean_cov(P)

    P = 0.5*(P + P');

    [V,D] = eig(P);
    d = diag(D);

    d(d < 0 & d > -1e-10) = 0;

    if any(d < -1e-8)
        warning('clean_cov:negativeEigenvalue', ...
            'Covariance has a significantly negative eigenvalue.');
    end

    d = max(d, 0);

    P = V*diag(d)*V';

    P = 0.5*(P + P');

end