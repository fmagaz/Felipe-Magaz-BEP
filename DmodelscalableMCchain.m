%% Scalable n-satellite 1D Kalman filter simulation

clear; clc; close all;

%% Settings
n  = 7;            % number of satellites
dt = 0.1;          % discrete time 
N  = 1000;
t  = (0:N)*dt;

%% Single-satellite model
A1 = [1 dt;
      0  1];

B1 = [0.5*dt^2;
      dt];

Q1 = [dt^4/4, dt^3/2;
      dt^3/2, dt^2];

%% Build system matrices
A = kron(eye(n), A1);      % 2n x 2n
B = kron(eye(n), B1);      % 2n x n

%% Initial state
x0 = zeros(2*n,1);
for i = 1:n
    x0(2*i-1) = 50*(i-1);   % spread positions
    x0(2*i)   = 0;      % initial velocity
end

%% Inputs (each satellite different)
u = zeros(n,N);
for i = 1:n
    u(i,:) = 0.25 * i * sawtooth(2*pi*(3+(i-1))*(1:N)/N - 0.5*pi*(i-1), 0.5);
end


%% Process noise
sigma_a = 0.1 * ones(n,1);

Q = zeros(2*n);
for i = 1:n
    idx = 2*i-1:2*i;
    Q(idx,idx) = sigma_a(i)^2 * Q1;
end

%% Monte carlo simulation loop
M = 5000;
err_hist = zeros(2*n, N+1, M);

for m = 1:M   
    %% True simulation
    x_true = zeros(2*n, N+1);
    x_true(:,1) = x0;

    w = mvnrnd(zeros(2*n, 1), Q, N)';
    for k = 1:N
        x_true(:,k+1) = A*x_true(:,k) + B*u(:,k) + w(:,k);
    end
    
    %% Measurement model
    
    % Absolute position measurements
    Cpos = kron(eye(n), [1 0]);   % n x 2n
    
    % Define relative measurement edges [i j] => p_j - p_i
    edges = zeros(n-1,2);
    
    for e = 1:(n-1)
        edges(e,1) = e;  % Assigning the first satellite in the edge
        edges(e,2) = e + 1;  % Assigning the next satellite in the edge
    end
    
    D = zeros(n-1,n);
    for e = 1:n-1
        i = edges(e,1);
        j = edges(e,2);
        D(e,i) = -1;
        D(e,j) =  1;
    end
    
    C_rel = D * Cpos;   % n-1 x 2n
    C_abs = Cpos;       % n x 2n
    
    C = [C_abs;
         C_rel];
    
    %% Measurement noise
    sigma_abs = 10;
    sigma_rel = 1;
    
    R_abs = sigma_abs^2 * eye(n);
    R_rel = sigma_rel^2 * eye(n-1);
    
    R = blkdiag(R_abs, R_rel);
    
    %% Measurement generation
    z = zeros(2*n-1, N+1);
    
    for k = 1:N+1
        e = mvnrnd(zeros(n+(n-1),1), R)'; %noise
        z(:,k) = C*x_true(:,k) + e;
    end
    
    %% Kalman filter
    
    x_hat = zeros(2*n, N+1);
    P_hist = zeros(2*n, 2*n, N+1);

    % Initial covariance
    P = blkdiag(kron(eye(n), diag([10^2 1^2]))); % 10 and 1 chosen as initial sensor standard deviation guess for position and velocity

    % Initial estimates
    x_hat(:,1) = x0 + mvnrnd(zeros(2*n, 1), P)';
    P_hist(:, :, 1) = P;

    I = eye(2*n);
    
    for k = 1:N
        % Predict
        xpred = A*x_hat(:,k) + B*u(:,k);
        Ppred = A*P*A' + Q;
    
        % Update
        y = z(:,k+1) - C*xpred;
        S = C*Ppred*C' + R;
        K = Ppred*C'/S;
    
        x_hat(:,k+1) = xpred + K*y;
        P = (I - K*C)*Ppred;
        P_hist(:, :, k+1) = P;
    end

    err_hist(:, :, m) = x_true - x_hat;

end
sigma_p = zeros(n, N+1);
sigma_v = zeros(n, N+1);
sigma_p_mc = zeros(n, N+1);
sigma_v_mc = zeros(n, N+1);
P_mc = zeros(2*n, 2*n, N+1);

for k = 1:N+1

    E = squeeze(err_hist(:,k,:));   % size 2*n x M
    P_mc(:,:,k) = cov(E.');

end

for k = 1:n

    sigma_p(k, :) = sqrt(squeeze(P_hist(2*k-1, 2*k-1, :)));
    sigma_v(k, :) = sqrt(squeeze(P_hist(2*k, 2*k, :)));
    sigma_p_mc(k, :) = sqrt(squeeze(P_mc(2*k-1, 2*k-1, :)));
    sigma_v_mc(k, :) = sqrt(squeeze(P_mc(2*k, 2*k, :)));

end





%% Example: relative position between satellite 1 and n
p_rel_true = x_true(2*n-1,:) - x_true(1,:);
p_rel_est  = x_hat(2*n-1,:) - x_hat(1,:);

%% Save data for separate plotting script

save_dir = fullfile(pwd, 'centralized_kf_saved_data');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Store representative run data
% Note: x_true, x_hat, and z are from the final Monte Carlo run.
x_true_rep = x_true;
x_hat_rep  = x_hat;
z_rep      = z;

% Store index helpers as arrays for easy use in another script
idx_p_all = 1:2:(2*n-1);
idx_v_all = 2:2:(2*n);

% Store metadata
sim_info = struct();
sim_info.estimator = 'Centralized KF';
sim_info.topology  = 'chain';
sim_info.n         = n;
sim_info.dt        = dt;
sim_info.N         = N;
sim_info.M         = M;
sim_info.sigma_abs = sigma_abs;
sim_info.sigma_rel = sigma_rel;
sim_info.sigma_a   = sigma_a;

% Save all data needed for plot-only scripts
save_filename = fullfile(save_dir, ...
    sprintf('centralized_KF_n_%d_MC_%d.mat', n, M));

save(save_filename, ...
    'sim_info', ...
    'n', 'dt', 'N', 'M', 't', ...
    'A1', 'B1', 'Q1', 'A', 'B', ...
    'x0', 'u', ...
    'sigma_a', 'Q', ...
    'sigma_abs', 'sigma_rel', 'R_abs', 'R_rel', 'R', ...
    'edges', 'D', 'Cpos', 'C_abs', 'C_rel', 'C', ...
    'x_true_rep', 'x_hat_rep', 'z_rep', ...
    'P_hist', 'P_mc', 'err_hist', ...
    'sigma_p', 'sigma_v', 'sigma_p_mc', 'sigma_v_mc', ...
    'idx_p_all', 'idx_v_all');

fprintf('Saved centralized KF plotting data to:\n%s\n', save_filename);

%% Plots

% figure;
% plot(t, p_rel_true, 'LineWidth',1.5); hold on;
% plot(t, p_rel_est, '--','LineWidth',1.5);
% xlabel('Time [s]');
% ylabel('Relative position [m]');
% legend('True','Estimated');
% grid on;
% title('Relative position (sat n - sat 1)');
% 
% %% Plot all positions
% figure;
% hold on
% for i = 1:n
%     plot(t, x(2*i-1,:), 'LineWidth',1);
% end
% xlabel('Time [s]');
% ylabel('Position [m]');
% title('True positions of all satellites');
% grid on;


%% Scalable plots

idx_p = @(i) 2*i - 1;
idx_v = @(i) 2*i;

%% 1) Absolute position figure for each satellite
for i = 1:n
    figure;
    plot(t, x_true(idx_p(i),:), 'LineWidth', 1.5); hold on;
    plot(t, z(i,:), '.', 'MarkerSize', 5);
    plot(t, x_hat(idx_p(i),:), 'LineWidth', 1.5);

    xlabel('Time [s]');
    ylabel(sprintf('Position p_%d [m]', i));
    legend('True','Measured abs','Estimated');
    grid on;
    title(sprintf('Satellite %d position', i));
end

%% 2) Relative position figure for each subsequent pair
% For chain topology: (1,2), (2,3), ..., (n-1,n)
for e = 1:(n-1)
    i = edges(e,1);
    j = edges(e,2);

    p_rel_true = x_true(idx_p(j),:) - x_true(idx_p(i),:);
    p_rel_est  = x_hat(idx_p(j),:) - x_hat(idx_p(i),:);

    figure;
    plot(t, p_rel_true, 'LineWidth', 1.5); hold on;
    plot(t, z(n+e,:), '.', 'MarkerSize', 5);
    plot(t, p_rel_est, 'LineWidth', 1.5);

    xlabel('Time [s]');
    ylabel(sprintf('Relative position p_%d - p_%d [m]', j, i));
    legend('True','Measured rel','Estimated');
    grid on;
    title(sprintf('Relative position: satellites %d and %d', i, j));
end

%% 3) One figure with all velocities
figure; hold on;
for i = 1:n
    plot(t, x_true(idx_v(i),:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('True v_%d', i));
    plot(t, x_hat(idx_v(i),:), '--', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Estimated v_%d', i));
end
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('Location','bestoutside');
grid on;
title('Velocities of all satellites');

%% 4) One figure with all inputs
figure; hold on;
for i = 1:n
    plot(t(1:N), u(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('u_%d', i));
end
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');
legend('Location','bestoutside');
grid on;
title('Control inputs of all satellites');

%% Position errors with centralized KF and empirical MC bounds

for i = 1:n
    figure('Name',sprintf('Position_error_satellite_%d',i), ...
           'Color','w','Units','centimeters','Position',[3 3 16 9]);

    plot(t, x_true(2*i-1,:) - x_hat(2*i-1,:), ...
        'k', 'LineWidth', 1.3);
    hold on;

    % Centralized KF theoretical bounds: blue dashed
    plot(t,  3*sigma_p(i,:), ...
        'Color', [0 0.4470 0.7410], ...
        'LineStyle', '--', ...
        'LineWidth', 1.5);
    plot(t, -3*sigma_p(i,:), ...
        'Color', [0 0.4470 0.7410], ...
        'LineStyle', '--', ...
        'LineWidth', 1.5);

    % Empirical Monte Carlo bounds: orange dotted
    plot(t,  3*sigma_p_mc(i,:), ...
        'Color', [0.8500 0.3250 0.0980], ...
        'LineStyle', ':', ...
        'LineWidth', 2.0);
    plot(t, -3*sigma_p_mc(i,:), ...
        'Color', [0.8500 0.3250 0.0980], ...
        'LineStyle', ':', ...
        'LineWidth', 2.0);

    xlabel('Time [s]', 'FontName','Times New Roman', 'FontSize',12, 'Color','k');
    ylabel('Position error [m]', 'FontName','Times New Roman', 'FontSize',12, 'Color','k');

    legend('Error', ...
           '+3\sigma Centralized KF', '-3\sigma Centralized KF', ...
           '+3\sigma empirical MC', '-3\sigma empirical MC', ...
           'Location','northeast');

    grid on;
    title(sprintf('Position error satellite %d', i), ...
      'Color','k', ...
      'FontName','Times New Roman', ...
      'FontSize',12, ...
      'FontWeight','bold');

    xlim([t(1) t(end)]);
end

%% Velocity errors with centralized KF and empirical MC bounds

for i = 1:n
    figure('Name',sprintf('Velocity_error_satellite_%d',i), ...
           'Color','w','Units','centimeters','Position',[3 3 16 9]);

    plot(t, x_true(2*i,:) - x_hat(2*i,:), ...
        'k', 'LineWidth', 1.3);
    hold on;

    % Centralized KF theoretical bounds: blue dashed
    plot(t,  3*sigma_v(i,:), ...
        'Color', [0 0.4470 0.7410], ...
        'LineStyle', '--', ...
        'LineWidth', 1.5);
    plot(t, -3*sigma_v(i,:), ...
        'Color', [0 0.4470 0.7410], ...
        'LineStyle', '--', ...
        'LineWidth', 1.5);

    % Empirical Monte Carlo bounds: orange dotted
    plot(t,  3*sigma_v_mc(i,:), ...
        'Color', [0.8500 0.3250 0.0980], ...
        'LineStyle', ':', ...
        'LineWidth', 2.0);
    plot(t, -3*sigma_v_mc(i,:), ...
        'Color', [0.8500 0.3250 0.0980], ...
        'LineStyle', ':', ...
        'LineWidth', 2.0);

    xlabel('Time [s]', 'FontName','Times New Roman', 'FontSize',12, 'Color','k');
    ylabel('Velocity error [m/s]', 'FontName','Times New Roman', 'FontSize',12, 'Color','k');

    legend('Error', ...
           '+3\sigma Centralized KF', '-3\sigma Centralized KF', ...
           '+3\sigma empirical MC', '-3\sigma empirical MC', ...
           'Location','northeast');

    grid on;
    title(sprintf('Velocity error satellite %d', i), ...
      'Color','k', ...
      'FontName','Times New Roman', ...
      'FontSize',12, ...
      'FontWeight','bold');

    xlim([t(1) t(end)]);
end


