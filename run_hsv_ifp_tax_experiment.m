function results = run_hsv_ifp_tax_experiment()
%RUN_HSV_IFP_TAX_EXPERIMENT
% Income fluctuation problem with inelastic labor supply and HSV taxes.
%
% Household budget constraint:
%
%   c + a' = a + y_aftertax
%
% where
%
%   y_pretax    = r*a + w*z
%   y_aftertax  = lambda * y_pretax^(1 - tau)
%   T(y)        = y_pretax - y_aftertax
%
% Labor supply is inelastic and normalized to one. z is idiosyncratic labor
% productivity.
%
% The code:
%   1. Solves the benchmark economy.
%   2. Computes the stationary distribution and model moments.
%   3. Loops over tau.
%   4. For each tau, adjusts lambda to keep total tax revenue equal to the
%      benchmark level.

    par = set_params();

    fprintf('\nSolving benchmark economy...\n');

    sol_bench = solve_model(par);

    target_revenue = sol_bench.moments.tax_revenue;

    fprintf('\nBenchmark moments:\n');
    disp(sol_bench.moments)

    % Tax experiment: vary progressivity tau and adjust lambda.
    tau_grid = par.experiment.tau_grid;
    nTau = numel(tau_grid);

    lambda_grid = nan(nTau, 1);
    tax_revenue_grid = nan(nTau, 1);
    K_grid = nan(nTau, 1);
    Y_grid = nan(nTau, 1);
    C_grid = nan(nTau, 1);
    welfare_grid = nan(nTau, 1);
    success_grid = false(nTau, 1);

    sol_experiment = cell(nTau, 1);

    fprintf('\nRunning revenue-neutral tax experiment...\n');

    for itau = 1:nTau

        tau_new = tau_grid(itau);

        fprintf('  tau = %.4f...\n', tau_new);

        par_tmp = par;
        par_tmp.tax.tau = tau_new;

        % Find lambda such that tax revenue equals benchmark revenue.
        [lambda_star, sol_star, success] = find_revenue_neutral_lambda( ...
            par_tmp, target_revenue);

        success_grid(itau) = success;

        if success
            lambda_grid(itau) = lambda_star;
            tax_revenue_grid(itau) = sol_star.moments.tax_revenue;
            K_grid(itau) = sol_star.moments.K;
            Y_grid(itau) = sol_star.moments.Y;
            C_grid(itau) = sol_star.moments.C;
            welfare_grid(itau) = sol_star.moments.welfare;
            sol_experiment{itau} = sol_star;
        end

    end

    experiment_table = table( ...
        tau_grid(:), ...
        lambda_grid, ...
        tax_revenue_grid, ...
        K_grid, ...
        Y_grid, ...
        C_grid, ...
        welfare_grid, ...
        success_grid, ...
        'VariableNames', {'tau', 'lambda', 'tax_revenue', 'K', 'Y', 'C', ...
                          'welfare', 'success'});

    fprintf('\nRevenue-neutral experiment results:\n');
    disp(experiment_table)

    results.par = par;
    results.benchmark = sol_bench;
    results.target_revenue = target_revenue;
    results.experiment_table = experiment_table;
    results.sol_experiment = sol_experiment;

end %end function


function par = set_params()
%SET_PARAMS Define all model, numerical, and experiment parameters.

    % Preferences
    par.beta = 0.96;
    par.sigma = 2.0;

    % Prices, partial equilibrium
    par.r = 0.03;
    par.w = 1.00;

    % Production function, used only for aggregate output accounting
    par.prod.A = 1.00;
    par.prod.alpha = 0.36;

    % HSV tax function:
    %   after-tax income = lambda * y^(1 - tau)
    %   tax liability    = y - lambda * y^(1 - tau)
    par.tax.lambda = 0.90;
    par.tax.tau = 0.18;

    % Asset grid
    par.grid.Na = 500;
    par.grid.a_min = 0.0;
    par.grid.a_max = 80.0;
    par.grid.curv = 2.0;

    x = linspace(0, 1, par.grid.Na)';
    par.grid.a = par.grid.a_min + ...
        (par.grid.a_max - par.grid.a_min) * x.^par.grid.curv;

    % Idiosyncratic productivity process
    % log z' = rho_z log z + eps
    par.z.Nz = 7;
    par.z.rho = 0.90;
    par.z.sigma_eps = 0.20;
    par.z.m = 3.0;

    [logz_grid, Pz] = tauchen( ...
        par.z.Nz, 0.0, par.z.rho, par.z.sigma_eps, par.z.m);

    z_grid = exp(logz_grid(:));

    % Normalize E[z] = 1 under the invariant distribution of z.
    pi_z = stationary_markov(Pz);
    z_grid = z_grid / (pi_z(:)' * z_grid);

    par.z.grid = z_grid;
    par.z.P = Pz;
    par.z.pi = stationary_markov(Pz);

    % VFI options
    par.vfi.max_iter = 1000;
    par.vfi.tol = 1e-8;
    par.vfi.verbose = false;

    % Distribution iteration options
    par.dist.max_iter = 5000;
    par.dist.tol = 1e-12;
    par.dist.verbose = false;

    % Revenue-neutral experiment
    par.experiment.tau_grid = linspace(0.05, 0.30, 11);
    par.experiment.lambda_bracket = [0.05, 2.00];

end %end function


function sol = solve_model(par)
%SOLVE_MODEL Solve household problem, stationary distribution, and moments.

    [V, policy_ind, policy_a, policy_c] = solve_vfi(par);

    mu = stationary_distribution(policy_ind, par);

    moments = compute_moments(mu, policy_c, V, par);

    sol.V = V;
    sol.policy_ind = policy_ind;
    sol.policy_a = policy_a;
    sol.policy_c = policy_c;
    sol.mu = mu;
    sol.moments = moments;

end %end function


function [V, policy_ind, policy_a, policy_c] = solve_vfi(par)
%SOLVE_VFI Value function iteration.

    a_grid = par.grid.a;
    z_grid = par.z.grid;
    Pz = par.z.P;

    Na = par.grid.Na;
    Nz = par.z.Nz;

    beta = par.beta;

    V = zeros(Na, Nz);
    V_new = zeros(Na, Nz);

    policy_ind = ones(Na, Nz);
    policy_a = zeros(Na, Nz);
    policy_c = zeros(Na, Nz);

    diff = Inf;

    for iter = 1:par.vfi.max_iter

        for iz = 1:Nz

            EV = V * Pz(iz, :)';

            z = z_grid(iz);

            for ia = 1:Na

                a = a_grid(ia);

                y_pre_tax = par.r * a + par.w * z;
                y_after_tax = hsv_after_tax_income(y_pre_tax, par.tax.lambda, par.tax.tau);

                cash_on_hand = a + y_after_tax;

                c_vec = cash_on_hand - a_grid;

                u_vec = utility(c_vec, par.sigma);

                rhs = u_vec + beta * EV;

                [V_new(ia, iz), best_ind] = max(rhs);

                policy_ind(ia, iz) = best_ind;
                policy_a(ia, iz) = a_grid(best_ind);
                policy_c(ia, iz) = c_vec(best_ind);

            end

        end

        diff = max(abs(V_new(:) - V(:)));

        V = V_new;

        if par.vfi.verbose && (mod(iter, 25) == 0 || iter == 1)
            fprintf('VFI iter %4d, diff = %.3e\n', iter, diff);
        end

        if diff < par.vfi.tol
            break
        end

    end

    if diff >= par.vfi.tol
        warning('VFI did not converge. Final diff = %.3e.', diff);
    end

end %end function


function mu = stationary_distribution(policy_ind, par)
%STATIONARY_DISTRIBUTION Compute invariant distribution over (a,z).

    Na = par.grid.Na;
    Nz = par.z.Nz;
    Pz = par.z.P;

    mu = ones(Na, Nz) / (Na * Nz);
    diff = Inf;

    for iter = 1:par.dist.max_iter

        mu_new = zeros(Na, Nz);

        for iz = 1:Nz
            for ia = 1:Na

                mass = mu(ia, iz);

                if mass > 0
                    iap = policy_ind(ia, iz);
                    mu_new(iap, :) = mu_new(iap, :) + mass * Pz(iz, :);
                end

            end
        end

        diff = max(abs(mu_new(:) - mu(:)));

        mu = mu_new;

        if par.dist.verbose && (mod(iter, 100) == 0 || iter == 1)
            fprintf('Distribution iter %4d, diff = %.3e\n', iter, diff);
        end

        if diff < par.dist.tol
            break
        end

    end

    mu = mu / sum(mu(:));

    if diff >= par.dist.tol
        warning('Distribution did not converge. Final diff = %.3e.', diff);
    end

end %end function


function moments = compute_moments(mu, policy_c, V, par)
%COMPUTE_MOMENTS Compute aggregate moments.

    a_grid = par.grid.a;
    z_grid = par.z.grid;

    [A_grid, Z_grid] = ndgrid(a_grid, z_grid);

    y_pre_tax = par.r * A_grid + par.w * Z_grid;

    y_after_tax = hsv_after_tax_income(y_pre_tax, par.tax.lambda, par.tax.tau);

    tax_liability = y_pre_tax - y_after_tax;

    K = sum(mu(:) .* A_grid(:));

    L_eff = sum(mu(:) .* Z_grid(:));

    Y = par.prod.A * K^par.prod.alpha * L_eff^(1 - par.prod.alpha);

    C = sum(mu(:) .* policy_c(:));

    tax_revenue = sum(mu(:) .* tax_liability(:));

    avg_tax_rate = tax_revenue / sum(mu(:) .* y_pre_tax(:));

    welfare = sum(mu(:) .* V(:));

    moments.K = K;
    moments.L_eff = L_eff;
    moments.Y = Y;
    moments.C = C;
    moments.tax_revenue = tax_revenue;
    moments.avg_tax_rate = avg_tax_rate;
    moments.welfare = welfare;
    moments.lambda = par.tax.lambda;
    moments.tau = par.tax.tau;

end %end function


function [lambda_star, sol_star, success] = find_revenue_neutral_lambda(par, target_revenue)
%FIND_REVENUE_NEUTRAL_LAMBDA
% Given tau, find lambda such that aggregate tax revenue equals target.

    lambda_bracket = par.experiment.lambda_bracket;

    log_lambda_low = log(lambda_bracket(1));
    log_lambda_high = log(lambda_bracket(2));

    revenue_gap = @(log_lambda) revenue_gap_given_log_lambda( ...
        log_lambda, par, target_revenue);

    f_low = revenue_gap(log_lambda_low);
    f_high = revenue_gap(log_lambda_high);

    success = true;

    % fzero needs the revenue gap to change sign over the lambda bracket.
    if sign(f_low) == sign(f_high)

        warning(['Could not bracket revenue-neutral lambda for tau = %.4f. ', ...
                 'Revenue gap at bracket endpoints: %.4e, %.4e.'], ...
                 par.tax.tau, f_low, f_high);

        lambda_star = NaN;
        sol_star = [];
        success = false;
        return

    end

    opts = optimset('TolX', 1e-5, 'Display', 'off');

    log_lambda_star = fzero(revenue_gap, [log_lambda_low, log_lambda_high], opts);

    lambda_star = exp(log_lambda_star);

    par_star = par;
    par_star.tax.lambda = lambda_star;
    sol_star = solve_model(par_star);

end %end function


function gap = revenue_gap_given_log_lambda(log_lambda, par, target_revenue)
%REVENUE_GAP_GIVEN_LOG_LAMBDA Auxiliary objective for root-finding.

    par_tmp = par;
    par_tmp.tax.lambda = exp(log_lambda);
    sol = solve_model(par_tmp);

    gap = sol.moments.tax_revenue - target_revenue;

end %end function


function y_after_tax = hsv_after_tax_income(y, lambda, tau)
%HSV_AFTER_TAX_INCOME
% Heathcote-Storesletten-Violante tax function:
%
%   y_after_tax = lambda * y^(1 - tau)
%   T(y)        = y - lambda * y^(1 - tau)

    y = max(y, 1e-12);

    y_after_tax = lambda .* y.^(1 - tau);

end %end function


function u = utility(c, sigma)
%UTILITY CRRA utility.

    u = -inf(size(c));

    good = c > 0;

    if abs(sigma - 1.0) < 1e-12
        u(good) = log(c(good));
    else
        u(good) = (c(good).^(1 - sigma) - 1) ./ (1 - sigma);
    end

end %end function


function [z_grid, P] = tauchen(N, mu, rho, sigma_eps, m)
%TAUCHEN Discretize AR(1):
%
%   z' = mu + rho z + eps, eps ~ N(0, sigma_eps^2)
%
% Returns grid for z and transition matrix P.

    sigma_z = sigma_eps / sqrt(1 - rho^2);

    z_min = mu - m * sigma_z;
    z_max = mu + m * sigma_z;

    z_grid = linspace(z_min, z_max, N)';

    step = z_grid(2) - z_grid(1);

    P = zeros(N, N);

    for i = 1:N
        for j = 1:N

            if j == 1

                upper = (z_grid(1) - mu - rho * z_grid(i) + step / 2) / sigma_eps;
                P(i, j) = normcdf_fast(upper);

            elseif j == N

                lower = (z_grid(N) - mu - rho * z_grid(i) - step / 2) / sigma_eps;
                P(i, j) = 1 - normcdf_fast(lower);

            else

                upper = (z_grid(j) - mu - rho * z_grid(i) + step / 2) / sigma_eps;
                lower = (z_grid(j) - mu - rho * z_grid(i) - step / 2) / sigma_eps;

                P(i, j) = normcdf_fast(upper) - normcdf_fast(lower);

            end

        end
    end

    P = P ./ sum(P, 2);

end %end function


function p = normcdf_fast(x)
%NORMCDF_FAST Normal CDF without requiring Statistics Toolbox.

    p = 0.5 * erfc(-x ./ sqrt(2));

end %end function


function pi = stationary_markov(P)
%STATIONARY_MARKOV Stationary distribution of a finite Markov chain.

    N = size(P, 1);

    pi = ones(1, N) / N;

    for iter = 1:10000

        pi_new = pi * P;

        if max(abs(pi_new - pi)) < 1e-14
            pi = pi_new;
            break
        end

        pi = pi_new;

    end

    pi = pi(:) / sum(pi);

end %end function
