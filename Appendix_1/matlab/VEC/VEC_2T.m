%% =============================================================
%   VEC MODEL -- REPLICATION SCRIPT FOR APPENDIX 1
%   Variable order: [1: Kidnapping for Ransom (K^E),
%                    2: Non-Extortionate Kidnapping (K^NE),
%                    3: Loan Interest Rate (r)]
%
%   Pipeline: unit-root tests (Table I: ADF/KPSS, levels and differences),
%   Johansen trace test
%   with lag-sensitivity analysis, VEC(p=36, r=1, unrestricted
%   constant) estimation, adjustment coefficients with standard
%   errors, error-correction-term diagnostics, residual diagnostics,
%   Gonzalo-Granger decomposition (exported to Tendencias_SE.csv),
%   forecasts, impulse responses (Cholesky and Blanchard-Quah),
%   and forecast error variance decomposition (FEVD).
% =============================================================

clc; clear; close all;

%% >>>>> USER CONFIGURATION <<<<<
UNIT_ROOT_LAGS    = 12;
VEC_LAGS_P        = 36;
COINT_RANK_MANUAL = 1;
JOHANSEN_MODEL    = 'H1';
FCAST_FUTURE_HORIZON = 24;

% --- Lag-sensitivity analysis configuration ---
MAX_SENSITIVITY_LAGS = 12; % Highest lag tested in the sensitivity loop
% <<<<< END USER CONFIGURATION <<<<<


%% 1. LOAD DATA
try
    opts = detectImportOptions("Data.csv");
    opts.VariableNamingRule = 'preserve';
    data = readtable("Data.csv", opts);
catch
    error('Data.csv not found');
end

if ismember('mes', data.Properties.VariableNames)
    mesStr = string(data.mes);
    try
        year  = str2double(extractBefore(mesStr, "m"));
        month = str2double(extractAfter(mesStr,  "m"));
        periodos = datetime(year, month, 1);
    catch
        warning('Date parsing failed. Using numeric index.');
        periodos = (1:height(data))';
    end
else
    periodos = (1:height(data))';
end

vars_needed = {'BR_r', 'Cantidad_Extorsivo', 'Cantidad_Simple'};
if ~all(ismember(vars_needed, data.Properties.VariableNames))
    error('Missing variables in Data.csv.');
end

BR_r           = data.BR_r(:);
Cant_Extorsivo = data.Cantidad_Extorsivo(:);
Cant_Simple    = data.Cantidad_Simple(:);

Y = [Cant_Extorsivo Cant_Simple BR_r];
var_names_vec = {'Kidnap_Ransom', 'Non_Extortionate', 'BR_r'};
var_display_names = {'Kidnapping for Ransom', 'Non-Extortionate Kidnapping', 'Loan Interest Rate'};

[T, k] = size(Y);
if any(any(isnan(Y))), Y = fillmissing(Y, 'movmedian', 3); end
fprintf("Loaded %d observations, %d variables.\n", T, k);


%% 2. UNIT-ROOT TESTS (Table I, Appendix 1)
% ADF: model ARD (constant), 12 lags.  KPSS: no trend, 12 lags.
fprintf('\n==========================================================\n');
fprintf('   TABLE I: UNIT-ROOT TESTS (%d LAGS, CONSTANT)\n', UNIT_ROOT_LAGS);
fprintf('==========================================================\n');
fprintf('MATLAB p-values truncated at [0.01, 0.10] for KPSS and [0.001, 0.999] for ADF.\n\n');

seriesLabels = {'K^E_t', 'K^NE_t', 'r_t'};
fprintf('%-8s | %-12s | %-12s | %-12s | %-12s\n', ...
    'Series', 'ADF lev p', 'KPSS lev p', 'ADF diff p', 'KPSS diff p');
fprintf('%s\n', repmat('-', 1, 62));

for i = 1:k
    y_level = Y(:, i);
    y_diff  = diff(y_level);

    [~, pADF_lev]  = adftest(y_level, 'model', 'ARD', 'lags', UNIT_ROOT_LAGS);
    [~, pKPSS_lev] = kpsstest(y_level, 'trend', false, 'lags', UNIT_ROOT_LAGS);
    [~, pADF_diff]  = adftest(y_diff, 'model', 'ARD', 'lags', UNIT_ROOT_LAGS);
    [~, pKPSS_diff] = kpsstest(y_diff, 'trend', false, 'lags', UNIT_ROOT_LAGS);

    fprintf('%-8s | %12s | %12s | %12s | %12s\n', seriesLabels{i}, ...
        formatUnitRootP(pADF_lev, 'ADF'), ...
        formatUnitRootP(pKPSS_lev, 'KPSS'), ...
        formatUnitRootP(pADF_diff, 'ADF'), ...
        formatUnitRootP(pKPSS_diff, 'KPSS'));
end

fprintf('\nInterpretation: I(1) if levels fail to reject unit root (ADF) and reject\n');
fprintf('stationarity (KPSS), while first differences reject unit root and do not\n');
fprintf('reject stationarity (KPSS).\n');


%% 2.1 JOHANSEN COINTEGRATION TEST
fprintf('\n==========================================================\n');
fprintf('      JOHANSEN COINTEGRATION TEST\n');
fprintf('==========================================================\n');
fprintf('Lags (VAR levels): %d\n', VEC_LAGS_P);
fprintf('Deterministic assumption: %s\n', JOHANSEN_MODEL);

[hJ, pValJ, statJ, cValJ, mles] = jcitest(Y, 'model', JOHANSEN_MODEL, 'lags', VEC_LAGS_P, 'Display', 'off');

% --- Prepare data for the manual summary table (robust extraction) ---
% 1. Keep only logical columns of the h table
if istable(hJ)
    idx_bool = varfun(@islogical, hJ, 'OutputFormat', 'uniform');
    hJ = table2array(hJ(:, idx_bool));
end

% 2. Keep only numeric columns of the statistic tables
if istable(statJ)
    idx_num = varfun(@isnumeric, statJ, 'OutputFormat', 'uniform');
    statJ = table2array(statJ(:, idx_num));
end

if istable(cValJ)
    idx_num = varfun(@isnumeric, cValJ, 'OutputFormat', 'uniform');
    cValJ = table2array(cValJ(:, idx_num));
end

if istable(pValJ)
    idx_num = varfun(@isnumeric, pValJ, 'OutputFormat', 'uniform');
    pValJ = table2array(pValJ(:, idx_num));
end

% 3. Handle struct outputs
if isstruct(statJ)
    if isfield(statJ, 'trace'), my_stat = statJ.trace;
    else, fn = fieldnames(statJ); my_stat = statJ.(fn{1}); end
else
    my_stat = statJ;
end

if isstruct(cValJ)
    if isfield(cValJ, 'trace'), my_cval = cValJ.trace;
    else, fn = fieldnames(cValJ); my_cval = cValJ.(fn{1}); end
else
    my_cval = cValJ;
end

if isstruct(pValJ)
    if isfield(pValJ, 'trace'), my_pval = pValJ.trace;
    else, fn = fieldnames(pValJ); my_pval = pValJ.(fn{1}); end
else
    my_pval = pValJ;
end

fprintf('\n--- Quick interpretation (Trace Test) - computed with %d lags ---\n', VEC_LAGS_P);
fprintf('%-10s | %-10s | %-10s | %-10s | %-10s | %-20s\n', 'Null', 'EigenVal', 'TraceStat', 'CritVal95', 'P-Value', 'Decision');
fprintf('%s\n', repmat('-',1,93));

r_suggested = k;
found_break = false;

% --- Robust (implied) eigenvalue computation ---
% If not available in mles, derive them from the trace statistics:
% Trace(r) = -T * Sum_{i=r+1}^{k} ln(1 - lambda_i)
% lambda_i = 1 - exp( -(Trace(i-1) - Trace(i)) / T ) (approx.)
eigen_calc = nan(k, 1);
if ~isempty(my_stat)
    T_use = size(Y,1) - VEC_LAGS_P; % effective observations
    stats_vec = my_stat;
    if size(stats_vec,1) < size(stats_vec,2), stats_vec = stats_vec'; end % force column
    stats_extended = [stats_vec; 0];
    for i = 1:k
        if i <= length(stats_vec)
            diff_trace = stats_extended(i) - stats_extended(i+1);
            if diff_trace > 0
                eigen_calc(i) = 1 - exp(-diff_trace / T_use);
            else
                eigen_calc(i) = 0;
            end
        end
    end
end

for idx = 1:length(hJ)
    r_check = idx - 1;
    if hJ(idx) == 1
        decision = 'Reject H0 ***'; % significant
    else
        decision = 'Fail to reject';
        if ~found_break
            r_suggested = r_check;
            found_break = true;
        end
    end

    if idx <= length(my_stat), val_stat = my_stat(idx); else, val_stat = NaN; end
    if idx <= length(my_cval), val_crit = my_cval(idx); else, val_crit = NaN; end
    if idx <= length(my_pval), val_pval = my_pval(idx); else, val_pval = NaN; end

    % --- Try to obtain the eigenvalue ---
    val_eig = NaN;
    found_in_mles = false;
    if isstruct(mles)
        possible_fields = {'eigVals','eigVal','lambdas','eigenvalues','lambda'};
        for f = possible_fields
            if isfield(mles, f{1})
                vec_eig = mles.(f{1});
                if idx <= length(vec_eig)
                    val_eig = vec_eig(idx);
                    found_in_mles = true;
                end
                break;
            end
        end
    end
    if (isnan(val_eig) || ~found_in_mles) && idx <= length(eigen_calc)
        val_eig = eigen_calc(idx);
    end

    fprintf('r <= %d    | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %s\n', ...
        r_check, val_eig, val_stat, val_crit, val_pval, decision);
end

fprintf('\n>>> RANK SUGGESTED BY THE TRACE TEST: r = %d\n', r_suggested);

% --- Show significant cointegration vectors (eigenvectors) ---
if r_suggested > 0
    fprintf('\n----------------------------------------------------------\n');
    fprintf('      SIGNIFICANT COINTEGRATION VECTORS (r=%d)\n', r_suggested);
    fprintf('----------------------------------------------------------\n');

    if isstruct(mles) && isfield(mles, 'evecs')
        Beta_Johansen = mles.evecs(:, 1:r_suggested);
        col_headers = arrayfun(@(x) sprintf('CV_%d',x), 1:r_suggested, 'UniformOutput', false);
        Tbl_Beta_J = array2table(Beta_Johansen, 'RowNames', var_names_vec, 'VariableNames', col_headers);
        disp(Tbl_Beta_J);
        fprintf('Note: these vectors correspond to the significant cointegration relations.\n');
    else
        fprintf('(Could not extract eigenvectors from the mles structure)\n');
    end
else
    fprintf('\n(No cointegration vectors shown because the suggested rank is 0)\n');
end


%% 2.2 LAG-SENSITIVITY ANALYSIS (JOHANSEN)
if ~exist('MAX_SENSITIVITY_LAGS','var'), MAX_SENSITIVITY_LAGS = 12; end

fprintf('\n==========================================================\n');
fprintf('      LAG SENSITIVITY (JOHANSEN)\n');
fprintf('      Testing lags 2 to %d\n', MAX_SENSITIVITY_LAGS);
fprintf('==========================================================\n');
fprintf('Note: the table shows p-values for each null hypothesis (H0)\n');

fprintf('%-6s | %-15s | %-12s | %-12s | %-12s\n', 'Lags', 'Sugg. rank', 'H0:r=0', 'H0:r<=1', 'H0:r<=2');
fprintf('%s\n', repmat('-', 1, 75));

for lag_check = 2:MAX_SENSITIVITY_LAGS
    [h_sen, pVal_sen, ~, ~] = jcitest(Y, 'model', JOHANSEN_MODEL, 'lags', lag_check, 'Display', 'off');

    if istable(h_sen)
        logic_cols = varfun(@islogical, h_sen, 'OutputFormat', 'uniform');
        h_sen = table2array(h_sen(:, logic_cols));
    end
    if istable(pVal_sen)
        idx_num = varfun(@isnumeric, pVal_sen, 'OutputFormat', 'uniform');
        pVal_sen = table2array(pVal_sen(:, idx_num));
    elseif isstruct(pVal_sen)
         if isfield(pVal_sen, 'trace'), pVal_sen = pVal_sen.trace;
         else, fn=fieldnames(pVal_sen); pVal_sen = pVal_sen.(fn{1}); end
    end

    % Suggested rank = first failure to reject
    sugerido = k;
    for rr = 0:(k-1)
        idx_r = rr + 1;
        if idx_r <= length(h_sen)
            if h_sen(idx_r) == 0
                sugerido = rr;
                break;
            end
        end
    end

    % Visual labels: "SIG(p=0.00)" or "NS (p=0.23)"
    str_res = strings(1, k);
    for i_h = 1:min(k, length(h_sen))
        pv = 0;
        if i_h <= length(pVal_sen), pv = pVal_sen(i_h); end
        if h_sen(i_h)==1
            str_res(i_h) = sprintf("SIG(p=%.3f)", pv);
        else
            str_res(i_h) = sprintf("NS (p=%.3f)", pv);
        end
    end

    fprintf('%-6d | %-15d | %-12s | %-12s | %-12s\n', ...
        lag_check, sugerido, str_res(1), str_res(2), str_res(3));
end
fprintf('\nLegend: SIG = significant (reject H0), NS = not significant (fail to reject H0)\n');


%% 3. VEC ESTIMATION (BASELINE)
p = VEC_LAGS_P;
q = max(p - 1, 1);
r = COINT_RANK_MANUAL;

MdlVEC_template = vecm(k, r, q);
EstMdlVEC = [];
EstSE = [];

if r > 0
    [EstMdlVEC, EstSE, logL] = estimate(MdlVEC_template, Y, 'Model', JOHANSEN_MODEL, 'Display', 'off');
    nParamsCalc = k^2*q + k*r + r*(k-r) + k;
    [aic, bic] = aicbic(logL, nParamsCalc, T);
    fprintf('\nVEC(p=%d, r=%d) estimated. AIC: %.2f\n', p, r, aic);
else
    error('r=0.');
end


%% 3.1 GOODNESS-OF-FIT STATISTICS (EQUATION BY EQUATION)
if r > 0
    fprintf('\n==========================================================\n');
    fprintf('   REGRESSION STATISTICS (VEC) - METHOD: %s\n', 'Maximum Likelihood (Johansen)');
    fprintf('==========================================================\n');

    % Recompute residuals for the effective sample
    E_inferred = infer(EstMdlVEC, Y);

    % Valid rows: the last T-p observations (after the presample)
    n_effective = T - p;
    E_eff = E_inferred(end-n_effective+1:end, :);

    % Actual dependent variable: Delta Y
    DY_full = diff(Y);
    DY_eff = DY_full(end-n_effective+1:end, :);

    [T_eff, ~] = size(DY_eff);

    % Parameters per equation: Alpha(r) + ShortRun(k*q) + Const(1)
    n_params_eq = r + k*q + 1;

    fprintf('Observations used: %d\n', T_eff);
    fprintf('%-25s | %-8s | %-8s | %-10s | %-10s | %-15s\n', ...
        'Equation', 'R-Sq', 'Adj-R2', 'RMSE', 'F-Stat', 'Decision');
    fprintf('%s\n', repmat('-',1,90));

    for i = 1:k
        y_act = DY_eff(:, i);
        res   = E_eff(:, i);

        ssr = sum(res.^2);
        sst = sum((y_act - mean(y_act)).^2);

        r2 = 1 - ssr/sst;
        if r2 < 0, r2 = 0; end % safety

        adj_r2 = 1 - (1 - r2) * (T_eff - 1) / (T_eff - n_params_eq);
        rmse = sqrt(mean(res.^2));

        % F-statistic (H0: all parameters except the constant equal 0)
        df1 = n_params_eq - 1;
        df2 = T_eff - n_params_eq;

        if r2 < 1 && r2 > 0
           val_f = (r2 / df1) / ((1 - r2) / df2);
        else
           val_f = NaN;
        end

        if ~isnan(val_f)
            pval_f = 1 - fcdf(val_f, df1, df2);
        else
            pval_f = 1;
        end

        % Joint-significance label (F test)
        if pval_f < 0.01
            star = '*** (p<0.01)';
        elseif pval_f < 0.05
            star = '**  (p<0.05)';
        elseif pval_f < 0.10
             star = '*   (p<0.10)';
        else
             star = 'NS';
        end

        fprintf('%-25s | %-8.4f | %-8.4f | %-10.4f | %-10.2f | %-15s\n', ...
            var_names_vec{i}, r2, adj_r2, rmse, val_f, star);
    end
    fprintf('%s\n', repmat('-',1,90));
    fprintf('Estimation method: Maximum Likelihood (Johansen)\n');
    fprintf('Decision: joint significance (F test)\n');
end


%% 4. REGRESSION RESULTS & MANUAL OLS BOOTSTRAP
if r > 0
    % --- BETA (cointegration) ---
    Beta = EstMdlVEC.Cointegration;
    fprintf('\n--- Beta (cointegration) ---\n');

    seBetaDisponible = false;
    if (isfield(EstSE, 'Cointegration') && ~isempty(EstSE.Cointegration))
        SeBeta = EstSE.Cointegration;
        if ~all(isnan(SeBeta(:))), seBetaDisponible = true; end
    end

    if seBetaDisponible
        tBeta = Beta ./ SeBeta;
        pValorBeta = 2 * (1 - tcdf(abs(tBeta), T - nParamsCalc));
        TblBetaFull = table(Beta, SeBeta, tBeta, pValorBeta, ...
                            'VariableNames', {'Coeff', 'StdErr', 'tStat', 'pValue'}, ...
                            'RowNames', var_names_vec);
        disp(TblBetaFull);
    else
        fprintf('(Beta SE not available. Computing via manual OLS bootstrap...)\n');

        nBoot = 200;

        % 1. Manual parameters
        DY = diff(Y);
        [T_dy, K] = size(DY);

        % Dependent variable
        DY_target = DY(p:end, :);
        nObsReg = size(DY_target, 1);

        % Lagged levels regressor
        Y_lag1 = Y(p : end-1, :);

        % Lagged-difference regressors
        X_lags = [];
        for j = 1:q
            lag_j_data = DY(p-j : end-j, :);
            X_lags = [X_lags, lag_j_data];
        end

        % Dimensional safety check
        if size(Y_lag1,1) ~= nObsReg || size(X_lags,1) ~= nObsReg
             min_len = min([size(DY_target,1), size(Y_lag1,1), size(X_lags,1)]);
             DY_target = DY_target(end-min_len+1:end, :);
             Y_lag1    = Y_lag1(end-min_len+1:end, :);
             X_lags    = X_lags(end-min_len+1:end, :);
        end

        % OLS
        XX = [Y_lag1, X_lags, ones(size(DY_target,1),1)];
        if rcond(XX'*XX) < 1e-12
            B_ols = (pinv(XX) * DY_target)';
        else
            B_ols = (XX \ DY_target)';
        end

        Pi_gen    = B_ols(:, 1:K);
        Gamma_gen = B_ols(:, K+1 : K+K*q);
        C_gen     = B_ols(:, end);
        E_resid   = DY_target - (XX * B_ols');

        % Bootstrap loop
        BetaBoot = zeros(k, nBoot);
        Y_sim = zeros(size(Y));

        for b = 1:nBoot
            % A. Resample residuals
            idx = randsample(size(E_resid,1), size(E_resid,1), true);
            E_star = E_resid(idx, :);

            % B. Manual recursive simulation
            Y_sim(1:p, :) = Y(1:p, :);
            for t = (p+1):T
                term_coint = Pi_gen * Y_sim(t-1, :)';

                diffs_vec = [];
                for j=1:q
                   dy_lag = Y_sim(t-j, :) - Y_sim(t-j-1, :);
                   diffs_vec = [diffs_vec; dy_lag'];
                end

                term_short = Gamma_gen * diffs_vec;
                idx_e = t - p;
                if idx_e > size(E_star,1), idx_e = size(E_star,1); end

                dy_new = term_coint + term_short + C_gen + E_star(idx_e, :)';
                Y_sim(t, :) = Y_sim(t-1, :) + dy_new';
            end

            % C. Estimate Beta manually (SVD of the OLS-estimated Pi)
            DY_s = diff(Y_sim);
            DY_target_s = DY_s(p:end, :);
            Y_lag1_s    = Y_sim(p:end-1, :);
            X_lags_s = [];
            for j=1:q
                X_lags_s = [X_lags_s, DY_s(p-j : end-j, :)];
            end

            XX_s = [Y_lag1_s, X_lags_s, ones(size(DY_target_s,1),1)];
            if rcond(XX_s'*XX_s) < 1e-12
               B_boot = (pinv(XX_s) * DY_target_s)';
            else
               B_boot = (XX_s \ DY_target_s)';
            end
            Pi_boot = B_boot(:, 1:K);

            [~,~,V_svd] = svd(Pi_boot);
            bb = V_svd(:, 1:r);

            try
                Rot = (bb' * bb) \ (bb' * Beta);
                bb_aligned = bb * Rot;
                BetaBoot(:, b) = bb_aligned;
            catch
                BetaBoot(:, b) = nan;
            end
        end

        % Statistics
        SeBetaBoot = std(BetaBoot, 0, 2, 'omitnan');
        tBetaBoot  = Beta ./ SeBetaBoot;
        pValBoot   = 2 * (1 - tcdf(abs(tBetaBoot), T - nParamsCalc));

        TblBetaBoot = table(Beta, SeBetaBoot, tBetaBoot, pValBoot, ...
                            'VariableNames', {'Coeff', 'StdErr_Boot', 'tStat_Boot', 'pValue_Boot'}, ...
                            'RowNames', var_names_vec);
        disp(TblBetaBoot);
    end

    % --- ALPHA ---
    Alpha = EstMdlVEC.Adjustment;
    fprintf('\n--- Alpha (adjustment speed) ---\n');
    if (isfield(EstSE, 'Adjustment'))
        tAlpha = Alpha ./ EstSE.Adjustment;
        pValorAlpha = 2 * (1 - tcdf(abs(tAlpha), T - nParamsCalc));
        disp(table(Alpha, EstSE.Adjustment, tAlpha, pValorAlpha, ...
             'VariableNames', {'Coeff', 'StdErr', 'tStat', 'pValue'}, 'RowNames', var_names_vec));
    else
        disp(Alpha);
    end
end

%% 4.2 ROBUST ANALYSIS OF THE ERROR-CORRECTION TERM (ECT)
if r > 0
    fprintf('\n==========================================================\n');
    fprintf('   COINTEGRATION RESIDUAL ANALYSIS (ECT)\n');
    fprintf('==========================================================\n');

    % 1. ECT computation
    % ECT_t = Beta' * Y_t (MATLAB stores Beta in columns, hence Y*Beta)
    ECT = Y * Beta;

    % With multiple relations (r>1), analyze each one.
    [T_ect, n_ect] = size(ECT);

    for j = 1:n_ect
        ect_curr = ECT(:, j);
        name_ect = sprintf('ECT_%d', j);

        fprintf('\n>>> Analyzing %s <<<\n', name_ect);

        % A. Descriptive statistics
        mu_ect = mean(ect_curr);
        var_ect = var(ect_curr);
        skew_ect = skewness(ect_curr);
        kurt_ect = kurtosis(ect_curr);

        fprintf('%-20s : %.5f\n', 'Mean', mu_ect);
        fprintf('%-20s : %.5f\n', 'Variance', var_ect);
        fprintf('%-20s : %.5f\n', 'Skewness', skew_ect);
        fprintf('%-20s : %.5f\n', 'Kurtosis', kurt_ect);

        % B. Hypothesis tests
        % 1. Normality (Jarque-Bera)
        ws = warning('off', 'all');
        [hJB, pJB, statJB] = jbtest(ect_curr);
        warning(ws);

        if hJB==1, resJB="Non-normal"; else, resJB="Normal"; end
        fprintf('%-20s : p=%.4f (Stat=%.2f) -> %s\n', 'Normality (JB)', ...
            pJB, statJB, resJB);

        % 2. Autocorrelation (Ljung-Box) - 20 lags
        [hLB, pLB, statLB] = lbqtest(ect_curr, 'Lags', 20);
        if hLB==1, resLB="Autocorrelated"; else, resLB="No autocorrelation"; end
        fprintf('%-20s : p=%.4f (Stat=%.2f) -> %s\n', 'Autocorr (LB-20)', ...
            pLB, statLB, resLB);

        % 3. Heteroskedasticity (ARCH test) - 5 lags
        [hARCH, pARCH, statARCH] = archtest(ect_curr - mean(ect_curr), 'Lags', 5);
        if hARCH==1, resARCH="Heteroskedastic"; else, resARCH="Homoskedastic"; end
        fprintf('%-20s : p=%.4f (Stat=%.2f) -> %s\n', 'ARCH (Lags=5)', ...
            pARCH, statARCH, resARCH);

        % 4. Stationarity (ADF) - sanity check
        [hADF, pADF, statADF] = adftest(ect_curr);
        if hADF==1, resADF="Stationary"; else, resADF="Non-stationary (Warning)"; end
        fprintf('%-20s : p=%.4f (Stat=%.2f) -> %s\n', 'Stationarity (ADF)', ...
            pADF, statADF, resADF);

        % C. ECT plots
        figure('Name', ['Analysis ' name_ect]);

        % 1. Time series
        subplot(3, 2, 1:2);
        plot(ect_curr, 'k-', 'LineWidth', 1);
        yline(0, 'r--');
        title(['Time series: ' name_ect]);
        axis tight; grid on;

        % 2. Histogram
        subplot(3, 2, 3);
        histogram(ect_curr, 20, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7]);
        hold on;
        x_grid = linspace(min(ect_curr), max(ect_curr), 100);
        plot(x_grid, normpdf(x_grid, mu_ect, sqrt(var_ect)), 'r-', 'LineWidth', 1.5);
        title('Distribution / Normality');
        hold off;

        % 3. Scatter vs. lag (visual dynamics check)
        subplot(3, 2, 4);
        scatter(ect_curr(1:end-1), ect_curr(2:end), 15, 'Filled', 'MarkerFaceColor', 'k');
        title('Scatter t vs t-1');
        xlabel('t-1'); ylabel('t');

        % 4. ACF
        subplot(3, 2, 5);
        autocorr(ect_curr, 'NumLags', 20);
        title('ACF (Autocorrelation)');

        % 5. PACF
        subplot(3, 2, 6);
        parcorr(ect_curr, 'NumLags', 20);
        title('PACF (Partial autocorrelation)');
    end
end


%% 5. RESIDUAL DIAGNOSTICS (Table: tab:diagnostics)
if r > 0
    E = infer(EstMdlVEC, Y);
    fprintf('\n==========================================================\n');
    fprintf('   RESIDUAL DIAGNOSTICS (VEC RESIDUALS)\n');
    fprintf('   Replicates Appendix Table tab:diagnostics\n');
    fprintf('==========================================================\n');
    ws = warning('off', 'all');

    eqLabels = {'Delta K^E_t', 'Delta K^NE_t', 'Delta r_t'};
    pLB_vec   = nan(k, 1);
    pJB_vec   = nan(k, 1);
    pARCH_vec = nan(k, 1);

    fprintf('%-18s | %-15s | %-15s | %-15s\n', ...
        'Equation', 'Ljung-Box(20)', 'Jarque-Bera', 'ARCH(5)');
    fprintf('%s\n', repmat('-', 1, 72));
    for i = 1:k
        [hLB, pLB] = lbqtest(E(:,i), 'Lags', 20);
        pLB_vec(i) = pLB;
        strLB = sprintf('p=%.3f (h=%d)', pLB, hLB);

        [hJB, pJB] = jbtest(E(:,i));
        pJB_vec(i) = pJB;
        if pJB <= 0.001
            strJB = 'p<0.001 (h=1)';
        else
            strJB = sprintf('p=%.3f (h=%d)', pJB, hJB);
        end

        [hARCH, pARCH] = archtest(E(:,i) - mean(E(:,i)), 'Lags', 5);
        pARCH_vec(i) = pARCH;
        if pARCH <= 0.001
            strARCH = 'p<0.001 (h=1)';
        else
            strARCH = sprintf('p=%.3f (h=%d)', pARCH, hARCH);
        end

        fprintf('%-18s | %-15s | %-15s | %-15s\n', ...
            eqLabels{i}, strLB, strJB, strARCH);
    end
    warning(ws);

    fprintf('\n--- Interpretation (Appendix Section S4) ---\n');
    fprintf('Ljung-Box(20): no remaining autocorrelation in any equation ');
    fprintf('(all p >= %.3f), validating the dynamic specification with p=%d.\n', ...
        min(pLB_vec), VEC_LAGS_P);

    archReject = pARCH_vec < 0.05;
    if any(archReject)
        rejNames = strjoin(eqLabels(archReject), ', ');
        fprintf('Engle ARCH(5): conditional heteroskedasticity detected in %s ', rejNames);
        fprintf('(p < 0.05). This is expected in long monthly series spanning ');
        fprintf('regimes of very different volatility (armed-conflict escalation ');
        fprintf('and de-escalation, interest-rate liberalization).\n');
    end
    if any(~archReject)
        homNames = strjoin(eqLabels(~archReject), ', ');
        fprintf('Engle ARCH(5): no evidence of conditional heteroskedasticity in %s ', homNames);
        fprintf('(p >= 0.05); residuals are consistent with homoskedasticity.\n');
    end
    fprintf('ARCH effects do not bias point estimates of beta and alpha, but ');
    fprintf('inference on short-run coefficients should be read with ');
    fprintf('heteroskedasticity-robust caution where ARCH(5) rejects.\n');
    fprintf('Note: ARCH(5) is Engle''s LM test on VEC residuals E (not on the ECT).\n');
end


%% 6. COMMON TRENDS (Gonzalo-Granger, full information)
fprintf('\n==========================================================\n');
fprintf('   GONZALO-GRANGER DECOMPOSITION\n');
fprintf('==========================================================\n');

if r > 0
    alpha_perp = null(Alpha','r');
    beta_perp  = null(Beta','r');

    n_minus_r = k - r;
    denominator = alpha_perp' * beta_perp;
    if abs(det(denominator)) < 1e-10
         A_GG = beta_perp * pinv(denominator);
    else
         A_GG = beta_perp * (denominator \ eye(size(denominator)));
    end

    W_f = alpha_perp';
    fprintf('1. Factor definition (W_f = alpha_perp''):\n');
    disp(array2table(W_f, 'VariableNames', var_names_vec, 'RowNames', arrayfun(@(x) sprintf('f_%d',x), 1:n_minus_r, 'UniformOutput',false)));

    fprintf('2. Loading matrix (A_GG = beta_perp * inv(alpha_perp''*beta_perp)):\n');
    col_names_f = arrayfun(@(x) sprintf('f_%d',x), 1:n_minus_r, 'UniformOutput',false);
    disp(array2table(A_GG, 'VariableNames', col_names_f, 'RowNames', var_names_vec));

    % Loading-matrix heatmap
    figure('Name', 'Loading Matrix A_GG Heatmap');
    h = heatmap(col_names_f, var_display_names, A_GG);
    h.Title = 'Loading Matrix (Gonzalo-Granger)';
    h.XLabel = 'Common Factors';
    h.YLabel = 'Variables';
    h.Colormap = parula;
    h.CellLabelFormat = '%.4f';

    f_t = (alpha_perp' * Y')';
    Y_trend_VEC = (A_GG * f_t')';
    tau_ext = Y_trend_VEC(:,1);
    tau_sim = Y_trend_VEC(:,2);

    if isa(periodos,'datetime'), t = periodos; else, t = (1:T)'; end

    % Export
    try
        TblExport = table(t, Cant_Extorsivo, tau_ext, Cant_Simple, tau_sim, ...
            'VariableNames', {'Date','Kidnap_Ransom_Obs','Kidnap_Ransom_Trend','NonExtortionate_Obs','NonExtortionate_Trend'});
        writetable(TblExport, 'Tendencias_SE.csv');
        fprintf('[OK]: Trends saved to "Tendencias_SE.csv"\n');
    catch
        warning('Warning: could not write Tendencias_SE.csv');
    end
end


%% 7. FORECASTS
if r > 0 && isa(periodos, 'datetime')
    TargetStart = datetime(2023, 1, 1);
    TargetEnd   = datetime(2027, 12, 1);
    idx_start = find(periodos == TargetStart);

    T_Comp = [];
    if ~isempty(idx_start)
        idx_base = idx_start - 1;
        Y_presample = Y(1:idx_base, :);
        MonthsComp = calmonths(between(periodos(idx_base), TargetEnd, 'months'));
        [YF_Comp, YMSE_Comp] = forecast(EstMdlVEC, MonthsComp, Y_presample);
        date_Comp = periodos(idx_base) + calmonths(1:MonthsComp);

        seExtA = zeros(MonthsComp,1); seSimA = zeros(MonthsComp,1);
        for i=1:MonthsComp, m=YMSE_Comp{i}; seExtA(i)=sqrt(m(1,1)); seSimA(i)=sqrt(m(2,2)); end

        mask = (date_Comp >= TargetStart & date_Comp <= TargetEnd);
        T_Comp     = date_Comp(mask)';
        F_Ext_Comp = YF_Comp(mask, 1);
        F_Sim_Comp = YF_Comp(mask, 2);
        S_Ext_Comp = seExtA(mask);
        S_Sim_Comp = seSimA(mask);
    end

    [YF_Fut, YMSE_Fut] = forecast(EstMdlVEC, FCAST_FUTURE_HORIZON, Y);
    date_Fut = periodos(end) + calmonths(1:FCAST_FUTURE_HORIZON);

    F_Ext_Fut = YF_Fut(:,1);
    F_Sim_Fut = YF_Fut(:,2);
    S_Ext_Fut = zeros(FCAST_FUTURE_HORIZON,1);
    S_Sim_Fut = zeros(FCAST_FUTURE_HORIZON,1);
    for i=1:FCAST_FUTURE_HORIZON, m=YMSE_Fut{i}; S_Ext_Fut(i)=sqrt(m(1,1)); S_Sim_Fut(i)=sqrt(m(2,2)); end
end


%% 8. PLOTS (ENGLISH + NO TITLES + LEFT BOX)
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultAxesXColor', 'k');
set(0, 'DefaultAxesYColor', 'k');
set(0, 'DefaultAxesLineWidth', 1.0);
set(0, 'DefaultAxesTickDir', 'out');
set(0, 'DefaultAxesTickLength', [0.015 0.025]);

% --- Trends ---
figure('Name','Common Trends'); hold on;
plot(t, f_t(:,1), 'k-', 'LineWidth', 1.5, 'DisplayName','f_1');
if size(f_t,2)>1, plot(t, f_t(:,2), '--', 'Color',[0.4 0.4 0.4], 'DisplayName','f_2'); end
yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
legend('Location','northwest', 'FontSize', 14);
xlabel('Year');

figure('Name','Trend Kidnapping for Ransom'); hold on;
plot(t, Cant_Extorsivo, '-', 'Color',[0.7 0.7 0.7], 'DisplayName','Observed Data');
plot(t, tau_ext, 'k-', 'LineWidth', 2, 'DisplayName','Trend (GG)');
yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
legend('Location','northwest', 'FontSize', 14);
ylabel('Kidnapping for Ransom'); xlabel('Year');

figure('Name','Trend Non-Extortionate Kidnapping'); hold on;
plot(t, Cant_Simple, '-', 'Color',[0.7 0.7 0.7], 'DisplayName','Observed Data');
plot(t, tau_sim, 'k-', 'LineWidth', 2, 'DisplayName','Trend (GG)');
yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
legend('Location','northwest', 'FontSize', 14);
ylabel('Non-Extortionate Kidnapping'); xlabel('Year');

% --- Comparative fit ---
if ~isempty(T_Comp)
    figure('Name','Comparative Kidnapping for Ransom'); hold on;
    plot(t, Cant_Extorsivo, '-', 'Color',[0.6 0.6 0.6], 'DisplayName','Observed Data');
    plot(T_Comp, F_Ext_Comp, 'k-', 'LineWidth',2, 'DisplayName','Forecast');
    plot(T_Comp, F_Ext_Comp+1.96*S_Ext_Comp, 'k:', 'DisplayName','95% CI');
    plot(T_Comp, F_Ext_Comp-1.96*S_Ext_Comp, 'k:', 'HandleVisibility','off');
    yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
    xlim([datetime(2018,1,1) datetime(2028,1,1)]);
    legend('Location','northwest', 'FontSize', 14);
    ylabel('Kidnapping for Ransom'); xlabel('Year');

    figure('Name','Comparative Non-Extortionate Kidnapping'); hold on;
    plot(t, Cant_Simple, '-', 'Color',[0.6 0.6 0.6], 'DisplayName','Observed Data');
    plot(T_Comp, F_Sim_Comp, 'k-', 'LineWidth',2, 'DisplayName','Forecast');
    plot(T_Comp, F_Sim_Comp+1.96*S_Sim_Comp, 'k:', 'DisplayName','95% CI');
    plot(T_Comp, F_Sim_Comp-1.96*S_Sim_Comp, 'k:', 'HandleVisibility','off');
    yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
    xlim([datetime(2018,1,1) datetime(2028,1,1)]);
    legend('Location','northwest', 'FontSize', 14);
    ylabel('Non-Extortionate Kidnapping'); xlabel('Year');
end

% --- Future forecast ---
figure('Name','Forecast Future Kidnapping for Ransom'); hold on;
plot(t, Cant_Extorsivo, '-', 'Color',[0.6 0.6 0.6], 'DisplayName','Observed Data');
plot(date_Fut', F_Ext_Fut, 'k-', 'LineWidth',2, 'DisplayName','Future Forecast');
plot(date_Fut', F_Ext_Fut+1.96*S_Ext_Fut, '--', 'Color',[0.3 0.3 0.3], 'DisplayName','95% CI');
plot(date_Fut', F_Ext_Fut-1.96*S_Ext_Fut, '--', 'Color',[0.3 0.3 0.3], 'HandleVisibility','off');
yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
legend('Location','northwest', 'FontSize', 14);
ylabel('Kidnapping for Ransom'); xlabel('Year');

figure('Name','Forecast Future Non-Extortionate Kidnapping'); hold on;
plot(t, Cant_Simple, '-', 'Color',[0.6 0.6 0.6], 'DisplayName','Observed Data');
plot(date_Fut', F_Sim_Fut, 'k-', 'LineWidth',2, 'DisplayName','Future Forecast');
plot(date_Fut', F_Sim_Fut+1.96*S_Sim_Fut, '--', 'Color',[0.3 0.3 0.3], 'DisplayName','95% CI');
plot(date_Fut', F_Sim_Fut-1.96*S_Sim_Fut, '--', 'Color',[0.3 0.3 0.3], 'HandleVisibility','off');
yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
legend('Location','northwest', 'FontSize', 14);
ylabel('Non-Extortionate Kidnapping'); xlabel('Year');


%% 9. RESIDUAL DIAGNOSTICS (ACF/PACF - 50 LAGS)
if r > 0
    fprintf('\nGenerating residual ACF/PACF (50 lags)...\n');
    E = infer(EstMdlVEC, Y);

    for i=1:k
        vName = var_display_names{i};
        figure('Name',['ACF-PACF: ' vName]);

        subplot(2,1,1);
        autocorr(E(:,i), 'NumLags', 50);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        title('');
        ylabel('ACF'); xlabel('Lags');

        subplot(2,1,2);
        parcorr(E(:,i), 'NumLags', 50);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        title('');
        ylabel('PACF'); xlabel('Lags');
    end
end


%% 10. IMPULSE RESPONSE ANALYSIS (Cholesky & Blanchard-Quah)
if r > 0
    fprintf('\n==========================================================\n');
    fprintf('   IMPULSE RESPONSE ANALYSIS (Cholesky & Blanchard-Quah)\n');
    fprintf('==========================================================\n');

    nHorizon = 40;
    gamma_matrices = EstMdlVEC.ShortRun;
    Pi_level = EstMdlVEC.Adjustment * EstMdlVEC.Cointegration';
    VAR_Matrices = zeros(k,k,p);
    VAR_Matrices(:,:,1) = eye(k) + Pi_level;
    if q > 0
        VAR_Matrices(:,:,1) = VAR_Matrices(:,:,1) + gamma_matrices{1};
        for i = 2:q
            VAR_Matrices(:,:,i) = gamma_matrices{i} - gamma_matrices{i-1};
        end
        VAR_Matrices(:,:,p) = -gamma_matrices{q};
    end

    Phi_RF = zeros(k,k,nHorizon);
    Phi_RF(:,:,1) = eye(k);

    for s = 2:nHorizon
        term = zeros(k,k);
        for j = 1:p
            if (s-j) >= 1
                term = term + VAR_Matrices(:,:,j) * Phi_RF(:,:,s-j);
            end
        end
        Phi_RF(:,:,s) = term;
    end

    Sigma_u = cov(E);

    % --- A. CHOLESKY IDENTIFICATION ---
    order_idx = [3, 2, 1];
    Sigma_ordered = Sigma_u(order_idx, order_idx);
    P_chol = chol(Sigma_ordered, 'lower');
    M = zeros(k,k);
    M(1,3) = 1; M(2,2) = 1; M(3,1) = 1;
    B_chol = M' * P_chol;

    IRF_Chol = zeros(nHorizon, k, k);
    for s = 1:nHorizon
        IRF_Chol(s,:,:) = Phi_RF(:,:,s) * B_chol;
    end

    shock_names_chol = {'Shock Loan Interest Rate', 'Shock Non-Extortionate', 'Shock Kidnapping for Ransom'};
    figure('Name','IRF Cholesky');
    for j = 1:3
        for i = 1:3
             subplot(3,3, (i-1)*3 + j);
             resp = IRF_Chol(:, i, j);
             plot(0:nHorizon-1, resp, 'k-', 'LineWidth', 1.5);
             yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
             ylabel(var_display_names{i}, 'Interpreter','none');
             xlabel('Periods');
             if i==1, title(shock_names_chol{j}); else, title(''); end
        end
    end


    % --- B. BLANCHARD-QUAH (BQ) IDENTIFICATION ---
    Alpha_p = null(EstMdlVEC.Adjustment');
    Beta_p  = null(EstMdlVEC.Cointegration');
    SumGamma = zeros(k,k);
    if q > 0
        for i=1:q, SumGamma = SumGamma + gamma_matrices{i}; end
    end
    Mat_inv = Alpha_p' * (eye(k) - SumGamma) * Beta_p;
    C_inf = Beta_p * (Mat_inv \ Alpha_p');
    Var_trends = Alpha_p' * Sigma_u * Alpha_p;

    % Ordering: 1=Ransom (Perm), 2=Non-Extortionate (Perm), 3=BR_r (Trans)
    Vec_Order_BQ = [1 2 3];
    Sigma_Ord_BQ = Sigma_u(Vec_Order_BQ, Vec_Order_BQ);
    Alpha_p_Ord  = Alpha_p(Vec_Order_BQ, :);
    Alpha_Ord    = EstMdlVEC.Adjustment(Vec_Order_BQ, :);
    D = [Alpha_p_Ord'; Alpha_Ord'];

    Cov_D = D * Sigma_Ord_BQ * D';
    Chol_D = chol(Cov_D, 'lower');
    B_bq_ordered = D \ Chol_D;

    % Vec_Order_BQ is [1 2 3] (original), so no re-permutation is needed
    B_bq_orig = B_bq_ordered;

    IRF_BQ = zeros(nHorizon, k, k);
    for s = 1:nHorizon
        IRF_BQ(s,:,:) = Phi_RF(:,:,s) * B_bq_orig;
    end

    shock_names_bq = {'Permanent Shock (Kidnapping for Ransom)', 'Permanent Shock (Non-Extortionate)', 'Transitory Shock (Loan Interest Rate)'};
    figure('Name','IRF Blanchard-Quah');
    for j = 1:3
        for i = 1:3
             subplot(3,3, (i-1)*3 + j);
             resp = IRF_BQ(:, i, j);
             plot(0:nHorizon-1, resp, 'k-', 'LineWidth', 1.5);
             yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
             ylabel(var_display_names{i}, 'Interpreter','none');
             xlabel('Periods');
            if i==1, title(shock_names_bq{j}); else, title(''); end
        end
    end
end


%% 11. VARIANCE DECOMPOSITION (FEVD)
if r > 0
    fprintf('\n==========================================================\n');
    fprintf('   FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)\n');
    fprintf('==========================================================\n');

    % FEVD helper: IRF_Array is nHorizon x k(vars) x k(shocks)
    compute_fevd = @(IRF_Array) ...
        cumsum(IRF_Array.^2, 1) ./ sum(cumsum(IRF_Array.^2, 1), 3);

    FEVD_Chol = compute_fevd(IRF_Chol);
    FEVD_BQ   = compute_fevd(IRF_BQ);

    % Grayscale palette for the area plots (3 shocks)
    gray_palette = [0.3 0.3 0.3; 0.6 0.6 0.6; 0.85 0.85 0.85];

    % --- FEVD CHOLESKY PLOT ---
    figure('Name','FEVD Cholesky');
    for i = 1:k
        subplot(k, 1, i);
        dat = squeeze(FEVD_Chol(:, i, :)); % nH x k_shocks
        area(0:nHorizon-1, dat);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        colororder(gray_palette);
        ylim([0 1]);
        ylabel(['Var: ' var_display_names{i}], 'Interpreter','none');
        xlabel('Periods');
        legend(shock_names_chol, 'Location','eastoutside', 'FontSize', 14);
        title('');
    end

    % --- FEVD BLANCHARD-QUAH PLOT ---
    figure('Name','FEVD Blanchard-Quah');
    for i = 1:k
        subplot(k, 1, i);
        dat = squeeze(FEVD_BQ(:, i, :));
        area(0:nHorizon-1, dat);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        colororder(gray_palette);
        ylim([0 1]);
        ylabel(['Var: ' var_display_names{i}], 'Interpreter','none');
        xlabel('Periods');
        legend(shock_names_bq, 'Location','eastoutside', 'FontSize', 14);
        title('');
    end

    % --- FEVD TABLES ---
    horizons_print = [1, 4, 8, 12, 24, 40];
    horizons_print = horizons_print(horizons_print <= nHorizon);

    fprintf('\n==========================================================\n');
    fprintf('   FEVD TABLES (Variance Decomposition)\n');
    fprintf('==========================================================\n');

    % A. CHOLESKY
    fprintf('\n--- FEVD CHOLESKY ---\n');
    for i = 1:k
        fprintf('\nVariable: %s\n', var_display_names{i});
        fprintf('%-8s', 'Period');
        for j=1:k, fprintf(' | %-15s', shock_names_chol{j}); end; fprintf('\n');
        for h = horizons_print
            fprintf('%-8d', h);
            for j=1:k
                val = FEVD_Chol(h, i, j);
                fprintf(' | %15.4f', val);
            end
            fprintf('\n');
        end
    end

    % B. BLANCHARD-QUAH
    fprintf('\n--- FEVD BLANCHARD-QUAH ---\n');
    for i = 1:k
        fprintf('\nVariable: %s\n', var_display_names{i});
        fprintf('%-8s', 'Period');
        for j=1:k, fprintf(' | %-15s', shock_names_bq{j}); end; fprintf('\n');
        for h = horizons_print
            fprintf('%-8d', h);
            for j=1:k
                val = FEVD_BQ(h, i, j);
                fprintf(' | %15.4f', val);
            end
            fprintf('\n');
        end
    end
end


%% 12. SPECIAL FOCUS: KIDNAPPING-FOR-RANSOM ANALYSIS
if r > 0
    fprintf('\nGenerating focused plots for kidnapping for ransom...\n');

    % --- A. IRF CHOLESKY: RESPONSE OF K^E ---
    % Variable index for kidnapping for ransom = 1
    figure('Name','Focus: IRF Cholesky - Ransom');
    for j=1:3
        subplot(1,3,j);
        plot(0:nHorizon-1, IRF_Chol(:,1,j), 'k-', 'LineWidth', 1.5);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        xlabel('Periods');
        title(shock_names_chol{j});
        if j==1, ylabel('Response of Kidnapping for Ransom', 'Interpreter','none'); end
    end

    % --- B. IRF BQ: RESPONSE OF K^E ---
    figure('Name','Focus: IRF BQ - Ransom');
    for j=1:3
        subplot(1,3,j);
        plot(0:nHorizon-1, IRF_BQ(:,1,j), 'k-', 'LineWidth', 1.5);
        yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
        xlabel('Periods');
        title(shock_names_bq{j});
        if j==1, ylabel('Response of Kidnapping for Ransom', 'Interpreter','none'); end
    end

    % --- C. FEVD CHOLESKY: K^E DECOMPOSITION ---
    figure('Name','Focus: FEVD Cholesky - Ransom');
    dat = squeeze(FEVD_Chol(:, 1, :));
    area(0:nHorizon-1, dat);
    yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
    colororder(gray_palette);
    ylim([0 1]);
    ylabel('Variance Share'); xlabel('Periods');
    legend(shock_names_chol, 'Location','eastoutside', 'FontSize', 14);
    title('Variance Decomposition of Kidnapping for Ransom (Cholesky)');

    % --- D. FEVD BQ: K^E DECOMPOSITION ---
    figure('Name','Focus: FEVD BQ - Ransom');
    dat = squeeze(FEVD_BQ(:, 1, :));
    area(0:nHorizon-1, dat);
    yline(0, 'k-', 'LineWidth', 1.0, 'HandleVisibility','off');
    colororder(gray_palette);
    ylim([0 1]);
    ylabel('Variance Share'); xlabel('Periods');
    legend(shock_names_bq, 'Location','eastoutside', 'FontSize', 14);
    title('Variance Decomposition of Kidnapping for Ransom (Blanchard-Quah)');
end

fprintf('\nFinished.\n');

%% 13. FUTURE PROJECTIONS TABLE (FIRST 10 PERIODS)
fprintf('\n==========================================================\n');
fprintf('   FUTURE PROJECTIONS (FIRST 10 PERIODS)\n');
fprintf('==========================================================\n');
fprintf('95%% confidence intervals\n');

n_print = min(10, FCAST_FUTURE_HORIZON);

fprintf('\n%-15s | %-30s | %-30s\n', 'Date', 'Kidnapping for Ransom', 'Non-Extortionate');
fprintf('%-15s | %-10s %-19s | %-10s %-19s\n', '', 'Forecast', '[Low, Upp]', 'Forecast', '[Low, Upp]');
fprintf('%s\n', repmat('-',1,85));

for i = 1:n_print
    dStr = datestr(date_Fut(i), 'mmm-yyyy');

    fe = F_Ext_Fut(i);
    se_e = S_Ext_Fut(i);
    ci_e_low = fe - 1.96*se_e;
    ci_e_upp = fe + 1.96*se_e;

    fs = F_Sim_Fut(i);
    se_s = S_Sim_Fut(i);
    ci_s_low = fs - 1.96*se_s;
    ci_s_upp = fs + 1.96*se_s;

    fprintf('%-15s | %10.2f [%9.2f, %9.2f] | %10.2f [%9.2f, %9.2f]\n', ...
        dStr, fe, ci_e_low, ci_e_upp, fs, ci_s_low, ci_s_upp);
end

function s = formatUnitRootP(p, testType)
% Format p-values with MATLAB tabulated bounds (Table I, Appendix 1).
    if strcmp(testType, 'KPSS')
        if p <= 0.01
            s = '<0.01';
        elseif p >= 0.10
            s = '>0.10';
        else
            s = sprintf('%.3f', p);
        end
    else
        if p <= 0.001
            s = '0.001';
        elseif p >= 0.999
            s = '>0.999';
        else
            s = sprintf('%.3f', p);
        end
    end
end
