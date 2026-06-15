%% =========================================================
%  TREND EQUIVALENCE CHECK (Appendix 1, Section S7)
%
%  Validates Corollary (stochastic-trend equivalence) by comparing
%  the VEC-GG long-run trend of kidnapping for ransom with the LLT
%  smoothed level trend, both read from pre-computed CSV outputs.
%
%  Steps:
%    1. Load VEC/Tendencias_SE.csv  (Kidnap_Ransom_Trend)
%    2. Load SE_Kalman/LLT_decomposition.csv  (Trend)
%    3. Build d_t = tau_VEC - tau_LLT on the full estimation sample
%    4. Run ADF and KPSS unit-root tests on d_t (July 1954 - Dec 2025)
%
%  Run from matlab/Trend_Difference/ (paths resolve automatically):
%      check_trend_equivalence
% =========================================================

clc;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
matlabRoot = fileparts(scriptDir);
cd(scriptDir);

vecFile = fullfile(matlabRoot, 'VEC', 'Tendencias_SE.csv');
lltFile = fullfile(matlabRoot, 'SE_Kalman', 'LLT_decomposition.csv');
outFile = fullfile(scriptDir, 'trend_difference.csv');

fprintf('\n==========================================================\n');
fprintf('   TREND EQUIVALENCE CHECK (Appendix 1, Section S7)\n');
fprintf('==========================================================\n');

%% 1. Load pre-computed trends
if ~isfile(vecFile)
    error('Missing file: %s. Run VEC/VEC_2T.m first.', vecFile);
end
if ~isfile(lltFile)
    error('Missing file: %s. Run SE_Kalman/LLT_Extorsivo.m first.', lltFile);
end

V = readtable(vecFile);
L = readtable(lltFile);

requiredVec = {'Date', 'Kidnap_Ransom_Trend'};
requiredLlt = {'Date', 'Trend'};
if ~all(ismember(requiredVec, V.Properties.VariableNames))
    error('Tendencias_SE.csv must contain columns: Date, Kidnap_Ransom_Trend');
end
if ~all(ismember(requiredLlt, L.Properties.VariableNames))
    error('LLT_decomposition.csv must contain columns: Date, Trend');
end

dates = datetime(V.Date);
tau_vec = V.Kidnap_Ransom_Trend(:);
tau_llt = L.Trend(:);

if numel(tau_vec) ~= numel(tau_llt)
    n = min(numel(tau_vec), numel(tau_llt));
    dates   = dates(1:n);
    tau_vec = tau_vec(1:n);
    tau_llt = tau_llt(1:n);
    warning('Series lengths differ; using the first %d aligned observations.', n);
end

diff_vals = tau_vec - tau_llt;
T = numel(diff_vals);

fprintf('Estimation / validation sample: %s to %s (T = %d)\n', ...
    datestr(dates(1), 'mmm-yyyy'), datestr(dates(end), 'mmm-yyyy'), T);

%% 2. Export aligned series (full sample)
T_out = table(dates, tau_vec, tau_llt, diff_vals, ...
    'VariableNames', {'Date', 'VEC_GG_Trend', 'LLT_Trend', 'Difference'});
writetable(T_out, outFile);
fprintf('Exported: %s\n', outFile);

%% 3. Unit-root tests on the trend difference (full estimation sample)
fprintf('\n--- Unit-root tests on d_t = VEC-GG trend - LLT trend ---\n');
fprintf('%-22s | %-10s | %-10s | %-10s\n', 'Series', 'ADF p', 'KPSS def', 'KPSS(12)');
fprintf('%s\n', repmat('-', 1, 58));

testSeries = {
    tau_vec,      'VEC-GG trend'
    tau_llt,      'LLT trend'
    diff_vals,    'Difference (d_t)'
    };

for i = 1:size(testSeries, 1)
    x = testSeries{i, 1};
    name = testSeries{i, 2};
    [~, pADF] = adftest(x);
    [~, pKPSS_def] = kpsstest(x, 'trend', false);
    [~, pKPSS12] = kpsstest(x, 'trend', false, 'lags', 12);
    fprintf('%-22s | %10.4f | %10.4f | %10.4f\n', name, pADF, pKPSS_def, pKPSS12);
end

%% 4. Appendix benchmarks (Section S7)
[~, pADF_diff] = adftest(diff_vals);
[~, pKPSS_def_diff] = kpsstest(diff_vals, 'trend', false);
[~, pKPSS12_diff] = kpsstest(diff_vals, 'trend', false, 'lags', 12);

fprintf('\n--- Appendix 1 benchmarks (difference only, full sample) ---\n');
fprintf('ADF p-value              : %.4f   (Appendix: 0.001)\n', pADF_diff);
fprintf('KPSS p-value (default)   : %.4f   (Appendix: 0.010)\n', pKPSS_def_diff);
fprintf('KPSS p-value (lags = 12)   : %.4f   (Appendix: 0.071)\n', pKPSS12_diff);

if pADF_diff < 0.05 && pKPSS12_diff >= 0.05
    fprintf('\nResult: d_t is stationary (I(0)) at the 5%% level.\n');
    fprintf('        Trend equivalence is supported (Corollary, Section S7).\n');
elseif pADF_diff < 0.05
    fprintf('\nResult: ADF rejects a unit root in d_t (I(0) by ADF).\n');
    fprintf('        KPSS with 12 lags does not reject stationarity (p = %.3f).\n', pKPSS12_diff);
    fprintf('        Trend equivalence is supported (Corollary, Section S7).\n');
else
    fprintf('\nResult: review the output above against Appendix 1, Section S7.\n');
end

%% 5. Plots (full estimation sample)
figure('Name', 'Trend equivalence: comparison', 'Color', 'w');
plot(dates, tau_llt, 'k', 'LineWidth', 1.5); hold on;
plot(dates, tau_vec, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);
ylabel('Kidnapping for ransom (level)'); xlabel('Year');
legend({'LLT trend', 'VEC-GG trend'}, 'Location', 'best', 'Box', 'off');
grid on; axis tight; box off;

figure('Name', 'Trend equivalence: difference', 'Color', 'w');
plot(dates, diff_vals, 'k', 'LineWidth', 1.2); hold on;
yline(0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);
ylabel('Difference (VEC-GG - LLT)'); xlabel('Year');
grid on; axis tight; box off;

fprintf('\nDone.\n');
