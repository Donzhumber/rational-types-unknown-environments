%% ============================================================
%  LOCAL LINEAR TREND MODEL -- REPLICATION SCRIPT FOR APPENDIX 1
%  Specification: stochastic level, AR(8) slope, AR(5) cycle,
%  irregular term. Maximum likelihood via the Kalman filter.
%  Target series: kidnapping for ransom (Cantidad_Extorsivo).
% ============================================================

clc; clear; close all;

%% 1. MODEL CONFIGURATION
% ------------------------------------------------------------
P_SLOPE = 8;
P_CYCLE = 5;
Q_CYCLE = 0;

%% 2. TIME CONFIGURATION
% ------------------------------------------------------------
MANUAL_START_YEAR  = 1954;
MANUAL_START_MONTH = 7;

fprintf('=== CONFIGURATION ===\n');
fprintf(' > Model: Slope AR(%d) | Cycle ARMA(%d,%d)\n', P_SLOPE, P_CYCLE, Q_CYCLE);

%% 3. DATA LOADING
% ------------------------------------------------------------
try
    data = readtable("Data.csv");
catch
    error('Could not read "Data.csv".');
end

% A. Target variable
target_vars = {'Cantidad_Extorsivo', 'Cantidad_Extorsivo_Cantidad_Simple', 'Extorsion'};
found = false;
for i = 1:length(target_vars)
    if ismember(target_vars{i}, data.Properties.VariableNames)
        y_raw = data.(target_vars{i}); found = true; break;
    end
end
if ~found, error('Target variable not found'); end

y = double(y_raw(:));
if any(isnan(y)), y = fillmissing(y,'linear'); end
T = length(y);
sigma_y = std(y);

% B. Time variable (t)
time_vars = {'mes', 'Mes', 'month', 'fecha', 'date', 'Date'};
found_t = false; t = [];
for i = 1:length(time_vars)
    if ismember(time_vars{i}, data.Properties.VariableNames)
        raw_t = data.(time_vars{i});
        if isdatetime(raw_t)
            t = raw_t;
        elseif iscell(raw_t) || isstring(raw_t)
            try t = datetime(string(raw_t)); catch, end
        elseif isnumeric(raw_t) && max(raw_t) > 30000 && max(raw_t) < 60000
            t = datetime(raw_t, 'ConvertFrom', 'excel');
        end
        if ~isempty(t), found_t = true; break; end
    end
end

if ~found_t || isempty(t) || length(t) ~= T
    start_dt = datetime(MANUAL_START_YEAR, MANUAL_START_MONTH, 1);
    t = start_dt + calmonths(0:T-1);
end
t = t(:);

%% 4. ESTIMATION
% ------------------------------------------------------------
fprintf('\n--- Estimating model... ---\n');

val_init_ar = 0.01;
val_init_ma = 0.01;

p0 = [log(0.1*sigma_y);         % Sigma Eta (level)
      log(0.01*sigma_y);        % Sigma Zeta (slope)
      0.1*ones(P_SLOPE,1);      % Slope AR
      log(0.1*sigma_y);         % Sigma Kappa (cycle)
      val_init_ar*ones(P_CYCLE,1); % Cycle AR
      val_init_ma*ones(Q_CYCLE,1); % Cycle MA
      log(0.1*sigma_y)];        % Sigma Eps (irregular)

ModelFcn = @(params) local_Generic_SSM(params, P_SLOPE, P_CYCLE, Q_CYCLE);
opts = optimoptions('fminunc','Display','iter','MaxIterations',2000, 'Algorithm','quasi-newton', 'StepTolerance',1e-6);

try
    Mdl = ssm(ModelFcn);
    [EstMdl, estParams, EstParamCov, logL] = estimate(Mdl, y, p0, 'Options',opts);
catch ME
    error('Estimation failed: %s', ME.message);
end

k_params = length(estParams);
aic = 2*k_params - 2*logL;
bic = k_params*log(T) - 2*logL;
fprintf('\nGlobal fit: LogL=%.2f, AIC=%.2f, BIC=%.2f\n', logL, aic, bic);

%% 5. COEFFICIENT TABLE AND P-VALUES
% ------------------------------------------------------------
fprintf('\n=== PARAMETER ESTIMATION TABLE ===\n');

std_errors = sqrt(diag(EstParamCov));
t_stats    = estParams ./ std_errors;
p_values   = 2 * (1 - tcdf(abs(t_stats), T - k_params));

paramNames = {};
paramNames{end+1,1} = 'Log_Sigma_Level(Eta)';
paramNames{end+1,1} = 'Log_Sigma_Slope(Zeta)';
for i = 1:P_SLOPE, paramNames{end+1,1} = sprintf('Slope_AR_L%d', i); end
paramNames{end+1,1} = 'Log_Sigma_Cycle(Kappa)';
for i = 1:P_CYCLE, paramNames{end+1,1} = sprintf('Cycle_AR_L%d', i); end
for i = 1:Q_CYCLE, paramNames{end+1,1} = sprintf('Cycle_MA_L%d', i); end
paramNames{end+1,1} = 'Log_Sigma_Irreg(Eps)';

EstTable = table(estParams, std_errors, t_stats, p_values, ...
    'RowNames', paramNames, ...
    'VariableNames', {'Coefficient','StdError','tStat','pValue'});

disp(EstTable);

significant = EstTable.pValue < 0.05;
if any(significant)
    fprintf('Parameters significant at the 5%% level:\n');
    disp(EstTable.Properties.RowNames(significant));
else
    fprintf('No parameter is significant at the 5%% level.\n');
end

fprintf('\nInnovation standard deviations (levels):\n');
fprintf('  sigma_eta (level)    = %.4f\n', exp(estParams(1)));
fprintf('  sigma_zeta (slope)   = %.4f\n', exp(estParams(2)));
fprintf('  sigma_kappa (cycle)  = %.4f\n', exp(estParams(2+P_SLOPE+1)));
fprintf('  sigma_eps (irregular)= %.4f\n', exp(estParams(end)));

%% 6. COMPONENTS AND PLOTS
% ------------------------------------------------------------
[xSm, varSm] = smooth(EstMdl, y);
level_sm = xSm(:, 1);
slope_sm = xSm(:, 2);
idx_cycle_start = 2 + max(1, P_SLOPE);
cycle_sm = xSm(:, idx_cycle_start);
res_obs  = y - (level_sm + cycle_sm);

T_Out = table(t, y, level_sm, slope_sm, cycle_sm, res_obs, ...
    'VariableNames', {'Date','Observed','Trend','Slope','Cycle','Residual'});
filename = 'LLT_decomposition.csv';
writetable(T_Out, filename);
fprintf('\nSaved: %s\n', filename);

figure('Name','Smoothed decomposition');
subplot(4,1,1); plot(t,y,'Color',[.7 .7 .7]); hold on; plot(t,level_sm,'r','LineWidth',1.5);
title('Level (Trend)'); legend('Data','Trend'); axis tight; grid on;
subplot(4,1,2); plot(t,slope_sm,'m'); title('Slope'); grid on;
subplot(4,1,3); plot(t,cycle_sm,'g'); title('Cycle'); grid on;
subplot(4,1,4); plot(t,res_obs,'k'); title('Residuals'); grid on;

function [A,B,C,D,Mean0,Cov0,StateType] = local_Generic_SSM(params, p_slope, p_cyc, q_cyc)
    curr=1;
    sigma_eta=exp(params(curr)); curr=curr+1;
    sigma_zeta=exp(params(curr)); curr=curr+1;
    slope_ar=[]; if p_slope>0, slope_ar=params(curr:curr+p_slope-1); curr=curr+p_slope; end
    sigma_kappa=exp(params(curr)); curr=curr+1;
    cycle_ar=[]; if p_cyc>0, cycle_ar=params(curr:curr+p_cyc-1); curr=curr+p_cyc; end
    cycle_ma=[]; if q_cyc>0, cycle_ma=params(curr:curr+q_cyc-1); curr=curr+q_cyc; end
    sigma_eps=exp(params(curr));

    dim_slp=max(1,p_slope); dim_cyc=max(p_cyc,q_cyc+1);
    num_st=1+dim_slp+dim_cyc;
    A=zeros(num_st); B=zeros(num_st,3); C=zeros(1,num_st);

    ilev=1; islp=2; icyc=2+dim_slp;
    A(ilev,ilev)=1; A(ilev,islp)=1; B(ilev,1)=sigma_eta; C(1)=1;
    if p_slope>0, for k=1:p_slope, A(islp,islp+k-1)=slope_ar(k); end; for k=2:dim_slp, A(islp+k-1,islp+k-2)=1; end; end
    B(islp,2)=sigma_zeta;
    if dim_cyc>0
        for k=1:p_cyc, A(icyc+k-1,icyc)=cycle_ar(k); end
        for k=1:dim_cyc-1, A(icyc+k-1,icyc+k)=1; end
        B(icyc,3)=sigma_kappa;
        for k=1:q_cyc, B(icyc+k,3)=cycle_ma(k)*sigma_kappa; end
        C(icyc)=1;
    end
    D=sigma_eps; Mean0=zeros(num_st,1); Cov0=eye(num_st)*10;
    StateType=[2; 2*ones(dim_slp,1); 0*ones(dim_cyc,1)];
end
