%% Diagnostic: compare replication outputs with Appendix_1.tex
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
matlabRoot = fileparts(scriptDir);
cd(scriptDir);

pathOf = @(parts) fullfile(matlabRoot, parts{:});

fprintf('\n========== APPENDIX 1 REPLICATION DIAGNOSTIC ==========\n');

%% 1. Check required files
files = {
    pathOf({'VEC','Data.csv'})
    pathOf({'VEC','VEC_2T.m'})
    pathOf({'VEC','Tendencias_SE.csv'})
    pathOf({'SE_Kalman','Data.csv'})
    pathOf({'SE_Kalman','LLT_Extorsivo.m'})
    pathOf({'SE_Kalman','LLT_decomposition.csv'})
    pathOf({'Trend_Difference','check_trend_equivalence.m'})
    };
fprintf('\n--- File inventory ---\n');
for i = 1:numel(files)
    if isfile(files{i})
        fprintf('[OK]   %s\n', files{i});
    else
        fprintf('[MISS] %s\n', files{i});
    end
end

%% 2. Unit-root tests (Table tab:unitroot in Appendix_1.tex)
opts = detectImportOptions(pathOf({'VEC','Data.csv'}));
opts.VariableNamingRule = 'preserve';
data = readtable(pathOf({'VEC','Data.csv'}), opts);
Y = [data.Cantidad_Extorsivo(:), data.Cantidad_Simple(:), data.BR_r(:)];
if any(any(isnan(Y))), Y = fillmissing(Y, 'movmedian', 3); end
vn = {'K_E','K_NE','r'};
lags = 12;

fprintf('\n--- Unit-root tests (ADF ARD + KPSS no trend, lags=%d) ---\n', lags);
fprintf('%-5s | %-12s | %-12s | %-12s | %-12s\n', 'Var', 'ADF lev p', 'KPSS lev p', 'ADF diff p', 'KPSS diff p');
for i = 1:3
    [~,pAl] = adftest(Y(:,i), 'model','ARD', 'lags', lags);
    [~,pKl] = kpsstest(Y(:,i), 'trend', false, 'lags', lags);
    d = diff(Y(:,i));
    [~,pAd] = adftest(d, 'model','ARD', 'lags', lags);
    [~,pKd] = kpsstest(d, 'trend', false, 'lags', lags);
    fprintf('%-5s | %12.3f | %12.3f | %12.3f | %12.3f\n', vn{i}, pAl, pKl, pAd, pKd);
end
fprintf('Appendix reports: levels ADF>0.05, KPSS<0.01; diff ADF=0.001, KPSS>0.10\n');

%% 3. VEC core estimates (quick re-estimation)
p = 36; r = 1; q = p-1; k = 3;
[EstMdl, EstSE, logL] = estimate(vecm(k,r,q), Y, 'Model','H1', 'Display','off');
Beta = EstMdl.Cointegration;
Alpha = EstMdl.Adjustment;
nParams = k^2*q + k*r + r*(k-r) + k;
[aic, bic] = aicbic(logL, nParams, size(Y,1));

fprintf('\n--- VEC(p=%d,r=%d) core estimates ---\n', p, r);
fprintf('logL=%.2f  AIC=%.2f  BIC=%.2f\n', logL, aic, bic);
fprintf('Beta  = [%.4f, %.4f, %.4f]\n', Beta(1), Beta(2), Beta(3));
fprintf('Alpha = [%.4f, %.4f, %.4f]\n', Alpha(1), Alpha(2), Alpha(3));
tA = Alpha ./ EstSE.Adjustment;
pA = 2*(1-tcdf(abs(tA), size(Y,1)-nParams));
fprintf('Alpha p-values = [%.4f, %.4f, %.4f]\n', pA(1), pA(2), pA(3));
fprintf('Speed |alpha*beta| = %.4f\n', abs(Alpha(1)*Beta(1)));

[hJ,pJ,sJ,cJ] = jcitest(Y,'model','H1','lags',p,'Display','off');
if istable(sJ), sJ = table2array(sJ(:,varfun(@isnumeric,sJ,'OutputFormat','uniform'))); end
if istable(pJ), pJ = table2array(pJ(:,varfun(@isnumeric,pJ,'OutputFormat','uniform'))); end
if istable(cJ), cJ = table2array(cJ(:,varfun(@isnumeric,cJ,'OutputFormat','uniform'))); end
fprintf('Johansen trace: [%.3f, %.3f, %.3f]  p=[%.3f, %.3f, %.3f]\n', sJ(1),sJ(2),sJ(3), pJ(1),pJ(2),pJ(3));

alpha_perp = null(Alpha','r');
beta_perp  = null(Beta','r');
A_GG = beta_perp * ((alpha_perp'*beta_perp) \ eye(k-r));
fprintf('GG W_f rows:\n'); disp(alpha_perp');
fprintf('GG A_GG:\n'); disp(A_GG);

E = infer(EstMdl, Y);
[~,pLB1] = lbqtest(E(:,1),'Lags',20);
[~,pLB2] = lbqtest(E(:,2),'Lags',20);
[~,pLB3] = lbqtest(E(:,3),'Lags',20);
fprintf('Residual Ljung-Box(20) p: [%.3f, %.3f, %.3f]\n', pLB1, pLB2, pLB3);
[~,pArch1] = archtest(E(:,1)-mean(E(:,1)),'Lags',5);
[~,pArch2] = archtest(E(:,2)-mean(E(:,2)),'Lags',5);
[~,pArch3] = archtest(E(:,3)-mean(E(:,3)),'Lags',5);
fprintf('Residual ARCH(5) p: [%.4f, %.4f, %.4f]\n', pArch1, pArch2, pArch3);

ECT = Y*Beta;
[~,pECT] = adftest(ECT);
fprintf('ECT ADF p = %.4f\n', pECT);

%% 4. Trend-equivalence test (Section S7, full estimation sample)
if isfile(pathOf({'VEC','Tendencias_SE.csv'})) && isfile(pathOf({'SE_Kalman','LLT_decomposition.csv'}))
    V = readtable(pathOf({'VEC','Tendencias_SE.csv'}));
    L = readtable(pathOf({'SE_Kalman','LLT_decomposition.csv'}));
    n = min(height(V), height(L));
    dtr = V.Kidnap_Ransom_Trend(1:n) - L.Trend(1:n);
    [~,pADF] = adftest(dtr);
    [~,pKPSS_def] = kpsstest(dtr,'trend',false);
    [~,pKPSS12] = kpsstest(dtr,'trend',false,'lags',12);
    fprintf('\n--- Trend difference (Jul-1954 to Dec-2025, n=%d) ---\n', n);
    fprintf('ADF p = %.4f  (Appendix: 0.001)\n', pADF);
    fprintf('KPSS default p = %.4f  (Appendix notes: 0.01 with default bandwidth)\n', pKPSS_def);
    fprintf('KPSS lags=12 p = %.4f  (Appendix: 0.071)\n', pKPSS12);
else
    fprintf('\n--- Trend difference test SKIPPED (missing CSV) ---\n');
end

fprintf('\nDONE.\n');
