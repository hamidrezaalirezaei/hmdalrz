function [MAPE_Target,Ypredict] = Forecaster(results, SVRt, options)
%%

M = options.Dimension;
Display = options.Display;

Kernel.arg = options.a ;
Kernel.Kernel = options.Kernel;

MinTrainingLoad = results.Data.MinTrainingLoad;
MaxTrainingLoad = results.Data.MaxTrainingLoad;

TargetLoadN = results.Target.LoadN;
TargetPeakLoad = results.Target.PeakLoad;

TargetTemp = results.Target.TempN;
% TargetTemp = 0.5 * ( results.Data.TempN(1:31) + results.Data.TempN(365+1 : 365+31) );
next = numel(TargetLoadN);

ytar=zeros(next, 1);
Ytarget=zeros(next, 1);
Ypredict=zeros(next, 1);

TA = zeros(31,M);
TA(1,:) = (results.Data.LoadN(end-M+1:end)');
if strcmp(results.Dataset, 'Iran') ==1
    TargetMatrix = zeros(31, M+8);
    TargetMatrix (:, M+1:end) = [ results.Target.Data(:,6:12), results.Target.Data(:,13)];
else
    TargetMatrix = zeros(31, M+9);
    TargetMatrix (:, M+1:end) = [ TargetTemp, results.Target.Data(:,6:12), results.Target.Data(:,13)];
end
S = SVRt.sv;
b = SVRt.b;
eta = SVRt.eta;
TrainingMatrix = SVRt.TrainingMatrix;

for ii=1:31
    TargetMatrix(ii,1:M) = TA(ii,:);
    
    i = 1:1:numel(S);
    y = sum( eta(i) * kernel(TrainingMatrix(i,:)', TargetMatrix(ii,:)', Kernel.Kernel, Kernel.arg) );
    Ytarget(ii) = y + b;

%     Ytarget(ii) = ytar(ii) + b ;
    Ypredict(ii) = (Ytarget(ii) *( MaxTrainingLoad - MinTrainingLoad)) +MinTrainingLoad;
    
    TA(ii+1,2:M) = TA(ii,1:M-1);
    TA(ii+1,1) = Ytarget(ii);
end

% Error Evaluate
% residuals = (ytarget' - LoadT);
% Target_Error_Percentage = (abs(residuals)./LoadT) *100;
% ytarget(1:30)=ytarget(2:31);
% ytarget(31)=[];
% Ypredict = ( MaxTrainingLoad - MinTrainingLoad)*ytarget + MinTrainingLoad;

Target_residuals = ( TargetPeakLoad(1:31) - Ypredict(1:31) );
Target_Error_Percentage = (abs(Target_residuals)./TargetPeakLoad(1:31)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next
MAPE_Target = 100/next*sum(abs((Target_residuals)./TargetPeakLoad(1:next)));
PAPE_Target = max(Target_Error_Percentage)
Max_Target = max(abs(Target_residuals))

TargetError = sprintf('Target MAPE = %f   ', MAPE_Target );
% TargetError = sprintf('Target MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Target, C, a, epsilon);


% Squared Error:
SqE_Target = sum (Target_residuals .^ 2);
% Root Squared Error:
RSE_Target = SqE_Target ^ 0.5;
% Mean Squared Error:
MSE_Target = mean(Target_residuals .^ 2);
% Root Mean Squared Error:
RMSE_Target = MSE_Target ^ 0.5;

figure;
plot(TargetPeakLoad(),'k:o');
hold on;
plot(Ypredict,'r-s');
legend ('Target ','Forecasted')
title (TargetError);
 