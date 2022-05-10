% function MAPE_Target = new_main(X)
clear all;
clc;
%% Load Data

%%%%%%%          Note               %%%%%%%
% Data [365 x 61] >>>  eache row : one day data
% column 1: year
% column 2: month of year
% column 3: Day of month
% column 4: Peak Load of Day
% column 5: Average Day Temp
% column 6:12 : weekday indeces [Mon .... Sun]
% column 13: Holiday index [ 0 or 1]
% column 14:61 : 30min load data of day

load('Eunite.mat');  %  

%  Normalize Daily Peak Load 
PeakLoad = Eunite.Models.AllDays.PeakLoad;
MinTrainingLoad = Eunite.MinPeakLoad;
MaxTrainingLoad = Eunite.MaxPeakLoad;
LoadN = Eunite.Models.AllDays.NormalPeakLoad;

% Normalize Average Daily Temprature :
MinTemp = Eunite.MinTemp;
MaxTemp = Eunite.MaxTemp;
TempN = Eunite.Models.AllDays.NormalTemp;
% 
% data = [LoadN, TempN];
% % Number of Clusters
% gamma = 100.150;
% [U,~,iter_num,center] = KFCMC(data, 2, gamma);
% str = sprintf('MAX Iteration Number  = %d' , iter_num);
% disp(str);
% 
% % Find the data points with highest grade of membership in clusters
% maxU = max(U);
% index1 = find(U(1,:) == maxU);
% index2 = find(U(2,:) == maxU);
% 
% if max(LoadN(index1) ) >= 0.5 
%     Subset1_Max = index1;
%     Subset2_min = index2;
% else
%     Subset1_Max = index2;
%     Subset2_min = index1;
% end
% 
% figure; hold on;
% plot(index1, LoadN(index1), 'rs');
% plot(index2, LoadN(index2), 'bd');
% 
% figure;
% subplot(211)
% plot(U(1,:))
% subplot(212)
% plot(U(2,:))
%% Set Options
% C = 0.128;
% epsilon = 0.010;
% a1 = 0.9000;
% C = 1.48230067;
% epsilon = 0.0089544128;
% a1 = 0.4025;

% No temp
X = [1.47399013864547,0.0163825885296101,0.405134851825157;];
% X = [ C, epsilon, a1];
% X = [0.128; 0.01; 0.9];

C = X(1);...0.128;
epsilon = X(2);...0.010;
a1 = X(3);...0.9000;

a2 = 1;       % second MultiLayer Perceptron parameters

Kernel{1} = 'linear';
Kernel{2} = 'poly';
Kernel{3} = 'rbf';
Kernel{4} = 'wavelet';
Kernel{5} = 'sigmoid';

options.C = C;
options.a(1) = a1;
options.a(2) = a2; 
options.epsilon = epsilon ;
options.Kernel = Kernel{4};
options.Dimension = 7;
options.Display = 'off';
options.solver = 'qp';
options.tolKKT = 1e-3;

%% SET SVM Data
Model  = Eunite.Models.AllDays.Subset1_max;

M = options.Dimension;

%%%%%%%         Set Training Data         %%%%%%%%%

ClassLoad = Model.LoadN;
ClassPeakLoad = Model.PeakLoad;
ClassTemp = Model.TempN;
ClassHDay = Model.Data(:,13);
ClassDayType = Model.Data(:,6:12);

LoadMatrix = zeros (size(ClassLoad,1)-(M) , M+1);
for ii = 1:M+1
    LoadMatrix(:,M+2-ii) = ClassLoad(ii:end-(M+1)+ii);
end

TrainingLoad = LoadMatrix(:,2:M+1);
TrainingTarget = LoadMatrix(:,1);
TrainingMatrix = [TrainingLoad, ClassTemp(M+1:end), ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
TrainingMatrixNoTemp = [TrainingLoad, ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];

%%%%%%%         Set Target Data         %%%%%%%%%
next = size(Eunite.Models.Target.PeakLoad,1);
InputPattern = zeros(next, M);
InputPattern(1,:) = (Model.LoadN(end-M+1:end)');

% Average Historical Temp:
TargetTemp = 0.5 * ( TempN(1:31) + TempN(365+1:365+31) );

TargetMatrix  = [InputPattern, TargetTemp, Eunite.Models.Target.Data(:,6:12), Eunite.Models.Target.Data(:,13)];
TargetMatrixNoTemp  = [InputPattern, Eunite.Models.Target.Data(:,6:12), Eunite.Models.Target.Data(:,13)];


%% SVM Train
% tic;
SVR_struct = svrtrain(TrainingMatrixNoTemp, TrainingTarget, options);
% TrainingTime = toc
%% SVM Regression
% tic;
Ytarget = svrforecaster(TargetMatrixNoTemp, SVR_struct, options);
% PredictTime = toc

Ypredict = ( ( MaxTrainingLoad - MinTrainingLoad) .* Ytarget ) +MinTrainingLoad;
%% Error Evaluate
TargetPeakLoad = Eunite.Models.Target.PeakLoad;

Target_residuals = ( TargetPeakLoad(1:next) - Ypredict(1:next) );
Target_Error_Percentage = ((Target_residuals)./TargetPeakLoad(1:next)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next;
MAPE_Target = 100/next*sum(abs((Target_residuals)./TargetPeakLoad(1:next)));
% MAPE_Target = mean(Target_Error_Percentage(~isinf(Target_Error_Percentage)));
TargetError = sprintf('Target MAPE = %f   ', MAPE_Target );
PAPE_Target = max(Target_Error_Percentage);
Max_Target = max(abs(Target_residuals));
% 
% Squared Error:
SqE_Target = sum (Target_residuals .^ 2);
% Root Squared Error:
RSE_Target = SqE_Target ^ 0.5;
% Mean Squared Error:
MSE_Target = mean(Target_residuals .^ 2);
% Root Mean Squared Error:
RMSE_Target = MSE_Target ^ 0.5;
% NMSE Error:
num = sum( (Target_residuals).^2 ) ./ next ;
den = sum( (TargetPeakLoad - mean(TargetPeakLoad) ).^2 ) ./ (numel(TargetPeakLoad) - 1);
NMSE_Target = num/den;
REP_Target = sqrt( sum( (Target_residuals).^2 ) / sum( (TargetPeakLoad).^2 ) ) * 100;

error.MAE = MAE_Target;
error.MAPE = MAPE_Target;
error.SqE = SqE_Target;
error.RSE = RSE_Target;
error.MSE = MSE_Target;
error.RMSE = RMSE_Target;
error.NMSE = NMSE_Target;
error.REP = REP_Target;
error.PAPE = PAPE_Target;
error.Max = Max_Target;
disp(error);
%% Plot Resulst
figure;
plot(TargetPeakLoad,'k:o');
hold on;
plot(Ypredict,'b-s');
legend ('Target ','Forecasted')
% plot(PeakLoad(1:31),'b:')
% plot(PeakLoad(699+1:699+31),'g-d')
title (TargetError);
% if strcmp (options.Display, 'on')
% end