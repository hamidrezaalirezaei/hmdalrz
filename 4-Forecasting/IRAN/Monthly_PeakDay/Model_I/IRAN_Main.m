% function MAPE_Target = new_main(X)
clear all;
% clc;
%% Load Data

load('IRAN.mat');  %  Data IRAN 
load('data1383.mat');
%%%%%%%          Note               %%%%%%%
% Data [365 x 37] >>>  eache row : one day data
% column 1: year
% column 2: month of year
% column 3: Day of month
% column 4: Peak Load of Day
% column 5: Average Day Temp
% column 6:12 : weekday indeces [Mon .... Sun]
% column 13: Holiday index [ 0 or 1]
% column 14:37 : hourly load data of day


%  Normalize Daily Peak Load 
% PeakLoad = IRAN.Models.AllDays.PeakLoad;
MinTrainingLoad = IRAN.MinPeakLoad;
MaxTrainingLoad = IRAN.MaxPeakLoad;
% LoadN = IRAN.Models.AllDays.NormalPeakLoad;

% Normalize Average Daily Temprature :
% MinTemp = IRAN.MinTemp;
% MaxTemp = IRAN.MaxTemp;
% TempN = IRAN.Models.AllDays.NormalTemp;

%% Set Options

% d=5  PSO      2.25 >>
% X = [7.14938347977905,0.104651753922395,3.61009850257487;];
% d=6  GA        2.00 >>
X = [5.20734018989457,0.0951310617791334,3.21187961761623;];
% % d=6  PSO      2.00 >>
% X = [6.17591756084650,0.0975610602877918,3.29456449999570;];
% % d=7  GA        2.11 >>
% X = [7.95403293305674,0.0859071087612386,2.34099553437260;];
% % d=7  PSO       2.21 >>
% X = [7.97505558451732,0.0859015157787134,2.34119727909012;];
% % d=8  GA         2.10 >>
% X = [9.43444414761939,0.0870903436889322,2.53143960586954;];
% % d=8  PSO      2.09 >>
% X =[6.65370127835451,0.0854063308859217,2.38945318086810;];
% % d=9  GA       2.38 >>
% X = [7.47840730974281,0.0702155513395804,3.82980766046512;];
% % d=9  PSO      2.33>>
% X = [8.18493363523657,0.0711318962950505,3.81790266151026;];
% % d=10  GA      2.49 >>
% X = [7.23822330168416,0.0945201609950933,3.32633880233298;];
% % d=10  PSO     2.52>>
% X = [77.8293472658001,0.0979193720790157,11.1959772621585;];

C =   X(1);...1.2298e+05;
epsilon = X(2);...0.1;
a1 = X(3);...1.53;

a2 = 0.45;       % second MultiLayer Perceptron parameters

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
options.Dimension = 6;
options.Display = 'off';
options.solver = 'qp';
options.tolKKT = 1e-6;

%% SET SVM Data
Model  = IRAN.Models.AllDays.Subset1_max;

M = options.Dimension;

%%%%%%%         Set Training Data         %%%%%%%%%


% ClassLoad = IRAN.Models.AllDays.NormalPeakLoad;
% ClassHDay = IRAN.Models.AllDays.Data (:,13);
% ClassDayType = IRAN.Models.AllDays.Data (:,6:12);
% ClassPeakLoad = Model.PeakLoad;

ClassLoad = Model.LoadN;
ClassTemp = Model.TempN;
ClassHDay = Model.Data(:,13);
ClassDayType = Model.Data(:,6:12);

% LoadN = IRAN.Models.AllDays.NormalPeakLoad;

LoadMatrix = zeros (size(ClassLoad,1)-(M) , M+1);
for ii = 1:M+1
    LoadMatrix(:,M+2-ii) = ClassLoad(ii:end-(M+1)+ii);
end

TrainingLoad = LoadMatrix(:,2:M+1);
TrainingTarget = LoadMatrix(:,1);
% TrainingMatrix = [TrainingLoad, ClassTemp(M+1:end), ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
TrainingMatrixNoTemp = [TrainingLoad, ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];

%%%%%%%         Set Target Data         %%%%%%%%%
next = 31;...size(IRAN.Models.Target.PeakLoad,1);
InputPattern = zeros(next, M);
InputPattern(1,:) = (Model.LoadN(end-M+1:end)');
% Average Historical Temp:
% TargetTemp = 0.5 * ( TempN(1:31) + TempN(365+1:365+31) );
% 
% TargetMatrix  = [InputPattern, IRAN.Models.Target.TempN, IRAN.Models.Target.Data(:,6:12), IRAN.Models.Target.Data(:,13)];
% TargetMatrixNoTemp  = [InputPattern, IRAN.Models.Target.Data(:,6:12), IRAN.Models.Target.Data(:,13)];
TargetMatrixNoTemp  = [InputPattern, Data1383(1:next, 6:12), Data1383(1:next,13)];


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
TargetPeakLoad = Data1383(1:next, 4);

Target_residuals = ( TargetPeakLoad(1:next) - Ypredict(1:next) );
Target_Error_Percentage = ((Target_residuals)./TargetPeakLoad(1:next)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next;
MAX_Target = max(abs(Target_residuals));
MAPE_Target = mean(abs(Target_Error_Percentage(~isinf(Target_Error_Percentage))));
PAPE_Target = max(Target_Error_Percentage);

% TargetError = sprintf('Target MAPE = %f   ', MAPE_Target );
% TargetError = sprintf('Target MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Target, C, a, epsilon);


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
% 
error.MAE = MAE_Target;
error.PAPE = PAPE_Target;
error.MAX = MAX_Target;
error.MAPE = MAPE_Target;
error.SqE = SqE_Target;
error.RSE = RSE_Target;
error.MSE = MSE_Target;
error.RMSE = RMSE_Target;
error.NMSE = NMSE_Target;
error.REP = REP_Target;
% 
disp(error);
%% Plot Resulst
figure;
plot(TargetPeakLoad ./1e3,'k:o');
hold on;
plot(Ypredict ./1e3,'b-s');
legend ('Target ','Forecasted')
% % plot(PeakLoad(1:31),'b:')
% % plot(PeakLoad(699+1:699+31),'g-d')
% title (TargetError);if strcmp (options.Display, 'on')
% end