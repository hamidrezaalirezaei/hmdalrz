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

load('IRAN.mat');  %  Data1997

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

% >>>> d=7      1.6023 
% X = [7.9539, 0.0859, 2.3410;];

%>>>> d=6       1.35
X = [4.75371838959148, 0.105796138913760, 2.95267765804490;];


C = X(1);
epsilon = X(2);
a1 = X(3);

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
options.tolKKT = 1e-3;

%% SET SVM Data
Model  = IRAN.Models.AllDays.Subset1_max;

M = options.Dimension;

%%%%%%%         Set Training Data         %%%%%%%%%

ClassLoad = Model.LoadN;
% ClassPeakLoad = Model.PeakLoad;
% ClassTemp = Model.TempN;
ClassHDay = Model.Data(:,13);
ClassDayType = Model.Data(:,6:12);

LoadMatrix = zeros (size(ClassLoad,1)-(M) , M+1);
for ii = 1:M+1
    LoadMatrix(:,M+2-ii) = ClassLoad(ii:end-(M+1)+ii);
end

TrainingLoad = LoadMatrix(:,2:M+1);
TrainingTarget = LoadMatrix(:,1);
% TrainingMatrix = [TrainingLoad, ClassTemp(M+1:end), ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
TrainingMatrixNoTemp = [TrainingLoad, ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
%%%%%%%         Set Target Data         %%%%%%%%%
LoadN = IRAN.Models.AllDays.NormalPeakLoad;
next = size(IRAN.Models.Target.PeakLoad,1);
InputPattern(1,:) = ([LoadN(1), LoadN(358), LoadN(365), LoadN(702), LoadN(709), LoadN(712), LoadN(716)]);
InputPattern(2,:) = fliplr([ LoadN(724), LoadN(368), LoadN(369),LoadN(372), LoadN(373), LoadN(380), LoadN(387)]);
InputPattern(3,:) = fliplr([ LoadN(725), LoadN(372), LoadN(373), LoadN(376), LoadN(380), LoadN(394), LoadN(401),]);
InputPattern(4,:) = fliplr([LoadN(726), LoadN(376), LoadN(380), LoadN(387), LoadN(394), LoadN(399), LoadN(401), ]);
InputPattern(5,:) = ([LoadN(11), LoadN(729), LoadN(386), LoadN(380), LoadN(379), LoadN(376), LoadN(372), ]);
InputPattern(6,:) = fliplr([LoadN(724), LoadN(372), LoadN(376), LoadN(379), LoadN(380), LoadN(386), LoadN(387)]);
InputPattern(7,:) = ([LoadN(11), LoadN(16), LoadN(369), LoadN(372), LoadN(373), LoadN(380), LoadN(387)]);
InputPattern(8,:) = fliplr([LoadN(29), LoadN(372), LoadN(375), LoadN(376), LoadN(379), LoadN(382), LoadN(386)]);
InputPattern(9,:) = fliplr([  LoadN(728), LoadN(376),  LoadN(377), LoadN(379), LoadN(380), LoadN(382), LoadN(386)]);
InputPattern(10,:) = ([   LoadN(375), LoadN(376), LoadN(379), LoadN(382), LoadN(385), LoadN(386), LoadN(372),]);
InputPattern(11,:) = fliplr([ LoadN(386),  LoadN(385), LoadN(382), LoadN(379), LoadN(376), LoadN(375),LoadN(372),]);
InputPattern(12,:) = ([LoadN(397), LoadN(386), LoadN(380), LoadN(379), LoadN(376), LoadN(375), LoadN(369)]);
InputPattern(13,:) = ([ LoadN(387), LoadN(394), LoadN(399),  LoadN(401), LoadN(407), LoadN(408), LoadN(373),]);
InputPattern(14,:) = ([LoadN(379), LoadN(376), LoadN(387), LoadN(372), LoadN(11), LoadN(369), LoadN(380)]);
InputPattern(15,:) = ([LoadN(389), LoadN(395), LoadN(396), LoadN(398), LoadN(402), LoadN(403), LoadN(404)]);
InputPattern(16,:) = ([LoadN(389), LoadN(390), LoadN(392), LoadN(395), LoadN(396), LoadN(398), LoadN(403)]);
InputPattern(17,:) = fliplr([LoadN(391), LoadN(393), LoadN(395), LoadN(398), LoadN(402), LoadN(404), LoadN(405)]);
InputPattern(18,:) = ([ LoadN(391), LoadN(395), LoadN(398), LoadN(402), LoadN(404), LoadN(405),LoadN(406),  ]);
InputPattern(19,:) = fliplr([ LoadN(409), LoadN(406), LoadN(405), LoadN(404), LoadN(382), LoadN(388), LoadN(402),]);
InputPattern(20,:) = fliplr([ LoadN(382), LoadN(391), LoadN(393), LoadN(395), LoadN(398), LoadN(400),  LoadN(405),   ]);
InputPattern(21,:) = ([ LoadN(414), LoadN(376), LoadN(379), LoadN(394), LoadN(400), LoadN(401), LoadN(408)]);
InputPattern(22,:) = ([  LoadN(391), LoadN(395), LoadN(398), LoadN(402), LoadN(403), LoadN(404),  LoadN(405)]);
InputPattern(23,:) = ([ LoadN(392), LoadN(396), LoadN(407), LoadN(408), LoadN(401), LoadN(399), LoadN(394),]);
InputPattern(24,:) = ([ LoadN(388), LoadN(395), LoadN(402), LoadN(404), LoadN(405), LoadN(406), LoadN(409)]);
InputPattern(25,:) = fliplr([  LoadN(406),LoadN(405),  LoadN(404),LoadN(402),   LoadN(398), LoadN(391), LoadN(388),]);
InputPattern(26,:) = fliplr([LoadN(402), LoadN(404),  LoadN(405), LoadN(406), LoadN(409), LoadN(412), LoadN(416)]);
InputPattern(27,:) = ([ LoadN(412), LoadN(406), LoadN(382), LoadN(391), LoadN(393), LoadN(404), LoadN(405),  ]);
InputPattern(28,:) = fliplr([LoadN(715), LoadN(713), LoadN(666), LoadN(646), LoadN(640), LoadN(387), LoadN(384)]);
InputPattern(29,:) = fliplr([LoadN(390), LoadN(395), LoadN(398), LoadN(402), LoadN(403), LoadN(404), LoadN(405)]);
InputPattern(30,:) = ([LoadN(395), LoadN(402), LoadN(404), LoadN(405), LoadN(406), LoadN(409), LoadN(416)]);
InputPattern(31,:) = ([  LoadN(373), LoadN(387), LoadN(394), LoadN(399), LoadN(401),LoadN(407),  LoadN(16) ]);

% Average Historical Temp:
% TargetTemp = 0.5 * ( TempN(1:31) + TempN(365+1:365+31) );

% TargetMatrix  = [InputPattern, IRAN.Models.Target.TempN, IRAN.Models.Target.Data(:,6:12), IRAN.Models.Target.Data(:,13)];
TargetMatrixNoTemp  = [InputPattern(:,1:M), IRAN.Models.Target.Data(:,6:12), IRAN.Models.Target.Data(:,13)];


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
TargetPeakLoad = IRAN.Models.Target.PeakLoad;

Target_residuals = ( TargetPeakLoad(1:next) - Ypredict(1:next) );
Target_Error_Percentage = ((Target_residuals)./TargetPeakLoad(1:next)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next;
MAPE_Target = 100/next*sum(abs((Target_residuals)./TargetPeakLoad(1:next)));
% MAPE_Target = mean(Target_Error_Percentage(~isinf(Target_Error_Percentage)));
% TargetError = sprintf('Target MAPE = %f   ', MAPE_Target );
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
plot(TargetPeakLoad ./1e3,'k:o');
hold on;
plot(Ypredict ./1e3,'r-s');
legend ('Target ','Forecasted')
% % plot(PeakLoad(1:31),'b:')
% % plot(PeakLoad(699+1:699+31),'g-d')
% title (TargetError);if strcmp (options.Display, 'on')
% end