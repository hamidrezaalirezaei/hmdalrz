% function MAPE_Target = corr_main(X)
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

load('Eunite.mat');  %  Data1997

%  Normalize Daily Peak Load 
PeakLoad = Eunite.Models.AllDays.PeakLoad;
MinTrainingLoad = Eunite.MinPeakLoad;
MaxTrainingLoad = Eunite.MaxPeakLoad;
LoadN = Eunite.Models.AllDays.NormalPeakLoad;

% Normalize Average Daily Temprature :
MinTemp = Eunite.MinTemp;
MaxTemp = Eunite.MaxTemp;
TempN = Eunite.Models.AllDays.NormalTemp;

%% Set Options

% >>>> 1.39    &  1.24
% X = [1.48230067;  0.0089544128;  0.4025;];

% >>>> 1.04    &  0.98
X = [1.4739;  0.01638;  0.4051; ];

% >>>> 0.98    &  0.91
% X = [1.65471423384417,0.0153407340704427,0.420352437204032;];
% 
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
% TrainingMatrix = [TrainingLoad, ClassTemp(M+1:end), ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
TrainingMatrixNoTemp = [TrainingLoad, ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];

%%%%%%%         Set Target Data         %%%%%%%%%
next = size(Eunite.Models.Target.PeakLoad,1);
InputPattern = zeros(next, M);

InputPattern(1,:) = ([LoadN(1), LoadN(12), LoadN(359), LoadN(360), LoadN(366), LoadN(724), LoadN(725)]);
InputPattern(2,:) = ([LoadN(2), LoadN(13), LoadN(360), LoadN(361), LoadN(367), LoadN(725), LoadN(726)]);
InputPattern(3,:) = ([LoadN(12), LoadN(26), LoadN(366), LoadN(396), LoadN(366), LoadN(724), LoadN(725)]);
InputPattern(4,:) = fliplr([LoadN(348), LoadN(376), LoadN(389), LoadN(396), LoadN(698), LoadN(711), LoadN(719)]);
InputPattern(5,:) = ([LoadN(332), LoadN(343), LoadN(344), LoadN(377), LoadN(384), LoadN(700), LoadN(706)]);
InputPattern(6,:) = ([LoadN(81), LoadN(361), LoadN(362), LoadN(368), LoadN(369), LoadN(382), LoadN(730)]);
InputPattern(7,:) = ([LoadN(699), LoadN(310), LoadN(349), LoadN(700), LoadN(332), LoadN(720), LoadN(672)]);
InputPattern(8,:) = ([LoadN(374), LoadN(328), LoadN(344), LoadN(325), LoadN(381), LoadN(680), LoadN(386)]);
InputPattern(9,:) = fliplr([  LoadN(728), LoadN(718), LoadN(704), LoadN(364), LoadN(371), LoadN(368),  LoadN(382)]);
InputPattern(10,:) = fliplr([ LoadN(348), LoadN(368), LoadN(371), LoadN(389), LoadN(711), LoadN(719), LoadN(729)]);
InputPattern(11,:) = fliplr([LoadN(336), LoadN(338), LoadN(377), LoadN(678), LoadN(699), LoadN(706), LoadN(715)]);
InputPattern(12,:) = ([LoadN(336), LoadN(374), LoadN(378), LoadN(379), LoadN(673), LoadN(680), LoadN(713)]);
InputPattern(13,:) = ([LoadN(314), LoadN(330), LoadN(336), LoadN(384), LoadN(685), LoadN(686), LoadN(715)]);
InputPattern(14,:) = fliplr([LoadN(708), LoadN(338), LoadN(667), LoadN(713), LoadN(695), LoadN(685), LoadN(686)]);
InputPattern(15,:) = fliplr([LoadN(322), LoadN(713), LoadN(715), LoadN(720), LoadN(700), LoadN(386), LoadN(686)]);
InputPattern(16,:) = fliplr([LoadN(368), LoadN(382), LoadN(711), LoadN(719), LoadN(690), LoadN(730), LoadN(728)]);
InputPattern(17,:) = ([LoadN(4), LoadN(719), LoadN(424), LoadN(725), LoadN(698), LoadN(403), LoadN(438)]);
InputPattern(18,:) = ([ LoadN(699), LoadN(378), LoadN(665), LoadN(667), LoadN(678), LoadN(685), LoadN(686), ]);
InputPattern(19,:) = ([ LoadN(664), LoadN(665), LoadN(675), LoadN(678), LoadN(686), LoadN(692), LoadN(693),]);
InputPattern(20,:) = fliplr([LoadN(374),LoadN(379), LoadN(387), LoadN(644), LoadN(646), LoadN(658), LoadN(685),   ]);
InputPattern(21,:) = ([LoadN(345), LoadN(387), LoadN(399), LoadN(673), LoadN(680), LoadN(714), LoadN(715)]);
InputPattern(22,:) = ([  LoadN(314), LoadN(374), LoadN(384), LoadN(387), LoadN(673), LoadN(679),  LoadN(716)]);
InputPattern(23,:) = ([ LoadN(719), LoadN(382), LoadN(368), LoadN(361), LoadN(690), LoadN(730), LoadN(711)]);
InputPattern(24,:) = fliplr([ LoadN(396), LoadN(698), LoadN(719), LoadN(368), LoadN(403), LoadN(371), LoadN(684)]);
InputPattern(25,:) = ([ LoadN(199), LoadN(667), LoadN(672), LoadN(639), LoadN(643),  LoadN(640), LoadN(636)]);
InputPattern(26,:) = ([LoadN(715), LoadN(667),  LoadN(640), LoadN(384), LoadN(349), LoadN(330), LoadN(314)]);
InputPattern(27,:) = fliplr([LoadN(715), LoadN(706), LoadN(700), LoadN(686), LoadN(667), LoadN(386), LoadN(384)]);
InputPattern(28,:) = fliplr([LoadN(715), LoadN(713), LoadN(666), LoadN(646), LoadN(640), LoadN(387), LoadN(384)]);
InputPattern(29,:) = fliplr([LoadN(676), LoadN(386), LoadN(665), LoadN(722), LoadN(688), LoadN(694), LoadN(671)]);
InputPattern(30,:) = fliplr([LoadN(368), LoadN(371), LoadN(382), LoadN(410), LoadN(690), LoadN(704), LoadN(719)]);
InputPattern(31,:) = fliplr([  LoadN(368), LoadN(371), LoadN(376), LoadN(396),LoadN(403),  LoadN(411), LoadN(712), ]);

% TargetMatrix = zeros(next, M+9);

% Average Historical Temp:
% TargetTemp = 0.5 * ( TempN(1:31) + TempN(365+1:365+31) );

% TargetMatrix  = [InputPattern, Eunite.Models.Target.TempN, Eunite.Models.Target.Data(:,6:12), Eunite.Models.Target.Data(:,13)];
TargetMatrixNoTemp  = [InputPattern, Eunite.Models.Target.Data(1:next,6:12), Eunite.Models.Target.Data(1:next,13)];


%% SVM Train
tic;
SVR_struct = svrtrain(TrainingMatrixNoTemp, TrainingTarget, options);
% TrainingTime = toc
%% SVM Regression
tic;
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
plot(Ypredict,'r-s');
legend ('Target ','Forecasted')
% % plot(PeakLoad(1:31),'b:')
% % plot(PeakLoad(699+1:699+31),'g-d')
% title (TargetError);
% if strcmp (options.Display, 'on')
% end