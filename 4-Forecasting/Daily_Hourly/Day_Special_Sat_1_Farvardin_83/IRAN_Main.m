% function MAPE_Target = Chaotic1(X)
clear all; ... clc;
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

load('IRAN_Target83.mat');  %  Data1381-1382

%  Normalize Daily Peak Load 
% PeakLoad = IRAN.Models.AllDays.PeakLoad;
MinTrainingLoad = IRAN.MinHourlyLoad;
MaxTrainingLoad = IRAN.MaxHourlyLoad;


%% Set Options
next = 24*1;...size(IRAN.Models.Target.PeakLoad,1);
TargetPeakLoad = IRAN.Models.Target.HourlyLoad(1:next);

% figure
% plot(IRAN.Models.AllDays.HourlyLoad(1:24) / 1e3);
% hold on
% plot(IRAN.Models.AllDays.HourlyLoad((24*365)+1:(24*365)+24) / 1e3);
% plot(TargetPeakLoad / 1e3)
% legend('1381 ¸Äjn»oÎ', '1382 ¸Äjn»oÎ',  '1383 ¸Äjn»oÎ');

%%   test 1 : 1 Farvardin Model: AllDay - History Data 7days

tao = 6;  dim = 5;   numday =20;      % mape =  2.5092  
X = [70879.0510064170,0.155372467105423,110.521978725076;];

Model =IRAN.Models.AllDays;
ClassLoad = Model.NormalHourlyLoad(end-(24*numday)+1 : end);

%%   test 2 : 1 Farvardin Model: SpecialDay - History Data 7days

% tao = 6;  dim = 4;    numday =10;     % MAPE =   2.3324
% X = [67389.87,0.05342,0.23824;];
% 
% Model =IRAN.Models.SpecialDays.AllDay;
% ClassLoad = Model.NormalHourlyLoad(end-(24*numday)+1 : end);

%%   test 3 : 1 Farvardin Model: Corr_Analysis - History Data 10 days
% numday = 7;     %mape : 3.5142
% X = [640528.827536199,0.000691471978804848,22.7683312340223;] ;
% tao = 7;
% dim = 7;        %  Cao : 5-6 , FNN: 6-8
% numday = 10;    %   mape = 2.4939     2.3855
% X = [308964.373792898,0.0683298929141030,0.407278490751580;];
% 
% load('InputDays1_7_corr0.90.mat');  %  Input Pattern 1 farvardin
% Model = InputDays(1, 1);
% ClassLoad = InputDays(1, 1).NormalHourlyLoad(end-(24*numday)+1 : end);

%%
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
options.Dimension = dim;
options.Display = 'off';
options.solver = 'qp';
options.tolKKT = 1e-3;

%% Phase space reconstruction
% nBins = [ 5000 ];
% nLags = 50;
% [MInf corrs]= ami(ClassLoad,nBins,nLags); 
% for i = 2:50
%     if MInf(i) < MInf(i-1) && MInf(i) < MInf(i+1)
%         tao = find(MInf == MInf(i));
%         break;
%     end
% end
% TSAnalysis.MInf = MInf;
% TSAnalysis.Correlation = corrs;
% 
% % tao =1;
% %%%%%%%%        FNN
% mmax = 20;
% rtol=10;
% atol=2;
% [FNN] = fnn_deneme(ClassLoad,tao,mmax,rtol,atol);
% dimFNN = find(FNN == min(FNN) ,1);
% 
% %%%%%%%%%        Cao's Method
% [E1 E2] = cao_deneme(ClassLoad,tao,mmax);
% for i = 2:mmax-1
%     if E1(i) - E1(i-1) <= 0.1 && E1(i+1) - E1(i) <= 0.1
%         dimCao = find(E1 == E1(i));
%         dimCao = dimCao +1;
%         break;
%     end
% end


%%
l = length(ClassLoad);
lbar = l - (dim +1- 1) * tao ;

Y=zeros(lbar,dim); 
for i=1:dim+1
    Y(:,i)=ClassLoad((1:lbar)+(i-1)*tao)';
end

xTrain = Y(:,1:dim);
yTrain = Y(:,dim+1);

%%%%%%      Train SVR
svrstruct = svrtrain(xTrain, yTrain, options);
% ytr = svrstruct.TrainingPredict;

%%%%%% Target data
% lt = numel(testdata);
% ytar=zeros(1,next);
% Ynextt=ClassLoad(1:next);
Ypredict=zeros(1,next);
Ytarget=zeros(1,next);
TargetMatrix = zeros(next, dim);

for ii = 0:dim-1
    TargetMatrix(1:(dim-ii)*tao, ii+1) = ClassLoad(end-(dim-ii)*tao+1:end);
end

for ii=1:next
    
    Ytar = svrforecaster(TargetMatrix(ii,:), svrstruct, options);
    Ytarget(ii) = Ytar;
    TargetMatrix( tao+ii, dim) = Ytarget(ii);
    for cc = 1:dim
         if  ii <= next-cc*tao;
             TargetMatrix( cc*tao+ii, dim +1 - cc) = Ytarget(ii);
         end
    end
    Ypredict (ii) = ( ( MaxTrainingLoad - MinTrainingLoad) .* Ytarget (ii) ) +MinTrainingLoad;

    
end

%% Error Evaluate
% TargetPeakLoad = IRAN.Models.Target.Data(1,14:37);
Target_residuals = ( TargetPeakLoad(1:next) - Ypredict(1:next)' );
MAPE_Target = 100/next*sum(abs((Target_residuals)./TargetPeakLoad(1:next)));
Target_Error_Percentage = ((Target_residuals)./TargetPeakLoad(1:next)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next;
TargetError = sprintf('Target MAPE = %f   ', MAPE_Target );
PAPE_Target = max(Target_Error_Percentage);
Max_Target = max(abs(Target_residuals));
% 
Ypredict = Ypredict';

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
plot(TargetPeakLoad / 1e3,'k:o');
hold on;
plot(Ypredict  / 1e3 ,'r-d');
% legend('Â÷¤H» nHk£¶','1 Ïk¶', '2 Ïk¶', '3 Ïk¶')
legend ('Target ','Forecasted')
% % plot(PeakLoad(1:31),'b:')
% % plot(PeakLoad(699+1:699+31),'g-d')
% title (TargetError);if strcmp (options.Display, 'on')
% end