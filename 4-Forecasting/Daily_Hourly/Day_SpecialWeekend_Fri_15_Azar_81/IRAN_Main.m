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
% load('InputDays1_7_corr0.90.mat');  %  Input Pattern 1 farvardin

%  Normalize Daily Peak Load 
% PeakLoad = IRAN.Models.AllDays.PeakLoad;
MinTrainingLoad = IRAN.MinHourlyLoad;
MaxTrainingLoad = IRAN.MaxHourlyLoad;


%% Set Options

% X = [653152.287994785	0.0134801506959253	6.38557316188163];

next = 24*1;...size(IRAN.Models.Target.PeakLoad,1);
% %     Day 684
selctedDay  = 261;

TargetPeakLoad = IRAN.Models.AllDays.HourlyLoad( (24 * (selctedDay-1) ) +1 : (24 * (selctedDay-1) ) + next);
TargetPeakLoad = TargetPeakLoad(:);
%%   test 1 : Fri 15 Azar 81 (HistoryData 261) Model: AllDay - History Data 7days
  
% tao = 6;  dim = 4;    numday =7;      % MAPE: 9.1528
% X = [662104.762261683,0.356435870083866,8529.66964244783;];

% tao = 6;  dim = 4;    numday =14;      % MAPE: 9.1529
% X = [780110.803750140	0.433862676344553	6189.07853005438];

tao = 6;  dim = 4;    numday =21;      % MAPE: 7.0009
X = [284956.58753	0.22951318094	474.22355];

% numday = 7;
% tao = 6;
% dim = 4;
% 
Model =IRAN.Models.AllDays;
ClassLoad = Model.NormalHourlyLoad(end-(24*numday)+1 : end);
% 
% plot(Model.HourlyLoad(24*(selctedDay-1-numday)+1 : 24*(selctedDay-1))/10e3);
% plot(24*numday + 0 : 24*numday+next , Model.HourlyLoad(24*(selctedDay-1) + 0: 24*(selctedDay-1)+next)/10e3,'r')
% legend('xp��A �I� �jHj' , 'y�I�pA �I� �jHj');

%%   test 2 : Fri 15 Azar 81 (HistoryData 261) Model: SpeciallDay - History Data 7days
% tao = 6;  dim = 4;    numday =7;      % MAPE: 2.2678
% X = [457497.161571438,1.00010240320104e-06,24.4775144164931;];

% tao = 5;  dim = 5;    numday =7;      % MAPE: 1.9242
% X = [613560.727909604,6.02754689551256e-05,36.0518443171809;];

% tao = 5;  dim = 5;    numday =10;      % MAPE: 1.8214
% X = [721070.236049328,0.000983883050572539,26.6069851959016;];

% tao = 5;  dim = 5;    numday =14;      % MAPE: 2.6247
% X = [288040.245215577,0.00643897021971211,4.07532430319481;];


% numday = 14;
% tao = 5;    % 6 and dim = 4
% dim = 5; ...FNN 5,      Cao : 6-8
% 
% Model =IRAN.Models.SpecialDays;
% daylimit = find (Model.index == selctedDay) -1;
% 
% ClassLoad = Model.AllDay.NormalHourlyLoad(24*(daylimit-numday-1)+1 : 24*(daylimit-1 ));
% 
% plot(Model.AllDay.HourlyLoad(24*(daylimit-1-numday)+1 : 24*(daylimit-1))/10e3);
% plot(24*numday + 0 : 24*numday+next , Model.AllDay.HourlyLoad(24*(daylimit-1) + 0: 24*(daylimit-1)+next)/10e3,'r')
% legend('xp��A �I� �jHj' , 'y�I�pA �I� �jHj');

%%   test 3 : Fri 15 Azar 81 (HistoryData 261) Model: Corr_Analysis - History Data 10 days

% tao = 4;  dim = 6;    numday =7;      % MAPE: 5.2995
% X = [242798.192400344	0.118559952903874	908.197842822860];

% tao = 5;  dim = 5;    numday =7;      % MAPE: 5.456
% X = [473904.523454458,0.153391980440138,1087.96600866354;];

% tao = 5;  dim = 5;    numday =10;      % MAPE: 6.2956
% X = [271651.773183781,0.162779760847868,781.640954110508;];

% tao = 6;  dim = 4;    numday =14;      % MAPE: 4.6316
% X = [813294.153534890	0.136742296874928	983.424656684504];

% tao = 4;  dim = 6;    numday =10;      % MAPE: 4.6249
% X =  [270068.953137355,0.109982867018717,459.103541511817;];

% tao = 4;  dim = 6;    numday =14;      % MAPE: 3.3818
% X = [396244.530289524,0.129271836911080,189.852109023691;] ;
% 
% numday = 14;
% tao = 4;
% dim = 6;        %  Cao : 5-6 , FNN: 4 or 6

% load 20inputDays_for_Fri_15_Azar81_261.mat;
% Model = InputDays;
% ClassLoad = Model.NormalHourlyLoad(end-(24*numday)+1 : end);
% 
% plot(InputDays.HourlyLoad(end-(24*numday)+1 : end)/10e3);
% plot(24*numday + 0 : 24*numday+next , IRAN.Models.AllDays.HourlyLoad(24*(selctedDay-1) + 0: 24*(selctedDay-1)+next)/10e3,'r')
% legend('xp��A �I� �jHj' , 'y�I�pA �I� �jHj');

%%
% TargetSampleindex = 252;
% HistoryLoadData = IRAN.Models.AllDays.HourLoad(1: selctedDay-1 , :)';
% % %% find Days of Pattern 
% % set Same Day of Last Year as Sample
% TargetLoadData = IRAN.Models.AllDays.Data(TargetSampleindex, 14:37 )';
% 
% 
% % Correlation Analysis between Target Days and Historical Data's
% Targetcorrelation = zeros(size(HistoryLoadData,2),1);
% for d = 1;
%     for pd = 1:size(HistoryLoadData,2);
%         correl=corr(TargetLoadData(:,d),HistoryLoadData(:,pd));
%         Targetcorrelation(pd,d) = correl(1,1);
% %         Targetcorrelation(d,pd) = Targetcorrelation(pd,d);
%     end
% end
% CorrelationAnalysis = [ (1:1:size(HistoryLoadData,2))' , Targetcorrelation];
%  maxcor =  find( CorrelationAnalysis (:,2) >= 0.85 );
%     HistoryDataaa = HistoryLoadData(:, maxcor);
%     scorrel (2:numel(maxcor)+1,1) = maxcor; 
%     scorrel (1,1:numel(maxcor)) = maxcor;
%     scorrel (numel(maxcor)+2,1:numel(maxcor)) = CorrelationAnalysis (maxcor,d); 
% while (numel(maxcor) >20 )
%         clear scorrel indexmax c candidate candidateindex;
%         scorrel (2:numel(maxcor)+1,1) = maxcor; 
%     scorrel (1,1:numel(maxcor)) = maxcor;
%     scorrel (numel(maxcor)+2,1:numel(maxcor)) = CorrelationAnalysis (maxcor,d); 
%     
%         for iii = 2:numel(maxcor)
%             for ddd = iii  : numel(maxcor)
%         scorrel (iii, ddd) = corr(HistoryDataaa(:,iii-1), HistoryDataaa(:, ddd) );
%             end
%         av = scorrel( iii, 2:end);
%         c(iii-1) =  max(av ) ;
%         indexmax(iii-1) = find ( av >= c(iii-1) ) +1;
%             if scorrel(end, indexmax(iii-1)) >= scorrel(end, iii-1) 
%             candidate(iii -1) = scorrel(1, indexmax(iii-1));
%             else
%             candidate(iii -1) = scorrel(iii, 1);
%             end
%         end
%         candidate = unique(candidate);
%         candidateindex = find(candidate > 0);
%         candidate = candidate(candidateindex);
%         Inputcandidate.conditate = [ candidate+0 ;CorrelationAnalysis(candidate,1)'];
%         Inputcandidate.conditatesort = sortrows(Inputcandidate.conditate', 2);
% HistoryDataaa = HistoryLoadData(:, candidate);
% maxcor = candidate;
% end
% InputDays.HourLoad = HistoryLoadData ( :, Inputcandidate.conditatesort(:,2))';
% InputDays.candidate = Inputcandidate;
% % Vectorized Hour Load
% [r,c] = size(InputDays.HourLoad);
% InputDays.HourlyLoad =[];
% for i = 1:r
%     InputDays.HourlyLoad = [InputDays.HourlyLoad ; InputDays.HourLoad(i,:)'];
% end
% 
% %%% Normalized Data
% InputDays.NormalHourlyLoad = Normalize_Fcn(InputDays.HourlyLoad, IRAN.MinHourlyLoad, IRAN.MaxHourlyLoad);
% save('85_inputDays_for_Fri_15_Azar81_261.mat', 'InputDays')

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
Target_residuals = ( TargetPeakLoad(1:next) - Ypredict(1:next)' );
MAPE_Target = 100/next*sum(abs((Target_residuals)./TargetPeakLoad(1:next)));
Target_Error_Percentage = ((Target_residuals)./TargetPeakLoad(1:next)) *100;
MAE_Target = (sum(abs(Target_residuals))) /next;
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
% str1 = sprintf( '   �� �k� \n 7/076 = I�i');
% str2 = sprintf( '   �j �k� \n  1/82 = I�i');
% str3 = sprintf( '   �w �k� \n  3/18 = I�i');
%% Plot Resulst
% figure;
plot(TargetPeakLoad ./ 1e3,'k:o');
hold on;
plot(Ypredict ./ 1e3, 'b-d');
% legend ('���H� nHk��', str1, str2, str3)

% legend ('Target ','Forecasted')
% % plot(PeakLoad(1:31),'b:')
% % plot(PeakLoad(699+1:699+31),'g-d')
% title (TargetError);if strcmp (options.Display, 'on')
% end