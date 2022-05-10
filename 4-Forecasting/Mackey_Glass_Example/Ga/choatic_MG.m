function MAPE_Target = choatic_MG(X)
% clear all;
%%
% Xmg = generate_MackeyGlass(4000,17);
% save('MackeyGlass.mat',  'Xmg');
% mg = Xmg(1000:end);
load MackeyGlass;
mg = Xmg;
next = 100;
train = 500;

% mind=min(mg);
% maxd=max(mg);
% mg = Normalize_Fcn(mg, mind, maxd);

traindata = mg(1:train);
testdata = mg(train+1:train+next);

% [dl,off,lnx,lne,aopt]=window(traindata);

%% Set Options
Kernelmod{1} = 'linear';
Kernelmod{2} = 'poly';
Kernelmod{3} = 'rbf';
Kernelmod{4} = 'wavelet';
Kernelmod{5} = 'sigmoid';

X = [787.532084753389,0.00190227099532228,0.330683050634140;];
% X=[44.2249480737837,0.000536590185395770,0.178889480773670;];
% X = [29.8210778340913,0.000913612348931974,0.151336427591399;];
options.C = X(1);...010.5;
options.epsilon =  X(2);...0.0001;
options.a(1) =  X(3);...0.170;
options.a(2) = 0.5; % second MultiLayer Perceptron parameters (Should be Negative )

options.Kernel = Kernelmod{4};
options.Display = 'no';
Kernel.arg = options.a;
epsilon = options.epsilon;
C = options.C;
Kernel.Kernel  = options.Kernel;
% Display = options.Display;
options.solver = 'smo';
options.tolKKT = 1e-6;...options.epsilon;
    
%%  
x = traindata;
%%%%%%%%%       Mutual Information
% v=ami(x);
% nBins = [ 5000 ];
% nLags = 50;
% [mi corrs]= ami(x,nBins,nLags); 
% for i = 2:50
%     if mi(i) < mi(i-1) && mi(i) < mi(i+1)
%         tao = find(mi == mi(i))
%         break;
%     end
% end
% %%%%%%%%%        FNN
% mmax = 20;
% rtol=15;
% atol=2;
% [FNN] = fnn_deneme(x,tao,mmax,rtol,atol);
% dimFNN = find(FNN == min(FNN) ,1);
% 
% %%%%%%%%%        Cao's Method
% [E1 E2] = cao_deneme(x,tao,mmax);
% for i = 2:mmax-1
%     if E1(i) - E1(i-1) <= 0.5 && E1(i+1) - E1(i) <= 0.5
%         dimCao = find(E1 == E1(i));
%         dimCao = dimCao +1;
%         break;
%     end
% end

%%%%%%%%%       Generalized Kernel Correlation Dimension Method
% [m,d,k,s,gki]=gka(x,1:mmax,tao, [], [], 0);
% c=1;
% for i = 2:mmax-1
%     if d(i) - d(i-1) <= 0.01 && d(i+1) - d(i) <= 0.01
%         dimst(c) = find(d == d(i-1))
%         dimst(c+1) = find(d == d(i+0))
%         dimst(c+2) = find(d == d(i+1))
%        break;
%     end
% end
% 
% EmbeddingRange = [dimst(1), dimst(end)]

% [amis corrs]= ami(x,[2 25],50);    

% dim = dimFNN;
% dim = dimCao+2;
tao = 10;
dim = 5;
options.Dimension = dim;

%% Phase space reconstruction

l = length(x);
lbar = l - (dim +1- 1) * tao ;

Y=zeros(lbar,dim); 
for i=1:dim+1
    Y(:,i)=x((1:lbar)+(i-1)*tao)';
end

xTrain = Y(:,1:dim);
yTrain = Y(:,dim+1);

%%%%%%      Train SVR
svrstruct = svrtrain(xTrain, yTrain, options);
ytr = svrstruct.TrainingPredict;

%%%%%% Target data
lt = numel(testdata);
Ytar=zeros(1,next);
Ytarget=zeros(1,next);
Ynextt=testdata(1:next);

TargetMatrix = zeros(next, dim);

for ii = 0:dim-1
    TargetMatrix(1:(dim-ii)*tao, ii+1) = x(end-(dim-ii)*tao+1:end);
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
    
end

%% Error Evaluate
% residual_Training = abs( ytr - yTrain);
numtr = sum( (yTrain - ytr).^2 );
dentr = sum( (yTrain - mean(yTrain) ).^2 );
% NMSE_Training = ( numtr / dentr );
% MAPE_Training = svrstruct.TrainingMAPE;

Target_residuals = ( Ynextt(1:next) - Ytarget(1:next) );
Target_Error_Percentage = (abs(Target_residuals)./Ynextt(1:next)) *100;
MAE_Target = max(abs(Target_residuals));
MAPE_Target = mean(Target_Error_Percentage(~isinf(Target_Error_Percentage)));
% TargetError = sprintf('Target MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Target, C, Kernel.arg, epsilon);


% Squared Error:
SqE_Target = sum (Target_residuals .^ 2);
% Root Squared Error:
RSE_Target = SqE_Target ^ 0.5;
% Mean Squared Error:
MSE_Target = mean(Target_residuals .^ 2);
% Root Mean Squared Error:
RMSE_Target = MSE_Target ^ 0.5;
% NMSE:
num = sum( (Ynextt - Ytarget).^2 );
den = sum( (Ynextt - mean(Ynextt) ).^2 );
NMSE_Target = num/den;
REP_Target = sqrt( sum( (Target_residuals).^2 ) / sum( (Ynextt).^2 ) ) * 100;

error.MAE = MAE_Target;
error.MAPE = MAPE_Target;
error.SqE = SqE_Target;
error.RSE_Target = RSE_Target;
error.MSE = MSE_Target;
error.RMSE = RMSE_Target;
error.NMSE = NMSE_Target;
error.REP = REP_Target;
% 
% if strcmp (options.Display, 'yes')
disp(error);
%% Plot result
%%%%%%%%%        plot series and attractor
% % Real Attractor with tao = 1
% figure;
% subplot(2,1,1);
% plot(traindata(1:end-1), traindata(2:end));
% axis('tight');
% xlabel('\itX_{t-1}', 'fontsize', 16);
% ylabel('\itX_{t}', 'fontsize', 16);
% title('t°¬ Â§¶ Áow ÂM¼{A JlI]', 'FontName', 'F_lotus Bold', 'fontsize', 20)
% subplot(2,1,2);
% plot(Y(:, 1), Y(:, 2));
% axis('tight');
% str = sprintf('  (¾ÃºIY) %d  ÂºI¶p oÃiáIU IM JlI] ÁpIwpIM ' , tao);
% title(str, 'fontsize', 18, 'FontName', 'F_lotus Bold');
% xlabel('\itX_{t}', 'fontsize', 16);
% ylabel('\itX_{t+\tau}', 'fontsize', 16);
% set(findobj('Type', 'line'), 'Color', 'k' , 'LineWidth', 4);
% set(findobj('Type', 'axes'), 'LineWidth', 3, 'Box', 'on', 'FontName', 'F_lotus Bold', 'fontsize', 18)
% 
% % subplot(2,2,[3,4]);
% figure
% plot(mg(500:1000));
% axis('tight');
% xlabel('(¾ÃºIY)ï·I¶p', 'FontName', 'F_lotus Bold', 'fontsize', 20);
% ylabel('\itX_{t}', 'fontsize', 16);
% title('t°¬ Â§¶ ÂM¼{A ÂºI¶p Áow ', 'FontName', 'F_lotus Bold', 'fontsize', 20)
% set(findobj('Type', 'line'), 'Color', 'k' , 'LineWidth', 4);
% set(findobj('Type', 'axes'), 'LineWidth', 3, 'Box', 'on', 'FontName', 'F_lotus Bold', 'fontsize', 18)
% 
% 
% %% plot forecasted result
% %%%%%%%%%        plot Training results
% % figure;
% % plot(yTrain,'k:o', 'LineWidth',2, 'MarkerSize', 10);
% % hold on;
% % plot(ytr,'r-*','LineWidth',1.5);
% % title(TrainingError);
% 
% %%%%%%%%%        plot Forecasted results
% % figure;
% % plot(Ynextt,'k:o');
% % hold on;
% % plot(Ytarget,'r-s');
% % legend ('Target ','Forecasted')
% % title (TargetError);
% 
% %%%%%%%%%         Plot both training & forecasted value
% strdata = sprintf('Â÷¤H» nHk£¶');
% % strtrain = sprintf('xp¼¶A ÁIõi  = %f %%', MAPE_Training);
% strtest = sprintf('Â¹ÃM yÃQ ÁIõi = %f %%', MAPE_Target);
% xtot = lbar+next;
% 
% figure;
% hold on;
% plot( [yTrain; testdata'] , 'k-');
% plot( 1:lbar, ytr, 'bd')
% plot( lbar+1:xtot, Ytarget' , 'r*');
% % legend(strdata, strtrain, strtest);
% line([lbar, lbar], [0.4, 1.5], 'Color', 'k');
% axis('tight');
% xlabel('(¾ÃºIY)ï·I¶p', 'FontName', 'F_lotus Bold', 'fontsize', 20);
% ylabel('\itX_{t}', 'fontsize', 20);
% title(' "t°¬ Â§¶"ïÂºI¶p Áow Â¹ÃM yÃQ ', 'FontName', 'F_lotus Bold', 'FontSize', 18)
% % set(gca, 'Box', 'on', 'FontName', 'F_lotus Bold', 'FontSize', 20);
% set(findobj('Type', 'line'), 'LineWidth', 2);
% set(findobj('Type', 'axes'), 'Box', 'on', 'FontName', 'F_lotus Bold', 'FontSize', 20, 'LineWidth', 2.5, 'Box', 'on')

%% %%%%%%%%%        plot Mutual Information results
% str = sprintf( 'Â±d¶ ¾¹Ãµ¨ ¸Ã²»H \n ÂºI¶p SMIY = %d', tao);
% figure;
% plot(v1, 'k-*', 'LineWidth', 4)
% line([tao, tao],[0.2 3.5], 'LineStyle', '-.', 'LineWidth', 4, 'Color', 'k');
% axis('tight');
% text(tao, 1.2, ' \leftarrow', 'FontName', 'F_lotus Bold', 'fontsize', 40);
% text(tao+1, 1.2, str, 'FontName', 'F_lotus Bold', 'fontsize', 30);
% ylabel('®MI£T¶ RIø°öH ¸Ã«ºIÃ¶ ', 'FontName', 'F_lotus Bold', 'fontsize', 30);
% xlabel('ÂºI¶p oÃiáIU \tau', 'FontName', 'F_lotus Bold', 'fontsize', 30);
% set(findobj('Type', 'axes'), 'FontName', 'F_lotus Bold', 'fontsize', 40) ; 
% 
% %%%%%%%%%        plot False Nearest Neighbours results
% str = sprintf( ' ¾¹Ã¿M ÂöId¶ k÷M = %d', dimFNN );
% figure;
% plot(1:length(FNN),FNN, 'k-h', 'LineWidth', 4)
% line([dimFNN, dimFNN],[0 max(FNN)], 'LineStyle', '-.', 'LineWidth', 4, 'Color', 'k');
% axis('tight');
% text(dimFNN+0.1, max(FNN)/2, '\leftarrow', 'FontName', 'F_lotus Bold', 'fontsize', 20);
% text(dimFNN+0.5, max(FNN)/2, str, 'FontName', 'F_lotus Bold', 'fontsize', 30)
% % title('Minimum embedding dimension with false nearest neighbours', 'FontName', 'F_lotus Bold', 'fontsize', 20)
% xlabel('ÂöId¶ k÷M ', 'FontName', 'F_lotus Bold', 'fontsize', 30)
% ylabel('JlI¨ ¾ÄIvµÀ ¸ÄoT§Äjqº % ', 'FontName', 'F_lotus Bold', 'fontsize', 30)
% set(gca, 'FontName', 'F_lotus Bold', 'fontsize', 40) ; 

%%%%%%%%%       Plot Generalized Kernel Correlation Dimension Method
% figure;
% plot(m,d,'k-');
% xlabel('ÂöId¶ k÷M ', 'FontName', 'F_lotus Bold', 'fontsize', 20);
% ylabel('Â«TvLµÀ k÷ÔM', 'FontName', 'F_lotus Bold', 'fontsize', 20);
% set(findobj('Type', 'axes'), 'LineWidth', 3, 'Box', 'on', 'FontName', 'F_lotus Bold', 'fontsize', 18)
% set(findobj('Type', 'line'), 'Color', 'k' , 'LineWidth', 4);

%%%%%%%%%        plot Cao's Embedding Dimension results
% str = sprintf( ' ¾¹Ã¿M ÂöId¶ k÷M = %d', dimCao );
% figure;
% hold on;
% plot(1:length(E1),E1,'k');
% plot(1:length(E2),E2, 'k-.');
% lgnd = legend('E1','E2',1);
% line([dimCao, dimCao],[0 E2(dimCao)], 'LineStyle', '-.', 'LineWidth', 3, 'Color', 'k');
% text(dimCao+0.1, E1(dimCao)-0.5, '\leftarrow', 'FontName', 'F_lotus Bold', 'fontsize', 20)
% text(dimCao+0.5, E1(dimCao)-0.5, str, 'FontName', 'F_lotus Bold', 'fontsize', 30)
% xlabel('ÂöId¶ k÷M ', 'FontName', 'F_lotus Bold', 'fontsize', 30)
% ylabel('E1 & E2 ', 'FontName', 'TimesNewRoman_Bold', 'fontsize', 28)
% axis([0, 20 0, 1.5]);
% % grid on;
% % title('"¼GIÃw" x»n ÂöId¶ k÷ÔM', 'FontName', 'F_lotus Bold', 'fontsize', 30)
% set(findobj('Type', 'line'), 'LineWidth', 4);
% set(findobj('Type', 'axes'), 'LineWidth', 3, 'Box', 'on', 'FontName', 'F_lotus Bold', 'fontsize', 40)
% set(lgnd, 'Box', 'off', 'FontName', 'TimesNewRoman', 'fontsize', 28)

% end