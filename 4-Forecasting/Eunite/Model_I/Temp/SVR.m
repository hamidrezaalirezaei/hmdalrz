%% Initialize SVR Model
function SVRt =SVR(results, options)

C = options.C;
Kernel.arg = options.a;
epsilon = options.epsilon;
Kernel.Kernel  = options.Kernel;
M = options.Dimension;
Display = options.Display;
Solver = options.solver;

MinTrainingLoad = results.Data.MinTrainingLoad;
MaxTrainingLoad = results.Data.MaxTrainingLoad;

ClassLoad = results.clustering.LoadN;
ClassPeakLoad = results.clustering.Data(:,4);
ClassTemp = results.clustering.TempN;
ClassHDay = results.clustering.Data(:,13);
ClassDayType = results.clustering.Data(:,6:12);

LoadMatrix = zeros (size(ClassLoad,1)-(M) , M+1);
for ii = 1:M+1
    LoadMatrix(:,M+2-ii) = ClassLoad(ii:end-(M+1)+ii);
end

TrainingLoad = LoadMatrix(:,2:M+1);
TrainingTarget = LoadMatrix(:,1);
if strcmp (results.Dataset , 'Iran') == 1;
    % No Temprature Data for yet
    TrainingMatrix = [TrainingLoad, results.Data.Data(M+1:end,6:12), results.Data.Data(M+1:end, 13), ];
else
    TrainingMatrix = [TrainingLoad, ClassTemp(M+1:end), ClassDayType(M+1:end,:), ClassHDay(M+1:end), ];
end
% Target Load
% TargetPeakLoad = results.Target.PeakLoad;
% nextNormal = numel(results.Target.Normal .index);
% TargetMatrix = zeros (nextNormal , size(TrainingMatrix,2));

xTrain = (TrainingMatrix);
yTrain = TrainingTarget;...ClassPeakLoad(M+1:end);

x = xTrain;
t = yTrain;

n=numel(t);
x=x';
H = kernel(x, Kernel.Kernel, Kernel.arg);
x = x';
% H=zeros(n,n);
% for i=1:n
%     for j=i:n
%         H(i,j)=CostumeKernel(Kernel, x(i,:),x(j,:));
%         H(j,i)=H(i,j);
%     end
% end

HH=[ H -H
         -H  H];

f=[-t', t']+epsilon;

Aeq=[ones(1,n) -ones(1,n)];
beq=0;

lb=zeros(2*n,1);
ub=C*ones(2*n,1);
Alg{1}='trust-region-reflective';
Alg{2}='interior-point-convex';
Alg{3}='active-set';



if (isempty(Solver)) == 1
    option=optimset('Algorithm',Alg{2},...
    'Display','off',...
    'MaxIter',50);
    [alpha,fval] = quadprog(HH,f,[],[],Aeq,beq,lb,ub,[],option);  
elseif strcmp ( Solver , 'smo')
    option.tolKKT = 0.001; 
    [alpha,fval] = gsmo(HH,f,[],[],Aeq,beq,lb,ub,[],option);
end



alpha=alpha';

AlmostZero=(abs(alpha)<max(abs(alpha))*1e-4);

alpha(AlmostZero)=0;

alpha_plus=alpha(1:n);
alpha_minus=alpha(n+1:end);

eta=alpha_plus-alpha_minus;

S= find( alpha_plus+alpha_minus>0 & alpha_plus+alpha_minus<C);

% Evaluate Training Data
yt=zeros(size(t));
% ytr=zeros(size(t));
% b=zeros(size(t));
for i=1:n
    yt(i)=MySVRFunc(eta(S),H(S,:) , i);
%     b(i)=mean(t(i)-yt(i)-sign(eta(i))'*epsilon);
%     ytr(i)=yt(i)+b(i);
end
b=mean(t(S)-yt(S)+sign(eta(S))'*epsilon);
ytr = yt +b;
% Bn = mean(t-yt-sign(eta)'*epsilon);
% B = sum(b)/n;
% Error Evaluate
residuals = ( t - ytr );

% Squared Error:
SqE_Training = sum (residuals .^ 2);
% Mean Squared Error:
MSE_Training =  (SqE_Training) / n ;
% Root Squared Error:
RSE_Training = SqE_Training ^ 0.5;
% Root Mean Squared Error:
RMSE_Training = MSE_Training ^ 0.5;

Training_Error_Percentage = (abs(residuals)./t) *100;
MAE_Training = mean(abs(Training_Error_Percentage));
MAPE_Training = mean(Training_Error_Percentage(~isinf(Training_Error_Percentage)));
TrainingError = sprintf('Training MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Training, C, Kernel.arg, epsilon);

SVRt.TrainingMatrix = TrainingMatrix(S,:);
SVRt.ninput = size(t,1);
SVRt.inputDim = size(x,2);
SVRt.eta = eta(S);
SVRt.b = b;
SVRt.nsv = numel(S);
SVRt.sv = (S);
SVRt.kernel = Kernel;

SVRt.error.SquaredError = SqE_Training;
SVRt.error.MeanSquaredError = MSE_Training;
SVRt.error.RootSquaredError = RSE_Training;
SVRt.error.RootMeanSquaredError = RMSE_Training;
SVRt.error.MeanAbsoluteError = MAE_Training;
SVRt.error.MeanAbsolutePercentageError = MAPE_Training;
SVRt.TrainingMAPE = MAPE_Training;

if strcmp ( Display , 'on') 
%% Plot Resulst
% Plot Training Results
figure;
plot(t,'k:o');
hold on;
plot(ytr,'r-s');
legend ('actual',' predicted')
title (TrainingError);
end

% end