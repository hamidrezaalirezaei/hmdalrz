function SVRStruct =svrtrain(TrainingMatrix, TrainingTarget, options)
%%%%%%%%%%%       svrtrain            %%%%%%%%%%%
% Train a e-support vector machine for Regression and function estimate
%
% SVRStruct =svrtrain(TrainingMatrix, TrainingTarget, options)
% 
%%% Inputs: 
%             TrainingMatrix [n x m]   Embedding Matrix [n x dim] 
%             TrainingTarget [n x 1 ]   Single Vertcal Matrix of Training Targets
%             
%             options.XXX :       XXX, specifies one or more of the following Values:
%                                             C:  Penalty Factor of Vapnik's Loss Function
%                                     epsilon:  Accepted error Margin of noisy data
%                                              a: [1 x 2 ] matrix of Kernel function's Arguments
%                                      Kernel: A string specifying the kernel function used 
%                                                 to represent the dot product in a new space.
%                                                  'linear'           ... linear kernel        k(a,b) = a'*b
%                                                  'poly'            ... polynomial         k(a,b) = (a'*b+arg[2])^arg[1]
%                                                  'rbf'              ... RBF (Gaussian) k(a,b) = exp(-0.5*||a-b||^2/arg[1]^2)
%                                                  'wavelet'       ... Wavelet              k(a,b) = prod ( cos(1.75 * (xi-xj) / arg[1] ) * exp(-0.5*||a-b||^2/arg[1]^2) );
%                                                  'sigmoid'      ... Sigmoidal           k(a,b) = tanh(arg[1]*(a'*b)+arg[2])
%                                 Dimension: Embedding Dimmension of systems Dynamic Matrix
%                                        solver:  A string specifying the Method used to Solve
%                                                    the Quadratic Problem of SVM:
%                                                    'smo'          ... Sequental Minimal optimization
%                                                    'qp'            ... quadprog Matlab Function's as Default Solver 
%                                       Display: 'yes' or 'no', Plot results of Training Stage
% 
%%% Outputs: 

%% Set Options
C = options.C;
Kernel.arg = options.a;
epsilon = options.epsilon;
Kernel.Kernel  = options.Kernel;
Display = options.Display;
Solver = options.solver;

%%%%% Set Training data
xTrain = (TrainingMatrix);
yTrain = TrainingTarget;
n=numel(yTrain);

%%%%% Ensemble of Kernel Matrix
H = kernel(xTrain', Kernel.Kernel, Kernel.arg);

% H=zeros(n,n);
% for i=1:n
%     for j=i:n
%         H(i,j)=Kernel.Kernel(Kernel, xTrain(i,:),xTrain(j,:));
%         H(j,i)=H(i,j);
%     end
% end
% 
% HH=[ H -H
%          -H  H];

f=[-yTrain', yTrain']+epsilon;

Aeq=[ones(1,n) -ones(1,n)];
beq=0;

lb=zeros(1, 2*n);
ub=C*ones(1, 2*n);..../n;

Alg{1}='trust-region-reflective';
Alg{2}='interior-point-convex';
Alg{3}='active-set';

%%%%% Solve Lagrange Dual Problem
tic;
if (isempty(Solver)) == 1 ||  strcmp ( Solver , 'qp')
    qpoptions=optimset('Algorithm',Alg{2},...
    'Display','off',...
    'MaxIter',10);
    [alpha,fval] = quadprog(HH,f,[],[],Aeq,beq,lb,ub,[],qpoptions);  
elseif strcmp ( Solver , 'smo')
    option.tolKKT = options.tolKKT; 
    [alpha,fval] = gsmo(HH,f,[],[],Aeq,beq,lb,ub,[],option);
end
% Solvingtime=toc
%%%%% Vector of Lagrange multipliers for the support vectors.
alpha=alpha';

AlmostZero=(abs(alpha)<max(abs(alpha))*1e-4);

alpha(AlmostZero)=0;

alpha_plus=alpha(1:n);
alpha_minus=alpha(n+1:end);


eta=alpha_plus-alpha_minus;

%%%%%  find support vectors
S= find( alpha_plus+alpha_minus > 0 & alpha_plus+alpha_minus <= C+epsilon);

%%%%%   Evaluate Training Data
yt=zeros(size(yTrain));
eta = eta(S);
H = H(S,:);
sv=numel(S);
%%%%%   Evaluate Support Vector Expantion:  w = sum(Bi * K(x , xi) )
for ii=1:n
%     y=0;
%     for i=1:sv
%         y=y+eta(i) * H (i,ii);
%     end
%     yt(ii)=y;
    i = 1:1:sv;
    y = sum(eta(i) * H (i,ii));
    yt(ii) = y;
end
b=mean(yTrain(S)-yt(S)+sign(eta)'*epsilon);
ytr = yt +b;


%%%%%   Error Evaluate
Training_residuals = ( yTrain - ytr );

Training_Error_Percentage = (abs(Training_residuals)./yTrain) *100;
MAE_Training = (sum(abs(Training_residuals))) /n;
MAX_Training = max(abs(Training_residuals));
MAPE_Training = mean(Training_Error_Percentage(~isinf(Training_Error_Percentage)));
PAPE_Training = max(Training_Error_Percentage);

% TrainingError = sprintf('Training MAPE = %f   ', MAPE_Training );
% TrainingError = sprintf('Training MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Training, C, a, epsilon);


% Squared Error:
SqE_Training = sum (Training_residuals .^ 2);
% Root Squared Error:
RSE_Training = SqE_Training ^ 0.5;
% Mean Squared Error:
MSE_Training = mean(Training_residuals .^ 2);
% Root Mean Squared Error:
RMSE_Training = MSE_Training ^ 0.5;
% NMSE Error:
num = sum( (Training_residuals).^2 ) ./ n ;
den = sum( (yTrain - mean(yTrain) ).^2 ) ./ (numel(yTrain) - 1);
NMSE_Training = num/den;
REP_Training = sqrt( sum( (Training_residuals).^2 ) / sum( (yTrain).^2 ) ) * 100;
% 
SVRStruct.error.MAE = MAE_Training;
SVRStruct.error.PAPE = PAPE_Training;
SVRStruct.error.MAX = MAX_Training;
SVRStruct.error.MAPE = MAPE_Training;
SVRStruct.error.SqE = SqE_Training;
SVRStruct.error.RSE_Training = RSE_Training;
SVRStruct.error.MSE = MSE_Training;
SVRStruct.error.RMSE = RMSE_Training;
SVRStruct.error.NMSE = NMSE_Training;
SVRStruct.error.REP = REP_Training;
% TrainingError = sprintf('Training MAPE = %f  C = %f a = %f Epsilon = %f ', MAPE_Training, C, Kernel.arg, epsilon);

%%%%%  SVR Structures:
SVRStruct.TrainingMatrix = TrainingMatrix(S,:);
SVRStruct.ninput = size(yTrain,1);
SVRStruct.inputDim = size(xTrain,2);
SVRStruct.eta = eta;
SVRStruct.b = b;
SVRStruct.nsv = numel(S);
SVRStruct.sv = (S);
SVRStruct.kernel = Kernel;
% 
% SVRStruct.error.SquaredError = SqE_Training;
% SVRStruct.error.MeanSquaredError = MSE_Training;
% SVRStruct.error.RootSquaredError = RSE_Training;
% SVRStruct.error.RootMeanSquaredError = RMSE_Training;
% SVRStruct.error.MeanAbsoluteError = MAE_Training;
% SVRStruct.error.MeanAbsolutePercentageError = MAPE_Training;
% SVRStruct.TrainingMAPE = MAPE_Training;
SVRStruct.TrainingPredict = ytr;

if strcmp ( Display , 'on') 
%% Plot Resulst
% Plot Training Results
figure;
plot(yTrain,'k:o');
hold on;
plot(ytr,'r-s');
legend ('actual',' predicted')
% title (TrainingError);
end