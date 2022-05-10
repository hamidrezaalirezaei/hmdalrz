function output = svrforecaster(TargetMatrix, SVRt, options)
%% Set Options:

M = options.Dimension;

Kernel.arg = options.a ;
Kernel.Kernel = options.Kernel;

%% Forecasting
next = size(TargetMatrix,1);
Ytarget=zeros(next, 1);
InputPattern=zeros(next, M);

S = SVRt.sv;
b = SVRt.b;
eta = SVRt.eta;
n=SVRt.nsv;

TrainingMatrix = SVRt.TrainingMatrix;

for ii=1:next
    
%     y=0;
%     for i=1:n
%         y=y+eta(i)*kernel(TrainingMatrix(i,:)', TargetMatrix(ii,:)', Kernel.Kernel, Kernel.arg);
%     end
    i = 1:1:n;
    y = sum( eta(i) * kernel(TrainingMatrix(i,:)', TargetMatrix(ii,:)', Kernel.Kernel, Kernel.arg) );
    Ytarget(ii) = y + b;
    
%     ytar(ii) = svrpredict(TargetMatrix(ii,:)', TrainingMatrix', eta, Kernel);
%     Ytarget(ii) = ytar(ii) + b ;
%     if ii >= 3;
%         load results;
% TargetMatrix(3,1:M) = [results.clustering.class1.LoadN(end-M+4:end)', Ytarget(1:3)' ];
% % 
%         InputPattern(ii+1,2:M) = TargetMatrix(ii,1:M-1);
%     InputPattern(ii+1,1) = Ytarget(ii);
%     TargetMatrix(ii+1,1:M) = InputPattern(ii+1,:);
%     end
% load results;
% TargetMatrix(2,1:M) = [results.clustering.class1.LoadN(end-M+2:end)', Ytarget(1:1) ];
% if ii >= 2
% %     load results;
% %     Input = [results.Data.LoadN(end-M+4:end)', Ytarget(1:3) ];
%     InputPattern(ii+1,2:M) = TargetMatrix(ii,1:M-1);
%     InputPattern(ii+1,1) = Ytarget(ii);
%     TargetMatrix(ii+1,1:M) = InputPattern(ii+1,:);
% end
%     load results;
%     InputPattern(ii,:) = [0, 0,0,0,0,0,0 ];
% end

%     InputPattern(ii+1,2:M) = TargetMatrix(ii+1,1:M-1);
%     InputPattern(ii+1,1) = Ytarget(ii);
%     TargetMatrix(12,1:M) = Ytarget(5:11);
%     end
end

output = Ytarget;
 