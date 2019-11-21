function [u,final_obj,iter_num, center] = KFCMC(Dataset, N, gamma)
%%%%%%     Kernel Based Fuzzy C-Mean Clustering    %%%%%%%
% 
% Inputs:
%                Dataset : [ N x M ]  matrix of Input Features
%                                              N : length on each Feature Vector
%                                              M : Number of Feature Vectors
%                                              feature Vector: Xi [N x 1] 
%                         N :      number of Class
%                 gamma :      Argumant of RBF Gaussian Kernel
%     
% Outputs:
%                         u :     Fuzzy Mebmership Degree of each Cluster
%      final objective :     Objective of each itteration Clustering Method
%             iter_num :      Number of passed Itterations to reach the stop Criteria
%                 center :      [ 2 x M ] center of each Class
% 
% References: 
%                     1) Zhong-dong Wu, et al, "Fuzzy C-Mean Clustering Algorithm based on kernel Method", 
%                     Proceedings of the Fifth International conference on Computational Intelligence and Multimedia Applications (ICCCIMA'03), 2003.
%                     
%                     2) Xiaowei Yang, et al., "A Kernel Fuzzyc-Means Clustering-Based Fuzzy Support Vector Machine Algorithm for Classification
%                     Problems With Outliers or Noises", IEEE TRANSACTIONS ON FUZZY SYSTEMS, VOL. 19, NO. 1, FEBRUARY 2011.
% 
% Author:     H.R. Alirezaei      ,Autumn  2012.

%% initial Constant Values
% global m; 
m=1.3;
if nargin < 2
    N = 2;
    gamma = 0.25;
elseif nargin < 3    
    gamma = 0.25;
end
max_iter = 100;
e = 1e-5;
% m = 1.3;
x = Dataset;
r = size(x,1);
% l = length(x);
if r == 1
    x = x';
end

data_n = size(x,1);
%% Gaussian Kernel
% GKernel=@(xi,xj) exp(-gamma*(norm(xi-xj)^2) );
Kernel.Kernel = 'rbf';
Kernel.arg = (gamma);
% K=zeros(data_n,data_n);

%%%%% Ensemble of Kernel Matrix
K = kernel(x', Kernel.Kernel, Kernel.arg);

% for ii=1:data_n
%     for j=ii:data_n
%         K(ii,j)=GKernel(x(ii,:),x(j,:));
%         K(j,ii)=K(ii,j);
%     end
% end  

%%
% Distance 
d_2 = @(xi,Vp) (2-2*kernel(xi',Vp', Kernel.Kernel, Kernel.arg) );
% d_2 = @(xi,Vp) (2-2*GKernel(xi,Vp) );
% Initialize Cluster Center

V = rand(N,1);
% V(1) = max(Dataset(1,:));
% V(2) = min(Dataset(1,:));  

u_old = zeros(data_n,N);
for ii=1:data_n
%     den = 0;
%     for p=1:N
%         den =  ( ( 1 / (d_2(x(ii,:), V(p) ) ) )^(1/m-1) ) + den;
%     end
    p = 1:1:N;
    den =  sum( ( ( 1 ./ (d_2(x(ii,:), V(p) ) ) ).^ (1/m-1) ) );
    for j=1:N
        num = (  ( 1 / (d_2(x(ii,:), V(j) ) ) )^(1/m-1) ) ;
        u_old(ii,j) = num / den;
    end 
%     j = 1:1:N;
% 	num =  ( 1 ./ (d_2(x(ii,:), V(j) ) ) ) .^ (1/m-1);
% 	u_old(ii,:) = num ./ den;
end
u=u_old';

% u = rand(N, data_n);
% col_sum = sum(u);
% u = u./col_sum(ones(N, 1), :);

obj_fcn = zeros (max_iter, N);
obj_fin = zeros (max_iter, 1);
for ii=1:max_iter
mf = u.^m;      % MF matrix after exponential modification  2-by-n
[U_new, dist_new] = U_matrix(mf,K,data_n,m,N);    % dist_new = [...]  n-by-2
% obj_fcn(i) = sum(sum(mf*num)) / (data_n*min(min(num)));  % objective function
% obj_fcn(ii,:) = sum(sum((d_2new)*mf));  %#ok<AGROW> % objective function
obj=(abs(u'-U_new));
obj_fcn(ii,:) = max(obj);

obj_fin (ii,1) = sum(sum((dist_new'.^2).*mf));
str = sprintf('  Iteration Number  = %d , achived Objective = % f  ' , ii, obj_fcn(ii,1) );
disp(str);


if ii > 1,
    final_obj = abs(obj_fin(ii-1,1) - obj_fin(ii,1));
    if abs(obj_fcn(ii-1,:) - obj_fcn(ii,:))  < e ;  iter_num = ii; break; end,
end
if ii == 99; break; end;
u = U_new';
mf = mf';
temp = ((ones(size(x, 2), 1)*sum(mf))');
mf = mf';
center = mf*x./temp;
end