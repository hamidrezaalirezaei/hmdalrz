function [U_new, dist_new] = U_matrix (mf,K,n,m, ClassNum)
% global m;
% U_new = zeros(data_n,C);

dist_new = Kernel_dist_mat(mf,K, ClassNum);   %  dist_new = [...] n-by-2

num = dist_new .^ ( -1 / (m-1) );
den = sum (dist_new' .^ ( -1 / (m-1) ) );

U_new = zeros(n, ClassNum);
for cc=1:ClassNum
    U_new(:,cc) = num (:,cc) ./ den';
end
% U_new(:,2) = num (:,2) ./ den';

