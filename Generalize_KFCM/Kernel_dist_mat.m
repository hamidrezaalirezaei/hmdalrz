function out = Kernel_dist_mat(mf,K, Cnum)
% global m;
% mf = [...] Cnum-by-n
den = sum(mf'); %#ok                % den = [...] Cnum-by-1
num1 =zeros(1, size(mf,1));         % num1 = [...] 1-by-Cnum
K_VpVp = size(num1);               % K_VpVp = [...] 1-by-Cnum 
n = size(K,1);                              % K = [...] n-by-n

% Evaluate Kpp  &  Kxp: 
mf = mf';                                   % mf = [...] n-by-Cnum
num2 = zeros(size(mf));             % num2 = [...] n-by-Cnum
K_XiVp = zeros(size(mf));         % K_XiVp = [...] n-by-Cnum

% [A1 B1] = meshgrid(mf(:,1) );
% [A2 B2] = meshgrid(mf(:,2) );
% num1 = [sum(sum(K.*B1.*A1)) , sum(sum(K.*B2.*A2)) ];
% num2 = [sum(K.*B1) ,sum(K.*B2)]

for kk = 1:Cnum
    num1(kk) = sum( sum ( K .* repmat( mf(:, kk) , [1 , n] ) .* repmat( mf(:, kk)', [n, 1] ) ) );
    K_VpVp(kk) = num1(kk) ./ (den(kk).^2);
    
    num2(: , kk) = sum ( K .* repmat( mf(:, kk) , [1 , n] ) );
    K_XiVp(: , kk) = num2(: , kk) ./ den(kk);
end
% K(xi,xi) = 1    &   K(xi, Vp) = [...] n-by-Cnum     &    K(Vp,Vp) = [...] 1-by-Cnum
% dist (x,Vp) = [ K(xi,xi) -2*K(xi, Vp) +K(Vp,Vp) ]  n-by-Cnum
out = ones(n,Cnum) - (2 .* K_XiVp) + (ones(n,1) * K_VpVp );   