n=50; X = rand(n,n); H = X'*X; f = -10*rand(n,1); 
 a=rand(n,1)+0.1; b=rand; tmp = rand(n,1); LB = tmp-1; UB = tmp+1;
 tic; [x1,fval1] = gsmo(H,f,a,b,LB,UB); fval1, toc
 tic; [x2,fval2] = quadprog(H,f,[],[],a',b,LB,UB); fval2, toc