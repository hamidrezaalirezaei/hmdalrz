% function MAPE_Target = main(X)
clear all;
clc;

%%
% results = eunite_clustering;
load results;

%%

C = 0.128;
epsilon = 0.010;
a1 = 00.9000;

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
options.Dimension = 7;
options.Display = 'off';
options.solver = [];
%

SVRt = SVR(results, options);
%

[MAPE_Target,Ypredict] = Forecaster(results, SVRt, options);
