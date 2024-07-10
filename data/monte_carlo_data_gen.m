%ESTIMATION CODE -Kernel Smoothing - Weitzman, UrsuSeilerHonka 2022
clc; clear all;

tic
%options for estimation
% options=optimset('Display','final','DiffMinChange',0.05,'FinDiffType','central','FunValCheck','on','MaxFunEvals',6000000,'MaxIter',6000000,'TolX',10^-6,'TolFun',10^-6);

%-------------------------------------------------------
% create table
%-------------------------------------------------------

% i=1;
% m=[-3.55:0.001:4]';
% for j=-3.55:0.001:4
%     c(i)=(1-normcdf(j))*((normpdf(j)/(1-normcdf(j)))-j);
%     i=i+1;
% end
% c=c';
% 
% 
% table=[m c];
% csvwrite('tableZ.csv',table);


%-------------------------------------------------------
% Simulate data
%-------------------------------------------------------

%simulation inputs
N_cons=1000;%num of consumers
N_prod=5;%num of products
param=[1 0.7 0.5 0.3 -3];%true parameter vector [4 brandFE, search cost constant (exp)]


%% 

for i=1:50
    %seed
    seed= i;
    
    %simulate data
    simWeitz(N_cons, N_prod, param,  seed);
    %csvwrite(sprintf('genWeitzDataS%d.csv',seed),output);
end 
