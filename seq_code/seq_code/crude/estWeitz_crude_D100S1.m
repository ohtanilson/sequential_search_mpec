%ESTIMATION CODE - Crude frequency simulator  - Weitzman, UrsuSeilerHonka 2022
clc; clear all;

tic
%options for estimation
options = optimset('Display', 'final','DiffMinChange',0.05,'FinDiffType','central','FunValCheck','on','MaxFunEvals',6000000,'MaxIter',6000000,'TolX',10^-6,'TolFun',10^-6);

%-------------------------------------------------------
% Setting parameters (obtained from file name)
%-------------------------------------------------------

filename=mfilename;

%number of epsilon+eta draws
D=100%str2num(filename(17:19));

%seed
seed=str2num(filename(21:end));

%-------------------------------------------------------
% Simulation
%-------------------------------------------------------

%simulation inputs
N_cons=1000;%num of consumers
N_prod=5;%num of products
param=[1 0.7 0.5 0.3 -3];%true parameter vector [4 brandFE, search cost constant (exp)]

%simulate data
simWeitz(N_cons, N_prod, param,  seed);
%load simulated data
data=load(sprintf('genWeitzDataS%d.mat',seed));data=cell2mat(struct2cell(data));


%-------------------------------------------------------
% Estimation
%-------------------------------------------------------

%initial parameter vector
param0=zeros(size(param));
%do estimation
tic
[be,val,exitflag,output,grad,hessian]=fminunc(@liklWeitz_crude_1,param0,options,data,D,seed);
toc
%compute standard errors
se=sqrt(diag(inv(hessian)));
se=real(se);

%save results
AS=[be'; se; val; exitflag];
csvwrite(sprintf('rezSimWeitz_crude_D%dS%d.csv',D,seed),AS);


toc