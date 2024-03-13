%ESTIMATION CODE -Kernel Smoothing - Weitzman, UrsuSeilerHonka 2022
clc; clear all;

tic
%options for estimation
options=optimset('Display','final','DiffMinChange',0.05,'FinDiffType','central','FunValCheck','on','MaxFunEvals',6000000,'MaxIter',6000000,'TolX',10^-6,'TolFun',10^-6);

%-------------------------------------------------------
% Setting parameters (obtained from file name)
%-------------------------------------------------------

filename=mfilename;

%number of epsilon+eta draws
D=str2num(filename(18:20));

%scaling: variables that smooth out the likelihood function; should be
%negative and range typically from -1 to -50; there are 3 of them, one for
%each element in the likelihood function; values need not all be the same;
%if all the same, they can be read from file name
%sl=str2num(filename(22:23));
%scaling= [-sl,-sl,-sl];
scaling = [-18,-4,-7];

%seed
seed=str2num(filename(25:end));

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
[be,val,exitflag,output,grad,hessian]=fminunc(@liklWeitz_kernel_1,param0,options,data,D,scaling,seed);

%compute standard errors
se=sqrt(diag(inv(hessian)));
se=real(se);

%save results
AS=[be'; se; val; exitflag];
csvwrite(sprintf('rezSimWeitz_kernel_D%dW%dS%d.csv',D,-scaling(1),seed),AS);

toc


