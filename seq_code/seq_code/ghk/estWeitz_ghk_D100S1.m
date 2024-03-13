%ESTIMATION CODE -Kernel Smoothing - Weitzman, UrsuSeilerHonka 2022
clc; clear all;

tic
%options for estimation
options = optimset('Display', 'final','DiffMinChange',0.05,'FinDiffType','central','FunValCheck','on','HessUpdate', 'bfgs', 'MaxFunEvals',6000000,'MaxIter',6000000,'TolX',10^-6,'TolFun',10^-6);

%-------------------------------------------------------
% Setting parameters (obtained from file name)
%-------------------------------------------------------

filename=mfilename;

%number of epsilon+eta draws
D=str2num(filename(15:17));

%seed
seed=str2num(filename(19:end));

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
f = @(x)liklWeitz_ghk_1(x, data, D, seed);
%problem has no linear constraints, so set those arguments to []
A = [];
b = [];
Aeq = [];
beq = [];
%upper and lower bounds on the parameters
lb = [-4 -4 -4 -4 -4];
ub = [4 4 4 4 4];
nonlcon = [];
[be,val,exitflag,output] = fminsearchcon(f,param0,lb,ub, A, b, nonlcon, options);
%info available at https://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd-fminsearchcon

%save results
AS=[be'; val; exitflag];
csvwrite(sprintf('rezSimWeitz_ghk_D%dW%dS%d.csv',D,seed),AS);

toc
