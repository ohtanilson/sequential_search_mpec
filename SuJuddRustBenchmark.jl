function tic()
    datetimenow = Dates.now()
    global start = Dates.unix2datetime(time())
end
function toc()
    finish = convert(Int, Dates.value(Dates.unix2datetime(time())-start))/1000
    println("Total elapsed time: ", finish, "seconds\n")
end

#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script file runs Monte Carlo experiments reported in Section 5 of
% the paper.
%
% Users first need to specify the true values of structural parameters
% as well as the discount factor in the data generating process.
% Then the program will execute the following steps:
%
% 1) Call AMPL to solve the integrated Bellman equations for the expected
%    value functions and compute the conditional choice probabilities to
%    simulate data of mileage transitions and decisions for 250 data sets;
%
% 2) Estimate the model using the constrained optimization approach with
%    AMPL/KNITRO implementation;
%
% 3) Estimate the model using the constrained optimization approach with
%    MATLAB/ktrlink implementation with first-order analytic derivatives;
%
% 4) Estimate the model using the NFXP algorithm with MATLAB/ktrlink
%    implementation with first-order analytic derivatives
%
% 5) Calculate summary statistics of the Monte Carlo experiments
%
% Source: Su and Judd (2011), Constrained Optimization Approaches to
% Estimation of Structural Models.
% Code Revised: Che-Lin Su, May 2010.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#

# Specify the true values of structural parameters in the data generating
# process. For Monte Carlo experiments with a different the discount factor,
# change the value for beta below.

using CSV
using Random
using Plots
using Distributions
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using Optim, NLSolversBase
mutable struct rust_struct
    beta::Float64;
    nT::Int64;
    nBus::Int64;
    N::Int64;
    M::Int64;
    RC::Float64;
    thetaCost::Float64;
    thetaProbs::Array{Float64,2};
    thetatrue::Array{Float64,2};
    MC::Int64;   # number of monte carlo replications
    multistarts::Int64;
end

println("From RustBusMLETableX_MC.m file in Su and Judd 2012")
function rust_mod(;beta = 0.975,#0.95
                   nT = 120,#1000
                   nBus = 50,
                   N = 175,
                   M = 5,
                   RC = 11.7257,
                   thetaCost = 2.4569,
                   thetaProbs = [0.0937 0.4475 0.4459 0.0127 0.0002 ],
                   MC = 250,  # number of monte carlo replications
                   multistarts = 5
                   )
    #thetatrue = [thetaCost; thetaProbs; RC];
    thetatrue = [thetaCost thetaProbs RC]
    return  param = rust_struct(beta, nT, nBus, N, M, RC, thetaCost, thetaProbs, thetatrue, MC, multistarts)
end
param = rust_mod(beta = 0.975)
#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1)
%%% Call AMPL to solve the integrated Bellman equations for the expected
%%% value functions and compute the conditional choice probabilities to
%%% simulate data of mileage transitions and decisions for 250 data sets;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#
println("Based on RustBusMLETableXSolveEV.mod in Su and Judd 2012")
function RustBusMLETableXSolveEV(param::rust_struct)
    """
    #################################
    # Based on RustBusMLETableXSolveEV.mod
    ##################################
    # input:
    #    param: contains true parameters
    # output:
    #    EV: true expected Value Function
    #    thetaProbs: true transition probability matrix
    #--------------------------------#
    """
    #  Define and process the data
    #param nBus; # number of buses in the data
    #set B := 1..nBus; # B is the index set of buses
    #param nT; # number of periods in the data
    #set T := 1..nT; # T is the vector of time indices
    N = param.N # number of states used in dynamic programming approximation
    nBus = param.nBus # number of buses in the data
    nT = param.nT # number of periods in the data

    # Define the state space used in the dynamic programming part
    X = Vector{Int64}(1:N); # X is the index set of states
    xmin = 0;
    xmax = 175 #100;
    x = Vector{Float64}(X);
    for i in 1:length(X) #  x[i] denotes state i;
        #x[i] = xmin + (xmax-xmin)/(N-1)*(i-1);
        x[i] = i
    end
    # Parameters and definition of transition process
    # In this example, M = 5: the bus mileage reading in the next period can either stay in current state or can move up to 4 states
    M = param.M;
    # Define discount factor. We fix beta since it can't be identified
    beta = param.beta;   # discount factor
    model = JuMP.Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=300.0))
    # Data: (xt, dt)
    ###################
    # dt, xt are given in MC_dt, MC_xt
    ###################
    @variable(model, dt[t=1:nT, b =1:nBus]); # decision of bus b at time t
    @variable(model, xt[t=1:nT, b =1:nBus]); # mileage (state) of bus b at time t
    # END OF MODEL and DATA SETUP #

    # DEFINING STRUCTURAL PARAMETERS and ENDOGENOUS VARIABLES TO BE SOLVED #
    # Parameters for (linear) cost function
    #    c(x, thetaCost) = 0.001*thetaCost*x ;
    #@variable(model, thetaCost[i = 1:2] >=0) # quadratic cost
    # thetaProbs[i] defines transition probability that mileage in next period moves up (i-1) grid point. M=5 in this example.
    # Replacement cost
    # Define true structural parameter values
    @variable(model, RC)
    @variable(model, thetaCost);
    @variable(model, thetaProbs[i=1:M]>=0);
    # TRUE PARAMETERS
    @constraint(model, RC==param.RC) # true
    @constraint(model, thetaCost==param.thetaCost) # true
    for i in 1:M
        @constraint(model, thetaProbs[i]==param.thetaProbs[i]) # true
    end
    # DECLARE EQUILIBRIUM CONSTRAINT VARIABLES
    # The NLP approach requires us to solve equilibrium constraint variables
    @variable(model, EV[i = 1:N], start = -50)       	# Expected Value Function of each state
    # END OF DEFINING STRUCTURAL PARAMETERS AND ENDOGENOUS VARIABLES

    #  DECLARE AUXILIARY VARIABLES  #
    #  Define auxiliary variables to economize on expressions
    #  Create Cost variable to represent the cost function;
    #  Cost[i] is the cost of regular maintenance at x[i].
    @variable(model,  Cost[i=1:N] >= 0)
    #  Let CbEV[i] represent - Cost[i] + beta*EV[i];
    #  this is the expected payoff at x[i] if regular maintenance is chosen
    @variable(model,  CbEV[i=1:N] )
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2)); #quadratic
        @NLconstraint(model, Cost[i] == 0.001*thetaCost*x[i]);
        @NLconstraint(model, CbEV[i] == - Cost[i] + beta*EV[i]);
    end

    #  Let PayoffDiff[i] represent -CbEV[i] - RC + CbEV[1];
    #  this is the difference in expected payoff at x[i] between engine replacement and regular maintenance
    @variable(model,  PayoffDiff[i=1:N])
    #  Let ProbRegMaint[i] represent 1/(1+exp(PayoffDiff[i]));
    #  this is the probability of performing regular maintenance at state x[i];
    @variable(model,  1>=ProbRegMaint[i=1:N]>=0)
    for i in 1:N
        @NLconstraint(model, PayoffDiff[i] == -CbEV[i] - RC + CbEV[1]);
        @NLconstraint(model, ProbRegMaint[i] ==  1/(1+exp(PayoffDiff[i])));
    end
    # BellmanViola represents violation of the Bellman equations.
    #var BellmanViola {i in 1..(N-M+1)} = sum {j in 0..(M-1)} log(exp(CbEV[i+j])+
    #exp(-RC + CbEV[1]))* thetaProbs[j+1] - EV[i];
    @variable(model, BellmanViola[i=1:convert(Int64, (N-M+1))])
    for i in 1:convert(Int64, (N-M+1))
        @NLconstraint(model, BellmanViola[i]
              == sum(log(exp(CbEV[i+(j-1)]) + exp(-RC + CbEV[1]))* thetaProbs[j] - EV[i] for j in 1:(convert(Int64,M))));
    end
    #  END OF DECLARING AUXILIARY VARIABLES #

    # DEFINE OBJECTIVE FUNCTION AND CONSTRAINTS
    # Since we are solving only for EV, we use 0 as the objective function.
    JuMP.@NLobjective(model, Max,  0)
    #  Bellman equation for states below N-M
    #Bellman_1toNminusM {i in X: i <= N-(M-1)}:
    #EV[i] = sum {j in 0..(M-1)}
    #log(exp(CbEV[i+j])+ exp(-RC + CbEV[1]))*thetaProbs[j+1];
    for i in 1:(convert(Int64,N-(M-1)))
        @NLconstraint(model, EV[i] == sum(log(exp(CbEV[i+j-1])
              + exp(-RC + CbEV[1]))* thetaProbs[j] for j in 1:(convert(Int64,M))));
    end
    #  Bellman equation for states above N-M, (we adjust transition probabilities to keep state in [xmin, xmax])
    for i in (convert(Int64,N-M)):(N-1)
        @NLconstraint(model, EV[i] == sum(log(exp(CbEV[i+j-1])+ exp(-RC + CbEV[1]))* thetaProbs[j] for j in 1:(convert(Int64,N-i)))
                                   + (1 - sum(thetaProbs[k] for k in 1:(convert(Int64,N-i)))) * log(exp(CbEV[N])+ exp(-RC + CbEV[1]))
                                   )
    end
    #  Bellman equation for state N
    @NLconstraint(model, EV[N] == log(exp(CbEV[N])+ exp(-RC + CbEV[1])));
    #  The probability parameters in transition process must add to one
    @NLconstraint(model, sum(thetaProbs[i] for i in 1:M) == 1)
    #  Put bound on EV; this should not bind, but is a cautionary step to help keep algorithm within bounds
    for i in 1:N
        @NLconstraint(model, EV[i] <= 500)
    end
    # END OF DEFINING OBJECTIVE FUNCTION AND CONSTRAINTS
    # DEFINE THE OPTIMIZATION PROBLEM #
    @time JuMP.optimize!(model)
    println("EV ",JuMP.value.(EV))
    println("RC ",JuMP.value.(RC))
    println("thetaCost ",JuMP.value.(thetaCost))
    println("thetaProbs",JuMP.value.(thetaProbs))
    println("objvalue= ", JuMP.objective_value(model))
    println("Locally optima? = ", termination_status(model))
    EV = JuMP.value.(EV)
    thetaProbs = JuMP.value.(thetaProbs)
    return EV, thetaProbs
end
#@time EV, thetaProbs = RustBusMLETableXSolveEV(param::rust_struct)

function EV_MPEC_check(param::rust_struct)
    @time EV, thetaProbs = RustBusMLETableXSolveEV(param::rust_struct)
    EV = EV[:,1]
    fig = Plots.plot(EV, label = "EV_simulated_by_MPEC", title = "EV_MPEC_check")
    ####################
    # EV from Conlon's data
    ####################
    EV_conlon = CSV.read("EV.sol", datarow=1)
    EV_conlon = EV_conlon[:,1]
    fig = Plots.plot!(EV_conlon,label = "EV_conlon's_file",color ="red", line = (:dashdot))
    @show fig
    savefig(fig,"EV_MPEC_check")
end
#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulate Data for 250 date sets -- (state, decision) = (xt,dt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#
println("The following is the adjusted section of the code from Su and Judd RustBusMLETableX_MC.m")
function simdata(param::rust_struct, EV::Array{Float64,1})
    #%The following is the adjusted section of the code from Su and Judd RustBusMLETableX_MC.m
    MC = param.MC;
    beta = param.beta;
    nT = param.nT;
    nBus = param.nBus;
    N = param.N;
    M = param.M;
    RC = param.RC;
    thetaCost = param.thetaCost;
    thetaProbs = param.thetaProbs;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%% Simulate Data for 250 date sets -- (state, decision) = (xt,dt)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = (1:N)';
    #P0 = 1./ (1 + exp( 0.001*thetaCost*x - beta.*EV - RC - 0.001*thetaCost*x(1)+ beta*EV(1)));
    #global x = Vector{Float64}(1:N);
    x = Vector{Float64}(1:N);
    P0 = 1 ./ (1 .+ exp.( 0.001.*thetaCost.*x .- beta.*EV .- RC .- 0.001.*thetaCost.*x[1].+ beta.*EV[1]));

    Random.seed!(1)
    MC_xt = zeros(nT, nBus, MC);
    MC_dt = zeros(nT, nBus, MC);
    MC_dx = zeros(nT, nBus, MC);
    for kk = 1:MC
        #Rx  = unifrnd(0, 1, nT, nBus);
        #Rd  = unifrnd(0, 1, nT, nBus);
        Rx  = rand(Distributions.Uniform(0,1),nT, nBus)
        Rd  = rand(Distributions.Uniform(0,1),nT, nBus)
        xt = ones(Int64, nT, nBus);
        dx = zeros(nT, nBus);
        dt = zeros(nT, nBus);
        #cumTransProb = cumsum(thetaProbs);
        cumTransProb = cumsum(thetaProbs, dims=2);
        for t = 1:nT
            #dt(t,:) = (Rd(t,:) >= P0(xt(t,:))');
            dt[t,:] = (Rd[t,:] .>= P0[xt[t,:]]);
            for i = 1:nBus
                #dx(t,i) = find(Rx(t,i) < cumTransProb,1);
                dx[t,i] = findfirst((Rx[t,i] .< cumTransProb)[1,:])
                if t < nT
                    if dt[t,i] == 1
                       xt[t+1,i] = 1 + dx[t,i]-1;
                    else
                      xt[t+1,i] = min(xt[t,i] + dx[t,i]-1, N);
                    end
                end
            end
        end
        MC_dt[:,:,kk] = dt;
        MC_xt[:,:,kk] = xt;
        MC_dx[:,:,kk] = dx;
    end
    return MC_dt, MC_xt, MC_dx
end
@time MC_dt, MC_xt, MC_dx = simdata(param, EV)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%% Setup Optimization Problem for MPEC
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setup_for_MPEC()
    #%Construct the sparsity pattern of the constraint Jacobian and Hessian
    N = param.N;
    M = param.M;
    beta = param.beta;
    #PayoffDiffPrime = zeros(7+N,length(x));
    PayoffDiffPrime = zeros(7+N,length(x));
    #PayoffDiffPrime(1,:)=0.001*(x'-repmat(x(1),1,length(x)));
    PayoffDiffPrime[1,:]=0.001.*(x'.-repeat([x[1]],1,length(x)));
    #PayoffDiffPrime(7,:)=-1;
    PayoffDiffPrime[7,:]=repeat([-1], 1, length(PayoffDiffPrime[7,:]));
    #PayoffDiffPrime(8,:)= beta;
    PayoffDiffPrime[8,:]= repeat([beta], 1, length(PayoffDiffPrime[8,:]));
    #PayoffDiffPrime = -beta*[zeros(7,N); eye(N)] + PayoffDiffPrime;
    PayoffDiffPrime = -beta*[zeros(7,N); LinearAlgebra.Diagonal(ones(N))] + PayoffDiffPrime;
    #CbEVPrime = zeros(length(x),7+N);
    CbEVPrime = zeros(length(x),7+N);
    #CbEVPrime(:,1)=-0.001*x;
    CbEVPrime[:,1]=-0.001*x;
    #CbEVPrime= beta*[zeros(N,7) eye(N)] + CbEVPrime;
    CbEVPrime = beta*[zeros(N,7) LinearAlgebra.Diagonal(ones(N))] + CbEVPrime;
    #CbEVPrime =  [ CbEVPrime; repmat(CbEVPrime(length(x),:),M,1)];
    CbEVPrime =  vcat(CbEVPrime, repeat(reshape(CbEVPrime[length(x),:], length(CbEVPrime[length(x),:]), 1) ,1,M)')

    #TransProbPrime = zeros(7+N,M);
    TransProbPrime = zeros(7+N,M);
    #TransProbPrime(2:6,:) = eye(M);
    TransProbPrime[2:6,:] = LinearAlgebra.Diagonal(ones(M));
    #RPrime = zeros(1,8-1+N);
    RPrime = zeros(1,8-1+N);
    #RPrime(:,7)=-1;
    RPrime[:,7] = fill!(RPrime[:,7], -1)
    #RPrime(:,8)=beta;
    RPrime[:,8] = fill!(RPrime[:,8], beta)
    #indices = repmat((1:N)',1,M)+repmat((1:M),N,1)-1;
    indices = repeat((1:N),1,M).+repeat((1:M)',N,1).-1;
    #d1 = ((CbEVPrime.*repmat(ones(N+M,1),1,N+7) + repmat(RPrime,N+M,1)))./(repmat(ones(N+M,1),1,N+7));
    d1 = ((CbEVPrime.*repeat(ones(N+M,1),1,N+7) + repeat(RPrime,N+M,1)))./(repeat(ones(N+M,1),1,N+7));
    #sum1 =  reshape(sum(reshape(d1(indices',:) .* repmat(repmat(ones(M,1),N,1),1,N+7),M,N, N+7 )),N, N+7);
    sum1 =  reshape(sum(reshape(d1[vec(indices'),:] .* repeat(repeat(ones(M,1),N,1),1,N+7),M,N, N+7 ), dims=1),N, N+7);
    #sum2 = ones(N,M)*TransProbPrime';
    sum2 = ones(N,M)*TransProbPrime';
    #EVPrime = [zeros(N,7) eye(N)];
    EVPrime = [zeros(N,7) LinearAlgebra.Diagonal(ones(N))];
    #JacobSpaPattern = (sum1 + sum2 - EVPrime);
    JacobSpaPattern = (sum1 + sum2 - EVPrime);
    #JacobSpaPattern = ceil.(abs.(JacobSpaPattern))./max(ceil.(abs.(JacobSpaPattern)),dims=1);
    #JacobSpaPattern = ceil.(abs.(JacobSpaPattern))./max(ceil.(abs.(JacobSpaPattern)),dims=1);

    #HessSpaPattern = ones(8,7+N);
    HessSpaPattern = ones(8,7+N);
    #HessSpaPattern = [HessSpaPattern; ones(N-1,8) eye(N-1)];
    HessSpaPattern = [HessSpaPattern; ones(N-1,8) LinearAlgebra.Diagonal(ones(N-1))];


    #%Define upper and lower bounds for the following decision varialbes:
    #%thetaCost,thetaProbs, and EV

    #%thetaCost
    lb = zeros(N+7,1);
    ub = zeros(N+7,1);
    lb[1]=0;
    ub[1]=1e-16;
    #%thetaProbs
    lb[2:6] = fill!(lb[2:6],0);
    ub[2:6] = fill!(ub[2:6],1);
    #%RC
    lb[7]=0;
    ub[7]=1e-16;

    #%EV
    #%Put bound on EV; this should not bind, but is a cautionary step to help keep algorithm within bounds
    #%lb(7:(7-1+N))=0;
    lb[8:end] = fill!(lb[8:end],-1e-1);
    #ub(8:end)=50;
    ub[8:end] = fill!(ub[8:end],50);
    #%The probability parameters in transition process must add to one
    #%Linear constraint : sum thetaProbs = 1
    Aeq = zeros(1,N+7);
    Aeq[2:6] = fill!(Aeq[2:6],1);
    beq=1;
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%% Setup Optimization Problem for NFXP
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setup_for_NFXP()
    #%Define upper and lower bounds for structural parameters:
    #%thetaCost
    #NFXPlb = zeros(7,1);
    NFXPlb = zeros(7,1);
    #NFXPub = zeros(7,1);
    NFXPub = zeros(7,1);
    #NFXPlb(1)=0;
    NFXPlb[1]=0;
    #NFXPub(1)=inf;
    NFXPub[1]=1e-16;
    #%thetaProbs
    #NFXPlb(2:6)=0;
    NFXPlb[2:6] = fill!(NFXPlb[2:6],0);
    #NFXPub(2:6)=1;
    NFXPub[2:6] = fill!(NFXPub[2:6],1);
    #NFXPlb(7)=0;
    NFXPlb[7]=0;
    #NFXPub(7)=inf;
    NFXPlb[7]=1e-16;
    #%The probability parameters in transition process must add to one
    #%Linear constraint : sum thetaProbs = 1

    #NFXPAeq = zeros(1, length(thetatrue));
    thetatrue = param.thetatrue
    NFXPAeq = zeros(1, length(thetatrue));
    #NFXPAeq(2:6)=1;
    NFXPAeq[2:6] = fill!(NFXPAeq[2:6],1);
    #NFXPbeq=1;
    NFXPbeq=1
end
#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 2)
%%% Estimate the model using the constrained optimization approach with
%%% AMPL/KNITRO implementation;
%%%
%%% IMPLEMENTATION 1: MPEC/AMPL
%%% We implement the MPEC approach using AMPL modeling language
%%% with KNITRO as the solver.
%%% AMPL supplies first-order and second-order analytical derivatives
%%% and the sparsity pattern of the constraint Jacobian and the Hessian
%%% to the solver.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#
#%Generating starting Points
println("enerating starting Points")
X0 = zeros(7+param.N,param.multistarts); #why +param.N?
X0[1,:]= (1:1:param.multistarts);
X0[2:6,:] = fill!(X0[2:6,:],1/param.M);
X0[7,:]= (4:1:4+(param.multistarts-1));

function RustBusMLETableX(param::rust_struct,
                          MC_dt, MC_xt, MC_dx,
                          kk, reps)
    #  Define and process the data
    N = param.N # number of states used in dynamic programming approximation
    nBus = param.nBus # number of buses in the data
    global X0 # initial value list
    # Define the state space used in the dynamic programming part
    nT = param.nT # number of periods in the data
    X = Vector{Int64}(1:N); # X is the index set of states
    xmin = 0;
    xmax = 175 #100;
    x = Vector{Float64}(X);
    for i in 1:length(X) #  x[i] denotes state i;
        #x[i] = xmin + (xmax-xmin)/(N-1)*(i-1);
        x[i] = i
    end
    # Parameters and definition of transition process
    # In this example, M = 5: the bus mileage reading in the next period can either stay in current state or can move up to 4 states
    M = param.M;
    # Define discount factor. We fix beta since it can't be identified
    beta = param.beta;   # discount factor
    model = JuMP.Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0))
    # Data: (xt, dt)
    #@variable(model, dt[t=1:nT, b =1:nBus]); # decision of bus b at time t
    #@variable(model, xt[t=1:nT, b =1:nBus]); # mileage (state) of bus b at time t
    # END OF MODEL and DATA SETUP #

    # DEFINING STRUCTURAL PARAMETERS and ENDOGENOUS VARIABLES TO BE SOLVED #
    # Parameters for (linear) cost function
    #    c(x, thetaCost) = 0.001*thetaCost*x ;
    @variable(model, thetaCost >=0, start = X0[1,reps]) # quadratic cost
    # thetaProbs[i] defines transition probability that mileage in next period moves up (i-1) grid point. M=5 in this example.
    @variable(model, 1>=thetaProbs[i = 1:M] >= 0.00001) # try another initial valu
    # Replacement cost
    #RC = m.RC
    @variable(model, RC >= 0, start = X0[7,reps])
    # Define true structural parameter values
    #=
    @constraint(model, RC==param.RC)
    @constraint(model, thetaCost==param.thetaCost)
    for i in 1:M
        @constraint(model, thetaProbs[i]==param.thetaProbs[i])
    end
    =#
    # DECLARE EQUILIBRIUM CONSTRAINT VARIABLES
    # The NLP approach requires us to solve equilibrium constraint variables
    @variable(model, EV[i = 1:N], start = X0[8,reps])       	# Expected Value Function of each state
    # END OF DEFINING STRUCTURAL PARAMETERS AND ENDOGENOUS VARIABLES

    #  DECLARE AUXILIARY VARIABLES  #
    #  Define auxiliary variables to economize on expressions

    #  Create Cost variable to represent the cost function;
    #  Cost[i] is the cost of regular maintenance at x[i].
    @variable(model,  Cost[i=1:N] >= 0)
    #  Let CbEV[i] represent - Cost[i] + beta*EV[i];
    #  this is the expected payoff at x[i] if regular maintenance is chosen
    @variable(model,  CbEV[i=1:N] )#NOT >=0 !!! it is infeasible!!
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2));
        @NLconstraint(model, Cost[i] == 0.001*thetaCost*x[i]);
        @NLconstraint(model, CbEV[i] == - Cost[i] + beta*EV[i]);
    end

    #  Let PayoffDiff[i] represent -CbEV[i] - RC + CbEV[1];
    #  this is the difference in expected payoff at x[i] between engine replacement and regular maintenance
    @variable(model,  PayoffDiff[i=1:N])
    #  Let ProbRegMaint[i] represent 1/(1+exp(PayoffDiff[i]));
    #  this is the probability of performing regular maintenance at state x[i];
    @variable(model,  1>=ProbRegMaint[i=1:N]>=0)
    for i in 1:N
        @NLconstraint(model, PayoffDiff[i] == -CbEV[i] - RC + CbEV[1]);
        @NLconstraint(model, ProbRegMaint[i] ==  1/(1+exp(PayoffDiff[i])));
    end
    # BellmanViola represents violation of the Bellman equations.
    @variable(model, BellmanViola[i=1:convert(Int64, (N-M+1))])
    for i in 1:convert(Int64, (N-M+1))
        @NLconstraint(model, BellmanViola[i] == sum(log(exp(CbEV[i+(j-1)]) + exp(-RC + CbEV[1]))* thetaProbs[j] - EV[i] for j in 1:(convert(Int64,M))));
    end
    #  END OF DECLARING AUXILIARY VARIABLES #
    ############################################################
    # DEFINE OBJECTIVE FUNCTION AND CONSTRAINTS
    # second term will attempt access out of the lengh of xt
    # so we modify some operation for the state transition
    #############################################################
    # Note that we use MPEC version of likelihood formula
    #sum {t in 2..nT, b in B} log(dt[t,b]*(1-ProbRegMaint[xt[t,b]])
    #                                + (1-dt[t,b])*ProbRegMaint[xt[t,b]])
    @NLexpression(model, RegMaint_term, sum(sum(log(dt[t,b]*(1-ProbRegMaint[convert(Int64,xt[t,b])])
          + (1-dt[t,b])*ProbRegMaint[convert(Int64,xt[t,b])]) for t=2:nT) for b=1:nBus)
                                     )
    #+ sum {t in 2..nT, b in B} log(dt[t-1,b]*(thetaProbs[xt[t,b]-1+1])
    #                               + (1-dt[t-1,b])*(thetaProbs[xt[t,b]-xt[t-1,b]+1]));
    @NLexpression(model, thetaProbs_term, sum(sum(log(dt[t-1,b]*(thetaProbs[convert(Int64, min(xt[t,b],4)-1+1)]) #thetaProbs[xt[t,b]-1+1]
          + (1-dt[t-1,b])*(thetaProbs[convert(Int64, max(xt[t,b]-xt[t-1,b],0)+1)])) for t=2:nT) for b=1:nBus)
                                     )
    JuMP.@NLobjective(model, Max, RegMaint_term + thetaProbs_term)
    #=
    f = 0; % value of the likelihood function
    g = zeros(length(X),1); % gradient of the likelihood funciton
    xt2 = xt(2:nT,:);
    xt1 = xt(1:nT-1,:);
    dtMinus = (dt(2:nT,:)==0);
    dtPlus = (dt(2:nT,:)==1);
    dtM1Minus = (dt(1:nT-1,:)==0);
    dtM1Plus = (dt(1:nT-1,:)==1);
    #% Constracut the value of the likelihood function
    f1 = 1-ProbRegMaint(xt2(dtPlus));
    f2 = ProbRegMaint(xt2(dtMinus));
    f3 = TransProb( xt2( dtM1Plus ));
    f4 = TransProb(xt2(dtM1Minus) - xt1(dtM1Minus)+1);
    f = -( sum(log(f1))+ sum(log(f2))+ sum(log(f3))+ sum(log(f4)));
    =#
    #=
    # original version of likelihood but incorrect.
    f = 0; #% value of the likelihood function
    #g = zeros(length(X),1); #% gradient of the likelihood funciton
    xt2 = xt[2:nT,:];
    xt1 = xt[1:nT-1,:];
    dtMinus = dt[2:nT,:].==0;
    dtPlus = dt[2:nT,:].==1;
    dtM1Minus = dt[1:nT-1,:].==0;
    dtM1Plus = dt[1:nT-1,:].==1;
    #% Constracut the value of the likelihood function
    @NLexpression(model, f1, ProbRegMaint[Vector{Int64}(xt2[dtPlus])])
    @NLexpression(model, f2, ProbRegMaint[Vector{Int64}(xt2[dtMinus])])
    @NLexpression(model, f3, thetaProbs[Vector{Int64}(xt2[dtM1Plus])])
    @NLexpression(model, f4, thetaProbs[Vector{Int64}(xt2[dtM1Minus] - xt1[dtM1Minus].+1)])
    @NLexpression(model, f, -( sum(log(1-f1))+ sum(log(f2))+ sum(log(f3))+ sum(log(f4))) )
    =#
    println("omit gradient g")
    #f1 = ProbRegMaint[Vector{Int64}(xt2[dtPlus])];
    #f2 = ProbRegMaint[Vector{Int64}(xt2[dtMinus])];
    #f3 = thetaProbs[Vector{Int64}(xt2[dtM1Plus])];
    #f4 = thetaProbs[Vector{Int64}(xt2[dtM1Minus] - xt1[dtM1Minus].+1)];
    #f = -( sum(log(1-f1))+ sum(log(f2))+ sum(log(f3))+ sum(log(f4)));
    #  Bellman equation for states below N-M
    for i in 1:(convert(Int64,N-(M-1)))
        @NLconstraint(model, EV[i] == sum(log(exp(CbEV[i+j-1])+ exp(-RC + CbEV[1]))* thetaProbs[j] for j in 1:(convert(Int64,M))));
    end
    #  Bellman equation for states above N-M, (we adjust transition probabilities to keep state in [xmin, xmax])
    for i in (convert(Int64,N-M)):(N-1)
        @NLconstraint(model, EV[i] == sum(log(exp(CbEV[i+j-1])+ exp(-RC + CbEV[1]))* thetaProbs[j] for j in 1:(convert(Int64,N-i)))
                                   + (1 - sum(thetaProbs[k] for k in 1:(convert(Int64,N-i)))) * log(exp(CbEV[N])+ exp(-RC + CbEV[1]))
                                   )
    end
    #  Bellman equation for state N
    @NLconstraint(model, EV[N] == log(exp(CbEV[N])+ exp(-RC + CbEV[1])));
    #  The probability parameters in transition process must add to one
    #@NLconstraint(model, sum(thetaProbs[i] for i in 1:3) == 1)
    #  Put bound on EV; this should not bind, but is a cautionary step to help keep algorithm within bounds
    for i in 1:N
        @NLconstraint(model, EV[i] <= 500)
    end
    # END OF DEFINING OBJECTIVE FUNCTION AND CONSTRAINTS
    # DEFINE THE OPTIMIZATION PROBLEM #
    @time JuMP.optimize!(model)
    println("EV ", JuMP.value.(EV))
    println("RC ", JuMP.value.(RC))
    println("thetaCost ", JuMP.value.(thetaCost))
    println("thetaProbs", JuMP.value.(thetaProbs))
    println("objvalue= ", JuMP.objective_value(model))
    println("Locally optima? = ", termination_status(model))
    return JuMP.value.(thetaCost),JuMP.value.(RC), JuMP.value.(EV), JuMP.value.(thetaProbs),JuMP.objective_value(model)
end

kk = 1 # index of monte carlo data
dt = MC_dt[:,:,kk]
xt = MC_xt[:,:,kk]
dx = MC_dx[:,:,kk]
test_reps = 1
@time thetaCost_MPEC, RC_MPEC, EV_MPEC, theta_Probs_MPEC, objval_MPEC = RustBusMLETableX(param, dt, xt, dx, kk, test_reps)
println("Estimated parameters:(thetaCost,[thetaProbs],RC) =\n ", thetaCost_MPEC, theta_Probs_MPEC,RC_MPEC)
println("true parameters: ", param.thetatrue)

function plot_MLE_MPEC(;grid_ind)
    temp = zeros(Float64,grid_ind*grid_ind,3)
    @time for i = 1:grid_ind
        for j = 1:grid_ind
            theta_list = [0:1:grid_ind;]
            theta_start = convert(Array{Float64,1},[i, j])
            param.thetaCost = theta_list[i]
            param.RC = theta_list[j]
            @time global  EV, thetaProbs = RustBusMLETableXSolveEV(param::rust_struct)
            @time global  MC_dt, MC_xt, MC_dx = simdata(param, EV)
            kk = 1 # index of monte carlo data
            global dt = MC_dt[:,:,kk]
            global xt = MC_xt[:,:,kk]
            global dx = MC_dx[:,:,kk]
            test_reps = 1
            temp[i+(j-1)*grid_ind,1] = theta_list[i]
            temp[i+(j-1)*grid_ind,2] = theta_list[j]
            ~, ~, ~, ~, objval_MPEC= RustBusMLETableX(param, dt, xt, dx, kk, test_reps)
            temp[i+(j-1)*grid_ind,3] = objval_MPEC
        end
    end
    global temp
    temp_plot = Plots.plot(xlabel = "thetaCost", ylabel ="RC" )
    #temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
    temp_plot = Plots.plot!(temp[:,1] ,temp[:,2], -temp[:,3], st=:surface)
    temp_plot = Plots.plot!(title="full_MLE_obj_MPEC")
    return temp_plot
end
#=
@time temp_plot = plot_MLE_MPEC(grid_ind=15)
@show temp_plot
savefig(temp_plot,"full_MLE_obj_MPEC")
open("temp_plot_MPEC.txt", "w") do f
    write(f, "$temp", "\n")
end
open("temp_plot_MPEC.txt", "r") do f
    temp=read(f)
end
temp_plot = Plots.plot(xlabel = "thetaCost", ylabel ="RC" )
#temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
temp_plot = Plots.plot!(temp[:,1] ,temp[:,2], -temp[:,3], st=:surface)
temp_plot = Plots.plot!(title="full_MLE_obj_MPEC")
savefig(temp_plot,"full_MLE_obj_MPEC")
=#

# multiple monte carlo
function MPEC_MC(param::rust_struct;
                 test_MC = 1,
                 test_multistarts = 5)
    MC = param.MC
    multistarts = param.multistarts
    N = param.N
    M = param.M
    # Restore the spaces
    KnitroExitAMPL = -10000*ones(MC,1);
    objvalAMPL = zeros(1,MC, multistarts);
    thetaCostAMPL = zeros(1,MC, multistarts);
    RCAMPL = zeros(1, MC, multistarts);
    EVAMPL = zeros(N, MC, multistarts);
    thetaProbsAMPL = zeros(M,MC, multistarts);
    SolveTimeAMPL = zeros(MC,1);
    ObjValAMPL = zeros(MC,1);
    IterAMPL = zeros(MC,1);
    FunEvalAMPL = zeros(MC,1);
    SuccessAMPL = zeros(MC,1);
    tic()
    for k = 1:test_MC
        for reps = 1:test_multistarts
            #fprintf('This is Monte Carlo run #%d out of %d\n \n', kk, MC);
            dt = MC_dt[:,:,k]
            xt = MC_xt[:,:,k]
            dx = MC_dx[:,:,k]
            @time thetaCostAMPL[1, k, reps], RCAMPL[1,k, reps],EVAMPL[:,k, reps], thetaProbsAMPL[:,k,reps],objvalAMPL[1,k,reps] = RustBusMLETableX(param, dt, xt, dx, k, reps)
            #@time sol_all = RustBusMLETableX(param, dt, xt, dx, k, reps)
            println("iteration counter: $k \n", "multistarts counter: $reps" )
        end
    end
    toc()
    open("MC$(MC)_beta$(param.beta)_thetaCostAMPL.txt", "w") do f
        write(f, "$thetaCostAMPL", "\n")
    end
    open("MC$(MC)_beta$(param.beta)_thetaCostAMPL.txt", "r") do f
        test_thetaCostAMPL=read(f)
    end

    open("MC$(MC)_beta$(param.beta)_RCAMPL.txt", "w") do f
        write(f, "$RCAMPL", "\n")
    end
    open("MC$(MC)_beta$(param.beta)_EVAMPL.txt", "w") do f
        write(f, "$EVAMPL", "\n")
    end
    open("MC$(MC)_beta$(param.beta)_thetaProbsAMPL.txt", "w") do f
        write(f, "$thetaProbsAMPL", "\n")
    end
    # Total elapsed time: 46516.19seconds
    println("SKIP [status,result] = system(strAmplSystemCall);")
end
#@time MPEC_MC(param,test_MC = 1,test_multistarts = 1) #5
#7.472057 seconds (13.33 M allocations: 319.549 MiB, 3.60% gc time)

#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Step 3)
    %%% Estimate the model using the constrained optimization approach with
    %%%  MATLAB/ktrlink implementation with first-order analytic derivatives;
    %%%
    %%% IMPLEMENTATION 2: MPEC/MATLAB
    %%% We implement the MPEC approach using MATLAB programming language
    %%% with KNITRO (ktrlink) as the solver.
    %%% We provide first-order analytical derivatives and sparsity pattern
    %%% of the constraint Jacobian to the solver.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#
println("Skip MATLAB/ktrlink implementation first-order analytical derivatives and sparsity pattern")

#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 4)
%%% Estimate the model using the NFXP algorithm with MATLAB/ktrlink
%%% implementation with first-order analytic derivatives
%%%
%%% IMPLEMENTATION 3: NFXP/MATLAB
%%% We implement the NFXP algorithm using MATLAB programming language
%%% with KNITRO (ktrlink) as the solver.
%%% We provide first-order analytical derivatives to the solver.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#
####################
# Reset EV from Conlon's data
####################
param = rust_mod(beta = 0.975)
EV = CSV.read("EV.sol", datarow=1)
EV = EV[:,1]
G = -1.0e10;
#for reps = 1:multistarts
reps = 1
#fprintf('Running NFXP in Monte Carlo run #%d out of %d replications \n', kk, MC);
#disp(['Running Starting Point #' num2str(reps)])
#t2 = cputime;
println("ktrlink(@likelihoodNFXP,theta0,[],[],NFXPAeq,NFXPbeq,NFXPlb,NFXPub,[],ktroptsNFXP,'knitroOptions.opt');")
#[thetaNFXP_reps fvalNFXP_reps flagNFXP_reps outputNFXP] = ktrlink(@likelihoodNFXP,theta0,[],[],NFXPAeq,NFXPbeq,NFXPlb,NFXPub,[],ktroptsNFXP,'knitroOptions.opt');
#=
tNFXP_reps = cputime - t2;
tNFXPsol(kk) = tNFXPsol(kk) + tNFXP_reps;
IterNFXPsol(kk) = IterNFXPsol(kk) + outputNFXP.iterations;
FunEvalNFXPsol(kk) = FunEvalNFXPsol(kk) + outputNFXP.funcCount;
numBellEvalsol(kk) = numBellEvalsol(kk) + BellEval;
=#
##################################
# define BeellmanContract and loglikelihood function for NFP
##################################
X = Vector{Int64}(1:param.N); # X is the index set of states
x = Vector{Float64}(1:param.N);
for i in 1:length(X) #  x[i] denotes state i;
    #x[i] = xmin + (xmax-xmin)/(N-1)*(i-1);
    x[i] = i
end
EVold = zeros(param.N,1);
indices = repeat((1:param.N),1,param.M).+repeat((1:param.M)',param.N,1).-1;

function BellContract_SA(param::rust_struct, thetaCost::Float64, TransProb::Array{Float64,2}, RC::Float64, tol_inner::Float64;inner_tol_max=500)
    #=
    % This m-file solves the integrated Bellman equation using constraction
    % mapping iteration in the NFXP algorithm.
    %
    % source: Su and Judd (2011), Constrained Optimization Approaches to
    % Estimation of Structural Models.
    % Code Revised: Che-Lin Su, May 2010.
    =#
    N = param.N
    M = param.M
    RC = param.RC
    beta = param.beta
    global x, indices;
    global BellEval;
    EV0 = zeros(N,1);;
    #% The cost function in Table X is: c(x) = 0.001*theta_1*x
    #Cost = (0.001*thetaCost.*x)' ;
    Cost = (0.001*thetaCost.*x) ;
    ii = 0;
    norm = 1;
    CbEV = Array{Float64,1}(undef,N+M)
    #################
    # SA contraction start
    #################
    while norm > tol_inner
        #println("Bellman evaluation count = ",ii)
        #% Let CbEV[i] represent - Cost[i] + beta*EV[i];
        #% this is the expected payoff at x[i] if regular maintenance is chosen
        #CbEV = - Cost + beta*EV0;
        CbEV[1:N] = - Cost .+ beta*EV0;
        #CbEV(N+1:N+M)=CbEV(N);
        CbEV[N+1:N+M] = fill!(CbEV[N+1:N+M],CbEV[N]);
        s1 = exp.(CbEV[indices]);
        s2 = exp.(-RC+CbEV[1]);
        s = s1 .+ s2;
        logs = log.(s);
        #EV = logs * TransProb;
        EV = logs * TransProb'; # updated EV
        # sup-norm
        global BellmanViola = abs.(EV - EV0);
        norm = maximum(BellmanViola);
        EV0 = EV;
        ii = ii+ 1;
        if inner_tol_max < ii
            break
        end
    end
    println("Bellman evaluation count = ",ii)
    #% BellEval is the number of contraction mapping iterations needed to solve
    #% the Bellman equation.
    #EVold = EV0;
    EVold = EV0;
    EVsol = EVold
    CbEVsol = - Cost .+ beta*EVsol;
    BellEval = ii + BellEval;
    u0 = -Cost
	u1 = -RC -Cost[1]
    Vkeep = u0 .+ beta .* EVsol
	Vchange = u1 .+ beta .* EVsol[1]
    pk = 1 ./ (1 .+exp.(Vchange.-Vkeep) )
    println("Uniform convergence (supnorm EV)= ",maximum(BellmanViola))
    println("The number of evaluation loops = ",BellEval)
    return EVsol, CbEVsol, norm, pk
end
tol_inner = 1.e-10;
BellEval = 0; #initialize
EV, CbEV, norm_SA, pk = BellContract_SA(param, param.thetaCost, param.thetaProbs, param.RC, tol_inner,inner_tol_max=1000)
EV = EV[:,1]
#reps = 1
#theta0 = X0[1:7,reps]; # initial value
function EV_NFXPSA_check()
    EV_conlon = CSV.read("EV.sol", datarow=1)
    EV_conlon = EV_conlon[:,1]
    fig = Plots.plot(EV_conlon, label = "EV_conlon's_file(true)",color ="red", title = "EV_NFXPSA_check_tol_inner")
    for i = 1:6
        tol_inner_list = [1.e-1, 1.e-2, 1.e-4, 1.e-6,1.e-8,1.e-10]
        @show tol_inner = tol_inner_list[i]
        #global tol_inner
        EV_NFXP, CbEV_NFXP, norm_SA_NFXP, pk_NFXP = BellContract_SA(param, param.thetaCost, param.thetaProbs, param.RC, tol_inner)
        EV_NFXP = EV_NFXP[:,1]
        fig = Plots.plot!(EV_NFXP,line = (:dashdot), label = "EV_simulated_by_NFXPSA_tol_inner_$(tol_inner)")
    end
    return fig
end
@show fig = EV_NFXPSA_check()
savefig(fig,"EV_NFXPSA_check")
#@NLconstraint(model, Cost[i] == 0.001*thetaCost*x[i]);
#@NLconstraint(model, CbEV[i] == - Cost[i] + beta*EV[i]);
#test_CbEV = - 0.001*sol_all[1]*x .+ beta.*sol_all[3]

println("likelihoodNFXP_fullMLE.m
we do not supply second-order analytic derivatives. Hence, the hessian of the likelihood function, h, is empty []. ")
#function [f, gNFXP, h] = likelihoodNFXP(theta)
#PayoffDiffPrime = zeros(7+N,length(x));
PayoffDiffPrime = zeros(7+N,length(x));
#PayoffDiffPrime(1,:)=0.001*(x'-repmat(x(1),1,length(x)));
PayoffDiffPrime[1,:]=0.001.*(x'.-repeat([x[1]],1,length(x)));
#PayoffDiffPrime(7,:)=-1;
PayoffDiffPrime[7,:]=repeat([-1], 1, length(PayoffDiffPrime[7,:]));
#PayoffDiffPrime(8,:)= beta;
PayoffDiffPrime[8,:]= repeat([beta], 1, length(PayoffDiffPrime[8,:]));
#PayoffDiffPrime = -beta*[zeros(7,N); eye(N)] + PayoffDiffPrime;
PayoffDiffPrime = -beta*[zeros(7,N); LinearAlgebra.Diagonal(ones(N))] + PayoffDiffPrime;
#CbEVPrime = zeros(length(x),7+N);
CbEVPrime = zeros(length(x),7+N);
#CbEVPrime(:,1)=-0.001*x;
CbEVPrime[:,1]=-0.001*x;
#CbEVPrime= beta*[zeros(N,7) eye(N)] + CbEVPrime;
CbEVPrime = beta*[zeros(N,7) LinearAlgebra.Diagonal(ones(N))] + CbEVPrime;
#CbEVPrime =  [ CbEVPrime; repmat(CbEVPrime(length(x),:),M,1)];
CbEVPrime =  vcat(CbEVPrime, repeat(reshape(CbEVPrime[length(x),:], length(CbEVPrime[length(x),:]), 1) ,1,M)')

#TransProbPrime = zeros(7+N,M);
TransProbPrime = zeros(7+N,M);
#TransProbPrime(2:6,:) = eye(M);
TransProbPrime[2:6,:] = LinearAlgebra.Diagonal(ones(M));
#RPrime = zeros(1,8-1+N);
RPrime = zeros(1,8-1+N);
#RPrime(:,7)=-1;
RPrime[:,7] = fill!(RPrime[:,7], -1)
#RPrime(:,8)=beta;
RPrime[:,8] = fill!(RPrime[:,8], beta)
#indices = repmat((1:N)',1,M)+repmat((1:M),N,1)-1;
indices = repeat((1:N),1,M).+repeat((1:M)',N,1).-1;
#d1 = ((CbEVPrime.*repmat(ones(N+M,1),1,N+7) + repmat(RPrime,N+M,1)))./(repmat(ones(N+M,1),1,N+7));
d1 = ((CbEVPrime.*repeat(ones(N+M,1),1,N+7) + repeat(RPrime,N+M,1)))./(repeat(ones(N+M,1),1,N+7));
#sum1 =  reshape(sum(reshape(d1(indices',:) .* repmat(repmat(ones(M,1),N,1),1,N+7),M,N, N+7 )),N, N+7);
sum1 =  reshape(sum(reshape(d1[vec(indices'),:] .* repeat(repeat(ones(M,1),N,1),1,N+7),M,N, N+7 ), dims=1),N, N+7);
#sum2 = ones(N,M)*TransProbPrime';
sum2 = ones(N,M)*TransProbPrime';
#EVPrime = [zeros(N,7) eye(N)];
EVPrime = [zeros(N,7) LinearAlgebra.Diagonal(ones(N))];
#JacobSpaPattern = (sum1 + sum2 - EVPrime);
JacobSpaPattern = (sum1 + sum2 - EVPrime);
#JacobSpaPattern = ceil.(abs.(JacobSpaPattern))./max(ceil.(abs.(JacobSpaPattern)),dims=1);
#JacobSpaPattern = ceil.(abs.(JacobSpaPattern))./max(ceil.(abs.(JacobSpaPattern)),dims=1);

#HessSpaPattern = ones(8,7+N);
HessSpaPattern = ones(8,7+N);
#HessSpaPattern = [HessSpaPattern; ones(N-1,8) eye(N-1)];
HessSpaPattern = [HessSpaPattern; ones(N-1,8) LinearAlgebra.Diagonal(ones(N-1))];


#%Define upper and lower bounds for the following decision varialbes:
#%thetaCost,thetaProbs, and EV

#%thetaCost

lb = zeros(param.N+7,1);
ub = zeros(param.N+7,1);
lb[1]=0;
ub[1]=1e-16;
#%thetaProbs
lb[2:6] = fill!(lb[2:6],0);
ub[2:6] = fill!(ub[2:6],1);
#%RC
lb[7]=0;
ub[7]=1e-16;

#%EV
#%Put bound on EV; this should not bind, but is a cautionary step to help keep algorithm within bounds
#%lb(7:(7-1+N))=0;
lb[8:end] = fill!(lb[8:end],-1e-1);
#ub(8:end)=50;
ub[8:end] = fill!(ub[8:end],50);
#%The probability parameters in transition process must add to one
#%Linear constraint : sum thetaProbs = 1
Aeq = zeros(1,param.N+7);
Aeq[2:6] = fill!(Aeq[2:6],1);
beq=1;

function likelihoodNFXP_fullMLE(param::rust_struct, theta)
    #=
    % This m-file computes the value and the gradient of the likelihood
    % function in the NFXP algorithm of the Harold Zurcher bus-engine replacement model.
    % f is the value of the likelihood function evaluated at the structural parameter vector theta.
    % g is the gradient of the likelihood function evaluated at the structural parameter vector theta.
    % In this implementation, we do not supply second-order analytic
    % derivatives. Hence, the hessian of the likelihood function, h, is empty [].
    %
    % source: Su and Judd (2011), Constrained Optimization Approaches to
    % Estimation of Structural Models.
    % Code Revised: Che-Lin Su, May 2010.
    =#
    #global dt xt nT nBus N M M1 M2 PayoffDiffPrime TransProbPrime beta CbEVPrime indices;
    global dt, xt, nT, indices
    N = param.N
    M = param.M
    nT = param.nT
    beta = param.beta
    nBus = param.nBus
    #global M1, M2,
    global PayoffDiffPrime, TransProbPrime, CbEVPrime;
    #% Define parameters for cost function
    thetaCost = theta[1];
    #% Define parameters and definition of transition process
    #% thetaProbs defines Markov chain
    thetaProbs = theta[2:6];
    #TransProb = thetaProbs;
    TransProb =  Array{Float64,2}(undef,1,5)
    TransProb[1,:] = thetaProbs;
    #% Define replacement cost parameter
    RC = theta[7];
    #ntheta = length(theta);
    ntheta = length(theta);
    #% Use constration mapping iteration to solve the integrated Bellman equations
    #[EV, CbEV] = BellContract_SA(thetaCost, TransProb, RC);
    EV, CbEV, norm_SA_NFXP, pk = BellContract_SA(param, thetaCost, TransProb, RC, tol_inner);
    #% Let PayoffDiff[i] represent -CbEV[i] - RC + CbEV[1];
    #% this is the difference in expected payoff at x[i] between engine replacement and regular maintenance
    #PayoffDiff = -CbEV - RC + CbEV(1);
    PayoffDiff = -CbEV .- RC .+ CbEV[1];
    #% Let ProbRegMaint[i] represent 1/(1+exp(PayoffDiff[i]));
    #% this is the probability of performing regular maintenance at state x[i];
    #ProbRegMaint = 1./(1+exp(PayoffDiff));
    ProbRegMaint = 1 ./(1 .+exp.(PayoffDiff));
    #=
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% OBJECTIVE AND CONSTRAINT DEFINITIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define objective function: Likelihood function
    % The likelihood function contains two pieces
    % First is the likelihood that the engine is replaced given time t state in the
    data.
    % Second is the likelihood that the observed transition between t-1 and t
    % would have occurred.
    =#
    f = 0;
    g = zeros(length(theta)+N,1);
    gPayoffPrime = g;
    gTransProbPrime = g;
    dtM1Minus = [];
    dtM1Plus = [];
    dtMinus = [];
    dtPlus = [];
    #for i = 1:nBus
    #=
    for i = 1:(nBus-1)
        #i=30 #test
        #dtM1Minus = find(dt(1:(nT-1),i)==0);
        dtM1Minus = findall(dt[1:(nT-1),i].==0)
        #dtM1Plus = find(dt(1:(nT-1),i)==1);
        dtM1Plus = findall(dt[1:(nT-1),i].==1);
        #dtMinus = find(dt((2:nT),i)==0)+1;
        dtMinus = findall(dt[(2:nT),i].==0).+1;
        #dtPlus = find(dt((2:nT),i)==1)+1;
        dtPlus = findall(dt[(2:nT),i].==1).+1;
        #ProbRegMaint(xt(dtPlus,i));
        #ProbRegMaint(xt(dtMinus,i));
        #TransProb( xt( dtM1Plus+1,i ) );
        #TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1);
        #= Following parts is no meaning on Matlab??
        ProbRegMaint[convert(Vector{Int64},xt[dtPlus,i])];
        ProbRegMaint[convert(Vector{Int64},xt[dtMinus,i])];
        TransProb[ convert(Vector{Int64},xt[dtM1Plus+1,i ]) ];
        TransProb[convert(Vector{Int64},xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1)];
        =#
        #% Compute the value of the likelihood function
        #=
        f = f -( sum( log( 1-ProbRegMaint(xt(dtPlus,i))))...
        + sum( log( ProbRegMaint(xt(dtMinus,i)))) ...
        + sum( log( TransProb( xt( dtM1Plus +1,i ) ) ))...
        + sum( log( TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1))) );
        =#
        f1 = sum( log.( 1 .-ProbRegMaint[convert(Vector{Int64}, xt[dtPlus,i])]))
        f2 = sum( log.( ProbRegMaint[convert(Vector{Int64}, xt[dtMinus,i])]))
        f3 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Plus.+1 ,i ]) ] ))
        f4 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Minus.+1, i]-xt[dtM1Minus,i].+1)]))
        f = f - ( f1 + f2 + f3 + f4 );
        println(f)
        #% Compute the gradient of the likelihood function
        #println("Compute the gradient of the likelihood function")
        #=
        #if nargout > 1
            #d1 = PayoffDiffPrime(:,xt(dtPlus,i))*ProbRegMaint(xt(dtPlus,i)) ;
            d1 = PayoffDiffPrime[:,convert(Int64,xt[dtPlus,i])]*ProbRegMaint[xt[dtPlus,i]] ;
            #d2 = - PayoffDiffPrime(:,xt(dtMinus,i))*( 1-ProbRegMaint(xt(dtMinus,i)));
            d2 = - PayoffDiffPrime[:,convert(Int64,xt[dtMinus,i])]*( 1-ProbRegMaint[xt[dtMinus,i]]);
            #d3 = TransProbPrime(:, xt( dtM1Plus +1,i ))*(1./TransProb( xt( dtM1Plus+1,i ) ));
            d3 = TransProbPrime[:, convert(Int64,xt[dtM1Plus+1,i])]*(1 /TransProb[xt[dtM1Plus+1,i ]]);
            #d4 = TransProbPrime(:,xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1)*(1./TransProb( xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1 ));
            d4 = TransProbPrime[:,convert(Int64,xt[dtM1Minus+1,i])-xt[dtM1Minus,i]+1]*(1 /TransProb[xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1 ]);
            gPayoffPrime = gPayoffPrime - (d1+d2);
            gTransProbPrime = gTransProbPrime - (d3+d4);
        #end
        =#
    end
    =#
    # transformed version from MPEC
    Cost = zeros(N)
    CbEV = zeros(N)
    PayoffDiff = zeros(N)
    ProbRegMaint = zeros(N)
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2));
        Cost[i] = 0.001*thetaCost*x[i]
        CbEV[i] = - Cost[i] + beta*EV[i]
    end
    for i in 1:N
        PayoffDiff[i] = -CbEV[i] - RC + CbEV[1];
        ProbRegMaint[i] =  1/(1+exp(PayoffDiff[i]));
    end
    RegMaint_term = sum(sum(log(dt[t,b]*(1-ProbRegMaint[convert(Int64,xt[t,b])])
          + (1-dt[t,b])*ProbRegMaint[convert(Int64,xt[t,b])]) for t=2:nT) for b=1:nBus)
    #+ sum {t in 2..nT, b in B} log(dt[t-1,b]*(thetaProbs[xt[t,b]-1+1])
    #                               + (1-dt[t-1,b])*(thetaProbs[xt[t,b]-xt[t-1,b]+1]));
    thetaProbs_term = sum(sum(log(dt[t-1,b]*(thetaProbs[convert(Int64, min(xt[t,b],4)-1+1)]) #thetaProbs[xt[t,b]-1+1]
          + (1-dt[t-1,b])*(thetaProbs[convert(Int64, max(xt[t,b]-xt[t-1,b],0)+1)])) for t=2:nT) for b=1:nBus)
    f = RegMaint_term + thetaProbs_term
    println("Continue to compute the gradient of the likelihood function\n")
    println("No hessian\n")
    println("Likelihood : ", f)
    #% Continue to compute the gradient of the likelihood function
    #=
    #if nargout > 1
        #gPayoffPrimetheta = gPayoffPrime(1:ntheta);
        gPayoffPrimetheta = gPayoffPrime[1:ntheta];
        #gPayoffPrimeEV = gPayoffPrime(ntheta+1:end);
        gPayoffPrimeEV = gPayoffPrime[ntheta+1:end];
        #gTransProbPrimetheta = gTransProbPrime(1:ntheta);
        gTransProbPrimetheta = gTransProbPrime[1:ntheta];
        s1 = exp(CbEV(indices));
        s2 = exp(-RC+CbEV(1));
        s = s1 + s2;
        logs = log(s);
        Rprime = zeros(1,ntheta+N);
        Rprime(ntheta)=-1;
        Rprime(ntheta+1)=beta;
        d1 = ((CbEVPrime.*repmat(exp(CbEV),1,ntheta + N) + exp(-RC+CbEV(1))*repmat(Rprime,N+M,1)))./(repmat(exp(CbEV)+exp(-RC+CbEV(1)),1,ntheta + N));
        sum1 = reshape(sum(reshape(d1(indices',:) .*
        repmat(repmat(TransProb,N,1),1,ntheta + N),M,N, ntheta + N )),N,ntheta + N);
        sum2 = logs*TransProbPrime';
        TPrime = sum1 + sum2;
        EVPrime = [zeros(N,ntheta) eye(N)];
        dTdtheta = TPrime(:,1:ntheta);
        dTdEV = TPrime(:,(ntheta+1):ntheta+N);
        gNFXP = dTdtheta'*(inv(eye(N)-dTdEV))'*gPayoffPrimeEV + gPayoffPrimetheta +gTransProbPrimetheta;
    #end
    if nargout > 2
        h=[];
    end
    =#
    #return f, gNFXP, h
    return f
end
println("Total maintanance on data sum(dt) =  ", sum(dt))
###############
# construct full-MLE NFP problem
###############
@time EV, thetaProbs = RustBusMLETableXSolveEV(param::rust_struct)
@time MC_dt, MC_xt, MC_dx = simdata(param, EV)
kk = 1 # index of monte carlo data
dt = MC_dt[:,:,kk]
xt = MC_xt[:,:,kk]
dx = MC_dx[:,:,kk]


f = likelihoodNFXP_fullMLE(param, param.thetatrue)
BellEval = 0
function neg_likelihoodNFXP_fullMLE(param,theta)
    neg_f = - likelihoodNFXP_fullMLE(param, theta)
    return neg_f
end
neg_f = neg_likelihoodNFXP_fullMLE(param, param.thetatrue)
@time thetaCost_MPEC, RC_MPEC, EV_MPEC, theta_Probs_MPEC, objval_MPEC = RustBusMLETableX(param, dt, xt, dx, kk, test_reps)
@show neg_f_MPEC = neg_likelihoodNFXP_fullMLE(param, vcat(thetaCost_MPEC, theta_Probs_MPEC, RC_MPEC))
@show objval_MPEC

f = neg_likelihoodNFXP_fullMLE(param, vcat(thetaCost_MPEC, theta_Probs_MPEC, RC_MPEC))
grid_ind = 15
temp = zeros(Float64,grid_ind*grid_ind,3)
@time for i = 1:grid_ind
	for j = 1:grid_ind
		theta_list = [0:1:15;]
		theta_start = convert(Array{Float64,1},[i, j])
		theta_start[1] = theta_list[i]
		theta_start[2] = theta_list[j]
		temp[i+(j-1)*grid_ind,1] = theta_start[1]
		temp[i+(j-1)*grid_ind,2] = theta_start[2]
		temp[i+(j-1)*grid_ind,3] = neg_likelihoodNFXP_fullMLE(param,vcat(theta_start[1], theta_Probs_MPEC, theta_start[2]))
	end
end
#5.680927 seconds (3.97 M allocations: 7.097 GiB, 22.12%
temp_plot = Plots.plot(xlabel = "thetaCost", ylabel ="RC" )
#temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
temp_plot = Plots.plot!(temp[:,1] ,temp[:,2], -temp[:,3], st=:surface)
temp_plot = Plots.plot!(title="full_MLE_obj_NFXP")
@show temp_plot



###############
# construct partial-MLE NFP problem
###############
function likelihoodNFXP_partialMLE(param::rust_struct, thetaCost::Float64, RC::Float64, firststage_thetaProbs::Array{Float64,2})
    #=
    % This m-file computes the value and the gradient of the likelihood
    % function in the NFXP algorithm of the Harold Zurcher bus-engine replacement model.
    % f is the value of the likelihood function evaluated at the structural parameter vector theta.
    % g is the gradient of the likelihood function evaluated at the structural parameter vector theta.
    % In this implementation, we do not supply second-order analytic
    % derivatives. Hence, the hessian of the likelihood function, h, is empty [].
    %
    % source: Su and Judd (2011), Constrained Optimization Approaches to
    % Estimation of Structural Models.
    % Code Revised: Che-Lin Su, May 2010.
    =#
    #global dt xt nT nBus N M M1 M2 PayoffDiffPrime TransProbPrime beta CbEVPrime indices;
    global dt, xt, nT, indices
    N = param.N
    M = param.M
    nT = param.nT
    beta = param.beta
    nBus = param.nBus
    #global M1, M2,
    global PayoffDiffPrime, TransProbPrime, CbEVPrime;
    firststage_thetaProbs # 1st estimates
    #% Define parameters for cost function
    #thetaCost = theta(1);
    #thetaCost = theta[1];
    #% Define parameters and definition of transition process
    #% thetaProbs defines Markov chain
    #thetaProbs = theta(2:6);
    thetaProbs = firststage_thetaProbs # 1st estimates
    #TransProb = thetaProbs;
    TransProb =  Array{Float64,2}(undef,1,5)
    TransProb[1,:] = thetaProbs;
    #% Define replacement cost parameter
    #RC = theta(7);
    #RC = theta[7];
    #ntheta = length(theta);
    theta = param.thetatrue
    ntheta = length(theta);
    #% Use constration mapping iteration to solve the integrated Bellman equations
    #[EV, CbEV] = BellContract_SA(thetaCost, TransProb, RC);
    EV, CbEV, norm_SA, pk = BellContract_SA(param, thetaCost, TransProb, RC, tol_inner);
    #% Let PayoffDiff[i] represent -CbEV[i] - RC + CbEV[1];
    #% this is the difference in expected payoff at x[i] between engine replacement and regular maintenance
    #PayoffDiff = -CbEV - RC + CbEV(1);
    PayoffDiff = -CbEV .- RC .+ CbEV[1];
    #% Let ProbRegMaint[i] represent 1/(1+exp(PayoffDiff[i]));
    #% this is the probability of performing regular maintenance at state x[i];
    #ProbRegMaint = 1./(1+exp(PayoffDiff));
    ProbRegMaint = 1 ./(1 .+exp.(PayoffDiff));
    #=
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% OBJECTIVE AND CONSTRAINT DEFINITIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define objective function: Likelihood function
    % The likelihood function contains two pieces
    % First is the likelihood that the engine is replaced given time t state in the
    data.
    % Second is the likelihood that the observed transition between t-1 and t
    % would have occurred.
    =#
    #f = 0;
    f = 0;
    #g = zeros(length(theta)+N,1);
    g = zeros(length(theta)+N,1);
    #gPayoffPrime = g;
    gPayoffPrime = g;
    #gTransProbPrime = g;
    gTransProbPrime = g;
    #dtM1Minus = [];
    dtM1Minus = [];
    #dtM1Plus = [];
    dtM1Plus = [];
    #dtMinus = [];
    dtMinus = [];
    #dtPlus = [];
    dtPlus = [];
    #for i = 1:nBus
    #=
    for i = 1:(nBus-1)
        #i=30 #test
        #dtM1Minus = find(dt(1:(nT-1),i)==0);
        dtM1Minus = findall(dt[1:(nT-1),i].==0)
        #dtM1Plus = find(dt(1:(nT-1),i)==1);
        dtM1Plus = findall(dt[1:(nT-1),i].==1);
        #dtMinus = find(dt((2:nT),i)==0)+1;
        dtMinus = findall(dt[(2:nT),i].==0).+1;
        #dtPlus = find(dt((2:nT),i)==1)+1;
        dtPlus = findall(dt[(2:nT),i].==1).+1;
        #ProbRegMaint(xt(dtPlus,i));
        #ProbRegMaint(xt(dtMinus,i));
        #TransProb( xt( dtM1Plus+1,i ) );
        #TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1);
        #= Following parts is no meaning on Matlab??
        ProbRegMaint[convert(Int64,xt[dtPlus,i])];
        ProbRegMaint[convert(Int64,xt[dtMinus,i])];
        TransProb[ convert(Int64,xt[dtM1Plus+1,i ]) ];
        TransProb[convert(Int64,xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1)];
        =#
        #% Compute the value of the likelihood function
        #=
        f = f -( sum( log( 1-ProbRegMaint(xt(dtPlus,i))))...
        + sum( log( ProbRegMaint(xt(dtMinus,i)))) ...
        + sum( log( TransProb( xt( dtM1Plus +1,i ) ) ))...
        + sum( log( TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1))) );
        =#
        f1 = sum( log.( 1 .-ProbRegMaint[convert(Vector{Int64},xt[dtPlus,i])]))
        f2 = sum( log.( ProbRegMaint[convert(Vector{Int64},xt[dtMinus,i])]))
        f3 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Plus.+1 ,i ]) ] ))
        f4 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Minus.+1, i]-xt[dtM1Minus,i].+1)]))
        f = f - ( f1 + f2 + f3 + f4 );
        println(f)
        if f == NaN
            break
        end
        #% Compute the gradient of the likelihood function
        println("Compute the gradient of the likelihood function")
        #if nargout > 1
            #d1 = PayoffDiffPrime(:,xt(dtPlus,i))*ProbRegMaint(xt(dtPlus,i)) ;
            d1 = PayoffDiffPrime[:,convert(Int64,xt[dtPlus,i])]*ProbRegMaint[xt[dtPlus,i]] ;
            #d2 = - PayoffDiffPrime(:,xt(dtMinus,i))*( 1-ProbRegMaint(xt(dtMinus,i)));
            d2 = - PayoffDiffPrime[:,convert(Int64,xt[dtMinus,i])]*( 1-ProbRegMaint[xt[dtMinus,i]]);
            #d3 = TransProbPrime(:, xt( dtM1Plus +1,i ))*(1./TransProb( xt( dtM1Plus+1,i ) ));
            d3 = TransProbPrime[:, convert(Int64,xt[dtM1Plus +1,i])]*(1./TransProb[xt[dtM1Plus+1,i ]]);
            #d4 = TransProbPrime(:,xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1)*(1./TransProb( xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1 ));
            d4 = TransProbPrime[:,xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1]*(1./TransProb[xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1 ]);
            gPayoffPrime = gPayoffPrime - (d1+d2);
            gTransProbPrime = gTransProbPrime - (d3+d4);
        #end
    end
    =#
    # transformed version from MPEC
    Cost = zeros(N)
    CbEV = zeros(N)
    PayoffDiff = zeros(N)
    ProbRegMaint = zeros(N)
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2));
        Cost[i] = 0.001*thetaCost*x[i]
        CbEV[i] = - Cost[i] + beta*EV[i]
    end
    for i in 1:N
        PayoffDiff[i] = -CbEV[i] - RC + CbEV[1];
        ProbRegMaint[i] =  1/(1+exp(PayoffDiff[i]));
    end
    RegMaint_term = sum(sum(log(dt[t,b]*(1-ProbRegMaint[convert(Int64,xt[t,b])])
          + (1-dt[t,b])*ProbRegMaint[convert(Int64,xt[t,b])]) for t=2:nT) for b=1:nBus)
    #+ sum {t in 2..nT, b in B} log(dt[t-1,b]*(thetaProbs[xt[t,b]-1+1])
    #                               + (1-dt[t-1,b])*(thetaProbs[xt[t,b]-xt[t-1,b]+1]));
    thetaProbs_term = sum(sum(log(dt[t-1,b]*(thetaProbs[convert(Int64, min(xt[t,b],4)-1+1)]) #thetaProbs[xt[t,b]-1+1]
          + (1-dt[t-1,b])*(thetaProbs[convert(Int64, max(xt[t,b]-xt[t-1,b],0)+1)])) for t=2:nT) for b=1:nBus)
    f = RegMaint_term + thetaProbs_term

    println("Continue to compute the gradient of the likelihood function\n")
    println("No hessian\n")
    println("Likelihood : ", f)
    #=
    #% Continue to compute the gradient of the likelihood function
    if nargout > 1
        gPayoffPrimetheta = gPayoffPrime(1:ntheta);
        gPayoffPrimeEV = gPayoffPrime(ntheta+1:end);
        gTransProbPrimetheta = gTransProbPrime(1:ntheta);
        s1 = exp(CbEV(indices));
        s2 = exp(-RC+CbEV(1));
        s = s1 + s2;
        logs = log(s);
        Rprime = zeros(1,ntheta+N);
        Rprime(ntheta)=-1;
        Rprime(ntheta+1)=beta;
        d1 = ((CbEVPrime.*repmat(exp(CbEV),1,ntheta + N) + exp(-RC+CbEV(1))*repmat(Rprime,N+M,1)))./(repmat(exp(CbEV)+exp(-RC+CbEV(1)),1,ntheta + N));
        sum1 = reshape(sum(reshape(d1(indices',:) .*
        repmat(repmat(TransProb,N,1),1,ntheta + N),M,N, ntheta + N )),N,ntheta + N);
        sum2 = logs*TransProbPrime';
        TPrime = sum1 + sum2;
        EVPrime = [zeros(N,ntheta) eye(N)];
        dTdtheta = TPrime(:,1:ntheta);
        dTdEV = TPrime(:,(ntheta+1):ntheta+N);
        gNFXP = dTdtheta'*(inv(eye(N)-dTdEV))'*gPayoffPrimeEV + gPayoffPrimetheta +gTransProbPrimetheta;
    end
    if nargout > 2
        h=[];
    end
    =#
    #return f, gNFXP, h
    return f
end


firststage_thetaProbs = param.thetaProbs
f = likelihoodNFXP_partialMLE(param,param.thetaCost,param.RC,firststage_thetaProbs)

BellEval = 0
function neg_likelihoodNFXP_partialMLE(param,theta)
    thetaCost = theta[1]
    RC = theta[2]
    firststage_thetaProbs = param.thetaProbs # true thetaProbs
    neg_f = - likelihoodNFXP_partialMLE(param,thetaCost,RC,firststage_thetaProbs)
    return neg_f
end
test_theta_PartialMLE = [param.thetaCost param.RC]
neg_f = neg_likelihoodNFXP_partialMLE(param, test_theta_PartialMLE)

# minimize negative likelihood
#func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar + 1]),
#                           ones(nvar+1); autodiff=:forward);
func_partialMLE = TwiceDifferentiable(theta_hat -> neg_likelihoodNFXP_partialMLE(param, theta_hat),
                           ones(length(test_theta_PartialMLE))); #zeros(length(theta))
# starting values are zeros
@time opt = Optim.optimize(func_partialMLE, zeros(length(test_theta_PartialMLE)))
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)
println("True (thetaCost, RC) = ",param.thetaCost," , ", param.RC)
# starting values are ones
@time opt = Optim.optimize(func_partialMLE, ones(length(test_theta_PartialMLE)))
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)
println("True (thetaCost, RC) = ",param.thetaCost," , ", param.RC)
# starting values are true values
@time opt = Optim.optimize(func_partialMLE, [param.thetaCost, param.RC])
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)
println("True (thetaCost, RC) = ",param.thetaCost," , ", param.RC)

#println(parameters)


function NFXP_partialMLE_MC(param::rust_struct;
                 test_MC = 1,
                 test_multistarts = 5)
    MC = param.MC
    multistarts = param.multistarts
    N = param.N
    M = param.M
    # Restore the spaces
    #KnitroExitAMPL = -10000*ones(MC,1);
    objvalNFXP = zeros(1,MC, multistarts);
    thetaCostNFXP = zeros(1,MC, multistarts);
    RCNFXP = zeros(1, MC, multistarts);
    EVNFXP = zeros(N, MC, multistarts);
    thetaProbsNFXP = zeros(M,MC, multistarts);
    #SolveTimeAMPL = zeros(MC,1);
    ObjValNFXP = zeros(MC,1);
    #IterAMPL = zeros(MC,1);
    #FunEvalAMPL = zeros(MC,1);
    #SuccessAMPL = zeros(MC,1);
    tic()
    for k = 1:test_MC
        for reps = 1:test_multistarts
            #fprintf('This is Monte Carlo run #%d out of %d\n \n', kk, MC);
            dt = MC_dt[:,:,k]
            xt = MC_xt[:,:,k]
            dx = MC_dx[:,:,k]
            #@time thetaCostAMPL[1, k, reps], RCAMPL[1,k, reps],EVAMPL[:,k, reps], thetaProbsAMPL[:,k,reps],objvalAMPL[1,k,reps] = RustBusMLETableX(param, dt, xt, dx, k, reps)
            func_partialMLE = TwiceDifferentiable(theta_hat -> neg_likelihoodNFXP_partialMLE(param, theta_hat),
                                       ones(length(test_theta_PartialMLE))); #zeros(length(theta))
            @time opt = Optim.optimize(func_partialMLE, zeros(length(test_theta_PartialMLE)))
            objvalNFXP[1,k,reps] = opt.minimum
            thetaCostNFXP[1, k, reps] = opt.minimizer[1]
            thetaProbsNFXP[:,k,reps] = param.thetaProbs # 1st stage = true
            RCNFXP[1,k, reps] = opt.minimizer[2]
            #@time sol_all = RustBusMLETableX(param, dt, xt, dx, k, reps)
            println("iteration counter: $k \n", "multistarts counter: $reps" )
        end
    end
    toc()
    open("MC$(MC)_beta$(param.beta)_thetaCostNFXP.txt", "w") do f
        write(f, "$thetaCostNFXP", "\n")
    end

    open("MC$(MC)_beta$(param.beta)_RCNFXP.txt", "w") do f
        write(f, "$RCNFXP", "\n")
    end
    open("MC$(MC)_beta$(param.beta)_thetaProbsNFXP.txt", "w") do f
        write(f, "$thetaProbsNFXP", "\n")
    end
    # Total elapsed time: 46516.19seconds
    println("SKIP [status,result] = system(strAmplSystemCall);")
    return thetaCostNFXP,RCNFXP,thetaProbsNFXP
end
@time thetaCostNFXP,RCNFXP,thetaProbsNFXP = NFXP_partialMLE_MC(param,test_MC = 1,test_multistarts = 1)
#109.763729 seconds (53.59 M allocations: 46.969 GiB, 4.13% gc time)

#--------------------------------------
#Plot surface of objective function of Partial MLE
#--------------------------------------
firststage_thetaProbs = param.thetaProbs
f = likelihoodNFXP_partialMLE(param,param.thetaCost,param.RC,firststage_thetaProbs)
grid_ind = 15
temp = zeros(Float64,grid_ind*grid_ind,3)
@time for i = 1:grid_ind
	for j = 1:grid_ind
		theta_list = [0:1:15;]
		theta_start = convert(Array{Float64,1},[i, j])
		theta_start[1] = theta_list[i]
		theta_start[2] = theta_list[j]
		temp[i+(j-1)*grid_ind,1] = theta_start[1]
		temp[i+(j-1)*grid_ind,2] = theta_start[2]
		temp[i+(j-1)*grid_ind,3] = likelihoodNFXP_partialMLE(param,theta_start[1],theta_start[2],firststage_thetaProbs)
	end
end
# 18.160824 seconds (6.32 M allocations: 7.130 GiB, 4.60% gc time)

temp_plot = Plots.plot(xlabel = "thetaCost", ylabel ="RC" )
#temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
temp_plot = Plots.plot!(temp[:,1] ,temp[:,2], -temp[:,3], st=:surface)
temp_plot = Plots.plot!(title="partial_MLE_obj_NFXP")
@show temp_plot
#savefig(temp_plot,"partial_MLE_obj_NFXP")

#=
#contour
temp_plot = Plots.plot(xlabel = "thetaCost", ylabel ="RC" )
temp_plot = Plots.plot!(temp[:,1] ,temp[:,2], -temp[:,3], st = [:contourf])
temp_plot = Plots.plot!(title="partial_MLE_obj_NFXP")
@show temp_plot
=#
##########################################################################
# Converted code of Su&Judd2012
# END
##########################################################################

###################
# NK written by sakaguchi
####################
using Printf
mutable struct rust_nk_struct
	N::Int64
	MM::Int64
	theta3::Vector{Float64}
	TM::Array{Float64,2}
	RC::Float64
	theta1::Float64
	beta::Float64
	x_grid::Vector{Float64}
	cost::Vector{Float64}
	u0::Vector{Float64}
	u1::Float64
	tol_SA::Float64
	Min_iter_SA::Int64
	Max_iter_SA::Int64
	tol_NK::Float64
	Min_iter_NK::Int64
	Max_iter_NK::Int64
	tol_ratio_NK::Float64
end

function rust_nk_struct_mod( ;N=175,
	MM=450,
	theta3=[0.0937,0.4475,0.4459,0.0127],
	beta=0.9999,
	RC=11.7257,
	theta1=2.45569,
	sqrtcost=0,
	tol_SA=1.0e-12,
	Min_iter_SA=10,
	Max_iter_SA=100,
	tol_NK=1.0e-12,
	Min_iter_NK=10,
	Max_iter_NK=100,
	tol_ratio_NK=1.0e-02)

	theta3=vcat(abs.(theta3), 1-sum(theta3)) #transition probability
	k3=length(theta3)
	TM=zeros(N,N)
	for i=1:N
		if i+k3-1<=N
	    	TM[i,i:i+k3-1]=theta3
		else
			over=i+k3-1-N
			theta3_t=theta3[1:k3-over]
			theta3_t[end]+=1-sum(theta3_t)
			TM[i,i:N]=theta3_t
		end
	end
	x_grid=Float64.(0.0:N-1)
	if sqrtcost==1
		cost=0.001*theta1.* (x_grid .^ 0.5)
	else
		cost=0.001*theta1.*x_grid
	end
	u0 = -cost
	u1 = -RC -cost[1]
	m=rust_nk_struct(N,MM,theta3,TM,RC,theta1,beta,x_grid,cost,u0,u1,tol_SA,Min_iter_SA,Max_iter_SA,tol_NK,Min_iter_NK,Max_iter_NK,tol_ratio_NK)
	return m
end



# EV(x,a)
function bellman_fullarg(u0,u1,beta,EV,TM)
	Vkeep = u0 .+ beta .* EV
	Vchange = u1 .+ beta .* EV[1]
	Vmax=maximum([maximum(Vkeep),maximum(Vchange)])
	newEV = TM * (Vmax .+ log.( exp.(Vkeep.-Vmax) .+ exp.(Vchange.-Vmax)))
	pk = 1 ./ (1 .+exp.(Vchange.-Vkeep) )
	return newEV, pk
end

function recover_EV_SA(bellman,N,tol,Max_iter,Min_iter;display=1,beta=0,tol_ratio=0)
	EV=zeros(N,1)
	# while d>=EV_tol
	supnorm=1
	convergenced=0
	pk=zeros(N,2)
	if display==1
		@printf "iter  sup-norm \n"
	end
	norms=zeros(Max_iter,2)
	for iter=1:Max_iter
	    newEV, pk = bellman(EV)
		supnorm_new = maximum( abs.( newEV .- EV ) )
		supnorm_relative=supnorm_new/supnorm
		supnorm=supnorm_new
		adj=ceil(log10(abs(maximum(newEV))))
		ltol= tol* 10^adj
		norms[iter,:]=[iter,supnorm]
		if display==1
			@printf "%4.0f   %4.15f \n" iter supnorm
		end
		if tol_ratio > 0
			if abs.(supnorm_relative-beta) < tol_ratio
				break
			end
		end
		if　supnorm<=tol || ( (iter>=Min_iter) && (supnorm < ltol) )
			convergenced=1
			break
		end
		EV=newEV
	end
	if display==1
		if convergenced==1
			println("SA Successes to convergence. Sup-norm of difeerence of EV is ",supnorm)
		else
			println("SA Fail in convergence. Sup-norm of difeerence of EV is ",supnorm)
		end
	end
	return EV,pk,norms
end


function derivatveGamma(beta,pk,TM)
	dGamma_dEV = Array(Diagonal(pk[:,1]))
	dGamma_dEV[:,1] = dGamma_dEV[:,1] .+ (1 .- pk)
  	dGamma_dEV=TM*beta*dGamma_dEV
end

function recover_EV_NK(bellman,N,TM,beta,tol_SA,Max_iter_SA,Min_iter_SA,tol_NK,Max_iter_NK,Min_iter_NK,tol_ratio_NK;display=1)
	if display==1
		println("First, calcurate starting value of EV by SA")
	end
	EV0,pk,norms_SA=recover_EV_SA(bellman,N,tol_SA,Max_iter_SA,Min_iter_SA,display=display,beta=beta,tol_ratio=tol_ratio_NK)
	EV=EV0
	convergenced=0
	supnorm=1
	if display==1
		@printf "iter  sup-norm \n"
	end
	norms_NK=zeros(Max_iter_NK,2)
	for iter=1:Max_iter_NK
		EV1, pk = bellman(EV0)
		dGamma_dEV=derivatveGamma(beta,pk,TM)
		F=Matrix(I,N,N)-dGamma_dEV
		EV=EV0 - inv(F)*(EV0-EV1)
		EV0, pk = bellman(EV)
		supnorm = maximum( abs.( EV .- EV0 ) )
		adj=ceil(log10(abs(maximum(EV0))))
		ltol= tol_NK* 10^adj
		norms_NK[iter,:]=[iter,supnorm]
		if display==1
			@printf "%4.0f   %4.15f \n" iter supnorm
		end
		if　supnorm<=tol_NK || ( (iter>=Min_iter_NK) && (supnorm < ltol) )
			convergenced=1
			break
		end
	end
	if display==1
		if convergenced==1
			println("N-K Successes to convergence in tolerance level. Sup-norm of difeerence of EV is ",supnorm)
		else
			println("N-K Fail in convergence. Sup-norm of difeerence of EV is ",supnorm)
		end
	end
	return EV,pk,norms_SA,norms_NK
end


function recoverEV_NFPA(m;NK=1,display=1)
	N=m.N
	u0=m.u0
	u1=m.u1
	beta=m.beta
	tol_SA=m.tol_SA
	Max_iter_SA=m.Max_iter_SA
	Min_iter_SA=m.Min_iter_SA
	tol_NK=m.tol_NK
	Max_iter_NK=m.Max_iter_NK
	Min_iter_NK=m.Min_iter_NK
	tol_ratio_NK=m.tol_ratio_NK
	TM=m.TM
	bellman(EV)=bellman_fullarg(u0,u1,beta,EV,TM)
	if NK==1
		EV,pk,norms_SA,norms_NK=recover_EV_NK(bellman,N,TM,beta,tol_SA,Max_iter_SA,Min_iter_SA,tol_NK,Max_iter_NK,Min_iter_NK,tol_ratio_NK,display=display)
	else
		EV,pk,norms_SA=recover_EV_SA(bellman,N,tol_SA,Max_iter_SA,Min_iter_SA,display=display)
		norms_NK=[]
	end
	return EV,pk,norms_SA,norms_NK
end

m= rust_nk_struct_mod()


function likelihoodNFXPNK_partialMLE(param::rust_struct, thetaCost::Float64, RC::Float64, firststage_thetaProbs::Array{Float64,2})
    global dt, xt, nT, indices
    N = param.N
    M = param.M
    nT = param.nT
    beta = param.beta
    nBus = param.nBus
    global PayoffDiffPrime, TransProbPrime, CbEVPrime;
    firststage_thetaProbs # 1st estimates
    #% Define parameters for cost function
    #thetaCost = theta(1);
    #thetaCost = theta[1];
    #% Define parameters and definition of transition process
    #% thetaProbs defines Markov chain
    #thetaProbs = theta(2:6);
    thetaProbs = firststage_thetaProbs # 1st estimates
    #TransProb = thetaProbs;
    TransProb =  Array{Float64,2}(undef,1,5)
    TransProb[1,:] = thetaProbs;
    #% Define replacement cost parameter
    #RC = theta(7);
    #RC = theta[7];
    #ntheta = length(theta);
    theta = param.thetatrue
    ntheta = length(theta);
    #% Use constration mapping iteration to solve the integrated Bellman equations
    #[EV, CbEV] = BellContract_SA(thetaCost, TransProb, RC);
    #---------------
    # NK version
    #---------------
    m=rust_nk_struct_mod(theta1=thetaCost,RC=RC,theta3=vec(firststage_thetaProbs))
	EV,pk,norms_SA,norms_NK=recoverEV_NFPA(m;display=0)
    Cost = zeros(N)
    CbEV = zeros(N)
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2));
        Cost[i] = 0.001*thetaCost*x[i]
        CbEV[i] = - Cost[i] + beta*EV[i]
    end
    #% Let PayoffDiff[i] represent -CbEV[i] - RC + CbEV[1];
    #% this is the difference in expected payoff at x[i] between engine replacement and regular maintenance
    #PayoffDiff = -CbEV - RC + CbEV(1);
    PayoffDiff = -CbEV .- RC .+ CbEV[1];
    #% Let ProbRegMaint[i] represent 1/(1+exp(PayoffDiff[i]));
    #% this is the probability of performing regular maintenance at state x[i];
    #ProbRegMaint = 1./(1+exp(PayoffDiff));
    ProbRegMaint = 1 ./(1 .+exp.(PayoffDiff));
    #=
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% OBJECTIVE AND CONSTRAINT DEFINITIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define objective function: Likelihood function
    % The likelihood function contains two pieces
    % First is the likelihood that the engine is replaced given time t state in the
    data.
    % Second is the likelihood that the observed transition between t-1 and t
    % would have occurred.
    =#
    #f = 0;
    f = 0;
    #g = zeros(length(theta)+N,1);
    g = zeros(length(theta)+N,1);
    #gPayoffPrime = g;
    gPayoffPrime = g;
    #gTransProbPrime = g;
    gTransProbPrime = g;
    #dtM1Minus = [];
    dtM1Minus = [];
    #dtM1Plus = [];
    dtM1Plus = [];
    #dtMinus = [];
    dtMinus = [];
    #dtPlus = [];
    dtPlus = [];
    #for i = 1:nBus
    #=
    for i = 1:(nBus-1)
        #i=30 #test
        #dtM1Minus = find(dt(1:(nT-1),i)==0);
        dtM1Minus = findall(dt[1:(nT-1),i].==0)
        #dtM1Plus = find(dt(1:(nT-1),i)==1);
        dtM1Plus = findall(dt[1:(nT-1),i].==1);
        #dtMinus = find(dt((2:nT),i)==0)+1;
        dtMinus = findall(dt[(2:nT),i].==0).+1;
        #dtPlus = find(dt((2:nT),i)==1)+1;
        dtPlus = findall(dt[(2:nT),i].==1).+1;
        #ProbRegMaint(xt(dtPlus,i));
        #ProbRegMaint(xt(dtMinus,i));
        #TransProb( xt( dtM1Plus+1,i ) );
        #TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1);
        #= Following parts is no meaning on Matlab??
        ProbRegMaint[convert(Int64,xt[dtPlus,i])];
        ProbRegMaint[convert(Int64,xt[dtMinus,i])];
        TransProb[ convert(Int64,xt[dtM1Plus+1,i ]) ];
        TransProb[convert(Int64,xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1)];
        =#
        #% Compute the value of the likelihood function
        #=
        f = f -( sum( log( 1-ProbRegMaint(xt(dtPlus,i))))...
        + sum( log( ProbRegMaint(xt(dtMinus,i)))) ...
        + sum( log( TransProb( xt( dtM1Plus +1,i ) ) ))...
        + sum( log( TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1))) );
        =#
        f1 = sum( log.( 1 .-ProbRegMaint[convert(Vector{Int64},xt[dtPlus,i])]))
        f2 = sum( log.( ProbRegMaint[convert(Vector{Int64},xt[dtMinus,i])]))
        f3 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Plus.+1 ,i ]) ] ))
        f4 = sum( log.( TransProb[convert(Vector{Int64}, xt[dtM1Minus.+1, i]-xt[dtM1Minus,i].+1)]))
        f = f - ( f1 + f2 + f3 + f4 );
        println(f)
        if f == NaN
            break
        end
        #% Compute the gradient of the likelihood function
        println("Compute the gradient of the likelihood function")
        #if nargout > 1
            #d1 = PayoffDiffPrime(:,xt(dtPlus,i))*ProbRegMaint(xt(dtPlus,i)) ;
            d1 = PayoffDiffPrime[:,convert(Int64,xt[dtPlus,i])]*ProbRegMaint[xt[dtPlus,i]] ;
            #d2 = - PayoffDiffPrime(:,xt(dtMinus,i))*( 1-ProbRegMaint(xt(dtMinus,i)));
            d2 = - PayoffDiffPrime[:,convert(Int64,xt[dtMinus,i])]*( 1-ProbRegMaint[xt[dtMinus,i]]);
            #d3 = TransProbPrime(:, xt( dtM1Plus +1,i ))*(1./TransProb( xt( dtM1Plus+1,i ) ));
            d3 = TransProbPrime[:, convert(Int64,xt[dtM1Plus +1,i])]*(1./TransProb[xt[dtM1Plus+1,i ]]);
            #d4 = TransProbPrime(:,xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1)*(1./TransProb( xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1 ));
            d4 = TransProbPrime[:,xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1]*(1./TransProb[xt[dtM1Minus+1,i]-xt[dtM1Minus,i]+1 ]);
            gPayoffPrime = gPayoffPrime - (d1+d2);
            gTransProbPrime = gTransProbPrime - (d3+d4);
        #end
    end
    =#
    # transformed version from MPEC
    Cost = zeros(N)
    CbEV = zeros(N)
    PayoffDiff = zeros(N)
    ProbRegMaint = zeros(N)
    for i in 1:N
        #@NLconstraint(model, Cost[i]== sum(thetaCost[j]*x[i]^j for j in 1:2));
        Cost[i] = 0.001*thetaCost*x[i]
        CbEV[i] = - Cost[i] + beta*EV[i]
    end
    for i in 1:N
        PayoffDiff[i] = -CbEV[i] - RC + CbEV[1];
        ProbRegMaint[i] =  1/(1+exp(PayoffDiff[i]));
    end
    RegMaint_term = sum(sum(log(dt[t,b]*(1-ProbRegMaint[convert(Int64,xt[t,b])])
          + (1-dt[t,b])*ProbRegMaint[convert(Int64,xt[t,b])]) for t=2:nT) for b=1:nBus)
    #+ sum {t in 2..nT, b in B} log(dt[t-1,b]*(thetaProbs[xt[t,b]-1+1])
    #                               + (1-dt[t-1,b])*(thetaProbs[xt[t,b]-xt[t-1,b]+1]));
    thetaProbs_term = sum(sum(log(dt[t-1,b]*(thetaProbs[convert(Int64, min(xt[t,b],4)-1+1)]) #thetaProbs[xt[t,b]-1+1]
          + (1-dt[t-1,b])*(thetaProbs[convert(Int64, max(xt[t,b]-xt[t-1,b],0)+1)])) for t=2:nT) for b=1:nBus)
    f = RegMaint_term + thetaProbs_term

    println("Continue to compute the gradient of the likelihood function\n")
    println("No hessian\n")
    println("Likelihood : ", f)
    #=
    #% Continue to compute the gradient of the likelihood function
    if nargout > 1
        gPayoffPrimetheta = gPayoffPrime(1:ntheta);
        gPayoffPrimeEV = gPayoffPrime(ntheta+1:end);
        gTransProbPrimetheta = gTransProbPrime(1:ntheta);
        s1 = exp(CbEV(indices));
        s2 = exp(-RC+CbEV(1));
        s = s1 + s2;
        logs = log(s);
        Rprime = zeros(1,ntheta+N);
        Rprime(ntheta)=-1;
        Rprime(ntheta+1)=beta;
        d1 = ((CbEVPrime.*repmat(exp(CbEV),1,ntheta + N) + exp(-RC+CbEV(1))*repmat(Rprime,N+M,1)))./(repmat(exp(CbEV)+exp(-RC+CbEV(1)),1,ntheta + N));
        sum1 = reshape(sum(reshape(d1(indices',:) .*
        repmat(repmat(TransProb,N,1),1,ntheta + N),M,N, ntheta + N )),N,ntheta + N);
        sum2 = logs*TransProbPrime';
        TPrime = sum1 + sum2;
        EVPrime = [zeros(N,ntheta) eye(N)];
        dTdtheta = TPrime(:,1:ntheta);
        dTdEV = TPrime(:,(ntheta+1):ntheta+N);
        gNFXP = dTdtheta'*(inv(eye(N)-dTdEV))'*gPayoffPrimeEV + gPayoffPrimetheta +gTransProbPrimetheta;
    end
    if nargout > 2
        h=[];
    end
    =#
    #return f, gNFXP, h
    return f
end


firststage_thetaProbs = param.thetaProbs
f = likelihoodNFXPNK_partialMLE(param,param.thetaCost,param.RC,firststage_thetaProbs)

BellEval = 0
function neg_likelihoodNFXPNK_partialMLE(param,theta)
    thetaCost = theta[1]
    RC = theta[2]
    firststage_thetaProbs = param.thetaProbs # true thetaProbs
    neg_f = - likelihoodNFXPNK_partialMLE(param,thetaCost,RC,firststage_thetaProbs)
    return neg_f
end
test_theta_PartialMLE = [param.thetaCost param.RC]
neg_f = neg_likelihoodNFXPNK_partialMLE(param, test_theta_PartialMLE)

# minimize negative likelihood
#func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar + 1]),
#                           ones(nvar+1); autodiff=:forward);
func_partialMLE = TwiceDifferentiable(theta_hat -> neg_likelihoodNFXPNK_partialMLE(param, theta_hat),
                           ones(length(test_theta_PartialMLE))); #zeros(length(theta))
# starting values are zeros
@time opt = Optim.optimize(func_partialMLE, zeros(length(test_theta_PartialMLE)))
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)
println("True (thetaCost, RC) = ",param.thetaCost," , ", param.RC)
# starting values are ones
@time opt = Optim.optimize(func_partialMLE, ones(length(test_theta_PartialMLE)))
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)
println("True (thetaCost, RC) = ",param.thetaCost," , ", param.RC)
# starting values are true values
@time opt = Optim.optimize(func_partialMLE, [param.thetaCost, param.RC])
parameters = Optim.minimizer(opt)
println("Estimated (thetaCost,RC) = ",parameters)




### Per Period Returns Function
function payoff(param::rust_struct, thetaCost::Float64, RC::Float64)
    N = param.N
    x = collect(1:N)' # Generate State Space range, i.e. [1, 2, 3, 4, ...]
    pi_0 = - 0.001 * thetaCost * x # Utility from not replacing
    pi_1 = - RC * ones(1, N) # Utility from replacing
    U = vcat(pi_0, pi_1) # Utility matrix
    return U
end
test_thetaProbs = param.thetaProbs[:] # test (and true)
test_thetaCost = 1.0
test_RC = 10.0
payoff(param, test_thetaCost, test_RC) # test
### Transition Probabilities
function transition_probs(param::rust_struct, thetaProbs::Vector{Float64})
    t = param.M #
    N = param.N
    ttmp = zeros(N - t, N) # Transition Probabilities
    for i in 1:N - t
        for j in 0:t-1
            ttmp[i, i + j] = thetaProbs[j+1]
        end
    end
    atmp = zeros(t,t) # Absorbing State Probabilities
    for i in 0:t - 1
        atmp[i+ 1,:] = [zeros(1,i) thetaProbs[1:t - i - 1]' ( 1 - sum(thetaProbs[1:t- i - 1]) ) ]
    end
    return [ttmp ; zeros(t, N - t) atmp]
end;
transition_probs(param, test_thetaProbs) # test

## Social Surplus function
function ss(param::rust_struct, EV::Array{Float64,1}, thetaCost::Float64, RC::Float64)
    N = param.N
    beta = param.beta
    payoff_list = payoff(param, thetaCost, RC)
    ss_val = (  exp.( payoff_list[1,:] + beta * EV - EV) +       # i=0
                exp.( payoff_list[2,:] + beta * EV[1] * ones(N,1) - EV) )    # i=1
    return EV + log.(ss_val)
end;
ss(param, EV, test_thetaCost, test_RC) # test
### Contraction Mapping
function contraction_mapping(param::rust_struct, thetaProbs::Any, thetaCost::Float64, RC::Float64)
    N = param.N
    P = transition_probs(param, thetaProbs) # Transition Matrix (K x K)
    eps_cp = 1 # Set epsilon to something greater than 0
    EV = ones(N,1) # initialized
    iter = 1
    while eps_cp > .000001
        EV1 = P * ss(param, EV, thetaCost, RC) # updated EV
        eps_cp = maximum(abs.(EV1 - EV))
        println("EV1-EV = ",eps_cp)
        println("iter = ",iter)
        EV = EV1
        iter += 1
    end
    return EV
end;
EV_NFXP = contraction_mapping(param, param.thetaProbs[:], param.thetaCost, param.RC) # test
@time EV_MPEC, thetaProbs = RustBusMLETableXSolveEV(param::rust_struct)
# Alomost same as previous EV (in ftn of `RustBusMLETableXSolveEV`)
Plots.plot([1:1:param.N], EV_NFXP, color = :red, title="EV_NFXP vs EV_MPEC")
Plots.plot!([1:1:param.N], EV_MPEC, color = :blue,line = (:dashdot))
println("Deviation between EV_NFXP and EV_MPEC : ",sum(EV_NFXP.-EV_MPEC))
### Choice Probabilities
############################
# Again, overflow is an issue, so the maximum value of EV
# is subtracted from the exponents in the numerator and denominator.
# This is equivalent to multiplying through by 1 and resolves the overflow.
#############################
function choice_prob(param::rust_struct, EV::Vector{Float64},thetaCost::Float64, RC::Float64)
    N = param.N
    beta = param.beta
    max_EV = maximum(EV) * ones(N,1)
    payoff_list = payoff(param, thetaCost, RC)
    enum = exp.( payoff_list[1,:] + beta *  EV - max_EV)
    denom = ( exp.( payoff_list[1, :] + beta *  EV - max_EV) +
        exp.( payoff_list[2,:] + beta * EV[1] * ones(N,1) - max_EV ) )
    P_k = enum ./ denom
    return P_k
end;
choice_prob(param, EV_NFXP, test_thetaCost, test_RC)
choice_prob(param, EV_MPEC, test_thetaCost, test_RC)

### Partial Log-Likelihood
function partial_LL(param::rust_struct, EV::Array{Float64,1},thetaCost::Float64, RC::Float64, MC_id::Int64)
    global MC_dt, MC_xt
    decision_obs = MC_dt[:,:,MC_id]# data.endog
    state_obs = MC_xt[:,:,MC_id]#data.exog
    cp_tmp = choice_prob(param, EV, thetaCost, RC)
    relevant_probs = [ cp_tmp[convert(Int, i)] for i in state_obs ] # ?
    p_ll = [ if decision == 0 log(r_p) else log(1 - r_p) end for (decision, r_p) in zip(decision_obs, relevant_probs)]
    println("partial_LL = ",-sum(p_ll))
    return -sum(p_ll)
end
MC_id = 1
partial_LL(param, EV_NFXP[:], test_thetaCost, test_RC, MC_id)
### Log-Likelihood
function objFunc(param::rust_struct, theta, thetaProbs_1st_stage::Any, MC_id::Int64)
    thetaCost = theta[1] # mutable struct change is OK?
    RC = theta[2]
    EV_NFXP = contraction_mapping(param, thetaProbs_1st_stage, thetaCost, RC)
    pll = partial_LL(param, EV_NFXP[:],thetaCost,RC, MC_id)
    return pll
end
# solve for MLE
firststage_thetaProbs = param.thetaProbs[:]
objFunc_for_Optim = TwiceDifferentiable(theta_hat -> objFunc(param, theta_hat, firststage_thetaProbs, MC_id),
                           ones(length(test_theta_PartialMLE)));
@time @show opt = Optim.optimize(objFunc_for_Optim, zeros(length(test_theta_PartialMLE)))
#1266.258673 seconds
parameters_NFP = Optim.minimizer(opt)
println("NFP Estimated (thetaCost,RC) =\n ",parameters_NFP)
println("True (thetaCost, RC) = ", param.thetaCost," , ", param.RC)
println("MPEC jointly Estimated:(thetaCost,[thetaProbs],RC) =\n ", parameters_MPEC)
println("true parameters: ", param.thetatrue)


firststage_thetaProbs = param.thetaProbs
grid_ind = 10
temp = zeros(Float64,grid_ind*grid_ind,3)
for i = 1:grid_ind
	for j = 1:grid_ind
		theta_list = [0:2:20;]
		theta_start = convert(Array{Float64,1},[i, j])
		theta_start[1] = theta_list[i]
		theta_start[2] = theta_list[j]
		temp[i+(j-1)*grid_ind,1] = theta_start[1]
		temp[i+(j-1)*grid_ind,2] = theta_start[2]
        EV_NFXP = contraction_mapping(param, vec(firststage_thetaProbs), theta_start[1], theta_start[2])
        temp[i+(j-1)*grid_ind,3] = partial_LL(param, EV_NFXP[:],theta_start[1],theta_start[2], MC_id)
	end
end

temp_plot = Plots.plot(ylabel ="RC", xlabel = "thetaCost" )
#temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
temp_plot = Plots.plot!(temp[:,1], temp[:,2] , temp[:,3], st=:surface)
temp_plot = Plots.plot!(title="partial_MLE_obj_NFXP_ponder")
@show temp_plot
savefig(temp_plot,"partial_MLE_obj_ponder_ponder")
