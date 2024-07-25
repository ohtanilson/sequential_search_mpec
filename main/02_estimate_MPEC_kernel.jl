#generate data from 01generate_data.jl
using LinearAlgebra
using Kronecker
using JuMP
using Ipopt
using Distributions
using CSV, DataFrames, DelimitedFiles, Statistics
using Base.Threads
using Random

function Kernel_MPEC(data::Matrix{Float64},maxtime::Float64,max_iter::Int64,tol::Float64 = 1e-6, seed::Int64 = 1)

    #tolerance of constraint/ objective/ parameter
    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>maxtime))
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", tol)
    #global param # initial value list
    @variable(model, par[i = 1:4],start = 0.0) 
    # @variable(model, par1,start = param[1]) 
    # @variable(model, par2,start = param[2]) 
    # @variable(model, par3,start = param[3]) 
    # @variable(model, par4,start = param[4]) 
    @variable(model, 2.0 >= c >= 0.0,start = exp(-3.0)) 
    @variable(model, m,start = 1.258) #scaler

    # Data features
    consumer = data[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    N_prod = data[:, end - 2]
    Js = unique(N_prod)
    nalt = Int.(Js[1])

    #error draws
    Random.seed!(seed)
    epsilonDraw = randn(N_obs, D)
    etaDraw = randn(N_obs, D)

    # Choices
    tran = data[:, end]
    searched = data[:, end - 1]
    lasts = data[:, end - 4]

    # Parameters
    outside = data[:, 3]
    #@NLexpression(model,c, exp(params[end]) )
    X = data[:, 4:7]
    # X1 = data[:, 4]
    # X2 = data[:, 5]
    # X3 = data[:, 6]
    # X4 = data[:, 7]

    eut = @expression(model, (X*par .+ etaDraw) .* (1 .- outside))
    #eut = @expression(model, ((X1*par1 .+ X2*par2 .+ X3*par3 .+ X4*par4 ).+ etaDraw) .* (1 .- outside))

    ut = @expression(model,eut .+ epsilonDraw)
    z = @expression(model, m .+ eut)
    # global  table
    # @NLconstraint(model, 
    #     ifelse(
    #         table[1, 2] >= c && c >= table[end, 2], m - table[argmin(abs.(table[:, 2] .- c)), 1],
    #         ifelse(table[1, 2] < c, m + c, m - 4.001)
    #         ) == 0)
    norm_cdf(x) = cdf(Normal(), x)
    norm_pdf(x) = pdf(Normal(), x)
    register(model, :norm_cdf, 1, norm_cdf; autodiff = true)
    register(model, :norm_pdf, 1, norm_pdf; autodiff = true)
    @NLconstraint(model, c == norm_pdf(m) + m *(norm_cdf(m) - 1))

    ut_unsearched = zeros(N_obs, D)
    searched2 = repeat(searched, 1, D)
    ut_unsearched[searched2 .== 0] .= -Inf
    ut_searched = @expression(model, ut .+ ut_unsearched)
    
    # Selection rule: z > z_next
    z_max = Array{NonlinearExpr,2}(undef,N_obs, D)
    for i = 1:N_cons
        k = nalt*(i-1) #last alternative for the previous consumer
        for j in 0:(nalt-1) #not for outside option
            for d = 1:D
                if j == 0
                    z_max[k + nalt-j,d] = @expression(model,z[k + nalt-j,d])
                else
                    z_max[k + nalt-j,d] = @expression(model,maximum(z[(k + nalt-j+1):(k + nalt),d]))
                end
            end
        end
    end   
    v1 = @expression(model, z - z_max)
    
    # # Stopping rule: z > u_so_far
    u_so_far = Array{NonlinearExpr,2}(undef,N_obs, D)
    for i = 1:N_cons
        k = nalt*(i-1) + 1 #first alternative
        #u_so_far[k,:] = @expression(model,ut_searched[k + (nalt-1),:]) #does not matter for optimization
        u_so_far[k,:] = @expression(model,ut_searched[k,:])
        for j in 1:(nalt-1)
            for d = 1:D
                u_so_far[k + j, d] = @expression(model,maximum(ut_searched[k:(k + j - 1), d]))
            end
        end
    end
    v2 = @expression(model, z .- u_so_far)
    v3 = @expression(model, u_so_far .- z)
    
    # Choice rule
    #u_max = max_{j ∈ S\ y} u_j, where y is the chosen alternative
    #by defining u_max in this way, we can avoid adding small number to u_y - u_max
    #u_y = @expression(model,Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran))

    # ut_searched_except_y_inf = zeros(N_obs, D)
    # ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
    # ut_searched_except_y  = @expression(model, ut_searched .+ ut_searched_except_y_inf)

    # ut_tran_inf = zeros(N_obs, D)
    # ut_tran_inf[repeat(tran, 1, D)  .== 0] .= -Inf
    # ut_tran = @expression(model, ut .+ ut_tran_inf)
    # u_max = Array{NonlinearExpr,2}(undef,N_cons, D)
    # u_y = Array{NonlinearExpr,2}(undef,N_obs, D)
    # for i = 1:N_cons
    #     for d = 1:D
    #         #u_max[i, d] = @expression(model, maximum(ut[(5*(i-1) + 1):5*i, d]))
    #         u_y[(5*(i-1) + 1):5*i, d] .= @expression(model, maximum( ut_tran[(5*(i-1) + 1):5*i, d])) 
    #     end
    # end
    u_y = @expression(model,(Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran)) ⊗ ones(nalt))
    v4 = @expression(model, (u_y - ut) )

    #denom
    # @NLexpression(model,denom[i = 1:N_obs, d = 1:D] , 
    #     exp(scaling[1]* v1[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v2[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v3[i,d]) * (1 - searched[i]) * (1 - outside[i])+
    #     exp( scaling[3] *v4[i,d]) * tran[i]
    #     )
    denom = @expression(model, exp.(scaling[1]* v1).*(1 .- outside) .* searched .*(1 .- lasts).+
         exp.(scaling[2]* v2).*(1 .- outside) .* searched .+
         exp.(scaling[2]* v3).*(1 .- searched) .*(1 .- outside) .+
         exp.(scaling[3] *v4) .*(1 .- tran).* searched
        )

    denom_sum  = Array{NonlinearExpr,2}(undef,N_cons, D)
    for i = 1:N_cons
        for d = 1:D
            denom_sum[i, d] = @expression(model,  sum( denom[(5*(i-1) + 1):5*i, d])) 
        end
    end
    
    #@NLexpression(model,denom_order[i = 1:N_obs, d = 1:D] , exp(scaling[1]* v1[i,d]) * searched[i,d] * (1 - outside[i,d])) #0 for outside option and searched = 0
    
    # denom_search1 = 
    #     #search until
    #     @expression(model, exp.(scaling[2]*v2).*(1 .- outside) .* searched )
    #     #not search from (search = 1 if outside = 1)
    # denom_search2 = @expression(model, exp.(scaling[2]*v3).*(1 .- searched) )#0 for outside option
    #@NLexpression(model,denom_ch[i = 1:N_cons, d = 1:D] , exp( scaling[3] *v4[i,d] ) )#(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
    # Combine all inputs
    prob = @expression(model, 1 ./ (1 .+ denom_sum))
    
    L_i = @expression(model, prob * ones(D)./D .+ 1e-10)

    JuMP.@NLobjective(model, Max, sum(log(L_i[i]) for i = 1:N_cons))

    @time JuMP.optimize!(model)

    
    return JuMP.value.(par),JuMP.value.(c),JuMP.value.(m),JuMP.objective_value(model),termination_status(model)
end

maxtime = 600.0
max_iter = 500
tol = 1e-3
scaling = [-20, -20, -20]
D = 100
# @time res = Kernel_MPEC(data,maxtime,max_iter,tol,1)

#estimate 100 Monte Carlo draws





results_MPEC = zeros(100, 7)
results_MPEC_term = []
fin = []
data_all = CSV.read("data/sim_data_100.csv", DataFrame) 
@time @threads for i = 1:100 #
    #data
    data = data_all[data_all[:, 1] .== i, 2:end] |> Matrix{Float64}    


    run_time = @elapsed begin 
            res_mpec = Kernel_MPEC(data,maxtime,max_iter,tol,i)
        end
    
    results_MPEC[i, 1:5] .= [res_mpec[1];log(res_mpec[2])] 
    results_MPEC[i, 6:6] .= res_mpec[4]
    results_MPEC[i, 7:7] .= run_time 
    append!(results_MPEC_term,[res_mpec[5]]) 
    
    append!(fin, i)
    println("finished: ", length(fin), "/", 100)
    GC.gc()
    GC.gc()
end

results_MPEC_df = DataFrame(hcat(results_MPEC, results_MPEC_term),:auto)
#column names
rename!(results_MPEC_df, names(results_MPEC_df) .=> ["beta1", "beta2", "beta3", "beta4", "logc", "loglik", "time", "converged"])

CSV.write("results/results_MPEC_1.csv",results_MPEC_df, writeheader=false)

param = [1.0, 0.7, 0.5, 0.3, -3.0]
mean(results_MPEC_df[:,1] .- param[1])
mean(results_MPEC_df[:,2] .- param[2])
mean(results_MPEC_df[:,3] .- param[3])
mean(results_MPEC_df[:,4] .- param[4])
mean(results_MPEC_df[:,5] .- param[5])

#standard deviation
std(results_MPEC_df[:,1])
std(results_MPEC_df[:,2])
std(results_MPEC_df[:,3])
std(results_MPEC_df[:,4])
std(results_MPEC_df[:,5])