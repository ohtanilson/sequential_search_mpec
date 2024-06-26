#generate data from 01generate_data.jl
using LinearAlgebra
using Kronecker
using JuMP
using Ipopt
using Distributions

# std_normal = Normal(0.0, 1.0)

scaling = [-18,-4,-7]

function Kernel_MPEC(maxtime::Float64,max_iter::Int64)

    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>maxtime))
    set_optimizer_attribute(model, "max_iter", max_iter)
    #global param # initial value list
    @variable(model, params[i = 1:4],start = 1.0) 
    @variable(model, c >= 0.0,start = 0.0) 
    @variable(model, m) #scaler

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

    # Parameters
    outside = data[:, 3]
    #@NLexpression(model,c, exp(params[end]) )
    X = data[:, 4:7]

    eut = @expression(model, ((X * params ).+ etaDraw) .* (1 .- outside))

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
                z_max[k + nalt-j,d] = @expression(model,maximum(z[(k + nalt-j):(k + nalt),d]))
            end
        end
    end   
    v1 = @expression(model, z - z_max)
    
    # # Stopping rule: z > u_so_far
    u_so_far = Array{NonlinearExpr,2}(undef,N_obs, D)
    for i = 1:N_cons
        k = nalt*(i-1) + 1 #first alternative
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

    ut_searched_except_y_inf = zeros(N_obs, D)
    ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
    ut_searched_except_y  = @expression(model, ut_searched .+ ut_searched_except_y_inf)

    ut_tran_inf = zeros(N_obs, D)
    ut_tran_inf[repeat(tran, 1, D)  .== 0] .= -Inf
    ut_tran = @expression(model, ut .+ ut_tran_inf)

    # u_max = Array{NonlinearExpr,2}(undef,N_cons, D)
    # u_y = Array{NonlinearExpr,2}(undef,N_cons, D)
    # for i = 1:N_cons
    #     for d = 1:D
    #         u_max[i, d] = @expression(model, maximum(ut[(5*(i-1) + 1):5*i, d]))
    #         u_y[i, d] = @expression(model, sum( (ut.* tran)[(5*(i-1) + 1):5*i, d])) 
    #     end
    # end
    @expression(model,u_max_[i= 1:N_cons, d= 1:D], maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, d]))
    @expression(model,u_y_[i= 1:N_cons, d= 1:D], maximum( ut_tran[(5*(i-1) + 1):5*i, d]))
    v4 = @expression(model, (u_y_ - u_max_) ⊗ ones(nalt) )

    #denom
    # @NLexpression(model,denom[i = 1:N_obs, d = 1:D] , 
    #     exp(scaling[1]* v1[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v2[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v3[i,d]) * (1 - searched[i]) * (1 - outside[i])+
    #     exp( scaling[3] *v4[i,d]) * tran[i]
    #     )
    denom = @expression(model, 
     exp.(scaling[1]* v1).*(1 .- outside) .* searched .+
        exp.(scaling[2]* v2).*(1 .- outside) .* searched .+
        exp.(scaling[2]* v3).*(1 .- searched) .*(1 .- outside) .+
        exp.(scaling[3] *v4) .* tran
        )
    #@NLexpression(model,denom_order[i = 1:N_obs, d = 1:D] , exp(scaling[1]* v1[i,d]) * searched[i,d] * (1 - outside[i,d])) #0 for outside option and searched = 0
    
    # denom_search1 = 
    #     #search until
    #     @expression(model, exp.(scaling[2]*v2).*(1 .- outside) .* searched )
    #     #not search from (search = 1 if outside = 1)
    # denom_search2 = @expression(model, exp.(scaling[2]*v3).*(1 .- searched) )#0 for outside option
    #@NLexpression(model,denom_ch[i = 1:N_cons, d = 1:D] , exp( scaling[3] *v4[i,d] ) )#(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
    # Combine all inputs
    #denom_order_search = @expression(model,denom_order .+ denom_search1 .+ denom_search2)
    #@NLexpression(model,denom_order_search_sum[i= 1:N_cons, d= 1:D], sum(denom_order_search[(5*(i-1) + 1):5*i, d], for i = 1:N_cons))

    #denom =  @expression(model,denom_order_search_sum .+ denom_ch) #denom_order_search .+ denom_ch
    #@NLexpression(model,prob[i = 1:N_obs, d = 1:D],1 / (1 + denom[i,d]))
    prob = @expression(model, 1 ./ (1 .+ denom))
    


    


    #choice = (u_y - u_max .>= 0)
    #@NLexpression(model,choice[i = 1:N_cons, d = 1:D] ,1 / (1 + exp( scaling[1] *v4[i,d] )) )


     #[i = 1:N_cons,  d = 1:D]
    #@NLexpression(model,choice[i = 1:N_cons, d = 1:D], (v4[i,d] >= 0))

    # Combine all inputs
    #L_i_d: (N x D) matrix
    #@expression(model, L_i_d, order .* search_1 .* search_2 .* search_3 .* choice)
    #@expression(model, L_i_d[i = 1:N_cons, d = 1:D], order[i,d] .* search_1[i,d] .* search_2[i,d] .* search_3[i,d] .* choice[i,d])

    #@expression(model, L_i[i = 1:N_cons], sum(L_i_d[i,d] for d=1:D))
    # for i = 1:N_cons
    #     #@NLconstraint(model, L_i_[i] == sum(choice[i,d] for d=1:D) + 1e-15)
        
    # end
    #@NLexpression(model,L_i[i = 1:N_cons], sum(prob[i,d] for d=1:D) + 1e-10)
    L_i = @expression(model, prob * ones(D)./D .+ 1e-10)

    JuMP.@NLobjective(model, Max, sum(log(L_i[i]) for i = 1:N_cons))

    @time JuMP.optimize!(model)

    
    return JuMP.value.(params),JuMP.value.(c),JuMP.objective_value(model)
end
maxtime = 10.0
max_iter = 1
@time params_,c_,objval_MPEC = Kernel_MPEC(maxtime)
[params_;c_]
param
liklWeitz_kernel_2_b([[1.0033809920108825, 0.8985401374613001, 0.5614162948784083, 0.5490977643171332];0.4569630773419061], dat, D, nalt, epsilonDraw, etaDraw,scaling)
JuMP.objective_value(model)
param[1:end-1],exp(param[end]),liklWeitz_kernel_2_b(param, dat, D, nalt, epsilonDraw, etaDraw,scaling)


