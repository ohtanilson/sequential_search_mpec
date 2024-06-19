using LinearAlgebra
using Kronecker
using JuMP
using Ipopt
using Distributions

# std_normal = Normal(0.0, 1.0)

scaling = [-10,-50,-1000]

#function Kernel_MPEC(data, D, seed)

    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>100.0))
    #global param # initial value list
    @variable(model, params[i = 1:4],start = 1.0) 
    @variable(model, c >= 0.0) 
    @variable(model, m) #scaler

    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = dat[:, end]
    searched = dat[:, end - 1]

    # Parameters
    outside = dat[:, 3]
    #@NLexpression(model,c, exp(params[end]) )
    X = dat[:, 4:7]

    ut = @expression(model, ( (X * params ) .+ etaDraw) .* (1 .- outside) .+ epsilonDraw)

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

    z = @expression(model, m .+ eut)

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
    u_y = @expression(model,Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran))

    ut_searched_except_y_inf = zeros(N_obs, D)
    ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
    ut_searched_except_y  = @expression(model, ut_searched .+ ut_searched_except_y_inf)

    # u_max = Array{AffExpr,2}(undef,N_cons, D)
    # for i = 1:N_cons
    #     for d = 1:D
    #         u_max[i, d] = @expression(model, maximum(ut[(5*(i-1) + 1):5*i, d]))
    #     end
    # end
    @expression(model,u_max_[i= 1:N_cons, d= 1:D], maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, d]))
    v4 = @expression(model, (u_y - u_max_) ⊗ ones(nalt) )

    #denom
    # @NLexpression(model,denom[i = 1:N_obs, d = 1:D] , 
    #     exp(scaling[1]* v1[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v2[i,d]) * searched[i] * (1 - outside[i]) +
    #     exp(scaling[2]* v3[i,d]) * (1 - searched[i]) * (1 - outside[i])+
    #     exp( scaling[3] *v4[i,d]) * tran[i]
    #     )
    denom = @expression(model, exp.(scaling[1]* v1).*(1 .- outside) .* searched .+
        exp.(scaling[2]* v2).*(1 .- outside) .* searched .+
        exp.(scaling[2]* v3).*(1 .- searched) .*(1 .- outside) .+
        exp.( scaling[3] *v4) .* tran)
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
    @NLexpression(model,prob[i = 1:N_obs, d = 1:D],1 / (1 + denom[i,d]))
    #prob = @expression(model, 1 ./ (1 .+ denom))
    
    

   
    
    

    
    
    # for i = 1:N_cons
    #     for d = 1:D
    #         v4[i, d] = @expression(model, u_y[i,d] - u_max[i,d])
    #     end
    # end
    
    


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
    @NLexpression(model,L_i[i = 1:N_cons], sum(prob[i,d] for d=1:D) + 1e-15)

JuMP.@NLobjective(model, Max, sum(log(L_i[i]) for i = 1:N_cons))

@time JuMP.optimize!(model)

JuMP.value.(params),JuMP.objective_value(model)
    #return 
#end




#example1:
model1 = Model(optimizer_with_attributes(Ipopt.Optimizer))
p0 = [10,6]
@variable(model1,x[1:2])
@NLparameter(model, p[i in 1:2] == p0[i])

p = [1,3,10]
function f_(x,p)
    return (x - p - 1)^2
end
f(3)
register(model,:f_, 2,f_; autodiff = true)
@NLobjective(model, Min, sum(f(x, p) for i in 1:2))
optimize!(model)
println(JuMP.value.(x))



#example2:
consumer = data[:, 1]
N_cons = length(Set(consumer))

#N_prod = data[:, end - 2]
N_prod = data[:, end - 2]
Js = unique(N_prod)
Num_J = length(Js)
consumerData = zeros(N_cons, 2)
consumer_num = 0

i = 1
nalt = Int.(Js[i])
dat = data[N_prod .== nalt, :]
N_obs = size(dat, 1)
uniCons = Int.(N_obs/nalt)
consid2 = reshape(dat[:, 1], nalt, uniCons)

# Generate random draws
Random.seed!(seed)
epsilonDraw = randn(N_obs, D)
etaDraw = randn(N_obs, D)
param0 = zeros(5)
#function liklWeitz_crude_2(param::Vector{Float64}, dat::Matrix{Float64}, D::Int64, nalt::Int64, epsilonDraw::Matrix{Float64}, etaDraw::Matrix{Float64})
function liklWeitz_kernel_2_b(param, dat, D, nalt, epsilonDraw, etaDraw,scaling)
    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = dat[:, end]
    searched = dat[:, end - 1]

    # Parameters
    outside = dat[:, 3]
    c = exp(param[end]) * ones(N_obs)
    X = dat[:, 4:7]
    xb = X * param[1:end-1] 
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    global  table
    m = zeros(N_obs)

    for i = 1:N_obs
        lookupvalue = abs.(table[:, 2] .- c[i])
        if (table[1, 2] >= c[i] && c[i] >= table[end, 2])
            index_m = argmin(lookupvalue)
            m[i] = table[index_m, 1]
        elseif table[1, 2] < c[i]
            m[i] = -c[i] # lower bound m
        elseif c[i] < table[end, 2]
            m[i] = 4.001 # upper bound m
        end
    end
    z = m .+ eut

    ut_searched = copy(ut)
    searched2 = repeat(searched, 1, D)
    ut_searched[searched2 .== 0] .= -Inf
    
    # Selection rule: z > z_next
    z_reshape = reshape(z, nalt, N_cons, D)
    z_max = copy(z_reshape)
    for i in 1:(nalt-2) #not for outside option
        z_max[nalt-i,:,:] = maximum(z_reshape[nalt-i:nalt,:,:] ,dims=1)
    end
    z_max = reshape(z_max,N_obs, D) 
    
    denom_order = exp.(scaling[1].*(z .- z_max)).* searched .* (1 .- outside) #0 for outside option and searched = 0

    # # Stopping rule: z > u_so_far
    ut_searched_reshape = reshape(ut_searched, nalt, N_cons, D)
    u_so_far = copy(ut_searched_reshape)
    for i in 2:nalt
        u_so_far[i, :, :] = max.(u_so_far[i, :, :], u_so_far[i - 1, :, :])
    end
    u_so_far =  circshift(u_so_far, 1) #same as cat(u_so_far[nalt,:,:],u_so_far[1:(nalt - 1),:,:], dims=1) #0.000707
    u_so_far = reshape(u_so_far, N_obs, D)
    
    denom_search1 = 
        #search until
        exp.(scaling[2]*(z .- u_so_far)).*(1 .- outside) .* searched 
        #not search from (search = 1 if outside = 1)
    
    denom_search2 = exp.(scaling[2]*(u_so_far .- z)).*(1 .- searched) #0 for outside option

    # Choice rule
    #u_max = max_{j ∈ S\ y} u_j, where y is the chosen alternative
    #by defining u_max in this way, we can avoid adding small number to u_y - u_max
    u_y = Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran) 
    ut_searched_except_y = copy(ut_searched)
    tran2 = repeat(tran, 1, D) 
    ut_searched_except_y[tran2 .== 1] .= -Inf
    ut_searched_ = reshape(ut_searched_except_y, nalt, N_cons, D)
    u_max = maximum(ut_searched_, dims=1)
    u_max = reshape(u_max, N_cons, D)
    
    #choice = (u_y - u_max .>= 0)
    denom_ch = exp.(scaling[3].*(u_y - u_max) ) #(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
    # Combine all inputs
    denom_order_search_reshape = reshape(denom_order .+ denom_search1 .+ denom_search2 , nalt, N_cons, D)#reshape(denom_order .+ denom_search1 .+ denom_search2, nalt, N_cons, D)
    denom_order_search = sum(denom_order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
    denom_order_search = reshape(denom_order_search, N_cons, D)

    denom =  denom_order_search .+ denom_ch #denom_order_search .+ denom_ch
    
    denfull_t = denom .> 0.0 .&& denom .< 2.2205e-16
    denom[denfull_t] .= 2.2205e-16
    denfull_t2 = denom .>  2.2205e+16
    denom[denfull_t2] .= 2.2205e+16

    prob = 1 ./ (1 .+ denom)
    
    # Average across D
    llk = mean(prob, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))

    println(param)
    println(ll)
    return -ll
end

function liklWeitz_kernel_2_btest(param, dat, D, nalt, epsilonDraw, etaDraw,scaling)
    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = dat[:, end]
    searched = dat[:, end - 1]

    # Parameters
    outside = dat[:, 3]
    c = exp(param[end]) * ones(N_obs)
    X = dat[:, 4:7]
    xb = X * param[1:end-1] 
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    global  table
    m = zeros(N_obs)

    for i = 1:N_obs
        lookupvalue = abs.(table[:, 2] .- c[i])
        if (table[1, 2] >= c[i] && c[i] >= table[end, 2])
            index_m = argmin(lookupvalue)
            m[i] = table[index_m, 1]
        elseif table[1, 2] < c[i]
            m[i] = -c[i] # lower bound m
        elseif c[i] < table[end, 2]
            m[i] = 4.001 # upper bound m
        end
    end
    z = m .+ eut

    ut_searched = copy(ut)
    searched2 = repeat(searched, 1, D)
    ut_searched[searched2 .== 0] .= -Inf
    
    # Selection rule: z > z_next
    z_max = zeros(N_obs, D)
    for i = 1:N_cons
        k = nalt*(i-1)
        for j in 0:(nalt-2) #not for outside option
            z_max[k + nalt-j,:] = maximum(z[(k + nalt-j):(k + nalt),:] ,dims=1)
        end
    end   
    
    denom_order = exp.(scaling[1].*(z .- z_max)).* searched .* (1 .- outside) #0 for outside option and searched = 0
    

    # # Stopping rule: z > u_so_far
    u_so_far = zeros(N_obs, D)
    for i = 1:N_cons
        k = nalt*(i-1) + 1
        for j in 1:(nalt-1)
            u_so_far[k + j, :] = maximum(ut_searched[k:(k + j - 1), :],dims = 1)
        end
    end
    
    denom_search1 = 
        #search until
        exp.(scaling[2]*(z .- u_so_far)).*(1 .- outside) .* searched 
        #not search from (search = 1 if outside = 1)
    
    denom_search2 = exp.(scaling[2]*(u_so_far .- z)).*(1 .- searched) #0 for outside option

    # Choice rule
    #u_max = max_{j ∈ S\ y} u_j, where y is the chosen alternative
    #by defining u_max in this way, we can avoid adding small number to u_y - u_max
    u_y = Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran)

    ut_searched_except_y_inf = zeros(N_obs, D)
    ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
    ut_searched_except_y  = ut_searched .+ ut_searched_except_y_inf

    #not good? => not work
    #u_max = @expression(model,reshape(maximum(reshape(ut_searched_except_y, nalt, N_cons, D), dims=1), N_cons, D))

    # for i = 1:N_cons
    #     for d = 1:D
    #         u_max[i, d] = @expression(model, maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, d]))
    #     end
    # end
    
    u_max = zeros(N_cons, D)
    for i = 1:N_cons
        u_max[i, :] = maximum(ut_searched_except_y[(nalt*(i-1) + 1):nalt*i, :],dims = 1)
    end
    
    
    #choice = (u_y - u_max .>= 0)
    denom_ch = exp.(scaling[3].*(u_y - u_max) ) #(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
    # Combine all inputs
    denom_order_search_reshape = reshape(denom_order .+ denom_search1 .+ denom_search2 , nalt, N_cons, D)#reshape(denom_order .+ denom_search1 .+ denom_search2, nalt, N_cons, D)
    denom_order_search = sum(denom_order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
    denom_order_search = reshape(denom_order_search, N_cons, D)

    denom =  denom_order_search .+ denom_ch #denom_order_search .+ denom_ch
    prob = 1 ./ (1 .+ denom)
    
    
    # Average across D
    llk = mean(prob, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))

    println(param)
    println(ll)
    return -ll
end

function liklWeitz_crude_2_b(param, dat, D, nalt, epsilonDraw, etaDraw)
    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = dat[:, end]
    searched = dat[:, end - 1]

    # Parameters
    outside = dat[:, 3]
    c = exp(param[end]) * ones(N_obs)
    X = dat[:, 4:7]
    xb = X * param[1:end-1] #param[1:end-1]
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    global  table
    m = zeros(N_obs)

    for i = 1:N_obs
        lookupvalue = abs.(table[:, 2] .- c[i])
        if (table[1, 2] >= c[i] && c[i] >= table[end, 2])
            index_m = argmin(lookupvalue)
            m[i] = table[index_m, 1]
        elseif table[1, 2] < c[i]
            m[i] = -c[i] # lower bound m
        elseif c[i] < table[end, 2]
            m[i] = 4.001 # upper bound m
        end
    end

    z = m .+ eut

    ut_searched = copy(ut)
    searched2 = repeat(searched, 1, D)
    ut_searched[searched2 .== 0] .= -Inf
    
    # Selection rule: z > z_next
    z_reshape = reshape(z, nalt, N_cons, D)
    z_max = copy(z_reshape)
    for i in 1:(nalt-2) #not for outside option
        z_max[nalt-i,:,:] = maximum(z_reshape[nalt-i:nalt,:,:] ,dims=1)
    end
    z_max = reshape(z_max,N_obs, D) 
    v1h = (z .- z_max) .* searched .* (1 .- outside) .+
            1.0 .* (1 .- (1 .- outside) .* searched)
    order = (v1h .>= 0)

    # # Stopping rule: z > u_so_far
    ut_searched_reshape = reshape(ut_searched, nalt, N_cons, D)
    u_so_far = copy(ut_searched_reshape)
    for i in 2:nalt
        u_so_far[i, :, :] = max.(u_so_far[i, :, :], u_so_far[i - 1, :, :])
    end
    u_so_far =  circshift(u_so_far, 1) #same as cat(u_so_far[nalt,:,:],u_so_far[1:(nalt - 1),:,:], dims=1) #0.000707
    u_so_far = reshape(u_so_far, N_obs, D)
    #search until
    v2h = (z .- u_so_far) .* (1 .- outside) .* searched .+ 1.0 .* (1 .- (1 .- outside) .* searched)  
    search_1 = (v2h .>= 0 )
    #not search from
    #search = 1 if outside = 1
    v3 = (u_so_far .- z) .* (1 .- searched) .+ 1.0 .* searched
    search_2 = (v3 .>= 0)
 
    
    # Choice rule
    u_y = Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran)
    ut_searched_reshape = reshape(ut_searched, nalt, N_cons, D)
    u_max = maximum(ut_searched_reshape, dims=1)
    u_max = reshape(u_max, N_cons, D)

    choice = (u_y - u_max .>= 0)

    # Combine all inputs
    order_search_reshape = reshape(order .* search_1 .* search_2   , nalt, N_cons, D)#reshape(order .* search_1 .* search_2, nalt, N_cons, D)
    order_search = minimum(order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
    order_search = reshape(order_search, N_cons, D)

    chain_mult =  order_search .* choice#order_search .* choice
    
    # Average across D
    llk = mean(chain_mult, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))
    println(param)
    println(ll)

    return -ll
end

scaling = [-10,-10,-10]
@time liklWeitz_kernel_2_b([25.300569540301815, 10.395989246759356, -11.990564263937848, -12.92572032559815,-3.0], dat, D, nalt, epsilonDraw, etaDraw,scaling) #0.672222 seconds/ 8336.632434540048
@time liklWeitz_kernel_2_btest(param0, dat, D, nalt, epsilonDraw, etaDraw,scaling) #0.262543 seconds
-sum(log.(1e-10 .+ liklWeitz_crude_2(param, dat, D, nalt, epsilonDraw, etaDraw)))
@time liklWeitz_crude_2_b(param, dat, D, nalt, epsilonDraw, etaDraw) #0.311123 seconds
@time liklWeitz_crude_1(param, data, D, seed) #0.899522 seconds

param0 = zeros(5)

@time result_crude = 
    Optim.optimize(
        param -> liklWeitz_crude_1(param, data, D, seed),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )

Optim.minimizer(result_crude);param
@time result_crude_b = 
    Optim.optimize(
        param -> liklWeitz_crude_2_b(param,  dat, D, nalt, epsilonDraw, etaDraw),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )

be_crude_b = Optim.minimizer(result_crude_b)

@time result_kernel = 
    Optim.optimize(
        param -> liklWeitz_kernel_1(param,data,D,scaling,seed),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )
be_kernel = Optim.minimizer(result_kernel)

@time result_kernel_b = 
    Optim.optimize(
        param -> liklWeitz_kernel_2_b(param,  dat, D, nalt, epsilonDraw, etaDraw,scaling),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )



be_kernel_NM = Optim.minimizer(result_kernel_b)
param
param0 = zeros(5)
param0[5] = -3.0
upper = repeat([10.0], 5)
lower = repeat([-10.0], 5)
@time result_kernel_LBFGS = 
    Optim.optimize(
        param -> liklWeitz_kernel_2_b(param,  dat, D, nalt, epsilonDraw, etaDraw,scaling),
        #lower, upper,
        param0,
        #Fminbox(BFGS()),
        BFGS(),
        #LBFGS(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )
        
be_kernel_LBFGS = Optim.minimizer(result_kernel_LBFGS)

scaling = [-100,-50,-10000]

liklWeitz_kernel_1(param,data,D,scaling,seed)
#liklWeitz_kernel_1(be_kernel, data, D,scaling,seed)
liklWeitz_kernel_1(be_crude_b, data, D,scaling,seed)
liklWeitz_kernel_1(be_kernel_NM, data, D,scaling,seed)
liklWeitz_kernel_1(be_kernel_LBFGS, data, D,scaling,seed)

liklWeitz_kernel_2_b([param[1:4];-2.4],  dat, D, nalt, epsilonDraw, etaDraw,scaling)
liklWeitz_kernel_2_b(param,  dat, D, nalt, epsilonDraw, etaDraw,scaling)
#liklWeitz_kernel_2_b(be_kernel,  dat, D, nalt, epsilonDraw, etaDraw,scaling)
liklWeitz_kernel_2_b(be_crude_b,  dat, D, nalt, epsilonDraw, etaDraw,scaling) #choice 936 order 3038 search 2060
liklWeitz_kernel_2_b(be_kernel_NM,  dat, D, nalt, epsilonDraw, etaDraw,scaling)#choice 954  order 3009 search 1970
liklWeitz_kernel_2_b(be_kernel_LBFGS,  dat, D, nalt, epsilonDraw, etaDraw,scaling)

liklWeitz_crude_2_b(param,  dat, D, nalt, epsilonDraw, etaDraw)
#liklWeitz_crude_2_b(be_kernel,  dat, D, nalt, epsilonDraw, etaDraw)
liklWeitz_crude_2_b(be_crude_b,  dat, D, nalt, epsilonDraw, etaDraw)#choice 941 order 2722 search 2129
liklWeitz_crude_2_b(be_kernel_NM,  dat, D, nalt, epsilonDraw, etaDraw)#choice 957 order 2547 search 2005
liklWeitz_crude_2_b(be_kernel_LBFGS,  dat, D, nalt, epsilonDraw, etaDraw)