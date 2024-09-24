function simWeitz(N_cons, N_prod, param, table, seed)
    # Set seed for replication
    Random.seed!(seed)

    # Number of observations
    N_obs = N_cons * N_prod
    consumer = repeat(1:N_cons, inner=N_prod)
    N_prod_cons = repeat(1:N_prod, outer=N_cons)

    # Product id
    prod = repeat(1:N_prod, outer=N_cons)

    # Outside option and brandFE
    outside = (prod .== 1)
    brand1 = (prod .== 2)
    brand2 = (prod .== 3)
    brand3 = (prod .== 4)
    brand4 = (prod .== 5)

    # Product characteristics: in this case only brand intercepts
    X = [brand1 brand2 brand3 brand4]
    index_temp = collect(1:length(consumer))
    index_first = index_temp[(index_temp.%N_prod) .== 1] #first entry for each consumer e.g.[1, 6, 11]
    index_last = index_temp[(index_temp.%N_prod) .== 0] #last entry for each consumer e.g.[5, 10, 15]
    
    # Parameters
    c = exp(param[end]) * ones(N_obs) # common search cost
    xb = X * param[1:end-1]
    # Errors
    epsilon = randn(N_obs)
    eta = randn(N_obs)

    # Expected utility and utility
    eut = (xb + eta) .* (1 .- outside)
    ut = eut + epsilon

    # Form Z (reservation utility)
    #table = readdlm("tableZ.csv", ',', Float64, header = true)
    m = zeros(N_obs)
    global table

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

    # Plug in large value for outside option
    z = 100000 * outside .+ z .* (1 .- outside)

    # Order data by z
    da = hcat(consumer, prod, outside, X, eut, ut, z)
    z_order = Int.(zeros(0))
    for i = 1:N_cons
        da[index_first[i]:index_last[i], end]
        temp_z_order = (i-1)*5 .+ sortperm(da[index_first[i]:index_last[i], end], rev=true)
        z_order = vcat(z_order, temp_z_order)
    end
    data = da[z_order, :]

    # Search decision: 1. Outside option always searched
    searched = outside

    # Search decision: 2. Search if z greater than all ut searched so far (because z's are ordered)
    for i = 1:(N_cons - 1)
        # For every product, except outside option
        for j = index_first[i] + 1: index_last[i]
            # Max ut so far
            relevant_ut_sofar = data[index_first[i]:j-1, end - 1] .* searched[index_first[i]:j-1]
            relevant_ut_sofar = filter(x -> x != 0, relevant_ut_sofar)
            max_ut_sofar = maximum(relevant_ut_sofar)

            # Search if z > ut_sofar
            if (data[j, end] > max_ut_sofar)
                searched[j] = 1
            end
        end
    end

    # Transaction: among those searched, pick max ut
    tran = zeros(N_obs)
    searched_ut = data[:, end - 1] .* searched

    for i = 1:(N_cons - 1)
        A = searched_ut[index_first[i]:index_last[i]]
        A[A .== 0] .= -100000
        indexch = argmax(A)
        tran[index_first[i] + indexch - 1] = 1
    end

    # Export data
    length_prod = repeat([N_prod], N_obs)
    searched_mat = reshape(searched, N_prod, N_cons)
    has_searched = searched_mat[2, :] # has_searched = 1 if searched at least one product
    #has_searched = repeat(has_searched,N_prod)
    has_searched = repeat(has_searched,inner=  N_prod)#modified (5th June)
    last = [zeros(N_prod - 1); 1]
    last = repeat(last, N_cons)
    #output = hcat(data[:, 1:end - 2], last, has_searched, length_prod, searched, tran)
    output = hcat(data[:, 1:7], last, has_searched, length_prod, searched, tran)
    #save("genWeitzDataS$seed.mat", Dict("data" => output))
    return output 
end


function Kernel_MPEC(data::Matrix{Float64},scaling::Vector{Int64},D::Int64,maxtime::Float64,max_iter::Int64,tol::Float64 = 1e-6, seed::Int64 = 1)

    #tolerance of constraint/ objective/ parameter
    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>maxtime))
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", tol)
    #global param # initial value list
    @variable(model, 3 >= par[i = 1:4] >= -2,start = 0.0) # to avoid very positive and negative domains
    # @variable(model, par1,start = param[1]) 
    # @variable(model, par2,start = param[2]) 
    # @variable(model, par3,start = param[3]) 
    # @variable(model, par4,start = param[4]) 
    @variable(model, 2.0 >= c >= 0.0,start = exp(0.0)) 
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

function estimate_kernel_NelderMead(data::Matrix{Float64},scaling::Vector{Int64},D::Int64,table::Matrix{Float64},seed::Int64 = 1)
    # Generate random draws
    Random.seed!(seed)
    epsilonDraw = randn(size(data, 1), D)
    etaDraw = randn(size(data, 1), D)

    param = [0.0, 0.0, 0.0, 0.0, 0.0]
    result_kernel = 
        Optim.optimize(
            param -> liklWeitz_kernel(param,  data, D, scaling, table, epsilonDraw, etaDraw),
            param,
            NelderMead(),
            #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
            autodiff=:central#,
            #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
            #finite_difference_increment=1e-8
            )

        return   Optim.minimizer(result_kernel), -Optim.minimum(result_kernel),Optim.converged(result_kernel)
end



function estimate_crude_NelderMead(data::Matrix{Float64},D::Int64,table::Matrix{Float64},seed::Int64 = 1)
    # Generate random draws
    Random.seed!(seed)
    epsilonDraw = randn(size(data, 1), D)
    etaDraw = randn(size(data, 1), D)

    param = [0.0, 0.0, 0.0, 0.0, 0.0]
    result_crude = 
        Optim.optimize(
            param -> liklWeitz_crude(param,  data, D, table, epsilonDraw, etaDraw),
            param,
            NelderMead(),
            #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
            autodiff=:central#,
            #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
            #finite_difference_increment=1e-8
            )

        return   Optim.minimizer(result_crude), -Optim.minimum(result_crude),Optim.converged(result_crude)
end




function liklWeitz_kernel(param, data, D, scaling, table, epsilonDraw, etaDraw)
    # Data features
    consumer = data[:, 1]
    N_cons = length(Set(consumer))

    #N_prod = data[:, end - 2]
    N_prod = data[:, end - 2]
    Js = unique(N_prod)

    i = 1

    nalt = Int.(Js[i])
    N_obs = size(data, 1)
    #uniCons = Int.(N_obs/nalt)
    #consid2 = reshape(dat[:, 1], nalt, uniCons)

    # Choices
    tran = data[:, end]
    searched = data[:, end - 1]
    lasts = data[:, end - 4]

    # Parameters
    outside = data[:, 3]
    c = exp(param[end]) 
    X = data[:, 4:7]
    xb = X * param[1:end-1] 
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    #global  table
    #m = zeros(1)
    
    #for i = 1:N_obs
    lookupvalue = abs.(table[:, 2] .- c)
    if (table[1, 2] >= c && c >= table[end, 2])
        index_m = argmin(lookupvalue)
        m = table[index_m, 1]
    elseif table[1, 2] < c
        m = -c # lower bound m
    elseif c < table[end, 2]
        m = 4.001 # upper bound m
    end
    #end
    z = m .+ eut

    ut_searched = copy(ut)
    searched2 = repeat(searched, 1, D)
    ut_searched[searched2 .== 0] .= -Inf
    
    # Selection rule: z > z_next
    z_reshape = reshape(z, nalt, N_cons, D)
    z_max = copy(z_reshape)
    for i in 1:(nalt-2) #not for outside option
        z_max[nalt-i,:,:] = maximum(z_reshape[nalt-i+1:nalt,:,:] ,dims=1)
    end
    #z_max[:,:,1];z_reshape[:,:,1]
    z_max = reshape(z_max,N_obs, D) 
    
    denom_order = exp.(scaling[1].*(z .- z_max)) .* searched .* (1 .- outside).*(1 .- lasts) #0 for outside option and searched = 0

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

    # u_y = Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran) 
    # ut_searched_except_y = copy(ut_searched)
    # tran2 = repeat(tran, 1, D) 
    # ut_searched_except_y[tran2 .== 1] .= -Inf
    # ut_searched_ = reshape(ut_searched_except_y, nalt, N_cons, D)
    # u_max = maximum(ut_searched_, dims=1)
    # u_max = reshape(u_max, N_cons, D)
    
    # ut_searched_except_y = copy(ut_searched)
    # ut_searched_except_y[repeat(tran, 1, D)  .== 1] .= -Inf
    # ut_tran = copy(ut)
    # ut_tran[repeat(tran, 1, D)  .== 0] .= -Inf
    
    # u_y = zeros(N_obs, D)
    # for i = 1:N_cons
    #    # u_max[i, :] = maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, :],dims = 1)
    #     u_y[(5*(i-1) + 1):5*i, :] .= maximum( ut_tran[(5*(i-1) + 1):5*i, :],dims = 1)
    # end 
    u_y = (Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran)) ⊗ ones(nalt)

    #choice = (u_y - u_max .>= 0)
    denom_ch=exp.(scaling[3].*(u_y - ut)).*(1 .- tran).*searched
    #denom_ch = exp.(scaling[3].*(u_y - u_max) ⊗ ones(nalt)).* tran #(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
    # Combine all inputs
    denom = reshape(denom_order .+denom_search1 .+ denom_search2 .+ denom_ch, 
    #denom = reshape(denom_ch ,
        nalt, N_cons, D)#reshape(denom_order .+ denom_search1 .+ denom_search2, nalt, N_cons, D)
    denom = sum(denom, dims=1) #prod(search_2_reshape, dims=1)
    denom = reshape(denom, N_cons, D)

    #denom =  denom_order_search .+ denom_ch #denom_order_search .+ denom_ch
    
    denfull_t = denom .> 0.0 .&& denom .< 2.2205e-16
    denom[denfull_t] .= 2.2205e-16
    denfull_t2 = denom .>  2.2205e+16
    denom[denfull_t2] .= 2.2205e+16

    prob = 1 ./ (1 .+ denom)
    
    # Average across D
    llk = mean(prob, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))

    #println(param)
    #println(ll)
    return -ll
end

function liklWeitz_crude(param, data, D, table, epsilonDraw, etaDraw)
    # Data features
    consumer = data[:, 1]
    N_cons = length(Set(consumer))

    #N_prod = data[:, end - 2]
    N_prod = data[:, end - 2]
    Js = unique(N_prod)

    i = 1

    nalt = Int.(Js[i])
    N_obs = size(data, 1)
    #uniCons = Int.(N_obs/nalt)
    #consid2 = reshape(dat[:, 1], nalt, uniCons)

    # Choices
    tran = data[:, end]
    searched = data[:, end - 1]

    # Parameters
    outside = data[:, 3]
    c = exp(param[end]) * ones(N_obs)
    X = data[:, 4:7]
    xb = X * param[1:end-1] #param[1:end-1]
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    #global  table
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
    order_search_reshape = reshape(order .* search_1 .* search_2   , 
    nalt, N_cons, D)#reshape(order .* search_1 .* search_2, nalt, N_cons, D)
    order_search = minimum(order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
    order_search = reshape(order_search, N_cons, D)

    chain_mult =  order_search .* choice#order_search .* choice
    
    # Average across D
    llk = mean(chain_mult, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))
    #println(param)
    #println(ll)

    return -ll
end