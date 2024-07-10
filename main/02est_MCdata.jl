#generate data from 01generate_data.jl
using LinearAlgebra
using Kronecker
using Distributions,Random
using CSV, DataFrames, DelimitedFiles, Statistics
using Base.Threads
using Optim
c = exp(-3.0)
scaling = [-18, -4, -7]
param = [1.0, 0.7, 0.5, 0.3, -3.0]
liklWeitz_kernel_2_b(param,  data, D, scaling, epsilonDraw, etaDraw)
liklWeitz_crude_2_b(param, data, D, epsilonDraw, etaDraw)
#function liklWeitz_crude_2(param::Vector{Float64}, dat::Matrix{Float64}, D::Int64, nalt::Int64, epsilonDraw::Matrix{Float64}, etaDraw::Matrix{Float64})
function liklWeitz_kernel_2_b(param, data, D, scaling, epsilonDraw, etaDraw)
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
    c = exp(param[end]) * ones(N_obs)
    X = data[:, 4:7]
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
    u_so_far[nalt,:,:] .= 2.0
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
    
    ut_searched_except_y = copy(ut_searched)
    ut_searched_except_y[repeat(tran, 1, D)  .== 1] .= -Inf

    ut_tran = copy(ut)
    ut_tran[repeat(tran, 1, D)  .== 0] .= -Inf
    
    #u_max = zeros(N_cons, D)
    #u_y = zeros(N_cons, D)
    u_y = zeros(N_obs, D)
    for i = 1:N_cons
       # u_max[i, :] = maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, :],dims = 1)
        u_y[(5*(i-1) + 1):5*i, :] .= maximum( ut_tran[(5*(i-1) + 1):5*i, :],dims = 1)
    end 

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

# function liklWeitz_kernel_2_btest(param, dat, D, nalt, epsilonDraw, etaDraw,scaling)
#     # Data features
#     consumer = dat[:, 1]
#     N_obs = length(consumer)
#     N_cons = length(Set(consumer))

#     # Choices
#     tran = dat[:, end]
#     searched = dat[:, end - 1]

#     # Parameters
#     outside = dat[:, 3]
#     c = exp(param[end]) * ones(N_obs)
#     X = dat[:, 4:7]
#     xb = X * param[1:end-1] 
#     eut = (xb .+ etaDraw) .* (1 .- outside)
#     ut = eut .+ epsilonDraw

#     # Form Z's using look-up table method
#     #table = readdlm("tableZ.csv", ',', Float64)
#     global  table
#     m = zeros(N_obs)

#     for i = 1:N_obs
#         lookupvalue = abs.(table[:, 2] .- c[i])
#         if (table[1, 2] >= c[i] && c[i] >= table[end, 2])
#             index_m = argmin(lookupvalue)
#             m[i] = table[index_m, 1]
#         elseif table[1, 2] < c[i]
#             m[i] = -c[i] # lower bound m
#         elseif c[i] < table[end, 2]
#             m[i] = 4.001 # upper bound m
#         end
#     end
#     z = m .+ eut

#     ut_searched = copy(ut)
#     searched2 = repeat(searched, 1, D)
#     ut_searched[searched2 .== 0] .= -Inf
    
#     # Selection rule: z > z_next
#     z_max = zeros(N_obs, D)
#     for i = 1:N_cons
#         k = nalt*(i-1)
#         for j in 0:(nalt-2) #not for outside option
#             z_max[k + nalt-j,:] = maximum(z[(k + nalt-j):(k + nalt),:] ,dims=1)
#         end
#     end   
    
#     denom_order = exp.(scaling[1].*(z .- z_max)).* searched .* (1 .- outside) #0 for outside option and searched = 0
    

#     # # Stopping rule: z > u_so_far
#     u_so_far = zeros(N_obs, D)
#     for i = 1:N_cons
#         k = nalt*(i-1) + 1
#         for j in 1:(nalt-1)
#             u_so_far[k + j, :] = maximum(ut_searched[k:(k + j - 1), :],dims = 1)
#         end
#     end
    
#     denom_search1 = 
#         #search until
#         exp.(scaling[2]*(z .- u_so_far)).*(1 .- outside) .* searched 
#         #not search from (search = 1 if outside = 1)
    
#     denom_search2 = exp.(scaling[2]*(u_so_far .- z)).*(1 .- searched) #0 for outside option

#     # Choice rule
#     #u_max = max_{j ∈ S\ y} u_j, where y is the chosen alternative
#     #by defining u_max in this way, we can avoid adding small number to u_y - u_max
#     u_y = Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran)

#     ut_searched_except_y_inf = zeros(N_obs, D)
#     ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
#     ut_searched_except_y  = ut_searched .+ ut_searched_except_y_inf

#     #not good? => not work
#     #u_max = @expression(model,reshape(maximum(reshape(ut_searched_except_y, nalt, N_cons, D), dims=1), N_cons, D))

#     # for i = 1:N_cons
#     #     for d = 1:D
#     #         u_max[i, d] = @expression(model, maximum(ut_searched_except_y[(5*(i-1) + 1):5*i, d]))
#     #     end
#     # end
    
#     u_max = zeros(N_cons, D)
#     for i = 1:N_cons
#         u_max[i, :] = maximum(ut_searched_except_y[(nalt*(i-1) + 1):nalt*i, :],dims = 1)
#     end
    
    
#     #choice = (u_y - u_max .>= 0)
#     denom_ch = exp.(scaling[3].*(u_y - u_max) ) #(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
#     # Combine all inputs
#     denom_order_search_reshape = reshape(denom_order .+ denom_search1 .+ denom_search2 , nalt, N_cons, D)#reshape(denom_order .+ denom_search1 .+ denom_search2, nalt, N_cons, D)
#     denom_order_search = sum(denom_order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
#     denom_order_search = reshape(denom_order_search, N_cons, D)

#     denom =  denom_order_search .+ denom_ch #denom_order_search .+ denom_ch
#     prob = 1 ./ (1 .+ denom)
    
    
#     # Average across D
#     llk = mean(prob, dims=2)
#     #return llk
#     ll = sum(log.(1e-10 .+ llk))

#     println(param)
#     println(ll)
#     return -ll
# end


function liklWeitz_crude_2_b(param, data, D, epsilonDraw, etaDraw)
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
    order_search_reshape = reshape(order .* search_1 .* search_2   , 
    nalt, N_cons, D)#reshape(order .* search_1 .* search_2, nalt, N_cons, D)
    order_search = minimum(order_search_reshape, dims=1) #prod(search_2_reshape, dims=1)
    order_search = reshape(order_search, N_cons, D)

    chain_mult =  order_search #.* choice#order_search .* choice
    
    # Average across D
    llk = mean(chain_mult, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))
    println(param)
    println(ll)

    return -ll
end

#monte carlo data
param0 = zeros(5)
param = [1.0, 0.7, 0.5, 0.3, -3.0]
D = 100

#fix seed
seed = 1


#Crude estimator
#create array to store results: params, log-likelihood, run time, convergence flag
#50 monte carlo runs
results_crude = zeros(50, 8)
fin = 0

@time @threads for i = 1:50
    #read csv
    data =  CSV.read("data/genWeitzDataS$i.csv", DataFrame,header=false) |> Matrix{Float64}

    # Generate random draws
    Random.seed!(i)
    epsilonDraw = randn(size(data, 1), D)
    etaDraw = randn(size(data, 1), D)


    run_time = @elapsed begin result_crude_b = 
        Optim.optimize(
            param -> liklWeitz_crude_2_b(param,  data, D, epsilonDraw, etaDraw),
            param,
            NelderMead(),
            #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
            autodiff=:central#,
            #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
            #finite_difference_increment=1e-8
            )#99.947188 seconds
        end
    
    results_crude[i, 1:5] .= Optim.minimizer(result_crude_b)
    results_crude[i, 6] .= -Optim.minimum(result_crude_b)
    results_crude[i, 7] .= run_time #for i = 1:2, draw errors for each evaluation of objective
    results_crude[i, 8] .= Optim.converged(result_crude_b)
    
    fin += 1
    println("finished: ", fin, "/", 50)
end

results_crude = DataFrame(results_crude,:auto)
#column names
rename!(results_crude, names(results_crude) .=> ["beta1", "beta2", "beta3", "beta4", "logc", "loglik", "time", "converged"])

CSV.write("results/results_crude.csv",results_crude, writeheader=false)

result_crude = CSV.read("results/results_crude.csv",DataFrame, header=false)
result_crude = result_crude |> Matrix
mean(result_crude[:,1] .- param[1])
mean(result_crude[:,2] .- param[2])
mean(result_crude[:,3] .- param[3])
mean(result_crude[:,4] .- param[4])
mean(result_crude[:,5] .- param[5])

#standard deviation
std(result_crude[:,1])
std(result_crude[:,2])
std(result_crude[:,3])
std(result_crude[:,4])
std(result_crude[:,5])


#krude
results_kernel = zeros(50, 8)
fin = []
scaling = [-18, -4, -7]
@time @threads for i = 1:50 
    #read csv
    data =  CSV.read("data/genWeitzDataS$i.csv", DataFrame,header=false) |> Matrix{Float64}

    # Generate random draws
    Random.seed!(i)
    epsilonDraw = randn(size(data, 1), D)
    etaDraw = randn(size(data, 1), D)


    run_time = @elapsed begin result = 
        Optim.optimize(
            param -> liklWeitz_kernel_2_b(param,  data, D, scaling, epsilonDraw, etaDraw),
            param,
            NelderMead(),
            #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
            autodiff=:central#,
            #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
            #finite_difference_increment=1e-8
            )#99.947188 seconds
        end
    
    results_kernel[i, 1:5] = Optim.minimizer(result)
    results_kernel[i, 6] = -Optim.minimum(result)
    results_kernel[i, 7] = run_time #for i = 1:2, draw errors for each evaluation of objective
    results_kernel[i, 8] = Optim.converged(result)

    append!(fin, i)
    
    println("finished: ", length(fin), "/", 50)
end

result_kernel = DataFrame(results_kernel,:auto)
#column names
rename!(result_kernel, names(result_kernel) .=> ["beta1", "beta2", "beta3", "beta4", "logc", "loglik", "time", "converged"])

CSV.write("results/results_kernel.csv",result_kernel, writeheader=false)
result_kernel = CSV.read("results/results_kernel.csv",DataFrame, header=false)
[result_kernel[1,:] |> Vector]
result_kernel = result_kernel |> Matrix
mean(result_kernel[:,1] .- param[1])
mean(result_kernel[:,2] .- param[2])
mean(result_kernel[:,3] .- param[3])
mean(result_kernel[:,4] .- param[4])
mean(result_kernel[:,5] .- param[5])

std(result_kernel[:,1])
std(result_kernel[:,2])
std(result_kernel[:,3])
std(result_kernel[:,4])
std(result_kernel[:,5])









#scaling = [-10,-10,-10]
@time liklWeitz_kernel_2_b(param, dat, D, nalt, epsilonDraw, etaDraw,scaling) #0.672222 seconds/ 5007.196627181486
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
        param0,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )#99.947188 seconds

[Optim.minimizer(result_crude_b)]
#[0.7957613433333975, 0.8443758981988342, 0.41763633473963296, 0.17236759142126237, -2.609205389630544]

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
        param -> liklWeitz_kernel_2_b(param,  data, D,scaling, epsilonDraw, etaDraw),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )  #346.455561 seconds 


#result: [0.7117998485990567, 0.44142513980478787, 0.41209526929584883, 0.37021667170942973, -2.44932136162212]
#scaling = [-20, -20, -20]
#[0.513017262232145, 0.24909507916195586, 0.1926051794640007, 0.1715571110832792, -2.9917871477587967]
#scaling = [-10, -5, -20]
#[0.8622056357675261, 0.5672262878918438, 0.4502427956054259, 0.38728259169998425, -2.480362177002503]

[Optim.minimizer(result_kernel_b)]
[param]
#[1.0, 0.7, 0.5, 0.3, -3.0]
[[params_;log(c_)]]
#scaling = [-18, -4, -7]
#unfinished: [1.0030456337591422, 0.8982042350667948, 0.5610812409840122, 0.5487639022089433, -0.783163699966503]

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