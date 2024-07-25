#generate data from 01generate_data.jl
using LinearAlgebra
using Kronecker
using Distributions,Random
using CSV, DataFrames, DelimitedFiles, Statistics
using Base.Threads
using Optim
using Plots

lookupvalue = abs.(table[:, 2] .- c)
if (table[1, 2] >= c && c >= table[end, 2])
    index_m = argmin(lookupvalue)
    m = table[index_m, 1]
elseif table[1, 2] < c
    m = -c # lower bound m
elseif c < table[end, 2]
    m = 4.001 # upper bound m
end

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
    c = exp(param[end]) 
    X = data[:, 4:7]
    xb = X * param[1:end-1] 
    eut = (xb .+ etaDraw) .* (1 .- outside)
    ut = eut .+ epsilonDraw

    # Form Z's using look-up table method
    #table = readdlm("tableZ.csv", ',', Float64)
    global  table
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

    chain_mult =  order_search .* choice#order_search .* choice
    
    # Average across D
    llk = mean(chain_mult, dims=2)
    #return llk
    ll = sum(log.(1e-10 .+ llk))
    #println(param)
    #println(ll)

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
#[result_kernel[1,:] |> Vector]
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


#shape of likelihood
c_vec = collect(-5.0:0.1:5.0)
scaling = [-18, -4, -7]
Random.seed!(1)
epsilonDraw_l = randn(size(data, 1), D)
etaDraw_l = randn(size(data, 1), D)
ll_store = zeros(length(c_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    ll_store[i] = liklWeitz_kernel_2_b([param[1:4];c_vec[i]],  data, D, scaling, epsilonDraw_l, etaDraw_l)
    append!(fin, i)
    
    println("finished: ", length(fin), "/", length(c_vec))
    GC.gc()
end

#plot
c_vec = collect(-5.0:0.1:5.0)
plot(c_vec, -ll_store, label="log-likelihood", xlabel="log c", ylabel="log-likelihood", title="Log-likelihood vs log c")
vline!([param[5]], label="true log c", color=:red)
vline!([c_vec[argmin(ll_store)]], label="argmax ll", color=:blue)

c_vec = collect(-5.0:0.1:5.0)
plot(exp.(c_vec[1:40]), -ll_store[1:40], label="log-likelihood", xlabel="c", ylabel="log-likelihood", title="Log-likelihood vs c")
vline!([exp(param[5])], label="true c", color=:red)
vline!([exp(c_vec[argmin(ll_store)])], label="argmax ll", color=:blue)


#plot(c_vec[1:50], -ll_store[1:50], label="log-likelihood", xlabel="c", ylabel="log-likelihood", title="Log-likelihood vs c")


#different seeds
c_vec = collect(-4.0:0.2:-1.0)
seed_vec = collect(1:10)

ll_store2 = zeros(length(c_vec),length(seed_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    for j = 1:length(seed_vec)
        Random.seed!(seed_vec[j])
        epsilonDraw_l = randn(size(data, 1), D)
        etaDraw_l = randn(size(data, 1), D)
        ll_store2[i,j] .= liklWeitz_kernel_2_b([param[1:4];c_vec[i]],  data, D, scaling, epsilonDraw_l, etaDraw_l)

        append!(fin, i)
        println("finished: ", length(fin), "/", length(c_vec)*length(seed_vec))
        GC.gc()
    end
end

#plot
c_vec = collect(-4.0:0.2:-1.0)
plot(c_vec, -ll_store2[:,1], label="log-likelihood", xlabel="c", ylabel="log-likelihood", title="Log-likelihood vs c",color=:black,legend = false)
vline!([c_vec[argmin(ll_store2[:,1])]])
for i = 2:10
    plot!(c_vec, -ll_store2[:,i],col = "black",color=:black)
    vline!([c_vec[argmin(ll_store2[:,i])]], label="argmax ll")
end
vline!([param[5]], label="true c", color=:red)
#error draw doesn't matter

#different datasets
c_vec = collect(-3.8:0.05:-1.6)
seed_vec = collect(1:50*length(c_vec))
ll_store3 = zeros(length(c_vec),length(seed_vec))
c_vec_store = zeros(length(c_vec),length(seed_vec))
fin = []
k = 1
@time for i = 1:length(c_vec) #@threads 
    for j = 2:10
        data =  CSV.read("data/genWeitzDataS$i.csv", DataFrame,header=false) |> Matrix{Float64}
        Random.seed!(seed_vec[k])
        epsilonDraw = randn(size(data, 1), D)
        etaDraw = randn(size(data, 1), D)
        c = c_vec[i] + 0.05*rand(1)[1]
        c_vec_store[i,j] = c
        ll_store3[i,j] .= liklWeitz_kernel_2_b([param[1:4];c],  data, D, scaling, epsilonDraw, etaDraw)

        append!(fin, i)
        println("finished: ", length(fin), "/", length(c_vec)*50)
        GC.gc()
        k += 1
    end
end

#plot
plot(c_vec_store[:,1], -ll_store3[:,1], label="log-likelihood", xlabel="c", ylabel="log-likelihood", title="Log-likelihood vs c",color=:black,legend = false)
vline!([c_vec_store[:,1][argmin(ll_store3[:,1])]])
for i = 2:9
    plot!(c_vec_store[:,i], -ll_store3[:,i],col = "black",color=:black)
    vline!([c_vec_store[:,i][argmin(ll_store3[:,i])]], label="argmax ll")
end
i = 10
plot!(c_vec_store[:,i], -ll_store3[:,i],col = "black",color=:black)
vline!([c_vec_store[:,i][argmin(ll_store3[:,i])]], label="argmax ll")


#shape of likelihood (crude)
c_vec = collect(-5.0:0.1:-1.0)
Random.seed!(1)
epsilonDraw = randn(size(data, 1), D)
etaDraw = randn(size(data, 1), D)
ll_store_c1 = zeros(length(c_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    ll_store_c1[i] .= liklWeitz_crude_2_b([param[1:4];c_vec[i]],  data, D, epsilonDraw, etaDraw)
    append!(fin, i)
    
    println("finished: ", length(fin), "/", length(c_vec))
    GC.gc()
end

c_vec = collect(-5.0:0.1:-1.0)
plot(c_vec, -ll_store_c1, label="log-likelihood", xlabel="log c", ylabel="log-likelihood ", title="Crude Log-likelihood vs log c")
vline!([param[5]], label="true log c", color=:red)
vline!([c_vec[argmin(ll_store_c1)]], label="argmax ll", color=:blue)

#different scaling (kernel)
scaling = [-1, -1, -1]
ll_store_scaling = zeros(length(c_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    ll_store_scaling[i] = liklWeitz_kernel_2_b([param[1:4];c_vec[i]],  data, D, scaling, epsilonDraw, etaDraw)
    append!(fin, i)
    
    println("finished: ", length(fin), "/", length(c_vec))
    GC.gc()
end

#plot
c_vec = collect(-5.0:0.1:-1.0)
plot(c_vec, -ll_store_scaling, label="log-likelihood", xlabel="log c", ylabel="log-likelihood", title="Scaling $scaling")
vline!([param[5]], label="true log c", color=:red)
vline!([c_vec[argmin(ll_store_scaling)]], label="argmax ll", color=:blue)

#scaling 2
scaling = [-10, -10, -10]
ll_store_scaling2 = zeros(length(c_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    ll_store_scaling2[i] = liklWeitz_kernel_2_b([param[1:4];c_vec[i]],  data, D, scaling, epsilonDraw, etaDraw)
    append!(fin, i)
    
    println("finished: ", length(fin), "/", length(c_vec))
    GC.gc()
end

#plot
c_vec = collect(-5.0:0.1:-1.0)
plot(c_vec, -ll_store_scaling2, label="log-likelihood", xlabel="log c", ylabel="log-likelihood", title="Scaling $scaling")
vline!([param[5]], label="true log c", color=:red)
vline!([c_vec[argmin(ll_store_scaling2)]], label="argmax ll", color=:blue)

#scaling 3
scaling = [-20, -20, -20]
ll_store_scaling3 = zeros(length(c_vec))
fin = []
@time for i = 1:length(c_vec) #@threads 
    ll_store_scaling3[i] = liklWeitz_kernel_2_b([param[1:4];c_vec[i]],  data, D, scaling, epsilonDraw, etaDraw)
    append!(fin, i)
    
    println("finished: ", length(fin), "/", length(c_vec))
    GC.gc()
end

#plot
c_vec = collect(-5.0:0.1:-1.0)
plot(c_vec, -ll_store_scaling3, label="log-likelihood", xlabel="log c", ylabel="log-likelihood", title="Scaling $scaling")
vline!([param[5]], label="true log c", color=:red)
vline!([c_vec[argmin(ll_store_scaling3)]], label="argmax ll", color=:blue)