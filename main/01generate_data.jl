using Optim, Random, Distributions, CSV, DataFrames, DelimitedFiles, Statistics
using JLD2, MAT

# construct look-up table
m = -3.55:0.001:4
c = similar(m)
i = 1
for j in m
    c[i] = (1 - cdf(Normal(), j)) * ((pdf(Normal(), j) / (1 - cdf(Normal(), j))) - j)
    global i += 1
end
table = hcat(m, c)
tableZ = copy(table)
#CSV.write("tableZ.csv", DataFrame(table, :auto), header=["m", "c"], delim=',')


function simWeitz(N_cons, N_prod, param, seed)
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
    has_searched = searched_mat[2, :]
    has_searched = repeat(has_searched, N_prod)
    last = [zeros(N_prod - 1); 1]
    last = repeat(last, N_cons)
    #output = hcat(data[:, 1:end - 2], last, has_searched, length_prod, searched, tran)
    output = hcat(data[:, 1:7], last, has_searched, length_prod, searched, tran)
    #save("genWeitzDataS$seed.mat", Dict("data" => output))
    return output 
end

function liklWeitz_crude_1(param, data, D, seed, epsilonDraw, etaDraw)
    consumer = data[:, 1]
    N_cons = length(Set(consumer))

    #N_prod = data[:, end - 2]
    N_prod = data[:, end - 2]
    Js = unique(N_prod)
    Num_J = length(Js)
    consumerData = zeros(N_cons, 2)
    consumer_num = 0

    # Construct likelihood for consumers with the same number of searches
    for i = 1:Num_J
        nalt = Int.(Js[i])
        dat = data[N_prod .== nalt, :]
        N_obs = size(dat, 1)
        uniCons = Int.(N_obs/nalt)
        consid2 = reshape(dat[:, 1], nalt, uniCons)

        # chosen consumer id and his likelihood
        consumerData[consumer_num + 1:consumer_num + uniCons, 1] .= consid2[1, :]
        consumerData[consumer_num + 1:consumer_num + uniCons, 2] .= liklWeitz_crude_2(param, dat, D, nalt, epsilonDraw, etaDraw)
        consumer_num += uniCons
    end

    # Sum over consumers
    # To guarantee llk is not zero within log
    llk = -sum(log.(1e-10 .+ consumerData[:, 2]))

    # Check for errors or save output
    if isnan(llk) || llk == Inf || llk == -Inf || !isreal(llk)
        loglik = 1e+300
    else
        loglik = llk
        println(param)
        println(loglik)
        paramLL = [param; loglik]
        # Save preliminary output
        #CSV.write("betaWeitz_crude_D$D""S$seed.csv", DataFrame(paramLL), writeheader=false)
    end

    return loglik
end

function liklWeitz_crude_2(param, dat, D, nalt, epsilonDraw, etaDraw)
    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = dat[:, end]
    searched = dat[:, end - 1]
    has_searched = dat[:, end - 3]
    last = dat[:, end - 4]
    
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
    ut_searched[searched2 .== 0] .= -9999
    prob = zeros(N_cons, D)
    for d = 1:D
        # Best ut_so_far
        # ymax = cummax(reshape(ut_searched[:, d], nalt, N_cons));
        ut_matrix = reshape(ut_searched[:, d], nalt, N_cons)
        for i = 1:N_cons
            temp_ymax = ut_matrix[:,i]
            temp_ymax = [maximum(temp_ymax[1:i]) for i = 1:length(temp_ymax)] 
            if i == 1
                ymax = temp_ymax
            else
                ymax = hcat(ymax, temp_ymax)
            end
        end
        # move outside to tail
        ymax = circshift(ymax, 1); 
        ymax = reshape(ymax, N_obs, 1);
    
        # Best z_next
        #zmax = Statistics.cummax(reshape(z(:, d), nalt, N_cons), 'reverse');
        z_matrix = reshape(z[:, d], nalt, N_cons)
        for i = 1:N_cons
            temp_zmax = reverse(z_matrix[:,i])
            temp_zmax = reverse([maximum(temp_zmax[1:i]) for i = 1:length(temp_zmax)])
            if i == 1
                zmax = temp_zmax
            else
                zmax = hcat(zmax, temp_zmax)
            end
        end
        zmax = circshift(zmax, -1);
        zmax = reshape(zmax, N_obs, 1);
    
        # Outside option for each consumer
        u0_2 = ut[:, d] .* outside
        u0_3 = reshape(u0_2, nalt, N_cons)
        u0_4 = repeat(sum(u0_3, dims=1), nalt, 1)
        u0_5 = reshape(u0_4, N_obs, 1)
    
        # Selection rule: z > z_next
        supp_var = ones(size(dat, 1), 1)
        order = (z[:, d] .- zmax) .* has_searched .* searched .* (1 .- outside) .* (1 .- last) .+
                supp_var .* last .+ 
                supp_var .* outside .+ 
                supp_var .* (1 .- has_searched) .+
                supp_var .* (1 .- searched)
        order .= order .> 0
    
        # Stopping rule: z > u_so_far
        search_1 = (z[:, d] .- ymax) .* has_searched .* searched .* (1 .- outside) .+
            supp_var .* outside .+
            supp_var .* (1 .- searched) .+
            supp_var .* (1 .- has_searched)
        search_1 .= search_1 .> 0
        search_2 = (ymax .- z[:, d]) .* has_searched .* (1 .- searched) .+
            supp_var .* (1 .- has_searched) .+
            supp_var .* searched
        search_2 .= search_2 .> 0
        search_3 = (u0_5 .- z[:, d]) .* (1 .- has_searched) .* (1 .- outside) .+
            supp_var .* has_searched .+
            supp_var .* outside
        search_3 .= search_3 .> 0
    
        # Choice rule
        u_ch2 = ut[:, d] .* tran
        u_ch3 = reshape(u_ch2, nalt, N_cons)
        u_ch4 = repeat(sum(u_ch3, dims=1), nalt, 1)
        u_ch5 = reshape(u_ch4, N_obs, 1)
        choice = (u_ch5 .- ut[:, d]) .* (1 .- tran) .* searched .+
            supp_var .* tran .+
            supp_var .* (1 .- searched)
        choice .= choice .> 0
    
        # Combine all inputs
        chain_mult = order .* search_1 .* search_2 .* search_2 .* search_3 .* choice;
    
        # Sum at the consumer level
        #final_result = accumarray(consumer, chain_mult, [N_cons 1], @prod);
        final_result = zeros(N_cons, 1)
        for i = 1:N_cons
            final_result[i] = Base.prod(chain_mult[consumer .== i])
        end
        # Probability for that d
        prob[:, d] = final_result;
    end
    
    # Average across D
    llk = mean(prob, dims=2);
    return llk
end

randn(1)
# ESTIMATION CODE - Crude frequency simulator - Weitzman, UrsuSeilerHonka 2022

# Options for estimation
#options = Optim.Options(g_tol=1e-6, f_tol=1e-6, iterations=6000000, show_trace=false)

# Setting parameters (obtained from file name)
#filename = string(Base.source_path())[end-22:end-2]  # Assuming the file name length is constant

# Number of epsilon+eta draws
#D = parse(Int, filename[17:19])
D = 100

# Seed
#seed = parse(Int, filename[21:end])
seed = 1

# Simulation inputs
N_cons = 1000  # num of consumers
N_prod = 5     # num of products
param = [1, 0.7, 0.5, 0.3, -3]  # true parameter vector [4 brandFE, search cost constant (exp)]

# Simulate data
data = simWeitz(N_cons, N_prod, param, seed)
#data = MAT.matread("genWeitzDataS1.mat")["output"]
# etaDraw = MAT.matread("etaDraw.mat")["etaDraw"]
# epsilonDraw = MAT.matread("epsilonDraw.mat")["epsilonDraw"]
# Load simulated data
# data = load("genWeitzDataS$seed.mat")
# data = data["data"]

# Estimation
# Initial parameter vector
param0 = zeros(size(param))

# Generate random draws
Random.seed!(seed)
epsilonDraw = randn(N_obs, D)
etaDraw = randn(N_obs, D)

# Define likelihood function
# function liklWeitz_crude_1_for_Optim(param)
#     # Call the liklWeitz_crude_1 function with appropriate arguments
#     # (Implementation of this function is assumed to be available)
#     # Replace the following line with the actual function call
#     return liklWeitz_crude_1(param, data, D, seed)
# end

# Perform estimation
@time result = 
    Optim.optimize(
        param -> liklWeitz_crude_1(param, data, D, seed, epsilonDraw, etaDraw),
        param0,
        #BFGS(),
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )


# Extract results
be = Optim.minimizer(result)
val = Optim.minimum(result)
exitflag = Optim.converged(result)

# Compute standard errors
hessian_inv = inv(Optim.hessian(result))
se = sqrt.(diag(hessian_inv))

# Save results
AS = hcat(be, se, val, exitflag)
#CSV.write("rezSimWeitz_crude_D$D"S$seed.csv", DataFrame(AS), writeheader=false)
