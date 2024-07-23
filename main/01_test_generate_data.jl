using Distributions,Random
using CSV, DataFrames, DelimitedFiles, Statistics
using Optim, JLD2, MAT

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


# Data generation
# Number of epsilon+eta draws
#D = parse(Int, filename[17:19])
D = 100

# Seed
#seed = parse(Int, filename[21:end])
seed = 1

# Simulation inputs
N_cons = 10^3  # num of consumers
N_prod = 5     # num of products
param = [1, 0.7, 0.5, 0.3, -3]  # true parameter vector [4 brandFE, search cost constant (exp)]

# Simulate data
data = simWeitz(N_cons, N_prod, param, seed)

#CSV.write("rezSimWeitz_crude_D$D"S$seed.csv", DataFrame(AS), writeheader=false)
