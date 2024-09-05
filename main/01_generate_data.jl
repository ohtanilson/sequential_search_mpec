
using Distributed
#using Optim, JLD2, MAT
Distributed.@everywhere include("../main/00_setting_julia.jl")
Distributed.@everywhere include("../main/00_functions.jl")

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
#Plots.plot(table)
#CSV.write("tableZ.csv", DataFrame(table, :auto), header=["m", "c"], delim=',')


# ESTIMATION CODE - Crude frequency simulator - Weitzman, UrsuSeilerHonka 2022

# Options for estimation
#options = Optim.Options(g_tol=1e-6, f_tol=1e-6, iterations=6000000, show_trace=false)

# Setting parameters (obtained from file name)
# Simulation inputs
#N_cons = 10^3  # num of consumers
N_prod = 5     # num of products
param = [1, 0.7, 0.5, 0.3, -3]  # true parameter vector [4 brandFE, search cost constant (exp)]
simulation_num = 50
N_cons_vec = [500,10^3,2*10^3,3*10^3]

# Simulate data
simulate_data = function(N_cons_vec, D, N_prod, param, table)
    for N_cons in N_cons_vec
        @show N_cons
        data = simWeitz(N_cons, N_prod, param, table, 1)
        data = hcat(repeat([1],size(data,1)),data)
        @time for i = 2:simulation_num
            seed = i
            data_i = simWeitz(N_cons, N_prod, param, table, seed)
            data_i = hcat(repeat([i],size(data_i,1)),data_i)
            data = vcat(data,data_i)
        end
        filename_begin = "../sequential_search_mpec/output/sim_data"
        filename_end   = ".csv"
        # if sigma == 1 || sigma == 2
        #     sigma = Int64(sigma)
        # end
        filename = filename_begin*"_consumer_"*string(N_cons)*"_error_draw_"*string(D)*filename_end
        CSV.write(filename, Tables.table(data))
    end
end


# Number of epsilon+eta draws
Random.seed!(1)
D_list = [200]#[100, 200]
for D in D_list
    @show D
    simulate_data(N_cons_vec, D, N_prod, param, table)
end

#dataa = CSV.read("data/sim_data_100.csv", DataFrame) |> Matrix



# Estimation
# Initial parameter vector
param0 = zeros(size(param))



# Define likelihood function
# function liklWeitz_crude_1_for_Optim(param)
#     # Call the liklWeitz_crude_1 function with appropriate arguments
#     # (Implementation of this function is assumed to be available)
#     # Replace the following line with the actual function call
#     return liklWeitz_crude_1(param, data, D, seed)
# end

# Perform estimation
if 0 == 1
    # Generate random draws
    Random.seed!(seed)
    epsilonDraw = randn(N_obs, D)
    etaDraw = randn(N_obs, D)
    
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
end

#CSV.write("rezSimWeitz_crude_D$D"S$seed.csv", DataFrame(AS), writeheader=false)
