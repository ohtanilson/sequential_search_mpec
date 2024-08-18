#generate data from 01_generate_data.jl
using Distributed
#using Optim, JLD2, MAT
Distributed.@everywhere include("../main/00_setting_julia.jl")
Distributed.@everywhere include("../main/00_functions.jl")

# setup for nested crude
D = 100
simulation_num = 50
N_cons_vec = [10^3,2*10^3,3*10^3]
table = readdlm("data/tableZ.csv", ',', Float64)

function estimate_crude_nested(D,simulation_num, N_cons_vec,table)
    for N_cons = N_cons_vec
        @show N_cons
        results_crude_nested = zeros(simulation_num, 7)
        results_crude_nested_term = []
        fin = []
        filename_begin = "../sequential_search_mpec/output/sim_data"
        filename_end   = ".csv"
        filename = filename_begin*"_consumer_"*string(N_cons)*"_error_draw_"*string(D)*filename_end
        data_all = CSV.read(filename, DataFrame) 
        #@time @threads for i = 1:simulation_num # for parallel computation
        @time for i = 1:simulation_num #100 #
            #data
            data = data_all[data_all[:, 1] .== i, 2:end] |> Matrix{Float64}    
            run_time = @elapsed begin 
                    res_crude_nested = estimate_crude_NelderMead(data,D,table,i)
                end
            
            results_crude_nested[i, 1:5] .= res_crude_nested[1]
            results_crude_nested[i, 6:6] .= res_crude_nested[2]
            results_crude_nested[i, 7:7] .= run_time 
            append!(results_crude_nested_term,[res_crude_nested[3]]) 
            
            append!(fin, i)
            println("finished: ", length(fin), "/", length(N_cons_vec))
            GC.gc()
            GC.gc()
        end
    
        results_crude_nested_df = DataFrame(hcat(results_crude_nested[1:simulation_num,:], results_crude_nested_term),:auto)
        #column names
        rename!(results_crude_nested_df, names(results_crude_nested_df) .=> ["beta1", "beta2", "beta3", "beta4", "logc", "loglik", "time", "converged"])
        filename_begin = "../sequential_search_mpec/output/results_nested_crude"
        filename_end   = ".csv"
        filename = filename_begin*"_consumer_"*string(N_cons[1])*"_error_draw_"*string(D)*filename_end
        CSV.write(filename, results_crude_nested_df, writeheader=false)
    
    end
    return results_crude_nested_df
end

@time results_crude_nested_df = 
    estimate_crude_nested(D,simulation_num, N_cons_vec,table)

    


param = [1.0, 0.7, 0.5, 0.3, -3.0]
mean(results_crude_nested_df[:,1] .- param[1])
mean(results_crude_nested_df[:,2] .- param[2])
mean(results_crude_nested_df[:,3] .- param[3])
mean(results_crude_nested_df[:,4] .- param[4])
mean(results_crude_nested_df[:,5] .- param[5])

#standard deviation
std(results_crude_nested_df[:,1])
std(results_crude_nested_df[:,2])
std(results_crude_nested_df[:,3])
std(results_crude_nested_df[:,4])
std(results_crude_nested_df[:,5])
