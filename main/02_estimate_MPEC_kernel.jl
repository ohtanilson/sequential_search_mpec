#generate data from 01_generate_data.jl
using Distributed
#using Optim, JLD2, MAT
Distributed.@everywhere include("../main/00_setting_julia.jl")
Distributed.@everywhere include("../main/00_functions.jl")

# setup for MPEC
maxtime = 300.0 #600.0
max_iter = 500
tol = 1e-3
D = 100
simulation_num = 50
scaling = [-20, -20, -20]
N_cons_vec = [10^3]#[10^3,2*10^3,3*10^3]
# @time res = Kernel_MPEC(data,maxtime,max_iter,tol,1)
#results_MPEC = zeros(100, 7)
results_MPEC_df = DataFrame()

function estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    for N_cons = N_cons_vec
        @show N_cons
        results_MPEC = zeros(simulation_num, 7)
        results_MPEC_term = []
        fin = []
        filename_begin = "../sequential_search_mpec/output/sim_data"
        filename_end   = ".csv"
        filename = filename_begin*"_consumer_"*string(N_cons)*"_error_draw_"*string(D)*filename_end
        data_all = CSV.read(filename, DataFrame) 
        #@time @threads for i = 1:simulation_num # for parallel computation
        @time for i = 1:simulation_num #100 #
            seed = i
            #data
            data = data_all[data_all[:, 1] .== i, 2:end] |> Matrix{Float64}    
            run_time = @elapsed begin 
                    res_mpec = Kernel_MPEC(data,scaling,D,maxtime,max_iter,tol,seed)
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
    
        results_MPEC_df = DataFrame(hcat(results_MPEC[1:simulation_num,:], results_MPEC_term),:auto)
        #column names
        rename!(results_MPEC_df, names(results_MPEC_df) .=> ["beta1", "beta2", "beta3", "beta4", "logc", "loglik", "time", "converged"])
        filename_begin = "../sequential_search_mpec/output/results_MPEC"
        filename_end   = ".csv"
        filename = filename_begin*"_consumer_"*string(N_cons[1])*"_error_draw_"*string(D)*"_scaling_"*string(scaling[1])*string(scaling[2])*string(scaling[3])*filename_end
        CSV.write(filename, results_MPEC_df, writeheader=false)
    
    end
    return results_MPEC_df
end

D_list = [100] #[100, 200]
for D in D_list
    @show D
    @time estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
end


# test for scaling 
if 0 == 1
    scaling = [-18, -4, -7]# ursu et al (2023)
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    scaling = [-50, -50, -50]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    scaling = [-100, -100, -100]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)

    scaling = [-5, -5, -5]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    scaling = [-10, -10, -10]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    scaling = [-2, -2, -2]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
    scaling = [-20, -20, -50]
    @time results_MPEC_df = 
        estimate_kernel_MPEC(maxtime,max_iter,tol,D,simulation_num,scaling, N_cons_vec)
end

    


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