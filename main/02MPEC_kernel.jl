#generate data from 01generate_data.jl
using LinearAlgebra
using Kronecker
using JuMP
using Ipopt
using Distributions
using CSV, DataFrames, DelimitedFiles, Statistics

scaling = [-20, -20, -20]
i = 1
seed = i
data =  CSV.read("data/genWeitzDataS$i.csv", DataFrame,header=false) |> Matrix{Float64}
D = 100

function Kernel_MPEC(maxtime::Float64,max_iter::Int64,tol::Float64 = 1e-6)

    #tolerance of constraint/ objective/ parameter
    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>maxtime))
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", tol)
    #global param # initial value list
    @variable(model, par[i = 1:4],start = 1.0) 
    # @variable(model, par1,start = param[1]) 
    # @variable(model, par2,start = param[2]) 
    # @variable(model, par3,start = param[3]) 
    # @variable(model, par4,start = param[4]) 
    @variable(model, 2.0 >= c >= 0.0,start = exp(-3.0)) 
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

    
    return JuMP.value.(par),JuMP.value.(c),JuMP.value.(m),JuMP.objective_value(model)
end
maxtime = 100.0 # => 1042 iterations
max_iter = 300
#not enough: maxtime = 3600.0, max_iter = 6000
#maxtime was binding first, and max_iter was around 2500(?) at end 
@time res = [Kernel_MPEC(maxtime,max_iter,1e-2)]
[[params_;log(c_)]] #[0.5872889646907962, 0.34346327406010513, 0.3724346107126989, 0.05837708724133401, -2.7583287415593416]
objval_MPEC #-4593.1152076843055
#julia benchmark:  [0.5863349297766002, 0.34251081276373263, 0.3715297786882639, 0.05747329606564218, -2.7592246317122147],  -4593.115152055782

Random.seed!(1)
epsilonDraw = randn(size(data, 1), D)
etaDraw = randn(size(data, 1), D)
result_k = 
    Optim.optimize(
        param -> liklWeitz_kernel_2_b(param,  data, D, scaling, epsilonDraw, etaDraw),
        param,
        NelderMead(),
        #BFGS(),LBFGS(), ConjugateGradient(), NelderMead(), Newton(), GradientDescent(), SimulatedAnnealing(), ParticleSwarm()
        autodiff=:central#,
        #optimizer = with_linesearch(BFGS(), Optim.HagerZhang()),
        #finite_difference_increment=1e-8
        )#99.947188 seconds
result_k.minimizer
GC.gc()


# function Kernel_MPEC2(maxtime::Float64,max_iter::Int64)

#     model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>maxtime))
#     set_optimizer_attribute(model, "max_iter", max_iter)
#     #global param # initial value list
#     @variable(model, par[i = 1:4],start = 1.0) 
#     @variable(model, c >= 0.0,start = exp(-3.0)) 
#     @variable(model, m) #scaler

#     # Data features
#     consumer = data[:, 1]
#     N_obs = length(consumer)
#     N_cons = length(Set(consumer))

#     N_prod = data[:, end - 2]
#     Js = unique(N_prod)
#     nalt = Int.(Js[1])

#     #error draws
#     Random.seed!(seed)
#     epsilonDraw = randn(N_obs, D)
#     etaDraw = randn(N_obs, D)

#     # Choices
#     tran = data[:, end]
#     searched = data[:, end - 1]
#     lasts = data[:, end - 4]
#     outside = data[:, 3] 

#     #reshape
#     tran = reshape(tran, nalt, N_cons)
#     searched = reshape(searched, nalt, N_cons)
#     lasts = reshape(lasts, nalt, N_cons)
#     outside = reshape(outside, nalt, N_cons)
#     epsilonDraw = reshape(epsilonDraw, nalt, N_cons, D)
#     etaDraw = reshape(etaDraw, nalt, N_cons, D)

#     #@NLexpression(model,c, exp(params[end]) )
#     X = data[:, 4:7]
#     X_reshape = reshape(X, nalt, N_cons, size(X, 2))
#     X_reshape2 = reshape(X_reshape[1,:,:]',1, N_cons* size(X, 2))
#     for i = 2:nalt
#         X_reshape2 = vcat(X_reshape2, reshape(X_reshape[i,:,:]',1, N_cons* size(X, 2)))
#     end
#     #X_reshape2 * (Diagonal(ones(N_cons))⊗ param[1:4])
#     Xb_resh = @expression(model, X_reshape2 * (Diagonal(ones(N_cons))⊗ par))
#     eut = @expression(model, (Xb_resh .+ etaDraw) .* (1 .- outside))

#     ut = @expression(model,eut .+ epsilonDraw)
#     z = @expression(model, m .+ eut)
#     # global  table
#     # @NLconstraint(model, 
#     #     ifelse(
#     #         table[1, 2] >= c && c >= table[end, 2], m - table[argmin(abs.(table[:, 2] .- c)), 1],
#     #         ifelse(table[1, 2] < c, m + c, m - 4.001)
#     #         ) == 0)
#     norm_cdf(x) = cdf(Normal(), x)
#     norm_pdf(x) = pdf(Normal(), x)
#     register(model, :norm_cdf, 1, norm_cdf; autodiff = true)
#     register(model, :norm_pdf, 1, norm_pdf; autodiff = true)
#     @NLconstraint(model, c == norm_pdf(m) + m *(norm_cdf(m) - 1))

#     ut_unsearched = zeros(nalt, N_cons, D)
#     searched2 = repeat(searched, 1, 1,D)
#     ut_unsearched[searched2 .== 0] .= -Inf
#     ut_searched = @expression(model, ut .+ ut_unsearched)
    
#     # Selection rule: z > z_next
#     z_max = Array{NonlinearExpr,3}(undef,nalt,N_cons, D)
#     z_max[nalt,:,:] = @expression(model,z[nalt,:,:])
#     #for i = 1:N_cons
#     #k = nalt*(i-1) #last alternative for the previous consumer
#     for j in 1:(nalt-1) #not for outside option
#         for d = 1:D
#             z_max[nalt-j,:,d] = @expression(model,maximum(z[nalt-j+1:nalt,:,d],dims=1))
#         end
#     end
#     #end   
    
#     v1 = @expression(model, z - z_max)
    
#     # # Stopping rule: z > u_so_far
#     u_so_far = Array{NonlinearExpr,3}(undef,nalt,N_cons, D)
#     #for i = 1:N_cons
#         #k = nalt*(i-1) + 1 #first alternative
#         #u_so_far[k,:] = @expression(model,ut_searched[k + (nalt-1),:]) #does not matter for optimization
#     u_so_far[1,:,:] = @expression(model,ut_searched[1,:,:])
#     u_so_far[2,:,:] = @expression(model,ut_searched[1,:,:])
#     for j in 3:nalt
#         for d = 1:D
#             u_so_far[j,:, d] = @expression(model,max.(ut_searched[j-1,:, d],u_so_far[j-1,:, d])) #maximum(ut_searched[1:(j - 1),:, d]))
#         end
#     end
#     #end
#     v2 = @expression(model, z .- u_so_far)
#     v3 = @expression(model, u_so_far .- z)
    
#     # Choice rule
#     #u_max = max_{j ∈ S\ y} u_j, where y is the chosen alternative
#     #by defining u_max in this way, we can avoid adding small number to u_y - u_max
#     #u_y = @expression(model,Diagonal(ones(N_cons)) ⊗ ones(1,nalt) * (ut.* tran))

#     # ut_searched_except_y_inf = zeros(N_obs, D)
#     # ut_searched_except_y_inf[repeat(tran, 1, D)  .== 1] .= -Inf
#     # ut_searched_except_y  = @expression(model, ut_searched .+ ut_searched_except_y_inf)

#     ut_tran_inf = zeros(nalt,N_cons, D)
#     ut_tran_inf[repeat(tran, 1,1, D)  .== 0] .= -Inf
#     ut_tran = @expression(model, ut .+ ut_tran_inf)
#     # u_max = Array{NonlinearExpr,2}(undef,N_cons, D)
#     u_y = Array{NonlinearExpr,3}(undef,nalt,N_cons, D)
#     #for i = 1:N_cons
#     for d = 1:D
#         u_y[:,:, d] .= @expression(model, maximum( ut_tran[:,:, d],dims = 1)) 
#     end
#     #end
#     v4 = @expression(model, (u_y - ut) )

#     #denom
#     # @NLexpression(model,denom[i = 1:N_obs, d = 1:D] , 
#     #     exp(scaling[1]* v1[i,d]) * searched[i] * (1 - outside[i]) +
#     #     exp(scaling[2]* v2[i,d]) * searched[i] * (1 - outside[i]) +
#     #     exp(scaling[2]* v3[i,d]) * (1 - searched[i]) * (1 - outside[i])+
#     #     exp( scaling[3] *v4[i,d]) * tran[i]
#     #     )
#     denom = @expression(model, exp.(scaling[1]* v1).*(1 .- outside) .* searched .*(1 .- lasts).+
#          exp.(scaling[2]* v2).*(1 .- outside) .* searched .+
#          exp.(scaling[2]* v3).*(1 .- searched) .*(1 .- outside) .+
#          exp.(scaling[3] *v4) .*(1 .- tran).* searched
#         )

#     denom_sum =  @expression(model, sum( denom, dims = 1)[1,:,:])
#     denom_sum  = Array{NonlinearExpr,2}(undef,N_cons, D)
#     # for i = 1:N_cons
#     for d = 1:D
#         denom_sum[:, d] = @expression(model, (ones(nalt)' * denom[:,:, d])')
#     end
#     # end
    
#     #@NLexpression(model,denom_order[i = 1:N_obs, d = 1:D] , exp(scaling[1]* v1[i,d]) * searched[i,d] * (1 - outside[i,d])) #0 for outside option and searched = 0
    
#     # denom_search1 = 
#     #     #search until
#     #     @expression(model, exp.(scaling[2]*v2).*(1 .- outside) .* searched )
#     #     #not search from (search = 1 if outside = 1)
#     # denom_search2 = @expression(model, exp.(scaling[2]*v3).*(1 .- searched) )#0 for outside option
#     #@NLexpression(model,denom_ch[i = 1:N_cons, d = 1:D] , exp( scaling[3] *v4[i,d] ) )#(not anymore: if u_y == u_max, choice = 0.5 even with scaling = 0, So add 1e-5)
    
#     # Combine all inputs
#     prob = @expression(model, 1 ./ (1 .+ denom_sum))
    
#     L_i = @expression(model, prob * ones(D)./D .+ 1e-10)

#     JuMP.@NLobjective(model, Max, sum(log(L_i[i]) for i = 1:N_cons))

#     @time JuMP.optimize!(model)

    
#     return JuMP.value.(params),JuMP.value.(c),JuMP.value.(m),JuMP.objective_value(model)
# end
