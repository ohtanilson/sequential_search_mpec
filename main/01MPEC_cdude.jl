using LinearAlgebra
using Kronecker

function liklWeitz_crude_1(param, data, D, seed)
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

        # Generate random draws
        Random.seed!(seed)
        epsilonDraw = randn(N_obs, D)
        etaDraw = randn(N_obs, D)

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



#function Crude_MPEC(data, D, seed)

    model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time"=>60.0))
    #global param # initial value list
    @variable(model, param[i = 1:4]) #5
    @variable(model, L_i_[i = 1:N_cons])
    #@variable(model, logL)

    # Data features
    consumer = data[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    # Choices
    tran = data[:, end]
    # searched = data[:, end - 1]
    # has_searched = data[:, end - 3]
    # last = data[:, end - 4]
    
    # Parameters
    outside = data[:, 3]
    #c = exp(param[end]) * ones(N_obs)
    X = data[:, 4:7]
    ut = @expression(model, ( (X * param) .+ etaDraw) .* (1 .- outside) .+ epsilonDraw)

    # Construct likelihood for consumers with the same number of searches
    # for i = 1:Num_J
    #     nalt = Int.(Js[i])
    #     dat = data[N_prod .== nalt, :]
    #     N_obs = size(dat, 1)
    #     uniCons = Int.(N_obs/nalt)
    #     consid2 = reshape(dat[:, 1], nalt, uniCons)

    #     # Generate random draws
    #     Random.seed!(seed)
    #     epsilonDraw = randn(N_obs, D)
    #     etaDraw = randn(N_obs, D)

    #     # chosen consumer id and his likelihood
    #     consumerData[consumer_num + 1:consumer_num + uniCons, 1] .= consid2[1, :]
    #     consumerData[consumer_num + 1:consumer_num + uniCons, 2] .= liklWeitz_crude_2(param, dat, D, nalt, epsilonDraw, etaDraw)
    #     consumer_num += uniCons
    # end

    # Sum over consumers
    # To guarantee llk is not zero within log
    # llk = -sum(log.(1e-10 .+ consumerData[:, 2]))

    # # Check for errors or save output
    # if isnan(llk) || llk == Inf || llk == -Inf || !isreal(llk)
    #     loglik = 1e+300
    # else
    #     loglik = llk
    #     println(param)
    #     println(loglik)
    #     paramLL = [param; loglik]
    #     # Save preliminary output
    #     #CSV.write("betaWeitz_crude_D$D""S$seed.csv", DataFrame(paramLL), writeheader=false)
    # end

    # function indicator(v)
    #     return v >= 0 ? 1 : 0
    # end

    # register(model, :indicator, autodiff = true)

    #choice
    u_y = @expression(model,Diagonal(ones(1000)) ⊗ ones(1,5) * (ut.* tran))
    for i = 1:N_cons
        for d = 1:D
            u_max[i, d] = @expression(model, maximum(ut[(5*(i-1) + 1):5*i, d]))
        end
    end
    for i = 1:N_cons
        for d = 1:D
            choice[i, d] = @NLexpression(model, u_y[i,d] - u_max[i,d] >= 0)
        end
    end
     #[i = 1:N_cons,  d = 1:D]
    #@NLexpression(model,choice[i = 1:N_cons, d = 1:D], (v4[i,d] >= 0))

    # Combine all inputs
    #L_i_d: (N x D) matrix
    #@expression(model, L_i_d, order .* search_1 .* search_2 .* search_3 .* choice)
    #@expression(model, L_i_d[i = 1:N_cons, d = 1:D], order[i,d] .* search_1[i,d] .* search_2[i,d] .* search_3[i,d] .* choice[i,d])

    #@expression(model, L_i[i = 1:N_cons], sum(L_i_d[i,d] for d=1:D))
    for i = 1:N_cons
        #@NLconstraint(model, L_i_[i] == sum(choice[i,d] for d=1:D) + 1e-15)
        
    end
    @NLexpression(model,L_i_[i = 1:N_cons], sum(choice[i,d] for d=1:D) + 1e-15)

JuMP.@NLobjective(model, Max, sum(log(L_i_[i]) for i = 1:N_cons))

@time JuMP.optimize!(model)

JuMP.value.(param),JuMP.objective_value(model)
    #return 
#end




#example1:
model = Model(optimizer_with_attributes(Ipopt.Optimizer))
p0 = [10,6]
@variable(model,x[1:2])
#@NLparameter(model, p[i in 1:2] == p0[i])

p = [1,3,10]
function f_(x)
    return (x - p - 1)^2
end
f(3)
register(model,:f_, 1,f_; autodiff = true)
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

#function liklWeitz_crude_2(param::Vector{Float64}, dat::Matrix{Float64}, D::Int64, nalt::Int64, epsilonDraw::Matrix{Float64}, etaDraw::Matrix{Float64})
function liklWeitz_crude_2_(param1::Float64,param2::Float64,param3::Float64,param4::Float64,param5::Float64)

    param = [param1;param2;param3;param4;param5]

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
    return sum(c)
    X = dat[:, 4:7]
    xb = X * param[1:end-1]
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
        chain_mult = order .* search_1 .* search_2 .* search_3 .* choice;
    
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
    #return llk
    ll = sum(log.(1e-10 .+ llk), dims=1)
    ll = ll[1]
    return ll
end
param = [1.0, 0.7, 0.5, 0.3, -3.0]
liklWeitz_crude_2_(1.0, 0.7, 0.5, 0.3, -3.0)
liklWeitz_crude_2(param, dat, D, nalt, epsilonDraw, etaDraw)
sum(log.(1e-10 .+ liklWeitz_crude_2(1, 0.7, 0.5, 0.3, -3)))

model = Model(optimizer_with_attributes(Ipopt.Optimizer))
@variable(model,param[1:5])
register(model,:liklWeitz_crude_2_, 5,liklWeitz_crude_2_; autodiff = true)
@NLobjective(model, Max, sum(log.(1e-10 .+ liklWeitz_crude_2(param[1],param[2],param[3],param[4],param[5]))))
optimize!(model)
println(JuMP.value.(param))