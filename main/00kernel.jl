function liklWeitz_kernel_1(param,data,D,scaling,seed)
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
        consumerData[consumer_num + 1:consumer_num + uniCons, 2] .= liklWeitz_kernel_2(param,dat,D,scaling,nalt,epsilonDraw,etaDraw)
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

function liklWeitz_kernel_2(param,dat,D,scaling,nalt,epsilonDraw,etaDraw)
    # Data features
    consumer = dat[:, 1]
    N_obs = length(consumer)
    N_cons = length(Set(consumer))

    scaling_order=scaling[1]
    scaling_search=scaling[2]
    scaling_choice=scaling[3]

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

        #selection rule: z>z_next
        denom_order=exp.(scaling_order*(z[:,d]-zmax)).*has_searched.*searched.*(1.-outside).*(1.-last)

        #stopping rule: z>u_so_far
        denom_search=exp.(scaling_search*(z[:,d]-ymax)).*has_searched.*searched.*(1-outside)
         + exp.(scaling_search*(ymax-z[:,d])).*has_searched.*(1-searched)
         + exp.(scaling_search*(u0_5-z[:,d])).*(1-has_searched).*(1-outside)

        #choice rule :
        # this is not what the paper says, but it should work
        # paper: v = u_trand - max_{i ∈ searched} u
        # here: v_i = u_trand - u_i (∀ i ∈ searched)
        u_ch2=ut[:,d].*tran
        u_ch3=reshape(u_ch2,nalt,N_cons)
        u_ch4=repeat(sum(u_ch3),nalt,1)
        u_ch5=reshape(u_ch4,N_obs,1)
        denom_ch=exp(scaling_choice*(u_ch5-ut[:,d])).*(1-tran).*searched

        #1. create denom with all inputs 
        denom=denom_order+denom_search+denom_ch

        #2. sum at the consumer level
        denfull2=accumarray(consumer,denom)
        denfull=denfull2[denfull2.!=0]

        #check for numerical errors
        denfull_t = denfull .> 0 .& denfull .< 2.2205e-16
        denfull[denfull_t] = 2.2205e-16
        denfull_t2 = denfull .>  2.2205e+16
        denfull[denfull_t2] = 2.2205e+16

        #3. compute prob=1/(1+denfull)
        prob[:,d]=1./(1+denfull)
    end

    llk=mean(prob,2)

    return llk
end


    