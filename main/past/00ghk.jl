function liklWeitz_ghk_1(param, data, D, seed)
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

        # # Generate random draws
        # Random.seed!(seed)
        # epsilonDraw = randn(N_obs, D)
        # etaDraw = randn(N_obs, D)

        # chosen consumer id and his likelihood
        consumerData[consumer_num + 1:consumer_num + uniCons, 1] .= consid2[1, :]
        consumerData[consumer_num + 1:consumer_num + uniCons, 2] .= liklWeitz_ghk_2(param, dat, D, nalt, seed)
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

function liklWeitz_ghk_2(param, dat, D, nalt, seed)

    #data features
    consumer=dat[:,1];
    N_obs=length(consumer);
    N_cons=length(unique(consumer));
    N_prod = nalt;
    
    #choices
    tran=dat[:,end];
    tran_ranking = sum(reshape(tran, N_prod, N_cons).* repeat(1:N_prod, 1, N_cons), dims=1)' .|> Int64
    searched=dat[:,end-1]
    
    searched_amt = sum(reshape(searched, N_prod, N_cons), dims=1)' .|> Int64
    last_searched = zeros(N_prod, N_cons);
    for i = 1:N_cons
        last_searched[searched_amt[i], i] = 1.0;
    end
    
    c=exp(param[end])*ones(N_obs);
    X=dat[:,4:3+size(param[1:end-1],1)];
    xb=X*param[1:end-1]
    
    ######Calculate m#########
    ###1. look-up table method
    #table=CSV.read("tableZ.csv");
    global table
    m=zeros(N_obs);
    for i=1:N_obs
        lookupvalue=abs.(table[:,2].-c[i]);
        if (table[1,2]>=c[i] && c[i]>=table[end,2])
            index_m=argmin(lookupvalue);
            m[i]=table[index_m,1];
        elseif table[1,2]<c[i]
            m[i]=-c[i];
        elseif c[i]<table[end,2]
            m[i]=4.001;
        end
    end
    
    # ###2. newton method
    # m=zeros(N_obs);
    # x0 = 0; # initial point
    # for i = 1:size(c, 1)
    #     m[i] = newtonZ(c[i], x0);
    # end
    
    # ###3. contraction mapping method
    # m=zeros(N_obs); # initial point
    # for i = 1:size(c, 1)
    #     m[i] = contractionZ(m[i], c[i]);
    # end
    
    Random.seed!(seed)
    etaDraw=randn(N_cons,D);
    xb_con_prod = reshape(xb, N_prod, N_cons);
    mc = reshape(m, N_prod, N_cons);

    L_ALL = zeros(N_cons, D);
    for con = 1:N_cons
        H = searched_amt[con];                              # The last searched item
        mu_i = zeros(N_prod, D);                            # Presearch shock vector
        eps_i = zeros(N_prod, D);                           # Postsearch shock vector
        S_bar = setdiff(1:N_prod, 1:H);                     # The set of unsearched products
        tr = tran_ranking[con];                             # Position of purchased product
        # Step 1: Sampling mu_i0   
        mu_i[1, :] = etaDraw[con, :];  # Sampling for the outside option   
        # Case 1: consumer searched products and only purchased the outside option
        if tr == 1 && H > 1
            # Step 2: Sampling mu_iH
            b_iH = mu_i[1, :] - xb_con_prod[H, con] - mc[H, con];
            rng(con*seed*1122);
            mu_i[H, :] = norminv.(rand(1, size(b_iH, 2)).*(1 .- cdf.(Normal(),b_iH)) .+ cdf.(Normal(),b_iH));
            # Step 3: Sampling mu_i(1:(H - 1))
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1, :] .- xb_con_prod[h, con];
                rng(con*seed*2233 + h);
                mu_i[h, :] = norminv.(rand(1, size(b_ih, 2)).*(1 .- cdf.(Normal(),b_ih)) .+ cdf.(Normal(),b_ih));
            end
            # Stopping rule
            l_stop = ones(1, D);
            for l = S_bar
                l_stop = l_stop.*cdf.(Normal(),mu_i[1, :] .- xb_con_prod[l, con] .- mc[l, con]);
            end
            # Continuation rule
            l_cont = 1 .- cdf.(Normal(),b_iH);
            # Selection rule
            l_selection = ones(1, D);
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1,
                    :] .- xb_con_prod[h, con];
                l_selection = l_selection.*(1 .- cdf.(Normal(),b_ih));
            end
            # Choice rule
            l_choice = ones(1, D);
            for h = 2:H
                l_choice = l_choice.*cdf.(Normal(),mu_i[1, :] .- mu_i[h, :] .- xb_con_prod[h, con]);
            end
        # Case 2 (Appendix A.1.1): consumer searched products and purchased the product searched last
        elseif tr == H && H > 1
            # Step 1-2: Sampling mu_il
            rng(seed*con);
            mu_i[S_bar, :] = randn(size(S_bar, 2), D);
            # Step 2: Sampling mu_iH
            b_iH = maximum.([mu_i[1, :] .- xb_con_prod[H, con] .- mc[H, con]; -xb_con_prod[H, con] .+ xb_con_prod[S_bar, con] .+ mu_i[S_bar, :]], dims=1);
            rng(seed*con*1122);
            mu_i[H, :] = norminv.(rand(1, size(b_iH, 2)).*(1 .- cdf.(Normal(),b_iH)) .+ cdf.(Normal(),b_iH));
            # Step 3: Sampling mu_i(1:(H - 1))
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1, :] .- xb_con_prod[h, con];
                rng(con*seed*2233 + h);
                mu_i[h, :] =  norminv.(rand(1, size(b_ih, 2)).*(1 .- cdf.(Normal(),b_ih)) .+ cdf.(Normal(),b_ih));
            end
            # Step 4: Sampling epsilon_i(1:H - 1)
            for h = 2:H - 1
                b_eps_ih = xb_con_prod[H, con] .+ mu_i[H, :] .+ mc[H, con] .- xb_con_prod[h, con] .- mu_i[h, :];
                rng(seed*con*3344 + h);
                eps_i[h, :] = norminv.(rand(1, size(b_eps_ih, 2)).*(cdf.(Normal(),b_eps_ih)));
            end
            # Stopping rule
            l_stop = 1 .- cdf.(Normal(),maximum([mu_i[1, :]; xb_con_prod[S_bar, con] .+ mu_i[S_bar, :] .+ mc[S_bar, con]; mu_i[2:H - 1, :] .+ xb_con_prod[2:H - 1, con] .+ eps_i[2:H - 1, :]], dims=1
                    ) .- xb_con_prod[H, con] .- mu_i[H, :]);
            # Continuation rule
            l_cont = 1 .- cdf.(Normal(),b_iH);
            # Selection rule
            l_selection = ones(1, D);
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1, :] .- xb_con_prod[h, con];
                l_selection = l_selection.*(1 .- cdf.(Normal(),b_ih));
            end
            # Choice rule
            l_choice = ones(1, D);
            for h = 2:H - 1
                l_choice = l_choice.*cdf.(Normal(),xb_con_prod[H, con] .+ mu_i[H, :] .+ mc[H, con] .- xb_con_prod[h, con] .- mu_i[h, :]);
            end
        # Case 3 (Appendix A.1.2): consumer searched products and purchased a product neither not searched last nor outside option
        elseif (tr > 1) && (tr < H)
            
            # Step 1-2: Sampling mu_il
            rng(seed*con);
            mu_i[S_bar, :] = randn(size(S_bar, 2), D);
            # Step 2: Sampling mu_iH
            b_iH = maximum.([mu_i[1, :] .- xb_con_prod[H, con] .- mc[H, con]; -xb_con_prod[H, con] .+ xb_con_prod[S_bar, con] .+ mu_i[S_bar, :]], dims=1);
            rng(seed*con*1122);
            mu_i[H, :] = norminv.(rand(1, size(b_iH, 2)).*(1 .- cdf.(Normal(),b_iH)) .+ cdf.(Normal(),b_iH));
            # Step 3: Sampling mu_i(1:(H - 1))
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1, :] .- xb_con_prod[h, con];
                rng(con*seed*2233 + h);
                mu_i[h, :] = norminv.(rand(1, size(b_ih, 2)).*(1 .- cdf.(Normal(),b_ih)) .+ cdf.(Normal(),b_ih));
            end
            # Step 4: Sampling epsilon for the purchased option
            b_eps_upper = xb_con_prod[H, con] .+ mu_i[H, :] .+ mc[H, con] .- xb_con_prod[tr, con] .- mu_i[tr, :];
            b_eps_lower = maximum.([xb_con_prod[S_bar, con] .+ mu_i[S_bar, :] .+ mc[S_bar, con]; mu_i[1, :]], dims=1) .- xb_con_prod[tr, con] .- mu_i[tr, :];
            rng(seed*con*3344);
            eps_i[tr, :] = norminv.(rand(1, size(b_eps_upper, 2)).*(cdf.(Normal(),b_eps_upper) .- cdf.(Normal(),b_eps_lower)) .+ cdf.(Normal(),b_eps_lower));
            # Stopping rule
            l_stop = cdf.(Normal(),b_eps_upper) .- cdf.(Normal(),b_eps_lower);
            # Continuation rule
            l_cont = 1 .- cdf.(Normal(),b_iH);
            # Selection rule
            l_selection = ones(1, D);
            for h = H - 1:-1:2
                b_ih = xb_con_prod[h + 1, con] .+ mu_i[h + 1, :] .- xb_con_prod[h, con];
                l_selection = l_selection.*(1 .- cdf.(Normal(),b_ih));
            end
            # Choice rule
            l_choice = ones(1, D);
            for h in setdiff(2:H, tr)
                l_choice = l_choice.*cdf.(Normal(),xb_con_prod[tr, con] .+ mu_i[tr, :] .+ eps_i[tr, :] .- xb_con_prod[h, con] .- mu_i[h, :]);
            end
        # Case 4: consumer only searched outside option and purchased the outside option
        elseif tr == 1 && H == 1
            # Stopping rule
            l_stop = ones(1, D);
            for l = S_bar
                l_stop = l_stop.*cdf.(Normal(),mu_i[1, :] .- xb_con_prod[l, con] .- mc[l, con]);
            end
            # Continuation rule
            l_cont = ones(1, D);
            # Selection rule
            l_selection = ones(1, D);
            # Choice rule
            l_choice = ones(1, D);
        end
        L_ALL[con, :] = l_stop.*l_cont.*l_selection.*l_choice;
    end
    llk = mean(L_ALL, dims=2);
    return llk
end

# Test: evaluate time
@elapsed begin
    for i = 1:5
        liklWeitz_ghk_1(param, data, D, seed)
    end     
end


