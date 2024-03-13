function loglik=liklWeitz_imp(param, W_PROP, L_ALL,WEIGHT, C, consumer, Js, D, M, seed)
    N_cons = size(W_PROP, 1);
    Num_J = length(Js);

    theta_mean0 = param(1, 1:4);
    theta_std0 = param(1, 6:9);
    c_mean0 = param(1, 5);
    c_std0 = param(1, 10);

    W_GUESS = zeros(N_cons, M, Num_J);
    for i = 1:Num_J
        N_prod = Js(i);
        theta_weight = WEIGHT{i};
        c_all = C{i};

        for pd = 1:M 
            theta_pd = reshape(theta_weight(:, pd), N_prod - 1, N_cons);
            for cons = 1:N_cons
                theta_n_pd = theta_pd(:, cons)';
                c_n_pd = c_all(consumer == cons, pd);
                w_theta = normpdf(theta_n_pd, theta_mean0, theta_std0);
                w_c = normpdf(log(c_n_pd), c_mean0, c_std0);
                W_GUESS(cons, pd) = prod(w_theta)*prod(w_c);
            end
        end
    end
    likl_prod = L_ALL.*W_GUESS./W_PROP;
    likl = mean(likl_prod, 2);
    %sum over consumers
    %to guarantee llk is not zero within log
    llk=-sum(log(10^-10+likl));

    %check for errors or save output
    if isnan(llk) == 1 || llk == Inf || llk == -Inf ||isreal(llk)==0
        loglik=1e+300
    else
        loglik=llk;
        disp(param);
        paramLL=[param loglik];
        %save preliminary output
        csvwrite(sprintf('betaWeitz_imp_D%dM%dS%d.csv',D,M,seed),paramLL);
    end
end