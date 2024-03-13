function loglik=liklWeitz_kernel_1(param,data,D,scaling,seed)

consumer=data(:,1);
N_cons=length(unique(consumer));

N_prod=data(:,end-2);
Js = unique(N_prod);
Num_J = length(Js);
consumerData = zeros(N_cons,2);
consumer_num = 0;

%construct likelihood for consumers with the same number of searches
for i = 1:Num_J
    nalt = Js(i);
    dat = data(N_prod == nalt,:);
    N_obs=length(dat);
    uniCons = N_obs / nalt;
    consid2 = reshape(dat(:,1),nalt,uniCons);
    
    rng('default'); rng(seed);
    epsilonDraw=randn(N_obs,D);
    etaDraw=randn(N_obs,D);
 
    consumerData(consumer_num+1:consumer_num+uniCons,1) = consid2(1,:)';
    consumerData(consumer_num+1:consumer_num+uniCons,2) = liklWeitz_kernel_2(param,dat,D,scaling,nalt,epsilonDraw,etaDraw);
    consumer_num = consumer_num + uniCons;
end

%sum over consumers
llk=-sum(log(consumerData(:,2)));

%check for errors or save output
if isnan(llk) == 1 || llk == Inf || llk == -Inf ||isreal(llk)==0
    loglik=1e+300
else
    loglik=llk;
    disp(param);
    paramLL=[param loglik];
    %save preliminary output
    csvwrite(sprintf('betaWeitz_kernel_D%dW%dS%d.csv',D,-scaling(1),seed),paramLL);
end

end

