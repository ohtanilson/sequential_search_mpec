function loglik=liklWeitz_crude_1(param,data,D,seed)

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
    consumerData(consumer_num+1:consumer_num+uniCons,2) = liklWeitz_crude_2(param,dat,D,nalt,epsilonDraw,etaDraw);
    consumer_num = consumer_num + uniCons;
end

%sum over consumers
%to guarantee llk is not zero within log
llk=-sum(log(10^-10+consumerData(:,2)));


%check for errors or save output
if isnan(llk) == 1 || llk == Inf || llk == -Inf ||isreal(llk)==0
    loglik=1e+300
else
    loglik=llk;
    disp(param);
    paramLL=[param loglik];
    %save preliminary output
    csvwrite(sprintf('betaWeitz_crude_D%dS%d.csv',D,seed),paramLL);
end

end

