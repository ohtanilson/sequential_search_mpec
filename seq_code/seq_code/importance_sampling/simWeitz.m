%SIMULATION CODE  - Weitzman, UrsuSeilerHonka 2022

function simulation = simWeitz(N_cons, N_prod, param, param_std, seed)

%set seed for replication
rng('default'); rng(seed);

%numb of observations
N_obs=N_cons*N_prod;
consumer=repelem(1:N_cons,N_prod); consumer=consumer'; 
N_prod_cons=repelem(N_prod,N_cons)';

%product id
prod=repmat([1:N_prod]',N_cons,1); 

%outside option (represents option of not buying; is always searched) and brandFE
outside = (prod==1);
brand1=(prod==2);
brand2=(prod==3);
brand3=(prod==4);
brand4=(prod==5);
 
%prod char: in this case only brand intercepts
X=[brand1 brand2 brand3 brand4];

[~,index_first,~]=unique(consumer,'stable');%first entry for each consumer e.g.[1, 6, 11]
index_last=cumsum(N_prod_cons);%last entry for each consumer e.g.[5, 10, 15]

%parameters (random coefficients)
c=exp(param(end) + param_std(end).*randn(N_obs,1));%search cost
xb=sum(X.*(param(1:end-1) + param_std(1:end-1).*randn(N_obs,1)),2);%utility from observables
%draws affecting utility
epsilon=randn(N_obs,1);
eta=randn(N_obs,1);
%expected utility and utility
eut=(xb+eta).*(1-outside);
ut=eut+epsilon;

%%%%%%FORM Z's%%%%%%%%%
%%%1. look-up table method
table=importdata('tableZ.csv');
m=zeros(N_obs,1);
for i=1:N_obs
    lookupvalue=abs(table(:,2)-c(i));
    if (table(1,2)>=c(i)&& c(i)>=table(end,2))
        [~,index_m]=min(lookupvalue);
        m(i)=table(index_m,1);
    elseif table(1,2)<c(i)
        m(i)=-c(i);
    elseif c(i)<table(end,2)
        m(i)=4.001;
    end
end
z=m+eut;


% %%%2. newton method
% m=zeros(N_obs,1);
% x0 = 0; % initial point
% for i = 1:size(c, 1)
%     m(i) = newtonZ(c(i), x0);
% end
% z= eut + m;

% %%%3. contraction mapping method
% m=zeros(N_obs,1); % initial point
% for i = 1:size(c, 1)
%     m(i) = contractionZ(m(i), c(i));
% end
% z = eut + m;

%plug in large value for outside option as it is always "searched" first 
z=100000*outside+z.*(1-outside);

%order data by z
da=[consumer prod outside X eut ut z];
whatz=size(da,2);
whatu=whatz-1;
whateu=whatu-1;

for i=1:N_cons
    [values(index_first(i):index_last(i)), order(index_first(i):index_last(i))]=sort(da(index_first(i):index_last(i),whatz),'descend');
end

long_first = repelem(index_first',N_prod_cons); long_first=long_first(:);
order2=order'+long_first-1;

data=da(order2,:);

%search decision: 1. outside option always searched
searched=outside;

%search decision: 2. search if z greater than all ut searched so far (because z's are ordered);
for i=1:N_cons
    %for every product, except outside option
    for j=index_first(i)+1:index_last(i)
        %max ut so far 
        relevant_ut_sofar=data(index_first(i):j-1,whatu).*searched(index_first(i):j-1,1);
        relevant_ut_sofar=relevant_ut_sofar(relevant_ut_sofar~=0);
        max_ut_sofar=max(relevant_ut_sofar);

        %search if z>ut_sofar 
        if (data(j,whatz)>max_ut_sofar)
            searched(j)=1;
        end
    end
end

%transaction: among those searched, pick max ut
tran=zeros(N_obs,1);
searched_ut=data(:,whatu).*searched;
for i=1:N_cons
    A=searched_ut(index_first(i):index_last(i));
    A(A == 0)=-100000;
    [~,indexch]=max(A);
    tran(index_first(i)+indexch-1)=1;
end

%export data
length=repelem(N_prod,N_obs,1);%number of products per consumer
searched_mat=reshape(searched,N_prod,N_cons);
has_searched=searched_mat(2,:);%did consumer search at least once
has_searched=repmat(has_searched,N_prod,1);
has_searched=has_searched(:);
last=[zeros(1,N_prod-1) 1]';
last=repmat(last,N_cons,1);

output=[data(:,1:whateu-1) last has_searched length searched tran];
save(sprintf('genWeitzDataS%d.mat',seed),'output');

end


