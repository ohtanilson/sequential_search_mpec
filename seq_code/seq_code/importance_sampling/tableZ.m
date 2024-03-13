clear all; clc;
i=1;
m=[-3.55:0.001:4]';
for j=-3.55:0.001:4
    c(i)=(1-normcdf(j))*((normpdf(j)/(1-normcdf(j)))-j);
    i=i+1;
end
c=c';


table=[m c];
csvwrite('tableZ.csv',table);







