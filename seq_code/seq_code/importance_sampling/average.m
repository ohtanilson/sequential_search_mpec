%AVERAGE

clc
clear all
d = dir('rezSimWeitz_imp_D100M100S*.csv');
for i = 1:length(d);  
    filename = d(i).name;
    data(:,i)=load(strcat(filename));
end
for i=1:100
    M(i,1)=mean(data(i,:));
    M2(i,1)=std(data(i,:));
    csvwrite('result_imp_mean_Sum.csv',M);
    csvwrite('result_imp_std_Sum.csv',M2);
end

