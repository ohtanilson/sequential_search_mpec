clc; clear all;
for i=2:50
    filename = sprintf('estWeitz_imp_D100M100S1.m',i);
    filename2=sprintf('estWeitz_imp_D100M100S%d.m',i);
    copyfile(filename,filename2)
end