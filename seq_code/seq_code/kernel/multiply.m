clc; clear all;
for i=2:50
    filename = sprintf('estWeitz_kernel_D100W10S1.m',i);
    filename2=sprintf('estWeitz_kernel_D100W10S%d.m',i);
    copyfile(filename,filename2)
end