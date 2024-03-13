

This folder contains estimation code for the Weitzman model using 4 different methods
Code for UrsuSeilerHonka, 2022

-each folder contains a readme file with additional details
-code written in Matlab R2018b
-methods may use different optimization packages in Matlab given specific numerical issues


* kernel-smoothed frequency simulator
    -uses fminunc and computes Hessian for standard errors
    -data simulated with homogenous parameters


* crude frequency simulator
    -uses fminunc and computes Hessian for standard errors
    -data simulated with homogenous parameters


* importance sampling
    -uses fmincon; will need to use a different method to compute standard errors (e.g. bootstrapping)
    -data simulated with random coefficients parameters


* GHK
    -uses fminsearchcon (to avoid the numerical issues of the max operator in GHK; fminsearchcon is a modified version of fminsearch); will need to use a different method to compute standard errors (e.g. bootstrapping)
    -data simulated with homogenous parameters

