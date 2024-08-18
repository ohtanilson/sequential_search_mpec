using Distributions,Random
using CSV, DataFrames, DelimitedFiles, Statistics
#using Plots
using LinearAlgebra, Kronecker
using JuMP, Ipopt # for MPEC
using Base.Threads # for parallel computation
using Optim #for Nelder Mead estimation