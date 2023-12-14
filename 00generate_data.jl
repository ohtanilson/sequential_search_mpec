using Optim

# ESTIMATION CODE - Crude frequency simulator - Weitzman, UrsuSeilerHonka 2022

# Options for estimation
#options = Optim.Options(g_tol=1e-6, f_tol=1e-6, iterations=6000000, show_trace=false)

# Setting parameters (obtained from file name)
#filename = string(Base.source_path())[end-22:end-2]  # Assuming the file name length is constant

# Number of epsilon+eta draws
#D = parse(Int, filename[17:19])

# Seed
#seed = parse(Int, filename[21:end])

# Simulation inputs
N_cons = 1000  # num of consumers
N_prod = 5     # num of products
param = [1, 0.7, 0.5, 0.3, -3]  # true parameter vector [4 brandFE, search cost constant (exp)]

# Simulate data
simWeitz(N_cons, N_prod, param, seed)

# Load simulated data
# data = load("genWeitzDataS$seed.mat")
# data = data["data"]

# Estimation
# Initial parameter vector
param0 = zeros(size(param))

# Define likelihood function
function liklWeitz_crude_1(param)
    # Call the liklWeitz_crude_1 function with appropriate arguments
    # (Implementation of this function is assumed to be available)
    # Replace the following line with the actual function call
    return liklWeitz_crude_1(param, data, D, seed)
end

# Perform estimation
result = optimize(liklWeitz_crude_1, param0, options)

# Extract results
be = Optim.minimizer(result)
val = Optim.minimum(result)
exitflag = Optim.converged(result)

# Compute standard errors
hessian_inv = inv(Optim.hessian(result))
se = sqrt.(diag(hessian_inv))

# Save results
AS = hcat(be, se, val, exitflag)
CSV.write("rezSimWeitz_crude_D$D"S$seed.csv", DataFrame(AS), writeheader=false)

# Display elapsed time
println("Elapsed time: ", time() - @elapsed begin
    # Simulation code
end, " seconds.")