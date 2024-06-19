consumer = data[:, 1]
N_cons = length(Set(consumer))

#N_prod = data[:, end - 2]
N_prod = data[:, end - 2]
Js = unique(N_prod)
Num_J = length(Js)

i = 1

nalt = Int.(Js[i])
dat = data[N_prod .== nalt, :]
N_obs = size(dat, 1)
#uniCons = Int.(N_obs/nalt)
#consid2 = reshape(dat[:, 1], nalt, uniCons)

# Generate random draws
Random.seed!(seed)
epsilonDraw = randn(N_obs, D)
etaDraw = randn(N_obs, D)
# Data features
# consumer = dat[:, 1]
# N_obs = length(consumer)
# N_cons = length(Set(consumer))

# Choices
# tran = dat[:, end]
# searched = dat[:, end - 1]
# has_searched = dat[:, end - 3]
# last = dat[:, end - 4]

# Parameters
#outside = dat[:, 3]
#c = exp(param[end]) * ones(N_obs)
#X = dat[:, 4:7]
#xb = X * param[1:end-1]
#eut = (xb .+ etaDraw) .* (1 .- outside)
#ut = eut .+ epsilonDraw