cd(@__DIR__)
using CSV, DataFrames, Statistics, LinearAlgebra, IterativeSolvers, Formatting, Plots

# 1
asset_data = CSV.read("asset_returns.csv", DataFrame)
returns_matrix = Matrix(asset_data)

# 2
μ_matrix = mean(returns_matrix, dims=1)

demeaned_returns_matrix = returns_matrix .- μ_matrix

m = size(returns_matrix, 1)

Σ = (demeaned_returns_matrix' * demeaned_returns_matrix) / (m-1)

# 3
n = length(μ_matrix)

A = zeros(n+2, n+2)

A[1:n, 1:n] = Σ

A[1:n, n+1] = μ_matrix
A[1:n, n+2] = ones(n)

A[n+1, 1:n] = μ_matrix'
A[n+2, 1:n] = ones(n)'

size(A) == (n+2, n+2)

A

κ = cond(A)
println("The condition number of A matrix: ", κ) # As the condition number of A matrix is approximately 21626, which indicates that the matrix is moderately ill-conditioned.

# 4 & 5
# (a) Blackslash operator
μ̄ = 0.10

b = zeros(n+2)
b[n+1] = μ̄ 
b[n+2] = 1
b

x_sol = A\b

elapsed_time = @elapsed x_sol = A \ b

residual_norm = norm(A * x_sol - b, 2)
b_norm = norm(b, 2)
relative_residual = residual_norm / b_norm

w = x_sol[1:n]           
λ1 = x_sol[n+1]      
λ2 = x_sol[n+2] 

weight_sum = sum(w)                       
expected_return = dot(w, μ_matrix)       
portfolio_variance = w' * Σ * w

# (b2) Jacobi
A_new = A' * A

tolerance = 1e-12
has_zeros = any(abs.(diag(A_new)) .< tolerance) # No zero on the diagonal
κ_new = cond(A_new)
all(sum(abs.(A_new),dims=2) .<= 2abs.(diag(A_new))) # As this matrix A_new is not diagonally dominant, so we don't implement the jacobi or  Gauss-Seidel method to solve this system.

# (c) Conjugate Gradient method
is_symmetric = issymmetric(A_new)
is_positive_definite = isposdef(A_new) # This matrix A_new is symmetric and positively definite.

b_new = A' * b

x_cg, ch_cg = cg(A_new, b_new, 
                 reltol=1e-12, 
                 maxiter=10_000,
                 log=true)

cg_iterations = ch_cg.iters
cg_time = @elapsed cg(A_new, b_new, reltol=1e-12, maxiter=10_000, log=true)
residual_norm_cg = norm(A * x_cg - b, 2)
relative_residual_cg = residual_norm_cg / norm(b, 2)

w_cg = x_cg[1:n]
λ1_cg = x_cg[n+1]
λ2_cg = x_cg[n+2]

weight_sum_cg = sum(w_cg)
expected_return_cg = dot(w_cg, μ_matrix)
portfolio_variance_cg = w_cg' * Σ * w_cg

# (d) GMRES
x_gmres, ch_gmres = gmres(A, b, 
                         reltol=1e-12,
                         abstol=1e-12,
                         restart=30,           
                         maxiter=1000,        
                         log=true,             
                         verbose=false)

gmres_total_iters = ch_gmres.iters             
gmres_residual_norms = ch_gmres.data[:resnorm] 

residual_norm_gmres = norm(A * x_gmres - b, 2)
relative_residual_gmres = residual_norm_gmres / norm(b, 2)

gmres_time = @elapsed gmres(A, b, 
                           reltol=1e-12, 
                           abstol=1e-12,
                           restart=30, 
                           maxiter=1000, 
                           log=true)

w_gmres = x_gmres[1:n]
λ1_gmres = x_gmres[n+1]
λ2_gmres = x_gmres[n+2]

weight_sum_gmres = sum(w_gmres)
expected_return_gmres = dot(w_gmres, μ_matrix)
portfolio_variance_gmres = w_gmres' * Σ * w_gmres

# (e) 
P_diag = zeros(n+2, n+2)

for i in 1:n
    P_diag[i,i] = Σ[i,i]    
end

P_diag[n+1, n+1] = 1.0
P_diag[n+2, n+2] = 1.0

A_p = P_diag^(-1) * A
b_p = P_diag^(-1) * b

x_gmres_p, ch_gmres_p = gmres(A_p, b_p, 
                         reltol=1e-12,
                         abstol=1e-12,
                         restart=30,           
                         maxiter=1000,        
                         log=true,             
                         verbose=false)

gmres_p_total_iters = ch_gmres_p.iters             
gmres_p_residual_norms = ch_gmres_p.data[:resnorm] 

residual_norm_gmres_p = norm(A_p * x_gmres_p - b_p, 2)
relative_residual_gmres_p = residual_norm_gmres_p / norm(b_p, 2)

gmres_time = @elapsed gmres(A_p, b_p, 
                           reltol=1e-12, 
                           abstol=1e-12,
                           restart=30, 
                           maxiter=1000, 
                           log=true)

w_gmres_p = x_gmres_p[1:n]
λ1_gmres_p = x_gmres_p[n+1]
λ2_gmres_p = x_gmres_p[n+2]

weight_sum_gmres_p = sum(w_gmres_p)
expected_return_gmres_p = dot(w_gmres_p, μ_matrix)
portfolio_variance_gmres_p = w_gmres_p' * Σ * w_gmres_p

# 6
methods = ["Backslash", "Conjugate Gradient", "GMRES", "Preconditioned GMRES"]
solutions = [x_sol, x_cg, x_gmres, x_gmres_p]
weights = [w, w_cg, w_gmres, w_gmres_p]
iterations = [1, cg_iterations, gmres_total_iters, gmres_p_total_iters]
times = [elapsed_time, cg_time, gmres_time, gmres_time]
residuals = [relative_residual, relative_residual_cg, relative_residual_gmres, relative_residual_gmres_p]

# Report results for each method
for i in 1:length(methods)
    println("\n$(methods[i]) Method:")
    println("Iterations: $(iterations[i])")
    println("Time: $(round(times[i], digits=6)) seconds")
    println("Relative residual: $(residuals[i])")
    
    w_current = weights[i]
    weight_sum = sum(w_current)
    expected_return = dot(w_current, μ_matrix)
    portfolio_variance = w_current' * Σ * w_current
    portfolio_std = sqrt(portfolio_variance)
    
    println("Weight sum: $(weight_sum)")
    println("Expected return: $(expected_return) (target: 0.10)")
    println("Portfolio variance: $(portfolio_variance)")
    println("Portfolio standard deviation: $(portfolio_std)")
    
    # Check constraints
    weight_ok = abs(weight_sum - 1.0) < 1e-8
    return_ok = abs(expected_return - 0.10) < 1e-8
    println("Constraints satisfied - Weight: $weight_ok, Return: $return_ok")
end

# Compare portfolio weights across methods
println("\nWEIGHT COMPARISON ACROSS METHODS")
println("Asset  |  Backslash  |      CG     |    GMRES    | Precond GMRES")
println("-------|-------------|-------------|-------------|---------------")
for i in 1:n
    println(format("{:6d} | {:11.6f} | {:11.6f} | {:11.6f} | {:11.6f}", 
                  i, w[i], w_cg[i], w_gmres[i], w_gmres_p[i]))
end

# Final verification
println("\nFINAL VERIFICATION:")
println("All methods should give approximately the same portfolio variance")
variances = [w'*Σ*w, w_cg'*Σ*w_cg, w_gmres'*Σ*w_gmres, w_gmres_p'*Σ*w_gmres_p]
println("Portfolio variances: ", variances)
println("Maximum variance difference: ", maximum(variances) - minimum(variances))

# 7
μ_range = range(0.01, 0.10, length=50)

all_weights = []
all_variances = []
all_std_devs = []

x_initial = zeros(n+2)
for μ_target in μ_range
    b = zeros(n+2)
    b[n+1] = μ_target
    b[n+2] = 1

    x_sol = A \ b 

    w = x_sol[1:n]
    push!(all_weights, w)
    variance = w' * Σ * w
    push!(all_variances, variance)
    push!(all_std_devs, sqrt(variance))

    x_initial = x_sol
end

plot(all_std_devs, μ_range, 
     xlabel="Portfolio Standard Deviation", 
     ylabel="Expected Return",
     title="Efficient Frontier",
     legend=false,
     marker=:circle)




