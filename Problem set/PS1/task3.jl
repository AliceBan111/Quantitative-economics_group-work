using Plots, NLsolve

# -zasoby
w11 = 1.0; w12 = 1.0
w21 = 0.5; w22 = 1.5

# dochód agenta
zasob(w1, w2, p) = p*w1 + w2

# lewastrona
# x = alfa_1 (udział agenta 1 w CES)
# d = sigma (parametr CES)
function lewastrona(p, x, d, w11, w12, w21, w22)
    m1 = zasob(w11, w12, p)
    m2 = zasob(w21, w22, p)

    s1 = (x^d * p^(1-d)) / (x^d * p^(1-d) + (1-x)^d)
    s2 = ((1-x)^d * p^(1-d)) / ((1-x)^d * p^(1-d) + x^d)

    c11 = s1 * m1 / p
    c21 = s2 * m2 / p

    return c11 + c21
end

# Prawa strona
prawastrona(w11,w21) = w11 + w21

#równianie
function rownanie!(F, p, x, d, w11, w12, w21, w22)
    F[1] = lewastrona(p[1], x, d, w11, w12, w21, w22) - prawastrona(w11, w21)
end

#Rozwiązanie równania
function rozwiazanie(x, d, w11, w12, w21, w22; p0=1.0)
    return nlsolve((F,p)->rownanie!(F,p,x,d,w11,w12,w21,w22), [p0], method=:newton, ftol=1e-9, iterations=100) 
end
# Justification of stopping criterion:
# Reason: The total endowment of goods in this economy is small (Total Supply ~ 2.5 units).
# An error of 1e-9 is extremely small when compared to 2.5.
# Therefore, this precision is sufficient to consider the market cleared and the solution an equilibrium.

# Funkcje obliczające popyty
function popyty(p, x, d, w11, w12, w21, w22)
    m1 = zasob(w11, w12, p)
    m2 = zasob(w21, w22, p)

    s1 = (x^d * p^(1-d)) / (x^d * p^(1-d) + (1-x)^d)
    s2 = ((1-x)^d * p^(1-d)) / ((1-x)^d * p^(1-d) + x^d)

    c11 = s1*m1/p
    c12 = (1-s1)*m1
    c21 = s2*m2/p
    c22 = (1-s2)*m2

    return c11, c12, c21, c22
end

# Verification function to check if the solution is an equilibrium
function verify_equilibrium(p, x, d, w11, w12, w21, w22; tol=1e-9)
    # Computing demands for both agents
    c11, c12, c21, c22 = popyty(p, x, d, w11, w12, w21, w22)
    
    # Computing total supply
    total_supply_1 = w11 + w21
    total_supply_2 = w12 + w22
    
    # Computing total demand
    total_demand_1 = c11 + c21
    total_demand_2 = c12 + c22
    
    # Computing market clearing errors
    error_market_1 = abs(total_demand_1 - total_supply_1)
    error_market_2 = abs(total_demand_2 - total_supply_2)
    
    # Checking if markets clear within tolerance
    market_1_clears = error_market_1 < tol
    market_2_clears = error_market_2 < tol
    
    # Printing verification results
    println("Market Clearing Verification:")
    println("  Good 1: Demand = $(round(total_demand_1, digits=6)), Supply = $(round(total_supply_1, digits=6)), Error = $(round(error_market_1, digits=8))")
    println("  Good 2: Demand = $(round(total_demand_2, digits=6)), Supply = $(round(total_supply_2, digits=6)), Error = $(round(error_market_2, digits=8))")
    println("  Market 1 clears: $market_1_clears")
    println("  Market 2 clears: $market_2_clears")
    
    return market_1_clears && market_2_clears
end

# Finding x values where both agents consume equal amounts of both goods
function find_equal_consumption(wyn, xs, sigma_val; tol=1e-3)
   
    equal_x = []
    
    for (i, x) in enumerate(xs)
        c11 = wyn[i][2]  # agent 1, good 1
        c12 = wyn[i][3]  # agent 1, good 2
        c21 = wyn[i][4]  # agent 2, good 1
        c22 = wyn[i][5]  # agent 2, good 2
        
        # Checking if consumptions are equal for both goods
        good1_equal = abs(c11 - c21) < tol
        good2_equal = abs(c12 - c22) < tol
        
        if good1_equal && good2_equal
            push!(equal_x, x)
            println("  x = $(round(x, digits=4)): c1=($(round(c11, digits=4)), $(round(c12, digits=4))), c2=($(round(c21, digits=4)), $(round(c22, digits=4)))")
        end
    end
    
    if isempty(equal_x)
        println("  No values of x found where agents consume equal amounts of both goods")
    else
        println("  Total $(length(equal_x)) value(s) found")
    end
    
    return equal_x
end

# Symulacja x w (0,1)
xs = collect(0.01:0.01:0.99)
sigma1 = 0.2
sigma2 = 5.0

function policz(x, d)
    res = rozwiazanie(x, d, w11, w12, w21, w22)
    if !res.f_converged
        @warn "Solver did not converge for x=$x, σ=$d"
    end
    p = res.zero[1]
    c11, c12, c21, c22 = popyty(p, x, d, w11, w12, w21, w22)
    return p, c11, c12, c21, c22
end

# Wyniki dla dwóch sigma
wyn1 = [policz(x, sigma1) for x in xs]
wyn2 = [policz(x, sigma2) for x in xs]

# Verifying equilibrium for selected values of x

# Testing for σ = 0.2
println("Testing σ = 0.2:")
test_indices = [1, 25, 50, 75, length(xs)]  # Testing at x ≈ 0.01, 0.25, 0.50, 0.75, 0.99
for idx in test_indices
    x_test = xs[idx]
    p_test = wyn1[idx][1]
    println("\nFor x = $(round(x_test, digits=2)):")
    verify_equilibrium(p_test, x_test, sigma1, w11, w12, w21, w22)
end

# Testing for σ = 5.0
println("\n\nTesting σ = 5.0:")
for idx in test_indices
    x_test = xs[idx]
    p_test = wyn2[idx][1]
    println("\nFor x = $(round(x_test, digits=2)):")
    verify_equilibrium(p_test, x_test, sigma2, w11, w12, w21, w22)
end

# Finding equal consumption points
equal_x_sigma1 = find_equal_consumption(wyn1, xs, sigma1)
equal_x_sigma2 = find_equal_consumption(wyn2, xs, sigma2)

# Summary
println("For σ = 0.2: $(length(equal_x_sigma1)) equal consumption point(s)")
println("For σ = 5.0: $(length(equal_x_sigma2)) equal consumption point(s)")

p1_1 = [w[1] for w in wyn1]
p1_2 = [w[1] for w in wyn2]

# Wykres ceny równowagi
plot(xs, p1_1, label="σ = 0.2", xlabel="x", ylabel="p*", title="Cena równowagi p(x)")
plot!(xs, p1_2, label="σ = 5.0")

# Konsumpcja dobra 1
c11_1 = [w[2] for w in wyn1]
c21_1 = [w[4] for w in wyn1]

c11_2 = [w[2] for w in wyn2]
c21_2 = [w[4] for w in wyn2]

plot(xs, c11_1, label="agent 1, σ=0.2", xlabel="x", ylabel="c1", title="Konsumpcja dobra 1")
plot!(xs, c21_1, label="agent 2, σ=0.2")
plot!(xs, c11_2, label="agent 1, σ=5.0")
plot!(xs, c21_2, label="agent 2, σ=5.0")


"""
Findings:
1. How does the elasticity of substitution affect equilibrium prices and allocations?

When elasticity of substitution is low (0.2), goods are poor substitutes. The equilibrium price is highly volatile to clear the market. Allocations are rigid.
When elasticity of substitution is high (5.0), goods are good substitutes. The equilibrium price is very stable. Allocations are highly sensitive to price signals.

2. When is the equilibrium price more sensitive to x?

(1) Elasticity of substitution is low.
(2) When x is near the boundaries.


"""
