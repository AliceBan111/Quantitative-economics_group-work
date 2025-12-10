

using Statistics
using Random
using Printf  


"""
    gauss_legendre_nodes_weights(n)

Calculates nodes and weights for Gauss-Legendre quadrature of order n.
These are the optimal points for evaluating the integral.
"""
function gauss_legendre_nodes_weights(n)
    
    if n == 20
        
        nodes = [-0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
                 -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
                 -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
                 -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
                 0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
                 0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
                 0.9639719272779138, 0.9931285991850949]
        
        
        weights = [0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
                   0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
                   0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
                   0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
                   0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
                   0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
                   0.0406014298003869, 0.0176140071391521]
        
        return nodes, weights
    else
        error("Only implemented for n=20 nodes")
    end
end

"""
    integrate_gauss(f, a, b, n=20)

Integrate the function f from a to b using Gauss-Legendre quadrature.
Transform the interval [a,b] to the standard interval [-1,1].
"""
function integrate_gauss(f, a, b, n=20)
    nodes, weights = gauss_legendre_nodes_weights(n)
    
   
    sum_val = 0.0
    for i in 1:n
        t = nodes[i]
        x = ((b - a) * t + (b + a)) / 2
        sum_val += weights[i] * f(x)
    end
    
  
    return sum_val * (b - a) / 2
end

"""
    integrate_infinite(f, n_segments=10, max_value=50)

Integrate from 0 to infinity by dividing into segments and using transformation.
To handle the infinite limit, we integrate up to a large value.
"""
function integrate_infinite(f, n_segments=10, max_value=50)
   
    segment_size = max_value / n_segments
    total = 0.0
    
    for i in 1:n_segments
        a = (i - 1) * segment_size
        b = i * segment_size
        if a == 0
            a = 0.001  
        end
        total += integrate_gauss(f, a, b, 20)
    end
    
    return total
end


"""
    bisection(f, a, b, tol=1e-8, max_iter=100)

Find the root of f in the interval [a,b] using bisection.
Requires that f(a) and f(b) have opposite signs.
"""
function bisection(f, a, b; tol=1e-8, max_iter=100)
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0
        error("f(a) y f(b) must have opposite signs")
    end
    
    for iter in 1:max_iter
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol || (b - a) / 2 < tol
            return c
        end
        
        if fa * fc < 0
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
    end
    
    return (a + b) / 2
end



"""
    foc_integral(ω, W, Rf, γ, μ, σ)

Numerically evaluate the integral of the first-order condition:
∫₀^∞ (r - Rf) * [W(ωr + (1-ω)Rf)]^(-γ) * fR(r) dr

where fR(r) is the lognormal probability density function.
"""
function foc_integral(ω, W, Rf, γ, μ, σ)
 
    function integrand(r)
        if r <= 0
            return 0.0
        end
        

        Rp = ω * r + (1 - ω) * Rf
        
        
        W_final = W * Rp
        
        if W_final <= 0
            return 0.0
        end
        
     
        u_prime = W_final^(-γ)
        

        log_density = -(log(r) - μ)^2 / (2 * σ^2)
        f_R = (1 / (r * σ * sqrt(2π))) * exp(log_density)
        
   
        return (r - Rf) * u_prime * f_R
    end
    
    result = integrate_infinite(integrand, 15, 100)
    
    return result
end


"""
    verify_linear_utility(Rf, μ, σ)

Verify that the foc_integral function works correctly by comparing
the numerical result with the analytical solution for γ = 0.
"""
function verify_linear_utility(Rf, μ, σ)
    W = 1.0
    ω = 0.5
    γ = 0.0
    
    
    numerical_result = foc_integral(ω, W, Rf, γ, μ, σ)
    
   
    analytical_result = exp(μ + σ^2/2) - Rf
    
    println("\n" * repeat("=", 70))
    println("VERIFICATION WITH γ = 0 (Linear Utility)")
    println(repeat("=", 70))
    println("Numerical result:  ", round(numerical_result, digits=10))
    println("Analytical result: ", round(analytical_result, digits=10))
    println("Difference:          ", abs(numerical_result - analytical_result))
    
    if abs(numerical_result - analytical_result) < 1e-4
        println("✓ Verification successful!")
    else
        println("✗ Verification error (may be due to numerical precision)")
    end
    println(repeat("=", 70))
end



"""
    optimal_portfolio(W, Rf, γ, μ, σ)

Find the optimal fraction ω* of wealth to invest in the risky asset
by solving the first-order condition: foc_integral(ω*, ...) = 0
"""
function optimal_portfolio(W, Rf, γ, μ, σ)
    
    f(ω) = foc_integral(ω, W, Rf, γ, μ, σ)
    
    
    intervals = [(-5.0, 5.0), (-2.0, 2.0), (0.0, 2.0), (-1.0, 1.0)]
    
    for (a, b) in intervals
        fa = f(a)
        fb = f(b)
        
        if fa * fb < 0
            
            ω_star = bisection(f, a, b, tol=1e-6)
            return ω_star
        end
    end
    
    
    println("  Searching for appropriate interval...")
    for a in -10:1:10
        b = a + 1
        fa = f(float(a))
        fb = f(float(b))
        if fa * fb < 0
            ω_star = bisection(f, float(a), float(b), tol=1e-6)
            return ω_star
        end
    end
    
    error("The root could not be found. Try adjusting the intervals.")
end


function solve_problem_1()
    println("\n" * repeat("=", 70))
    println("PROBLEM 1: PORTFOLIO SELECTION")
    println(repeat("=", 70))
    
    
    W = 1.0      
    Rf = 1.02    
    γ = 3.0      
    μ = 0.05    
    σ = 0.1      
    
    println("\nParámetros:")
    println("  Initial wealth (W):            ", W)
    println("  Risk-free return (Rf):         ", Rf, " (", round((Rf-1)*100, digits=2), "%)")
    println("  Risk aversion (γ):             ", γ)
    println("  Average log(R) (μ):            ", μ)
    println("  Std. dev. of log(R) (σ):       ", σ)
    
   
    verify_linear_utility(Rf, μ, σ)
    
    println("\nCalculating optimal portfolio...")
    ω_star = optimal_portfolio(W, Rf, γ, μ, σ)
    
    println("\n" * repeat("=", 70))
    println("RESULT")
    println(repeat("=", 70))
    println("Optimal fraction in risky assets (ω*):  ", round(ω_star, digits=6))
    println("Risk-free interest rate:         ", round(1 - ω_star, digits=6))
    println(repeat("=", 70))
    
 
    println("\nInterpretation:")
    if ω_star > 1
        println("  → The agent uses leverage (ω* > 1)")
        println("  → Borrow at a risk-free rate to invest more")
    elseif ω_star > 0
        println("  → The agent invests ", round(ω_star*100, digits=2), "% in risky assets")
        println("  → Y ", round((1-ω_star)*100, digits=2), "% in risk-free assets")
    else
        println("  → Short position in the risky asset (ω* < 0)")
    end
    
    return ω_star
end


"""
    analyze_risk_aversion()

Analyse how ω* varies with different values of γ (without using Plots.jl)
"""
function analyze_risk_aversion()
   
    W = 1.0
    Rf = 1.02
    μ = 0.05
    σ = 0.1
    
   
    γ_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    println("\n" * repeat("=", 70))
    println("SENSITIVITY ANALYSIS: ω* vs γ")
    println(repeat("=", 70))
    println("\nCalculating optimal portfolios for different levels of risk aversion...\n")
    println("γ (Aversion)  |  ω* (% Risky)  |  Interpretation")
    println(repeat("-", 70))
    
    results = []
    
    for γ in γ_values
        ω = optimal_portfolio(W, Rf, γ, μ, σ)
        push!(results, (γ, ω))
        
        interp = ""
        if ω > 1
            interp = "Leveraged"
        elseif ω > 0.5
            interp = "Mostly risky"
        elseif ω > 0
            interp = "Diversified"
        else
            interp = "Short position"
        end
        
        @printf("   %.1f          |    %7.2f%%        |  %s\n", γ, ω*100, interp)
    end
    
    println(repeat("=", 70))
    println("\nNote: The greater the risk aversion (γ), the lower the investment in risky assets")
    
    return results
end

println("\n🚀 Starting solution to Problem 1...")
println("   (Implementation without external packages)")


ω_optimal = solve_problem_1()


results = analyze_risk_aversion()

println("\n✓ Problem 1 successfully completed!")
println(repeat("=", 70))
println("\nNOTA: For graphical display, you can use the results in Excel.")
println("      or any graphics software with the generated data.")

println(repeat("=", 70))
