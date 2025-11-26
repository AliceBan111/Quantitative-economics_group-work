using LinearAlgebra, Printf


function solve_exact(α, β)
    return [1.0, 1.0, 1.0, 1.0, 1.0]
end

function solve_and_analyze(α, β)
    A = [1.0  -1.0   0.0   α-β    β
         0.0   1.0  -1.0   0.0  0.0
         0.0   0.0   1.0  -1.0  0.0
         0.0   0.0   0.0   1.0 -1.0
         0.0   0.0   0.0   0.0  1.0]
    
   
    b = [α, 0.0, 0.0, 0.0, 1.0]
    
 
    x_exact = solve_exact(α, β)
    x_computed = A \ b
    
   
    cond_num = cond(A)
    residual = norm(A * x_computed - b) / norm(b)
    
    return x_exact, x_computed, cond_num, residual
end

solve_and_analyze(2.0, 3.0)
solve_and_analyze(1e6, 1e6) 
solve_and_analyze(1e-6, 1e-6) 


α = 0.1
println("β\t\tExact x1\tBackslash x1\tCondition Number\tRelative Residual")
for i in 0:12
    β = 10.0^i
    x_ex, x_comp, cn, rr = solve_and_analyze(α, β)
    @printf("%.0e\t\t%.6f\t%.6f\t\t%.2e\t%.2e\n", β, x_ex[1], x_comp[1], cn, rr)
end

"""
Finding:
(1) As β increases from 1 to 10¹², the condition number increases from 9.47 to 1.41 × 10²⁴ Moreover, as β increases one order of magnitude, the conditon number increases by about two orders of magnitude.
(2) As β increases, relative residuals also increase, which indicates he accumulation of numerical errors in the solution computed by the backslash operator.
(3) Despite the condition number reaching 1.41 × 10²⁴, the numerical solution is still 1.000000 due to limited output precision.

"""


