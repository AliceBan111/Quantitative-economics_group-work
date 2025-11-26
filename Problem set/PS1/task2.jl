using LinearAlgebra


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

α = 0.1
println("β\t\tx1_exact\tx1_computed\tCond_Number\tRel_Residual")
println("-"^80)

for i in 0:12
    β = 10.0^i
    x_ex, x_comp, cn, rr = solve_and_analyze(α, β)
    @printf("%.0e\t\t%.6f\t%.6f\t\t%.2e\t%.2e\n", β, x_ex[1], x_comp[1], cn, rr)
end