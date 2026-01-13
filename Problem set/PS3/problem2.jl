using Printf

const T = 20          
const Smax = 10       
const λ = 0.05        
const δ = 0.25        
const ρ = 5.0         


const c0 = 0.5
const c1 = 0.5
const c2 = 0.1

const α = 5.0         
const ψ = 10.0        
const p = 100.0       
const κ = 0.5         
const θ = 2.0         
const ω = 2.5         


const UNBLOOMED = 0   
const BLOOMED = 1     
const DEAD = -1       


const WAIT = 0        
const APPLY = 1       




function reward(O::Int, t::Int)
    
    anxiety = -(c0 + c1*t + c2*t^2)  
    
    if O == UNBLOOMED
        return anxiety
    elseif O == BLOOMED
        return α  
    else  
        return anxiety - ψ  
    end
end




function terminal_reward(O::Int, price::Float64)
    
    if O == UNBLOOMED
        return -κ * price  
    elseif O == BLOOMED
        return θ * price   
    else  
        return -ω * price  
    end
end




function factorial_approx(n::Int)

    if n == 0
        return 1.0
    end

    if n <= 20
        result = 1.0
        for i in 1:n
            result *= i
        end
        return result
    end

    return sqrt(2π*n) * (n/ℯ)^n
end

function poisson_pmf(k::Int, λ::Float64)

    return exp(-λ) * λ^k / factorial_approx(k)
end

function poisson_probs(ρ::Float64, max_k::Int=20)


    probs = zeros(Float64, max_k + 1)
    for k in 0:max_k
        probs[k+1] = poisson_pmf(k, ρ)
    end

    probs = probs / sum(probs)
    return probs
end

function state_to_idx(O::Int)

    if O == DEAD
        return 1
    elseif O == UNBLOOMED
        return 2
    else  
        return 3
    end
end




function solve_bellman()

    

    V = zeros(Float64, T+1, 3, Smax+1)
    policy = zeros(Int, T+1, 3, Smax+1)
    

    max_stress_increase = 20
    π = poisson_probs(ρ, max_stress_increase)
    

    for S in 0:Smax
        V[T+1, state_to_idx(DEAD), S+1] = reward(DEAD, T) + terminal_reward(DEAD, p)
        V[T+1, state_to_idx(UNBLOOMED), S+1] = reward(UNBLOOMED, T) + terminal_reward(UNBLOOMED, p)
        V[T+1, state_to_idx(BLOOMED), S+1] = reward(BLOOMED, T) + terminal_reward(BLOOMED, p)
    end
    

    for t in (T-1):-1:1
        if t % 5 == 0
            println("  Processing day t=$t...")
        end
        

        for S in 0:Smax
            V[t+1, state_to_idx(DEAD), S+1] = reward(DEAD, t) + V[t+2, state_to_idx(DEAD), S+1]
            V[t+1, state_to_idx(BLOOMED), S+1] = reward(BLOOMED, t) + V[t+2, state_to_idx(BLOOMED), S+1]
        end
        

        for S in 0:Smax

            V_wait = reward(UNBLOOMED, t) + 
                     λ * V[t+2, state_to_idx(BLOOMED), S+1] + 
                     (1 - λ) * V[t+2, state_to_idx(UNBLOOMED), S+1]
            

            V_apply = reward(UNBLOOMED, t)
            

            for k in 0:max_stress_increase
                new_S = S + k
                
                if new_S > Smax

                    V_apply += π[k+1] * V[t+2, state_to_idx(DEAD), Smax+1]
                else

                    prob_bloom = λ + δ
                    V_apply += π[k+1] * (prob_bloom * V[t+2, state_to_idx(BLOOMED), new_S+1] + 
                                        (1 - prob_bloom) * V[t+2, state_to_idx(UNBLOOMED), new_S+1])
                end
            end
            

            if V_wait >= V_apply
                V[t+1, state_to_idx(UNBLOOMED), S+1] = V_wait
                policy[t+1, state_to_idx(UNBLOOMED), S+1] = WAIT
            else
                V[t+1, state_to_idx(UNBLOOMED), S+1] = V_apply
                policy[t+1, state_to_idx(UNBLOOMED), S+1] = APPLY
            end
        end
    end

    return V, policy
end




function simulate_path(V::Array{Float64,3}, policy::Array{Int,3})

    O = UNBLOOMED
    S = 0
    fertilizer_count = 0
    π = poisson_probs(ρ, 20)
    
    for t in 1:T
        if O == DEAD || O == BLOOMED
            continue  
        end
        
        
        action = policy[t+1, state_to_idx(O), S+1]
        
        if action == WAIT
        
            if rand() < λ
                O = BLOOMED
            end
        else  
            fertilizer_count += 1
            
            r = rand()
            cumsum_prob = 0.0
            k = 0
            for i in 1:length(π)
                cumsum_prob += π[i]
                if r <= cumsum_prob
                    k = i - 1
                    break
                end
            end
            
            new_S = S + k
            
            if new_S > Smax
                O = DEAD
                S = Smax
            else
                S = new_S
                if rand() < λ + δ
                    O = BLOOMED
                end
            end
        end
    end
    
    return O, fertilizer_count
end




function print_policy_table(policy::Array{Int,3}, times::Vector{Int})

    println("\n" * "="^60)
    println("Optimal Policy: (0=Esperar, 1=Fertilizar)")
    println("="^60)
    

    print("Estrés | ")
    for t in times
        print("t=$t  ")
    end
    println()
    println("-"^60)
    

    for S in 0:Smax
        @printf "%2d     | " S
        for t in times
            action = policy[t+1, state_to_idx(UNBLOOMED), S+1]
            symbol = action == WAIT ? " W " : " F "
            print(symbol * "  ")
        end
        println()
    end
    println("="^60)
    println("W = Wait | F = Fertilizer")
    println()
end




println("\n" * "="^60)
println("SOLVING THE PROBLEM OF BASIL'S ORCHID")
println("="^60)


V, policy = solve_bellman()




initial_action = policy[2, state_to_idx(UNBLOOMED), 1]
action_name = initial_action == WAIT ? "WAIT" : "APPLY FERTILIZER"
println("\n📋 Question 4(a):")
println("Optimal action at t=1, O=0, S=0: $action_name")




println("\n📋 Question 4(b):")
times_to_plot = [1, 5, 10, 15, 19]
print_policy_table(policy, times_to_plot)

println("Interpretation:")
println("As the deadline approaches (T increases), Basil becomes")
println("More willing to use fertiliser because time is running out.")




expected_utility = V[2, state_to_idx(UNBLOOMED), 1]
println("\n📋 Question 4(c):")
@printf "Total utility expected since (t=1, O=0, S=0): %.2f\n" expected_utility




println("\n📋 Questions 4(d) y 4(e):")
n_sims = 1000
println("Simulating $n_sims trayectories...")

outcomes = Dict(BLOOMED => 0, UNBLOOMED => 0, DEAD => 0)
fertilizer_applications = Int[]

for sim in 1:n_sims
    if sim % 200 == 0
        println(" Simulation $sim/$n_sims...")
    end
    final_O, fert_count = simulate_path(V, policy)
    outcomes[final_O] += 1
    push!(fertilizer_applications, fert_count)
end

println("\n" * "="^60)
println("Results of $n_sims Simulations:")
println("="^60)
@printf "✓ Blossomed correctly:  %5.1f%%  (%d simulaciones)\n" (outcomes[BLOOMED]/n_sims*100) outcomes[BLOOMED]
@printf "○ It never blossomed:         %5.1f%%  (%d simulaciones)\n" (outcomes[UNBLOOMED]/n_sims*100) outcomes[UNBLOOMED]
@printf "✗ Died:                  %5.1f%%  (%d simulaciones)\n" (outcomes[DEAD]/n_sims*100) outcomes[DEAD]

avg_fertilizer = sum(fertilizer_applications) / length(fertilizer_applications)
println("\n" * "-"^60)
@printf "Average fertilizer application: %.2f\n" avg_fertilizer
@printf "Compared to T=%d Total days: %.1f%% of the days\n" T (avg_fertilizer/T*100)
println("="^60)

println("\nEND")
