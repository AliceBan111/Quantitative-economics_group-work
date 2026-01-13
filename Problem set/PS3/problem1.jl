using LinearAlgebra, Plots, QuantEcon

P_Z = [0.6 0.3 0.1;
       0.2 0.6 0.2;
       0.1 0.3 0.6]
X_states = 0:5
Z_indices = 1:3

get_idx(x, z_idx) = x * 3 + z_idx

Q = zeros(18, 18)

for x in X_states
    for z_idx in Z_indices
        current_idx = get_idx(x, z_idx)
        if z_idx == 1     
            next_x = 0
        elseif z_idx == 2
            next_x = x
        elseif z_idx == 3
            next_x = (x == 5) ? 3 : min(x + 1, 5)
        end
        for next_z_idx in Z_indices
            next_idx = get_idx(next_x, next_z_idx)
            Q[current_idx, next_idx] += P_Z[z_idx, next_z_idx]
        end
    end
end

mc = MarkovChain(Q)
ψ_star = stationary_distributions(mc)[1]

ψ_X = [sum(ψ_star[get_idx(x, 1):get_idx(x, 3)]) for x in X_states]

ψ_Z = [sum(ψ_star[z_idx:3:18]) for z_idx in Z_indices]

mean_X = sum(X_states .* ψ_X)

cond_means = zeros(3)
for z_idx in Z_indices
    prob_z = ψ_Z[z_idx]
    joint_probs = [ψ_star[get_idx(x, z_idx)] for x in X_states]
    cond_means[z_idx] = sum(X_states .* joint_probs) / prob_z
end

p1 = bar(X_states, ψ_X, title="Marginal Distribution of X", label="", xlabel="X")
p2 = bar(["z1", "z2", "z3"], ψ_Z, title="Marginal Distribution of Z", label="", xlabel="Z")
p3 = bar(["z1", "z2", "z3"], cond_means, title="Conditional Mean E[X | Z]", label="", xlabel="Z")

plot(p1, p2, p3, layout=(3,1), size=(600, 800))
