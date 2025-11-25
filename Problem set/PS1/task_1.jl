
using Distributions, Random, Plots, StatsPlots


# Poiss Distributions
dist_poiss = Poisson(1)

# Probe size
n = [5, 25, 100, 1000]

# Matrix for the avrerage
mat = zeros(1000, 4)

# generating avg values with standarization
for k in 1:4
    c = n[k]
    for i in 1:1000
        mat[i,k] = (mean(rand(dist_poiss, c)) - 1) * sqrt(c)
    end
end

# generating a vector with four blank plots
plots = Vector{Plots.Plot}(undef, 4)

# X 
x = -4:0.01:4
normal_pdf = pdf.(Normal(0,1), x)

# Histogramy + gęstość N(0,1)
for j in 1:4
    p = histogram(
        mat[:,j],
        bins = 30,
        normalize = true,
        title = "n = $(n[j])",
        xlabel = "Wartości standaryzowane",
        ylabel = "Gęstość prawdopodobieństwa",
        legend = false,
        color = :dodgerblue,
        alpha = 0.6
    )

    # adding density line N(0,1)
    plot!(p, x, normal_pdf, color=:red, lw=2)

    plots[j] = p
end

# 2×2
plot(plots..., layout=(2, 2), size=(900, 700))
