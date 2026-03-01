using LinearAlgebra, Statistics, Distributions, Optim, Printf, Random

# PART 1: PARAMETERS

struct Params
    # Production
    alpha::Float64    # capital share
    nu::Float64       # labor share
    delta::Float64    # depreciation
    r::Float64        # interest rate
    w::Float64        # wage
    beta::Float64     # discount factor = 1/(1+r)
    # Productivity AR(1)
    rho::Float64
    sigma::Float64
    z_tilde::Float64
    # Adjustment costs
    gamma::Float64    # convex cost coefficient
    F::Float64        # fixed cost coefficient
    ps::Float64       # resale price (irreversibility)
    # Subsidy
    tau::Float64      # investment subsidy rate
end

function make_params(; gamma=0.5, F=0.02, ps=0.9, tau=0.0)
    alpha = 0.30
    nu    = 0.60
    delta = 0.08
    r     = 0.04
    w     = 1.0
    beta  = 1.0 / (1.0 + r)
    rho   = 0.90
    sigma = 0.12
    z_tilde = exp(-sigma^2 / (2*(1 - rho^2)))
    return Params(alpha, nu, delta, r, w, beta, rho, sigma, z_tilde,
                  gamma, F, ps, tau)
end

# PART 2: ANALYTICAL SOLUTION FOR LABOR AND PROFIT

"""
FOC of z*k^alpha*h^nu - w*h
"""
function optimal_labor(z, k, p::Params)
    return (p.nu * z * k^p.alpha / p.w)^(1.0 / (1.0 - p.nu))
end

"""
Reduced-form profit (after substituting out optimal labor):
"""
function profit(z, k, p::Params)
    coeff = (1.0 - p.nu) * (p.nu / p.w)^(p.nu / (1.0 - p.nu))
    return coeff * z^(1.0 / (1.0 - p.nu)) * k^(p.alpha / (1.0 - p.nu))
end

# PART 3: GRIDS

"""
z_grid: Nz points, transition matrix Pi (Nz x Nz)
"""
function tauchen(rho, sigma, z_tilde, Nz; m=3)
    log_z_tilde = log(z_tilde)
    std_z = sigma / sqrt(1 - rho^2)
    log_z_max = log_z_tilde + m * std_z
    log_z_min = log_z_tilde - m * std_z
    log_z_grid = range(log_z_min, log_z_max, length=Nz)
    dz = (log_z_max - log_z_min) / (Nz - 1)

    Pi = zeros(Nz, Nz)
    d = Normal()
    for i in 1:Nz
        for j in 1:Nz
            if j == 1
                Pi[i,j] = cdf(d, (log_z_grid[j] - rho*log_z_grid[i] - (1-rho)*log(z_tilde) + dz/2) / sigma)
            elseif j == Nz
                Pi[i,j] = 1.0 - cdf(d, (log_z_grid[j] - rho*log_z_grid[i] - (1-rho)*log(z_tilde) - dz/2) / sigma)
            else
                Pi[i,j] = cdf(d, (log_z_grid[j] - rho*log_z_grid[i] - (1-rho)*log(z_tilde) + dz/2) / sigma) -
                           cdf(d, (log_z_grid[j] - rho*log_z_grid[i] - (1-rho)*log(z_tilde) - dz/2) / sigma)
            end
        end
        Pi[i,:] ./= sum(Pi[i,:])
    end
    z_grid = exp.(log_z_grid)
    return z_grid, Pi
end

"""
Capital grid: Nk points between k_min and k_max (log-spaced).
"""
function build_capital_grid(Nk; k_min=0.1, k_max=20.0)
    return exp.(range(log(k_min), log(k_max), length=Nk))
end

# PART 4: VALUE FUNCTION ITERATION

"""
Adjustment cost function: c(i, k) = (gamma/2)*(i/k)^2*k + F*k*I{i!=0}
"""
function adj_cost(i, k, p::Params)
    c = (p.gamma / 2.0) * (i / k)^2 * k
    if abs(i) > 1e-10
        c += p.F * k
    end
    return c
end

"""
Price of investment: p(i; tau) = (1-tau) if i>=0, ps if i<0
"""
function inv_price(i, p::Params)
    if i >= 0.0
        return 1.0 - p.tau
    else
        return p.ps
    end
end

"""
One-period payoff for firm with (k, z) choosing investment i.
"""
function payoff(z, k, i, p::Params)
    pi = profit(z, k, p)
    cost = inv_price(i, p) * i + adj_cost(i, k, p)
    return pi - cost
end

"""
Solve VFI. Returns value function V (Nk x Nz) and investment policy i_star (Nk x Nz).
"""
function solve_vfi(p::Params; Nk=300, Nz=50, tol=1e-6, max_iter=2000, verbose=true)
    k_grid = build_capital_grid(Nk)
    z_grid, Pi = tauchen(p.rho, p.sigma, p.z_tilde, Nz)

    V = zeros(Nk, Nz)
    V_new = similar(V)
    i_star = zeros(Nk, Nz)

    # For each (k, z), the next period capital k' = (1-delta)*k + i
    # So i = k' - (1-delta)*k

    for iter in 1:max_iter
        for iz in 1:Nz
            z = z_grid[iz]
            EV = Pi[iz, :] # row of transition matrix
            EV_kprime = V * EV  # Nk vector: EV_kprime[ik'] = sum_iz' Pi[iz,iz']*V[ik',iz']

            for ik in 1:Nk
                k = k_grid[ik]
                k_nodep = (1.0 - p.delta) * k  # k' if i=0

                best_val = -Inf
                best_i = 0.0

                for ik2 in 1:Nk
                    kp = k_grid[ik2]
                    i = kp - k_nodep
                    pay = payoff(z, k, i, p)
                    val = pay + p.beta * EV_kprime[ik2]
                    if val > best_val
                        best_val = val
                        best_i = i
                    end
                end

                # Also allow exact inaction i=0 (k' = k_nodep, interpolate)
                # Interpolate EV at k_nodep
                ev_inact = interp1d(k_grid, EV_kprime, k_nodep)
                val_inact = profit(z, k, p) + p.beta * ev_inact  # no adjustment cost when i=0

                if val_inact > best_val
                    best_val = val_inact
                    best_i = 0.0
                end

                V_new[ik, iz] = best_val
                i_star[ik, iz] = best_i
            end
        end

        diff = maximum(abs.(V_new - V))
        V .= V_new

        if verbose && iter % 100 == 0
            @printf("  VFI iter %4d, diff = %.2e\n", iter, diff)
        end
        if diff < tol
            verbose && @printf("  VFI converged at iter %d, diff = %.2e\n", iter, diff)
            break
        end
        if iter == max_iter
            verbose && @printf("  VFI reached max_iter, diff = %.2e\n", diff)
        end
    end

    return V, i_star, k_grid, z_grid, Pi
end

"""
Linear interpolation: evaluate f at x given grid xg and values fg.
"""
function interp1d(xg, fg, x)
    n = length(xg)
    if x <= xg[1]; return fg[1]; end
    if x >= xg[n]; return fg[n]; end
    j = searchsortedfirst(xg, x) - 1
    j = clamp(j, 1, n-1)
    t = (x - xg[j]) / (xg[j+1] - xg[j])
    return fg[j] * (1-t) + fg[j+1] * t
end

# PART 5: STATIONARY DISTRIBUTION

"""
Compute stationary distribution mu (Nk x Nz) by iteration.
i_star is the investment policy (Nk x Nz).
"""
function stationary_distribution(i_star, k_grid, z_grid, Pi, p::Params;
                                  tol=1e-8, max_iter=5000)
    Nk, Nz = size(i_star)
    mu = ones(Nk, Nz) / (Nk * Nz)
    mu_new = similar(mu)

    for iter in 1:max_iter
        mu_new .= 0.0
        for iz in 1:Nz
            for ik in 1:Nk
                k = k_grid[ik]
                inv = i_star[ik, iz]
                kp = (1 - p.delta) * k + inv
                # find ik' by interpolation (split mass between adjacent grid points)
                Nk2 = length(k_grid)
                if kp <= k_grid[1]
                    ik2_lo = 1; ik2_hi = 1; wlo = 1.0
                elseif kp >= k_grid[Nk2]
                    ik2_lo = Nk2; ik2_hi = Nk2; wlo = 1.0
                else
                    ik2_lo = searchsortedfirst(k_grid, kp) - 1
                    ik2_lo = clamp(ik2_lo, 1, Nk2-1)
                    ik2_hi = ik2_lo + 1
                    wlo = (k_grid[ik2_hi] - kp) / (k_grid[ik2_hi] - k_grid[ik2_lo])
                end
                for iz2 in 1:Nz
                    mu_new[ik2_lo, iz2] += wlo * Pi[iz, iz2] * mu[ik, iz]
                    if ik2_hi != ik2_lo
                        mu_new[ik2_hi, iz2] += (1-wlo) * Pi[iz, iz2] * mu[ik, iz]
                    end
                end
            end
        end
        diff = maximum(abs.(mu_new - mu))
        mu .= mu_new
        if diff < tol
            break
        end
    end

    mu ./= sum(mu)
    return mu
end

# PART 6: MOMENTS

"""
Compute model moments given policy and stationary distribution.
"""
function compute_moments(i_star, mu, k_grid, z_grid, p::Params)
    Nk, Nz = size(i_star)
    avg_inv_rate = 0.0
    inaction_rate = 0.0
    neg_inv_frac = 0.0
    pos_spike = 0.0
    neg_spike = 0.0

    for iz in 1:Nz
        for ik in 1:Nk
            k = k_grid[ik]
            inv = i_star[ik, iz]
            ir = inv / k
            w = mu[ik, iz]
            avg_inv_rate += ir * w
            inaction_rate += (abs(ir) < 0.01 ? 1.0 : 0.0) * w
            neg_inv_frac  += (inv < 0 ? 1.0 : 0.0) * w
            pos_spike     += (ir > 0.20 ? 1.0 : 0.0) * w
            neg_spike     += (ir < -0.20 ? 1.0 : 0.0) * w
        end
    end

    return [avg_inv_rate, inaction_rate, neg_inv_frac, pos_spike, neg_spike]
end

# Data moments from Cooper & Haltiwanger (2006)
const DATA_MOMENTS = [0.122, 0.081, 0.104, 0.18, 0.014]
const MOMENT_NAMES = ["Avg Inv Rate", "Inaction Rate", "Neg Inv Frac",
                      "Pos Spike Rate", "Neg Spike Rate"]

# PART 7: METHOD OF MOMENTS ESTIMATION
function grid_search_smm(W; Nk=60, Nz=12)
    gamma_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    F_vals     = [0.0, 0.001, 0.005, 0.01, 0.02]
    ps_vals    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    best_val = Inf
    best_theta = [0.5, 0.02, 0.9]  
    
    for gamma in gamma_vals
        for F in F_vals
            for ps in ps_vals
                val = smm_objective_safe([gamma, F, ps], W; Nk=Nk, Nz=Nz)
                println("gamma=$gamma, F=$F, ps=$ps => obj=$val")
                if val < best_val
                    best_val = val
                    best_theta = [gamma, F, ps]
                end
            end
        end
    end
    
    println("Grid search best: gamma=$(best_theta[1]), F=$(best_theta[2]), ps=$(best_theta[3]), obj=$best_val")
    return best_theta
end

"""
Objective function for SMM: weighted distance between model and data moments.
"""
function smm_objective_safe(theta_raw, W; Nk=60, Nz=12)
    gamma, F, ps = theta_raw
    gamma = clamp(gamma, 0.05, 9.0)
    F     = clamp(F, 0.0, 0.1)
    ps    = clamp(ps, 0.01, 0.98)
    
    p = make_params(gamma=gamma, F=F, ps=ps)
    try
        V, i_star, k_grid, z_grid, Pi = solve_vfi(p; Nk=Nk, Nz=Nz, verbose=false)
        mu = stationary_distribution(i_star, k_grid, z_grid, Pi, p)
        m_model = compute_moments(i_star, mu, k_grid, z_grid, p)
        diff = m_model - DATA_MOMENTS
        return dot(diff, W * diff)
    catch e
        println("VFI failed: $e")
        return 1e10
    end
end

"""
Estimate theta = (gamma, F, ps) via SMM.
"""
function estimate_params(; Nk=60, Nz=12)
    W = I

    println("Stage 1: Grid Search")
    theta0 = grid_search_smm(W; Nk=Nk, Nz=Nz)

    println("\nStage 2: NelderMead refinement")
    println("Starting from: gamma=$(theta0[1]), F=$(theta0[2]), ps=$(theta0[3])")
    
    result = optimize(
        th -> smm_objective_safe(th, W; Nk=Nk, Nz=Nz),
        theta0,
        NelderMead(),
        Optim.Options(
            iterations = 1000,
            show_trace = true,
            f_abstol   = 1e-5,
            x_abstol   = 1e-4
        )
    )
    
    gamma_hat, F_hat, ps_hat = Optim.minimizer(result)

    gamma_hat = clamp(gamma_hat, 0.05, 5.0)
    F_hat     = clamp(F_hat, 0.0, 0.1)
    ps_hat    = clamp(ps_hat, 0.1, 1.0)
    
    @printf("Estimated: gamma=%.4f, F=%.4f, ps=%.4f\n", gamma_hat, F_hat, ps_hat)
    return gamma_hat, F_hat, ps_hat
end

# PART 8: AGGREGATE STATISTICS

"""
Compute aggregate K, H, Y and other statistics.
"""
function compute_aggregates(i_star, mu, k_grid, z_grid, p::Params)
    Nk, Nz = size(i_star)
    K = 0.0; H = 0.0; Y = 0.0
    corr_k = Float64[]; corr_z = Float64[]; corr_w = Float64[]
    subsidy_cost = 0.0; total_output = 0.0

    for iz in 1:Nz
        z = z_grid[iz]
        for ik in 1:Nk
            k = k_grid[ik]
            inv = i_star[ik, iz]
            h = optimal_labor(z, k, p)
            y = z * k^p.alpha * h^p.nu
            w = mu[ik, iz]
            K += k * w
            H += h * w
            Y += y * w
            push!(corr_k, k); push!(corr_z, z); push!(corr_w, w)
            if inv > 0
                subsidy_cost += p.tau * inv * w
            end
            total_output += y * w
        end
    end

    # Weighted correlation between k and z
    mu_k = sum(corr_k .* corr_w)
    mu_z = sum(corr_z .* corr_w)
    var_k = sum((corr_k .- mu_k).^2 .* corr_w)
    var_z = sum((corr_z .- mu_z).^2 .* corr_w)
    cov_kz = sum((corr_k .- mu_k) .* (corr_z .- mu_z) .* corr_w)
    corr_kz = cov_kz / sqrt(var_k * var_z)

    return K, H, Y, corr_kz, subsidy_cost, total_output
end

# PART 8.5: DISTRIBUTION ANALYSIS

function analyze_distributions(i_star, mu, k_grid, z_grid, p::Params)
    Nk, Nz = size(i_star)
    
    # --- 1. Marginal distribution of k ---
    mu_k = sum(mu, dims=2)[:, 1]  # Nk vector, sum over z
    
    # Weighted mean of k
    mean_k = sum(k_grid .* mu_k)
    
    # Weighted median of k
    cumulative = cumsum(mu_k)
    median_idx = searchsortedfirst(cumulative, 0.5)
    median_k = k_grid[clamp(median_idx, 1, Nk)]
    
    # Fraction of firms in different k ranges
    frac_low  = sum(mu_k[k_grid .< 1.0])
    frac_mid  = sum(mu_k[(k_grid .>= 1.0) .& (k_grid .<= 6.0)])
    frac_high = sum(mu_k[k_grid .> 6.0])
    
    println("\n--- Marginal Distribution of Capital k ---")
    @printf("  Mean(k)   = %.4f\n", mean_k)
    @printf("  Median(k) = %.4f\n", median_k)
    @printf("  Mean > Median: %s (right-skewed if true)\n", mean_k > median_k ? "true" : "false")
    @printf("  Fraction k < 1.0  : %.4f\n", frac_low)
    @printf("  Fraction 1 ≤ k ≤ 6: %.4f\n", frac_mid)
    @printf("  Fraction k > 6.0  : %.4f\n", frac_high)
    
    # --- 2. Investment rate histogram ---
    # Collect weighted investment rates
    bins = [-Inf, -0.20, -0.01, 0.01, 0.10, 0.20, 0.40, Inf]
    bin_labels = ["<-20%", "-20% to -1%", "-1% to 1% (inaction)",
                  "1% to 10%", "10% to 20%", "20% to 40%", ">40%"]
    bin_mass = zeros(length(bins) - 1)
    
    for iz in 1:Nz
        for ik in 1:Nk
            k   = k_grid[ik]
            inv = i_star[ik, iz]
            ir  = inv / k
            w   = mu[ik, iz]
            for b in 1:length(bin_labels)
                if bins[b] < ir <= bins[b+1]
                    bin_mass[b] += w
                    break
                end
            end
        end
    end
    # Handle exact lower bound for first bin
    for iz in 1:Nz
        for ik in 1:Nk
            ir = i_star[ik, iz] / k_grid[ik]
            if ir == -Inf || ir < bins[1]
                bin_mass[1] += mu[ik, iz]
            end
        end
    end
    
    println("\n--- Investment Rate Histogram ---")
    @printf("  %-25s %10s\n", "Bin", "Mass")
    for (label, mass) in zip(bin_labels, bin_mass)
        bar = "█" ^ Int(round(mass * 100))
        @printf("  %-25s %8.4f  %s\n", label, mass, bar)
    end
    
    # Check for bimodality: mass near zero vs mass at large positive rates
    mass_near_zero    = bin_mass[3]           # -1% to 1%
    mass_large_pos    = bin_mass[6] + bin_mass[7]  # >20%
    mass_middle       = bin_mass[4] + bin_mass[5]  # 1% to 20%
    println()
    @printf("  Mass near zero (|i/k|<1%%):    %.4f\n", mass_near_zero)
    @printf("  Mass at large positive (>20%%): %.4f\n", mass_large_pos)
    @printf("  Mass in middle (1%%-20%%):       %.4f\n", mass_middle)
    is_bimodal = (mass_near_zero > mass_middle * 0.5) && (mass_large_pos > mass_middle * 0.5)
    println("  Bimodal pattern detected: ", is_bimodal ? "YES" : "NO")
    
    return mu_k, bin_mass, bin_labels
end

# PART 9: EXPLORATION (STEP 2)

function explore_model()
    println("\n" * "="^60)
    println("STEP 2: MODEL EXPLORATION")
    println("="^60)

    configs = [
        ("Convex costs only (F=0, ps=1)",    make_params(gamma=0.5, F=0.0, ps=1.0)),
        ("Adding fixed costs (F=0.02, ps=1)", make_params(gamma=0.5, F=0.02, ps=1.0)),
        ("Full model (F=0.02, ps=0.9)",       make_params(gamma=0.5, F=0.02, ps=0.9)),
    ]

    for (desc, p) in configs
        println("\n--- $desc ---")
        V, i_star, k_grid, z_grid, Pi = solve_vfi(p; Nk=50, Nz=10, verbose=true)
        mu = stationary_distribution(i_star, k_grid, z_grid, Pi, p)
        m = compute_moments(i_star, mu, k_grid, z_grid, p)
        println("  Moments:")
        for (name, val) in zip(MOMENT_NAMES, m)
            @printf("    %-20s: %.4f\n", name, val)
        end
    end
end

# PART 10: SENSITIVITY TO GRID SIZE

function grid_sensitivity(p::Params)
    println("\n" * "="^60)
    println("GRID SENSITIVITY ANALYSIS")
    println("="^60)
    configs = [(30, 8), (50, 12), (80, 15), (120, 20), (200, 30)]
    for (Nk, Nz) in configs
        V, i_star, k_grid, z_grid, Pi = solve_vfi(p; Nk=Nk, Nz=Nz, verbose=false)
        mu = stationary_distribution(i_star, k_grid, z_grid, Pi, p)
        m = compute_moments(i_star, mu, k_grid, z_grid, p)
        @printf("Nk=%3d, Nz=%2d | AvgInvRate=%.4f, InactionRate=%.4f\n",
                Nk, Nz, m[1], m[2])
    end
end

# PART 11: POLICY ANALYSIS (SUBSIDY)


function policy_analysis(gamma_hat, F_hat, ps_hat; Nk=300, Nz=50)
    println("\n" * "="^60)
    println("POLICY ANALYSIS: Investment Subsidy tau=0.10 vs tau=0")
    println("="^60)

    results = Dict()
    for tau in [0.0, 0.10]
        p = make_params(gamma=gamma_hat, F=F_hat, ps=ps_hat, tau=tau)
        println("\nSolving model for tau=$tau...")
        V, i_star, k_grid, z_grid, Pi = solve_vfi(p; Nk=Nk, Nz=Nz, verbose=true)
        mu = stationary_distribution(i_star, k_grid, z_grid, Pi, p)
        m = compute_moments(i_star, mu, k_grid, z_grid, p)
        K, H, Y, corr_kz, sub_cost, tot_out = compute_aggregates(i_star, mu, k_grid, z_grid, p)
        results[tau] = (V=V, i_star=i_star, k_grid=k_grid, z_grid=z_grid,
                        Pi=Pi, mu=mu, moments=m, K=K, H=H, Y=Y,
                        corr_kz=corr_kz, sub_cost=sub_cost, tot_out=tot_out)
    end

    r0 = results[0.0]
    r1 = results[0.10]

    println("\n--- Investment Moments Comparison ---")
    println(@sprintf("%-22s %10s %10s %10s", "Moment", "tau=0", "tau=0.10", "% Change"))
    for (i, name) in enumerate(MOMENT_NAMES)
        v0 = r0.moments[i]; v1 = r1.moments[i]
        pct = (v1 - v0) / abs(v0) * 100
        @printf("%-22s %10.4f %10.4f %10.2f%%\n", name, v0, v1, pct)
    end

    println("\n--- Aggregates ---")
    for (name, v0, v1) in [("K", r0.K, r1.K), ("H", r0.H, r1.H), ("Y", r0.Y, r1.Y)]
        pct = (v1 - v0) / abs(v0) * 100
        @printf("%-10s: baseline=%.4f, subsidy=%.4f, change=%+.2f%%\n", name, v0, v1, pct)
    end

    println("\n--- Cross-sectional Patterns ---")
    @printf("Corr(k,z) baseline: %.4f\n", r0.corr_kz)
    @printf("Corr(k,z) subsidy:  %.4f\n", r1.corr_kz)

    println("\n--- Subsidy Cost ---")
    @printf("Subsidy cost / Output: %.4f (%.2f%%)\n",
            r1.sub_cost / r1.tot_out, 100*r1.sub_cost/r1.tot_out)

    return results
end

# PART 12: PRINT CALIBRATION RESULTS

function print_calibration(gamma_hat, F_hat, ps_hat; Nk=300, Nz=50)
    println("\n" * "="^60)
    println("CALIBRATION RESULTS")
    println("="^60)
    @printf("gamma = %.4f\n", gamma_hat)
    @printf("F     = %.4f\n", F_hat)
    @printf("ps    = %.4f\n", ps_hat)

    p = make_params(gamma=gamma_hat, F=F_hat, ps=ps_hat)
    V, i_star, k_grid, z_grid, Pi = solve_vfi(p; Nk=Nk, Nz=Nz, verbose=false)
    mu = stationary_distribution(i_star, k_grid, z_grid, Pi, p)
    m_model = compute_moments(i_star, mu, k_grid, z_grid, p)

    println("\n--- Moment Fit ---")
    @printf("%-22s %10s %10s\n", "Moment", "Data", "Model")
    for (i, name) in enumerate(MOMENT_NAMES)
        @printf("%-22s %10.4f %10.4f\n", name, DATA_MOMENTS[i], m_model[i])
    end
    mu_k, bin_mass, bin_labels = analyze_distributions(i_star, mu, k_grid, z_grid, p)
    return V, i_star, k_grid, z_grid, Pi, mu
end

# PART 13: PRINT POLICY FUNCTIONS SUMMARY

function print_policy_summary(i_star, k_grid, z_grid)
    Nk, Nz = size(i_star)
    println("\n--- Policy Function i*(k, z) Summary ---")
    println("(Investment rates i/k for selected k and z levels)")
    println()
    # Select indices
    k_idx = [1, div(Nk,4), div(Nk,2), 3*div(Nk,4), Nk]
    z_idx = [1, div(Nz,4)+1, div(Nz,2)+1, 3*div(Nz,4)+1, Nz]
    # Header
    print(@sprintf("%-8s", "k\\z"))
    for iz in z_idx
        print(@sprintf("   z=%.2f", z_grid[iz]))
    end
    println()
    for ik in k_idx
        k = k_grid[ik]
        print(@sprintf("k=%-6.2f", k))
        for iz in z_idx
            ir = i_star[ik, iz] / k
            print(@sprintf("  %7.4f", ir))
        end
        println()
    end
end

# MAIN EXECUTION

function main()
    println("="^60)
    println("HETEROGENEOUS FIRM MODEL - INVESTMENT SUBSIDY ANALYSIS")
    println("="^60)
    println()

    # --- Step 1: Exploration ---
    explore_model()

    # --- Step 2: Grid sensitivity with default params ---
    p_default = make_params(gamma=0.3, F=0.01, ps=0.7)
    grid_sensitivity(p_default)

    # --- Step 3: Calibration via SMM ---
    gamma_hat, F_hat, ps_hat = estimate_params(Nk=60, Nz=12)

    # Pre-calibrated values (set after running estimation):
    # gamma_hat = 0.8
    # F_hat     = 0.00001
    # ps_hat    = 0.70

    # --- Step 4: Print calibration results ---
    V, i_star, k_grid, z_grid, Pi, mu = print_calibration(gamma_hat, F_hat, ps_hat;
                                                            Nk=300, Nz=50)

    # --- Step 5: Print policy function summary ---
    print_policy_summary(i_star, k_grid, z_grid)

    # --- Step 6: Policy analysis ---
    results = policy_analysis(gamma_hat, F_hat, ps_hat; Nk=300, Nz=50)

    println("\nDone.")
end

# Run
main()