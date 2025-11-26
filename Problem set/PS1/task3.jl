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
    return nlsolve((F,p)->rownanie!(F,p,x,d,w11,w12,w21,w22), [p0], method=:newton)
end

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

# Symulacja x w (0,1)
xs = collect(0.01:0.01:0.99)
sigma1 = 0.2
sigma2 = 5.0

function policz(x, d)
    res = rozwiazanie(x, d, w11, w12, w21, w22)
    p = res.zero[1]
    c11, c12, c21, c22 = popyty(p, x, d, w11, w12, w21, w22)
    return p, c11, c12, c21, c22
end

# Wyniki dla dwóch sigma
wyn1 = [policz(x, sigma1) for x in xs]
wyn2 = [policz(x, sigma2) for x in xs]

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
