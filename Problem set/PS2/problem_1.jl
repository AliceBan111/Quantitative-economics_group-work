# ============================================================================
# Problema 1: Elección de Portafolio - SIN PAQUETES EXTERNOS
# Cuantitative Economics - Problem Set 2
# ============================================================================

# Solo usamos paquetes base de Julia (vienen preinstalados)
using Statistics
using Random
using Printf  # Para formatear la salida con @printf

# ----------------------------------------------------------------------------
# IMPLEMENTACIÓN PROPIA: Integración Numérica (Cuadratura de Gauss-Legendre)
# ----------------------------------------------------------------------------

"""
    gauss_legendre_nodes_weights(n)

Calcula nodos y pesos para cuadratura de Gauss-Legendre de orden n.
Estos son los puntos óptimos para evaluar la integral.
"""
function gauss_legendre_nodes_weights(n)
    # Para simplificar, usamos valores precalculados para n=20
    # Estos nodos y pesos están en el intervalo [-1, 1]
    if n == 20
        # Nodos de Gauss-Legendre (20 puntos)
        nodes = [-0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
                 -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
                 -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
                 -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
                 0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
                 0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
                 0.9639719272779138, 0.9931285991850949]
        
        # Pesos correspondientes
        weights = [0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
                   0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
                   0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
                   0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
                   0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
                   0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
                   0.0406014298003869, 0.0176140071391521]
        
        return nodes, weights
    else
        error("Solo implementado para n=20 nodos")
    end
end

"""
    integrate_gauss(f, a, b, n=20)

Integra la función f de a a b usando cuadratura de Gauss-Legendre.
Transforma el intervalo [a,b] al intervalo [-1,1] estándar.
"""
function integrate_gauss(f, a, b, n=20)
    nodes, weights = gauss_legendre_nodes_weights(n)
    
    # Transformación del intervalo [-1,1] a [a,b]
    # x = ((b-a)*t + (b+a))/2, donde t ∈ [-1,1]
    sum_val = 0.0
    for i in 1:n
        t = nodes[i]
        x = ((b - a) * t + (b + a)) / 2
        sum_val += weights[i] * f(x)
    end
    
    # Factor de escala por el cambio de variable
    return sum_val * (b - a) / 2
end

"""
    integrate_infinite(f, n_segments=10, max_value=50)

Integra de 0 a infinito dividiendo en segmentos y usando transformación.
Para manejar el límite infinito, integramos hasta un valor grande.
"""
function integrate_infinite(f, n_segments=10, max_value=50)
    # Dividimos [0, max_value] en segmentos
    segment_size = max_value / n_segments
    total = 0.0
    
    for i in 1:n_segments
        a = (i - 1) * segment_size
        b = i * segment_size
        if a == 0
            a = 0.001  # Evitamos evaluar exactamente en 0
        end
        total += integrate_gauss(f, a, b, 20)
    end
    
    return total
end

# ----------------------------------------------------------------------------
# IMPLEMENTACIÓN PROPIA: Método de Bisección para encontrar raíces
# ----------------------------------------------------------------------------

"""
    bisection(f, a, b, tol=1e-8, max_iter=100)

Encuentra la raíz de f en el intervalo [a,b] usando bisección.
Requiere que f(a) y f(b) tengan signos opuestos.
"""
function bisection(f, a, b; tol=1e-8, max_iter=100)
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0
        error("f(a) y f(b) deben tener signos opuestos")
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

# ----------------------------------------------------------------------------
# Paso 1: Función para evaluar la integral de la condición de primer orden
# ----------------------------------------------------------------------------

"""
    foc_integral(ω, W, Rf, γ, μ, σ)

Evalúa numéricamente la integral de la condición de primer orden:
∫₀^∞ (r - Rf) * [W(ωr + (1-ω)Rf)]^(-γ) * fR(r) dr

donde fR(r) es la densidad de probabilidad lognormal.
"""
function foc_integral(ω, W, Rf, γ, μ, σ)
    # Definimos el integrando
    function integrand(r)
        if r <= 0
            return 0.0
        end
        
        # Retorno del portafolio: Rp = ωr + (1-ω)Rf
        Rp = ω * r + (1 - ω) * Rf
        
        # Riqueza final: W * Rp
        W_final = W * Rp
        
        if W_final <= 0
            return 0.0
        end
        
        # Utilidad marginal: u'(W) = W^(-γ)
        u_prime = W_final^(-γ)
        
        # Densidad lognormal: fR(r) = (1/(r*σ*√(2π))) * exp(-(log(r)-μ)²/(2σ²))
        log_density = -(log(r) - μ)^2 / (2 * σ^2)
        f_R = (1 / (r * σ * sqrt(2π))) * exp(log_density)
        
        # Retornamos: (r - Rf) * u'(W*Rp) * fR(r)
        return (r - Rf) * u_prime * f_R
    end
    
    # Integramos de 0 a infinito
    result = integrate_infinite(integrand, 15, 100)
    
    return result
end

# ----------------------------------------------------------------------------
# Paso 2: Verificación con γ = 0 (utilidad lineal)
# ----------------------------------------------------------------------------

"""
    verify_linear_utility(Rf, μ, σ)

Verifica que la función foc_integral funciona correctamente comparando
el resultado numérico con la solución analítica para γ = 0.
"""
function verify_linear_utility(Rf, μ, σ)
    W = 1.0
    ω = 0.5
    γ = 0.0
    
    # Cálculo numérico
    numerical_result = foc_integral(ω, W, Rf, γ, μ, σ)
    
    # Cálculo analítico: E[R] - Rf
    analytical_result = exp(μ + σ^2/2) - Rf
    
    println("\n" * repeat("=", 70))
    println("VERIFICACIÓN CON γ = 0 (Utilidad Lineal)")
    println(repeat("=", 70))
    println("Resultado numérico:  ", round(numerical_result, digits=10))
    println("Resultado analítico: ", round(analytical_result, digits=10))
    println("Diferencia:          ", abs(numerical_result - analytical_result))
    
    if abs(numerical_result - analytical_result) < 1e-4
        println("✓ Verificación exitosa!")
    else
        println("✗ Error en la verificación (puede ser por precisión numérica)")
    end
    println(repeat("=", 70))
end

# ----------------------------------------------------------------------------
# Paso 3: Función para encontrar el portafolio óptimo
# ----------------------------------------------------------------------------

"""
    optimal_portfolio(W, Rf, γ, μ, σ)

Encuentra la fracción óptima ω* de riqueza a invertir en el activo riesgoso
resolviendo la condición de primer orden: foc_integral(ω*, ...) = 0
"""
function optimal_portfolio(W, Rf, γ, μ, σ)
    # Definimos la función cuya raíz queremos encontrar
    f(ω) = foc_integral(ω, W, Rf, γ, μ, σ)
    
    # Buscamos primero el signo para definir intervalo
    # Probamos diferentes intervalos hasta encontrar un cambio de signo
    intervals = [(-5.0, 5.0), (-2.0, 2.0), (0.0, 2.0), (-1.0, 1.0)]
    
    for (a, b) in intervals
        fa = f(a)
        fb = f(b)
        
        if fa * fb < 0
            # Encontramos cambio de signo, usamos bisección
            ω_star = bisection(f, a, b, tol=1e-6)
            return ω_star
        end
    end
    
    # Si no encontramos, usamos búsqueda más amplia
    println("  Búsqueda de intervalo apropiado...")
    for a in -10:1:10
        b = a + 1
        fa = f(float(a))
        fb = f(float(b))
        if fa * fb < 0
            ω_star = bisection(f, float(a), float(b), tol=1e-6)
            return ω_star
        end
    end
    
    error("No se pudo encontrar la raíz. Intenta ajustar los intervalos.")
end

# ----------------------------------------------------------------------------
# Paso 4: Cálculo del portafolio óptimo con parámetros específicos
# ----------------------------------------------------------------------------

function solve_problem_1()
    println("\n" * repeat("=", 70))
    println("PROBLEMA 1: ELECCIÓN DE PORTAFOLIO")
    println(repeat("=", 70))
    
    # Parámetros del problema
    W = 1.0      # Riqueza inicial
    Rf = 1.02    # Retorno libre de riesgo (2%)
    γ = 3.0      # Aversión al riesgo
    μ = 0.05     # Media de log(R)
    σ = 0.1      # Desviación estándar de log(R)
    
    println("\nParámetros:")
    println("  Riqueza inicial (W):           ", W)
    println("  Retorno libre de riesgo (Rf):  ", Rf, " (", round((Rf-1)*100, digits=2), "%)")
    println("  Aversión al riesgo (γ):        ", γ)
    println("  Media de log(R) (μ):           ", μ)
    println("  Desv. std. de log(R) (σ):      ", σ)
    
    # Primero verificamos con γ = 0
    verify_linear_utility(Rf, μ, σ)
    
    # Calculamos el portafolio óptimo
    println("\nCalculando portafolio óptimo...")
    ω_star = optimal_portfolio(W, Rf, γ, μ, σ)
    
    println("\n" * repeat("=", 70))
    println("RESULTADO")
    println(repeat("=", 70))
    println("Fracción óptima en activo riesgoso (ω*): ", round(ω_star, digits=6))
    println("Fracción en activo libre de riesgo:      ", round(1 - ω_star, digits=6))
    println(repeat("=", 70))
    
    # Interpretación económica
    println("\nInterpretación:")
    if ω_star > 1
        println("  → El agente usa apalancamiento (ω* > 1)")
        println("  → Pide prestado a tasa libre de riesgo para invertir más")
    elseif ω_star > 0
        println("  → El agente invierte ", round(ω_star*100, digits=2), "% en el activo riesgoso")
        println("  → Y ", round((1-ω_star)*100, digits=2), "% en el activo libre de riesgo")
    else
        println("  → Posición corta en el activo riesgoso (ω* < 0)")
    end
    
    return ω_star
end

# ----------------------------------------------------------------------------
# Paso 5: Análisis de sensibilidad (en lugar de gráfico con Plots.jl)
# ----------------------------------------------------------------------------

"""
    analyze_risk_aversion()

Analiza cómo varía ω* con diferentes valores de γ (sin usar Plots.jl)
"""
function analyze_risk_aversion()
    # Parámetros fijos
    W = 1.0
    Rf = 1.02
    μ = 0.05
    σ = 0.1
    
    # Valores de γ a probar
    γ_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    println("\n" * repeat("=", 70))
    println("ANÁLISIS DE SENSIBILIDAD: ω* vs γ")
    println(repeat("=", 70))
    println("\nCalculando portafolios óptimos para diferentes niveles de aversión al riesgo...\n")
    println("γ (Aversión)  |  ω* (% Riesgoso)  |  Interpretación")
    println(repeat("-", 70))
    
    results = []
    
    for γ in γ_values
        ω = optimal_portfolio(W, Rf, γ, μ, σ)
        push!(results, (γ, ω))
        
        interp = ""
        if ω > 1
            interp = "Apalancado"
        elseif ω > 0.5
            interp = "Mayormente riesgoso"
        elseif ω > 0
            interp = "Diversificado"
        else
            interp = "Posición corta"
        end
        
        @printf("   %.1f          |    %7.2f%%        |  %s\n", γ, ω*100, interp)
    end
    
    println(repeat("=", 70))
    println("\nObservación: A mayor aversión al riesgo (γ), menor inversión en activo riesgoso")
    
    return results
end

# ----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------------------

println("\n🚀 Iniciando solución del Problema 1...")
println("   (Implementación sin paquetes externos)")

# Resolvemos el problema principal
ω_optimal = solve_problem_1()

# Análisis de sensibilidad
results = analyze_risk_aversion()

println("\n✓ Problema 1 completado exitosamente!")
println(repeat("=", 70))
println("\nNOTA: Para visualización gráfica, puedes usar los resultados en Excel")
println("      o cualquier software de gráficos con los datos generados.")
println(repeat("=", 70))