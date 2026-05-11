"""
=============================================================================
MÉTODO DE PANELES - Distribución de Manantiales (q_j) + Torbellinos (γ)
de Intensidad Constante sobre cada Panel
=============================================================================
Asignatura: Aerodinámica y Aeroelasticidad (AyA) - Vehículos Aeroespaciales
ETSIAE - Universidad Politécnica de Madrid

Metodología: Tema 3, diapositivas 87-97 (ANEXO: Método de Paneles)
Guion de trabajo: Apartados 2 y 3

Singularidades por panel:
  - Manantiales q_j de intensidad CONSTANTE en cada panel (distinta por panel)
  - Torbellinos γ de intensidad CONSTANTE e IGUAL para TODOS los paneles
  Incógnitas: q_1, q_2, ..., q_N, γ  →  N + 1 incógnitas
  Ecuaciones: N condiciones geométricas + 1 condición de Kutta = N + 1

Referencia principal: Theory of Wing Sections, Abbott & Von Doenhoff
                      + diapositivas de clase (Luis Ayuso y Rodolfo Sant)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECCIÓN 1: GEOMETRÍA DEL PERFIL NACA DE 4 CIFRAS
# =============================================================================

def linea_media_naca4(x, f, xf):
    """
    Calcula la línea media (zc) y su derivada (dzc/dx) para un perfil NACA 4 cifras.
    Se trabaja en el plano xz (x = dirección de la cuerda, z = perpendicular).
    
    Parámetros:
        x  : coordenada adimensional x/c (0 a 1)
        f  : curvatura máxima como fracción de la cuerda (ej: 0.02 para 2%)
        xf : posición de la curvatura máxima como fracción (ej: 0.40 para 40%)
    
    Retorna:
        zc    : ordenada z de la línea media
        dzc_dx: derivada de la línea media dz_c/dx
    
    Fórmulas (Tema 3, diap. 9 / Guion diap. 9):
        Para 0 ≤ x ≤ xf:  zc = (f/xf²) * (2·xf·x - x²)
        Para xf ≤ x ≤ 1:  zc = (f/(1-xf)²) * [(1-2·xf) + 2·xf·x - x²]
    """
    # Caso sin curvatura (perfil simétrico)
    if f == 0 or xf == 0:
        return np.zeros_like(x), np.zeros_like(x)
    
    zc = np.where(
        x <= xf,
        (f / xf**2) * (2.0 * xf * x - x**2),
        (f / (1.0 - xf)**2) * ((1.0 - 2.0 * xf) + 2.0 * xf * x - x**2)
    )
    dzc_dx = np.where(
        x <= xf,
        (2.0 * f / xf**2) * (xf - x),
        (2.0 * f / (1.0 - xf)**2) * (xf - x)
    )
    return zc, dzc_dx


def espesor_naca4(x, t):
    """
    Distribución de espesor para perfiles NACA de 4 cifras (plano xz).
    
    Parámetros:
        x : coordenada adimensional x/c
        t : espesor máximo como fracción de la cuerda (ej: 0.12 para 12%)
    
    Retorna:
        zt: semi-espesor en la posición x (se suma/resta en z a la línea media)
    
    Fórmula (Tema 3, diap. 9 / Guion diap. 9):
        zt(x) = ±5t·(a·√x + b·x + c·x² + d·x³ + e·x⁴)
        con a=0.2969, b=-0.1260, c=-0.3516, d=0.2843, e=-0.1015
    """
    # Coeficientes estándar NACA
    a, b, c, d, e = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
    
    zt = 5.0 * t * (a * np.sqrt(np.maximum(x, 0.0)) + b * x + c * x**2
                     + d * x**3 + e * x**4)
    return zt


def generar_perfil_naca4(f, xf, t, N):
    """
    Genera las coordenadas de un perfil NACA de 4 cifras con N paneles.
    
    Panelización con distribución coseno (concentra nodos en BA y BS).
    El recorrido va desde el Borde de Salida (BS) por el INTRADÓS,
    pasando por el Borde de Ataque (BA), y volviendo por el EXTRADÓS al BS.
    
    Parámetros:
        f  : curvatura máxima (fracción de cuerda)
        xf : posición de curvatura máxima (fracción de cuerda)
        t  : espesor máximo (fracción de cuerda)
        N  : número de paneles (N+1 nodos)
    
    Retorna:
        x_nodos, z_nodos: arrays de N+1 coordenadas
    
    Distribución coseno (Tema 3, diap. 87):
        β_i = β_1 + i·Δβ,  con Δβ = 2π/N,  β_1 = 0
        x_i = (c/2)·(1 + cos(β_i))
        
        β = 0    →  x = c  (BS)
        β = π    →  x = 0  (BA)
        β = 2π   →  x = c  (BS)
    """
    N1 = N + 1  # Número de nodos
    beta = np.linspace(0, 2.0 * np.pi, N1)
    
    # Coordenadas x a lo largo de la cuerda (distribución coseno)
    x_cos = 0.5 * (1.0 + np.cos(beta))
    
    x_nodos = np.zeros(N1)
    z_nodos = np.zeros(N1)
    
    for i in range(N1):
        xn = np.clip(x_cos[i], 0.0, 1.0)
        
        # Línea media y su derivada (en el plano xz)
        if f == 0 or xf == 0:
            zc, dzc = 0.0, 0.0
        else:
            if xn <= xf:
                zc = (f / xf**2) * (2.0 * xf * xn - xn**2)
                dzc = (2.0 * f / xf**2) * (xf - xn)
            else:
                zc = (f / (1.0 - xf)**2) * ((1.0 - 2.0 * xf) + 2.0 * xf * xn - xn**2)
                dzc = (2.0 * f / (1.0 - xf)**2) * (xf - xn)
        
        # Semi-espesor (perpendicular a la línea media)
        zt = 5.0 * t * (0.2969 * np.sqrt(max(xn, 0.0)) - 0.1260 * xn
                        - 0.3516 * xn**2 + 0.2843 * xn**3 - 0.1015 * xn**4)
        
        # Ángulo local de la línea media
        theta_c = np.arctan(dzc)
        
        if beta[i] <= np.pi:
            # INTRADÓS (lower surface): desde BS hacia BA
            x_nodos[i] = xn + zt * np.sin(theta_c)
            z_nodos[i] = zc - zt * np.cos(theta_c)
        else:
            # EXTRADÓS (upper surface): desde BA hacia BS
            x_nodos[i] = xn - zt * np.sin(theta_c)
            z_nodos[i] = zc + zt * np.cos(theta_c)
    
    return x_nodos, z_nodos


# =============================================================================
# SECCIÓN 2: MÉTODO DE PANELES
# (Distribución de manantiales q_j + torbellinos γ constantes)
# =============================================================================

def metodo_paneles(x_nodos, z_nodos, alpha_deg, V_inf=1.0):
    """
    Resuelve el flujo potencial alrededor del perfil usando el método de paneles
    con distribución de manantiales (q_j) y torbellinos (γ) de intensidad constante.
    
    METODOLOGÍA (Tema 3, diapositivas 87-97):
    
    PASO 1 - Geometría de paneles:
        - Punto de control: centro de cada panel (x̄_i, z̄_i)
        - Inclinación: θ_i = arctan(Δz_i / Δx_i)
        - Normal:  n_i = (-sin θ_i, cos θ_i)   [apunta hacia FUERA]
        - Tangente: τ_i = (cos θ_i, sin θ_i)
    
    PASO 2 - Coeficientes de influencia (diap. 89-93):
        Para el par (i, j), se calculan en ejes LOCALES del panel j:
        
        Coordenadas locales del p.c. del panel i:
            x_ij = (x̄_i - x_j)·cos θ_j + (z̄_i - z_j)·sin θ_j
            z_ij = -(x̄_i - x_j)·sin θ_j + (z̄_i - z_j)·cos θ_j
        
        Distancias y ángulos:
            r₁² = x_ij² + z_ij²
            r₂² = (x_ij - c_j)² + z_ij²
            θ₁ = arctan(z_ij / x_ij)
            θ₂ = arctan(z_ij / (x_ij - c_j))
        
        Coeficientes adimensionales en ejes locales (diap. 92):
        
            Manantiales:                     Torbellinos:
            u*_q = -(1/2π)·ln(r₂/r₁)       u*_γ = (1/2π)·(θ₂ - θ₁)
            w*_q = (1/2π)·(θ₂ - θ₁)        w*_γ = (1/2π)·ln(r₂/r₁)
        
        Auto-influencia (i = j):
            u*_q = 0,  w*_q = 1/2            u*_γ = 1/2,  w*_γ = 0
    
    PASO 3 - Rotación a ejes perfil (diap. 93):
        u_q = u*_q·cos θ_j - w*_q·sin θ_j    (idem para u_γ)
        w_q = u*_q·sin θ_j + w*_q·cos θ_j    (idem para w_γ)
    
    PASO 4 - Proyecciones normal y tangencial (diap. 94):
        V_n  = u·(-sin θ_i) + w·cos θ_i       [normal al panel i]
        V_τ  = u·cos θ_i + w·sin θ_i           [tangente al panel i]
    
    PASO 5 - Sistema de ecuaciones (diap. 94-95):
        N ecuaciones geométricas: Σ_j a_ij·q_j + a_{i,N+1}·γ = I_i
        1 ecuación de Kutta: V_τ,1 + V_τ,N = 0  (panel 1 y panel N en el BS)
        
        donde I_i = V∞·sin(θ_i - α)
    
    Parámetros:
        x_nodos, z_nodos : coordenadas de los N+1 nodos
        alpha_deg        : ángulo de ataque en grados
        V_inf            : velocidad del flujo libre (por defecto 1.0)
    
    Retorna:
        Cl, Cm_c4, Cp, vel_tang, xc, zc, gamma, q, Cd, Cl_i
        (todo en el plano xz)
    """
    alpha = np.radians(alpha_deg)
    N = len(x_nodos) - 1  # Número de paneles
    
    # ---- PASO 1: Geometría de paneles ----
    # Punto de control (centro del panel)
    xc = 0.5 * (x_nodos[:-1] + x_nodos[1:])
    zc = 0.5 * (z_nodos[:-1] + z_nodos[1:])
    
    # Incrementos
    dx = x_nodos[1:] - x_nodos[:-1]
    dz = z_nodos[1:] - z_nodos[:-1]
    
    # Longitud de cada panel
    c_panel = np.sqrt(dx**2 + dz**2)
    
    # Inclinación de cada panel (diap. 87): θ_i = arctan(Δz/Δx)
    # NOTA: se usa arctan2 para obtener el cuadrante correcto, -π ≤ θ ≤ π
    theta = np.arctan2(dz, dx)
    
    # ---- PASO 2-4: Coeficientes de influencia ----
    # Matrices de influencia:
    #   An[i,j] : contribución NORMAL de manantial q_j sobre panel i
    #   Ang[i]  : contribución NORMAL de torbellino γ (acumulada de todos los paneles j)
    #   At[i,j] : contribución TANGENCIAL de manantial q_j sobre panel i
    #   Atg[i]  : contribución TANGENCIAL de torbellino γ (acumulada)
    
    An  = np.zeros((N, N))
    Ang = np.zeros(N)
    At  = np.zeros((N, N))
    Atg = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            # -- Coordenadas locales del p.c. del panel i en ejes del panel j --
            # (Tema 3, diap. 88)
            x_ij = ((xc[i] - x_nodos[j]) * np.cos(theta[j])
                   + (zc[i] - z_nodos[j]) * np.sin(theta[j]))
            z_ij = (-(xc[i] - x_nodos[j]) * np.sin(theta[j])
                    + (zc[i] - z_nodos[j]) * np.cos(theta[j]))
            
            if i == j:
                # ============================================================
                # AUTO-INFLUENCIA (diap. 92)
                # ============================================================
                # CUIDADO: esta es una de las fuentes de error más comunes
                # de la IA según los profesores (error en la diagonal).
                #
                # Manantiales:  u*_q_ii = 0,     w*_q_ii = q_i/2  → u*=0, w*=1/2
                # Torbellinos:  u*_γ_ii = γ/2,   w*_γ_ii = 0      → u*=1/2, w*=0
                # ============================================================
                uq_star = 0.0
                wq_star = 0.5
                ug_star = 0.5
                wg_star = 0.0
            else:
                # ============================================================
                # INFLUENCIA CRUZADA (diap. 89, 92)
                # ============================================================
                # Distancias desde el p.c. i a los extremos del panel j
                r1_sq = x_ij**2 + z_ij**2
                r2_sq = (x_ij - c_panel[j])**2 + z_ij**2
                
                # Protección numérica contra log(0)
                r1 = np.sqrt(max(r1_sq, 1e-30))
                r2 = np.sqrt(max(r2_sq, 1e-30))
                
                # Ángulos (diap. 89): θ₁ y θ₂ medidos desde eje local x'
                # NOTA IMPORTANTE sobre el orden r₂/r₁ en el logaritmo:
                # El signo del logaritmo es CRUCIAL. Según la diap. 92:
                #   u*_q = -(1/2π)·ln(r₂/r₁)
                #   w*_γ = +(1/2π)·ln(r₂/r₁)
                theta1 = np.arctan2(z_ij, x_ij)
                theta2 = np.arctan2(z_ij, x_ij - c_panel[j])
                
                log_ratio = np.log(r2 / r1)
                dtheta = theta2 - theta1
                
                # Coeficientes adimensionales en ejes locales (diap. 92)
                uq_star = -(1.0 / (2.0 * np.pi)) * log_ratio    # manantial → u'
                wq_star =  (1.0 / (2.0 * np.pi)) * dtheta        # manantial → w'
                ug_star =  (1.0 / (2.0 * np.pi)) * dtheta        # torbellino → u'
                wg_star =  (1.0 / (2.0 * np.pi)) * log_ratio     # torbellino → w'
            
            # ---- PASO 3: Rotación a ejes globales (perfil) ----
            # (Tema 3, diap. 93)
            cos_j = np.cos(theta[j])
            sin_j = np.sin(theta[j])
            
            # Velocidad inducida por manantiales, en ejes perfil
            uq_global = uq_star * cos_j - wq_star * sin_j
            wq_global = uq_star * sin_j + wq_star * cos_j
            
            # Velocidad inducida por torbellinos, en ejes perfil
            ug_global = ug_star * cos_j - wg_star * sin_j
            wg_global = ug_star * sin_j + wg_star * cos_j
            
            # ---- PASO 4: Proyecciones normal y tangencial ----
            # Normal:  n_i = (-sin θ_i, cos θ_i)
            # Tangente: τ_i = (cos θ_i, sin θ_i)
            sin_i = np.sin(theta[i])
            cos_i = np.cos(theta[i])
            
            # Contribución NORMAL del panel j (manantiales y torbellinos)
            An[i, j] = -uq_global * sin_i + wq_global * cos_i
            Ang[i]  += -ug_global * sin_i + wg_global * cos_i
            
            # Contribución TANGENCIAL del panel j (manantiales y torbellinos)
            At[i, j] =  uq_global * cos_i + wq_global * sin_i
            Atg[i]  +=  ug_global * cos_i + wg_global * sin_i
    
    # ---- PASO 5: Ensamblaje del sistema de ecuaciones ----
    # Tamaño (N+1) x (N+1)
    A_sis = np.zeros((N + 1, N + 1))
    b_sis = np.zeros(N + 1)
    
    # --- Primeras N filas: condición geométrica (V·n = 0) ---
    # Σ_j An[i,j]·q_j + Ang[i]·γ = I_i
    # donde I_i = V∞·sin(θ_i - α)  [proyección de V∞ sobre la normal, cambiada de signo]
    A_sis[:N, :N] = An           # Coeficientes de q_j
    A_sis[:N, N]  = Ang          # Coeficiente de γ
    b_sis[:N]     = V_inf * np.sin(theta - alpha)  # Término independiente I_i
    
    # --- Fila N+1: Condición de Kutta (diap. 94) ---
    # La presión debe ser igual en extradós e intradós en el BS.
    # Esto se traduce en: V_τ,1 + V_τ,N = 0
    # (velocidades tangenciales en el primer y último panel suman cero)
    #
    # V_τ,i = Σ_j At[i,j]·q_j + Atg[i]·γ + V∞·cos(θ_i - α)
    #
    # Para panel 1 (i=0) y panel N (i=N-1):
    A_sis[N, :N] = At[0, :] + At[N-1, :]
    A_sis[N, N]  = Atg[0] + Atg[N-1]
    b_sis[N]     = -V_inf * (np.cos(theta[0] - alpha) + np.cos(theta[N-1] - alpha))
    
    # ---- Resolución del sistema lineal ----
    solucion = np.linalg.solve(A_sis, b_sis)
    q     = solucion[:N]     # Intensidades de los manantiales (una por panel)
    gamma = solucion[N]      # Intensidad del torbellino (única para todos los paneles)
    
    # ---- Cálculo de la velocidad tangencial en cada panel ----
    # V_τ,i = Σ_j At[i,j]·q_j + Atg[i]·γ + V∞·cos(θ_i - α)
    # (Tema 3, diap. 96)
    vel_tang = At @ q + Atg * gamma + V_inf * np.cos(theta - alpha)
    
    # ---- Coeficiente de presión ----
    # Cp = 1 - (V_local / V∞)²
    Cp = 1.0 - (vel_tang / V_inf)**2
    
    # ================================================================
    # CÁLCULO DE Cl, Cm_c/4 - PANEL POR PANEL
    # ================================================================
    # NOTA IMPORTANTE: El Cl se calcula integrando Cp sobre cada panel
    # (fuerza de presión × proyección), NO como un único torbellino
    # usando Kutta-Yukovski Γ = ρV∞Cl·c/2.
    # Esto es una fuente de error frecuente señalada por los profesores.
    #
    # La fuerza de presión sobre el panel i actúa en dirección normal:
    #   dF = -Cp_i · q∞ · c_i · n_i
    #
    # En componentes del perfil (body axes):
    #   dCFx_i = Cp_i · Δz_i / c    (fuerza en x por unidad de q∞·c)
    #   dCFz_i = -Cp_i · Δx_i / c   (fuerza en z por unidad de q∞·c)
    #
    # Coeficientes aerodinámicos:
    #   CN = Σ dCFz = -Σ Cp_i · Δx_i / c     (fuerza normal, + hacia arriba)
    #   CA = Σ dCFx =  Σ Cp_i · Δz_i / c     (fuerza axial, + hacia BS)
    #   Cl = CN·cos α - CA·sin α
    #   Cd = CN·sin α + CA·cos α  (≈ 0 en flujo potencial)
    # ================================================================
    
    c_cuerda = 1.0  # Cuerda unitaria
    
    # Fuerza normal (perpendicular a la cuerda, positiva hacia arriba)
    CN = -np.sum(Cp * dx) / c_cuerda
    
    # Fuerza axial (a lo largo de la cuerda, positiva hacia BS)
    CA = np.sum(Cp * dz) / c_cuerda
    
    # Coeficiente de sustentación (perpendicular a V∞)
    Cl = CN * np.cos(alpha) - CA * np.sin(alpha)
    
    # Coeficiente de resistencia de presión (debería ser ≈ 0 en flujo potencial)
    Cd = CN * np.sin(alpha) + CA * np.cos(alpha)
    
    # ================================================================
    # MOMENTO RESPECTO AL BORDE DE ATAQUE (positivo nariz arriba)
    # ================================================================
    # Cm_le = Σ_i [-Cp_i · (x̄_i · Δx_i + z̄_i · Δz_i)] / c²
    #
    # NOTA: El signo se define cuidadosamente:
    # El momento "matemático" (producto cruz) da positivo = nariz abajo.
    # La convención aeronáutica usa positivo = nariz arriba.
    # Derivación:
    #   M_le (math, CCW+) = Σ [x̄_i · dFz_i - z̄_i · dFx_i]
    #   Cm_le (aero, nose-up+) = -M_le / (q∞·c²)
    #   = -Σ [x̄_i·(-Cp_i·Δx_i) - z̄_i·(Cp_i·Δz_i)] / c²
    #   = Σ Cp_i · (x̄_i·Δx_i + z̄_i·Δz_i) / c²
    # ================================================================
    
    Cm_le = np.sum(Cp * (xc * dx + zc * dz)) / c_cuerda**2
    
    # Traslación al cuarto de cuerda:
    # Cm_c/4 = Cm_le + CN · 0.25
    # (Tema 3, diap. 6: Cm_B = Cm_A + Cl·(xB - xA)/c, con CN ≈ Cl para α pequeño)
    Cm_c4 = Cm_le + CN * 0.25
    Cl_i = (-Cp * dx * np.cos(alpha) - Cp * dz * np.sin(alpha)) / c_cuerda
    return Cl, Cm_c4, Cp, vel_tang, xc, zc, gamma, q, Cd, Cl_i


# =============================================================================
# SECCIÓN 3: ESTUDIO DE CONVERGENCIA
# =============================================================================

def estudio_convergencia(f, xf, t, alpha_deg, N_inicio=40, factor=1.2,
                         eps_objetivo=0.01, N_max=1000):
    """
    Estudio de convergencia según el criterio del guion (diap. 11):
    
    1. Comenzar con N_inicio paneles (40).
    2. Calcular Cl_i.
    3. Aumentar paneles un 20% (factor=1.2). Calcular Cl_{i+1}.
    4. Error relativo: ε_i = |Cl_{i+1} - Cl_i| / |Cl_i|
    5. Repetir hasta ε < 1%.
    
    Retorna:
        N_list    : lista de números de paneles usados
        Cl_list   : lista de Cl correspondientes
        eps_list  : lista de errores relativos
        N_opt     : número de paneles óptimo (primer N con ε < 1%)
    """
    N_list = []
    Cl_list = []
    eps_list = []
    
    N_actual = N_inicio
    
    # Primera iteración
    x, z = generar_perfil_naca4(f, xf, t, N_actual)
    Cl_actual, _, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
    
    N_list.append(N_actual)
    Cl_list.append(Cl_actual)
    
    N_opt = None
    
    while N_actual < N_max:
        # Aumentar paneles un 20%
        N_nuevo = int(np.ceil(N_actual * factor))
        if N_nuevo == N_actual:
            N_nuevo += 2  # Asegurar incremento (y par para simetría)
        # Hacer que sea par para simetría intradós/extradós
        if N_nuevo % 2 != 0:
            N_nuevo += 1
        
        x, z = generar_perfil_naca4(f, xf, t, N_nuevo)
        Cl_nuevo, _, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
        
        # Error relativo
        if abs(Cl_actual) > 1e-10:
            eps = abs(Cl_nuevo - Cl_actual) / abs(Cl_actual)
        else:
            eps = abs(Cl_nuevo - Cl_actual)
        
        N_list.append(N_nuevo)
        Cl_list.append(Cl_nuevo)
        eps_list.append(eps)
        
        # Verificar convergencia
        if eps < eps_objetivo and N_opt is None:
            N_opt = N_nuevo
        
        Cl_actual = Cl_nuevo
        N_actual = N_nuevo
        
        # Si ya convergió, hacer algunas iteraciones más para el gráfico
        if N_opt is not None and N_actual > N_opt * 2:
            break
    
    if N_opt is None:
        N_opt = N_list[-1]
        print(f"  ⚠ No se alcanzó convergencia < {eps_objetivo*100}%. Usando N = {N_opt}")
    
    return N_list, Cl_list, eps_list, N_opt


# =============================================================================
# SECCIÓN 4: RETO DE DISEÑO (Apartado 3 del guion)
# =============================================================================

def reto_diseno(f_base, xf_base, t_base, alpha_deg, delta_Cl=0.2, N=200):
    """
    Optimización paramétrica para incrementar Cl en +0.2 (plano xz),
    manteniendo |ΔCm_c/4| lo más bajo posible.
    
    Estrategia analítica (Tema 3, diap. 15-16):
        Cl ≈ 2π(α + 2f/c)    → f es el DRIVER PRINCIPAL de Cl
        Cm_c/4 ≈ -π·f/c      → Cm depende SOLO de f
        Cl_α ≈ 2π(1+0.83t/c) → t tiene efecto SECUNDARIO
        xf no cambia Cl en flujo potencial (solo cambia forma del Cp)
    """
    # ---- Valores base ----
    x_base, z_base = generar_perfil_naca4(f_base, xf_base, t_base, N)
    Cl_base, Cm_base, _, _, _, _, _, _, _, _ = metodo_paneles(x_base, z_base, alpha_deg)
    Cl_objetivo = Cl_base + delta_Cl
    
    print(f"\n{'='*70}")
    print(f"  APARTADO 3: RETO DE DISEÑO E INGENIERÍA INVERSA")
    print(f"{'='*70}")
    print(f"  Perfil base: f={f_base*100:.1f}%, xf={xf_base*100:.0f}%, t={t_base*100:.0f}%")
    print(f"  α = {alpha_deg}°")
    print(f"  Cl_base    = {Cl_base:.6f}")
    print(f"  Cm_base    = {Cm_base:.6f}")
    print(f"  Cl_objetivo = {Cl_objetivo:.6f}  (ΔCl = +{delta_Cl})")
    
    # =================================================================
    # PASO 1: ANÁLISIS DE SENSIBILIDAD - Efecto de cada parámetro
    # =================================================================
    print(f"\n{'-'*70}")
    print(f"  PASO 1: ANÁLISIS DE SENSIBILIDAD (un parámetro a la vez)")
    print(f"{'-'*70}")
    
    # --- 1a. Efecto de la curvatura f ---
    print(f"\n  1a) Efecto de la curvatura f (xf={xf_base*100:.0f}% fijo, t={t_base*100:.0f}% fijo):")
    print(f"      TPL predice: dCl/d(f/c) ≈ 4π ≈ 12.57")
    print(f"      {'f (%)':>8s}  {'Cl':>10s}  {'ΔCl':>10s}  {'Cm_c/4':>10s}  {'Δ|Cm|':>10s}")
    print(f"      {'-'*55}")
    f_scan = np.arange(max(0.5, f_base*100 - 1), f_base*100 + 6, 0.5) / 100.0
    for fv in f_scan:
        x, z = generar_perfil_naca4(fv, xf_base, t_base, N)
        Cl_v, Cm_v, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
        marca = " ◄── base" if abs(fv - f_base) < 1e-6 else ""
        marca = " ◄── OBJETIVO" if abs(Cl_v - Cl_objetivo) < 0.015 else marca
        print(f"      {fv*100:8.1f}  {Cl_v:10.6f}  {Cl_v-Cl_base:+10.4f}  {Cm_v:10.6f}  {abs(Cm_v)-abs(Cm_base):+10.4f}{marca}")
    
    # --- 1b. Efecto de la posición de curvatura xf ---
    print(f"\n  1b) Efecto de la posición xf (f={f_base*100:.1f}% fijo, t={t_base*100:.0f}% fijo):")
    print(f"      TPL predice: Cl NO depende de xf → efecto pequeño")
    print(f"      {'xf (%)':>8s}  {'Cl':>10s}  {'ΔCl':>10s}  {'Cm_c/4':>10s}  {'Δ|Cm|':>10s}")
    print(f"      {'-'*55}")
    xf_scan = np.arange(20, 70, 10) / 100.0
    for xfv in xf_scan:
        x, z = generar_perfil_naca4(f_base, xfv, t_base, N)
        Cl_v, Cm_v, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
        marca = " ◄── base" if abs(xfv - xf_base) < 1e-6 else ""
        print(f"      {xfv*100:8.0f}  {Cl_v:10.6f}  {Cl_v-Cl_base:+10.4f}  {Cm_v:10.6f}  {abs(Cm_v)-abs(Cm_base):+10.4f}{marca}")
    
    # --- 1c. Efecto del espesor t ---
    print(f"\n  1c) Efecto del espesor t (f={f_base*100:.1f}% fijo, xf={xf_base*100:.0f}% fijo):")
    print(f"      TPL predice: Cl_α ≈ 2π(1+0.83·t/c) → efecto secundario")
    print(f"      {'t (%)':>8s}  {'Cl':>10s}  {'ΔCl':>10s}  {'Cm_c/4':>10s}  {'Δ|Cm|':>10s}")
    print(f"      {'-'*55}")
    t_scan = np.arange(max(6, t_base*100 - 4), t_base*100 + 8, 2) / 100.0
    for tv in t_scan:
        x, z = generar_perfil_naca4(f_base, xf_base, tv, N)
        Cl_v, Cm_v, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
        marca = " ◄── base" if abs(tv - t_base) < 1e-6 else ""
        print(f"      {tv*100:8.0f}  {Cl_v:10.6f}  {Cl_v-Cl_base:+10.4f}  {Cm_v:10.6f}  {abs(Cm_v)-abs(Cm_base):+10.4f}{marca}")
    
    # --- Conclusión del análisis de sensibilidad ---
    print(f"\n  CONCLUSIÓN DE SENSIBILIDAD:")
    print(f"    → f (curvatura) es el parámetro DOMINANTE: ΔCl ≈ 12·Δ(f/c)")
    print(f"    → xf (posición) apenas cambia Cl en flujo potencial")
    print(f"    → t (espesor) tiene efecto pequeño a través del factor (1+0.83·t/c)")
    print(f"    → Para ΔCl = +0.2: Δf/c ≈ 0.2/4π ≈ 0.016 → f_nueva ≈ {f_base*100:.1f}+1.6 = {f_base*100+1.6:.1f}%")
    
    # =================================================================
    # PASO 2: BARRIDO PARAMÉTRICO FINO
    # =================================================================
    print(f"\n{'-'*70}")
    print(f"  PASO 2: BARRIDO PARAMÉTRICO FINO")
    print(f"{'-'*70}")
    
    # Rangos : f fino (es el driver), xf y t solo 3 valores cada uno
    f_range = np.arange(max(0.5, f_base*100), f_base*100 + 5, 0.1) / 100.0
    xf_range = np.array([max(10, xf_base*100 - 10), xf_base*100, min(90, xf_base*100 + 10)]) / 100.0
    t_range = np.array([t_base*100, t_base*100 + 1, t_base*100 + 2]) / 100.0
    
    n_total = len(f_range) * len(xf_range) * len(t_range)
    print(f"  Combinaciones a evaluar: {len(f_range)} f × {len(xf_range)} xf × {len(t_range)} t = {n_total}")
    
    # Usar pocos paneles para la búsqueda (rápido), luego verificar con N completo
    N_busqueda = 80
    print(f"  Paneles en búsqueda: {N_busqueda} (se verifica después con N={N})")
    
    mejor_resultado = None
    mejor_delta_cm = np.inf
    resultados = []
    n_eval = 0
    
    for t_val in t_range:
        for xf_val in xf_range:
            for f_val in f_range:
                n_eval += 1
                if n_eval % 200 == 0:
                    print(f"    ... evaluando {n_eval}/{n_total}")
                x, z = generar_perfil_naca4(f_val, xf_val, t_val, N_busqueda)
                Cl, Cm, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
                
                if abs(Cl - Cl_objetivo) < 0.01:
                    delta_cm = abs(Cm - Cm_base)
                    resultados.append({
                        'f': f_val, 'xf': xf_val, 't': t_val,
                        'Cl': Cl, 'Cm': Cm, 'delta_Cm': delta_cm
                    })
                    if delta_cm < mejor_delta_cm:
                        mejor_delta_cm = delta_cm
                        mejor_resultado = {
                            'f': f_val, 'xf': xf_val, 't': t_val,
                            'Cl': Cl, 'Cm': Cm, 'delta_Cm': delta_cm
                        }
    
    if mejor_resultado is None:
        print("  ⚠ No se encontró solución con |ΔCl| < 0.01. Buscando la más cercana...")
        mejor_delta_cl = np.inf
        for t_val in t_range:
            for xf_val in xf_range:
                for f_val in f_range:
                    x, z = generar_perfil_naca4(f_val, xf_val, t_val, N_busqueda)
                    Cl, Cm, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
                    delta_cl_abs = abs(Cl - Cl_objetivo)
                    delta_cm = abs(Cm - Cm_base)
                    if delta_cl_abs < mejor_delta_cl:
                        mejor_delta_cl = delta_cl_abs
                        mejor_resultado = {
                            'f': f_val, 'xf': xf_val, 't': t_val,
                            'Cl': Cl, 'Cm': Cm, 'delta_Cm': delta_cm
                        }
    
    # VERIFICACIÓN: recalcular el óptimo y las top soluciones con N paneles completo
    print(f"\n  Verificando mejores candidatos con N={N} paneles...")
    for r in resultados:
        x, z = generar_perfil_naca4(r['f'], r['xf'], r['t'], N)
        Cl, Cm, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
        r['Cl'] = Cl
        r['Cm'] = Cm
        r['delta_Cm'] = abs(Cm - Cm_base)
    
    # Recalcular el mejor con N completo
    x, z = generar_perfil_naca4(mejor_resultado['f'], mejor_resultado['xf'], mejor_resultado['t'], N)
    Cl, Cm, _, _, _, _, _, _, _, _ = metodo_paneles(x, z, alpha_deg)
    mejor_resultado['Cl'] = Cl
    mejor_resultado['Cm'] = Cm
    mejor_resultado['delta_Cm'] = abs(Cm - Cm_base)
    
    # =================================================================
    # PASO 3: RESULTADOS
    # =================================================================
    print(f"\n{'-'*70}")
    print(f"  PASO 3: RESULTADOS DEL RETO DE DISEÑO")
    print(f"{'-'*70}")
    
    print(f"\n  Soluciones que cumplen |Cl - Cl_obj| < 0.01: {len(resultados)}")
    if len(resultados) > 0:
        # Mostrar las 10 mejores
        resultados_ord = sorted(resultados, key=lambda r: r['delta_Cm'])
        n_mostrar = min(10, len(resultados_ord))
        print(f"\n  Top {n_mostrar} soluciones (ordenadas por menor |ΔCm|):")
        print(f"  {'#':>3s}  {'f(%)':>6s}  {'xf(%)':>6s}  {'t(%)':>6s}  {'Cl':>10s}  {'Cm_c/4':>10s}  {'ΔCl':>8s}  {'Δ|Cm|':>8s}")
        print(f"  {'-'*68}")
        for k, r in enumerate(resultados_ord[:n_mostrar]):
            print(f"  {k+1:3d}  {r['f']*100:6.1f}  {r['xf']*100:6.0f}  {r['t']*100:6.0f}  "
                  f"{r['Cl']:10.6f}  {r['Cm']:10.6f}  {r['Cl']-Cl_base:+8.4f}  {r['delta_Cm']:8.4f}")
    
    f_opt  = mejor_resultado['f']
    xf_opt = mejor_resultado['xf']
    t_opt  = mejor_resultado['t']
    
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │          PERFIL OPTIMIZADO SELECCIONADO       │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  f/c   = {f_opt*100:5.1f}%  (base: {f_base*100:.1f}%)            │")
    print(f"  │  xf    = {xf_opt*100:5.0f}%  (base: {xf_base*100:.0f}%)             │")
    print(f"  │  t/c   = {t_opt*100:5.0f}%  (base: {t_base*100:.0f}%)             │")
    print(f"  │  Cl    = {mejor_resultado['Cl']:.6f}  (ΔCl = {mejor_resultado['Cl']-Cl_base:+.4f})  │")
    print(f"  │  Cm_c/4= {mejor_resultado['Cm']:.6f}  (Δ|Cm|= {mejor_resultado['delta_Cm']:.4f})   │")
    print(f"  └─────────────────────────────────────────────┘")
    
    # --- Explicación física de por qué varía cada parámetro ---
    print(f"\n  ANÁLISIS CRÍTICO (¿Por qué estos valores?):")
    print(f"  ─────────────────────────────────────────────")
    
    delta_f = f_opt - f_base
    delta_xf = xf_opt - xf_base
    delta_t = t_opt - t_base
    
    print(f"\n  • Curvatura f: {f_base*100:.1f}% → {f_opt*100:.1f}% (Δf = {delta_f*100:+.1f}%)")
    print(f"    Es el cambio principal. En TPL: Cl_0 = 4π·f/c. Aumentar f")
    print(f"    incrementa directamente la sustentación a ángulo de ataque cero")
    print(f"    porque la línea media más curvada genera mayor asimetría entre")
    print(f"    extradós e intradós, acelerando el flujo arriba y decelerándolo abajo.")
    print(f"    Coste: Cm_c/4 = -π·f/c, así que más curvatura → más momento picador.")
    
    if abs(delta_xf) > 1e-6:
        print(f"\n  • Posición xf: {xf_base*100:.0f}% → {xf_opt*100:.0f}% (Δxf = {delta_xf*100:+.0f}%)")
        print(f"    En flujo potencial, xf no cambia Cl de forma significativa.")
        print(f"    Su efecto principal es redistribuir la carga de presión a lo largo")
        print(f"    de la cuerda: xf más adelante concentra la succión cerca del BA.")
        print(f"    En flujo viscoso, esto sí importa: afecta al gradiente de presión")
        print(f"    adverso y por tanto a la posición de desprendimiento y al Cl_max.")
    else:
        print(f"\n  • Posición xf: se mantiene en {xf_opt*100:.0f}%")
        print(f"    Coherente con TPL: xf no afecta a Cl en flujo potencial.")
    
    if abs(delta_t) > 1e-6:
        print(f"\n  • Espesor t: {t_base*100:.0f}% → {t_opt*100:.0f}% (Δt = {delta_t*100:+.0f}%)")
        print(f"    El espesor afecta a Cl a través del factor (1+0.83·t/c) que multiplica")
        print(f"    a 2πα. A mayor espesor, más sustentación para un α dado, pero el efecto")
        print(f"    es pequeño comparado con cambiar f. No afecta a Cm_c/4 significativamente.")
    else:
        print(f"\n  • Espesor t: se mantiene en {t_opt*100:.0f}%")
        print(f"    Confirma que el espesor no es un driver relevante para ΔCl en potencial.")
    
    return f_opt, xf_opt, t_opt, mejor_resultado, Cl_base, Cm_base, resultados


# =============================================================================
# SECCIÓN 5: EXPORTACIÓN DE COORDENADAS PARA XFOIL
# =============================================================================

def exportar_para_xfoil(x_nodos, z_nodos, nombre_archivo="perfil_optimizado.dat"):
    """
    Exporta las coordenadas del perfil en formato compatible con XFOIL.
    XFOIL espera: coordenadas desde BS por extradós, BA, intradós, BS.
    Nuestro formato va: BS → intradós → BA → extradós → BS.
    Hay que invertir el orden.
    """
    # Invertir: XFOIL usa extradós → BA → intradós
    x_xfoil = x_nodos[::-1]
    z_xfoil = z_nodos[::-1]
    
    with open(nombre_archivo, 'w') as f:
        f.write(f"Perfil NACA optimizado\n")
        for i in range(len(x_xfoil)):
            f.write(f"  {x_xfoil[i]:.6f}  {z_xfoil[i]:.6f}\n")
    
    print(f"  Coordenadas exportadas a: {nombre_archivo}")
    return nombre_archivo


# =============================================================================
# SECCIÓN 6: GRÁFICOS DE CALIDAD
# =============================================================================

def configurar_grafico():
    """Configuración global de matplotlib para gráficos de calidad profesional."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })

configurar_grafico()


def graficar_perfil(x, z, titulo="Perfil NACA", archivo=None):
    """Dibuja el perfil aerodinámico con ejes proporcionados."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, z, 'b-', linewidth=1.5)
    ax.fill(x, z, alpha=0.1, color='blue')
    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c')
    ax.set_title(titulo)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, bbox_inches='tight')
    plt.show()


def graficar_cp(xc, Cp, titulo="Distribución de Cp", archivo=None,
                xc2=None, Cp2=None, label1="Panel Method", label2="XFOIL"):
    """Dibuja la distribución de Cp (eje y invertido, convención aeronáutica)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xc, Cp, 'b-o', markersize=2, label=label1)
    if xc2 is not None and Cp2 is not None:
        ax.plot(xc2, Cp2, 'r--s', markersize=2, label=label2)
        ax.legend()
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title(titulo)
    ax.invert_yaxis()  # Convención: Cp negativo arriba
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, bbox_inches='tight')
    plt.show()


def graficar_convergencia(N_list, Cl_list, eps_list, N_opt, archivo=None):
    """
    Gráfico de Cl(N) y ε(N) con doble escala (apartado 2.4 del guion).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Número de paneles (N)')
    ax1.set_ylabel('Cl', color=color1)
    ax1.plot(N_list, Cl_list, 'o-', color=color1, label='Cl(N)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=N_opt, color='green', linestyle='--', alpha=0.7, label=f'N_opt = {N_opt}')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Error relativo ε (%)', color=color2)
    # eps_list tiene un elemento menos que N_list
    ax2.plot(N_list[1:], [e * 100 for e in eps_list], 's--', color=color2, label='ε(N) [%]')
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Objetivo: ε = 1%')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax1.set_title('Estudio de Convergencia: Cl(N) y Error Relativo ε(N)')
    fig.tight_layout()
    if archivo:
        plt.savefig(archivo, bbox_inches='tight')
    plt.show()


def graficar_comparacion_perfiles(x1, z1, x2, z2, label1, label2, archivo=None):
    """Dibuja dos perfiles superpuestos (apartado 3.2)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x1, z1, 'b-', linewidth=1.5, label=label1)
    ax.plot(x2, z2, 'r--', linewidth=1.5, label=label2)
    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c')
    ax.set_title('Comparación de perfiles')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, bbox_inches='tight')
    plt.show()


def graficar_cp_comparacion(xc1, Cp1, xc2, Cp2, label1, label2, archivo=None):
    """Dibuja dos distribuciones de Cp superpuestas (apartado 3.3)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xc1, Cp1, 'b-o', markersize=2, label=label1)
    ax.plot(xc2, Cp2, 'r--s', markersize=2, label=label2)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('Comparación de distribuciones de Cp')
    ax.invert_yaxis()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, bbox_inches='tight')
    plt.show()


# =============================================================================
# SECCIÓN 7: EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecución completa de los apartados 2 y 3 del trabajo.
    
    ⚠ IMPORTANTE: Modifica las variables f, xf, t y alpha según tu DNI.
    Las fórmulas del guion (diap. 8) son:
    
    Para DNI: 12345XYZ
        Perfil NACA:
            f   = 1 + E(X/8) · 3        [en % de cuerda]
            xf  = 20 + 10·E(Y/8) · 2    [en % de cuerda]
            t   = 10 + E(Z/9) · 8        [en % de cuerda]
        
        Perfil triangular:
            α   = (1 + 0.3·X)°
            xv  = 0.25 + 0.05·Y
            zv  = 0.1 + 0.01·Z
    
    donde E(x) es la parte entera de x.
    
    EJEMPLO: DNI 12345678 → X=6, Y=7, Z=8
        f  = 1 + E(6/8)·3 = 1 + 0·3 = 1%
        xf = 20 + 10·E(7/8)·2 = 20 + 10·0·2 = 20%
        t  = 10 + E(8/9)·8 = 10 + 0·8 = 10%
        → NACA 1210
    """
    
    print("=" * 70)
    print("  MÉTODO DE PANELES - PERFILES NACA DE 4 CIFRAS")
    print("  Distribución de manantiales (qj) + torbellinos (γ) constantes")
    print("=" * 70)
    
    # ================================================================
    # PARÁMETROS DEL PERFIL - MODIFICAR SEGÚN TU DNI
    # ================================================================
    f  = 0.03    # Curvatura máxima (fracción de cuerda)
    xf = 0.3    # Posición de curvatura máxima (fracción de cuerda)
    t  = 0.16    # Espesor máximo (fracción de cuerda)
    
    # Ángulo de ataque del perfil triangular (se usa también para el NACA)
    # α = (1 + 0.3·X)°  → Modificar según DNI
    alpha_deg = 2.8  # Ejemplo para X=6: α = 1 + 0.3·6 = 2.8°
    
    nombre_perfil = f"NACA {int(f*100)}{int(xf*10)}{int(t*100):02d}"
    
    print(f"\n  Perfil: {nombre_perfil}")
    print(f"  f = {f*100:.1f}%, xf = {xf*100:.0f}%, t = {t*100:.0f}%")
    print(f"  α = {alpha_deg}°\n")
    
    # ================================================================
    # APARTADO 2.1: Código de paneles con 100 paneles iniciales
    # ================================================================
    print("-" * 60)
    print("  APARTADO 2.1: Implementación con 100 paneles")
    print("-" * 60)
    
    N_inicial = 100
    x_100, z_100 = generar_perfil_naca4(f, xf, t, N_inicial)
    Cl_100, Cm_100, Cp_100, vel_100, xc_100, zc_100, gamma_100, q_100, Cd_100, Cl_i_100 = \
    metodo_paneles(x_100, z_100, alpha_deg)
    
    print(f"  N = {N_inicial} paneles:")
    print(f"    Cl     = {Cl_100:.6f}")
    print(f"    Cm_c/4 = {Cm_100:.6f}")
    print(f"    Cd     = {Cd_100:.6f}  (debería ser ≈ 0)")
    print(f"    γ      = {gamma_100:.6f}")
    
    # Dibujar el perfil
    graficar_perfil(x_100, z_100,
                    titulo=f"Perfil {nombre_perfil} ({N_inicial} paneles)",
                    archivo="fig_perfil_base.png")
    
    # ================================================================
    # APARTADO 2.2: Comparación con XFOIL no viscoso
    # ================================================================
    print(f"\n{'-'*60}")
    print(f"  APARTADO 2.2: Comparación con XFOIL (valores de referencia)")
    print(f"{'-'*60}")
    print(f"  → Ejecuta XFOIL con el perfil {nombre_perfil} a α = {alpha_deg}°")
    print(f"    en modo NO viscoso (inviscid) y anota Cl y Cm_c/4.")
    print(f"  → Introduce aquí los valores de XFOIL para calcular el error %.")
    
    # Valores de XFOIL (REEMPLAZAR con tus datos):
    Cl_xfoil = None  # Ej: 0.7523
    Cm_xfoil = None  # Ej: -0.0532
    
    if Cl_xfoil is not None:
        error_Cl = abs(Cl_100 - Cl_xfoil) / abs(Cl_xfoil) * 100
        print(f"\n  Cl (código)  = {Cl_100:.6f}")
        print(f"  Cl (XFOIL)   = {Cl_xfoil:.6f}")
        print(f"  Error Cl     = {error_Cl:.2f}%")
    if Cm_xfoil is not None:
        error_Cm = abs(Cm_100 - Cm_xfoil) / abs(Cm_xfoil) * 100
        print(f"  Cm (código)  = {Cm_100:.6f}")
        print(f"  Cm (XFOIL)   = {Cm_xfoil:.6f}")
        print(f"  Error Cm     = {error_Cm:.2f}%")
    
    # ================================================================
    # APARTADO 2.3: Distribución de Cp
    # ================================================================
    print(f"\n{'-'*60}")
    print(f"  APARTADO 2.3: Distribución de Cp(x)")
    print(f"{'-'*60}")
    
    graficar_cp(xc_100, Cp_100,
                titulo=f"Distribución de Cp - {nombre_perfil} a α = {alpha_deg}° (N = {N_inicial})",
                archivo="fig_cp_100paneles.png")
    
    # ================================================================
    # APARTADO 2.4-2.5: Estudio de convergencia
    # ================================================================
    print(f"\n{'-'*60}")
    print(f"  APARTADO 2.4-2.5: Estudio de convergencia")
    print(f"{'-'*60}")
    
    N_list, Cl_list, eps_list, N_opt = estudio_convergencia(
        f, xf, t, alpha_deg, N_inicio=40, factor=1.2, eps_objetivo=0.01
    )
    
    print(f"\n  Resultados del estudio de convergencia:")
    print(f"  {'N':>6s}  {'Cl':>12s}  {'ε (%)':>10s}")
    print(f"  {'-'*30}")
    for k in range(len(N_list)):
        if k == 0:
            print(f"  {N_list[k]:6d}  {Cl_list[k]:12.6f}  {'---':>10s}")
        else:
            print(f"  {N_list[k]:6d}  {Cl_list[k]:12.6f}  {eps_list[k-1]*100:10.4f}")
    
    print(f"\n  ✓ Número de paneles óptimo: N_opt = {N_opt}")
    
    graficar_convergencia(N_list, Cl_list, eps_list, N_opt,
                          archivo="fig_convergencia.png")
    
    # Recalcular con N_opt
    x_opt, z_opt = generar_perfil_naca4(f, xf, t, N_opt)
    Cl_opt, Cm_opt, Cp_opt, vel_opt, xc_opt, zc_opt, _, _, Cd_opt, _ = \
    metodo_paneles(x_opt, z_opt, alpha_deg)
    
    print(f"\n  Resultados con N_opt = {N_opt}:")
    print(f"    Cl     = {Cl_opt:.6f}")
    print(f"    Cm_c/4 = {Cm_opt:.6f}")
    print(f"    Cd     = {Cd_opt:.6f}")
    
    graficar_cp(xc_opt, Cp_opt,
                titulo=f"Distribución de Cp - {nombre_perfil} a α = {alpha_deg}° (N_opt = {N_opt})",
                archivo="fig_cp_Nopt.png")
    
    # ================================================================
    # APARTADO 3: RETO DE DISEÑO (la función reto_diseno imprime todo)
    # ================================================================
    
    N_diseno = min(N_opt, 200)
    
    f_new, xf_new, t_new, resultado, Cl_base, Cm_base, todos_resultados = \
        reto_diseno(f, xf, t, alpha_deg, delta_Cl=0.2, N=N_diseno)
    
    # Generar el perfil optimizado para gráficos
    x_new, z_new = generar_perfil_naca4(f_new, xf_new, t_new, N_diseno)
    Cl_new, Cm_new, Cp_new, vel_new, xc_new, zc_new, _, _, _, _ = \
    metodo_paneles(x_new, z_new, alpha_deg)
    
    nombre_nuevo = f"Modificado (f={f_new*100:.1f}%, xf={xf_new*100:.0f}%, t={t_new*100:.0f}%)"
    
    # Apartado 3.2: Comparación de perfiles
    x_base_plot, z_base_plot = generar_perfil_naca4(f, xf, t, N_diseno)
    graficar_comparacion_perfiles(
        x_base_plot, z_base_plot, x_new, z_new,
        label1=nombre_perfil, label2=nombre_nuevo,
        archivo="fig_comparacion_perfiles.png"
    )
    
    # Apartado 3.3: Comparación de Cp
    Cl_b, Cm_b, Cp_b, _, xc_b, _, _, _, _, _ = metodo_paneles(x_base_plot, z_base_plot, alpha_deg)
    graficar_cp_comparacion(
        xc_b, Cp_b, xc_new, Cp_new,
        label1=nombre_perfil, label2=nombre_nuevo,
        archivo="fig_comparacion_cp.png"
    )
    
    # ================================================================
    # EXPORTAR PERFIL OPTIMIZADO PARA XFOIL (Apartado 4)
    # ================================================================
    print(f"\n{'-'*60}")
    print(f"  EXPORTACIÓN para XFOIL (Apartado 4)")
    print(f"{'-'*60}")
    
    # Generar con muchos paneles para exportación suave
    x_export, z_export = generar_perfil_naca4(f_new, xf_new, t_new, 200)
    exportar_para_xfoil(x_export, z_export, "perfil_optimizado.dat")
    
    # También exportar el perfil base
    x_exp_base, z_exp_base = generar_perfil_naca4(f, xf, t, 200)
    exportar_para_xfoil(x_exp_base, z_exp_base, "perfil_base.dat")
    
    print(f"\n  Para el apartado 4, en XFOIL:")
    print(f"  1. Cargar 'perfil_optimizado.dat' con LOAD")
    print(f"  2. OPER → VISC 3000000  (Re = 3·10⁶)")
    print(f"  3. ASEQ αmin αmax Δα")
    print(f"  4. Comparar curvas Cl(α) y Cm(α) viscoso vs. no viscoso")
    
    # ================================================================
    # RESUMEN FINAL
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  RESUMEN DE RESULTADOS")
    print(f"{'='*70}")
    print(f"  Perfil base: {nombre_perfil}")
    print(f"    Cl = {Cl_base:.6f},  Cm_c/4 = {Cm_base:.6f}")
    print(f"  Perfil optimizado: {nombre_nuevo}")
    print(f"    Cl = {Cl_new:.6f},  Cm_c/4 = {Cm_new:.6f}")
    print(f"    ΔCl = {Cl_new - Cl_base:.4f}")
    print(f"    Δ|Cm_c/4| = {abs(Cm_new) - abs(Cm_base):.6f}")
    print(f"  N_opt = {N_opt} paneles")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
