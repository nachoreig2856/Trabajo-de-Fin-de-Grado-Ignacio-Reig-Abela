import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------
# Error relativo para el método del gradiente
# --------------------------------------------------
def error_relativo(x, x_opt):
    return np.linalg.norm(x - x_opt) / np.linalg.norm(x_opt)


# --------------------------------------------------
# Función objetivo
# --------------------------------------------------
def F(A, b, x):
    return 0.5 * np.linalg.norm(A @ x - b)**2


# --------------------------------------------------
# Método del gradiente
# No lo compilamos con Numba porque ya usa productos matriciales de NumPy
# --------------------------------------------------
def gradiente_minimos(A, b, x0, x_opt, n_epocas, tol):
    xk = x0.copy().astype(float)

    errores = []
    valores_F = []

    L = np.linalg.norm(A, 2)**2

    r = A @ xk - b
    errores.append(error_relativo(xk, x_opt))
    valores_F.append(0.5 * np.linalg.norm(r)**2)

    for epoca in range(n_epocas):

        if errores[-1] < tol:
            break

        grad_k = A.T @ r

        xk = xk - (1 / L) * grad_k

        r = A @ xk - b

        errores.append(error_relativo(xk, x_opt))
        valores_F.append(0.5 * np.linalg.norm(r)**2)

    return np.array(errores), np.array(valores_F)


# --------------------------------------------------
# Funciones auxiliares compiladas con Numba
# --------------------------------------------------
@njit
def error_relativo_numba(x, x_opt):
    suma = 0.0
    suma_opt = 0.0

    for i in range(x.shape[0]):
        diff = x[i] - x_opt[i]
        suma += diff * diff
        suma_opt += x_opt[i] * x_opt[i]

    return np.sqrt(suma) / np.sqrt(suma_opt)


@njit
def funcion_objetivo_residual_numba(r):
    suma = 0.0

    for i in range(r.shape[0]):
        suma += r[i] * r[i]

    return 0.5 * suma


@njit
def calcular_L_columnas_numba(A):
    p = A.shape[0]
    n = A.shape[1]

    L = np.empty(n)

    for j in range(n):
        suma = 0.0
        for i in range(p):
            suma += A[i, j] * A[i, j]
        L[j] = suma

    return L


@njit
def calcular_residual_inicial_numba(A, b, x):
    p = A.shape[0]
    n = A.shape[1]

    r = np.empty(p)

    for i in range(p):
        suma = 0.0
        for j in range(n):
            suma += A[i, j] * x[j]
        r[i] = suma - b[i]

    return r


# --------------------------------------------------
# Método cíclico por épocas con Numba
# --------------------------------------------------
@njit
def ciclico_minimos_numba(A, b, x0, x_opt, n_epocas, tol):
    p = A.shape[0]
    n = A.shape[1]

    xk = x0.copy()

    errores = np.empty(n_epocas + 1)
    valores_F = np.empty(n_epocas + 1)

    L = calcular_L_columnas_numba(A)
    r = calcular_residual_inicial_numba(A, b, xk)

    contador = 0
    j = 0

    errores[contador] = error_relativo_numba(xk, x_opt)
    valores_F[contador] = funcion_objetivo_residual_numba(r)
    contador += 1

    for epoca in range(n_epocas):

        if errores[contador - 1] < tol:
            break

        # Una época = n actualizaciones coordenadas
        for _ in range(n):

            if L[j] <= 1e-14:
                j = (j + 1) % n
                continue

            # grad_j = A[:, j]^T r
            grad_j = 0.0
            for i in range(p):
                grad_j += A[i, j] * r[i]

            paso = grad_j / L[j]

            # x_j = x_j - paso
            xk[j] -= paso

            # r = r - paso * A[:, j]
            for i in range(p):
                r[i] -= paso * A[i, j]

            j = (j + 1) % n

        errores[contador] = error_relativo_numba(xk, x_opt)
        valores_F[contador] = funcion_objetivo_residual_numba(r)
        contador += 1

    return errores[:contador], valores_F[:contador]


# --------------------------------------------------
# Método greedy por épocas con Numba
# --------------------------------------------------
@njit
def greedy_minimos_numba(A, b, x0, x_opt, G, L, n_epocas, tol):
    p = A.shape[0]
    n = A.shape[1]

    xk = x0.copy()

    errores = np.empty(n_epocas + 1)
    valores_F = np.empty(n_epocas + 1)

    r = calcular_residual_inicial_numba(A, b, xk)

    # Gradiente inicial: grad = A^T r
    grad_k = np.empty(n)

    for j in range(n):
        suma = 0.0
        for i in range(p):
            suma += A[i, j] * r[i]
        grad_k[j] = suma

    contador = 0

    errores[contador] = error_relativo_numba(xk, x_opt)
    valores_F[contador] = funcion_objetivo_residual_numba(r)
    contador += 1

    for epoca in range(n_epocas):

        if errores[contador - 1] < tol:
            break

        # Una época = n actualizaciones greedy
        for _ in range(n):

            # Coordenada greedy: mayor valor absoluto del gradiente
            j_max = 0
            maximo = abs(grad_k[0])

            for j in range(1, n):
                valor = abs(grad_k[j])
                if valor > maximo:
                    maximo = valor
                    j_max = j

            if L[j_max] <= 1e-14:
                grad_k[j_max] = 0.0
                continue

            grad_j = grad_k[j_max]
            paso = grad_j / L[j_max]

            # Actualización de la coordenada
            xk[j_max] -= paso

            # Actualización del residual
            for i in range(p):
                r[i] -= paso * A[i, j_max]

            # Actualización eficiente del gradiente:
            # grad = grad - paso * G[:, j_max]
            for j in range(n):
                grad_k[j] -= paso * G[j, j_max]

        errores[contador] = error_relativo_numba(xk, x_opt)
        valores_F[contador] = funcion_objetivo_residual_numba(r)
        contador += 1

    return errores[:contador], valores_F[:contador]


# --------------------------------------------------
# Método aleatorio por épocas con Numba
# --------------------------------------------------
@njit
def aleatorio_minimos_numba(A, b, x0, x_opt, n_epocas, tol, beta, seed):
    p = A.shape[0]
    n = A.shape[1]

    np.random.seed(seed)

    xk = x0.copy()

    errores = np.empty(n_epocas + 1)
    valores_F = np.empty(n_epocas + 1)

    L = calcular_L_columnas_numba(A)

    # Probabilidades proporcionales a L_j^beta
    pesos = np.empty(n)
    suma_pesos = 0.0

    for j in range(n):
        if L[j] > 1e-14:
            pesos[j] = L[j] ** beta
        else:
            pesos[j] = 0.0

        suma_pesos += pesos[j]

    probabilidades_acumuladas = np.empty(n)
    acumulada = 0.0

    for j in range(n):
        acumulada += pesos[j] / suma_pesos
        probabilidades_acumuladas[j] = acumulada

    r = calcular_residual_inicial_numba(A, b, xk)

    contador = 0

    errores[contador] = error_relativo_numba(xk, x_opt)
    valores_F[contador] = funcion_objetivo_residual_numba(r)
    contador += 1

    for epoca in range(n_epocas):

        if errores[contador - 1] < tol:
            break

        # Una época = n actualizaciones aleatorias
        for _ in range(n):

            u = np.random.random()

            j = n - 1
            for jj in range(n):
                if u <= probabilidades_acumuladas[jj] and L[jj] > 1e-14:
                    j = jj
                    break

            # grad_j = A[:, j]^T r
            grad_j = 0.0
            for i in range(p):
                grad_j += A[i, j] * r[i]

            paso = grad_j / L[j]

            # Actualización de la coordenada
            xk[j] -= paso

            # Actualización del residual
            for i in range(p):
                r[i] -= paso * A[i, j]

        errores[contador] = error_relativo_numba(xk, x_opt)
        valores_F[contador] = funcion_objetivo_residual_numba(r)
        contador += 1

    return errores[:contador], valores_F[:contador]


# --------------------------------------------------
# Métricas finales
# --------------------------------------------------
def calcular_metricas(nombre, errores, valores_F, tiempo):
    return {
        "Método": nombre,
        "Épocas": len(errores) - 1,
        "Tiempo": tiempo,
        "F(x_final)": valores_F[-1],
        "Residual": np.sqrt(2 * valores_F[-1]),
        "Error relativo": errores[-1]
    }




# --------------------------------------------------
# Configuracion del experimento
# --------------------------------------------------
n_repeticiones = 5
p = 200
n = 150
n_epocas_gradiente = 2_000_000
n_epocas_coord = 1_000_000
tol = 1e-4
betas = [0.0, 0.5, 1.0]
seed_base_problema = 1
seed_base_aleatorio = 123
metodos = ["Gradiente", "Ciclico", "Greedy", "Aleatorio beta=0", "Aleatorio beta=0.5", "Aleatorio beta=1"]

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13
})


# --------------------------------------------------
# Generacion del problema
# --------------------------------------------------
def generar_problema(seed):
    np.random.seed(seed)
    v = np.random.randn(p)
    Z = np.random.randn(p, n)
    eps = 0.14
    A = v[:, None] + eps * Z
    A = np.asfortranarray(A)
    x_opt = np.random.randn(n).astype(np.float64)
    b = A @ x_opt
    b = b.astype(np.float64)
    x0 = np.zeros(n, dtype=np.float64)
    return A, b, x0, x_opt


# --------------------------------------------------
# Calentamiento de Numba
# --------------------------------------------------
print("Compilando funciones con Numba...")
A_w = np.asfortranarray(np.random.randn(5, 3))
x_opt_w = np.random.randn(3).astype(np.float64)
b_w = (A_w @ x_opt_w).astype(np.float64)
x0_w = np.zeros(3, dtype=np.float64)
_ = ciclico_minimos_numba(A_w, b_w, x0_w, x_opt_w, 1, tol)
G_w = A_w.T @ A_w
L_w = np.diag(G_w).copy()
_ = greedy_minimos_numba(A_w, b_w, x0_w, x_opt_w, G_w, L_w, 1, tol)
_ = aleatorio_minimos_numba(A_w, b_w, x0_w, x_opt_w, 1, tol, 0.0, seed_base_aleatorio)
print("Compilacion terminada.\n")


# --------------------------------------------------
# Ejecucion de las repeticiones
# --------------------------------------------------
resultados_todos = {metodo: [] for metodo in metodos}
curvas_F_todas = {metodo: [] for metodo in metodos}
curvas_iter = {}

for rep in range(n_repeticiones):
    print(f"Repeticion {rep + 1}/{n_repeticiones}")
    A, b, x0, x_opt = generar_problema(seed_base_problema + rep)

    # Gradiente
    t0 = time.perf_counter()
    err, vals = gradiente_minimos(A, b, x0, x_opt, n_epocas_gradiente, tol)
    tiempo = time.perf_counter() - t0
    resultados_todos["Gradiente"].append(calcular_metricas("Gradiente", err, vals, tiempo))
    curvas_F_todas["Gradiente"].append(vals)

    # Ciclico
    t0 = time.perf_counter()
    err, vals = ciclico_minimos_numba(A, b, x0, x_opt, n_epocas_coord, tol)
    tiempo = time.perf_counter() - t0
    resultados_todos["Ciclico"].append(calcular_metricas("Ciclico", err, vals, tiempo))
    curvas_F_todas["Ciclico"].append(vals)
    if rep == 0:
        curvas_iter["Ciclico"] = (np.arange(len(vals)) * n, vals)

    # Greedy
    t0 = time.perf_counter()
    G = A.T @ A
    L_greedy = np.diag(G).copy()
    err, vals = greedy_minimos_numba(A, b, x0, x_opt, G, L_greedy, n_epocas_coord, tol)
    tiempo = time.perf_counter() - t0
    resultados_todos["Greedy"].append(calcular_metricas("Greedy", err, vals, tiempo))
    curvas_F_todas["Greedy"].append(vals)
    if rep == 0:
        curvas_iter["Greedy"] = (np.arange(len(vals)) * n, vals)

    # Aleatorio para beta = 0, 0.5, 1
    for beta in betas:
        nombre = f"Aleatorio beta={beta:g}"
        t0 = time.perf_counter()
        err, vals = aleatorio_minimos_numba(
            A, b, x0, x_opt, n_epocas_coord, tol, beta, seed_base_aleatorio + rep
        )
        tiempo = time.perf_counter() - t0
        resultados_todos[nombre].append(calcular_metricas(nombre, err, vals, tiempo))
        curvas_F_todas[nombre].append(vals)
        if rep == 0:
            curvas_iter[nombre] = (np.arange(len(vals)) * n, vals)


# --------------------------------------------------
# Tabla de resultados medios
# --------------------------------------------------
print("\nResultados medios en", n_repeticiones, "repeticiones\n")
print(f"{'Metodo':<20} {'Epocas':<12} {'Tiempo':<12} {'F(x_final)':<15} {'Error relativo':<15}")
print("-" * 80)

for metodo in metodos:
    epocas = np.mean([r["Épocas"] for r in resultados_todos[metodo]])
    tiempo = np.mean([r["Tiempo"] for r in resultados_todos[metodo]])
    f_final = np.mean([r["F(x_final)"] for r in resultados_todos[metodo]])
    error = np.mean([r["Error relativo"] for r in resultados_todos[metodo]])
    print(f"{metodo:<20} {epocas:<12.2f} {tiempo:<12.6f} {f_final:<15.6e} {error:<15.6e}")


# --------------------------------------------------
# Grafica 1: evolucion media de F por epocas
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for metodo in metodos:
    longitud = min(len(c) for c in curvas_F_todas[metodo])
    matriz = np.vstack([c[:longitud] for c in curvas_F_todas[metodo]])
    media = np.mean(matriz, axis=0)
    plt.semilogy(media + 1e-16, label=metodo)

plt.xlabel("Epoca")
plt.ylabel(r"$F(x_k)$")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Grafica 2: F(x_k)-F(x_opt) por iteraciones
# Se usa la primera repeticion y puntos al final de cada epoca
# --------------------------------------------------
def formato_eje_x(x, pos):
    if abs(x) < 1:
        return "0"
    mantisa, exponente = f"{x:.0e}".split("e")
    return f"{mantisa}e{int(exponente)}"

plt.figure(figsize=(9, 6))

for metodo, (it, vals) in curvas_iter.items():
    plt.semilogy(it, vals + 1e-16, label=metodo)

plt.xlabel("Iteracion")
plt.ylabel(r"$F(x_k)-F(x_{\mathrm{opt}})$")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(FuncFormatter(formato_eje_x))
plt.tight_layout()
plt.show()
