import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit


# ==================================================
# Función objetivo y gradiente
# ==================================================

@njit
def F_logistico(X, y, x, s, lam):
    return np.mean(np.log(1.0 + np.exp(-y * s))) + (lam / 2.0) * np.sum(x ** 2)


@njit
def gradiente_logistico(X, y, x, s, lam):
    n, m = X.shape
    coef = -y / (1.0 + np.exp(y * s))
    return (X @ coef) / m + lam * x


@njit
def constantes_Li(X, lam):
    n, m = X.shape
    Li = np.zeros(n)

    for r in range(n):
        Li[r] = np.sum(X[r, :] ** 2) / (4.0 * m) + lam

    return Li


# ==================================================
# Métodos
# ==================================================

@njit
def metodo_gradiente_logistico(X, y, x0, lam, max_iter, tol, L):
    x = x0.copy()
    s = X.T @ x

    valores_F = np.zeros(max_iter + 1)
    alpha = 1.0 / L

    for k in range(max_iter + 1):

        grad = gradiente_logistico(X, y, x, s, lam)
        norma_grad = np.sqrt(np.sum(grad ** 2))

        valores_F[k] = F_logistico(X, y, x, s, lam)

        if norma_grad < tol:
            return x, k, valores_F[:k+1]

        if k == max_iter:
            break

        x = x - alpha * grad
        s = X.T @ x

    return x, max_iter, valores_F


@njit
def metodo_ciclico_logistico(X, y, x0, lam, max_epocas, tol):
    n, m = X.shape
    x = x0.copy()
    s = X.T @ x

    valores_F = np.zeros(max_epocas + 1)
    Li = constantes_Li(X, lam)

    for epoca in range(max_epocas + 1):

        grad = gradiente_logistico(X, y, x, s, lam)
        norma_grad = np.sqrt(np.sum(grad ** 2))

        valores_F[epoca] = F_logistico(X, y, x, s, lam)

        if norma_grad < tol:
            return x, epoca, valores_F[:epoca+1]

        if epoca == max_epocas:
            break

        for r in range(n):
            grad_r = -np.sum(y * X[r, :] / (1.0 + np.exp(y * s))) / m + lam * x[r]
            delta = -grad_r / Li[r]

            x[r] += delta
            s += delta * X[r, :]

    return x, max_epocas, valores_F


@njit
def metodo_aleatorio_logistico(X, y, x0, lam, max_epocas, tol, beta, seed):
    np.random.seed(seed)

    n, m = X.shape
    x = x0.copy()
    s = X.T @ x

    valores_F = np.zeros(max_epocas + 1)
    Li = constantes_Li(X, lam)

    pesos = Li ** beta
    probs = pesos / np.sum(pesos)

    acumuladas = np.zeros(n)
    acumulado = 0.0

    for r in range(n):
        acumulado += probs[r]
        acumuladas[r] = acumulado

    for epoca in range(max_epocas + 1):

        grad = gradiente_logistico(X, y, x, s, lam)
        norma_grad = np.sqrt(np.sum(grad ** 2))

        valores_F[epoca] = F_logistico(X, y, x, s, lam)

        if norma_grad < tol:
            return x, epoca, valores_F[:epoca+1]

        if epoca == max_epocas:
            break

        for k in range(n):

            u = np.random.random()

            r = 0
            while r < n - 1 and u > acumuladas[r]:
                r += 1

            grad_r = -np.sum(y * X[r, :] / (1.0 + np.exp(y * s))) / m + lam * x[r]
            delta = -grad_r / Li[r]

            x[r] += delta
            s += delta * X[r, :]

    return x, max_epocas, valores_F


@njit
def metodo_greedy_logistico(X, y, x0, lam, max_epocas, tol):
    n, m = X.shape
    x = x0.copy()
    s = X.T @ x

    valores_F = np.zeros(max_epocas + 1)
    Li = constantes_Li(X, lam)

    for epoca in range(max_epocas + 1):

        grad = gradiente_logistico(X, y, x, s, lam)
        norma_grad = np.sqrt(np.sum(grad ** 2))

        valores_F[epoca] = F_logistico(X, y, x, s, lam)

        if norma_grad < tol:
            return x, epoca, valores_F[:epoca+1]

        if epoca == max_epocas:
            break

        for k in range(n):

            grad = gradiente_logistico(X, y, x, s, lam)

            r = 0
            maximo = abs(grad[0])

            for j in range(1, n):
                if abs(grad[j]) > maximo:
                    maximo = abs(grad[j])
                    r = j

            delta = -grad[r] / Li[r]

            x[r] += delta
            s += delta * X[r, :]

    return x, max_epocas, valores_F


# ==================================================
# Generación de matrices
# ==================================================

def generar_etiquetas(X, ruido_etiquetas, seed):
    rng = np.random.default_rng(seed)

    n, m = X.shape
    x_real = rng.standard_normal(n)

    s = X.T @ x_real
    y = np.sign(s)
    y[y == 0] = 1

    num_ruido = int(ruido_etiquetas * m)
    idx = rng.choice(m, size=num_ruido, replace=False)
    y[idx] *= -1

    return y.astype(float)


def matriz_normal(n, m, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    return X


def matriz_correlacionada(n, m, seed):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, m))

    # Factor común para crear dependencia entre variables
    factor_comun = rng.standard_normal(m)
    X = X + factor_comun

    return X


def matriz_escalada(n, m, seed):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, m))

    # Escalas distintas para crear L_i diferentes
    escalas = np.exp(rng.uniform(0.0, np.log(10.0), size=n))
    X = X * escalas[:, None]

    return X


def matriz_dispersa(n, m, densidad, seed):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, m))
    mascara = rng.random((n, m)) < densidad
    X = X * mascara

    return X


def matriz_todas(n, m, densidad, seed):
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, m))

    # Correlación
    factor_comun = rng.standard_normal(m)
    X = X + factor_comun

    # Escalas distintas
    escalas = np.exp(rng.uniform(0.0, np.log(10.0), size=n))
    X = X * escalas[:, None]

    # Dispersión
    mascara = rng.random((n, m)) < densidad
    X = X * mascara

    return X


# ==================================================
# Parámetros
# ==================================================

m = 4250
n = 2250

lam = 0.5
tol = 1e-5

max_iter = 10000
max_epocas = 1000

n_experimentos = 5
densidad = 0.05
ruido_etiquetas = 0.10

betas = [0.0, 0.5, 1.0]

casos = [
    "Normal",
    "Correlacionada",
    "Escalada",
    "Dispersa",
    "Todas"
]

# Zoom distinto para cada matriz
zooms = {
    "Normal": 20,
    "Correlacionada": 40,
    "Escalada": 60,
    "Dispersa": 15,
    "Todas": 20
}


# ==================================================
# Calentamiento Numba
# ==================================================

print("Compilando funciones...")

X = matriz_normal(n, m, seed=0)
y = generar_etiquetas(X, ruido_etiquetas, seed=100)
x0 = np.zeros(n)

L = (np.linalg.norm(X, 2) ** 2) / (4.0 * m) + lam

_ = metodo_gradiente_logistico(X, y, x0, lam, 1, tol, L)
_ = metodo_ciclico_logistico(X, y, x0, lam, 1, tol)
_ = metodo_aleatorio_logistico(X, y, x0, lam, 1, tol, 0.0, 1)
_ = metodo_greedy_logistico(X, y, x0, lam, 1, tol)

print("Compilación terminada.\n")


# ==================================================
# Experimentos
# ==================================================

for caso in casos:

    print("=" * 80)
    print(f"CASO: {caso}")
    print("=" * 80)

    metodos = [
        "Gradiente",
        "Cíclico",
        "Aleatorio beta=0.0",
        "Aleatorio beta=0.5",
        "Aleatorio beta=1.0",
        "Greedy"
    ]

    epocas = {metodo: [] for metodo in metodos}
    tiempos = {metodo: [] for metodo in metodos}
    valores_finales = {metodo: [] for metodo in metodos}
    curvas_F = {metodo: [] for metodo in metodos}

    for exp in range(n_experimentos):

        print(f"Experimento {exp + 1}/{n_experimentos}")

        seed = exp + 1

        if caso == "Normal":
            X = matriz_normal(n, m, seed)

        elif caso == "Correlacionada":
            X = matriz_correlacionada(n, m, seed)

        elif caso == "Escalada":
            X = matriz_escalada(n, m, seed)

        elif caso == "Dispersa":
            X = matriz_dispersa(n, m, densidad, seed)

        elif caso == "Todas":
            X = matriz_todas(n, m, densidad, seed)

        y = generar_etiquetas(X, ruido_etiquetas, seed + 1000)
        x0 = np.zeros(n)

        L = (np.linalg.norm(X, 2) ** 2) / (4.0 * m) + lam

        # --------------------------------------------------
        # Gradiente completo
        # --------------------------------------------------

        print("  Gradiente...")
        inicio = time.time()

        x, it, F = metodo_gradiente_logistico(
            X, y, x0, lam, max_iter, tol, L
        )

        tiempo = time.time() - inicio

        epocas["Gradiente"].append(it)
        tiempos["Gradiente"].append(tiempo)
        valores_finales["Gradiente"].append(F[-1])
        curvas_F["Gradiente"].append(F)

        print(f"    {it} etapas, {tiempo:.4f}s, F = {F[-1]:.6e}")

        # --------------------------------------------------
        # Cíclico
        # --------------------------------------------------

        print("  Cíclico...")
        inicio = time.time()

        x, it, F = metodo_ciclico_logistico(
            X, y, x0, lam, max_epocas, tol
        )

        tiempo = time.time() - inicio

        epocas["Cíclico"].append(it)
        tiempos["Cíclico"].append(tiempo)
        valores_finales["Cíclico"].append(F[-1])
        curvas_F["Cíclico"].append(F)

        print(f"    {it} épocas, {tiempo:.4f}s, F = {F[-1]:.6e}")

        # --------------------------------------------------
        # Aleatorio con distintos beta
        # --------------------------------------------------

        for beta in betas:

            nombre = f"Aleatorio beta={beta}"

            print(f"  {nombre}...")
            inicio = time.time()

            x, it, F = metodo_aleatorio_logistico(
                X, y, x0, lam, max_epocas, tol, beta, seed
            )

            tiempo = time.time() - inicio

            epocas[nombre].append(it)
            tiempos[nombre].append(tiempo)
            valores_finales[nombre].append(F[-1])
            curvas_F[nombre].append(F)

            print(f"    {it} épocas, {tiempo:.4f}s, F = {F[-1]:.6e}")

        # --------------------------------------------------
        # Greedy
        # --------------------------------------------------

        print("  Greedy...")
        inicio = time.time()

        x, it, F = metodo_greedy_logistico(
            X, y, x0, lam, max_epocas, tol
        )

        tiempo = time.time() - inicio

        epocas["Greedy"].append(it)
        tiempos["Greedy"].append(tiempo)
        valores_finales["Greedy"].append(F[-1])
        curvas_F["Greedy"].append(F)

        print(f"    {it} épocas, {tiempo:.4f}s, F = {F[-1]:.6e}")

        print("")

    # ==================================================
    # Tabla media del caso
    # ==================================================

    print(f"Resultados medios - {caso}")
    print(f"{'Método':<22} {'Épocas':<12} {'Tiempo':<15} {'F(x_final)':<18}")
    print("-" * 75)

    for metodo in metodos:
        print(
            f"{metodo:<22} "
            f"{np.mean(epocas[metodo]):<12.2f} "
            f"{np.mean(tiempos[metodo]):<15.6f} "
            f"{np.mean(valores_finales[metodo]):<18.6e}"
        )

    print("")

    # ==================================================
    # Gráfica media del caso: F(x_k)/F(x_0)
    # Sin sombreado de desviación típica
    # ==================================================

    plt.figure(figsize=(10, 6), dpi=150)

    zoom_epocas = zooms[caso]

    medias_guardadas = []

    for metodo in metodos:

        max_len = max(len(c) for c in curvas_F[metodo])

        # Matriz de curvas normalizadas:
        # cada curva se representa como F(x_k) / F(x_0).
        # Si una curva termina antes, se prolonga con su último valor
        # para que la media siempre use los 5 experimentos.
        matriz = np.zeros((n_experimentos, max_len))

        for i in range(n_experimentos):
            curva = curvas_F[metodo][i]
            curva_rel = curva / curva[0]

            matriz[i, :len(curva_rel)] = curva_rel
            matriz[i, len(curva_rel):] = curva_rel[-1]

        media = np.mean(matriz, axis=0)
        medias_guardadas.append(media)

        eje_x = np.arange(len(media))

        plt.plot(
            eje_x,
            media,
            label=metodo,
            linewidth=2.4
        )

    # Zoom horizontal
    plt.xlim(0, zoom_epocas)

    # Zoom vertical automático usando solo la zona visible
    y_min = np.inf
    y_max = -np.inf

    for media in medias_guardadas:

        limite = min(zoom_epocas + 1, len(media))

        y_min = min(y_min, np.min(media[:limite]))
        y_max = max(y_max, np.max(media[:limite]))

    margen = 0.06 * (y_max - y_min)

    if margen == 0:
        margen = 0.01

    plt.ylim(y_min - margen, y_max + margen)

    plt.xlabel("Épocas", fontsize=16)
    plt.ylabel(r"$F(x_k)/F(x_0)$", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="upper right",
        fontsize=12,
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(f"grafica_{caso}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"grafica_{caso}.pdf", bbox_inches="tight")

    plt.show()
    plt.close()


    # ==================================================
    # Gráfica F(x_k) - F_ref
    # Solo métodos por coordenadas y eje horizontal en iteraciones
    # ==================================================

    metodos_coordenadas = [metodo for metodo in metodos if metodo != "Gradiente"]

    # Para cada repetición se toma como F_ref el menor valor final obtenido
    # entre todos los métodos en esa misma repetición.
    F_refs = np.zeros(n_experimentos)

    for i in range(n_experimentos):
        F_refs[i] = min(curvas_F[metodo][i][-1] for metodo in metodos)

    plt.figure(figsize=(10, 6), dpi=150)

    for metodo in metodos_coordenadas:

        max_len = max(len(c) for c in curvas_F[metodo])

        matriz = np.zeros((n_experimentos, max_len))

        for i in range(n_experimentos):
            curva = curvas_F[metodo][i] - F_refs[i]
            curva = np.maximum(curva, 1e-16)

            matriz[i, :len(curva)] = curva
            matriz[i, len(curva):] = curva[-1]

        media = np.mean(matriz, axis=0)

        # En métodos por coordenadas, una época equivale a n actualizaciones.
        iteraciones = np.arange(len(media)) * n

        plt.semilogy(
            iteraciones,
            media + 1e-16,
            label=metodo,
            linewidth=2.4
        )

    plt.xlim(0, zoom_epocas * n)

    plt.xlabel("Iteraciones", fontsize=16)
    plt.ylabel(r"$F(x_k)-F_{\mathrm{ref}}$", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="upper right",
        fontsize=12,
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(f"grafica_FmenosFref_{caso}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"grafica_FmenosFref_{caso}.pdf", bbox_inches="tight")

    plt.show()
    plt.close()
