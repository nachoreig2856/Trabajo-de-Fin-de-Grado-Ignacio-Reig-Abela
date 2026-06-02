import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numba import njit


# ==================================================
# PARÁMETROS DEL EXPERIMENTO
# ==================================================

N = 350

# Si quieres usar una imagen externa, pon aquí la ruta:
# IMAGE_PATH = "mi_imagen.png"
IMAGE_PATH = "je.jpg"
alpha = 0.6
radio = 1
porcentaje_eliminado = 0.30

lambda_reg = 5e-3
mu = 1e-6

tol = 1e-5
max_epocas_gradiente = 500
max_epocas_coord = 250

seed = 123


# ==================================================
# TAMAÑOS DE LETRA PARA FIGURAS
# ==================================================

TAM_TITULO_IMAGEN = 26
TAM_TITULO_GRAFICA = 24
TAM_EJES = 22
TAM_TICKS = 18
TAM_LEYENDA = 18
GROSOR_LINEA = 2.8


# ==================================================
# IMAGEN DE PRUEBA / CARGA DE IMAGEN
# ==================================================

def imagen_prueba(N):
    """
    Imagen RGB sencilla por si no quieres cargar una imagen externa.
    """
    X = np.zeros((N, N, 3), dtype=np.float64)

    # Fondo con degradado de color
    for i in range(N):
        for j in range(N):
            X[i, j, 0] = i / (N - 1)
            X[i, j, 1] = j / (N - 1)
            X[i, j, 2] = 0.5

    # Cuadrado rojo grande
    a = N // 4
    b = 3 * N // 4
    X[a:b, a:b, 0] = 1.0
    X[a:b, a:b, 1] = 0.1
    X[a:b, a:b, 2] = 0.1

    # Cuadrado blanco centrado
    lado_blanco = N // 4
    ini_blanco = (N - lado_blanco) // 2
    fin_blanco = ini_blanco + lado_blanco

    X[ini_blanco:fin_blanco, ini_blanco:fin_blanco, 0] = 1.0
    X[ini_blanco:fin_blanco, ini_blanco:fin_blanco, 1] = 1.0
    X[ini_blanco:fin_blanco, ini_blanco:fin_blanco, 2] = 1.0

    # Cuadrado azul centrado dentro del blanco
    lado_azul = N // 8
    ini_azul = (N - lado_azul) // 2
    fin_azul = ini_azul + lado_azul

    X[ini_azul:fin_azul, ini_azul:fin_azul, 0] = 0.0
    X[ini_azul:fin_azul, ini_azul:fin_azul, 1] = 0.2
    X[ini_azul:fin_azul, ini_azul:fin_azul, 2] = 1.0

    return X


def cargar_imagen_rgb(path, N):
    """
    Carga una imagen desde disco, la convierte a RGB y la redimensiona a N x N.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((N, N), Image.Resampling.LANCZOS)
    X = np.asarray(img, dtype=np.float64) / 255.0
    return X


# ==================================================
# CONSTRUCCIÓN DE MATRICES DISPERSAS
# ==================================================

def indice(i, j, N):
    return i * N + j


def construir_matriz_difuminado(N, radio=1, alpha=0.8):
    """
    Construye A = (1-alpha)I + alpha B.
    B sustituye cada píxel por el promedio local de una ventana de radio dado.
    """
    n = N * N

    rows = []
    cols = []
    data = []

    for i in range(N):
        for j in range(N):
            p = indice(i, j, N)

            vecinos = []
            for di in range(-radio, radio + 1):
                for dj in range(-radio, radio + 1):
                    ii = i + di
                    jj = j + dj

                    if 0 <= ii < N and 0 <= jj < N:
                        vecinos.append(indice(ii, jj, N))

            peso = 1.0 / len(vecinos)

            for q in vecinos:
                rows.append(p)
                cols.append(q)
                data.append(peso)

    B = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    I = sparse.eye(n, format="csr")

    A = (1.0 - alpha) * I + alpha * B
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()

    return A


def construir_matriz_diferencias(N):
    """
    Construye la matriz D de diferencias finitas horizontales y verticales.
    """
    n = N * N

    rows = []
    cols = []
    data = []
    fila = 0

    # Diferencias horizontales
    for i in range(N):
        for j in range(N - 1):
            p = indice(i, j, N)
            q = indice(i, j + 1, N)

            rows.extend([fila, fila])
            cols.extend([q, p])
            data.extend([1.0, -1.0])
            fila += 1

    # Diferencias verticales
    for i in range(N - 1):
        for j in range(N):
            p = indice(i, j, N)
            q = indice(i + 1, j, N)

            rows.extend([fila, fila])
            cols.extend([q, p])
            data.extend([1.0, -1.0])
            fila += 1

    D = sparse.csr_matrix((data, (rows, cols)), shape=(fila, n))
    D.sum_duplicates()
    D.eliminate_zeros()

    return D


# ==================================================
# FUNCIÓN OBJETIVO
# ==================================================

def funcion_objetivo(x, A, D, mask, b, lambda_reg, mu):
    """
    F(x) = 1/2 ||M A x - b||^2 + lambda/2 ||D x||^2 + mu/2 ||x||^2.
    """
    r_datos = mask * (A @ x) - b
    r_suavidad = D @ x

    return (
        0.5 * np.dot(r_datos, r_datos)
        + 0.5 * lambda_reg * np.dot(r_suavidad, r_suavidad)
        + 0.5 * mu * np.dot(x, x)
    )


def estimar_L_global(H):
    """
    Estima L = lambda_max(H), constante Lipschitz global.
    """
    try:
        L = eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
        return float(1.05 * L)
    except Exception:
        rng = np.random.default_rng(0)
        n = H.shape[0]

        v = rng.normal(size=n)
        v /= np.linalg.norm(v)

        for _ in range(100):
            w = H @ v
            nw = np.linalg.norm(w)

            if nw == 0:
                break

            v = w / nw

        L = float(v @ (H @ v))
        return 1.05 * L


def guardar_rgb(nombre, img):
    img = np.clip(img, 0.0, 1.0)
    Image.fromarray((255 * img).astype(np.uint8)).save(nombre)


# ==================================================
# FUNCIONES NUMBA PARA MÉTODOS POR COORDENADAS
# ==================================================

@njit
def norma2_numba(v):
    s = 0.0
    for i in range(v.size):
        s += v[i] * v[i]
    return np.sqrt(s)


@njit
def actualizar_gradiente_columna(i, delta, g, indptr, indices, data):
    ini = indptr[i]
    fin = indptr[i + 1]

    for k in range(ini, fin):
        fila = indices[k]
        valor = data[k]
        g[fila] += delta * valor


@njit
def coord_ciclico_numba(c, Li, indptr, indices, data, tol, max_epocas):
    n = c.size

    x = np.zeros(n)
    g = -c.copy()

    g0_norm = norma2_numba(g)

    grad_hist = np.empty(max_epocas + 1)
    grad_hist[0] = 1.0

    x_hist = np.empty((max_epocas + 1, n))
    x_hist[0, :] = x

    epocas_realizadas = 0

    for epoca in range(1, max_epocas + 1):

        for i in range(n):
            delta = -g[i] / Li[i]

            if delta != 0.0:
                x[i] += delta
                actualizar_gradiente_columna(i, delta, g, indptr, indices, data)

        grad_rel = norma2_numba(g) / g0_norm

        grad_hist[epoca] = grad_rel
        x_hist[epoca, :] = x
        epocas_realizadas = epoca

        if grad_rel < tol:
            break

    return x, g, grad_hist[:epocas_realizadas + 1], x_hist[:epocas_realizadas + 1, :], epocas_realizadas


@njit
def coord_aleatorio_numba(c, Li, indptr, indices, data, coords, tol, max_epocas):
    n = c.size

    x = np.zeros(n)
    g = -c.copy()

    g0_norm = norma2_numba(g)

    grad_hist = np.empty(max_epocas + 1)
    grad_hist[0] = 1.0

    x_hist = np.empty((max_epocas + 1, n))
    x_hist[0, :] = x

    epocas_realizadas = 0

    for epoca in range(1, max_epocas + 1):

        for k in range(n):
            i = coords[epoca - 1, k]

            delta = -g[i] / Li[i]

            if delta != 0.0:
                x[i] += delta
                actualizar_gradiente_columna(i, delta, g, indptr, indices, data)

        grad_rel = norma2_numba(g) / g0_norm

        grad_hist[epoca] = grad_rel
        x_hist[epoca, :] = x
        epocas_realizadas = epoca

        if grad_rel < tol:
            break

    return x, g, grad_hist[:epocas_realizadas + 1], x_hist[:epocas_realizadas + 1, :], epocas_realizadas


@njit
def coord_greedy_numba(c, Li, indptr, indices, data, tol, max_epocas):
    n = c.size

    x = np.zeros(n)
    g = -c.copy()

    g0_norm = norma2_numba(g)

    grad_hist = np.empty(max_epocas + 1)
    grad_hist[0] = 1.0

    x_hist = np.empty((max_epocas + 1, n))
    x_hist[0, :] = x

    epocas_realizadas = 0

    for epoca in range(1, max_epocas + 1):

        for _ in range(n):

            imax = 0
            maxval = abs(g[0])

            for i in range(1, n):
                val = abs(g[i])
                if val > maxval:
                    maxval = val
                    imax = i

            delta = -g[imax] / Li[imax]

            if delta != 0.0:
                x[imax] += delta
                actualizar_gradiente_columna(imax, delta, g, indptr, indices, data)

        grad_rel = norma2_numba(g) / g0_norm

        grad_hist[epoca] = grad_rel
        x_hist[epoca, :] = x
        epocas_realizadas = epoca

        if grad_rel < tol:
            break

    return x, g, grad_hist[:epocas_realizadas + 1], x_hist[:epocas_realizadas + 1, :], epocas_realizadas


# ==================================================
# MÉTODOS
# ==================================================

def calcular_F_hist(x_hist, A, D, mask, b, lambda_reg, mu):
    F_hist = np.empty(x_hist.shape[0])

    for k in range(x_hist.shape[0]):
        F_hist[k] = funcion_objetivo(x_hist[k], A, D, mask, b, lambda_reg, mu)

    return F_hist


def metodo_gradiente(H, c, A, D, mask, b, lambda_reg, mu, L, tol, max_epocas):
    n = H.shape[0]

    x = np.zeros(n)
    g = H @ x - c
    g0_norm = np.linalg.norm(g)

    grad_hist = [1.0]
    x_hist = [x.copy()]

    paso = 1.0 / L

    t0 = time.perf_counter()

    for _ in range(max_epocas):

        x = x - paso * g
        g = H @ x - c

        grad_rel = np.linalg.norm(g) / g0_norm

        grad_hist.append(grad_rel)
        x_hist.append(x.copy())

        if grad_rel < tol:
            break

    tiempo = time.perf_counter() - t0

    x_hist = np.array(x_hist)
    grad_hist = np.array(grad_hist)
    F_hist = calcular_F_hist(x_hist, A, D, mask, b, lambda_reg, mu)

    return {
        "x": x,
        "epocas": len(grad_hist) - 1,
        "tiempo": tiempo,
        "F_final": F_hist[-1],
        "grad_rel": grad_hist[-1],
        "F_hist": F_hist,
        "grad_hist": grad_hist,
    }


def metodo_ciclico(H_csc, c, Li, A, D, mask, b, lambda_reg, mu, tol, max_epocas):
    indptr = H_csc.indptr.astype(np.int64)
    indices = H_csc.indices.astype(np.int64)
    data = H_csc.data.astype(np.float64)

    t0 = time.perf_counter()

    x, g, grad_hist, x_hist, epocas = coord_ciclico_numba(
        c.astype(np.float64),
        Li.astype(np.float64),
        indptr,
        indices,
        data,
        tol,
        max_epocas
    )

    tiempo = time.perf_counter() - t0
    F_hist = calcular_F_hist(x_hist, A, D, mask, b, lambda_reg, mu)

    return {
        "x": x,
        "epocas": epocas,
        "tiempo": tiempo,
        "F_final": F_hist[-1],
        "grad_rel": grad_hist[-1],
        "F_hist": F_hist,
        "grad_hist": grad_hist,
    }


def metodo_aleatorio(H_csc, c, Li, A, D, mask, b, lambda_reg, mu, tol, max_epocas, seed=0):
    n = c.size
    rng = np.random.default_rng(seed)

    coords = rng.integers(0, n, size=(max_epocas, n), dtype=np.int64)

    indptr = H_csc.indptr.astype(np.int64)
    indices = H_csc.indices.astype(np.int64)
    data = H_csc.data.astype(np.float64)

    t0 = time.perf_counter()

    x, g, grad_hist, x_hist, epocas = coord_aleatorio_numba(
        c.astype(np.float64),
        Li.astype(np.float64),
        indptr,
        indices,
        data,
        coords,
        tol,
        max_epocas
    )

    tiempo = time.perf_counter() - t0
    F_hist = calcular_F_hist(x_hist, A, D, mask, b, lambda_reg, mu)

    return {
        "x": x,
        "epocas": epocas,
        "tiempo": tiempo,
        "F_final": F_hist[-1],
        "grad_rel": grad_hist[-1],
        "F_hist": F_hist,
        "grad_hist": grad_hist,
    }


def metodo_greedy(H_csc, c, Li, A, D, mask, b, lambda_reg, mu, tol, max_epocas):
    indptr = H_csc.indptr.astype(np.int64)
    indices = H_csc.indices.astype(np.int64)
    data = H_csc.data.astype(np.float64)

    t0 = time.perf_counter()

    x, g, grad_hist, x_hist, epocas = coord_greedy_numba(
        c.astype(np.float64),
        Li.astype(np.float64),
        indptr,
        indices,
        data,
        tol,
        max_epocas
    )

    tiempo = time.perf_counter() - t0
    F_hist = calcular_F_hist(x_hist, A, D, mask, b, lambda_reg, mu)

    return {
        "x": x,
        "epocas": epocas,
        "tiempo": tiempo,
        "F_final": F_hist[-1],
        "grad_rel": grad_hist[-1],
        "F_hist": F_hist,
        "grad_hist": grad_hist,
    }


# ==================================================
# RESOLUCIÓN DE UN CANAL
# ==================================================

def resolver_canal(nombre_canal, x_true, A, D, mask, H, Li, L_global, tol, seed):
    print(f"\nCanal {nombre_canal}")
    print("-" * 80)

    b = mask * (A @ x_true)
    c = A.T @ b

    resultados = {}

    resultados["Gradiente"] = metodo_gradiente(
        H, c, A, D, mask, b, lambda_reg, mu,
        L_global, tol, max_epocas_gradiente
    )

    resultados["Cíclico"] = metodo_ciclico(
        H, c, Li, A, D, mask, b, lambda_reg, mu,
        tol, max_epocas_coord
    )

    resultados["Greedy"] = metodo_greedy(
        H, c, Li, A, D, mask, b, lambda_reg, mu,
        tol, max_epocas_coord
    )

    resultados["Aleatorio"] = metodo_aleatorio(
        H, c, Li, A, D, mask, b, lambda_reg, mu,
        tol, max_epocas_coord,
        seed=seed
    )

    print(f"{'Método':<12}{'Épocas':<12}{'Tiempo (s)':<15}{'F(x_final)':<18}{'Grad. relativo':<18}")
    print("-" * 80)

    for metodo, res in resultados.items():
        print(
            f"{metodo:<12}"
            f"{res['epocas']:<12}"
            f"{res['tiempo']:<15.6f}"
            f"{res['F_final']:<18.6e}"
            f"{res['grad_rel']:<18.6e}"
        )

    return resultados, b


# ==================================================
# EJECUCIÓN COMPLETA RGB
# ==================================================

rng = np.random.default_rng(seed)

if IMAGE_PATH is None:
    img_true = imagen_prueba(N)
else:
    img_true = cargar_imagen_rgb(IMAGE_PATH, N)

n = N * N

print(f"Imagen RGB: {N} x {N}")
print(f"Variables por canal: {n}")
print(f"Variables totales tratadas en RGB: {3 * n}")

print("\nConstruyendo matrices dispersas...")

A = construir_matriz_difuminado(N, radio=radio, alpha=alpha)
D = construir_matriz_diferencias(N)

mask = (rng.random(n) > porcentaje_eliminado).astype(np.float64)
M = sparse.diags(mask, format="csr")

H = A.T @ (M @ A) + lambda_reg * (D.T @ D) + mu * sparse.eye(n, format="csr")
H = H.tocsc()
H.sum_duplicates()
H.eliminate_zeros()

Li = H.diagonal()

print("Matrices construidas.")
print(f"No nulos de A: {A.nnz}")
print(f"No nulos de D: {D.nnz}")
print(f"No nulos de H: {H.nnz}")
print(f"Li mínimo: {Li.min():.6e}")
print(f"Li máximo: {Li.max():.6e}")

print("\nEstimando constante Lipschitz global...")

L_global = estimar_L_global(H)

print(f"L global aproximada: {L_global:.6e}")
print(f"Paso gradiente 1/L: {1 / L_global:.6e}")
print(f"Paso coordenado mínimo 1/max Li: {1 / Li.max():.6e}")
print(f"Paso coordenado máximo 1/min Li: {1 / Li.min():.6e}")
print(f"Cota aproximada L/mu: {L_global / mu:.6e}")

print("\nCalentando funciones de numba...")

c_dummy = np.ones(n, dtype=np.float64)
Li_dummy = Li.astype(np.float64)

indptr_dummy = H.indptr.astype(np.int64)
indices_dummy = H.indices.astype(np.int64)
data_dummy = H.data.astype(np.float64)

_ = coord_ciclico_numba(
    c_dummy,
    Li_dummy,
    indptr_dummy,
    indices_dummy,
    data_dummy,
    1e-1,
    1
)

coords_dummy = rng.integers(0, n, size=(1, n), dtype=np.int64)

_ = coord_aleatorio_numba(
    c_dummy,
    Li_dummy,
    indptr_dummy,
    indices_dummy,
    data_dummy,
    coords_dummy,
    1e-1,
    1
)

_ = coord_greedy_numba(
    c_dummy,
    Li_dummy,
    indptr_dummy,
    indices_dummy,
    data_dummy,
    1e-1,
    1
)

print("Calentamiento completado.")

canales = {
    "rojo": img_true[:, :, 0].reshape(-1),
    "verde": img_true[:, :, 1].reshape(-1),
    "azul": img_true[:, :, 2].reshape(-1),
}

resultados_rgb = {}
b_rgb = {}

for k, (nombre, x_true_canal) in enumerate(canales.items()):
    resultados, b = resolver_canal(
        nombre,
        x_true_canal,
        A,
        D,
        mask,
        H,
        Li,
        L_global,
        tol,
        seed + k
    )

    resultados_rgb[nombre] = resultados
    b_rgb[nombre] = b


# ==================================================
# TABLA TOTAL RGB
# ==================================================

metodos = ["Gradiente", "Cíclico", "Greedy", "Aleatorio"]

print("\nCoste total RGB")
print("-" * 70)
print(f"{'Método':<12}{'Épocas totales':<18}{'Tiempo total (s)':<18}")
print("-" * 70)

for metodo in metodos:
    epocas_totales = sum(resultados_rgb[canal][metodo]["epocas"] for canal in resultados_rgb)
    tiempo_total = sum(resultados_rgb[canal][metodo]["tiempo"] for canal in resultados_rgb)

    print(
        f"{metodo:<12}"
        f"{epocas_totales:<18}"
        f"{tiempo_total:<18.6f}"
    )


# ==================================================
# RECONSTRUCCIONES RGB
# ==================================================

def reconstruir_rgb(resultados_rgb, metodo, N):
    R = resultados_rgb["rojo"][metodo]["x"].reshape(N, N)
    G = resultados_rgb["verde"][metodo]["x"].reshape(N, N)
    B = resultados_rgb["azul"][metodo]["x"].reshape(N, N)

    img = np.stack([R, G, B], axis=-1)
    return np.clip(img, 0.0, 1.0)


img_blur = np.zeros_like(img_true)
img_obs = np.zeros_like(img_true)

for idx, canal in enumerate(["rojo", "verde", "azul"]):
    x_true_canal = canales[canal]

    Ax = A @ x_true_canal

    X_blur = Ax.reshape(N, N)
    X_obs = X_blur.copy()
    X_obs[mask.reshape(N, N) == 0] = 0.0

    img_blur[:, :, idx] = X_blur
    img_obs[:, :, idx] = X_obs

img_blur = np.clip(img_blur, 0.0, 1.0)
img_obs = np.clip(img_obs, 0.0, 1.0)

img_grad = reconstruir_rgb(resultados_rgb, "Gradiente", N)
img_cic = reconstruir_rgb(resultados_rgb, "Cíclico", N)
img_gre = reconstruir_rgb(resultados_rgb, "Greedy", N)
img_ale = reconstruir_rgb(resultados_rgb, "Aleatorio", N)

guardar_rgb("rgb_original.png", img_true)
guardar_rgb("rgb_difuminada.png", img_blur)
guardar_rgb("rgb_observada.png", img_obs)
guardar_rgb("rgb_reconstruccion_gradiente.png", img_grad)
guardar_rgb("rgb_reconstruccion_ciclico.png", img_cic)
guardar_rgb("rgb_reconstruccion_greedy.png", img_gre)
guardar_rgb("rgb_reconstruccion_aleatorio.png", img_ale)


# ==================================================
# FIGURA 1: ORIGINAL, DIFUMINADA Y OBSERVADA
# ==================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(img_true)
axes[0].set_title("Original", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[0].axis("off")

axes[1].imshow(img_blur)
axes[1].set_title("Difuminada", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[1].axis("off")

axes[2].imshow(img_obs)
axes[2].set_title("Observada", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("rgb_fila_original_difuminada_observada.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# FIGURA 2: RECONSTRUCCIONES 2x2
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 13))

axes[0, 0].imshow(img_grad)
axes[0, 0].set_title("Gradiente", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[0, 0].axis("off")

axes[0, 1].imshow(img_cic)
axes[0, 1].set_title("Cíclico", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[0, 1].axis("off")

axes[1, 0].imshow(img_gre)
axes[1, 0].set_title("Greedy", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[1, 0].axis("off")

axes[1, 1].imshow(img_ale)
axes[1, 1].set_title("Aleatorio", fontsize=TAM_TITULO_IMAGEN, pad=14)
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("rgb_reconstrucciones_2x2.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# GRÁFICA: VALOR DE LA FUNCIÓN EN EL CANAL ROJO
# ==================================================

canal_grafica = "rojo"

plt.figure(figsize=(10, 7))

for metodo, res in resultados_rgb[canal_grafica].items():
    epocas = np.arange(len(res["F_hist"]))
    plt.plot(epocas, res["F_hist"], label=metodo, linewidth=GROSOR_LINEA)

plt.xlabel("Épocas", fontsize=TAM_EJES)
plt.ylabel(r"$F(x_k)$", fontsize=TAM_EJES)
plt.xticks(fontsize=TAM_TICKS)
plt.yticks(fontsize=TAM_TICKS)
plt.legend(fontsize=TAM_LEYENDA)
plt.grid(True)
plt.tight_layout()
plt.savefig("rgb_evolucion_funcion_canal_rojo.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# GRÁFICA: GRADIENTE RELATIVO EN EL CANAL ROJO
# ==================================================

plt.figure(figsize=(10, 7))

for metodo, res in resultados_rgb[canal_grafica].items():
    epocas = np.arange(len(res["grad_hist"]))
    plt.semilogy(epocas, res["grad_hist"], label=metodo, linewidth=GROSOR_LINEA)

plt.xlabel("Épocas", fontsize=TAM_EJES)
plt.ylabel(r"$\|\nabla F(x_k)\|/\|\nabla F(x_0)\|$", fontsize=TAM_EJES)
plt.xticks(fontsize=TAM_TICKS)
plt.yticks(fontsize=TAM_TICKS)
plt.legend(fontsize=TAM_LEYENDA)
plt.grid(True)
plt.tight_layout()
plt.savefig("rgb_gradiente_relativo_canal_rojo.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# ARCHIVOS GUARDADOS
# ==================================================

print("\nArchivos guardados:")
print(" - rgb_original.png")
print(" - rgb_difuminada.png")
print(" - rgb_observada.png")
print(" - rgb_reconstruccion_gradiente.png")
print(" - rgb_reconstruccion_ciclico.png")
print(" - rgb_reconstruccion_greedy.png")
print(" - rgb_reconstruccion_aleatorio.png")
print(" - rgb_fila_original_difuminada_observada.png")
print(" - rgb_reconstrucciones_2x2.png")
print(" - rgb_evolucion_funcion_canal_rojo.png")
print(" - rgb_gradiente_relativo_canal_rojo.png")