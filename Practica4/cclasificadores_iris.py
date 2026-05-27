"""
=============================================================
  CLASIFICADORES KNN / EUCLIDIANO - DATASET IRIS (UCI)
  Implementación desde cero (sin scikit-learn para clasificadores)
=============================================================
  Requisitos: numpy, matplotlib, pandas
  Instalar: pip install numpy matplotlib pandas
  Uso:       python clasificadores_iris.py
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  1.  DATASET IRIS (incrustado, sin sklearn)
# ─────────────────────────────────────────────
def cargar_iris():
    """
    Carga el dataset Iris directamente desde el repositorio UCI.
    Si no hay conexión, usa los datos incrustados.
    """
    try:
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        with urllib.request.urlopen(url, timeout=5) as r:
            raw = r.read().decode()
        rows = [l.split(",") for l in raw.strip().split("\n") if l]
        X = np.array([[float(v) for v in r[:4]] for r in rows])
        clases_str = [r[4].strip() for r in rows]
        nombres = sorted(set(clases_str))
        mapa = {n: i for i, n in enumerate(nombres)}
        y = np.array([mapa[c] for c in clases_str])
        return X, y, nombres
    except Exception:
        # Datos incrustados (primeras filas de cada clase para arrancar sin internet)
        pass

    # Dataset Iris completo (150 muestras)
    datos = np.array([
        [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],
        [4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],
        [5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
        [5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],
        [4.8,3.4,1.9,0.2],[5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],
        [5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],
        [5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],[5.0,3.2,1.2,0.2],
        [5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],
        [5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],
        [5.1,3.8,1.9,0.4],[4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],
        [5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],
        [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],
        [6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],
        [6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],
        [5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
        [5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],
        [6.4,2.9,4.3,1.3],[6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],
        [6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],
        [5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],[6.0,3.4,4.5,1.6],
        [6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],
        [5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],
        [5.6,2.7,4.2,1.3],[5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],
        [5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],
        [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],
        [6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],
        [6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],
        [6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],
        [6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
        [6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],
        [6.7,3.3,5.7,2.1],[7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],
        [6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],
        [6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3.0,6.1,2.3],
        [6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],
        [6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],
        [6.7,3.3,5.7,2.5],[6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],
        [6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8],
    ])
    etiquetas = np.array([0]*50 + [1]*50 + [2]*50)
    nombres = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    return datos, etiquetas, nombres


# ─────────────────────────────────────────────
#  2.  EDA
# ─────────────────────────────────────────────
def eda(X, y, nombres_clases, nombres_attrs):
    print("=" * 60)
    print("  ANÁLISIS EXPLORATORIO DE DATOS (EDA) — IRIS")
    print("=" * 60)
    print(f"\n  Número de filas    : {X.shape[0]}")
    print(f"  Número de columnas : {X.shape[1] + 1}  (4 atributos + 1 clase)")
    print(f"  Valores nulos      : 0")

    print("\n  Distribución de clases:")
    for i, nombre in enumerate(nombres_clases):
        n = np.sum(y == i)
        print(f"    {nombre:<20}: {n:3d}  ({n/len(y)*100:.1f}%)")

    print("\n  Estadística fundamental por atributo:")
    header = f"  {'Atributo':<16} {'Media':>8} {'Desv.Est':>10} {'Mín':>7} {'Máx':>7}"
    print(header)
    print("  " + "-" * 54)
    for i, attr in enumerate(nombres_attrs):
        col = X[:, i]
        print(f"  {attr:<16} {np.mean(col):>8.3f} {np.std(col):>10.3f} "
              f"{np.min(col):>7.2f} {np.max(col):>7.2f}")

    print("\n  Estadística por clase:")
    for ci, cn in enumerate(nombres_clases):
        print(f"\n    [{cn}]")
        print(f"    {'Atributo':<16} {'Media':>8} {'Desv.Est':>10}")
        print("    " + "-" * 36)
        for ai, attr in enumerate(nombres_attrs):
            col = X[y == ci, ai]
            print(f"    {attr:<16} {np.mean(col):>8.3f} {np.std(col):>10.3f}")
    print()


# ─────────────────────────────────────────────
#  3.  VISUALIZACIÓN 2D
# ─────────────────────────────────────────────
def visualizar(X, y, nombres_clases, nombres_attrs):
    colores = ["#1D9E75", "#378ADD", "#D85A30"]
    marcadores = ["o", "s", "^"]

    pares = [(0,1), (0,2), (0,3), (2,3)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Iris — Pares de atributos por clase", fontsize=14, fontweight="bold", y=1.01)

    for ax, (xi, yi_) in zip(axes.flat, pares):
        for ci in range(len(nombres_clases)):
            mask = y == ci
            ax.scatter(X[mask, xi], X[mask, yi_],
                       color=colores[ci], marker=marcadores[ci],
                       alpha=0.75, s=45, label=nombres_clases[ci], edgecolors="white", linewidths=0.4)
        ax.set_xlabel(nombres_attrs[xi], fontsize=10)
        ax.set_ylabel(nombres_attrs[yi_], fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

    handles = [mpatches.Patch(color=colores[i], label=nombres_clases[i])
               for i in range(len(nombres_clases))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    plt.tight_layout()
    plt.savefig("iris_scatter.png", dpi=120, bbox_inches="tight")
    print("  Gráfica guardada: iris_scatter.png")
    plt.show()

    # Histogramas
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3))
    fig2.suptitle("Distribución de atributos por clase", fontsize=13, fontweight="bold")
    for ai, ax in enumerate(axes2):
        for ci in range(len(nombres_clases)):
            ax.hist(X[y == ci, ai], bins=15, alpha=0.55,
                    color=colores[ci], label=nombres_clases[ci], edgecolor="none")
        ax.set_title(nombres_attrs[ai], fontsize=10)
        ax.set_xlabel("cm", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, axis="y")
    handles2 = [mpatches.Patch(color=colores[i], label=nombres_clases[i])
                for i in range(len(nombres_clases))]
    fig2.legend(handles=handles2, loc="lower center", ncol=3, fontsize=9,
                bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig("iris_histogramas.png", dpi=120, bbox_inches="tight")
    print("  Gráfica guardada: iris_histogramas.png")
    plt.show()


# ─────────────────────────────────────────────
#  4.  FUNCIONES DE DISTANCIA
# ─────────────────────────────────────────────
def dist_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def dist_manhattan(a, b):
    return np.sum(np.abs(a - b))

def dist_chebyshev(a, b):
    return np.max(np.abs(a - b))

def dist_coseno(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def dist_canberra(a, b):
    denom = np.abs(a) + np.abs(b)
    mask = denom > 0
    return np.sum(np.abs(a[mask] - b[mask]) / denom[mask])

DISTANCIAS = {
    "Euclidiana": dist_euclidiana,
    "Manhattan" : dist_manhattan,
    "Chebyshev" : dist_chebyshev,
    "Coseno"    : dist_coseno,
    "Canberra"  : dist_canberra,
}


# ─────────────────────────────────────────────
#  5.  CLASIFICADORES
# ─────────────────────────────────────────────

# 5a. Clasificador Euclidiano por centroides
def clasificador_euclidiano(X_train, y_train, x_test, dist_fn):
    """Calcula el centroide de cada clase y predice la clase del centroide más cercano."""
    clases = np.unique(y_train)
    centroides = {c: X_train[y_train == c].mean(axis=0) for c in clases}
    distancias = {c: dist_fn(x_test, centro) for c, centro in centroides.items()}
    return min(distancias, key=distancias.get)


# 5b. KNN general (sirve para 1NN, 3NN, 5NN, 7NN, 9NN, 11NN)
def clasificador_knn(X_train, y_train, x_test, k, dist_fn):
    """Clasifica x_test por mayoría de votos entre los k vecinos más cercanos."""
    distancias = [(dist_fn(x_test, X_train[i]), y_train[i])
                  for i in range(len(X_train))]
    distancias.sort(key=lambda t: t[0])
    vecinos = [lbl for _, lbl in distancias[:k]]
    return Counter(vecinos).most_common(1)[0][0]


def predecir(X_train, y_train, X_test, clasificador, k, dist_fn):
    """Aplica el clasificador a cada muestra de X_test y devuelve predicciones."""
    preds = []
    for x in X_test:
        if clasificador == "euclid":
            preds.append(clasificador_euclidiano(X_train, y_train, x, dist_fn))
        else:
            preds.append(clasificador_knn(X_train, y_train, x, k, dist_fn))
    return np.array(preds)


def exactitud(y_real, y_pred):
    return np.mean(y_real == y_pred)


# ─────────────────────────────────────────────
#  6.  MÉTODOS DE VALIDACIÓN
# ─────────────────────────────────────────────

# 6a. Leave-One-Out
def leave_one_out(X, y, clasificador, k, dist_fn):
    n = len(X)
    correctos = 0
    for i in range(n):
        mask_train = np.ones(n, dtype=bool)
        mask_train[i] = False
        X_train, y_train = X[mask_train], y[mask_train]
        pred = predecir(X_train, y_train, X[i:i+1], clasificador, k, dist_fn)
        if pred[0] == y[i]:
            correctos += 1
    return correctos / n


# 6b. K-Fold Cross-Validation
def kfold_cv(X, y, k_folds, clasificador, k_clf, dist_fn, semilla=42):
    n = len(X)
    indices = np.arange(n)
    rng = np.random.default_rng(semilla)
    rng.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    accs = []
    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_idx])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val     = X[val_idx],   y[val_idx]
        y_pred = predecir(X_train, y_train, X_val, clasificador, k_clf, dist_fn)
        accs.append(exactitud(y_val, y_pred))
    return np.mean(accs)


# 6c. Hold-Out 70 / 30
def hold_out(X, y, proporcion_train, clasificador, k, dist_fn, semilla=42):
    n = len(X)
    indices = np.arange(n)
    rng = np.random.default_rng(semilla)
    rng.shuffle(indices)
    n_train = int(n * proporcion_train)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test,  y_test  = X[indices[n_train:]], y[indices[n_train:]]
    y_pred = predecir(X_train, y_train, X_test, clasificador, k, dist_fn)
    return exactitud(y_test, y_pred)


# ─────────────────────────────────────────────
#  7.  EXPERIMENTO COMPLETO
# ─────────────────────────────────────────────
def ejecutar_experimentos(X, y):
    clasificadores_cfg = [
        ("Euclidiano", "euclid", None),
        ("1NN",        "knn",    1),
        ("3NN",        "knn",    3),
        ("5NN",        "knn",    5),
        ("7NN",        "knn",    7),
        ("9NN",        "knn",    9),
        ("11NN",       "knn",    11),
    ]
    validaciones = ["LOO", "10-Fold CV", "Hold-Out 70/30"]

    print("=" * 60)
    print("  EJECUTANDO EXPERIMENTOS...")
    print("=" * 60)

    resultados = {}   # (distancia, validacion, clf_nombre) -> acc

    for nombre_dist, dist_fn in DISTANCIAS.items():
        print(f"\n  Distancia: {nombre_dist}")
        for val_nombre in validaciones:
            print(f"    Validación: {val_nombre:<18}", end="")
            for clf_nombre, clf_tipo, k_val in clasificadores_cfg:
                if val_nombre == "LOO":
                    acc = leave_one_out(X, y, clf_tipo, k_val, dist_fn)
                elif val_nombre == "10-Fold CV":
                    acc = kfold_cv(X, y, 10, clf_tipo, k_val, dist_fn)
                else:
                    acc = hold_out(X, y, 0.70, clf_tipo, k_val, dist_fn)
                resultados[(nombre_dist, val_nombre, clf_nombre)] = acc
            print("OK")

    return resultados, clasificadores_cfg, validaciones


# ─────────────────────────────────────────────
#  8.  TABLA COMPARATIVA
# ─────────────────────────────────────────────
def imprimir_tabla(resultados, clasificadores_cfg, validaciones):
    clf_nombres = [c[0] for c in clasificadores_cfg]
    ancho_clf = 12

    print("\n")
    print("=" * 80)
    print("  TABLA COMPARATIVA DE EXACTITUD (%)")
    print("=" * 80)

    for nombre_dist in DISTANCIAS:
        print(f"\n  Distancia: {nombre_dist}")
        # Encabezado
        header = f"  {'Validación':<18}" + "".join(f"{n:>{ancho_clf}}" for n in clf_nombres)
        print(header)
        print("  " + "-" * (18 + ancho_clf * len(clf_nombres)))
        for val_nombre in validaciones:
            fila = f"  {val_nombre:<18}"
            for clf_nombre in clf_nombres:
                acc = resultados[(nombre_dist, val_nombre, clf_nombre)]
                fila += f"{acc*100:>{ancho_clf}.2f}"
            print(fila)

    # Mejor resultado global
    mejor_key = max(resultados, key=resultados.get)
    mejor_acc = resultados[mejor_key]
    print(f"\n  Mejor resultado: {mejor_acc*100:.2f}%  →  "
          f"Distancia={mejor_key[0]}, Validación={mejor_key[1]}, Clasificador={mejor_key[2]}")


def graficar_tabla(resultados, clasificadores_cfg, validaciones):
    """Genera un heatmap de exactitudes por distancia y clasificador (promedio de validaciones)."""
    clf_nombres = [c[0] for c in clasificadores_cfg]
    dist_nombres = list(DISTANCIAS.keys())

    # Promedio sobre validaciones
    matriz = np.zeros((len(dist_nombres), len(clf_nombres)))
    for di, d in enumerate(dist_nombres):
        for ci, c in enumerate(clf_nombres):
            vals = [resultados[(d, v, c)] for v in validaciones]
            matriz[di, ci] = np.mean(vals) * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matriz, aspect="auto", cmap="YlGn", vmin=80, vmax=100)
    ax.set_xticks(range(len(clf_nombres)))
    ax.set_xticklabels(clf_nombres, fontsize=10)
    ax.set_yticks(range(len(dist_nombres)))
    ax.set_yticklabels(dist_nombres, fontsize=10)
    ax.set_title("Exactitud promedio (%) por distancia y clasificador", fontsize=12, pad=12)

    for di in range(len(dist_nombres)):
        for ci in range(len(clf_nombres)):
            val = matriz[di, ci]
            color = "black" if val < 95 else "white"
            ax.text(ci, di, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Exactitud (%)")
    plt.tight_layout()
    plt.savefig("iris_heatmap.png", dpi=120, bbox_inches="tight")
    print("  Gráfica guardada: iris_heatmap.png")
    plt.show()

    # Gráfica por validación
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colores_dist = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#BA7517"]
    for vi, (val_nombre, ax) in enumerate(zip(validaciones, axes)):
        for di, (dist_nombre, color) in enumerate(zip(dist_nombres, colores_dist)):
            accs = [resultados[(dist_nombre, val_nombre, c)] * 100 for c in clf_nombres]
            ax.plot(clf_nombres, accs, marker="o", markersize=6,
                    label=dist_nombre, color=color, linewidth=1.8)
        ax.set_title(val_nombre, fontsize=11)
        ax.set_ylim(70, 102)
        ax.set_xlabel("Clasificador", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(True, alpha=0.25)
        if vi == 0:
            ax.set_ylabel("Exactitud (%)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
                bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig2.suptitle("Exactitud por clasificador, distancia y método de validación",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("iris_lineas.png", dpi=120, bbox_inches="tight")
    print("  Gráfica guardada: iris_lineas.png")
    plt.show()


# ─────────────────────────────────────────────
#  9.  MAIN
# ─────────────────────────────────────────────
def main():
    ATTRS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    print("\n  Cargando dataset Iris...")
    X, y, nombres_clases = cargar_iris()
    print(f"  Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} atributos.\n")

    # 1. EDA
    eda(X, y, nombres_clases, ATTRS)

    # 2. Visualización
    print("  Generando gráficas EDA...")
    visualizar(X, y, nombres_clases, ATTRS)

    # 3. Clasificadores + validación
    resultados, clasificadores_cfg, validaciones = ejecutar_experimentos(X, y)

    # 4. Tabla comparativa
    imprimir_tabla(resultados, clasificadores_cfg, validaciones)

    # 5. Gráficas de resultados
    print("\n  Generando gráficas de resultados...")
    graficar_tabla(resultados, clasificadores_cfg, validaciones)

    print("\n  ¡Listo! Archivos generados:")
    print("    iris_scatter.png, iris_histogramas.png,")
    print("    iris_heatmap.png, iris_lineas.png\n")


if __name__ == "__main__":
    main()
