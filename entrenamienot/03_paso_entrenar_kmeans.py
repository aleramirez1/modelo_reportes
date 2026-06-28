from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES_PRIORIZADAS = [
    "groserias",
    "porcentaje_groserias",
    "palabras_repetidas",
    "caracteres_especiales",
    "caracteres_especiales_repitidos",
    "contiene_url",
    "palabras_claves",
    "palabras",
]

PESOS_PRIORIZADOS = np.array([3.0, 3.0, 1.5, 0.3, 0.3, 2.0, 2.0, 1.0], dtype=float)


def ejecutar_paso_3():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "csv_dataset_new.csv"
    artefactos_dir = base_dir / "entrenamienot" / "artefactos"
    artefactos_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {data_path}")

    df = pd.read_csv(data_path)
    if df.shape[1] < 3:
        raise ValueError("El dataset debe tener al menos 3 columnas para omitir las primeras dos")

    if {"groserias", "palabras"}.issubset(df.columns):
        denom = df["palabras"].replace(0, np.nan)
        df["porcentaje_groserias"] = (df["groserias"] / denom * 100).fillna(0.0)

    faltantes = [c for c in FEATURES_PRIORIZADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas priorizadas en el dataset: {faltantes}")

    columnas_modelo = FEATURES_PRIORIZADAS.copy()

    for c in columnas_modelo:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nulos = int(df[columnas_modelo].isnull().sum().sum())
    if nulos > 0:
        raise ValueError(f"Hay nulos en columnas del modelo: {nulos}. Revísalos antes de entrenar.")

    X = df[columnas_modelo].values

    scaler_path = artefactos_dir / "scaler_paso2.joblib"
    weights_path = artefactos_dir / "weights_paso2.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", None) != X.shape[1]:
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, scaler_path)

    joblib.dump(PESOS_PRIORIZADOS, weights_path)

    X_scaled = scaler.transform(X)
    X_modelo = X_scaled * PESOS_PRIORIZADOS

    resultados = []
    mejor_k = None
    mejor_score = -1.0
    mejor_modelo = None

    k_min = 2
    k_max = 9

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = km.fit_predict(X_modelo)
        score = silhouette_score(X_modelo, labels)
        resultados.append((k, score, km.inertia_))
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            mejor_modelo = km

    kmeans_path = artefactos_dir / "kmeans_paso3.joblib"
    metricas_path = artefactos_dir / "metricas_kmeans_paso3.joblib"
    features_path = artefactos_dir / "features_paso2.joblib"

    joblib.dump(mejor_modelo, kmeans_path)
    joblib.dump(
        {
            "mejor_k": int(mejor_k),
            "mejor_silhouette": float(mejor_score),
            "resultados": resultados,
            "columnas_modelo": columnas_modelo,
            "pesos_modelo": PESOS_PRIORIZADOS.tolist(),
        },
        metricas_path,
    )

    joblib.dump(columnas_modelo, features_path)

    print("=" * 80)
    print("PASO 3 - ENTRENAMIENTO K-MEANS")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Filas: {X.shape[0]}")
    print(f"Columnas usadas: {X.shape[1]}")
    print(f"Features priorizadas: {columnas_modelo}")
    print(f"Pesos aplicados: {PESOS_PRIORIZADOS.tolist()}")
    print(f"Rango evaluado de k: {k_min} a {k_max}")
    print("\nResultados por k:")
    for k, score, inertia in resultados:
        print(f"- k={k} | silhouette={score:.4f} | inertia={inertia:.2f}")

    print(f"\nK recomendado por silhouette: {mejor_k}")
    print(f"Mejor silhouette: {mejor_score:.4f}")

    print("\nArtefactos guardados:")
    print(f"- {kmeans_path}")
    print(f"- {metricas_path}")
    print("\nEstado: OK para avanzar a Paso 4 (asignacion de clusters y riesgo)")


if __name__ == "__main__":
    ejecutar_paso_3()
