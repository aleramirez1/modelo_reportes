from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def ejecutar_paso_4():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "csv_dataset_new.csv"
    artefactos_dir = base_dir / "entrenamienot" / "artefactos"
    salida_path = base_dir / "data" / "dataset_reportes_clusterizado.csv"
    perfil_path = artefactos_dir / "perfil_clusters_paso4.csv"
    riesgo_meta_path = artefactos_dir / "riesgo_paso4.joblib"

    scaler_path = artefactos_dir / "scaler_paso2.joblib"
    features_path = artefactos_dir / "features_paso2.joblib"
    weights_path = artefactos_dir / "weights_paso2.joblib"
    kmeans_path = artefactos_dir / "kmeans_paso3.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {data_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No existe scaler del paso 2: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"No existe features del paso 2: {features_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe weights del paso 2: {weights_path}")
    if not kmeans_path.exists():
        raise FileNotFoundError(f"No existe modelo del paso 3: {kmeans_path}")

    df = pd.read_csv(data_path)

    columnas_modelo = joblib.load(features_path)
    if "porcentaje_groserias" in columnas_modelo and "porcentaje_groserias" not in df.columns:
        if {"groserias", "palabras"}.issubset(df.columns):
            denom = pd.to_numeric(df["palabras"], errors="coerce").replace(0, np.nan)
            numer = pd.to_numeric(df["groserias"], errors="coerce")
            df["porcentaje_groserias"] = (numer / denom * 100).fillna(0.0)
        else:
            raise ValueError("No se puede derivar porcentaje_groserias: faltan groserias/palabras")

    for c in columnas_modelo:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nulos = int(df[columnas_modelo].isnull().sum().sum())
    if nulos > 0:
        raise ValueError(f"Hay nulos en columnas del modelo: {nulos}. Revísalos antes de asignar clusters.")

    X = df[columnas_modelo].values

    scaler = joblib.load(scaler_path)
    weights = np.asarray(joblib.load(weights_path), dtype=float)
    kmeans = joblib.load(kmeans_path)

    X_scaled = scaler.transform(X)
    X_modelo = X_scaled * weights
    clusters = kmeans.predict(X_modelo)
    distancias = kmeans.transform(X_modelo).min(axis=1)

    p90 = float(np.percentile(distancias, 90))
    p97 = float(np.percentile(distancias, 97))

    nivel_riesgo = np.where(distancias >= p97, "alto", np.where(distancias >= p90, "medio", "bajo"))

    df_salida = df.copy()
    df_salida["cluster"] = clusters
    df_salida["distancia_centroide"] = distancias
    df_salida["nivel_riesgo"] = nivel_riesgo

    perfil = df_salida.groupby("cluster")[columnas_modelo + ["distancia_centroide"]].mean().round(4)
    perfil["cantidad_reportes"] = df_salida.groupby("cluster").size()

    df_salida.to_csv(salida_path, index=False, encoding="utf-8")
    perfil.to_csv(perfil_path, encoding="utf-8")

    joblib.dump(
        {
            "p90_distancia": p90,
            "p97_distancia": p97,
            "columnas_modelo": columnas_modelo,
            "pesos_modelo": weights.tolist(),
        },
        riesgo_meta_path,
    )

    conteo_clusters = df_salida["cluster"].value_counts().sort_index()
    conteo_riesgo = df_salida["nivel_riesgo"].value_counts()

    print("=" * 80)
    print("PASO 4 - ASIGNACION DE CLUSTERS Y NIVEL DE RIESGO")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Filas procesadas: {len(df_salida)}")
    print(f"Columnas modelo: {len(columnas_modelo)}")

    print("\nDistribucion por cluster:")
    for c, n in conteo_clusters.items():
        pct = n / len(df_salida) * 100
        print(f"- cluster {c}: {n} ({pct:.1f}%)")

    print("\nDistribucion por nivel de riesgo:")
    for nivel, n in conteo_riesgo.items():
        pct = n / len(df_salida) * 100
        print(f"- {nivel}: {n} ({pct:.1f}%)")

    print("\nUmbrales de riesgo por distancia:")
    print(f"- p90: {p90:.4f}")
    print(f"- p97: {p97:.4f}")

    print("\nArchivos generados:")
    print(f"- {salida_path}")
    print(f"- {perfil_path}")
    print(f"- {riesgo_meta_path}")
    print("\nEstado: OK para avanzar a Paso 5 (deteccion de anomalias con Isolation Forest)")


if __name__ == "__main__":
    ejecutar_paso_4()
