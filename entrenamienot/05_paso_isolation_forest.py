from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def ejecutar_paso_5():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "csv_dataset_new.csv"
    clusterizado_path = base_dir / "data" / "dataset_reportes_clusterizado.csv"
    salida_path = base_dir / "data" / "dataset_reportes_clusterizado_anomalias.csv"
    artefactos_dir = base_dir / "entrenamienot" / "artefactos"

    scaler_path = artefactos_dir / "scaler_paso2.joblib"
    features_path = artefactos_dir / "features_paso2.joblib"
    weights_path = artefactos_dir / "weights_paso2.joblib"
    modelo_path = artefactos_dir / "isolation_forest_paso5.joblib"
    metricas_path = artefactos_dir / "metricas_isolation_paso5.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset base: {data_path}")
    if not clusterizado_path.exists():
        raise FileNotFoundError(f"No existe dataset clusterizado del paso 4: {clusterizado_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No existe scaler del paso 2: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"No existe listado de features del paso 2: {features_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe listado de pesos del paso 2: {weights_path}")

    columnas_modelo = joblib.load(features_path)

    df_base = pd.read_csv(data_path)
    df_cluster = pd.read_csv(clusterizado_path)

    if "porcentaje_groserias" in columnas_modelo and "porcentaje_groserias" not in df_base.columns:
        if {"groserias", "palabras"}.issubset(df_base.columns):
            denom = pd.to_numeric(df_base["palabras"], errors="coerce").replace(0, np.nan)
            numer = pd.to_numeric(df_base["groserias"], errors="coerce")
            df_base["porcentaje_groserias"] = (numer / denom * 100).fillna(0.0)
        else:
            raise ValueError("No se puede derivar porcentaje_groserias: faltan groserias/palabras")

    for c in columnas_modelo:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce")

    nulos = int(df_base[columnas_modelo].isnull().sum().sum())
    if nulos > 0:
        raise ValueError(f"Hay nulos en columnas del modelo: {nulos}. Revísalos antes de detectar anomalías.")

    X = df_base[columnas_modelo].values
    scaler = joblib.load(scaler_path)
    weights = np.asarray(joblib.load(weights_path), dtype=float)
    X_scaled = scaler.transform(X)
    X_modelo = X_scaled * weights

    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=300)
    pred = iso.fit_predict(X_modelo)
    score = iso.decision_function(X_modelo)

    score_invertido = -score
    smin = float(score_invertido.min())
    smax = float(score_invertido.max())
    score_norm = (score_invertido - smin) / (smax - smin + 1e-9)

    df_salida = df_cluster.copy()
    df_salida["anomalia_if"] = (pred == -1).astype(int)
    df_salida["score_anomalia_if"] = score
    df_salida["riesgo_if_0_1"] = score_norm

    q90 = float(np.percentile(score_norm, 90))
    q97 = float(np.percentile(score_norm, 97))
    nivel_if = np.where(score_norm >= q97, "alto", np.where(score_norm >= q90, "medio", "bajo"))
    df_salida["nivel_riesgo_if"] = nivel_if

    if "nivel_riesgo" in df_salida.columns:
        combinado = []
        for i in range(len(df_salida)):
            a = str(df_salida.loc[i, "nivel_riesgo"])
            b = str(df_salida.loc[i, "nivel_riesgo_if"])
            if a == "alto" or b == "alto":
                combinado.append("alto")
            elif a == "medio" or b == "medio":
                combinado.append("medio")
            else:
                combinado.append("bajo")
        df_salida["nivel_riesgo_final"] = combinado
    else:
        df_salida["nivel_riesgo_final"] = df_salida["nivel_riesgo_if"]

    joblib.dump(iso, modelo_path)
    joblib.dump(
        {
            "contamination": 0.1,
            "q90_score_norm": q90,
            "q97_score_norm": q97,
            "columnas_modelo": columnas_modelo,
            "pesos_modelo": weights.tolist(),
            "anomalias": int(df_salida["anomalia_if"].sum()),
            "pct_anomalias": float(df_salida["anomalia_if"].mean() * 100),
        },
        metricas_path,
    )

    df_salida.to_csv(salida_path, index=False, encoding="utf-8")

    conteo_if = df_salida["anomalia_if"].value_counts().to_dict()
    conteo_nivel = df_salida["nivel_riesgo_final"].value_counts().to_dict()

    print("=" * 80)
    print("PASO 5 - DETECCION DE ANOMALIAS CON ISOLATION FOREST")
    print("=" * 80)
    print(f"Dataset base: {data_path}")
    print(f"Filas procesadas: {len(df_salida)}")
    print(f"Columnas modelo: {len(columnas_modelo)}")
    print("\nConteo anomalia_if (0=normal, 1=anomalia):")
    print(conteo_if)
    print("\nDistribucion nivel_riesgo_final:")
    print(conteo_nivel)
    print("\nUmbrales score IF normalizado:")
    print(f"- q90: {q90:.4f}")
    print(f"- q97: {q97:.4f}")
    print("\nArchivos generados:")
    print(f"- {salida_path}")
    print(f"- {modelo_path}")
    print(f"- {metricas_path}")
    print("\nEstado: OK para avanzar a Paso 6 (resumen final y exportes de operacion)")


if __name__ == "__main__":
    ejecutar_paso_5()
