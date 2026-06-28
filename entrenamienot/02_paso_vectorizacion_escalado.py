from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


def ejecutar_paso_2():
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
        raise ValueError(f"Hay nulos en columnas del modelo: {nulos}. Revísalos antes de escalar.")

    X = df[columnas_modelo].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = artefactos_dir / "scaler_paso2.joblib"
    features_path = artefactos_dir / "features_paso2.joblib"
    weights_path = artefactos_dir / "weights_paso2.joblib"

    joblib.dump(scaler, scaler_path)
    joblib.dump(columnas_modelo, features_path)
    joblib.dump(PESOS_PRIORIZADOS, weights_path)

    print("=" * 80)
    print("PASO 2 - VECTORIZACION TABULAR Y ESCALADO")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Filas: {X.shape[0]}")
    print(f"Columnas usadas: {X.shape[1]}")
    print("\nColumnas priorizadas del modelo:")
    for c in columnas_modelo:
        print(f"- {c}")

    print("\nPesos aplicados por feature:")
    for c, w in zip(columnas_modelo, PESOS_PRIORIZADOS.tolist()):
        print(f"- {c}: {w}")

    print("\nMuestra X sin escalar (primeras 3 filas):")
    print(pd.DataFrame(X[:3], columns=columnas_modelo))

    print("\nMuestra X escalado (primeras 3 filas):")
    print(pd.DataFrame(X_scaled[:3], columns=columnas_modelo).round(4))

    print("\nArtefactos guardados:")
    print(f"- {scaler_path}")
    print(f"- {features_path}")
    print(f"- {weights_path}")
    print("\nEstado: OK para avanzar a Paso 3 (entrenar K-Means y elegir k)")


if __name__ == "__main__":
    ejecutar_paso_2()
