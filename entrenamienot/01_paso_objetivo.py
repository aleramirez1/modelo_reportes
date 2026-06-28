from pathlib import Path

import pandas as pd


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


def ejecutar_paso_1():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "csv_dataset_new.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {data_path}")

    df = pd.read_csv(data_path)
    if df.shape[1] < 3:
        raise ValueError("El dataset debe tener al menos 3 columnas para omitir las primeras dos")

    columnas_omitidas = df.columns.tolist()[:2]
    faltantes = [c for c in FEATURES_PRIORIZADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas priorizadas en el dataset: {faltantes}")

    columnas_modelo = FEATURES_PRIORIZADAS.copy()

    objetivo_modelo = {
        "tipo": "no_supervisado",
        "algoritmo_principal": "kmeans",
        "entrada": "vector tabular numerico estandarizado",
        "salida": "cluster por reporte",
        "uso": "agrupar patrones y preparar deteccion de rareza",
    }

    print("=" * 80)
    print("PASO 1 - DEFINICION DEL MODELO")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Columnas omitidas por configuracion: {columnas_omitidas}")
    print("\nObjetivo del paso 1:")
    for k, v in objetivo_modelo.items():
        print(f"- {k}: {v}")

    print("\nVariables priorizadas para vectorizacion:")
    for c in columnas_modelo:
        print(f"- {c}")

    print("\nEstado: OK para avanzar a Paso 2 (vectorizacion y escalado)")


if __name__ == "__main__":
    ejecutar_paso_1()
