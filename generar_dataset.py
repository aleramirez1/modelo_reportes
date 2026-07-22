from pathlib import Path

import numpy as np
import pandas as pd


N_FILAS = 1000
SEED = 42


def generar_dataset(n=N_FILAS, seed=SEED):
    rng = np.random.default_rng(seed)

    n_anomalos = int(n * 0.10)
    n_normales = n - n_anomalos

    filas = []

    for _ in range(n_normales):
        palabras = int(rng.integers(3, 20))
        groserias = 0
        palabras_repetidas = int(rng.integers(0, max(1, palabras // 4) + 1))
        caracteres_especiales = int(rng.integers(0, 3))
        caracteres_especiales_repitidos = 0
        cantidad_numeros = int(rng.integers(0, 3))
        tiempo = int(rng.integers(1, 240))
        contiene_url = 0
        palabras_claves = int(rng.integers(0, min(palabras, 6) + 1))
        porcentaje_groserias = 0.0
        filas.append([
            groserias, palabras_repetidas, caracteres_especiales,
            caracteres_especiales_repitidos, cantidad_numeros, palabras,
            tiempo, contiene_url, porcentaje_groserias, palabras_claves,
        ])

    for _ in range(n_anomalos):
        palabras = int(rng.integers(1, 15))
        groserias = int(rng.integers(1, max(2, palabras) + 1))
        groserias = min(groserias, palabras)
        palabras_repetidas = int(rng.integers(0, palabras + 1))
        caracteres_especiales = int(rng.integers(0, 15))
        caracteres_especiales_repitidos = int(rng.integers(0, 8))
        cantidad_numeros = int(rng.integers(0, 6))
        tiempo = int(rng.integers(1, 400))
        contiene_url = int(rng.integers(0, 2))
        palabras_claves = int(rng.integers(0, min(palabras, 3) + 1))
        porcentaje_groserias = round(groserias / palabras * 100, 6) if palabras > 0 else 0.0
        filas.append([
            groserias, palabras_repetidas, caracteres_especiales,
            caracteres_especiales_repitidos, cantidad_numeros, palabras,
            tiempo, contiene_url, porcentaje_groserias, palabras_claves,
        ])

    columnas = [
        "groserias",
        "palabras_repetidas",
        "caracteres_especiales",
        "caracteres_especiales_repitidos",
        "cantidad_numeros",
        "palabras",
        "tiempo_desde_ultimo_reporte_min",
        "contiene_url",
        "porcentaje_groserias",
        "palabras_claves",
    ]

    df = pd.DataFrame(filas, columns=columnas)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    salida = data_dir / "csv_dataset_new.csv"

    df = generar_dataset()
    df.to_csv(salida, index=False, encoding="utf-8")

    print(f"Dataset generado: {salida}")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {df.shape[1]}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
