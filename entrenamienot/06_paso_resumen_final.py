from pathlib import Path

import json
import joblib
import pandas as pd


def ejecutar_paso_6():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "dataset_reportes_clusterizado_anomalias.csv"
    artefactos_dir = base_dir / "entrenamienot" / "artefactos"
    salida_resumen = artefactos_dir / "resumen_final_paso6.json"
    salida_tabla = artefactos_dir / "resumen_final_paso6.csv"
    salida_operativa = base_dir / "data" / "dataset_reportes_final.csv"

    metricas_kmeans_path = artefactos_dir / "metricas_kmeans_paso3.joblib"
    metricas_iso_path = artefactos_dir / "metricas_isolation_paso5.joblib"
    perfil_clusters_path = artefactos_dir / "perfil_clusters_paso4.csv"
    riesgo_meta_path = artefactos_dir / "riesgo_paso4.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset final esperado: {data_path}")

    df = pd.read_csv(data_path)

    columnas_requeridas = ["cluster", "distancia_centroide", "anomalia_if", "nivel_riesgo_final"]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias para el resumen final: {faltantes}")

    resumen_general = {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "clusters_unicos": int(df["cluster"].nunique()),
        "anomalias": int(df["anomalia_if"].sum()),
        "pct_anomalias": float(df["anomalia_if"].mean() * 100),
        "riesgo_bajo": int((df["nivel_riesgo_final"] == "bajo").sum()),
        "riesgo_medio": int((df["nivel_riesgo_final"] == "medio").sum()),
        "riesgo_alto": int((df["nivel_riesgo_final"] == "alto").sum()),
    }

    resumen_clusters = (
        df.groupby("cluster")
        .agg(
            cantidad_reportes=("cluster", "size"),
            distancia_promedio=("distancia_centroide", "mean"),
            distancia_maxima=("distancia_centroide", "max"),
            anomalias=("anomalia_if", "sum"),
        )
        .round(4)
        .reset_index()
    )

    resumen_riesgo = (
        df.groupby("nivel_riesgo_final")
        .agg(
            cantidad_reportes=("nivel_riesgo_final", "size"),
            distancia_promedio=("distancia_centroide", "mean"),
            anomalias=("anomalia_if", "sum"),
        )
        .round(4)
        .reset_index()
    )

    if metricas_kmeans_path.exists():
        metricas_kmeans = joblib.load(metricas_kmeans_path)
    else:
        metricas_kmeans = {}

    if metricas_iso_path.exists():
        metricas_iso = joblib.load(metricas_iso_path)
    else:
        metricas_iso = {}

    if perfil_clusters_path.exists():
        perfil_clusters = pd.read_csv(perfil_clusters_path)
    else:
        perfil_clusters = pd.DataFrame()

    if riesgo_meta_path.exists():
        riesgo_meta = joblib.load(riesgo_meta_path)
    else:
        riesgo_meta = {}

    payload = {
        "resumen_general": resumen_general,
        "metricas_kmeans": metricas_kmeans,
        "metricas_isolation_forest": metricas_iso,
        "riesgo_meta": riesgo_meta,
        "resumen_clusters": resumen_clusters.to_dict(orient="records"),
        "resumen_riesgo": resumen_riesgo.to_dict(orient="records"),
    }

    with open(salida_resumen, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    resumen_tabla = pd.DataFrame([
        {"seccion": "general", **resumen_general}
    ])
    resumen_tabla.to_csv(salida_tabla, index=False, encoding="utf-8")

    df.to_csv(salida_operativa, index=False, encoding="utf-8")

    print("=" * 80)
    print("PASO 6 - RESUMEN FINAL Y EXPORTACION OPERATIVA")
    print("=" * 80)
    print(f"Dataset final: {data_path}")
    print(f"Filas: {resumen_general['filas']}")
    print(f"Columnas: {resumen_general['columnas']}")
    print(f"Clusters unicos: {resumen_general['clusters_unicos']}")
    print(f"Anomalias: {resumen_general['anomalias']} ({resumen_general['pct_anomalias']:.1f}%)")
    print("\nRiesgo final:")
    print(f"- bajo: {resumen_general['riesgo_bajo']}")
    print(f"- medio: {resumen_general['riesgo_medio']}")
    print(f"- alto: {resumen_general['riesgo_alto']}")
    print("\nResumen por cluster:")
    print(resumen_clusters.to_string(index=False))
    print("\nResumen por nivel de riesgo final:")
    print(resumen_riesgo.to_string(index=False))
    print("\nArchivos generados:")
    print(f"- {salida_resumen}")
    print(f"- {salida_tabla}")
    print(f"- {salida_operativa}")
    print("\nEstado: pipeline cerrado y listo para uso operativo")


if __name__ == "__main__":
    ejecutar_paso_6()
