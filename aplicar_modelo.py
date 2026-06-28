from pathlib import Path
import argparse
import json
import ast
import re
import joblib
import numpy as np
import pandas as pd


def cargar_artefactos(base_dir):
    artefactos_dir = base_dir / "entrenamienot" / "artefactos"
    scaler = joblib.load(artefactos_dir / "scaler_paso2.joblib")
    features = joblib.load(artefactos_dir / "features_paso2.joblib")
    weights = np.asarray(joblib.load(artefactos_dir / "weights_paso2.joblib"), dtype=float)
    kmeans = joblib.load(artefactos_dir / "kmeans_paso3.joblib")
    isolation = joblib.load(artefactos_dir / "isolation_forest_paso5.joblib")
    riesgo_meta = joblib.load(artefactos_dir / "riesgo_paso4.joblib")
    return scaler, features, weights, kmeans, isolation, riesgo_meta


def transformar_entrada(df, features, scaler, weights):
    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nulos = int(df[features].isnull().sum().sum())
    if nulos > 0:
        raise ValueError(f"Hay nulos en columnas del modelo: {nulos}")

    X = df[features].values
    X_scaled = scaler.transform(X)
    X_modelo = X_scaled * weights
    return df, X_modelo


def extraer_caracteristicas_reporte(texto, tiempo_desde_ultimo_reporte_min=None):
    texto = str(texto or "")
    tokens = re.findall(r"\b\w+\b", texto.lower(), flags=re.UNICODE)
    palabras = len(tokens)
    palabras_repetidas = max(0, palabras - len(set(tokens)))
    caracteres_especiales = sum(1 for ch in texto if not ch.isalnum() and not ch.isspace())
    caracteres_especiales_repitidos = sum(len(m.group(0)) - 1 for m in re.finditer(r"([^\w\s])\1+", texto))
    cantidad_numeros = sum(ch.isdigit() for ch in texto)
    groserias = sum(1 for t in tokens if t in {"malo", "horrible", "asco", "tonto", "idiota", "estupido", "mierda", "puta", "puto", "pendejo", "menso", "culero"})
    contiene_url = int(bool(re.search(r"https?://|www\.|\.com|\.net|\.org", texto, flags=re.IGNORECASE)))

    palabras_clave_diccionario = {
        "basura", "bolsas", "costales", "contenedor", "contenedores", "recoleccion", "recolección",
        "camion", "camión", "unidad", "compactador", "tolva", "ruta", "recorrido", "servicio",
        "carga", "descarga", "averia", "avería", "descompuesto", "descompuso", "falla", "fallando",
        "motor", "bateria", "batería", "radiador", "frenos", "embrague", "aceite", "llanta",
        "ponchada", "ponchado", "diesel", "diésel", "calentamiento", "calle", "avenida", "boulevard",
        "boulevar", "privada", "andador", "esquina", "crucero", "colonia", "barrio", "fraccionamiento",
        "mercado", "tianguis", "parque", "escuela", "trafico", "tráfico", "accidente", "choque",
        "bloqueado", "bloqueada", "cerrada", "cierre", "desvio", "desvío", "manifestacion",
        "manifestación", "protesta", "arbol", "árbol", "rama", "ramas", "poste", "cables", "cable",
        "escombro", "escombros", "bache", "socavon", "socavón", "inundacion", "inundación", "charco",
        "lodo", "tierra", "carro", "auto", "camioneta", "trailer", "tráiler", "vehiculo", "vehículo",
        "estacionado", "obstruye", "obstruido", "lluvia", "lloviendo", "tormenta", "granizo", "viento",
        "neblina", "vecinos", "ciudadanos", "persona", "personas", "policia", "policía", "transito",
        "tránsito", "recoger", "recogiendo", "recogio", "recogió", "levantar", "levanto", "levantó",
        "cargar", "cargando", "vaciar", "limpiar", "limpieza", "pasar", "entrar", "salir", "esperando",
        "detenido", "retraso", "apoyo", "lleno", "desbordado", "tirada", "regada", "perro", "animales",
        "fiesta", "evento"
    }
    palabras_claves = sum(1 for t in tokens if t in palabras_clave_diccionario)

    if tiempo_desde_ultimo_reporte_min is None:
        tiempo_desde_ultimo_reporte_min = 0

    porcentaje_groserias = (groserias / palabras * 100) if palabras > 0 else 0.0

    return {
        "groserias": groserias,
        "palabras_repetidas": palabras_repetidas,
        "caracteres_especiales": caracteres_especiales,
        "caracteres_especiales_repitidos": caracteres_especiales_repitidos,
        "cantidad_numeros": cantidad_numeros,
        "palabras": palabras,
        "tiempo_desde_ultimo_reporte_min": tiempo_desde_ultimo_reporte_min,
        "contiene_url": contiene_url,
        "porcentaje_groserias": porcentaje_groserias,
        "palabras_claves": palabras_claves,
    }


def agregar_porcentaje_groserias(df):
    if "porcentaje_groserias" not in df.columns and {"groserias", "palabras"}.issubset(df.columns):
        g = pd.to_numeric(df["groserias"], errors="coerce")
        p = pd.to_numeric(df["palabras"], errors="coerce").replace(0, np.nan)
        df["porcentaje_groserias"] = (g / p * 100).fillna(0.0)
    return df


def preparar_df_desde_reporte(texto_reporte, tiempo_desde_ultimo_reporte_min=None):
    return pd.DataFrame([
        extraer_caracteristicas_reporte(texto_reporte, tiempo_desde_ultimo_reporte_min)
    ])


def cargar_entrada(args):
    if args.record_file:
        with open(args.record_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame([data])

    if args.record:
        try:
            data = json.loads(args.record)
        except json.JSONDecodeError:
            data = ast.literal_eval(args.record)
        return pd.DataFrame([data])

    if args.payload:
        try:
            data = json.loads(args.payload)
        except json.JSONDecodeError:
            data = ast.literal_eval(args.payload)
        return pd.DataFrame([data])

    return pd.read_csv(args.input)


def aplicar_modelo(input_path, output_path):
    base_dir = Path(__file__).resolve().parent
    scaler, features, weights, kmeans, isolation, riesgo_meta = cargar_artefactos(base_dir)

    df = pd.read_csv(input_path)
    df = agregar_porcentaje_groserias(df)

    df, X_modelo = transformar_entrada(df, features, scaler, weights)

    clusters = kmeans.predict(X_modelo)
    distancias = kmeans.transform(X_modelo).min(axis=1)
    anomalias = isolation.predict(X_modelo)
    scores_if = isolation.decision_function(X_modelo)

    p90 = float(riesgo_meta["p90_distancia"])
    p97 = float(riesgo_meta["p97_distancia"])

    nivel_riesgo = np.where(distancias >= p97, "alto", np.where(distancias >= p90, "medio", "bajo"))
    nivel_riesgo_final = np.where((anomalias == -1) & (distancias >= p97), "alto", np.where((anomalias == -1) | (distancias >= p90), "medio", "bajo"))

    df["cluster"] = clusters
    df["distancia_centroide"] = distancias
    df["anomalia_if"] = (anomalias == -1).astype(int)
    df["score_anomalia_if"] = scores_if
    df["nivel_riesgo"] = nivel_riesgo
    df["nivel_riesgo_final"] = nivel_riesgo_final

    df.to_csv(output_path, index=False, encoding="utf-8")

    print("=" * 80)
    print("APLICACION DEL MODELO")
    print("=" * 80)
    print(f"Entrada: {input_path}")
    print(f"Salida: {output_path}")
    print(f"Filas procesadas: {len(df)}")
    print(f"Clusters unicos: {df['cluster'].nunique()}")
    print(f"Anomalias IF: {int(df['anomalia_if'].sum())}")
    print("\nDistribucion nivel_riesgo_final:")
    print(df["nivel_riesgo_final"].value_counts().to_string())
    print("=" * 80)


def aplicar_modelo_reporte(texto_reporte, tiempo_desde_ultimo_reporte_min=None):
    base_dir = Path(__file__).resolve().parent
    scaler, features, weights, kmeans, isolation, riesgo_meta = cargar_artefactos(base_dir)

    df = preparar_df_desde_reporte(texto_reporte, tiempo_desde_ultimo_reporte_min)
    df = agregar_porcentaje_groserias(df)

    df, X_modelo = transformar_entrada(df, features, scaler, weights)

    cluster = int(kmeans.predict(X_modelo)[0])
    distancia = float(kmeans.transform(X_modelo).min(axis=1)[0])
    anomalia_if = int(isolation.predict(X_modelo)[0] == -1)
    score_if = float(isolation.decision_function(X_modelo)[0])

    p90 = float(riesgo_meta["p90_distancia"])
    p97 = float(riesgo_meta["p97_distancia"])
    nivel_riesgo = "alto" if distancia >= p97 else "medio" if distancia >= p90 else "bajo"

    nivel_riesgo_final = nivel_riesgo
    if anomalia_if == 1 and distancia >= p97:
        nivel_riesgo_final = "alto"
    elif anomalia_if == 1 or distancia >= p90:
        nivel_riesgo_final = "medio"

    resultado = {
        "caracteristicas_extraidas": df.iloc[0].to_dict(),
        "cluster": cluster,
        "distancia_centroide": distancia,
        "anomalia_if": anomalia_if,
        "score_anomalia_if": score_if,
        "nivel_riesgo": nivel_riesgo,
        "nivel_riesgo_final": nivel_riesgo_final,
    }

    print("=" * 80)
    print("APLICACION DE UN SOLO REPORTE")
    print("=" * 80)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("=" * 80)

    return resultado


def aplicar_modelo_payload(payload_json):
    base_dir = Path(__file__).resolve().parent
    scaler, features, weights, kmeans, isolation, riesgo_meta = cargar_artefactos(base_dir)

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        payload = ast.literal_eval(payload_json)

    if "reporte" not in payload:
        raise ValueError('El payload debe incluir la clave "reporte"')

    texto_reporte = payload.get("reporte", "")
    tiempo = payload.get("tiempo")

    df = preparar_df_desde_reporte(texto_reporte, tiempo)
    df = agregar_porcentaje_groserias(df)

    df, X_modelo = transformar_entrada(df, features, scaler, weights)

    cluster = int(kmeans.predict(X_modelo)[0])
    distancia = float(kmeans.transform(X_modelo).min(axis=1)[0])
    anomalia_if = int(isolation.predict(X_modelo)[0] == -1)
    score_if = float(isolation.decision_function(X_modelo)[0])

    p90 = float(riesgo_meta["p90_distancia"])
    p97 = float(riesgo_meta["p97_distancia"])
    nivel_riesgo = "alto" if distancia >= p97 else "medio" if distancia >= p90 else "bajo"

    nivel_riesgo_final = nivel_riesgo
    if anomalia_if == 1 and distancia >= p97:
        nivel_riesgo_final = "alto"
    elif anomalia_if == 1 or distancia >= p90:
        nivel_riesgo_final = "medio"

    resultado = {
        "caracteristicas_extraidas": df.iloc[0].to_dict(),
        "cluster": cluster,
        "distancia_centroide": distancia,
        "anomalia_if": anomalia_if,
        "score_anomalia_if": score_if,
        "nivel_riesgo": nivel_riesgo,
        "nivel_riesgo_final": nivel_riesgo_final,
    }

    print("=" * 80)
    print("APLICACION DE UN SOLO REPORTE")
    print("=" * 80)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("=" * 80)

    return resultado


def aplicar_modelo_entrada_unica(record_json):
    base_dir = Path(__file__).resolve().parent
    scaler, features, weights, kmeans, isolation, riesgo_meta = cargar_artefactos(base_dir)

    try:
        data = json.loads(record_json)
    except json.JSONDecodeError:
        data = ast.literal_eval(record_json)

    df = pd.DataFrame([data])
    df = agregar_porcentaje_groserias(df)

    df, X_modelo = transformar_entrada(df, features, scaler, weights)

    cluster = int(kmeans.predict(X_modelo)[0])
    distancia = float(kmeans.transform(X_modelo).min(axis=1)[0])
    anomalia_if = int(isolation.predict(X_modelo)[0] == -1)
    score_if = float(isolation.decision_function(X_modelo)[0])

    p90 = float(riesgo_meta["p90_distancia"])
    p97 = float(riesgo_meta["p97_distancia"])
    nivel_riesgo = "alto" if distancia >= p97 else "medio" if distancia >= p90 else "bajo"

    nivel_riesgo_final = nivel_riesgo
    if anomalia_if == 1 and distancia >= p97:
        nivel_riesgo_final = "alto"
    elif anomalia_if == 1 or distancia >= p90:
        nivel_riesgo_final = "medio"

    resultado = {
        "cluster": cluster,
        "distancia_centroide": distancia,
        "anomalia_if": anomalia_if,
        "score_anomalia_if": score_if,
        "nivel_riesgo": nivel_riesgo,
        "nivel_riesgo_final": nivel_riesgo_final,
    }

    print("=" * 80)
    print("APLICACION DE UN SOLO REPORTE")
    print("=" * 80)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("=" * 80)

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent / "data" / "csv_dataset_new.csv"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "data" / "dataset_reportes_aplicado.csv"))
    parser.add_argument("--record", help="JSON con un solo reporte para inferencia puntual")
    parser.add_argument("--record-file", help="Ruta a un archivo JSON con un solo reporte")
    parser.add_argument("--payload", help="JSON con {reporte, tiempo} para inferencia puntual")
    parser.add_argument("--payload-file", help="Ruta a un archivo JSON con {reporte, tiempo}")
    parser.add_argument("--reporte", help="Texto completo del reporte para inferencia puntual")
    parser.add_argument("--tiempo", type=int, default=None, help="Minutos desde el ultimo reporte, opcional")
    args = parser.parse_args()

    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = f.read()
        aplicar_modelo_payload(payload)
        raise SystemExit(0)

    if args.payload:
        aplicar_modelo_payload(args.payload)
        raise SystemExit(0)

    if args.reporte:
        aplicar_modelo_reporte(args.reporte, args.tiempo)
        raise SystemExit(0)

    if args.record_file:
        df = cargar_entrada(args)
        df = agregar_porcentaje_groserias(df)
        base_dir = Path(__file__).resolve().parent
        scaler, features, weights, kmeans, isolation, riesgo_meta = cargar_artefactos(base_dir)

        df, X_modelo = transformar_entrada(df, features, scaler, weights)

        clusters = kmeans.predict(X_modelo)
        distancias = kmeans.transform(X_modelo).min(axis=1)
        anomalias = isolation.predict(X_modelo)
        scores_if = isolation.decision_function(X_modelo)

        p90 = float(riesgo_meta["p90_distancia"])
        p97 = float(riesgo_meta["p97_distancia"])
        nivel_riesgo = np.where(distancias >= p97, "alto", np.where(distancias >= p90, "medio", "bajo"))
        nivel_riesgo_final = np.where((anomalias == -1) & (distancias >= p97), "alto", np.where((anomalias == -1) | (distancias >= p90), "medio", "bajo"))

        df["cluster"] = clusters
        df["distancia_centroide"] = distancias
        df["anomalia_if"] = (anomalias == -1).astype(int)
        df["score_anomalia_if"] = scores_if
        df["nivel_riesgo"] = nivel_riesgo
        df["nivel_riesgo_final"] = nivel_riesgo_final

        print("=" * 80)
        print("APLICACION DE UN SOLO REPORTE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        raise SystemExit(0)

    if args.record:
        aplicar_modelo_entrada_unica(args.record)
        raise SystemExit(0)

    aplicar_modelo(Path(args.input), Path(args.output))
