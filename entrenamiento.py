from pathlib import Path
import importlib.util


BASE_DIR = Path(__file__).resolve().parent
PASOS_DIR = BASE_DIR / "entrenamienot"


def cargar_modulo(nombre_archivo, nombre_modulo):
    ruta = PASOS_DIR / nombre_archivo
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modulo: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ejecutar_entrenamiento():
    paso_1 = cargar_modulo("01_paso_objetivo.py", "paso_1")
    paso_2 = cargar_modulo("02_paso_vectorizacion_escalado.py", "paso_2")
    paso_3 = cargar_modulo("03_paso_entrenar_kmeans.py", "paso_3")
    paso_4 = cargar_modulo("04_paso_clusters_riesgo.py", "paso_4")
    paso_5 = cargar_modulo("05_paso_isolation_forest.py", "paso_5")
    paso_6 = cargar_modulo("06_paso_resumen_final.py", "paso_6")

    print("=" * 80)
    print("ENTRENAMIENTO COMPLETO")
    print("=" * 80)

    paso_1.ejecutar_paso_1()
    paso_2.ejecutar_paso_2()
    paso_3.ejecutar_paso_3()
    paso_4.ejecutar_paso_4()
    paso_5.ejecutar_paso_5()
    paso_6.ejecutar_paso_6()

    print("=" * 80)
    print("FIN DEL ENTRENAMIENTO COMPLETO")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_entrenamiento()
