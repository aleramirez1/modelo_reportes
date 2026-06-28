# Recolecta API

API en capas para ejecutar inferencias no supervisadas y consultar el historial.

## Endpoints

- `POST /infer` realiza una inferencia sobre un reporte.
- `GET /inferences` lista inferencias realizadas con paginación.
- `GET /health` verifica que la API esté viva.

## Swagger

La documentación automática está disponible en:

- `/docs`
- `/redoc`
- `/openapi.json`

## OpenAPI

La API expone especificacion OpenAPI con:

- metadata de la aplicacion
- tags por grupo de endpoints
- ejemplos de request y response
- descripciones de parametros y respuestas

Esto alimenta directamente Swagger UI y ReDoc.

## Estructura

- `api/core`: configuración global.
- `api/db`: engine, sesión y `Base` de SQLAlchemy.
- `api/models`: entidades ORM.
- `api/schemas`: contratos de entrada y salida.
- `api/repositories`: acceso a datos.
- `api/services`: lógica de inferencia y orquestación.
- `api/routers`: endpoints HTTP.
- `api/sql`: script SQL para crear la base y tablas.

## Ejecución local

1. Instala dependencias:

```bash
pip install -r api/requirements.txt
```

2. Levanta la API:

```bash
uvicorn api.main:app --reload
```

## Variables de entorno

- `DATABASE_URL`: opcional. Si no se define, se usa MySQL en `mysql+pymysql://root:root@localhost:3306/recolecta_api`.

## MySQL

Si quieres usar tu propia instancia, define `DATABASE_URL` con el formato:

```text
mysql+pymysql://usuario:password@host:3306/recolecta_api
```

Antes de arrancar la API, crea la base de datos ejecutando el script en `api/sql/schema.sql`.
