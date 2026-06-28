CREATE DATABASE IF NOT EXISTS recolecta_api;
USE recolecta_api;

CREATE TABLE IF NOT EXISTS inferencias (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    reporte TEXT NOT NULL,
    tiempo_desde_ultimo_reporte_min INTEGER,
    cluster INTEGER NOT NULL,
    distancia_centroide DOUBLE NOT NULL,
    anomalia_if INTEGER NOT NULL,
    score_anomalia_if DOUBLE NOT NULL,
    nivel_riesgo VARCHAR(16) NOT NULL,
    nivel_riesgo_final VARCHAR(16) NOT NULL,
    caracteristicas_extraidas JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inferencias_created_at ON inferencias (created_at);
CREATE INDEX idx_inferencias_cluster ON inferencias (cluster);
CREATE INDEX idx_inferencias_nivel_riesgo_final ON inferencias (nivel_riesgo_final);
