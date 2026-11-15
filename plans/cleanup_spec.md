# Plan de Saneamiento y Arquitectura: football-moneyball (2026)

## 1. Diagnóstico de Arquitectura y Auditoría Estática (`chief-auditor`)
* **Solapamientos de Ingesta:** Coexistencia de `src/scrapers/` con `src/data/advanced_sources_hub.py`. Se unifica la arquitectura para que `src/scrapers/` opere como adaptadores específicos de formato, mientras `MasterAnalyticsDataHub` coordina todas las fuentes externas.
* **Calidad del Dato y Entity Resolution (`entity-resolver`):** Se integran `pandera` para validación estricta de esquemas y `rapidfuzz` / `splink` para emparejamiento probabilístico y resolución de alias y tildes entre FBref, Transfermarkt y Sofascore.
* **Procesamiento PySpark (`spark-warehouse`):** Se ha dotado a Spark del driver nativo PostgreSQL (`postgresql-42.6.0.jar`) con conexión directa y configuración idempotente en PostgreSQL (`analytics_scouting_metrics`).
* **Desacoplamiento de Modelos:**
  - `src/models_scouting/`: Algoritmos puros de scouting, K-NN y valoración salarial.
  - `src/models_analytics/`: Modelos matemáticos (Poisson bivariado, Dixon-Coles, SLSQP, ELO).
  - `src/models_tactical/`: Simulador táctico, escritorios de análisis y emparejamientos.
* **Monolito de UI (`ui-integrator`):** `src/app/app.py` mantiene la exclusividad de presentación gráfica mediante consumo reactivo de los servicios desacoplados.

## 2. Mapa de Propiedad Estricta de Archivos (Strict File Ownership)
1. `01-chief-auditor`: Solo lectura de arquitectura y planes en `plans/`.
2. `02-garbage-collector`: Deprecación limpia y saneamiento de dependencias en `requirements.txt`.
3. `03-ingestion-harvester`: `src/data/advanced_sources_hub.py` y `src/data/football_api_connector.py`.
4. `04-entity-resolver`: `src/data/global_player_database.py` y `src/data/real_dataset_loader.py` con `pandera` y `rapidfuzz`.
5. `05-spark-warehouse`: `src/config/db_init.py` y `src/processing/`.
6. `06-scouting-quant`: `src/models_scouting/`.
7. `07-math-oracle`: `src/models_analytics/`.
8. `08-tactical-engine`: `src/models_tactical/`.
9. `09-ui-integrator`: `src/app/app.py`.
10. `10-qa-gatekeeper`: `tests/`.
