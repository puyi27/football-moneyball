# Especificación Técnica de Saneamiento y Arquitectura: football-moneyball (2026)
**Documento de Auditoría Integral — Oleada 1**  
**Auditor Responsable:** `chief-auditor`  
**Fecha de Auditoría:** 15 de Noviembre de 2026  
**Entorno de Ejecución:** Docker Container `fb_jupyter` (Python 3.11.6, PySpark 3.5.0, PostgreSQL 16)

---

## 1. Resumen Ejecutivo y Diagnóstico Global

Tras la auditoría estática y arquitectónica integral ejecutada con **Vulture 2.16**, **Deptry 0.25.1**, **Pytest 9.1.1** y rastreos semánticos avanzados en el repositorio, se emite un dictamen de **ARQUITECTURA COMPROMETIDA POR DEUDA TÉCNICA Y DATOS MOCK**.

A pesar de que la suite de pruebas unitarias actual aprueba **71/71 tests** (28.5 s), este resultado se sustenta sobre **datos sintéticos hardcodeados**, validaciones eludidas y capas desacopladas que ocultan inconsistencias severas en producción.

### Indicadores Clave de Auditoría
* **Codigo Muerto Detectado (Vulture):** 46 símbolos y bloques no referenciados (clases completas, métodos y atributos huérfanos).
* **Dependencias Huérfanas y Desalineadas (Deptry):** 15 incidencias (7 dependencias no utilizadas en producción, 4 dependencias críticas ausentes de `requirements.txt`).
* **Pureza del Dato (ISO/IEC 25012):** Críticamente deficiente en `src/data/`. Más de **600 líneas de código con futbolistas y estadísticas inventadas** en `global_player_database.py` y `real_dataset_loader.py`.
* **Validación de Esquemas (Pandera):** Degradada. `validate_players_schema()` captura silenciosamente las excepciones y devuelve el DataFrame no conforme sin lanzar error.
* **Resolución de Entidades (RapidFuzz / Splink):** `resolve_player_entity()` está desconectado del flujo de ingesta; `splink` está instalado pero nunca importado en `src/`.
* **Violación de Contratos de Datos:** Colisión crɭtica entre `metrics_calculator.py` (PySpark) y `global_player_database.py` (Pandas) sobre la tabla de PostgreSQL `analytics_scouting_metrics` (incompatibilidad de columnas y tipos).

---


## 2. Auditoria (`vulture` & `deptry`)
### 2.1. Diagnostico de Dependencias (deptry . --known-first-party src)

#### A. Dependencias Declaradas pero Huerfanas en el Codigo (DEP002)
Las siguientes 7 librerias estan presentes en requirements.txt sin ser importadas en src/:
1. pytest>=8.0.0: Dependencia exclusiva de pruebas. Debe segregarse a requirements-dev.txt.
2. vulture>=2.11: Herramienta de analisis estatico. Debe segregarse a requirements-dev.txt.
3. deptry>=0.16.0: Herramienta de auditoria de dependencias. Debe segregarse a requirements-dev.txt.
4. html5lib>=1.1: Parser HTML legacy declarado para scrapers antiguos; BeautifulSoup en src/scrapers/fbref.py emplea html.parser.
5. splink>=4.0.0: Libreria de resolucion probabilistica de entidades declarada en Fase 8 pero sin un solo import en src/.
6. ScraperFC>=4.5.0: Declarada para scraping de SofaScore/Transfermarkt pero sin imports en el codigo productivo.
7. LanusStats>=1.5.0: Declarada para metricas sudamericanas pero sin uso activo en src/.

#### B. Dependencias de Produccion no Declaradas (DEP003 - Transitive Dependencies)
Modulos importados activamente en codigo de produccion en src/ pero ausentes de requirements.txt:
1. matplotlib: Importado en src/app/app.py:415 y src/models_scouting/player_profiler.py:342, 344.
2. mplsoccer: Importado en src/models_scouting/player_profiler.py:341.
3. pyspark: Importado en src/processing/metrics_calculator.py:2-4.
4. urllib3: Importado en src/data/advanced_sources_hub.py:538.

---

### 2.2. Diagnostico de Codigo Muerto (vulture src/)

46 elementos de codigo muerto identificados con confianza >= 60%:
- src/scrapers/fbref.py: Clase FBrefScraper (L8) y metodos get_supported_leagues (L45), fetch_league_stats (L67), parse_player_stats_html (L86), parse_defense_stats_html (L161).
- src/scrapers/football_data.py: Clase FootballDataScraper (L6) y metodos get_supported_leagues (L46), download_top5_leagues (L85), download_scouting_leagues (L89), parse_csv (L93).
- src/processing/db_loader.py: Clase DBLoader (L5) y metodos load_matches_df (L27), load_player_stats_df (L71).
- src/app/app.py: get_db_engine (L120), variables c_f2 (L356), canterano_def (L587).
- src/data/advanced_sources_hub.py: get_matches_for_competition (L68), base_xg (L180), get_league_season_matches (L387), get_player_market_value_evolution (L493), calculate_amortized_cost (L590), get_sample_club_payroll (L603), atributos openfootball (L636), capology (L639), get_system_health (L642).
- src/data/football_api_connector.py: test_api_football_connection (L44), test_football_data_connection (L88), search_live_player (L127), fetch_competition_teams (L169), normalize_api_player_to_schema (L216).
- src/data/global_player_database.py: resolve_player_entity (L42).
- src/models_analytics/: update_ratings (elo_model.py:18), calculate_metrics (evaluation.py:5), total_minutes (team_profiler.py:52).
- src/models_scouting/: get_academy_prospects (academy_scout.py:81), recommend_career_pathway (L127), compare_academy_vs_market (L286), generate_pitch_heatmap (player_profiler.py:206), calculate_moneyball_bargains (L312), ax (L401), find_similar_players (recommender.py:39).
- src/models_tactical/: simulate_situational_match (match_scenario_simulator.py:21), recommender (tactical_analyst_desk.py:39), audit_club_squad (L51), scout_transfer_targets (L74), simulate_transfer_scenario (L119).

---

## 3. Auditoria de Solapamientos e Ingesta Duplicada

### 3.1. Dicotomia src/scrapers/ vs src/data/advanced_sources_hub.py
Se constata una fractura arquitectonica entre dos generaciones del proyecto:
1. Generacion Fase 2 (src/scrapers/):
   - fbref.py: Scraping HTML directo via requests y BeautifulSoup. Inestable ante bloqueos Cloudflare (HTTP 429).
   - football_data.py: Descarga de CSVs estaticos de football-data.co.uk.
   - Ambos scrapers estan aislados y solo subsisten para alimentar tests/test_fase2_scrapers.py.
2. Generacion Fase 8/9 (src/data/advanced_sources_hub.py):
   - Emplea soccerdata.FBref con caching estructurado en disco y cabeceras respetuosas.
   - Posee adaptadores para StatsBomb, Understat, Kloppy, OpenFootball y FPL.
   - Incoherencia: La interfaz Streamlit (app.py) no consulta src/scrapers/ ni aprovecha el scraping real de advanced_sources_hub.py, sino que consume directamente global_player_database.py.

### 3.2. Redundancia en Ingesta de Partidos y Resultados
Existen 3 canales desconectados para obtener resultados deportivos:
- Canal A: FootballDataScraper.parse_csv() en src/scrapers/football_data.py (cuotas Bet365 y resultados).
- Canal B: OpenFootballFeed.get_league_season_matches() en src/data/advanced_sources_hub.py (JSON de GitHub).
- Canal C: FootballAPIConnector.fetch_competition_teams() en src/data/football_api_connector.py (REST API football-data.org).

---

## 4. Auditoria de Pureza del Dato y Validacion ISO/IEC 25012

### 4.1. Deteccion de Datos Sinteticos y Diccionarios Hardcodeados

#### A. src/data/global_player_database.py
- Defecto Critico: generate_worldwide_player_network() y generate_all_seasons_network() acumulan mas de 650 lineas de inserciones manuales add_p().
- Aunque los nombres son reales (Bade, Lukebakio, Vinicius), las metricas p90 asociadas (xg: 0.05, xa: 0.02, interc: 2.45, tackles: 2.85) son valores inventados a mano.
- La evolucion multitemporada (2024/2025 -> 2025/2026 -> 2026/2027) aplica multiplicadores heuristicos ficticios (+25% valor si edad <= 21, -30% si edad >= 32).

#### B. src/data/real_dataset_loader.py
- generate_curated_real_dataset() contiene 144 lineas con 120 diccionarios de jugadores (player_id: 101...). Duplica de forma desincronizada a global_player_database.py.

#### C. src/data/advanced_sources_hub.py
- L107-123 (StatsBombEngine): Fallback sintetico con np.random.seed(match_id % 1000) que genera 1,500 eventos inventados.
- L163-228 (UnderstatShotEngine): Fallback con np.random.seed(42) que genera disparos simulados para 10 jugadores.
- L454-490 (TransfermarktDatasetHub): Diccionario estatico de jugadores en fin de contrato.
- L493-511: Curvas de valor de mercado simuladas con bifurcaciones if/else sobre nombres.
- L568-575 (FPLEngine): Lista estatica de 5 futbolistas con indices ICT simulados.
- L605-620 (CapologyFinancialEngine): Nominas y amortizaciones estaticas.

#### D. src/models_scouting/player_profiler.py
- L161-203 (get_pitch_heatmap_data): Genera coordenadas (x, y) sinteticas mediante distribuciones normales (np.random.normal) con semilla hash del nombre.

### 4.2. Evaluacion contra Estandar ISO/IEC 25012
- Exactitud Sintactica: ACEPTABLE (tipos conformes con pandas/postgres).
- Exactitud Semantica: CRITICA (metricas inventadas, no reflejan observaciones empiricas de partidos).
- Completitud: DEFICIENTE (sin profundidad en ligas internacionales).
- Consistencia: INSUFICIENTE (Spark minutes_played vs Pandas minutes).
- Actualidad: COMPROMETIDA (fichajes y lesiones no se actualizan en tiempo real).
- Cumplimiento Normativo: DEFICIENTE (Pandera silencia excepciones en validate_players_schema con try/except).

### 4.3. Entity Resolution (RapidFuzz / Splink)
- resolve_player_entity() en global_player_database.py funciona en unit tests pero esta DESCONECTADO de los pipelines de ingesta.
- splink esta en requirements.txt pero con 0 imports en todo el repositorio.

---

## 5. Violaciones de Contratos de Datos y Esquemas

### 5.1. Colision Destructiva en PostgreSQL (analytics_scouting_metrics)
Existe una carrera de escritura destructiva entre dos subsistemas:
- Motor PySpark (src/processing/metrics_calculator.py):
  Lee fact_player_stats y dim_players, calcula percentiles posicionales y vuelca a PostgreSQL via JDBC con .mode('overwrite').
  Columnas resultantes: player_id, minutes_played, goals, assists, xg, xa, position, goals_p90, assists_p90, xg_p90, xa_p90, is_reliable, percentiles.
  Campos omitidos: player_name, team_name, league, age, market_value_m, season.
- Motor Pandas (src/data/global_player_database.py):
  Ejecuta to_sql('analytics_scouting_metrics', engine, if_exists='replace').
  Columnas resultantes: player_id, season, player_name, team_name, league, position, age, market_value_m, minutes, xg_p90, xa_p90, goals_p90, assists_p90, interceptions_p90, tackles_p90, dribbles_p90, prog_passes_p90, is_reliable, potential_ovr, is_academy, academy_origin.
- Impacto: Si corre PySpark, destruye las columnas necesarias para la UI y los modelos de scouting (age, market_value_m, player_name). Si corre Pandas, destruye los percentiles posicionales de Spark y cambia minutes_played por minutes.

---

## 6. Plan de Saneamiento y Asignacion de Tareas por Subagente

### Tareas Asignadas a 02-garbage-collector
Ambito de Escritura Exclusivo: requirements.txt, src/scrapers/, limpieza de imports no usados.
1. Saneamiento de requirements.txt:
   - Purgar dependencias dev: pytest, vulture, deptry.
   - Purgar dependencias huerfanas: html5lib, splink, ScraperFC, LanusStats.
   - Declarar formalmente dependencias requeridas: matplotlib, mplsoccer, urllib3.
2. Refactorizacion Limpia de Scrapers Legacy:
   - Convertir src/scrapers/fbref.py en fachada delegada a SoccerDataHub de src/data/advanced_sources_hub.py, manteniendo compatibilidad con tests/test_fase2_scrapers.py.
   - Preservar en src/scrapers/football_data.py unicamente el parser de cuotas y resultados para fact_matches.
3. Limpieza de Metodos Muertos:
   - Purgar funciones y variables muertas detectadas por Vulture en src/processing/db_loader.py y src/app/app.py (get_db_engine, c_f2, canterano_def).

---

### Tareas Asignadas a 03-ingestion-harvester
Ambito de Escritura Exclusivo: src/data/advanced_sources_hub.py, src/data/football_api_connector.py.
1. Eliminacion de Semillas y Fallbacks Sinteticos:
   - Retirar np.random.seed y la generacion simulada en StatsBombEngine y UnderstatShotEngine.
   - Implementar persistencia offline con serializacion Parquet/CSV en data/cache_sources/ con datos reales descargados de StatsBomb Open Data y soccerdata.
2. Robustez de Red y Gestion de Rate Limits:
   - Garantizar que ante caidas de red o HTTP 429, el hub consuma el snapshot local cacheado sin inventar datos sinteticos.
3. Unificacion de Calendarios:
   - Consolidar la ingesta de partidos y calendarios en un unico pipeline desacoplado.

---

### Tareas Asignadas a 04-entity-resolver
Ambito de Escritura Exclusivo: src/data/global_player_database.py, src/data/real_dataset_loader.py.
1. Erradicacion de Diccionarios Hardcodeados:
   - Purgar las 650+ lineas de inserciones estaticas add_p() en generate_worldwide_player_network().
   - Configurar load_global_players() para nutrirse de la tabla PostgreSQL o de un archivo consolidado generado por ingesta real.
   - Eliminar generate_curated_real_dataset() en real_dataset_loader.py y reorientar load_real_players_df() al almacen unificado.
2. Validacion Pandera Estricta (ISO/IEC 25012):
   - Modificar validate_players_schema() para ejecutar validacion estricta sin silenciar excepciones, e incorporar rangos para xg_p90, xa_p90 y market_value_m.
3. Conexion Operativa de RapidFuzz:
   - Integrar resolve_player_entity() como paso obligatorio en la ingesta para consolidar futbolistas entre FBref, Transfermarkt y APIs.

---

### Tareas Asignadas a 05-spark-warehouse
Ambito de Escritura Exclusivo: src/config/db_init.py, src/processing/db_loader.py, src/processing/metrics_calculator.py.
1. Unificacion del DDL en PostgreSQL:
   - Anadir formalmente analytics_scouting_metrics al DDL en src/config/db_init.py con clave primaria compuesta (player_id, season).
   - Estandarizar nombres de columnas: unificar minutes vs minutes_played y tipos numericos.
2. Carga Idempotente con UPSERT:
   - Reemplazar if_exists='replace' y mode('overwrite') por ON CONFLICT (player_id, season) DO UPDATE.
3. Pipeline PySpark:
   - Preservar dimensiones descriptivas (player_name, team_name, market_value_m, age) uniendo fact_player_stats con dim_players y dim_teams.

---

### Tareas Asignadas a 06-scouting-quant, 07-math-oracle, 08-tactical-engine, 09-ui-integrator
- Scouting Quant: Conectar mapas de calor con coordenadas reales de eventos estandarizados por Kloppy.
- Math Oracle & Tactical Engine: Purgar metodos duplicados identificados por Vulture (update_ratings, audit_club_squad, recommender).
- UI Integrator: Consumir datos exclusivamente a traves de MasterAnalyticsDataHub o load_global_players().
- QA Gatekeeper: Asegurar que tras las limpiezas la suite de pytest complete continue en verde (71/71 passed).

---

## 7. Matriz de Control y Reglas de No Regresion

| Regla de Control | Requisito Obligatorio | Verificacion Automatizada |
| :--- | :--- | :--- |
| Cero Codigo Muerto | Ningun archivo nuevo o modificado puede contener clases o metodos no referenciados. | python -m vulture src/ --min-confidence 80 |
| Pureza de Dependencias | Todas las librerias importadas deben estar en requirements.txt. | python -m deptry . --known-first-party src |
| Cero Datos Sinteticos en Ingesta | Prohibido el uso de np.random y listas de futbolistas hardcodeados en src/data/. | grep -rn 'np.random\|seed' src/data/ devuelve 0 |
| Contrato Pandera Activo | Las excepciones de validacion de esquema deben lanzarse en caso de anomalia. | Tests de validacion con datos no conformes |
| No Regresion en Pruebas | Los 71 tests existentes deben mantenerse en aprobado. | pytest tests/ (71 passed) |
