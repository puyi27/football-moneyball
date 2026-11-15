import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round, percent_rank
from pyspark.sql.window import Window
from src.processing.db_loader import DBLoader

def get_spark_session() -> SparkSession:
    """
    Configura e inicializa la sesión de PySpark asegurando el driver JDBC nativo
    de PostgreSQL (/usr/local/spark-3.5.0-bin-hadoop3/jars/postgresql-42.6.0.jar)
    o el paquete Maven en caso de ejecución distribuida desacoplada.
    """
    builder = SparkSession.builder.appName("Moneyball_Scouting_Metrics")
    jar_path = "/usr/local/spark-3.5.0-bin-hadoop3/jars/postgresql-42.6.0.jar"
    if os.path.exists(jar_path):
        builder = (
            builder.config("spark.jars", jar_path)
            .config("spark.driver.extraClassPath", jar_path)
            .config("spark.executor.extraClassPath", jar_path)
        )
    else:
        builder = builder.config("spark.jars.packages", "org.postgresql:postgresql:42.6.0")
    return builder.getOrCreate()

def calculate_scouting_metrics(df_stats, df_players, min_minutes_threshold=450):
    """
    Calcula métricas normalizadas por 90 minutos (p90) y percentiles por posición.
    Aplica el umbral de fiabilidad de minutos jugados (por defecto 450 min).
    
    Garantiza:
    - Normalización p90 evitando división por cero.
    - Flag booleana is_reliable según el umbral de minutos.
    - Cálculo exacto de percentiles sobre Window particionado por posición.
    """
    # Unir estadísticas con datos del jugador para obtener la posición si se proporciona df_players
    if df_players is not None:
        # Evitar columnas duplicadas en el join
        player_cols = [c for c in df_players.columns if c not in df_stats.columns or c == "player_id"]
        df_joined = df_stats.join(df_players.select(*player_cols), on="player_id", how="inner")
    else:
        df_joined = df_stats

    # Identificar columna de minutos jugados (minutes_played o minutes)
    mins_col_name = "minutes_played" if "minutes_played" in df_joined.columns else "minutes"
    mins_col = col(mins_col_name)

    # 1. Normalización por 90 minutos (Evitar división por cero)
    candidate_metrics = [
        "goals", "assists", "xg", "xa",
        "tackles", "interceptions", "shots", "passes_completed",
        "dribbles", "prog_passes"
    ]
    metrics = [m for m in candidate_metrics if m in df_joined.columns]

    df_p90 = df_joined
    for m in metrics:
        df_p90 = df_p90.withColumn(
            f"{m}_p90",
            when(mins_col > 0, round((col(m) / mins_col) * 90, 4)).otherwise(0.0)
        )

    # 2. Evaluación de fiabilidad (Umbral por defecto: 450 minutos)
    df_p90 = df_p90.withColumn(
        "is_reliable",
        when(mins_col >= min_minutes_threshold, True).otherwise(False)
    )

    # 3. Cálculo de percentiles por posición
    if "position" in df_p90.columns:
        window_spec = Window.partitionBy("position")
    else:
        window_spec = Window.partitionBy()

    for m in metrics:
        if f"{m}_p90" in df_p90.columns:
            window_order = window_spec.orderBy(col(f"{m}_p90").asc())
            df_p90 = df_p90.withColumn(
                f"{m}_percentile",
                round(percent_rank().over(window_order) * 100, 2)
            )

    return df_p90

def run_metrics_pipeline(db_url, db_user, db_password, season="2026/2027"):
    """
    Ejecuta el pipeline distribuido PySpark conectándose a PostgreSQL mediante JDBC nativo
    y realizando una carga UPSERT idempotente en la tabla 'analytics_scouting_metrics'.
    """
    spark = get_spark_session()

    try:
        # Cargar datos desde PostgreSQL
        df_stats = spark.read.format("jdbc") \
            .option("url", db_url) \
            .option("dbtable", "fact_player_stats") \
            .option("user", db_user) \
            .option("password", db_password) \
            .option("driver", "org.postgresql.Driver") \
            .load()

        df_players = spark.read.format("jdbc") \
            .option("url", db_url) \
            .option("dbtable", "dim_players") \
            .option("user", db_user) \
            .option("password", db_password) \
            .option("driver", "org.postgresql.Driver") \
            .load()

        # Calcular métricas normalizadas
        df_metrics = calculate_scouting_metrics(df_stats, df_players)

        # Carga idempotente en PostgreSQL mediante staging + UPSERT ON CONFLICT
        staging_table = "stage_analytics_scouting_metrics"
        df_metrics.write.format("jdbc") \
            .option("url", db_url) \
            .option("dbtable", staging_table) \
            .option("user", db_user) \
            .option("password", db_password) \
            .option("driver", "org.postgresql.Driver") \
            .mode("overwrite") \
            .save()

        loader = DBLoader()
        loader.upsert_from_staging(staging_table=staging_table, season=season)
        print("Métricas calculadas y consolidadas con UPSERT idempotente en 'analytics_scouting_metrics'.")
    except Exception as e:
        print(f"Error en ejecución del pipeline PySpark: {e}")
        # En caso de contingencia JDBC staging, usar DBLoader directo si df_metrics existe
        if "df_metrics" in locals():
            loader = DBLoader()
            loader.load_scouting_metrics_df(df_metrics, season=season)
        else:
            raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    DB_URL = "jdbc:postgresql://fb_postgres:5432/football_db"
    DB_USER = os.getenv("DB_USER", "moneyball")
    DB_PASS = os.getenv("DB_PASSWORD", "scouting_admin")
    run_metrics_pipeline(DB_URL, DB_USER, DB_PASS)