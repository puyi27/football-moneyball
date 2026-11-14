import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round, percent_rank
from pyspark.sql.window import Window

def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("Moneyball_Scouting_Metrics") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .getOrCreate()

def calculate_scouting_metrics(df_stats, df_players):
    """
    Calcula métricas por 90 minutos y percentiles por posición.
    Filtra la fiabilidad basándose en un umbral mínimo de minutos.
    """
    # Unir estadísticas con datos del jugador para obtener la posición
    df_joined = df_stats.join(df_players, on="player_id", how="inner")
    
    # 1. Normalización por 90 minutos (Evitar división por cero)
    metrics = ["goals", "assists", "xg", "xa"]
    df_p90 = df_joined
    
    for m in metrics:
        if m in df_p90.columns:
            df_p90 = df_p90.withColumn(
                f"{m}_p90",
                when(col("minutes_played") > 0, round((col(m) / col("minutes_played")) * 90, 4)).otherwise(0.0)
            )

    # 2. Evaluación de fiabilidad (Umbral: 450 minutos)
    df_p90 = df_p90.withColumn(
        "is_reliable",
        when(col("minutes_played") >= 450, True).otherwise(False)
    )
    
    # 3. Cálculo de percentiles por posición (Solo usando datos fiables para la distribución)
    # Se calculan los percentiles para comparar a los jugadores frente a sus pares
    window_spec = Window.partitionBy("position")
    
    for m in metrics:
        if f"{m}_p90" in df_p90.columns:
            # Ordenamos por la métrica. Si un jugador no es fiable, su percentil será referencial
            window_order = window_spec.orderBy(col(f"{m}_p90").asc())
            df_p90 = df_p90.withColumn(
                f"{m}_percentile",
                round(percent_rank().over(window_order) * 100, 2)
            )
            
    return df_p90

def run_metrics_pipeline(db_url, db_user, db_password):
    """Ejecuta el pipeline conectándose a PostgreSQL."""
    spark = get_spark_session()
    
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

    df_metrics = calculate_scouting_metrics(df_stats, df_players)
    
    # Escribir resultados en una tabla analítica (sobrescribe en cada ejecución batch)
    df_metrics.write.format("jdbc") \
        .option("url", db_url) \
        .option("dbtable", "analytics_scouting_metrics") \
        .option("user", db_user) \
        .option("password", db_password) \
        .option("driver", "org.postgresql.Driver") \
        .mode("overwrite") \
        .save()
        
    print("Métricas calculadas y guardadas en 'analytics_scouting_metrics'.")
    spark.stop()

if __name__ == "__main__":
    DB_URL = "jdbc:postgresql://fb_postgres:5432/moneyball"
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASSWORD", "postgres")
    run_metrics_pipeline(DB_URL, DB_USER, DB_PASS)