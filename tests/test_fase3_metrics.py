import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from src.processing.metrics_calculator import calculate_scouting_metrics

@pytest.fixture(scope="module")
def spark():
    """Crea una sesión de Spark local para los tests."""
    return SparkSession.builder \
        .appName("Test_Scouting_Metrics") \
        .master("local[1]") \
        .getOrCreate()

def test_calculate_scouting_metrics(spark):
    """Verifica la normalización p90, umbral de fiabilidad y percentiles."""
    
    # 1. Preparar datos simulados (Mock Data)
    schema_players = StructType([
        StructField("player_id", IntegerType(), True),
        StructField("position", StringType(), True)
    ])
    data_players = [
        (1, "Forward"),  # Jugador fiable y goleador
        (2, "Forward"),  # Jugador poco fiable (pocos minutos)
        (3, "Defender")  # Jugador en otra posición
    ]
    df_players = spark.createDataFrame(data_players, schema_players)

    schema_stats = StructType([
        StructField("player_id", IntegerType(), True),
        StructField("minutes_played", IntegerType(), True),
        StructField("goals", IntegerType(), True),
        StructField("assists", IntegerType(), True),
        StructField("xg", DoubleType(), True),
        StructField("xa", DoubleType(), True)
    ])
    data_stats = [
        (1, 900, 5, 2, 4.5, 1.2),  # 900 mins (Fiable). 5 goles en 900 min = 0.5 goles p90
        (2, 90, 1, 0, 0.5, 0.0),   # 90 mins (NO fiable). 1 gol en 90 min = 1.0 goles p90
        (3, 900, 0, 1, 0.1, 0.5)   # 900 mins (Fiable).
    ]
    df_stats = spark.createDataFrame(data_stats, schema_stats)

    # 2. Ejecutar la transformación
    df_result = calculate_scouting_metrics(df_stats, df_players).collect()

    # Convertir a diccionario para facilitar validaciones
    results = {row["player_id"]: row for row in df_result}

    # 3. Validaciones (Asserts)
    
    # Comprobar jugador 1 (Fiable y cálculos p90)
    assert results[1]["is_reliable"] is True, "Jugador con 900 mins debería ser fiable"
    assert results[1]["goals_p90"] == 0.5, "Error en normalización de goles por 90 min"
    
    # Comprobar jugador 2 (No fiable, pero p90 alto)
    assert results[2]["is_reliable"] is False, "Jugador con 90 mins no debería ser fiable"
    assert results[2]["goals_p90"] == 1.0, "Error en normalización de goles por 90 min"
    
    # Comprobar percentiles particionados por posición (Forwards)
    # El Jugador 2 tiene 1.0 goles p90, el Jugador 1 tiene 0.5 goles p90.
    # Por tanto, Jugador 2 debe tener un percentil mayor (100.0) que el Jugador 1 (0.0).
    assert results[2]["goals_percentile"] == 100.0, "Error en cálculo de percentil máximo"
    assert results[1]["goals_percentile"] == 0.0, "Error en cálculo de percentil mínimo"