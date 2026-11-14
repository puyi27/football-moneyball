import pytest
import psycopg2
from pyspark.sql import SparkSession
import requests
import bs4

def test_conexion_postgres():
    """Prueba que el contenedor de Jupyter puede conectarse al de PostgreSQL."""
    try:
        conexion = psycopg2.connect(
            host="fb_postgres", 
            port="5432",
            database="football_db",
            user="moneyball",
            password="scouting_admin"
        )
        cursor = conexion.cursor()
        cursor.execute("SELECT 1;")
        resultado = cursor.fetchone()
        assert resultado[0] == 1, "La consulta a la base de datos no devolvió el valor esperado."
        cursor.close()
        conexion.close()
    except Exception as e:
        pytest.fail(f"Fallo crítico conectando a PostgreSQL: {e}")

def test_sesion_spark():
    """Prueba que PySpark puede levantar una sesión local y procesar datos."""
    try:
        spark = SparkSession.builder \
            .appName("Test_Moneyball") \
            .master("local[*]") \
            .getOrCreate()
        
        df_prueba = spark.createDataFrame([(1, 'A'), (2, 'B')], ["id", "valor"])
        conteo = df_prueba.count()
        
        assert conteo == 2, "Spark no procesó correctamente el DataFrame de prueba."
        spark.stop()
    except Exception as e:
        pytest.fail(f"Fallo crítico levantando la sesión de PySpark: {e}")

def test_librerias_scraping():
    """Prueba que las librerías de extracción están instaladas y accesibles."""
    assert requests.__version__ is not None, "La librería requests no está instalada."
    assert bs4.__version__ is not None, "La librería BeautifulSoup4 no está instalada."