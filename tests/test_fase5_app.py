import importlib.util
from pathlib import Path
import pytest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app" / "app.py")

def test_streamlit_app_compilation():
    """
    Verifica que el script principal de Streamlit existe, 
    sus dependencias están instaladas y compila sin errores de sintaxis.
    """
    path = APP_PATH
    spec = importlib.util.spec_from_file_location("app", path)
    
    assert spec is not None, f"No se encontró el archivo de la aplicación en {path}"
    
    app_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(app_module)
    except ImportError as e:
        pytest.fail(f"Faltan dependencias para Streamlit: {e}")
    except SyntaxError as e:
        pytest.fail(f"Error de sintaxis en app.py: {e}")
    except Exception as e:
        pytest.fail(f"Error crítico al compilar la app: {e}")


def test_streamlit_app_all_modules_and_interactions():
    """
    Verifica la ejecución de los 4 macro-módulos y botones principales mediante AppTest de Streamlit,
    asegurando 0 excepciones y 0 DuplicateWidgetID.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 no disponible")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Excepción en Módulo I: {at.exception}"

    # Módulo II
    at.sidebar.radio[0].set_value("Módulo II: Dirección Deportiva & Plantillas").run()
    assert len(at.exception) == 0, f"Excepción en Módulo II: {at.exception}"

    # Módulo III
    at.sidebar.radio[0].set_value("Módulo III: Pizarra Táctica & Simulador (Digital Twin)").run()
    assert len(at.exception) == 0, f"Excepción en Módulo III: {at.exception}"

    # Probar botones de Módulo III
    for btn in at.button:
        if "Plan" in btn.label or "Digital Twin" in btn.label:
            btn.click().run()
            assert len(at.exception) == 0, f"Excepción en botón de Módulo III ({btn.label}): {at.exception}"

    # Módulo IV
    at.sidebar.radio[0].set_value("Módulo IV: Data Lakehouse & Big Data Hub").run()
    assert len(at.exception) == 0, f"Excepción en Módulo IV: {at.exception}"
