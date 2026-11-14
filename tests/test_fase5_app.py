import importlib.util
import pytest

def test_streamlit_app_compilation():
    """
    Verifica que el script principal de Streamlit existe, 
    sus dependencias están instaladas y compila sin errores de sintaxis.
    """
    path = "src/app/app.py"
    spec = importlib.util.spec_from_file_location("app", path)
    
    assert spec is not None, f"No se encontró el archivo de la aplicación en {path}"
    
    app_module = importlib.util.module_from_spec(spec)
    try:
        # Intenta cargar y compilar el módulo sin lanzar la interfaz
        spec.loader.exec_module(app_module)
    except ImportError as e:
        pytest.fail(f"Faltan dependencias para Streamlit: {e}")
    except SyntaxError as e:
        pytest.fail(f"Error de sintaxis en app.py: {e}")
    except Exception as e:
        pytest.fail(f"Error crítico al compilar la app: {e}")