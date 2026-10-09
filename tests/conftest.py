import sys
import os

# Asegurar que el directorio raiz del proyecto este en sys.path para pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
