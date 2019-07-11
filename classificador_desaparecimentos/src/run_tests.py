import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../packages/Unidecode-1.1.1-py2.py3-none-any.whl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../packages/mock-3.0.5-py2.py3-none-any.whl')
pytest.main()