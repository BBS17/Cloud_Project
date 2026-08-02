"""
Prevent real model loading during tests.

Injected into sys.modules before any test file is imported, so that
`from app.final_model import ...` in app.main resolves to this mock.
torch and transformers are never imported during the test run.
"""
import sys
from unittest.mock import MagicMock

_mock = MagicMock()
_mock.predict_text.return_value = {"label": "Truth", "confidence": 99.0}
_mock.is_model_loaded.return_value = True
_mock.load_model.return_value = None

sys.modules["app.final_model"] = _mock
