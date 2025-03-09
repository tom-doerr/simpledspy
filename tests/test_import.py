import sys
import unittest
from unittest.mock import patch
import warnings

class TestImport(unittest.TestCase):
    
    @patch.dict(sys.modules, {"simpledspy": None})
    def test_incorrect_import_warning(self):
        # Simulate incorrect import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Force __name__ to be something other than "simpledspy"
            with patch.object(sys.modules["simpledspy"], "__name__", "not_simpledspy"):
                # Re-import to trigger the warning
                import importlib
                importlib.reload(sys.modules["simpledspy"])
            
            # Check if warning was raised
            self.assertTrue(any("importing 'simpledspy' incorrectly" in str(warning.message) 
                              for warning in w))

if __name__ == "__main__":
    unittest.main()
