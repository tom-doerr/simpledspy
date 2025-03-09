import sys
import unittest
import warnings
import importlib

# Import simpledspy to make it available in sys.modules
import simpledspy

class TestImport(unittest.TestCase):
    
    def test_incorrect_import_warning(self):
        # Simulate incorrect import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Save original name
            original_name = simpledspy.__name__
            
            try:
                # Temporarily change the module name to simulate incorrect import
                simpledspy.__name__ = "not_simpledspy"
                
                # Re-import to trigger the warning
                importlib.reload(simpledspy)
                
                # Check if warning was raised
                self.assertTrue(any("importing 'simpledspy' incorrectly" in str(warning.message) 
                                for warning in w), 
                                "No warning was raised for incorrect import")
            finally:
                # Restore original name
                simpledspy.__name__ = original_name

if __name__ == "__main__":
    unittest.main()
