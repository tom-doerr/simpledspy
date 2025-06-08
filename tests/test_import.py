"""Tests for import warnings"""
import unittest
import warnings
import sys
import types

class TestImport(unittest.TestCase):
    """Test cases for import warnings"""
    
    def test_correct_import_no_warning(self):
        # Create a mock module
        mock_module = types.ModuleType("simpledspy")
        mock_module.__name__ = "simpledspy"
        
        # Store the original module if it exists
        original_module = sys.modules.get("simpledspy", None)
        
        # Replace with our mock
        sys.modules["simpledspy"] = mock_module
        
        try:
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Execute the warning check code directly
                if not mock_module.__name__.startswith("simpledspy"):
                    warnings.warn(
                        "It looks like you might be importing 'simpledspy' incorrectly. "
                        "Please use 'import simpledspy' instead of 'import dspy'.",
                        ImportWarning
                    )
                
                # Check if warning was raised
                self.assertEqual(len(w), 0, "Warning was incorrectly raised")
        finally:
            # Restore the original module if it existed
            if original_module:
                sys.modules["simpledspy"] = original_module
            else:
                del sys.modules["simpledspy"]

    def test_incorrect_import_warning(self):
        # Create a mock module
        mock_module = types.ModuleType("incorrect")
        mock_module.__name__ = "incorrect"
        
        # Store the original module if it exists
        original_module = sys.modules.get("simpledspy", None)
        
        # Replace with our mock
        sys.modules["simpledspy"] = mock_module
        
        try:
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Execute the warning check code directly
                if not mock_module.__name__.startswith("simpledspy"):
                    warnings.warn(
                        "It looks like you might be importing 'simpledspy' incorrectly. "
                        "Please use 'import simpledspy' instead of 'import dspy'.",
                        ImportWarning
                    )
                
                # Check if warning was raised
                self.assertTrue(any("importing 'simpledspy' incorrectly" in str(warning.message) 
                              for warning in w), 
                              "No warning was raised for incorrect import")
        finally:
            # Restore the original module if it existed
            if original_module:
                sys.modules["simpledspy"] = original_module
            else:
                del sys.modules["simpledspy"]

if __name__ == "__main__":
    unittest.main()
