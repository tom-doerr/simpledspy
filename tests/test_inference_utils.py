"""Comprehensive tests for inference_utils.py"""

import pytest
import ast
import inspect
from unittest.mock import Mock, patch
from simpledspy.inference_utils import InferenceUtils
from simpledspy.exceptions import SecurityError


class TestASTDepthCalculation:
    """Test AST depth calculation"""
    
    def test_get_ast_depth_simple(self):
        """Test depth calculation for simple AST"""
        tree = ast.parse("x = 1")
        depth = InferenceUtils.get_ast_depth(tree)
        assert depth == 3  # Module -> Assign -> Name/Constant
    
    def test_get_ast_depth_nested(self):
        """Test depth calculation for nested AST"""
        tree = ast.parse("f(g(h(x)))")
        depth = InferenceUtils.get_ast_depth(tree)
        assert depth >= 5  # Module -> Expr -> Call -> Call -> Call
    
    def test_get_ast_depth_non_ast_node(self):
        """Test depth calculation with non-AST node"""
        depth = InferenceUtils.get_ast_depth("not an AST", 5)
        assert depth == 5


class TestSafeParseAST:
    """Test safe AST parsing with security checks"""
    
    def test_safe_parse_ast_valid_code(self):
        """Test parsing valid code"""
        code = "x = predict('test')"
        tree = InferenceUtils.safe_parse_ast(code)
        assert isinstance(tree, ast.AST)
    
    def test_safe_parse_ast_code_too_long(self):
        """Test rejection of overly long code"""
        code = "x" * 1001
        with pytest.raises(SecurityError) as exc_info:
            InferenceUtils.safe_parse_ast(code)
        assert "Code too long" in str(exc_info.value)
    
    def test_safe_parse_ast_too_deep(self):
        """Test rejection of deeply nested AST"""
        # Create deeply nested code
        code = "f(" * 15 + "x" + ")" * 15
        with pytest.raises(SecurityError) as exc_info:
            InferenceUtils.safe_parse_ast(code, max_depth=10)
        assert "AST too deep" in str(exc_info.value)
    
    def test_safe_parse_ast_syntax_error(self):
        """Test handling of syntax errors"""
        code = "x = = 1"  # Invalid syntax
        with pytest.raises(SecurityError) as exc_info:
            InferenceUtils.safe_parse_ast(code)
        assert "Failed to parse AST" in str(exc_info.value)


class TestInferOutputNames:
    """Test output name inference"""
    
    def test_infer_output_names_no_frame(self):
        """Test with no frame"""
        result = InferenceUtils.infer_output_names(None)
        assert result == ["output"]
    
    def test_infer_output_names_single_assignment(self):
        """Test single variable assignment"""
        mock_frame = Mock()
        mock_frame.code_context = ["result = predict('test')"]
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["result = predict('test')"]
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["result"]
    
    def test_infer_output_names_tuple_assignment(self):
        """Test tuple assignment"""
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["name1, name2 = predict('test')"]
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["name1", "name2"]
    
    def test_infer_output_names_no_assignment(self):
        """Test standalone call without assignment"""
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict('test')"]
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["output"]
    
    def test_infer_output_names_nested_frames(self):
        """Test searching through nested frames"""
        mock_frame = Mock()
        mock_frame.f_back = Mock()
        mock_frame.f_back.f_back = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            # First frame doesn't have predict
            mock_info.side_effect = [
                Mock(code_context=["some_other_code()"]),
                Mock(code_context=["result = chain_of_thought('test')"])
            ]
            
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["result"]
    
    def test_infer_output_names_no_code_context(self):
        """Test with no code context"""
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = None
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["output"]
    
    def test_infer_output_names_exception_handling(self):
        """Test exception handling"""
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo', side_effect=AttributeError):
            result = InferenceUtils.infer_output_names(mock_frame)
            assert result == ["output"]


class TestInferInputNames:
    """Test input name inference"""
    
    def test_infer_input_names_no_frame(self):
        """Test with no frame"""
        args = ("arg1", "arg2")
        result = InferenceUtils.infer_input_names(args, None)
        assert result == ["arg0", "arg1"]
    
    def test_infer_input_names_simple_variables(self):
        """Test simple variable names"""
        args = ("value1", "value2")
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict(var1, var2)"]
            
            with patch.object(InferenceUtils, 'safe_parse_ast') as mock_parse:
                # Create AST with variable names
                call_node = ast.Call(
                    func=ast.Name(id='predict'),
                    args=[ast.Name(id='var1'), ast.Name(id='var2')],
                    keywords=[]
                )
                tree = ast.Module(body=[ast.Expr(value=call_node)])
                mock_parse.return_value = tree
                
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["var1", "var2"]
    
    def test_infer_input_names_self_attributes(self):
        """Test self.attribute names"""
        args = ("value",)
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict(self.data)"]
            
            with patch.object(InferenceUtils, 'safe_parse_ast') as mock_parse:
                # Create AST with self.attribute
                call_node = ast.Call(
                    func=ast.Name(id='predict'),
                    args=[ast.Attribute(
                        value=ast.Name(id='self'),
                        attr='data'
                    )],
                    keywords=[]
                )
                tree = ast.Module(body=[ast.Expr(value=call_node)])
                mock_parse.return_value = tree
                
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["data"]
    
    def test_infer_input_names_mixed_types(self):
        """Test mixed argument types"""
        args = ("value1", "value2", "value3")
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict(var1, 'literal', func())"]
            
            with patch.object(InferenceUtils, 'safe_parse_ast') as mock_parse:
                # Create AST with mixed types
                call_node = ast.Call(
                    func=ast.Name(id='predict'),
                    args=[
                        ast.Name(id='var1'),
                        ast.Constant(value='literal'),
                        ast.Call(func=ast.Name(id='func'), args=[], keywords=[])
                    ],
                    keywords=[]
                )
                tree = ast.Module(body=[ast.Expr(value=call_node)])
                mock_parse.return_value = tree
                
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["var1", "arg1", "arg2"]
    
    def test_infer_input_names_reserved_words(self):
        """Test sanitization of reserved words"""
        args = ("value1", "value2", "value3")
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict(args, kwargs, self)"]
            
            with patch.object(InferenceUtils, 'safe_parse_ast') as mock_parse:
                # Create AST with reserved words
                call_node = ast.Call(
                    func=ast.Name(id='predict'),
                    args=[
                        ast.Name(id='args'),
                        ast.Name(id='kwargs'),
                        ast.Name(id='self')
                    ],
                    keywords=[]
                )
                tree = ast.Module(body=[ast.Expr(value=call_node)])
                mock_parse.return_value = tree
                
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["arg0", "arg1", "arg2"]
    
    def test_infer_input_names_no_call_node(self):
        """Test when no call node is found"""
        args = ("value1", "value2")
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["x = 1"]
            
            with patch.object(InferenceUtils, 'safe_parse_ast') as mock_parse:
                # Create AST without call node
                tree = ast.parse("x = 1")
                mock_parse.return_value = tree
                
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["arg0", "arg1"]
    
    def test_infer_input_names_no_code_context(self):
        """Test with no code context"""
        args = ("value",)
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = None
            result = InferenceUtils.infer_input_names(args, mock_frame)
            assert result == ["arg0"]
    
    def test_infer_input_names_ast_parse_error(self):
        """Test AST parsing error fallback"""
        args = ("value1", "value2")
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo') as mock_info:
            mock_info.return_value.code_context = ["predict(var1, var2)"]
            
            # When safe_parse_ast raises ValueError (which is caught), it falls back to arg0, arg1
            with patch.object(InferenceUtils, 'safe_parse_ast', side_effect=ValueError("Test error")):
                result = InferenceUtils.infer_input_names(args, mock_frame)
                assert result == ["arg0", "arg1"]
    
    def test_infer_input_names_exception_handling(self):
        """Test general exception handling"""
        args = ("value",)
        mock_frame = Mock()
        
        with patch('inspect.getframeinfo', side_effect=AttributeError):
            result = InferenceUtils.infer_input_names(args, mock_frame)
            assert result == ["arg0"]


class TestGetTypeHintsFromSignature:
    """Test type hint extraction from function signatures"""
    
    def test_get_type_hints_no_frame(self):
        """Test with no frame"""
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            None, ["input"], ["output"]
        )
        assert input_types == {}
        assert output_types == {}
    
    def test_get_type_hints_from_function(self):
        """Test extracting type hints from annotated function"""
        def test_func(text: str) -> str:
            return text
        
        mock_frame = Mock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_locals = {"test_func": test_func}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["text"], ["output"]
        )
        
        assert input_types == {"text": str}
        assert output_types == {"output": str}
    
    def test_get_type_hints_multiple_outputs(self):
        """Test type hints with tuple return type"""
        from typing import Tuple
        
        def test_func(x: int) -> Tuple[str, int]:
            return "result", 42
        
        mock_frame = Mock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_locals = {"test_func": test_func}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["x"], ["out1", "out2"]
        )
        
        assert input_types == {"x": int}
        assert output_types == {"out1": str, "out2": int}
    
    def test_get_type_hints_no_annotations(self):
        """Test function without type hints"""
        def test_func(x):
            return x
        
        mock_frame = Mock()
        mock_frame.f_code.co_name = "test_func"
        mock_frame.f_locals = {"test_func": test_func}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["x"], ["output"]
        )
        
        assert input_types == {}
        assert output_types == {}
    
    def test_get_type_hints_function_not_found(self):
        """Test when function is not found"""
        mock_frame = Mock()
        mock_frame.f_code.co_name = "nonexistent"
        mock_frame.f_locals = {}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["input"], ["output"]
        )
        
        assert input_types == {}
        assert output_types == {}
    
    def test_get_type_hints_from_outer_frame(self):
        """Test finding function in outer frame"""
        def test_func(x: int) -> str:
            return str(x)
        
        inner_frame = Mock()
        inner_frame.f_code.co_name = "test_func"
        inner_frame.f_locals = {}
        inner_frame.f_globals = {}
        
        outer_frame = Mock()
        outer_frame.f_locals = {"test_func": test_func}
        outer_frame.f_globals = {}
        
        inner_frame.f_back = outer_frame
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            inner_frame, ["x"], ["output"]
        )
        
        assert input_types == {"x": int}
        assert output_types == {"output": str}
    
    def test_get_type_hints_not_callable(self):
        """Test when found object is not callable"""
        mock_frame = Mock()
        mock_frame.f_code.co_name = "not_func"
        mock_frame.f_locals = {"not_func": "not a function"}
        mock_frame.f_globals = {}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["input"], ["output"]
        )
        
        assert input_types == {}
        assert output_types == {}
    
    def test_get_type_hints_signature_error(self):
        """Test when signature inspection fails"""
        # Built-in functions may not have inspectable signatures
        mock_frame = Mock()
        mock_frame.f_code.co_name = "print"
        mock_frame.f_locals = {}
        mock_frame.f_globals = {"print": print}
        mock_frame.f_back = None
        
        input_types, output_types = InferenceUtils.get_type_hints_from_signature(
            mock_frame, ["value"], ["output"]
        )
        
        # Should handle gracefully
        assert isinstance(input_types, dict)
        assert isinstance(output_types, dict)
    
    def test_get_type_hints_exception_handling(self):
        """Test general exception handling"""
        mock_frame = Mock()
        mock_frame.f_code.co_name = "test"
        
        # Cause an exception
        with patch('inspect.signature', side_effect=Exception("Test error")):
            input_types, output_types = InferenceUtils.get_type_hints_from_signature(
                mock_frame, ["input"], ["output"]
            )
            
            assert input_types == {}
            assert output_types == {}


class TestProcessReturnAnnotation:
    """Test return annotation processing"""
    
    def test_process_return_annotation_empty(self):
        """Test with empty annotation"""
        output_names = ["output"]
        output_types = {}
        
        InferenceUtils._process_return_annotation(
            inspect.Signature.empty, output_names, output_types
        )
        
        assert output_types == {}
    
    def test_process_return_annotation_single_type(self):
        """Test single return type"""
        output_names = ["result"]
        output_types = {}
        
        InferenceUtils._process_return_annotation(
            str, output_names, output_types
        )
        
        assert output_types == {"result": str}
    
    def test_process_return_annotation_tuple_type(self):
        """Test tuple return type"""
        from typing import Tuple
        
        output_names = ["out1", "out2", "out3"]
        output_types = {}
        
        # Create a Tuple type hint
        tuple_type = Tuple[str, int, bool]
        
        InferenceUtils._process_return_annotation(
            tuple_type, output_names, output_types
        )
        
        assert output_types == {"out1": str, "out2": int, "out3": bool}
    
    def test_process_return_annotation_mismatched_length(self):
        """Test when tuple length doesn't match output names"""
        from typing import Tuple
        
        output_names = ["out1", "out2"]
        output_types = {}
        
        # Tuple with 3 types but only 2 output names
        tuple_type = Tuple[str, int, bool]
        
        InferenceUtils._process_return_annotation(
            tuple_type, output_names, output_types
        )
        
        # Should not update output_types due to mismatch
        assert output_types == {}
    
    def test_process_return_annotation_no_args_attribute(self):
        """Test type without __args__ attribute"""
        output_names = ["out1", "out2"]
        output_types = {}
        
        # List doesn't have __args__ in the way we expect
        InferenceUtils._process_return_annotation(
            list, output_names, output_types
        )
        
        # Should not update output_types
        assert output_types == {}