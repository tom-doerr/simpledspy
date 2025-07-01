"""Utilities for inferring input and output names from code context"""

import inspect
import ast
from typing import List, Dict, Any, Tuple
from .exceptions import SecurityError


class InferenceUtils:
    """Utilities for inferring variable names from calling context"""
    
    @staticmethod
    def get_ast_depth(node: ast.AST, current_depth: int = 0) -> int:
        """Calculate the depth of an AST node"""
        if not isinstance(node, ast.AST):
            return current_depth
        
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = InferenceUtils.get_ast_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    @staticmethod
    def safe_parse_ast(code: str, max_depth: int = 10) -> ast.AST:
        """Safely parse AST with depth limiting"""
        try:
            # Limit code length to prevent DoS
            if len(code) > 1000:
                raise SecurityError("Code too long for AST parsing")
            
            tree = ast.parse(code)
            
            # Check AST depth
            if InferenceUtils.get_ast_depth(tree) > max_depth:
                raise SecurityError("AST too deep")
            
            return tree
        except (SyntaxError, ValueError) as e:
            raise SecurityError(f"Failed to parse AST: {e}") from e
    
    @staticmethod
    def infer_output_names(frame: Any) -> List[str]:
        """Infer output names based on assignment context"""
        if frame is None:
            return ["output"]

        try:
            # Get the code context lines
            lines = inspect.getframeinfo(frame).code_context
            if not lines:
                return ["output"]
            line = lines[0].strip()

            # Skip frames that don't contain our actual call
            # Look for 'predict(' or 'chain_of_thought(' in the line
            max_depth = 5
            current_frame = frame
            for _ in range(max_depth):
                if current_frame is None:
                    break
                lines = inspect.getframeinfo(current_frame).code_context
                if lines and (
                    "predict(" in lines[0] or "chain_of_thought(" in lines[0]
                ):
                    line = lines[0].strip()
                    break
                current_frame = current_frame.f_back

            # Handle multi-line assignment
            if "=" in line:
                lhs = line.split("=")[0].strip()
                # Handle tuple assignment: name1, name2 = ...
                if "," in lhs:
                    output_names = [name.strip() for name in lhs.split(",")]
                # Handle single assignment: name = ...
                else:
                    output_names = [lhs]
            # Handle no assignment (standalone call)
            else:
                output_names = ["output"]

            return output_names
        except (AttributeError, IndexError, TypeError):
            return ["output"]
    
    @staticmethod
    def infer_input_names(args: tuple, frame: Any) -> List[str]:
        """Infer input variable names using frame inspection"""
        try:
            if frame is None:
                return [f"arg{i}" for i in range(len(args))]

            # Get the code context of the call
            context_lines = inspect.getframeinfo(frame).code_context
            if not context_lines:
                return [f"arg{i}" for i in range(len(args))]

            call_line = context_lines[0].strip()

            # Try the AST method with safety checks
            try:
                tree = InferenceUtils.safe_parse_ast(call_line)
                call_node = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        call_node = node
                        break
                if call_node is None:
                    return [f"arg{i}" for i in range(len(args))]

                arg_names = []
                for arg in call_node.args:
                    if isinstance(arg, ast.Name):
                        name = arg.id
                        arg_names.append(name)
                    elif (
                        isinstance(arg, ast.Attribute)
                        and isinstance(arg.value, ast.Name)
                        and arg.value.id == "self"
                    ):
                        name = arg.attr
                        arg_names.append(name)
                    else:
                        # For any other type of node, assign argX name
                        arg_names.append(f"arg{len(arg_names)}")

                # Sanitize reserved words
                reserved = ["args", "kwargs", "self"]
                for i, name in enumerate(arg_names):
                    if name in reserved:
                        arg_names[i] = f"arg{i}"

                return arg_names
            except (SyntaxError, TypeError, ValueError):
                # If AST method fails, fall back to arg0, arg1, ...
                return [f"arg{i}" for i in range(len(args))]
        except (AttributeError, ValueError, IndexError, TypeError):
            return [f"arg{i}" for i in range(len(args))]
    
    @staticmethod
    def get_type_hints_from_signature(
        frame: Any, input_names: List[str], output_names: List[str]
    ) -> Tuple[Dict[str, type], Dict[str, type]]:
        """Get input/output types from function signature"""
        input_types: Dict[str, type] = {}
        output_types: Dict[str, type] = {}
        if frame is None:
            return input_types, output_types

        try:
            # Get the function object from the frame hierarchy
            func_name = frame.f_code.co_name
            func = None

            # Check current frame's locals
            func = frame.f_locals.get(func_name, None)
            if func is None:
                # Check current frame's globals
                func = frame.f_globals.get(func_name, None)

            # If not found, check the outer frame
            if func is None and frame.f_back:
                outer_frame = frame.f_back
                func = outer_frame.f_locals.get(
                    func_name, outer_frame.f_globals.get(func_name, None)
                )

            if func and callable(func):
                try:
                    signature = inspect.signature(func, follow_wrapped=True)

                    # Get input parameter types
                    for param_name, param in signature.parameters.items():
                        if (
                            param_name in input_names
                            and param.annotation != inspect.Parameter.empty
                        ):
                            input_types[param_name] = param.annotation

                    # Get return type hints
                    return_ann = signature.return_annotation
                    InferenceUtils._process_return_annotation(
                        return_ann, output_names, output_types
                    )
                except (ValueError, TypeError):
                    # Signature inspection failed
                    pass
        except Exception:  # pylint: disable=broad-except
            # Failed to get type hints
            pass

        return input_types, output_types
    
    @staticmethod
    def _process_return_annotation(return_ann, output_names, output_types):
        """Helper to process return annotation types"""
        if return_ann == inspect.Signature.empty:
            return

        if len(output_names) == 1:
            output_types[output_names[0]] = return_ann
        elif hasattr(return_ann, "__args__") and len(return_ann.__args__) == len(
            output_names
        ):
            tuple_types = return_ann.__args__
            for i, t in enumerate(tuple_types):
                if i < len(output_names):
                    output_types[output_names[i]] = t
