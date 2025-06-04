"""Factory for creating DSPy signatures and modules

Provides:
- create_signature: Creates DSPy Signature classes
- create_module: Creates DSPy Predict/ChainOfThought modules
"""
import dspy
from typing import List, Dict, Any

class ModuleFactory:
    def create_signature(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Signature:
        signature_fields = {}
        
        # Create descriptions for inputs
        for inp in inputs:
            field_type = input_types.get(inp) if input_types else None
            type_info = f" (type: {field_type.__name__})" if field_type else ""
            signature_fields[inp] = dspy.InputField(desc=f"{inp}{type_info}")
            
        # Create descriptions for outputs
        for outp in outputs:
            field_type = output_types.get(outp) if output_types else None
            type_info = f" (type: {field_type.__name__})" if field_type else ""
            signature_fields[outp] = dspy.OutputField(desc=f"{outp}{type_info}")

        # Create better default instructions
        if description:
            instructions = description
        elif inputs and outputs:
            instructions = f"Process inputs ({', '.join(inputs)}) to produce outputs ({', '.join(outputs)})"
        elif inputs:
            instructions = f"Process inputs ({', '.join(inputs)})"
        else:
            instructions = f"Produce outputs ({', '.join(outputs)})"
            
        return type(
            'Signature',
            (dspy.Signature,),
            {
                '__doc__': instructions,
                **signature_fields
            }
        )

    def create_module(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Module:
        """
        Creates a DSPy module with the specified signature
        
        Args:
            inputs: List of input field names
            outputs: List of output field names
            input_types: Optional dictionary mapping input names to types
            output_types: Optional dictionary mapping output names to types
            description: Custom description for the module signature
            
        Returns:
            DSPy Predict module with the configured signature
        """
        signature_class = self.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        
        # Create a custom module that appends "processed:" to inputs
        class CustomModule(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.signature = signature
                self.predict = dspy.Predict(signature)
                
            def forward(self, **kwargs):
                # Preprocess inputs by adding "processed:" prefix
                processed_kwargs = {k: f"processed: {v}" for k, v in kwargs.items()}
                return self.predict(**processed_kwargs)
                
        return CustomModule(signature_class)
