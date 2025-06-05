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
        
        # Create simple descriptions for inputs
        for inp in inputs:
            signature_fields[inp] = dspy.InputField(desc=inp)
            
        # Create simple descriptions for outputs
        for outp in outputs:
            signature_fields[outp] = dspy.OutputField(desc=outp)

        # Create better default instructions
        if description:
            instructions = description
        elif inputs and outputs:
            instructions = f"Given inputs: {', '.join(inputs)}. Produce outputs: {', '.join(outputs)}"
        elif inputs:
            instructions = f"Process inputs: {', '.join(inputs)}"
        else:
            instructions = f"Produce outputs: {', '.join(outputs)}"
            
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
        
        return dspy.Predict(signature_class)
