import dspy
from typing import List, Dict, Any

class ModuleFactory:
    def create_module(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Module:
        """
        Creates a DSPy module with specified inputs and outputs.
        
        Args:
            inputs: List of input field names
            outputs: List of output field names
            input_types: Dictionary mapping input names to types
            output_types: Dictionary mapping output names to types
            description: Optional description of the module's purpose
            
        Returns:
            dspy.Module: Configured DSPy module
        """
        signature_fields = {}
        
        # Create input fields with type hints
        for inp in inputs:
            # Handle input types safely
            if input_types and isinstance(input_types, dict):
                field_type = input_types.get(inp)
            else:
                field_type = None
                
            desc = f"Input field {inp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            signature_fields[inp] = dspy.InputField(desc=desc)
            
        # Create output fields with type hints
        for outp in outputs:
            field_type = output_types.get(outp) if output_types else None
            desc = f"Output field {outp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            signature_fields[outp] = dspy.OutputField(desc=desc)

        # Create signature class
        instructions = description or f"Given the fields {', '.join(inputs)}, produce the fields {', '.join(outputs)}."
        signature_class = type(
            'Signature',
            (dspy.Signature,),
            {
                '__doc__': instructions,
                **signature_fields
            }
        )
        
        # Create and return a Predict module with the signature
        return dspy.Predict(signature_class)




        # Create signature class with proper instructions
        instructions = description or f"Given the fields {', '.join(inputs)}, produce the fields {', '.join(outputs)}."
        return type(
            'Signature',
            (dspy.Signature,),
            {
                '__doc__': instructions,
                **signature_fields
            }
        )
