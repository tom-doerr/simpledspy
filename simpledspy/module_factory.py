import dspy
from typing import List, Dict, Any

class ModuleFactory:
    def create_module(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Module:
        signature_fields = {}
        
        # Always use input_1, input_2, ... for inputs
        for i, inp in enumerate(inputs):
            field_type = input_types.get(inp) if input_types else None
            desc = f"Input field {inp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            field_name = f"input_{i+1}"
            signature_fields[field_name] = dspy.InputField(desc=desc)
            
        # Always use output_1, output_2, ... for outputs
        for i, outp in enumerate(outputs):
            field_type = output_types.get(outp) if output_types else None
            desc = f"Output field {outp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            field_name = f"output_{i+1}"
            signature_fields[field_name] = dspy.OutputField(desc=desc)

        instructions = description or f"Given the fields {', '.join(inputs)}, produce the fields {', '.join(outputs)}."
        signature_class = type(
            'Signature',
            (dspy.Signature,),
            {
                '__doc__': instructions,
                **signature_fields
            }
        )
        
        return dspy.Predict(signature_class)
