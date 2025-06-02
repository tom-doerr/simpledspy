import dspy
from typing import List, Dict, Any

class ModuleFactory:
    def create_module(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Module:
        signature_fields = {}
        
        # Use original input names
        for inp in inputs:
            field_type = input_types.get(inp) if input_types else None
            desc = f"Input field {inp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            signature_fields[inp] = dspy.InputField(desc=desc)
            
        # Use original output names
        for outp in outputs:
            field_type = output_types.get(outp) if output_types else None
            desc = f"Output field {outp}"
            if field_type:
                desc += f" of type {field_type.__name__}"
            signature_fields[outp] = dspy.OutputField(desc=desc)

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
