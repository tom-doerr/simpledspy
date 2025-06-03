import dspy
from typing import List, Dict, Any

class ModuleFactory:
    def create_signature(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> dspy.Signature:
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
        signature_class = self.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        return dspy.Predict(signature_class)
