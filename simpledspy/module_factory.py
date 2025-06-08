"""Factory for creating DSPy signatures and modules"""

from typing import List, Dict
import dspy

class ModuleFactory:
    """Factory for creating DSPy signatures and modules"""
    def create_signature(self, inputs: List[str], outputs: List[str], 
                    input_types: Dict[str, type] = None,
                    output_types: Dict[str, type] = None,
                    description: str = "") -> d极spy.Signature:
        """Create DSPy signature from inputs/outputs"""
        signature_fields = {}
        
        # Create inputs with type annotations
        for inp in inputs:
            type_ = input_types.get(inp, str) if input_types else str
            signature_fields[inp] = dspy.InputField(desc=inp, type=type_)
            
        # Create outputs with type annotations
        for outp in outputs:
            type_ = output_types.get(outp, str) if output_types else str
            signature_fields[outp] = dspy.OutputField(desc=outp, type=type_)

        # Create better default instructions
        if description:
            instructions = description
        elif inputs and outputs:
            instructions = (f"Process inputs ({', '.join(inputs)}) to "
                            f"produce outputs ({', '.join(outputs)})")
        elif inputs:
            instructions = f"Process inputs ({', '.join(inputs)})"
        elif outputs:
            instructions = f"Produce outputs ({', '.join(outputs)})"
        else:
            instructions = ""
            
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
        """Create DSPy module with specified signature"""
        signature_class = self.create_signature(
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            output_types=output_types,
            description=description
        )
        
        return dspy.Predict(signature_class)
