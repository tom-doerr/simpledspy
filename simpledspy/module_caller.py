from typing import Any, Dict, List
import dspy

class ModuleCaller:
    def __init__(self, module_type: type, inputs: List[str], outputs: List[str], 
                 description: str = "", input_types: Dict[str, type] = None,
                 output_types: Dict[str, type] = None):
        """
        Initialize with DSPy module type and signature parameters.
        
        Args:
            module_type: DSPy module class (e.g., dspy.Predict, dspy.ChainOfThought)
            inputs: List of input field names
            outputs: List of output field names
            description: Optional description of the module's purpose
            input_types: Dictionary mapping input names to types
            output_types: Dictionary mapping output names to types
        """
        self.module_type = module_type
        self.inputs = inputs
        self.outputs = outputs
        self.description = description
        self.input_types = input_types or {}
        self.output_types = output_types or {}
        
        # Create signature
        self._create_signature()
        
        # Create module
        self.module = self.module_type(self.signature)
    
    def _create_signature(self):
        """Create DSPy signature from parameters"""
        signature_fields = {}
        
        # Create input fields
        for inp in self.inputs:
            desc = f"Input field {inp}"
            if inp in self.input_types:
                desc += f" of type {self.input_types[inp].__name__}"
            signature_fields[inp] = dspy.InputField(desc=desc)
            
        # Create output fields
        for outp in self.outputs:
            desc = f"Output field {outp}"
            if outp in self.output_types:
                desc += f" of type {self.output_types[outp].__name__}"
            signature_fields[outp] = dspy.OutputField(desc=desc)
        
        # Create signature class
        instructions = self.description or f"Given {', '.join(self.inputs)}, produce {', '.join(self.outputs)}."
        self.signature = type(
            'Signature',
            (dspy.Signature,),
            {
                '__doc__': instructions,
                **signature_fields
            }
        )
    
    def __call__(self, **kwargs) -> Any:
        """
        Execute the module with given inputs.
        
        Args:
            **kwargs: Input values keyed by input field names
            
        Returns:
            Module output value(s)
        """
        # Validate inputs
        for inp in self.inputs:
            if inp not in kwargs:
                raise ValueError(f"Missing required input: {inp}")
                
        # Run prediction
        prediction = self.module(**kwargs)
        
        # Handle outputs
        results = []
        for outp in self.outputs:
            value = getattr(prediction, outp)
            
            # Apply type conversion if specified
            if outp in self.output_types:
                target_type = self.output_types[outp]
                try:
                    if target_type is int:
                        value = int(float(value.replace(',', '').strip()))
                    elif target_type is float:
                        value = float(value.replace(',', '').strip())
                    elif target_type is bool:
                        value = value.lower() == 'true' if isinstance(value, str) else bool(value)
                    elif target_type is str:
                        value = str(value)
                except (ValueError, TypeError):
                    pass  # Keep original value on conversion failure
            results.append(value)
            
        return results[0] if len(results) == 1 else tuple(results)
