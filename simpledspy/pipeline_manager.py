from typing import List, Any, Tuple, Dict
import dspy

class PipelineManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Ensure _steps is initialized only once for the singleton instance
            cls._instance._steps = [] 
        return cls._instance

    def register_step(self, inputs: List[str], outputs: List[str], module: Any):
        # Stores a tuple: (list_of_input_names, list_of_output_names, module_instance)
        self._steps.append((inputs, outputs, module))

    def assemble_pipeline(self):
        """
        Assembles the registered steps into a DSPy pipeline module.
        
        Returns:
            dspy.Module: The assembled DSPy pipeline.
        """
        if not self._steps:
            raise ValueError("No pipeline steps registered. Please make pipe calls before assembling the pipeline.")

        class Pipeline(dspy.Module):
            def __init__(self, steps_data: List[Tuple[List[str], List[str], dspy.Module]]):
                super().__init__()
                self.steps = steps_data  # Expose steps data for testing
                # Register each module instance with a unique name like 'step_0', 'step_1', etc.
                for i, step_data_tuple in enumerate(self.steps):
                    module_instance = step_data_tuple[2] # The module is the 3rd element
                    setattr(self, f'step_{i}', module_instance)

            def forward(self, **inputs):
                # Start with the initial inputs
                data = inputs.copy()
                
                for i, step_data_tuple in enumerate(self.steps):
                    input_names_for_this_step = step_data_tuple[0]
                    output_names_for_this_step = step_data_tuple[1]
                    module_instance = getattr(self, f'step_{i}')
                    
                    # Prepare the input dictionary for this step
                    step_inputs = {}
                    for name in input_names_for_this_step:
                        if name not in data:
                            raise ValueError(f"Pipeline Step {i}: Missing input '{name}'")
                        step_inputs[name] = data[name]
                    
                    # Execute the module
                    prediction_object = module_instance(**step_inputs)
                    
                    # Update data with outputs from this step
                    for name in output_names_for_this_step:
                        if not hasattr(prediction_object, name):
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found")
                        data[name] = getattr(prediction_object, name)
                
                # Return the final outputs as they were last stored
                if len(output_names_for_this_step) == 1:
                    return data[output_names_for_this_step[0]]
                return tuple(data[name] for name in output_names_for_this_step)

        return Pipeline(self._steps)
