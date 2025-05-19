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
                self.pipeline_steps_data = steps_data
                # Register each module instance with a unique name like 'step_0', 'step_1', etc.
                for i, step_data_tuple in enumerate(self.pipeline_steps_data):
                    module_instance = step_data_tuple[2] # The module is the 3rd element
                    setattr(self, f'step_{i}', module_instance)

            def forward(self, *args):
                # Initial values for the first step, passed as positional arguments
                current_values_for_next_step: Any = args 

                for i, step_data_tuple in enumerate(self.pipeline_steps_data):
                    input_names_for_this_step = step_data_tuple[0]
                    output_names_for_this_step = step_data_tuple[1]
                    # Retrieve the pre-registered module instance for this step
                    module_instance = getattr(self, f'step_{i}')

                    # Prepare kwargs for the current module
                    # Ensure current_values_for_next_step is a tuple for zipping
                    if not isinstance(current_values_for_next_step, tuple):
                        current_values_for_next_step = (current_values_for_next_step,)

                    if len(input_names_for_this_step) != len(current_values_for_next_step):
                        raise ValueError(
                            f"Pipeline Step {i}: Mismatch between number of expected inputs "
                            f"({len(input_names_for_this_step)} names: {input_names_for_this_step}) and "
                            f"number of available values ({len(current_values_for_next_step)} values: {current_values_for_next_step})."
                        )
                    
                    kwargs_for_module = dict(zip(input_names_for_this_step, current_values_for_next_step))
                    
                    # Execute the current step's module
                    prediction_object = module_instance(**kwargs_for_module) # dspy.Predict returns a dspy.Prediction
                    
                    # Extract results from the prediction_object based on the registered output_names_for_this_step
                    # These will become current_values_for_next_step for the *next* iteration (or final result)
                    extracted_outputs_list = []
                    for name in output_names_for_this_step:
                        if hasattr(prediction_object, name):
                            extracted_outputs_list.append(getattr(prediction_object, name))
                        else:
                            # This case should ideally not happen if module signature and output_names are consistent
                            raise ValueError(f"Pipeline Step {i}: Output field '{name}' not found in prediction object.")

                    if len(extracted_outputs_list) == 1:
                        current_values_for_next_step = extracted_outputs_list[0]
                    else:
                        current_values_for_next_step = tuple(extracted_outputs_list)
                
                # The result of the last step is the result of the pipeline
                return current_values_for_next_step

        return Pipeline(self._steps)
