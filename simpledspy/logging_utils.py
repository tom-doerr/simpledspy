"""Utilities for logging module results"""

from typing import List, Dict, Any
from .logger import Logger
from .settings import settings as global_settings


class LoggingUtils:
    """Utilities for logging module execution results"""
    
    @staticmethod
    def log_results(
        module_name: str,
        input_dict: Dict[str, Any],
        input_names: List[str],
        output_names: List[str],
        output_values: List[Any],
        description: str = None,
    ):
        """Log module inputs and outputs with meaningful names"""
        # Create input structure with both names and values
        inputs_data = []
        for name in input_names:
            if name in input_dict:
                inputs_data.append({"name": name, "value": input_dict[name]})

        # Create output structure with both names and values
        outputs_data = []
        for i, name in enumerate(output_names):
            outputs_data.append({"name": name, "value": output_values[i]})

        # Create logger for this specific module
        base_dir = global_settings.log_dir if global_settings.log_dir else ".simpledspy"
        logger = Logger(module_name=module_name, base_dir=base_dir)
        logger.log(
            {
                "module": module_name,
                "inputs": inputs_data,
                "outputs": outputs_data,
                "description": description,
            }
        )
