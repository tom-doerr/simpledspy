API Reference (Simple)
======================

This is a simplified API reference for SimpleDSPy that doesn't require importing the modules.

Core Classes
------------

Predict
~~~~~~~

.. code-block:: python

   class Predict(BaseCaller):
       """Predict module caller for simple input-output transformations."""
       
       def __call__(self, *args, inputs=None, outputs=None, 
                    description=None, lm_params=None, 
                    name=None, trainset=None):
           """
           Execute a prediction.
           
           Args:
               *args: Input values to the module
               inputs: Optional list of input names
               outputs: Optional list of output names
               description: Task description for the LLM
               lm_params: LLM parameter overrides
               name: Module name for logging
               trainset: Training examples
               
           Returns:
               Single output value or tuple of outputs
           """

ChainOfThought
~~~~~~~~~~~~~~

.. code-block:: python

   class ChainOfThought(BaseCaller):
       """Chain of Thought module for reasoning tasks."""
       
       # Same interface as Predict but generates reasoning steps

Pipeline Management
-------------------

PipelineManager
~~~~~~~~~~~~~~~

.. code-block:: python

   class PipelineManager:
       """Manages multi-step LLM pipelines."""
       
       def add_step(self, name, module, description=None):
           """Add a step to the pipeline."""
           
       def register_step(self, inputs, outputs, module):
           """Register a step with explicit inputs/outputs."""
           
       def step(self, func):
           """Decorator to add a function as a pipeline step."""
           
       def run(self, **initial_inputs):
           """Execute the pipeline with initial inputs."""
           
       def assemble_pipeline(self):
           """Assemble steps into an executable pipeline."""

Optimization
------------

OptimizationManager
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class OptimizationManager:
       """Manages optimization of DSPy modules."""
       
       def optimize(self, module, trainset, valset=None, 
                    metric=None, teleprompter="BootstrapFewShot",
                    **kwargs):
           """
           Optimize a module using DSPy teleprompters.
           
           Args:
               module: Module to optimize
               trainset: Training examples
               valset: Validation examples
               metric: Evaluation metric function
               teleprompter: Optimization strategy
               **kwargs: Teleprompter-specific arguments
           """

Evaluator
~~~~~~~~~

.. code-block:: python

   class Evaluator:
       """Evaluates LLM outputs on a 1-10 scale."""
       
       def evaluate(self, prediction, expected=None, 
                    criteria=None, strict=False):
           """
           Evaluate prediction quality.
           
           Args:
               prediction: Model output
               expected: Expected output
               criteria: Custom evaluation criteria
               strict: Use strict evaluation
               
           Returns:
               Score from 1-10
           """

Configuration
-------------

Settings
~~~~~~~~

.. code-block:: python

   class Settings:
       """Global configuration for SimpleDSPy."""
       
       # LLM Configuration
       default_lm: str = None  # Default language model
       lm: Any = None  # DSPy LM instance
       
       # Logging
       logging_enabled: bool = True
       log_dir: str = ".simpledspy"
       
       # Retry Configuration
       retry_attempts: int = 3
       retry_delay: float = 1.0
       
       # Rate Limiting (deprecated)
       rate_limit_calls: int = 60
       rate_limit_window: int = 60

Utilities
---------

Logger
~~~~~~

.. code-block:: python

   class Logger:
       """Handles logging of module inputs/outputs."""
       
       def __init__(self, module_name, base_dir=".simpledspy"):
           """Initialize logger for a specific module."""
           
       def log(self, data):
           """Log execution data."""
           
       def load_training_data(self):
           """Load previously logged data for training."""

InferenceUtils
~~~~~~~~~~~~~~

Utilities for inferring variable names and types from code context:

- ``infer_input_names(args, frame)``: Infer input variable names
- ``infer_output_names(frame)``: Infer output variable names  
- ``get_type_hints_from_signature(frame, input_names, output_names)``: Extract type hints
- ``safe_parse_ast(code, max_depth=10)``: Safely parse Python AST

TrainingUtils
~~~~~~~~~~~~~

Utilities for handling training data:

- ``format_example(example)``: Format logged data for DSPy
- ``load_and_prepare_demos(name)``: Load training demos from logs
- ``apply_training_data(module, trainset=None, name=None)``: Apply training data to module

Exceptions
----------

.. code-block:: python

   # Base exception
   class SimpleDSPyError(Exception):
       """Base exception for SimpleDSPy errors."""

   # Specific exceptions
   class ValidationError(SimpleDSPyError):
       """Raised when input validation fails."""
       
   class ConfigurationError(SimpleDSPyError):
       """Raised when configuration is invalid."""
       
   class ModuleError(SimpleDSPyError):
       """Raised when module execution fails."""
       
   class PipelineError(SimpleDSPyError):
       """Raised when pipeline operations fail."""
       
   class SecurityError(SimpleDSPyError):
       """Raised when security constraints are violated."""

Retry Configuration
-------------------

.. code-block:: python

   @dataclass
   class RetryConfig:
       """Configuration for retry behavior."""
       
       max_attempts: int = 3
       initial_delay: float = 1.0
       max_delay: float = 60.0
       exponential_base: float = 2.0
       exceptions: tuple = (Exception,)
       
   def with_retry(config: RetryConfig):
       """Decorator to add retry logic to functions."""