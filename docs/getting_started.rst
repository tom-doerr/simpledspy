Getting Started
===============

This guide will help you get started with SimpleDSPy.

Installation
------------

Install SimpleDSPy using pip:

.. code-block:: bash

   pip install simpledspy

Or install from source:

.. code-block:: bash

   git clone https://github.com/simpledspy/simpledspy.git
   cd simpledspy
   pip install -e .

Basic Usage
-----------

SimpleDSPy provides two main ways to create LLM modules:

1. **Predict**: For simple input-output transformations
2. **ChainOfThought**: For reasoning tasks that benefit from step-by-step thinking

Simple Prediction
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict

   # SimpleDSPy automatically infers variable names
   question = "What is the capital of France?"
   answer = Predict()(question)
   print(answer)  # Output: "Paris"

Chain of Thought
~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import ChainOfThought

   # For complex reasoning tasks
   problem = "If a train travels 60 mph for 2 hours, how far does it go?"
   solution = ChainOfThought()(problem)
   print(solution)  # Output includes reasoning steps

Using Type Hints
----------------

SimpleDSPy can automatically generate signatures from type hints:

.. code-block:: python

   from simpledspy import Predict

   def summarize_text(text: str) -> str:
       """Summarize the given text."""
       summary = Predict()(text)
       return summary

   long_text = "..."  # Your long text here
   summary = summarize_text(long_text)

Multiple Inputs and Outputs
---------------------------

.. code-block:: python

   from simpledspy import Predict

   # Multiple inputs
   context = "The weather is sunny and warm."
   question = "What's the weather like?"
   answer = Predict()(context, question)

   # Multiple outputs with explicit names
   text = "Python is a great programming language."
   sentiment, confidence = Predict(outputs=["sentiment", "confidence"])(text)

Custom Descriptions
-------------------

Add descriptions to help the LLM understand the task:

.. code-block:: python

   from simpledspy import Predict

   query = "machine learning applications"
   results = Predict(
       description="Generate relevant search results for the query"
   )(query)

Pipeline Creation
-----------------

Chain multiple operations together:

.. code-block:: python

   from simpledspy import PipelineManager, Predict

   # Create a pipeline
   pipeline = PipelineManager()

   # Add steps
   @pipeline.step
   def extract_entities(text: str) -> str:
       entities = Predict()(text)
       return entities

   @pipeline.step  
   def classify_entities(entities: str) -> str:
       categories = Predict()(entities)
       return categories

   # Run the pipeline
   text = "Apple Inc. was founded by Steve Jobs in Cupertino."
   result = pipeline.run(text)

Configuration
-------------

Configure SimpleDSPy settings:

.. code-block:: python

   from simpledspy import settings

   # Set default LLM
   settings.default_lm = "openai/gpt-4"

   # Enable logging
   settings.logging_enabled = True

   # Set retry configuration
   settings.retry_attempts = 3
   settings.retry_delay = 1.0

Next Steps
----------

* Explore the :doc:`api_reference` for detailed documentation
* Check out :doc:`examples` for more complex use cases
* Learn about optimization and training data collection