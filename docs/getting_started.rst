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

   git clone https://github.com/tom-doerr/simpledspy.git
   cd simpledspy
   pip install -e .

Configure a DSPy language model before making LLM calls:

.. code-block:: python

   import dspy
   import simpledspy

   simpledspy.configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"))

Basic Usage
-----------

SimpleDSPy provides two main ways to create LLM modules:

1. **Predict**: For simple input-output transformations
2. **ChainOfThought**: For reasoning tasks that benefit from step-by-step thinking

Simple Prediction
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import predict

   # SimpleDSPy automatically infers variable names
   question = "What is the capital of France?"
   answer = predict(question)
   print(answer)  # Output: "Paris"

Chain of Thought
~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import chain_of_thought

   # For complex reasoning tasks
   problem = "If a train travels 60 mph for 2 hours, how far does it go?"
   solution = chain_of_thought(problem)
   print(solution)  # Output includes reasoning steps

Using Type Hints
----------------

SimpleDSPy can automatically generate signatures from type hints:

.. code-block:: python

   from simpledspy import predict

   def summarize_text(text: str) -> str:
       """Summarize the given text."""
       summary = predict(text)
       return summary

   long_text = "..."  # Your long text here
   summary = summarize_text(long_text)

Multiple Inputs and Outputs
---------------------------

.. code-block:: python

   from simpledspy import predict

   # Multiple inputs
   context = "The weather is sunny and warm."
   question = "What's the weather like?"
   answer = predict(context, question)

   # Multiple outputs with explicit names
   text = "Python is a great programming language."
   sentiment, confidence = predict(text, outputs=["sentiment", "confidence"])

Custom Descriptions
-------------------

Add descriptions to help the LLM understand the task:

.. code-block:: python

   from simpledspy import predict

   query = "machine learning applications"
   results = predict(
       query,
       description="Generate relevant search results for the query"
   )

Pipeline Creation
-----------------

Chain multiple operations together:

.. code-block:: python

   from simpledspy import ModuleFactory, PipelineManager

   factory = ModuleFactory()
   pipeline = PipelineManager()

   pipeline.register_step(
       inputs=["text"],
       outputs=["entities"],
       module=factory.create_module(
           inputs=["text"],
           outputs=["entities"],
           description="Extract named entities from the text",
       ),
   )
   pipeline.register_step(
       inputs=["entities"],
       outputs=["categories"],
       module=factory.create_module(
           inputs=["entities"],
           outputs=["categories"],
           description="Classify each entity by type",
       ),
   )

   compiled_pipeline = pipeline.assemble_pipeline()
   text = "Apple Inc. was founded by Steve Jobs in Cupertino."
   result = compiled_pipeline(text=text)
   print(result.categories)

Configuration
-------------

Configure SimpleDSPy settings:

.. code-block:: python

   import dspy
   import simpledspy

   # Set default LLM
   simpledspy.configure(lm=dspy.LM(model="openai/gpt-4"))

   # Enable logging
   simpledspy.configure(logging_enabled=True)

   # Set retry configuration
   simpledspy.configure(retry_attempts=3, retry_delay=1.0)

Next Steps
----------

* Explore the :doc:`api_reference` for detailed documentation
* Check out :doc:`examples` for more complex use cases
* Learn about optimization and training data collection
