.. SimpleDSPy documentation master file

SimpleDSPy Documentation
========================

SimpleDSPy is a lightweight wrapper around DSPy that simplifies the creation of LLM pipelines through a reflection-based API. It allows users to build complex LLM workflows with minimal boilerplate code.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_simple
   examples

Key Features
------------

* **Reflection-based API**: Automatically infers variable names from calling context
* **Minimal Boilerplate**: Build complex pipelines with simple, intuitive code
* **Type Hints Support**: Automatic signature generation from type annotations
* **Pipeline Management**: Chain multiple LLM operations seamlessly
* **Optimization Ready**: Built-in support for DSPy optimizers
* **Logging & Training**: Automatic training data collection and management

Quick Example
-------------

.. code-block:: python

   from simpledspy import Predict

   # Simple prediction
   question = "What is the capital of France?"
   answer = Predict()(question)
   print(answer)  # "Paris"

   # With type hints
   def qa_system(question: str) -> str:
       answer = Predict()(question)
       return answer

Installation
------------

.. code-block:: bash

   pip install simpledspy

Requirements
------------

* Python 3.9+
* DSPy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

