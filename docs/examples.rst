Examples
========

This page contains various examples demonstrating SimpleDSPy's capabilities.

Basic Examples
--------------

Question Answering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict

   # Simple Q&A
   question = "What is the speed of light?"
   answer = Predict()(question)
   print(f"Answer: {answer}")

   # With context
   context = "The speed of light in vacuum is exactly 299,792,458 meters per second."
   question = "How fast does light travel?"
   answer = Predict()(context, question)

Text Summarization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict

   article = """
   Artificial intelligence has transformed many industries over the past decade.
   From healthcare to finance, AI applications are becoming increasingly sophisticated.
   Machine learning models can now perform tasks that were once thought to be 
   exclusively human domains.
   """
   
   summary = Predict(
       outputs=["summary"],
       description="Summarize the article in one sentence"
   )(article)

Advanced Examples
-----------------

Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import ChainOfThought

   text = "Apple Inc. announced that Tim Cook will visit Paris next month."
   
   # Extract entities with reasoning
   entities = ChainOfThought(
       outputs=["entities"],
       description="Extract all named entities and their types"
   )(text)

Multi-step Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import PipelineManager, Predict, ChainOfThought

   # Create a document processing pipeline
   pipeline = PipelineManager()

   @pipeline.step
   def extract_key_points(document: str) -> str:
       """Extract main points from document"""
       points = ChainOfThought(
           description="Extract 3-5 key points from the document"
       )(document)
       return points

   @pipeline.step
   def generate_summary(key_points: str) -> str:
       """Generate executive summary"""
       summary = Predict(
           description="Create a concise executive summary"
       )(key_points)
       return summary

   @pipeline.step
   def create_action_items(key_points: str, summary: str) -> str:
       """Create action items based on the document"""
       action_items = ChainOfThought(
           description="Generate actionable next steps"
       )(key_points, summary)
       return action_items

   # Process a document
   document = "..."  # Your document here
   results = pipeline.run(document)

Type-Guided Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import List, Dict
   from simpledspy import Predict

   def analyze_sentiment(reviews: List[str]) -> Dict[str, float]:
       """Analyze sentiment of multiple reviews"""
       
       sentiments = []
       for review in reviews:
           # Type hints guide the output format
           sentiment_score = Predict(
               outputs=["score"],
               description="Rate sentiment from 0.0 (negative) to 1.0 (positive)"
           )(review)
           sentiments.append(float(sentiment_score))
       
       return {
           "average": sum(sentiments) / len(sentiments),
           "max": max(sentiments),
           "min": min(sentiments)
       }

Training and Optimization
-------------------------

Collecting Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict, settings

   # Enable logging to collect training data
   settings.logging_enabled = True
   settings.log_dir = "./training_data"

   # Use modules normally - data is automatically collected
   for question in questions:
       answer = Predict(name="qa_module")(question)

   # Later, use collected data for optimization
   from simpledspy import OptimizationManager

   optimizer = OptimizationManager()
   optimized_module = optimizer.optimize(
       module_name="qa_module",
       metric="accuracy"
   )

Custom Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Evaluator, Predict

   # Create custom evaluation criteria
   evaluator = Evaluator()

   def evaluate_qa_response(question, answer, expected):
       score = evaluator.evaluate(
           prediction=answer,
           expected=expected,
           criteria={
               "accuracy": "How factually correct is the answer?",
               "completeness": "Does it fully address the question?",
               "clarity": "Is the answer clear and well-structured?"
           }
       )
       return score

   # Use in optimization
   results = []
   for q, expected_a in test_set:
       predicted_a = Predict()(q)
       score = evaluate_qa_response(q, predicted_a, expected_a)
       results.append(score)

Error Handling
--------------

Retry Logic
~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict, settings

   # Configure retry behavior
   settings.retry_attempts = 5
   settings.retry_delay = 2.0  # seconds

   # Automatic retry on failures
   try:
       result = Predict()(complex_input)
   except Exception as e:
       print(f"Failed after {settings.retry_attempts} attempts: {e}")

Custom Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import Predict
   from simpledspy.exceptions import ValidationError, ModuleError

   try:
       # Input validation
       if not text:
           raise ValidationError("Input text cannot be empty")
       
       result = Predict()(text)
       
   except ValidationError as e:
       print(f"Invalid input: {e}")
   except ModuleError as e:
       print(f"Module execution failed: {e}")

Best Practices
--------------

1. **Use descriptive variable names** - SimpleDSPy infers names from your variables
2. **Add type hints** - Helps with automatic signature generation
3. **Provide descriptions** - Improves LLM understanding of the task
4. **Enable logging** - Collect training data for optimization
5. **Handle errors gracefully** - Use try-except blocks for production code
6. **Optimize iteratively** - Start simple, then optimize based on real usage