Examples
========

This page contains various examples demonstrating SimpleDSPy's capabilities.

Basic Examples
--------------

Question Answering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import predict

   # Simple Q&A
   question = "What is the speed of light?"
   answer = predict(question)
   print(f"Answer: {answer}")

   # With context
   context = "The speed of light in vacuum is exactly 299,792,458 meters per second."
   question = "How fast does light travel?"
   answer = predict(context, question)

Text Summarization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import predict

   article = """
   Artificial intelligence has transformed many industries over the past decade.
   From healthcare to finance, AI applications are becoming increasingly sophisticated.
   Machine learning models can now perform tasks that were once thought to be 
   exclusively human domains.
   """
   
   summary = predict(
       article,
       outputs=["summary"],
       description="Summarize the article in one sentence",
   )

Advanced Examples
-----------------

Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import chain_of_thought

   text = "Apple Inc. announced that Tim Cook will visit Paris next month."
   
   # Extract entities with reasoning
   entities = chain_of_thought(
       text,
       outputs=["entities"],
       description="Extract all named entities and their types",
   )

Multi-step Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import ModuleFactory, PipelineManager

   factory = ModuleFactory()
   pipeline = PipelineManager()

   pipeline.register_step(
       inputs=["document"],
       outputs=["key_points"],
       module=factory.create_module(
           inputs=["document"],
           outputs=["key_points"],
           description="Extract 3-5 key points from the document",
           module_type="chain_of_thought",
       ),
   )
   pipeline.register_step(
       inputs=["key_points"],
       outputs=["summary"],
       module=factory.create_module(
           inputs=["key_points"],
           outputs=["summary"],
           description="Create a concise executive summary",
       ),
   )
   pipeline.register_step(
       inputs=["key_points", "summary"],
       outputs=["action_items"],
       module=factory.create_module(
           inputs=["key_points", "summary"],
           outputs=["action_items"],
           description="Generate actionable next steps",
           module_type="chain_of_thought",
       ),
   )

   # Process a document
   document = "..."  # Your document here
   compiled_pipeline = pipeline.assemble_pipeline()
   results = compiled_pipeline(document=document)

Type-Guided Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import List, Dict
   from simpledspy import predict

   def analyze_sentiment(reviews: List[str]) -> Dict[str, float]:
       """Analyze sentiment of multiple reviews"""
       
       sentiments = []
       for review in reviews:
           # Type hints guide the output format
           sentiment_score = predict(
               review,
               outputs=["score"],
               description="Rate sentiment from 0.0 (negative) to 1.0 (positive)",
           )
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

   import dspy
   import simpledspy
   from simpledspy import predict

   # Enable logging to collect training data
   simpledspy.configure(logging_enabled=True, log_dir="./training_data")

   # Use modules normally - data is automatically collected
   for question in questions:
       answer = predict(question, name="qa_module")

   # Later, use collected data for optimization
   from simpledspy import ModuleFactory, OptimizationManager

   trainset = [
       dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
       dspy.Example(question="What is 3 * 3?", answer="9").with_inputs("question"),
   ]
   factory = ModuleFactory()
   module = factory.create_module(inputs=["question"], outputs=["answer"])
   optimizer = OptimizationManager()
   optimized_module = optimizer.optimize(module=module, trainset=trainset)

Custom Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import predict
   from simpledspy.evaluator import Evaluator

   # Create custom evaluation criteria
   evaluator = Evaluator()

   def evaluate_qa_response(question, answer, expected):
       score = evaluator.evaluate(
           inputs={"question": question, "expected": expected},
           outputs={"answer": answer},
           evaluation_instructions=[
               "Rate accuracy, completeness, and clarity from 1-10"
           ],
       )
       return score

   # Use in optimization
   results = []
   for q, expected_a in test_set:
       predicted_a = predict(q)
       score = evaluate_qa_response(q, predicted_a, expected_a)
       results.append(score)

Error Handling
--------------

Retry Logic
~~~~~~~~~~~

.. code-block:: python

   import simpledspy
   from simpledspy import predict

   # Configure retry behavior
   simpledspy.configure(retry_attempts=5, retry_delay=2.0)

   # Automatic retry on failures
   try:
       result = predict(complex_input)
   except Exception as e:
       print(f"Prediction failed: {e}")

Custom Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simpledspy import predict
   from simpledspy.exceptions import ValidationError, ModuleError

   try:
       # Input validation
       if not text:
           raise ValidationError("Input text cannot be empty")
       
       result = predict(text)
       
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
