"""Optimization example for SimpleDSPy - Improve module performance"""

import dspy
from simpledspy import predict, configure, OptimizationManager
from simpledspy.evaluator import Evaluator

# Configure language model
configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"), temperature=0.7, max_tokens=150)


def create_training_examples():
    """Create training examples for optimization"""
    # In practice, you would load these from a dataset
    training_examples = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
        {"question": "What is 2 + 2?", "answer": "4"},
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
        },
        {"question": "What year did World War II end?", "answer": "1945"},
    ]
    return training_examples


def basic_optimization_example():
    """Example: Basic module optimization"""
    print("=== Basic Optimization Example ===")

    # Create a question-answering module
    def qa_module(question: str) -> str:
        answer = predict(question, outputs=["answer"])
        return answer

    # Create training data
    trainset = create_training_examples()

    # Initialize optimization manager
    optimizer = OptimizationManager()

    # Configure optimization settings
    optimizer.configure(
        strategy="bootstrap",  # or "simba" for more advanced optimization
        n_iterations=3,
        n_demos=3,
    )

    # Optimize the module
    print("Optimizing module...")
    optimized_module = optimizer.optimize_module(
        module=qa_module, trainset=trainset, input_key="question", output_key="answer"
    )

    # Test the optimized module
    test_question = "What is the speed of light?"
    print(f"\nTest Question: {test_question}")

    # Original module
    original_answer = qa_module(test_question)
    print(f"Original Answer: {original_answer}")

    # Note: In practice, the optimized module would be used differently
    # This is a simplified example
    print("\nOptimization complete! The module now has improved prompts.")
    print()


def evaluation_example():
    """Example: Using custom evaluation criteria"""
    print("=== Custom Evaluation Example ===")

    # Create an evaluator with custom instructions
    evaluator = Evaluator(
        evaluation_instruction="Rate the answer quality from 1-10 based on accuracy, completeness, and clarity"
    )

    # Example question and answers
    question = "Explain photosynthesis"

    answer1 = "Plants make food from sunlight"
    answer2 = "Photosynthesis is the process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen"

    # Evaluate answers
    score1 = evaluator.evaluate(
        inputs={"question": question}, outputs={"answer": answer1}
    )

    score2 = evaluator.evaluate(
        inputs={"question": question}, outputs={"answer": answer2}
    )

    print(f"Question: {question}")
    print(f"\nAnswer 1: {answer1}")
    print(f"Score: {score1}/10")
    print(f"\nAnswer 2: {answer2}")
    print(f"Score: {score2}/10")
    print()


def optimization_with_validation():
    """Example: Optimization with validation set"""
    print("=== Optimization with Validation ===")

    # Split data into train and validation
    all_examples = create_training_examples()
    trainset = all_examples[:3]
    valset = all_examples[3:]

    # Create module to optimize
    def fact_checker(statement: str) -> str:
        verification = predict(
            statement,
            outputs=["verification"],
            description="Verify if this statement is true or false and explain why",
        )
        return verification

    # Configure optimization
    optimizer = OptimizationManager()
    optimizer.configure(strategy="bootstrap", n_iterations=2, n_demos=2)

    print("Training examples:")
    for ex in trainset:
        print(f"  Q: {ex['question']} -> A: {ex['answer']}")

    print("\nValidation examples:")
    for ex in valset:
        print(f"  Q: {ex['question']} -> A: {ex['answer']}")

    print("\nOptimization would improve the module's performance on similar tasks.")
    print()


def advanced_optimization_strategies():
    """Example: Different optimization strategies"""
    print("=== Advanced Optimization Strategies ===")

    optimizer = OptimizationManager()

    # Bootstrap strategy (default)
    print("1. Bootstrap Strategy:")
    print("   - Creates diverse prompts through bootstrapping")
    print("   - Good for general-purpose optimization")
    optimizer.configure(strategy="bootstrap", n_iterations=5)

    # SIMBA strategy (if available)
    print("\n2. SIMBA Strategy:")
    print("   - State-of-the-art optimization using Bayesian methods")
    print("   - Better for complex tasks requiring nuanced understanding")
    try:
        optimizer.configure(strategy="simba", n_iterations=10)
    except ValueError:
        print("   - Note: SIMBA may require additional dependencies")

    print("\nChoose strategy based on your task complexity and requirements.")
    print()


def logging_and_training_data():
    """Example: Using logged data for continuous improvement"""
    print("=== Logging for Continuous Improvement ===")

    # Enable logging to collect training data
    configure(
        lm=dspy.LM(model="openai/gpt-3.5-turbo"),
        logging_enabled=True,
        log_dir=".simpledspy",  # Training data will be saved here
    )

    # Use modules with logging enabled
    question = "What causes rain?"
    answer = predict(
        question,
        name="weather_qa",  # Name helps organize logged data
        description="Answer questions about weather phenomena",
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\nThis interaction has been logged and can be used for future optimization.")
    print("Check .simpledspy/modules/weather_qa/ for logged data.")

    # Disable logging when not needed
    configure(logging_enabled=False)
    print()


if __name__ == "__main__":
    # Run optimization examples
    basic_optimization_example()
    evaluation_example()
    optimization_with_validation()
    advanced_optimization_strategies()
    logging_and_training_data()

    print("=== Optimization examples completed! ===")
    print("Optimization helps improve module performance on your specific tasks.")
