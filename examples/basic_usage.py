"""Basic usage examples for SimpleDSPy"""

import dspy
from simpledspy import predict, chain_of_thought, configure

# Configure with your preferred language model
# You can use any model supported by DSPy
configure(
    lm=dspy.LM(model="openai/gpt-3.5-turbo"),  # or "anthropic/claude-3", etc.
    temperature=0.7,
    max_tokens=150,
)


def basic_examples():
    """Demonstrate basic predict and chain_of_thought usage"""
    print("=== Basic Predict Example ===")

    # Simple prediction - variable names are automatically inferred
    question = "What is the capital of France?"
    answer = predict(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

    # Chain of thought reasoning
    print("=== Chain of Thought Example ===")
    problem = "If a train travels 120 miles in 2 hours, what is its average speed?"
    solution = chain_of_thought(problem)
    print(f"Problem: {problem}")
    print(f"Solution: {solution}\n")


def custom_names_example():
    """Demonstrate custom input/output names"""
    print("=== Custom Names Example ===")

    # You can specify custom names for inputs and outputs
    text = "The quick brown fox jumps over the lazy dog."
    sentiment = predict(text, inputs=["text"], outputs=["sentiment"])
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")


def multiple_outputs_example():
    """Demonstrate multiple outputs"""
    print("=== Multiple Outputs Example ===")

    # Generate multiple outputs in one call
    article = """
    Python is a high-level programming language known for its simplicity 
    and readability. It was created by Guido van Rossum and released in 1991.
    """

    title, summary, keywords = predict(
        article, inputs=["article"], outputs=["title", "summary", "keywords"]
    )

    print(f"Article: {article.strip()}")
    print(f"Title: {title}")
    print(f"Summary: {summary}")
    print(f"Keywords: {keywords}\n")


def description_example():
    """Demonstrate using descriptions to guide the model"""
    print("=== Description Example ===")

    # Add descriptions to provide context
    code = """
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    """

    explanation = predict(
        code,
        inputs=["code"],
        outputs=["explanation"],
        description="Explain this Python code in simple terms for beginners",
    )

    print(f"Code: {code.strip()}")
    print(f"Explanation: {explanation}\n")


def type_hints_example():
    """Demonstrate type hints for better results"""
    print("=== Type Hints Example ===")

    # Define a function with type hints
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment with type hints"""
        sentiment = predict(text)
        return sentiment

    review = "This product exceeded my expectations! Highly recommended."
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}\n")


if __name__ == "__main__":
    # Run all examples
    basic_examples()
    custom_names_example()
    multiple_outputs_example()
    description_example()
    type_hints_example()

    print("=== Examples completed! ===")
    print("Check out the other example files for more advanced usage:")
