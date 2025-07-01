"""Advanced usage examples for SimpleDSPy"""

import dspy
from typing import List, Optional, Dict
from simpledspy import predict, chain_of_thought, configure

# Configure language model
configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"), temperature=0.7, max_tokens=200)


def type_hints_advanced():
    """Advanced type hints for complex scenarios"""
    print("=== Advanced Type Hints ===")

    # Complex type hints are automatically handled
    def analyze_data(
        data: List[Dict[str, float]],
    ) -> tuple[str, List[str], Optional[float]]:
        """Analyze data with complex type hints"""
        summary, insights, confidence = predict(data)
        return summary, insights, confidence

    # Example data
    sales_data = [
        {"month": "Jan", "revenue": 100000, "growth": 0.15},
        {"month": "Feb", "revenue": 110000, "growth": 0.10},
        {"month": "Mar", "revenue": 125000, "growth": 0.14},
    ]

    summary, insights, confidence = analyze_data(sales_data)
    print(f"Data: {sales_data}")
    print(f"Summary: {summary}")
    print(f"Insights: {insights}")
    print(f"Confidence: {confidence}")
    print()


def context_preservation():
    """Example: Preserving context across calls"""
    print("=== Context Preservation ===")

    # SimpleDSPy can maintain context through variable names
    context = "We are discussing artificial intelligence and its impact on society."

    # First question
    question1 = "What are the main benefits?"
    answer1 = predict(
        context, question1, inputs=["context", "question1"], outputs=["answer1"]
    )
    print(f"Context: {context}")
    print(f"Q1: {question1}")
    print(f"A1: {answer1}")

    # Follow-up question (context is preserved through naming)
    question2 = "What about the risks?"
    answer2 = predict(
        context,
        answer1,
        question2,
        inputs=["context", "answer1", "question2"],
        outputs=["answer2"],
    )
    print(f"Q2: {question2}")
    print(f"A2: {answer2}")
    print()


def custom_module_parameters():
    """Example: Using custom LM parameters per call"""
    print("=== Custom Module Parameters ===")

    text = "The quantum computer successfully demonstrated supremacy by solving a problem in 200 seconds that would take classical computers 10,000 years."

    # Creative summary (high temperature)
    creative_summary = predict(
        text,
        outputs=["summary"],
        description="Create a creative, engaging summary",
        lm_params={"temperature": 0.9},
    )

    # Factual summary (low temperature)
    factual_summary = predict(
        text,
        outputs=["summary"],
        description="Create a precise, factual summary",
        lm_params={"temperature": 0.1},
    )

    print(f"Original: {text}")
    print(f"\nCreative Summary: {creative_summary}")
    print(f"Factual Summary: {factual_summary}")
    print()


def programmatic_module_creation():
    """Example: Creating modules programmatically"""
    print("=== Programmatic Module Creation ===")

    # You can create modules dynamically based on runtime conditions
    def create_translator(target_language: str):
        """Factory function to create language-specific translators"""

        def translator(text: str) -> str:
            translation = predict(
                text,
                target_language,
                inputs=["text", "target_language"],
                outputs=["translation"],
                description=f"Translate the text to {target_language}",
            )
            return translation

        return translator

    # Create specific translators
    french_translator = create_translator("French")
    spanish_translator = create_translator("Spanish")

    text = "Hello, how are you?"
    print(f"Original: {text}")
    print(f"French: {french_translator(text)}")
    print(f"Spanish: {spanish_translator(text)}")
    print()


def error_handling_and_validation():
    """Example: Robust error handling and validation"""
    print("=== Error Handling and Validation ===")

    def safe_predict(text: str, task: str) -> Optional[str]:
        """Predict with error handling"""
        try:
            # Validate inputs
            if not text or not text.strip():
                return "Error: Empty input text"

            if len(text) > 1000:
                return "Error: Text too long (max 1000 characters)"

            # Perform prediction
            result = predict(
                text,
                task,
                inputs=["text", "task"],
                outputs=["result"],
                lm_params={"max_tokens": 100},
            )

            # Validate output
            if not result or result.strip() == "":
                return "Error: Empty response"

            return result

        except Exception as e:
            return f"Error: {str(e)}"

    # Test cases
    test_cases = [
        ("Analyze this text", "sentiment analysis"),
        ("", "summary"),  # Empty input
        ("Short", "expand into paragraph"),
    ]

    for text, task in test_cases:
        result = safe_predict(text, task)
        print(f"Input: '{text}' | Task: '{task}'")
        print(f"Result: {result}")
        print()


def batch_processing():
    """Example: Efficient batch processing"""
    print("=== Batch Processing ===")

    # Process multiple items efficiently
    emails = [
        "Thanks for your help with the project. The results look great!",
        "I'm disappointed with the service. This is unacceptable.",
        "Could you please send me the report by Friday?",
        "Great job on the presentation! Very impressed.",
    ]

    # Process in batch
    results = []
    for email in emails:
        sentiment, priority = predict(
            email,
            outputs=["sentiment", "priority"],
            description="Analyze email sentiment and assign priority (high/medium/low)",
        )
        results.append(
            {
                "email": email[:50] + "...",  # Truncate for display
                "sentiment": sentiment,
                "priority": priority,
            }
        )

    # Display results
    print("Email Batch Analysis:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['email']}")
        print(f"   Sentiment: {result['sentiment']}, Priority: {result['priority']}")
    print()


def dynamic_output_handling():
    """Example: Dynamic output based on input"""
    print("=== Dynamic Output Handling ===")

    def analyze_content(content: str, analysis_type: str) -> Dict:
        """Dynamically determine outputs based on analysis type"""

        if analysis_type == "sentiment":
            sentiment, confidence = predict(
                content,
                outputs=["sentiment", "confidence"],
                description="Analyze sentiment with confidence score",
            )
            return {"sentiment": sentiment, "confidence": confidence}

        elif analysis_type == "summary":
            summary, key_points = predict(
                content,
                outputs=["summary", "key_points"],
                description="Create summary and extract key points",
            )
            return {"summary": summary, "key_points": key_points}

        elif analysis_type == "full":
            sentiment, summary, key_points, recommendation = predict(
                content,
                outputs=["sentiment", "summary", "key_points", "recommendation"],
                description="Comprehensive analysis with all aspects",
            )
            return {
                "sentiment": sentiment,
                "summary": summary,
                "key_points": key_points,
                "recommendation": recommendation,
            }
        else:
            return {"error": "Unknown analysis type"}

    # Example content
    content = """
    The new product launch was highly successful, exceeding our initial projections 
    by 40%. Customer feedback has been overwhelmingly positive, with particular 
    praise for the innovative features and user-friendly design.
    """

    # Different analysis types
    for analysis_type in ["sentiment", "summary", "full"]:
        print(f"\nAnalysis Type: {analysis_type}")
        result = analyze_content(content, analysis_type)
        for key, value in result.items():
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    # Run advanced examples
    type_hints_advanced()
    context_preservation()
    custom_module_parameters()
    programmatic_module_creation()
    error_handling_and_validation()
    batch_processing()
    dynamic_output_handling()

    print("=== Advanced examples completed! ===")
    print("These examples show more sophisticated uses of SimpleDSPy.")
