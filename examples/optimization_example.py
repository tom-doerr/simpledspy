"""Optimization examples for SimpleDSPy."""

import dspy

from simpledspy import ModuleFactory, OptimizationManager, configure, predict
from simpledspy.evaluator import Evaluator


configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"), temperature=0.7, max_tokens=150)


def create_training_examples():
    """Create training examples for optimization."""
    return [
        dspy.Example(
            question="What is the capital of France?", answer="Paris"
        ).with_inputs("question"),
        dspy.Example(
            question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
        ).with_inputs("question"),
        dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
        dspy.Example(
            question="What is the largest planet in our solar system?", answer="Jupiter"
        ).with_inputs("question"),
        dspy.Example(
            question="What year did World War II end?", answer="1945"
        ).with_inputs("question"),
    ]


def basic_optimization_example():
    """Optimize a DSPy module created by ModuleFactory."""
    print("=== Basic Optimization Example ===")

    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["question"],
        outputs=["answer"],
        description="Answer factual questions concisely",
    )
    trainset = create_training_examples()

    optimizer = OptimizationManager()
    optimizer.configure(strategy="bootstrap_few_shot", max_labeled_demos=3)

    print("Optimizing module...")
    optimized_module = optimizer.optimize(module, trainset=trainset)

    test_question = "What is the speed of light?"
    result = optimized_module(question=test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {result.answer}\n")


def evaluation_example():
    """Use custom evaluation instructions."""
    print("=== Custom Evaluation Example ===")

    evaluator = Evaluator(
        evaluation_instruction=(
            "Rate the answer quality from 1-10 based on accuracy, "
            "completeness, and clarity"
        )
    )

    question = "Explain photosynthesis"
    answer1 = "Plants make food from sunlight"
    answer2 = (
        "Photosynthesis is the process by which plants convert light energy "
        "into chemical energy, using carbon dioxide and water to produce "
        "glucose and oxygen."
    )

    score1 = evaluator.evaluate(
        inputs={"question": question}, outputs={"answer": answer1}
    )
    score2 = evaluator.evaluate(
        inputs={"question": question}, outputs={"answer": answer2}
    )

    print(f"Question: {question}")
    print(f"Answer 1: {answer1}")
    print(f"Score: {score1}/10")
    print(f"Answer 2: {answer2}")
    print(f"Score: {score2}/10\n")


def available_optimization_strategies():
    """Show the supported optimization strategies."""
    print("=== Optimization Strategies ===")

    optimizer = OptimizationManager()
    for strategy in [
        "bootstrap_few_shot",
        "mipro",
        "bootstrap_random",
        "simba",
    ]:
        optimizer.configure(strategy=strategy)
        print(f"Configured strategy: {strategy}")
    print()


def logging_and_training_data():
    """Collect logged interactions for later training data curation."""
    print("=== Logging for Continuous Improvement ===")

    configure(
        lm=dspy.LM(model="openai/gpt-3.5-turbo"),
        logging_enabled=True,
        log_dir=".simpledspy",
    )

    question = "What causes rain?"
    answer = predict(
        question,
        name="weather_qa",
        description="Answer questions about weather phenomena",
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("This interaction was logged under .simpledspy/modules/weather_qa/.")

    configure(logging_enabled=False)
    print()


if __name__ == "__main__":
    basic_optimization_example()
    evaluation_example()
    available_optimization_strategies()
    logging_and_training_data()

    print("=== Optimization examples completed! ===")
