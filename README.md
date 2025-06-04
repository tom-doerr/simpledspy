<h1 align="center">SimpleDSPy</h1>

<p align="center">
  <a href="https://pypi.org/project/simpledspy/">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=PyPI&message=simpledspy&color=blue" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/simpledspy/">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=Python&message=3.9+%7C+3.10+%7C+3.11&color=blue" alt="Python Version">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=License&message=MIT&color=blue" alt="License: MIT">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=Code%20Style&message=Black&color=black" alt="Code style: black">
  </a>
  <a href="https://github.com/tomdoerr/simpledspy/actions/workflows/tests.yml">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=Tests&message=Passing&color=green" alt="Tests">
  </a>
</p>

<p align="center">
  SimpleDSPy is a lightweight Python library that simplifies building and running DSPy pipelines with an intuitive interface.
</p>

> **Note:** This project is currently a work in progress and is not officially affiliated with or endorsed by the DSPy project. It is an independent effort to create a simplified interface for working with DSPy pipelines.

> **Important:** Always import this package as `import simpledspy`, not as `import dspy`. The latter will import the actual DSPy library instead of SimpleDSPy.

## Features

- **Predict and ChainOfThought Modules**: Easy-to-use function calls for predictions
- **Pipeline Manager**: Create complex pipelines of DSPy modules
- **Optimization**: Use DSPy teleprompters to optimize your modules
- **Evaluation**: Evaluate outputs on a scale of 1-10 and log inputs/outputs
- **CLI**: Command-line interface for running modules and pipelines
- **Automatic Module Creation**: Generate DSPy modules from input/output specifications
- **Type Hinting**: Support for flexible type annotations
- **Logging**: Built-in logging of inputs/outputs for training data collection

## Installation

```bash
pip install simpledspy
```

## Usage

### Basic Prediction
```python
from simpledspy import predict

result = predict("Hello, world!")
print(result)
```

### Chain of Thought
```python
from simpledspy import chain_of_thought

result = chain_of_thought("What is the capital of France?", 
                         description="Reason step by step")
print(result)
```

### Pipeline Management
```python
from simpledspy import PipelineManager, predict

manager = PipelineManager()
manager.register_step(
    inputs=["text"],
    outputs=["cleaned"],
    module=predict("Clean text")
)
manager.register_step(
    inputs=["cleaned"],
    outputs=["sentiment"],
    module=predict("Analyze sentiment")
)

pipeline = manager.assemble_pipeline()
result = pipeline(text="I love this product!")
print(result.sentiment)  # "positive"
```

### Optimization
```python
from simpledspy import predict
from simpledspy.optimization_manager import OptimizationManager

# Create training data
trainset = [
    {"input": "2+2", "output": "4"},
    {"input": "3*3", "output": "9"}
]

# Optimize the module
manager = OptimizationManager()
optimized_predict = manager.optimize(predict, trainset)

# Use optimized module
result = optimized_predict("4*4")
print(result)  # "16"
```

### Evaluation
```python
from simpledspy.evaluator import Evaluator

evaluator = Evaluator("Rate output quality 1-10")
score = evaluator.evaluate(
    {"question": "2+2"},
    {"answer": "4"}
)
print(score)  # 10
```

## CLI Usage

```bash
# Basic prediction
python -m simpledspy.cli "Hello, world!" --description "Extract greeting"

# Chain of thought reasoning
python -m simpledspy.cli "What is 2+2?" --module chain_of_thought --description "Reason step by step"

# Enable optimization
python -m simpledspy.cli "Solve math problem" --optimize --trainset data/trainset.json

# Evaluate outputs
python -m simpledspy.cli "What is the capital of France?" --evaluation-instruction "Check if answer is correct"

# Build pipelines
python -m simpledspy.cli "Input text" --pipeline "Clean text" "Analyze sentiment" "Summarize"
```

## How It Works

SimpleDSPy provides:
1. **High-level API**: `predict()` and `chain_of_thought()` functions handle module creation
2. **Pipeline Management**: Track and assemble multi-step pipelines
3. **Optimization**: Improve module performance with training data
4. **Evaluation**: Score outputs using custom instructions
5. **Logging**: Record inputs/outputs for analysis and training

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

MIT License
