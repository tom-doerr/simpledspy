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
  <a href="https://github.com/tomdoerr/simpledspy/actions/workflows/ci.yml">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=CI&message=Passing&color=green" alt="CI">
  </a>
</p>

<p align="center">
  SimpleDSPy is a lightweight Python library that simplifies building and running DSPy pipelines with an intuitive interface.
</p>

> **⚠️ Careful, vibe coded project!**

> **Note:** This project is currently a work in progress and is not officially affiliated with or endorsed by the DSPy project. It is an independent effort to create a simplified interface for working with DSPy pipelines.

> **Important:** Always import this package as `import simpledspy`, not as `import dspy`. The latter will import the actual DSPy library instead of SimpleDSPy.

## Features

- **Predict and ChainOfThought Modules**: Easy-to-use function calls for predictions
- **Pipeline Manager**: Create complex pipelines of DSPy modules
- **Optimization**: Use DSPy teleprompters to optimize your modules
- **Evaluation**: Evaluate outputs on a scale of 1-10 and log inputs/outputs
- **Reward Tracking**: Track cumulative discounted rewards over time
- **Advice Generation**: Generate optimization advice from positive/negative examples
- **CLI**: Command-line interface for running modules and pipelines
- **Automatic Module Creation**: Generate DSPy modules from input/output specifications
- **Type Hinting**: Support for flexible type annotations
- **Logging**: Built-in logging of inputs/outputs for training data collection

## Installation

```bash
pip install simpledspy
```

## Configuration

Before using SimpleDSPy, configure your language model:

```python
import dspy
dspy.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo'))
```

## Usage

### Basic Prediction
```python
from simpledspy import predict

# Basic prediction with default parameters
result = predict("Hello, world!")
print(result)

# Set LLM parameters directly in the call
result = predict("Write a short greeting", 
                lm_params={"max_tokens": 50, "temperature": 0.7})
print(result)
```

### Chain of Thought
```python
from simpledspy import chain_of_thought

# Basic chain of thought with default parameters
result = chain_of_thought("What is the capital of France?", 
                         description="Reason step by step")
print(result)

# Set temperature directly in the call
result = chain_of_thought("Solve this complex math problem", 
                         description="Show reasoning steps",
                         lm_params={"temperature": 0.3})
print(result)
```

### Setting LLM Parameters
You can set LLM parameters directly in module calls using the `lm_params` argument. Supported parameters include:
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature (0.0-1.0)
- `top_p`: Nucleus sampling probability
- `n`: Number of completions to generate
- `stop`: Stop sequences
- ...and other model-specific parameters

Example:
```python
result = predict("Generate creative text", 
                lm_params={
                    "max_tokens": 100,
                    "temperature": 0.9,
                    "top_p": 0.95
                })
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

### Reward Tracking and Advice Generation
```python
from simpledspy.evaluator import Evaluator

# Create evaluator with reward group
evaluator = Evaluator("Rate output quality 1-10", reward_group="math")
score = evaluator.evaluate(
    {"question": "2+2"},
    {"answer": "4"}
)
print(score)  # 10

# Get cumulative reward
cumulative = evaluator.get_cumulative_reward()
print(f"Cumulative reward: {cumulative:.2f}")

# Generate advice from examples
advice = evaluator.get_advice()
print(f"Optimization advice: {advice}")
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
1. **High-level API**: `predict()` and `chain_of_thought()` functions
2. **Pipeline Management**: Multi-step DSPy pipelines
3. **Optimization**: Module optimization with training data
4. **Evaluation**: 1-10 scoring with custom instructions
5. **Logging**: Input/output logging for training data
6. **Reward Function**: `evaluate()` method for cumulative rewards
7. **Training Data**: Extract high-scoring examples from logs

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Publishing the Package

To publish a new version of SimpleDSPy:

1. Update the version in `pyproject.toml` and `simpledspy/__init__.py`
2. Build the package:
   ```bash
   poetry build
   ```
3. Publish using Poetry's built-in command:
   ```bash
   poetry publish
   ```
   Poetry will prompt you for your PyPI credentials

4. Verify the published package:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple simpledspy==0.3.0
   ```
   Replace 0.3.0 with your actual version number

For API token authentication:
```bash
poetry config pypi-token.pypi your-api-token
```
Then run `poetry publish` without credentials prompt

## License

MIT License
