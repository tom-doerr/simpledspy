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

- Automatic module creation from input/output names
- Pipeline management and step tracking
- Clean, minimal API
- Built-in caching and configuration
- Type hints and documentation
- Support for different module types (Predict, ChainOfThought)

## Installation

```bash
pip install simpledspy
```

## Quick Start

```python
from simpledspy import predict, chain_of_thought

# Basic text processing
cleaned_text = predict("Some messy   text with extra spaces")
print(cleaned_text)  # "Some messy text with extra spaces"

# Multiple inputs/outputs
name, age = predict("John Doe, 30 years old")
print(name)  # "John Doe"
print(age)   # 30

# Chain of thought reasoning
result = chain_of_thought(
    "If I have 5 apples and eat 2, how many do I have left?",
    description="Reason step by step"
)
print(result)  # "3"
```

## How It Works

The `predict` and `chain_of_thought` functions automatically:
1. Detect input variable names
2. Create appropriate DSPy modules
3. Track pipeline steps
4. Return processed outputs

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

MIT License
