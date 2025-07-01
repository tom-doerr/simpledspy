# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
SimpleDSPy is a lightweight wrapper around DSPy that simplifies the creation of LLM pipelines through a reflection-based API. It allows users to build complex LLM workflows with minimal boilerplate code.

## Architecture

### Core Components
- **BaseCaller** (`base_caller.py`): Foundation class for all modules, handles DSPy module initialization and invocation
- **module_caller.py**: Creates specialized callers (ChainOfThoughtCaller, ProgrammaticCaller) with automatic signature generation
- **module_factory.py**: Factory pattern for creating different types of modules
- **pipeline_manager.py**: Manages multi-step LLM pipelines with sequential module execution
- **optimization_manager.py**: Handles optimization of pipelines using DSPy teleprompters
- **evaluator.py**: Evaluation system with 1-10 scoring for optimization
- **settings.py**: Global configuration management
- **logger.py**: Optional logging system that saves training data to `.simpledspy/` directory

### Key Design Patterns
- Reflection-based API that infers variable names from calling context
- Automatic signature generation from type hints
- Factory pattern for module creation
- Pipeline pattern for chaining operations

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module_caller.py

# Run tests matching pattern
pytest -k "test_chain"

# Run with coverage
pytest --cov=simpledspy
```

### Linting & Formatting
```bash
# Format code
black .

# Check formatting
black --check .

# Type checking (optional)
mypy simpledspy
```

### Building & Publishing
```bash
# Build package
poetry build

# Publish to PyPI
poetry publish
```

### Environment Setup
```bash
# Create virtual environment and install dependencies
./setup_env.sh
```

## Testing Approach
- Unit tests for all major components using pytest
- Mock DSPy objects to test SimpleDSPy logic in isolation
- CI/CD runs tests on Python 3.9, 3.10, and 3.11
- Black formatting is enforced in CI

## Important Implementation Details

### Logging System
- When `settings.enable_logging = True`, creates `.simpledspy/` directory
- Logs are stored in `.simpledspy/logs/`
- Training data saved to `.simpledspy/training_data/`
- Each module call generates a timestamped JSON file with inputs/outputs

### Module Creation Pattern
```python
# SimpleDSPy uses reflection to infer variable names
question = "What is the capital of France?"
answer = module_caller.call_chain_of_thought()  # Automatically uses 'question' as input
```

### Pipeline Usage
```python
# Pipelines chain multiple modules
pipeline = PipelineManager()
pipeline.add_step("step1", module1)
pipeline.add_step("step2", module2)
result = pipeline.run(initial_input)
```

### Type Hints for Signatures
- When provided, type hints automatically generate DSPy signatures
- Example: `def process(text: str) -> summary: str` creates signature "text -> summary"

## CI/CD Configuration
- GitHub Actions workflow in `.github/workflows/python-package.yml`
- Runs on push to main/master and pull requests
- Tests multiple Python versions
- Enforces black formatting
- Publishes to PyPI on version tags