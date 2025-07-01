# SimpleDSPy Examples

This directory contains example code demonstrating various features and use cases of SimpleDSPy.

## Getting Started

Before running any examples, make sure you have:

1. Installed SimpleDSPy: `pip install simpledspy`
2. Configured your language model credentials (e.g., OpenAI API key)

## Examples Overview

### 1. [basic_usage.py](basic_usage.py)
Learn the fundamentals of SimpleDSPy:
- Simple predictions with automatic variable name inference
- Chain of thought reasoning
- Custom input/output names
- Multiple outputs in a single call
- Using descriptions to guide the model
- Type hints for better results

### 2. [pipeline_example.py](pipeline_example.py)
Build multi-step processing pipelines:
- Translation pipeline (multi-language round-trip)
- Text analysis pipeline (extract, summarize, analyze)
- Data processing pipeline (parse, clean, enrich)
- Pipeline inspection and assembly

### 3. [optimization_example.py](optimization_example.py)
Optimize your modules for better performance:
- Basic module optimization with training data
- Custom evaluation criteria
- Different optimization strategies (Bootstrap, SIMBA)
- Using logged data for continuous improvement
- Validation sets for testing

### 4. [advanced_usage.py](advanced_usage.py)
Advanced techniques and patterns:
- Complex type hints with nested structures
- Context preservation across multiple calls
- Custom LM parameters per call
- Programmatic module creation
- Error handling and input validation
- Batch processing for efficiency
- Dynamic output handling

## Running the Examples

Each example file can be run independently:

```bash
python examples/basic_usage.py
python examples/pipeline_example.py
python examples/optimization_example.py
python examples/advanced_usage.py
```

## Customizing Examples

Feel free to modify these examples for your use case:

1. **Change the Language Model**: Update the `configure()` call to use your preferred model:
   ```python
   configure(lm=dspy.LM(model="anthropic/claude-3"))
   ```

2. **Adjust Parameters**: Modify temperature, max_tokens, and other parameters:
   ```python
   configure(temperature=0.5, max_tokens=300)
   ```

3. **Add Your Data**: Replace example data with your own datasets and use cases.

## Best Practices

1. **Variable Naming**: Use descriptive variable names - SimpleDSPy infers context from them
2. **Type Hints**: Add type hints to your functions for better results
3. **Descriptions**: Use the `description` parameter to provide context
4. **Logging**: Enable logging to collect training data for future optimization
5. **Error Handling**: Always validate inputs and handle potential errors

## Need Help?

- Check the [SimpleDSPy documentation](https://github.com/thompsonjeff/simpledspy)
- Review the source code for more details on available features
- Open an issue on GitHub for bugs or feature requests

Happy coding with SimpleDSPy!