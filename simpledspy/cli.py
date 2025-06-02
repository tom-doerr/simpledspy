#!/usr/bin/env python3
import sys
import argparse
import dspy
from .module_factory import ModuleFactory

def main():
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument('inputs', nargs='*', help="Input strings to process (use - for stdin)")
    parser.add_argument('-d', '--description', help="Description of the processing task")
    parser.add_argument('-m', '--module', choices=['predict', 'chain_of_thought'], 
                       default='chain_of_thought', help="DSPy module type to use")
    parser.add_argument('--optimize', action='store_true', help="Enable pipeline optimization")
    parser.add_argument('--strategy', choices=['bootstrap_few_shot', 'mipro'], 
                       default='bootstrap_few_shot', help="Optimization strategy")
    parser.add_argument('--max-demos', type=int, default=4, help="Maximum demonstrations for optimization")
    args = parser.parse_args()
    
    # Check if stdin has data
    if not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
        inputs = [input_text]
    else:
        inputs = args.inputs
        
    # Create module factory
    factory = ModuleFactory()
    
    # Create module with proper signature
    input_names = [f"input_{i+1}" for i in range(len(inputs))]
    module = factory.create_module(
        inputs=input_names,
        outputs=["output"],
        description=args.description
    )
    
    # Prepare inputs
    input_dict = {name: value for name, value in zip(input_names, inputs)}
    
    # Run prediction
    result = module(**input_dict)
    
    # Print result
    print(result.output)

if __name__ == "__main__":
    main()
