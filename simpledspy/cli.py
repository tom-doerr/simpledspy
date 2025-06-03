#!/usr/bin/env python3
import sys
import argparse
import dspy
import json
from .module_factory import ModuleFactory

def main():
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument('inputs', nargs='*', help="Input strings to process (use - for stdin)")
    parser.add_argument('-d', '--description', help="Description of the processing task")
    parser.add_argument('-m', '--module', choices=['predict', 'chain_of_thought'], 
                       default='predict', help="DSPy module type to use")
    parser.add_argument('--optimize', action='store_true', help="Enable pipeline optimization")
    parser.add_argument('--strategy', choices=['bootstrap_few_shot', 'mipro'], 
                       default='bootstrap_few_shot', help="Optimization strategy")
    parser.add_argument('--max-demos', type=int, default=4, help="Maximum demonstrations for optimization")
    parser.add_argument('--json', action='store_true', help="Output in JSON format")
    args = parser.parse_args()
    
    # Use command-line arguments if present, otherwise check stdin
    if args.inputs:
        inputs = args.inputs
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
        inputs = [input_text]
    else:
        inputs = []
        
    # Create module factory
    factory = ModuleFactory()
    
    # Create module with proper signature
    input_names = [f"input_{i+1}" for i in range(len(inputs))]
    output_names = ["output"]
    module = factory.create_module(
        inputs=input_names,
        outputs=output_names,
        description=args.description
    )
    
    # Prepare inputs
    input_dict = {name: value for name, value in zip(input_names, inputs)}
    
    # Handle optimization if enabled
    if args.optimize:
        from .optimization_manager import OptimizationManager
        manager = OptimizationManager()
        manager.configure(strategy=args.strategy, max_bootstrapped_demos=args.max_demos)
        
        # Create simple trainset for demonstration (should be provided by user)
        trainset = [input_dict]
        module = manager.optimize(module, trainset)
    
    # Run prediction
    result = module(**input_dict)
    
    # Handle output
    output_data = {name: getattr(result, name) for name in output_names}
    
    # Format output
    if args.json:
        print(json.dumps(output_data))
    else:
        if len(output_data) == 1:
            print(next(iter(output_data.values())))
        else:
            for name, value in output_data.items():
                print(f"{name}: {value}")

if __name__ == "__main__":
    main()
