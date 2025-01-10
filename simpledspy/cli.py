#!/usr/bin/env python3
import sys
import argparse
from simpledspy import pipe
from .optimization_manager import OptimizationManager

def main():
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument('inputs', nargs='+', help="Input strings to process")
    parser.add_argument('-d', '--description', help="Description of the processing task")
    parser.add_argument('--optimize', action='store_true', help="Enable optimization")
    parser.add_argument('--strategy', choices=['bootstrap_few_shot', 'mipro'], 
                       default='bootstrap_few_shot', help="Optimization strategy")
    parser.add_argument('--max-demos', type=int, default=4, 
                       help="Maximum number of demonstrations")
    
    args = parser.parse_args()
    
    # Configure optimization if enabled
    if args.optimize:
        from simpledspy import OptimizationManager
        optimizer = OptimizationManager()
        optimizer.configure(
            strategy=args.strategy,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos
        )
        # TODO: Add training data loading and optimization
    
    # Process inputs
    result = pipe(*args.inputs, description=args.description)
    
    # Print results
    if isinstance(result, tuple):
        for i, res in enumerate(result, 1):
            print(f"Output {i}: {res}")
    else:
        print("Result:", result)

if __name__ == "__main__":
    main()
