#!/usr/bin/env python3
import sys
import argparse
from simpledspy import pipe
from .optimization_manager import OptimizationManager

def main():
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument('inputs', nargs='*', help="Input strings to process (use - for stdin)")
    parser.add_argument('-d', '--description', help="Description of the processing task")
    parser.add_argument('--optimize', action='store_true', help="Enable optimization")
    parser.add_argument('--strategy', choices=['bootstrap_few_shot', 'mipro'], 
                       default='bootstrap_few_shot', help="Optimization strategy")
    parser.add_argument('--max-demos', type=int, default=4, 
                       help="Maximum number of demonstrations")
    parser.add_argument('--metric', type=str,
                       help="Custom metric function to use for optimization")
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
    
    # Check if stdin has data
    if not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
        inputs = [input_text]
    else:
        inputs = args.inputs
        
    # Process inputs
    if len(inputs) == 1:
        answer = pipe(inputs[0], description=args.description)
    else:
        answer = pipe(*inputs, description=args.description)
    
    # Print results
    if isinstance(answer, tuple):
        for res in answer:
            print(res)
    else:
        print(answer)

if __name__ == "__main__":
    main()
