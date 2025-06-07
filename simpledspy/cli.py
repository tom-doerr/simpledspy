#!/usr/bin/env python3
"""Command Line Interface for SimpleDSPy

Features:
- Run DSPy modules and pipelines
- Enable optimization with different strategies
- Evaluate outputs with custom instructions
- Generate training data from logs

Note: You must configure your language model before use.
See documentation for configuration instructions.
"""
import sys
import argparse
import json
from .module_factory import ModuleFactory
from .evaluator import Evaluator
from .optimization_manager import OptimizationManager
from .pipeline_manager import PipelineManager

def main():
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument('inputs', nargs='*', help="Input strings to process (use - for stdin)")
    parser.add_argument('-d', '--description', help="Description of the processing task")
    parser.add_argument('-m', '--module', choices=['predict', 'chain_of_thought'], 
                       default='predict', help="DSPy module type to use")
    parser.add_argument('--optimize', action='store_true', help="Enable pipeline optimization")
    parser.add_argument('--strategy', choices=['bootstrap_few_shot', 'mipro', 'bootstrap_random', 'simba'], 
                       default='bootstrap_few_shot', help="DSPy optimization algorithm")
    parser.add_argument('--max-demos', type=int, default=4, help="Maximum demonstrations for optimization")
    parser.add_argument('--trainset', type=str, help="Path to JSON file containing training set")
    parser.add_argument('--json', action='store_true', help="Output in JSON format")
    parser.add_argument('--pipeline', nargs='+', help="Run a pipeline with multiple step descriptions")
    parser.add_argument('--evaluation-instruction', type=str, default="", 
                       help="Instruction for evaluating outputs on a 1-10 scale")
    parser.add_argument('--log-file', type=str, default="dspy_logs.jsonl", 
                       help="File to store input/output logs")
    args = parser.parse_args()
    
    # Use command-line arguments if present, otherwise check stdin
    if args.inputs:
        inputs = args.inputs
    elif not sys.stdin.isatty():
        inputs = [sys.stdin.read().strip()]
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
    
    # Setup evaluator if evaluation instruction provided
    if args.evaluation_instruction:
        from .evaluator import Evaluator
        evaluator = Evaluator(evaluation_instruction=args.evaluation_instruction, 
                             log_file=args.log_file)
    else:
        evaluator = None
    
    # Prepare inputs
    input_dict = dict(zip(input_names, inputs))
    
    # Handle optimization if enabled
    if args.optimize:
        manager = OptimizationManager()
        manager.configure(strategy=args.strategy, max_bootstrapped_demos=args.max_demos)
        
        if args.trainset:
            try:
                with open(args.trainset, 'r', encoding='utf-8') as f:
                    trainset = json.load(f)
            except OSError as e:
                print(f"Error loading trainset: {e}")
                sys.exit(1)
        else:
            print("Warning: Using single input as trainset. Provide --trainset for better optimization.")
            trainset = [input_dict]
            
        module = manager.optimize(module, trainset)
    
    # Handle pipeline execution if enabled
    if args.pipeline:
        manager = PipelineManager()
        manager.reset()
        
        # Create steps from descriptions
        for i, desc in enumerate(args.pipeline):
            # For first step, use the main input names
            if i == 0:
                step_inputs = input_names
            # For subsequent steps, use previous step's output
            else:
                step_inputs = [f"output_{i}"]
                
            step_outputs = [f"output_{i+1}"]
            
            step_module = factory.create_module(
                inputs=step_inputs,
                outputs=step_outputs,
                description=desc
            )
                
            manager.register_step(
                inputs=step_inputs,
                outputs=step_outputs,
                module=step_module
            )
        
        pipeline = manager.assemble_pipeline()
        result = pipeline(**input_dict)
        
        # Get final output - use the output name from the pipeline step
        output_name = f"output_{len(args.pipeline)}"
        output_value = getattr(result, output_name)
        output_data = {output_name: output_value}
    else:
        # Run single module prediction
        result = module(**input_dict)
        output_data = {name: getattr(result, name) for name in output_names}
    
    # Format output as string representations
    if args.json:
        print(json.dumps(output_data))
    else:
        # For pipelines we get a dictionary of outputs - we want just the values
        # Convert all values to strings by applying str()
        output_lines = []
        output_values = []
        for name, value in output_data.items():
            # Handle objects that have the expected attribute
            if hasattr(value, name):
                actual_value = getattr(value, name)
            else:
                actual_value = value
                    
            # Get the string representation
            str_value = str(actual_value)
            output_lines.append(f"{name}: {str_value}")
            output_values.append(str_value)

        # For single output: print the value directly
        if len(output_values) == 1:
            print(output_values[0])
        else:
            for line in output_lines:
                print(line)
    
    # Log with evaluation if evaluator is set
    if evaluator:
        evaluator.log_with_evaluation(
            module=args.module,
            inputs=input_dict,
            outputs=output_data,
            description=args.description
        )

if __name__ == "__main__":
    main()
