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


def _run_pipeline(args, factory, input_names, input_dict):
    """Run a pipeline of modules."""
    manager = PipelineManager()
    manager.reset()

    for i, desc in enumerate(args.pipeline):
        step_inputs = input_names if i == 0 else [f"output_{i}"]
        step_outputs = [f"output_{i+1}"]

        step_module = factory.create_module(
            inputs=step_inputs, outputs=step_outputs, description=desc
        )

        manager.register_step(
            inputs=step_inputs, outputs=step_outputs, module=step_module
        )

    pipeline = manager.assemble_pipeline()
    result = pipeline(**input_dict)
    output_name = f"output_{len(args.pipeline)}"
    output_value = getattr(result, output_name)
    return {output_name: output_value}


def _run_single_module(module, input_dict, output_names):
    """Run a single module."""
    result = module(**input_dict)
    return {name: getattr(result, name) for name in output_names}


def main():  # pylint: disable=too-many-branches,too-many-statements
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="SimpleDSPy command line interface")
    parser.add_argument(
        'inputs', nargs='*', help="Input strings to process (use - for stdin)"
    )
    parser.add_argument(
        '-d', '--description', help="Description of the processing task"
    )
    parser.add_argument(
        '-m', '--module', choices=['predict', 'chain_of_thought'], default='predict'
    )
    parser.add_argument(
        '--optimize', action='store_true', help="Enable pipeline optimization"
    )
    parser.add_argument(
        '--strategy',
        choices=['bootstrap_few_shot', 'mipro', 'bootstrap_random', 'simba'],
        default='bootstrap_few_shot',
    )
    parser.add_argument(
        '--max-demos', type=int, default=4, help="Maximum demonstrations for optimization"
    )
    parser.add_argument(
        '--trainset', type=str, help="Path to JSON file with training set"
    )
    parser.add_argument('--json', action='store_true', help="Output in JSON format")
    parser.add_argument(
        '--pipeline', nargs='+', help="Run a pipeline with multiple step descriptions"
    )
    parser.add_argument(
        '--evaluation-instruction',
        type=str,
        default="",
        help="Instruction for evaluating outputs",
    )
    parser.add_argument(
        '--log-file', type=str, default="dspy_logs.jsonl", help="File to store logs"
    )
    args = parser.parse_args()

    if args.inputs:
        inputs = args.inputs
    elif not sys.stdin.isatty():
        inputs = sys.stdin.read().strip().split('\n')
    else:
        inputs = []

    factory = ModuleFactory()
    input_names = [f"input_{i+1}" for i in range(len(inputs))]
    output_names = ["output"]
    module = factory.create_module(
        inputs=input_names, outputs=output_names, description=args.description
    )

    if args.optimize:
        manager = OptimizationManager()
        manager.configure(
            strategy=args.strategy, max_bootstrapped_demos=args.max_demos
        )
        trainset = []
        if args.trainset:
            try:
                with open(args.trainset, 'r', encoding='utf-8') as f:
                    trainset = json.load(f)
            except OSError as e:
                print(f"Error loading trainset: {e}")
                sys.exit(1)
        else:
            print("Warning: Using single input as trainset for optimization.")
            trainset = [dict(zip(input_names, inputs))]
        module = manager.optimize(module, trainset)

    input_dict = dict(zip(input_names, inputs))
    if args.pipeline:
        output_data = _run_pipeline(args, factory, input_names, input_dict)
    else:
        output_data = _run_single_module(module, input_dict, output_names)

    if args.json:
        print(json.dumps(output_data))
    else:
        for value in output_data.values():
            print(str(value))

    if args.evaluation_instruction:
        evaluator = Evaluator(
            evaluation_instruction=args.evaluation_instruction,
            log_file=args.log_file,
        )
        evaluator.log_with_evaluation(
            module=args.module,
            inputs=input_dict,
            outputs=output_data,
            description=args.description,
        )


if __name__ == "__main__":
    main()
