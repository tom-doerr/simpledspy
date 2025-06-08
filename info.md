i want to be able to use different types of modules for inference and not just pipe, so e.g. name, age = chain_of_thought(raw_string)
pipe() should be replaced with predict()
signatures for the modules should be constructed using the input and output variable names 
the project should be well documented with docstrings
all inputs and outputs should be logged by default into a file
the same file should allow me to set a string for evaluation instruction
there should be an evaluator that runs and ranks on a scale from 1 to 10 which is used for optimization
there should be a cli tool where i can start optimization 
the cli tool should allow me to select the dspy optimization algorithm
all functionality is documented in the readme
the readmes should be very comprehensive
it should be possible to set llm call params right in the module call, like max tokens or temperature; i want to be able to set them in the call as attributes and not just in a dictionary which would complicate the syntax 
i want to be able to set the evaluation instruction per module together with the training data and not just when callig the cli since each module might have its own evaluation criteria
i eant to be able to set multiple evaluation instructions / evaluation criteria per module which then get averaged
don't use dspy.OpenAI anymore, it has been replaced with dspy.LM


# data
there should be a way to enable automatig data saving so we save input and output for all modules
the file allows to easily move from logged to the section that is for training data so i can easily add good samples to the training/optimization dataset
each module should write the logging info to a seperate file 
logging should not happen by default but should be required to be turned on in the code
simpledspy should create some dot directory where we create all the files for the modules
by default there should already be a seciotn for the training which of course is empty by default
it then should be easy to move lines from logged input output to the train data section
it should be easy for the user to adapt training samples by modifying input and output text of a samples
the train data section should be abouve the log data section, which should be at the bottom since it likely will contain the most data
there should be the possibility to set custom names for modules when called
when no custom name for a module is set, the name consists out of ouputs, module type, and inputs: <ouput_0>_<ouput_1>__<moduletype>__<input_0>_<input_1>, e.g. finished__predict__context_steps
we do need to save the datetime together with inputs and outputs 
the datetime should be in human readable format, so maybe the T notation is best
the description should be saved in those datapoints
if no other optimized model is saved/available, the trainset/optimization set should be used for the fewshot optimizer when the module is initilized so it has the trainset as few-shot examples


# optimization
when no evaluation criteria is provided, optimization should just work with DSPy's FewShot optimizer (assuming there are any samples in the training data)
the newest optimized modules should be automatically loaded by default

# settings
there should be a simpledspy way to set global default settings like lm, temperature, max tokens and so on that doesn't require me to import dspy
as with the other code, this should be well covered by tests


# reflection
simpledspy inferes the number of outputs and ouput labels (signature) using python reflection
it also uses reflection to find out what the inputs are
it should be possible to use an arbitrary number of outputs on the left side
if python type hints are used, those types should be set in the dspy signature
it should not be necessary to set inputs or ouputs labels when calling a module
please make sure we have tests for mapping input variable names to the signature input variable names
we should use the actual assign output names even when we have multiple outputs and shouldn't just use e.g. output0, output1 
reflection should also work when the variable(s) are self.<var>


# usage examples 
please make sure we have tests for the components that make these use cases possible:
```
from simpledspy import predict
a = 'jkl'
b = 'abc'
a_reversed, b_repeated = predict(b, a)
```

# legacy functionality that should be removed
the reward function and the online rl with few-shot examples is legacy and shouldn't be part of the project anymore; optimization should only happend with dspy optimizers
interactive cli capability for running inference is legacy and should no longer be included; new versions should only have cli capability for optimizing modules
legacy: there should be a reward function i can call anywhere 
legacy: the reward feature should enable an additional layer of optimization where i can over time use cummulative discounted rewards to evaluate advice and examples i can give as input to the modules
legacy: the over time weighted few-shot examples (maybe with certainty / error margin) should be usable to generate advice by reflecting on a samples that includes examples that have a positive and examples that have a negative impact on overall reward over time 
legacy: maybe we could also use the reward calls for the dspy optimization but I'm not sure yet
legacy: we do need differente reward groups so we can optimize, e.g. a judge module as well as a modules being judged by the judge 
legacy: if not reward group is specified, rewards should be attributed to the default reward group and modules without reward group should be optimized using the default reward group rewards
legacy: it would be great if we somehow had a way to repeat inferences or multiple inferences / parts of the program when some type of assertion fails; this is really helpful if we need output or results to be a certain way to be able to continue. just continuing on and letting the rest off the execution fail and hope that over time negative reward will make it improve is not a good solution. so it should retry somehow for a number of times and give negative reward as feeedback every time as to make that part of the program more reliable over time
legacy: each module should store all inputs and ouputs by default and do bayesian optimization to optimize the reward as well as the number of used few shot samples (maybe indirectly through optimizing which few-shot examples to include)
legacy: the rewads should be discounted and applied to all used few-shot examples, of course multiple times when used multiple times
legacy: there should be a way to mark the end of an episode or a new episode or similar to create a cutoff so rewards are not propagated too far
legacy: i want to be able to hand over one or more python functions as an argument to dspy modules 
legacy: handing over a single function without a list should be possible as well as handing over multiple functions in a list or dict or similar should be possible
legacy: the module should assign a negative reward when it fails to run the function without errors, do its bayesian optimization and retry up to a configurable number of times
legacy: it should return the return value of the python object as a proper python object and not as string so i can continue right away using that data
