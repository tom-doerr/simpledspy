i want to be able to use different types of modules for inference and not just pipe, so e.g. name, age = chain_of_thought(raw_string)
pipe() should be replaced with predict()
signatures for the modules should be constructed using the input and output variable names 
the project should be well documented with docstrings
all inputs and outputs should be logged by default into a file
the file allows to easily move from logged to the section that is for training data so i can easily add good samples to the training/optimization dataset
the same file should allow me to set a string for evaluation instruction
there should be an evaluator that runs and ranks on a scale from 1 to 10 which is used for optimization
there should be a cli tool where i can start optimization 
the cli tool should allow me to select the dspy optimization algorithm
all functionality is documented in the readme
the readmes should be very comprehensive
there should be a reward function i can call anywhere 
the reward feature should enable an additional layer of optimization where i can over time use cummulative discounted rewards to evaluate advice and examples i can give as input to the modules
the over time weighted few-shot examples (maybe with certainty / error margin) should be usable to generate advice by reflecting on a samples that includes examples that have a positive and examples that have a negative impact on overall reward over time 
maybe we could also use the reward calls for the dspy optimization but I'm not sure yet
we do need differente reward groups so we can optimize, e.g. a judge module as well as a modules being judged by the judge 
if not reward group is specified, rewards should be attributed to the default reward group and modules without reward group should be optimized using the default reward group rewards
