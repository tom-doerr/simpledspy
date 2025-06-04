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
