import pytest
from simpledspy import pipe

def test_multiple_outputs():
    """Test handling multiple return values"""
    text = "John Doe, 30 years old"
    
    name, age = pipe(text)
    assert name == "John Doe"
    assert age == "30" # LLM often returns numbers as strings

def test_two_outputs():
    """Test extracting second words from two lists"""
    list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
    list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
    
    # The order of assignment (oqc, jkl) vs. order of args (jkl, oqc)
    # tests if the _get_caller_context can correctly map them if names are found,
    # or if the LLM can infer from "second_word_list_oqc" and "second_word_list_jkl"
    # which input to use. If _get_caller_context fails for inputs, they become input_1, input_2.
    # The output names are always taken from assignment.
    second_word_list_oqc, second_word_list_jkl = pipe(list_jkl, list_oqc)
    assert second_word_list_jkl == "def"
    assert second_word_list_oqc == "iwfo"

def test_single_output():
    """Test extracting second word from single list assignment"""
    list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
    list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
    
    # Here, only one variable is assigned.
    # The pipe function will create a module asking for "second_word_list_oqc".
    # The LLM needs to infer from this name and the inputs which value to produce.
    second_word_list_oqc = pipe(list_jkl, list_oqc)
    assert second_word_list_oqc == "iwfo"

def test_third_word():
    """Test extracting third word from text"""
    third_word = pipe("abc def ghi jkl")
    assert third_word == "ghi"

# def test_cli_biggest_number():
    # """Test CLI interface for finding biggest number"""
    # import subprocess
    
    # # Run the CLI command
    # result = subprocess.run(
        # ['simpledspy', '54 563 125', '-d', 'get the biggest number'],
        # capture_output=True,
        # text=True
    # )
    
    # # Check the output
    # assert result.returncode == 0
    # assert "563" in result.stdout

def test_cli_stdin_biggest_number():
    """Test CLI interface with stdin for finding biggest number"""
    import subprocess
    
    # Run the CLI command with stdin
    result = subprocess.run(
        ['simpledspy', '-d', 'get the biggest number of the numbers listed'],
        input='54 563 125\n',
        capture_output=True,
        text=True,
        check=False # Added to avoid raising CalledProcessError on non-zero exit
    )
    
    # Check the output
    assert result.returncode == 0
    assert "563" in result.stdout

def test_type_hints():
    """Test type hint support in pipe function"""
    text = "John Doe, 30 years old"
    
    # Test with type hints for string
    name: str = pipe(text, description="Extract the full name.")
    assert isinstance(name, str)
    assert name == "John Doe"
    
    # Test with type hints for int
    # The LLM might return "30" or "30 years old". Description helps.
    age_str: str = pipe(text, description="Extract the age as a number.")
    # The LLM is likely to return a string "30".
    # The type conversion in pipe will attempt to convert it.
    
    age_int: int = pipe(text, description="Extract the age as a number.")
    assert isinstance(age_int, int)
    assert age_int == 30

    # Test with type hints for float
    # If LLM returns "30", it should be converted to 30.0
    # If LLM returns "30.0", it should be converted to 30.0
    age_float: float = pipe(text, description="Extract the age as a number.")
    assert isinstance(age_float, float)
    assert age_float == 30.0

    # Test boolean
    is_major_str: str = pipe("The subject is an adult.", description="Is the subject an adult? Reply true or false.")
    is_major_bool: bool = pipe("The subject is an adult.", description="Is the subject an adult? Reply true or false.")
    
    assert isinstance(is_major_bool, bool)
    # This depends on LLM output, "true" or "True"
    # Assuming LLM returns "true" or "True" (case-insensitive due to .lower() in pipe)
    if "true" in is_major_str.lower():
        assert is_major_bool is True
    elif "false" in is_major_str.lower():
        assert is_major_bool is False
    else:
        # If LLM returns something unexpected, the bool conversion might be True for non-empty strings
        # or False for empty ones, depending on the raw string.
        # The current bool conversion is `value.lower() == 'true'`.
        pytest.fail(f"LLM returned unexpected string for boolean: {is_major_str}")

