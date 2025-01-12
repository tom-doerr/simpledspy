import pytest
from simpledspy import pipe

def test_multiple_outputs():
    """Test handling multiple return values"""
    text = "John Doe, 30 years old"
    
    name, age = pipe(text)
    assert name == "John Doe"
    assert age == "30"

def test_two_outputs():
    """Test extracting second words from two lists"""
    list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
    list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
    
    second_word_list_oqc, second_word_list_jkl = pipe(list_jkl, list_oqc)
    assert second_word_list_jkl == "def"
    assert second_word_list_oqc == "iwfo"

def test_single_output():
    """Test extracting second word from single list"""
    list_jkl = "abc def ghi jkl iowe afj wej own iow jklwe"
    list_oqc = "oid iwfo fjs wjiof sfio we x dso weop vskl we"
    
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
        text=True
    )
    
    # Check the output
    assert result.returncode == 0
    assert "563" in result.stdout

def test_type_hints():
    """Test type hint support in pipe function"""
    text = "John Doe, 30 years old"
    
    # Test with type hints
    name: str = pipe(text)
    assert isinstance(name, str)
    
    # Test with multiple outputs and types
    name, age = pipe(text)
    assert isinstance(name, str)
    assert isinstance(age, int)
    assert name == "John Doe"
    assert age == 30

