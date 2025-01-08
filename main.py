from simpledspy import pipe

if __name__ == "__main__":
    # Test basic functionality
    result = pipe("abc def ghi jkl")
    print("Result:", result)  # Should print: ghi
    
    # Test concatenation
    combined = pipe("Hello", "World")
    print("Combined:", combined)  # Should print: Hello World
    
    # Test word count
    word_count = pipe("This text has exactly five words")
    print("Word count:", word_count)  # Should print: 5
