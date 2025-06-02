import pytest
from simpledspy.module_factory import ModuleFactory
import dspy
from typing import List, Dict, Any, Optional, Union

def test_create_module_basic():
    """Test basic module creation without type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["input1", "input2"],
        outputs=["output1"]
    )
    
    # Check that it's a DSPy module
    assert isinstance(module, dspy.Module)
    
    # Check that the signature has the expected fields
    signature = module.signature
    
    # Check that fields are defined in the signature
    assert 'input1' in signature.model_fields
    assert 'input2' in signature.model_fields
    assert 'output1' in signature.model_fields
    
    # Check field types
    assert signature.model_fields['input1'].annotation == str
    assert signature.model_fields['input2'].annotation == str
    assert signature.model_fields['output1'].annotation == str

def test_create_module_with_types():
    """Test module creation with type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["name", "age"],
        outputs=["greeting"],
        input_types={"name": str, "age": int},
        output_types={"greeting": str}
    )
    
    # Get the signature
    signature = module.signature
    
    # Check that fields exist
    assert 'name' in signature.model_fields
    assert 'age' in signature.model_fields
    assert 'greeting' in signature.model_fields
    
    # Get field objects
    name_field = signature.model_fields['name']
    age_field = signature.model_fields['age']
    greeting_field = signature.model_fields['greeting']
    
    # Check that type information is included in descriptions
    assert "of type str" in name_field.json_schema_extra['desc']
    assert "of type int" in age_field.json_schema_extra['desc']
    assert "of type str" in greeting_field.json_schema_extra['desc']

def test_create_module_with_description():
    """Test module creation with custom description"""
    factory = ModuleFactory()
    description = "Extract the person's name and age from the text"
    module = factory.create_module(
        inputs=["text"],
        outputs=["name", "age"],
        description=description
    )
    
    # Check that the description is set
    assert description == module.signature.__doc__

def test_create_module_with_missing_types():
    """Test module creation with partial type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["name", "age"],
        outputs=["greeting", "farewell"],
        input_types={"name": str},  # Only one input has type
        output_types={"greeting": str}  # Only one output has type
    )
    
    # Get the signature
    signature = module.signature
    
    # Get field objects
    name_field = signature.model_fields['name']
    age_field = signature.model_fields['age']
    greeting_field = signature.model_fields['greeting']
    farewell_field = signature.model_fields['farewell']
    
    # Check that type information is included only where provided
    assert "of type str" in name_field.json_schema_extra['desc']
    assert "of type str" not in age_field.json_schema_extra['desc']
    assert "of type str" in greeting_field.json_schema_extra['desc']
    assert "of type str" not in farewell_field.json_schema_extra['desc']

def test_create_module_with_complex_types():
    """Test module creation with complex type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["items", "config"],
        outputs=["result", "metadata"],
        input_types={"items": List[str], "config": Dict[str, Any]},
        output_types={"result": List[int], "metadata": Dict[str, Any]}
    )
    
    # Get the signature
    signature = module.signature
    
    # Get field objects
    items_field = signature.model_fields['items']
    config_field = signature.model_fields['config']
    result_field = signature.model_fields['result']
    metadata_field = signature.model_fields['metadata']
    
    # Check that type information is included in descriptions
    assert "of type List" in items_field.json_schema_extra['desc']
    assert "of type Dict" in config_field.json_schema_extra['desc']
    assert "of type List" in result_field.json_schema_extra['desc']
    assert "of type Dict" in metadata_field.json_schema_extra['desc']

def test_create_module_with_optional_types():
    """Test module creation with Optional type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["name", "age"],
        outputs=["greeting"],
        input_types={"name": str, "age": Optional[int]},
        output_types={"greeting": Optional[str]}
    )
    
    # Get the signature
    signature = module.signature
    
    # Get field objects
    name_field = signature.model_fields['name']
    age_field = signature.model_fields['age']
    greeting_field = signature.model_fields['greeting']
    
    # Check that type information is included in descriptions
    assert "of type str" in name_field.json_schema_extra['desc']
    assert "of type Optional" in age_field.json_schema_extra['desc']
    assert "of type Optional" in greeting_field.json_schema_extra['desc']

def test_create_module_with_union_types():
    """Test module creation with Union type hints"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["value"],
        outputs=["result"],
        input_types={"value": Union[str, int]},
        output_types={"result": Union[str, int]}
    )
    
    # Get the signature
    signature = module.signature
    
    # Get field objects
    value_field = signature.model_fields['value']
    result_field = signature.model_fields['result']
    
    # Check that type information is included in descriptions
    assert "of type Union" in value_field.json_schema_extra['desc']
    assert "of type Union" in result_field.json_schema_extra['desc']

def test_create_module_with_empty_inputs():
    """Test module creation with no inputs"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=[],
        outputs=["result"]
    )
    
    # Get the signature
    signature = module.signature
    
    # Check that output field exists
    assert 'result' in signature.model_fields
    # Check field description
    assert "Output field result" in signature.model_fields['result'].json_schema_extra['desc']

def test_create_module_with_empty_outputs():
    """Test module creation with no outputs"""
    factory = ModuleFactory()
    
    # Create module with empty outputs - this should work in the implementation
    module = factory.create_module(
        inputs=["input1"],
        outputs=[]
    )
    
    # Get the signature
    signature = module.signature
    
    # Check that input field exists
    assert 'input1' in signature.model_fields
    # Check field description
    assert "Input field input1" in signature.model_fields['input1'].json_schema_extra['desc']

def test_create_module_with_duplicate_names():
    """Test module creation with duplicate field names"""
    factory = ModuleFactory()
    
    # Create module with duplicate names
    module = factory.create_module(
        inputs=["name", "name"],
        outputs=["result", "result"]
    )
    
    # Get the signature
    signature = module.signature
    
    # Check that fields exist (only once each)
    assert 'name' in signature.model_fields
    assert 'result' in signature.model_fields
    
    # Count the number of fields
    fields = [f for f in signature.model_fields.keys() if not f.startswith('_') and f not in dir(type(signature))]
    assert len(fields) == 2  # Only two fields should exist

def test_create_module_with_long_description():
    """Test module creation with a very long description"""
    factory = ModuleFactory()
    long_description = "This is a very long description. " * 50
    
    module = factory.create_module(
        inputs=["input1"],
        outputs=["output1"],
        description=long_description
    )
    
    # Check that the description is set
    assert module.signature.__doc__ == long_description

def test_module_execution():
    """Test that created module can be executed"""
    factory = ModuleFactory()
    module = factory.create_module(
        inputs=["text"],
        outputs=["result"]
    )
    
    # We can't actually execute the module without an LLM,
    # but we can check that it has the expected structure
    assert hasattr(module, 'forward')
    assert callable(module.forward)
