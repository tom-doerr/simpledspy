"""Comprehensive tests for training_utils.py"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import dspy
from simpledspy.training_utils import TrainingUtils
from simpledspy.settings import settings


class TestFormatExample:
    """Test example formatting functionality"""
    
    def test_format_example_new_format(self):
        """Test formatting with new inputs/outputs format"""
        example = {
            "inputs": [
                {"name": "question", "value": "What is 2+2?"},
                {"name": "context", "value": "Basic math"}
            ],
            "outputs": [
                {"name": "answer", "value": "4"},
                {"name": "reasoning", "value": "Addition"}
            ],
            "timestamp": "2024-01-01",
            "module": "test_module"
        }
        
        result = TrainingUtils.format_example(example)
        
        assert result == {
            "question": "What is 2+2?",
            "context": "Basic math",
            "answer": "4",
            "reasoning": "Addition"
        }
    
    def test_format_example_old_format(self):
        """Test formatting with old top-level keys format"""
        example = {
            "question": "What is 2+2?",
            "answer": "4",
            "section": "math",
            "timestamp": "2024-01-01",
            "module": "test_module",
            "description": "Test description"
        }
        
        result = TrainingUtils.format_example(example)
        
        # Reserved keys should be excluded
        assert result == {
            "question": "What is 2+2?",
            "answer": "4"
        }
    
    def test_format_example_empty_inputs_outputs(self):
        """Test formatting with empty inputs/outputs"""
        example = {
            "inputs": [],
            "outputs": [],
            "module": "test"
        }
        
        result = TrainingUtils.format_example(example)
        assert result == {}
    
    def test_format_example_mixed_format(self):
        """Test when both formats are present (new format takes precedence)"""
        example = {
            "inputs": [{"name": "q", "value": "new"}],
            "outputs": [{"name": "a", "value": "format"}],
            "q": "old",
            "a": "format"
        }
        
        result = TrainingUtils.format_example(example)
        
        # New format should take precedence
        assert result == {
            "q": "new",
            "a": "format"
        }
    
    def test_format_example_missing_keys(self):
        """Test handling of malformed examples"""
        # Missing 'value' key - this will raise KeyError
        example = {
            "inputs": [{"name": "q"}],  # Missing value
            "outputs": [{"name": "a", "value": "answer"}]
        }
        
        # Should raise KeyError for missing value
        with pytest.raises(KeyError):
            TrainingUtils.format_example(example)


class TestLoadAndPrepareDemos:
    """Test demo loading and preparation"""
    
    @patch('simpledspy.training_utils.Logger')
    def test_load_and_prepare_demos_with_data(self, mock_logger_class):
        """Test loading demos when training data exists"""
        # Mock logger instance
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        # Mock training data
        training_data = [
            {
                "inputs": [{"name": "text", "value": "Hello"}],
                "outputs": [{"name": "response", "value": "Hi"}]
            },
            {
                "inputs": [{"name": "text", "value": "Goodbye"}],
                "outputs": [{"name": "response", "value": "Bye"}]
            }
        ]
        mock_logger.load_training_data.return_value = training_data
        
        # Test
        demos = TrainingUtils.load_and_prepare_demos("test_module")
        
        # Verify logger was created with correct parameters
        mock_logger_class.assert_called_once_with(
            module_name="test_module",
            base_dir=settings.log_dir if settings.log_dir else ".simpledspy"
        )
        
        # Verify demos were created
        assert len(demos) == 2
        assert all(isinstance(d, dspy.Example) for d in demos)
        assert demos[0].text == "Hello"
        assert demos[0].response == "Hi"
        assert demos[1].text == "Goodbye"
        assert demos[1].response == "Bye"
    
    @patch('simpledspy.training_utils.Logger')
    def test_load_and_prepare_demos_no_data(self, mock_logger_class):
        """Test loading demos when no training data exists"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        mock_logger.load_training_data.return_value = []
        
        demos = TrainingUtils.load_and_prepare_demos("test_module")
        
        assert demos == []
    
    @patch('simpledspy.training_utils.Logger')
    def test_load_and_prepare_demos_with_log_dir(self, mock_logger_class):
        """Test that custom log_dir from settings is used"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        mock_logger.load_training_data.return_value = []
        
        # Set custom log_dir
        with patch.object(settings, 'log_dir', '/custom/log/dir'):
            TrainingUtils.load_and_prepare_demos("test_module")
            
            # Verify logger was created with custom log_dir
            mock_logger_class.assert_called_once_with(
                module_name="test_module",
                base_dir="/custom/log/dir"
            )
    
    @patch('simpledspy.training_utils.Logger')
    def test_load_and_prepare_demos_malformed_data(self, mock_logger_class):
        """Test handling of malformed training data"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        # Mix of valid and invalid data
        training_data = [
            {
                "inputs": [{"name": "text", "value": "Valid"}],
                "outputs": [{"name": "response", "value": "Good"}]
            },
            {  # Valid old format
                "text": "Old format",
                "response": "Works too"
            },
            {"invalid": "dict without expected keys"},  # This will actually create a valid Example with one field
        ]
        mock_logger.load_training_data.return_value = training_data
        
        demos = TrainingUtils.load_and_prepare_demos("test_module")
        
        # Should have 3 demos (all create valid Examples)
        assert len(demos) == 3
        assert demos[0].text == "Valid"
        assert demos[1].text == "Old format"
        assert demos[2].invalid == "dict without expected keys"
    
    @patch('simpledspy.training_utils.Logger')
    @patch('simpledspy.training_utils.dspy.Example')
    def test_load_and_prepare_demos_example_creation_error(self, mock_example, mock_logger_class):
        """Test handling when Example creation fails"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        training_data = [
            {"text": "data1", "response": "resp1"},
            {"text": "data2", "response": "resp2"}
        ]
        mock_logger.load_training_data.return_value = training_data
        
        # Make Example creation fail for first item
        mock_example.side_effect = [TypeError("Invalid"), Mock(text="data2", response="resp2")]
        
        demos = TrainingUtils.load_and_prepare_demos("test_module")
        
        # Should skip the failed one
        assert len(demos) == 1
        assert demos[0].text == "data2"


class TestApplyTrainingData:
    """Test applying training data to modules"""
    
    def test_apply_training_data_with_trainset(self):
        """Test applying explicit trainset"""
        mock_module = Mock()
        
        # Test with dict trainset
        trainset = [
            {"input": "test1", "output": "result1"},
            {"input": "test2", "output": "result2"}
        ]
        
        TrainingUtils.apply_training_data(mock_module, trainset=trainset)
        
        # Verify demos were set
        assert hasattr(mock_module, 'demos')
        assert len(mock_module.demos) == 2
        assert all(isinstance(d, dspy.Example) for d in mock_module.demos)
    
    def test_apply_training_data_with_example_objects(self):
        """Test applying trainset that already contains Example objects"""
        mock_module = Mock()
        
        # Trainset with Example objects
        example1 = dspy.Example(input="test1", output="result1")
        example2 = dspy.Example(input="test2", output="result2")
        trainset = [example1, example2]
        
        TrainingUtils.apply_training_data(mock_module, trainset=trainset)
        
        # Should use existing Example objects
        assert mock_module.demos == [example1, example2]
    
    def test_apply_training_data_mixed_trainset(self):
        """Test applying trainset with mixed dict and Example objects"""
        mock_module = Mock()
        
        example1 = dspy.Example(input="test1", output="result1")
        trainset = [
            example1,
            {"input": "test2", "output": "result2"}
        ]
        
        TrainingUtils.apply_training_data(mock_module, trainset=trainset)
        
        assert len(mock_module.demos) == 2
        assert mock_module.demos[0] == example1
        assert isinstance(mock_module.demos[1], dspy.Example)
        assert mock_module.demos[1].input == "test2"
    
    @patch.object(TrainingUtils, 'load_and_prepare_demos')
    def test_apply_training_data_with_name(self, mock_load):
        """Test applying training data by loading from name"""
        mock_module = Mock()
        
        # Mock loaded demos
        demo1 = dspy.Example(text="test1", response="resp1")
        demo2 = dspy.Example(text="test2", response="resp2")
        mock_load.return_value = [demo1, demo2]
        
        TrainingUtils.apply_training_data(mock_module, name="test_module")
        
        # Verify loading was called
        mock_load.assert_called_once_with("test_module")
        
        # Verify demos were set
        assert mock_module.demos == [demo1, demo2]
    
    @patch.object(TrainingUtils, 'load_and_prepare_demos')
    def test_apply_training_data_with_name_no_demos(self, mock_load):
        """Test when loading by name returns no demos"""
        # Use MagicMock with spec to avoid auto-creating attributes
        mock_module = MagicMock(spec=[])
        mock_load.return_value = []
        
        TrainingUtils.apply_training_data(mock_module, name="test_module")
        
        # When no demos are loaded, module.demos is not set
        assert not hasattr(mock_module, 'demos')
    
    def test_apply_training_data_no_params(self):
        """Test when neither trainset nor name is provided"""
        mock_module = Mock()
        
        # Make sure module doesn't already have demos
        if hasattr(mock_module, 'demos'):
            delattr(mock_module, 'demos')
        
        TrainingUtils.apply_training_data(mock_module)
        
        # Module should not have demos attribute set
        assert not hasattr(mock_module, 'demos')
    
    def test_apply_training_data_trainset_takes_precedence(self):
        """Test that trainset takes precedence over name"""
        mock_module = Mock()
        
        trainset = [{"input": "explicit", "output": "data"}]
        
        with patch.object(TrainingUtils, 'load_and_prepare_demos') as mock_load:
            TrainingUtils.apply_training_data(
                mock_module, 
                trainset=trainset, 
                name="should_be_ignored"
            )
            
            # load_and_prepare_demos should not be called
            mock_load.assert_not_called()
            
            # Should use explicit trainset
            assert len(mock_module.demos) == 1
            assert mock_module.demos[0].input == "explicit"
    
    def test_apply_training_data_empty_trainset(self):
        """Test with empty trainset"""
        mock_module = Mock()
        
        TrainingUtils.apply_training_data(mock_module, trainset=[])
        
        # Should set empty demos list
        assert hasattr(mock_module, 'demos')
        assert mock_module.demos == []


class TestIntegration:
    """Integration tests for training utilities"""
    
    @patch('simpledspy.training_utils.Logger')
    def test_full_workflow(self, mock_logger_class):
        """Test complete workflow from loading to applying"""
        # Setup mock logger
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        # Mock training data with various formats
        training_data = [
            {
                "inputs": [
                    {"name": "question", "value": "What is AI?"},
                    {"name": "context", "value": "Technology"}
                ],
                "outputs": [
                    {"name": "answer", "value": "Artificial Intelligence"}
                ]
            },
            {
                "question": "What is ML?",
                "answer": "Machine Learning",
                "timestamp": "2024-01-01"
            }
        ]
        mock_logger.load_training_data.return_value = training_data
        
        # Create mock module
        mock_module = Mock()
        
        # Apply training data by name
        TrainingUtils.apply_training_data(mock_module, name="qa_module")
        
        # Verify the complete flow
        assert hasattr(mock_module, 'demos')
        assert len(mock_module.demos) == 2
        
        # Check first demo (new format)
        assert mock_module.demos[0].question == "What is AI?"
        assert mock_module.demos[0].context == "Technology"
        assert mock_module.demos[0].answer == "Artificial Intelligence"
        
        # Check second demo (old format)
        assert mock_module.demos[1].question == "What is ML?"
        assert mock_module.demos[1].answer == "Machine Learning"