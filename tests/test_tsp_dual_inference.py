import pytest
import numpy as np
from unittest.mock import MagicMock
from src.tsp_dual_inference import predict_dual_bound, load_model

def test_load_model(mock_model, mock_tokenizer):
    """Test that model loading call works (and is mocked)."""
    model, tokenizer = load_model("mock_path")
    assert model == mock_model
    assert tokenizer == mock_tokenizer

def test_predict_dual_bound_success(mock_model, mock_tokenizer):
    """Test successful bound prediction with mocked output."""
    # Setup mock to return valid float in tags
    mock_tokenizer.decode.return_value = "Thought process... <answer>123.45</answer>"
    
    coords = np.array([[0,0], [1,1], [2,2]])
    bound = predict_dual_bound(mock_model, mock_tokenizer, coords)
    
    assert bound == 123.45
    # Verify generate was called
    assert mock_model.generate.called

def test_predict_dual_bound_raw_number(mock_model, mock_tokenizer):
    """Test resilience when model output is just a number."""
    mock_tokenizer.decode.return_value = "Just the number: 123.45"
    
    coords = np.array([[0,0], [1,1]])
    bound = predict_dual_bound(mock_model, mock_tokenizer, coords)
    
    assert bound == 123.45

def test_predict_dual_bound_failure(mock_model, mock_tokenizer):
    """Test fallback when no number is found."""
    mock_tokenizer.decode.return_value = "I cannot solve this."
    
    coords = np.array([[0,0], [1,1]])
    bound = predict_dual_bound(mock_model, mock_tokenizer, coords, max_retries=1)
    
    assert bound is None
