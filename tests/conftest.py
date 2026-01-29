import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session")
def mock_tokenizer():
    """Returns a mock tokenizer that behaves like a HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.decode.return_value = "<answer>123.45</answer>"
    tokenizer.batch_decode.return_value = ["<answer>123.45</answer>"]
    # Mock __call__ to return a dict with input_ids
    tokenizer.return_value = {
        "input_ids": MagicMock(shape=(1, 10)),
        "attention_mask": MagicMock(shape=(1, 10))
    }

    # Handle apply_chat_template
    tokenizer.apply_chat_template.return_value = "User: Prompt\nAssistant:"
    return tokenizer

@pytest.fixture(scope="session")
def mock_model():
    """Returns a mock model that can generate text."""
    model = MagicMock()
    model.device = "cpu"
    # Mock generate to return a tensor of shape (1, 20)
    generated_ids = MagicMock()
    generated_ids.shape = (1, 20)
    # Slicing support: output_ids = generated_ids[:, input_len:]
    generated_ids.__getitem__.return_value = MagicMock(shape=(1, 10))
    model.generate.return_value = generated_ids
    return model

@pytest.fixture(autouse=True)
def patch_transformers(mock_tokenizer, mock_model):
    """Automatically patch AutoTokenizer and AutoModelForCausalLM for all tests."""
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoModelForVision2Seq.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoProcessor.from_pretrained", return_value=mock_tokenizer):
        yield
