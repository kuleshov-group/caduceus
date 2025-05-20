import pytest
import torch
from collections import defaultdict
from typing import Optional, Dict, Any, Literal
from torch import nn
from torch.utils.data import Dataset

from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import CaduceusForMaskedLM
from transformers import TrainingArguments, Trainer
from composer import Trainer as ComposerTrainer
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW


# Reentrant gradient checkpointing is required to force recomputation
# on backward passes, which is necessary for tests that verify checkpointing
# usage based on observing those recomputations
USE_REENTRANT_CHECKPOINTS = True

def create_test_model(
    config_overrides: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cuda"),
    seed: int = 0
) -> CaduceusForMaskedLM:
    """Create a CaduceusForMaskedLM model with test configuration."""
    torch.random.manual_seed(seed)
    
    # Default test configuration
    config_params: Dict[str, Any] = {
        'd_model': 128,
        'n_layer': 4,
        'vocab_size': 10,
        'gradient_checkpointing_stride': 1,
        'pad_token_id': -100
    }
    
    # Update with any overrides
    if config_overrides:
        config_params.update(config_overrides)
    
    config = CaduceusConfig(**config_params)
    return CaduceusForMaskedLM(config).to(device)


def create_test_inputs(
    model: CaduceusForMaskedLM,
    batch_size: Optional[int] = 2,
    seq_len: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Create random input tensors for testing."""
    shape = (batch_size, seq_len) if batch_size is not None else (seq_len,)
    return {
        'input_ids': torch.randint(0, model.config.vocab_size, shape, device=device),
        'labels': torch.randint(0, model.config.vocab_size, shape, device=device)
    }


def test_caduceus_masked_lm():
    """Test basic CaduceusForMaskedLM functionality with default settings."""
    # Create model with default config
    model = create_test_model()
    
    # Generate random input
    batch_size, seq_len = 3, 128
    inputs = create_test_inputs(model, batch_size=batch_size, seq_len=seq_len, device=model.device)
    
    # Run forward pass
    outputs = model(**inputs)
    
    # Check output shapes
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size), "Unexpected logits shape"
    
    # Check loss is computed and backpropagates
    assert outputs.loss is not None, "Loss should be computed when labels are provided"
    outputs.loss.backward()
    
    # Check all parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

def run_direct_pass(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool
) -> Dict[nn.Module, int]:
    """Run single forward and backward pass explicitly and return forward pass counts."""
    forward_counts = defaultdict(int)
    def count_forwards(module, input, output):
        forward_counts[module] += 1
    
    # Register hooks on each layer
    for layer in model.caduceus.backbone.layers:
        layer.register_forward_hook(count_forwards)
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT_CHECKPOINTS})
    else:
        model.gradient_checkpointing_disable()
        
    outputs = model(**inputs)
    outputs.loss.backward()
    
    return dict(forward_counts)
    

def run_hf_training(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool,
    output_dir: str
) -> Dict[nn.Module, int]:
    """Run training using HF Trainer and return forward pass counts.
    
    See Also:
        - [Transformers Documentation - Trainer](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.Trainer)
        - [Transformers Source - Trainer implementation](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/trainer.py#L312)
    """
    forward_counts = defaultdict(int)
    def count_forwards(module, input, output):
        forward_counts[module] += 1
    
    # Register hooks on each layer
    for layer in model.caduceus.backbone.layers:
        layer.register_forward_hook(count_forwards)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        disable_tqdm=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT_CHECKPOINTS},
        max_steps=1,
        logging_strategy="no",
        report_to="none",
        # Ensure that serialization is tested as well
        save_strategy="steps",
        save_steps=1,
        # Use safetensors instead of native serialization to avoid errors about 
        # saving shared tensors in different modules (as `BiMambaWrapper` does), e.g.:
        # RuntimeError: The weights trying to be saved contained shared tensors [<tensor_dict>] that are mismatching the transformers base configuration.
        save_safetensors=False,
        # Overwrite must be enabled as pytest tmp_dir is named systematically, e.g.:
        # /tmp/pytest-of-ubuntu/pytest-3/test_activation_checkpointing_1
        overwrite_output_dir=True, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SimpleDataset(inputs),
    )
    
    trainer.train()
    return dict(forward_counts)

def run_mosaic_training(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool,
    output_dir: str
) -> Dict[nn.Module, int]:
    """Run training using MosaicML Composer Trainer and return forward pass counts.
    
    See Also:
        - [Composer Documentation - Trainer](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.Trainer.html)
        - [Composer Source - Trainer implementation](https://docs.mosaicml.com/projects/composer/en/latest/_modules/composer/trainer/trainer.html#Trainer)
    """
    forward_counts = defaultdict(int)
    def count_forwards(module, input, output):
        forward_counts[module] += 1
    
    # Register hooks on each layer
    for layer in model.caduceus.backbone.layers:
        layer.register_forward_hook(count_forwards)
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT_CHECKPOINTS})
    else:
        model.gradient_checkpointing_disable()

    # Wrap the model for Composer
    composer_model = HuggingFaceModel(model)

    optimizer = DecoupledAdamW(model.parameters())

    trainer = ComposerTrainer(
        optimizers=optimizer,
        model=composer_model,
        log_to_console=False,
        progress_bar=False,
        train_dataloader=SimpleDataset(inputs),
        max_duration='1ba',
        device_train_microbatch_size=1,
        # Ensure that serialization is tested as well
        save_folder=output_dir,
        save_interval='1ep',
        # Overwrite must be enabled as pytest tmp_dir is named systematically, e.g.:
        # /tmp/pytest-of-ubuntu/pytest-3/test_activation_checkpointing_1
        save_overwrite=True,
    )
    
    trainer.fit()
    return dict(forward_counts)

class SimpleDataset(Dataset):
    """Simple dataset wrapper for a single input (or single batch of inputs)."""
    
    def __init__(self, inputs: Dict[str, torch.Tensor]):
        self.inputs = inputs
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        return self.inputs
    
@pytest.mark.parametrize("gradient_checkpointing_stride", [1, 2])
@pytest.mark.parametrize("mode", ["direct", "huggingface", "mosaic"])
def test_activation_checkpointing_recomputation(
    gradient_checkpointing_stride: int,
    mode: Literal["direct", "huggingface", "mosaic"],
    tmp_path,
):
    """Test that activation checkpointing causes expected recomputation."""
    # Create model with specified stride
    model = create_test_model({
        'gradient_checkpointing_stride': gradient_checkpointing_stride
    })
    
    # Generate random input
    # Trainer APIs handle device placement and batching so we only those it in `direct` mode
    batch_size = 2 if mode == "direct" or mode == "mosaic" else None
    device = model.device if mode == "direct" or mode == "mosaic" else None
    inputs = create_test_inputs(model, batch_size=batch_size, device=device)
    
    # Run training with and without checkpointing
    run_fn = (run_hf_training if mode == "huggingface" 
             else run_mosaic_training if mode == "mosaic"
             else run_direct_pass)
    kwargs = {"output_dir": str(tmp_path)} if mode in ["huggingface", "mosaic"] else {}
    
    forward_counts_no_checkpoint = run_fn(model, inputs, False, **kwargs)
    forward_counts_checkpoint = run_fn(model, inputs, True, **kwargs)
    
    # Verify counts
    for layer in model.caduceus.backbone.layers:
        layer_idx = layer.layer_idx
        if layer_idx % gradient_checkpointing_stride == 0:
            # Checkpointed layers should be computed twice
            assert forward_counts_checkpoint[layer] == 2 * forward_counts_no_checkpoint[layer]
        else:
            # Non-checkpointed layers should be computed the same number of times
            assert forward_counts_checkpoint[layer] == forward_counts_no_checkpoint[layer]

def test_invalid_gradient_checkpointing_stride():
    """Test that invalid gradient_checkpointing_stride raises ValueError."""
    # Test stride = 0
    with pytest.raises(ValueError, match=r".*must be between 1 and \d+.*"):
        create_test_model({'gradient_checkpointing_stride': 0})
    
    # Test stride > n_layer
    with pytest.raises(ValueError, match=r".*must be between 1 and \d+.*"):
        create_test_model({'gradient_checkpointing_stride': 5})


