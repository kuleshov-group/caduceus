import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fsspec
import torch
import torch.nn as nn
import torch.utils

# Check if file exists using fsspec
def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)

# List directory contents using fsspec
def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def get_last_embedding_dimension(model: nn.Module, rcps=False) -> int:
    """
    Function to get the last embedding dimension of a PyTorch model by passing
    a random tensor through the model and inspecting the output shape.
    This is done with gradients disabled and always on GPU.

    Args:
        model (nn.Module): The PyTorch model instance.

    Returns:
        int: The last embedding dimension (i.e., the last dimension of the output tensor).
    """
    # Move the model to GPU
    model = model.to('cuda')

    # Try to determine the input shape based on the first layer of the model
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Assume a common image input size if it's a Conv2d layer
            input_shape = (3, 224, 224)  # RGB image of size 224x224
            break
        elif isinstance(module, nn.Linear):
            # Assume a 1D input size for a fully connected layer
            input_shape = (module.in_features,)
            break
        elif isinstance(module, nn.Embedding):
            # Assume a single index for an Embedding layer
            input_shape = (1,)
            break
    else:
        raise ValueError("Unable to determine the input shape automatically.")

    # Generate a random input tensor and move it to GPU
    random_input = torch.randint(low=0, high=16, size=(1, *input_shape)).to('cuda')  # Add batch size of 1

    # Pass the tensor through the model with no gradients
    with torch.no_grad():
        output = model(random_input)[0]

    # Get the shape of the output tensor
    last_embedding_dimension = output.shape[-1]

    if rcps:
        last_embedding_dimension //= 2

    # Return the last dimension of the output tensor
    return last_embedding_dimension
