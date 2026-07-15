import os
import tempfile

import torch


def get_model_size(model) -> float:
    """
    Returns the serialized model size in MB.
    """

    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 ** 2)

    os.remove(f.name)

    return size_mb