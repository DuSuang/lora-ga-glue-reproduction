"""Full-size model loading and random state for a single selected CUDA device."""
import random

import numpy as np
import torch
from transformers import T5ForConditionalGeneration


def select_device(device):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Specify an indexed NVIDIA device such as cuda:0")
    torch.cuda.set_device(device)
    return device


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    torch.cuda.default_generators[device.index].manual_seed(seed)


def base(model_dir, device):
    return T5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    ).to(device).eval()


@torch.inference_mode()
def probe(model, batch):
    return model.eval()(**batch).logits.detach().cpu()
