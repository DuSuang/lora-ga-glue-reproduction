"""Run the offline CPU acceptance suite with active CUDA and pretrained-model tripwires."""
from contextlib import ExitStack, contextmanager
import os
import unittest
from unittest.mock import patch

from glue import machine
from glue.machine import ROOT


@contextmanager
def guarded():
    machine.setup_cache()
    os.environ["HF_HUB_OFFLINE"]="1"
    import torch
    from transformers import PreTrainedModel
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA already initialized")
    torch.set_num_threads(2)
    with ExitStack() as stack:
        cuda=stack.enter_context(patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")))
        pretrained=stack.enter_context(patch.object(PreTrainedModel,"from_pretrained",
                                                   side_effect=AssertionError("pretrained model forbidden")))
        yield
        print(f"CUDA initialization calls={cuda.call_count}; pretrained weight loads={pretrained.call_count}")
        assert not torch.cuda.is_initialized() and not cuda.called and not pretrained.called


def run(names):
    with guarded():
        loader=unittest.defaultTestLoader
        tests=loader.loadTestsFromNames(names) if names else loader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
        result=unittest.TextTestRunner(verbosity=2).run(tests)
    return result.wasSuccessful()
