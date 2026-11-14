"""The upstream FSQ, with exact integer validation and 1D shapes."""

import torch

from cosmos_predict1.tokenizer.modules.quantizers import FSQuantizer

from .config import CODEBOOK_SIZE, FSQ_LEVELS


def validate_ids(ids):
    if ids.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint16):
        raise ValueError("tokens must be integer IDs")
    if ids.numel() and (
        ids.to(torch.int64).min() < 0 or ids.to(torch.int64).max() >= CODEBOOK_SIZE
    ):
        raise ValueError("FSQ ID outside [0,64000)")


def make_fsq():
    # Float32 prevents a non-power-of-two code spacing being rounded before indexing.
    return FSQuantizer(levels=list(FSQ_LEVELS), dtype=torch.float32)


def quantize(fsq, values):
    with torch.autocast(values.device.type, enabled=False):
        ids, codes, loss = fsq(values.float())
    return ids, codes, loss.mean()
