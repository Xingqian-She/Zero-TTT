from __future__ import annotations

import torch
from zero_ttt.precision import configure_strict_fp32


def test_strict_fp32_disables_reduced_precision_backends() -> None:
    previous_matmul = torch.get_float32_matmul_precision()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        configure_strict_fp32()
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(previous_matmul)
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
