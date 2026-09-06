from __future__ import annotations

import pytest
import torch
from torch import nn
from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.model.base import BasePolicyValueModel
from zero_ttt.model.contracts import BackboneOutput, ModelDiagnostics, ModelPredictions


def model_inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch, 25, 19, 19),
        torch.zeros(batch, 5),
        torch.ones(batch, 362, dtype=torch.bool),
    )


def test_base_model_owns_macro_forward_order_and_masking() -> None:
    calls: list[str] = []

    class RecordingModel(BasePolicyValueModel):
        def __init__(self) -> None:
            super().__init__(
                input_planes=1,
                global_features=1,
                board_size=1,
                action_size=2,
            )
            self.weight = nn.Parameter(torch.ones(()))

        def _encode_tokens(self, board_features, global_features):
            calls.append("encode")
            return (board_features.flatten(1) + global_features)[:, :, None]

        def _forward_backbone(self, hidden):
            calls.append("backbone")
            zero = hidden.new_zeros(())
            return BackboneOutput(hidden, ModelDiagnostics(zero, zero, zero, zero))

        def _decode_outputs(self, hidden):
            calls.append("decode")
            value = hidden[:, 0] * self.weight
            return ModelPredictions(
                policy_logits=torch.cat((value, value), dim=-1),
                value=value,
                ownership=value,
                score_margin=value,
            )

        def _parameter_group_members(self):
            return {"base": (self.weight,)}

        def configure_execution(self, config):
            del config

        def compile_training_components(self, *, dynamic, mode):
            del dynamic, mode

    model = RecordingModel()
    output = model(
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1),
        torch.tensor([[False, True]]),
    )
    assert calls == ["encode", "backbone", "decode"]
    assert torch.isneginf(output.policy_logits[0, 0])
    assert torch.isfinite(output.policy_logits[0, 1])


def test_model_rejects_invalid_input_contracts() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    board, global_features, legal = model_inputs()
    with pytest.raises(TypeError, match="boolean"):
        model(board, global_features, legal.float())
    with pytest.raises(TypeError, match=r"torch\.float32"):
        model(board.double(), global_features, legal)
    with pytest.raises(TypeError, match=r"torch\.float32"):
        model(board.to(torch.bfloat16), global_features.to(torch.bfloat16), legal)
    with pytest.raises(ValueError, match="cannot be empty"):
        model(board[:0], global_features[:0], legal[:0])
