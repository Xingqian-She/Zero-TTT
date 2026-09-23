"""Run phase-separated strict-FP32 GPU memory smokes for production configs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
from zero_ttt.config import ExperimentConfig, load_config
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE
from zero_ttt.model import ModelOutput, PolicyValueTransformer
from zero_ttt.precision import configure_strict_fp32, require_fp32_module
from zero_ttt_trainer.ema import update_slow_weights
from zero_ttt_trainer.gradients import clip_model_gradients, parameters_are_finite
from zero_ttt_trainer.losses import TrainingTargets, compute_losses

MEMORY_LIMIT_BYTES = int(14.5 * 1024**3)
PARAMETER_LIMIT = 630_000_000
EXPECTED_PARAMETERS = {
    "configs/profiles/rtx4090l.toml": 43_371_150,
    "configs/profiles/rtx4090l_baseline.toml": 41_189_893,
    "configs/profiles/rtx4090l_625m_future.toml": 625_357_745,
    "configs/profiles/rtx4090l_625m_future_baseline.toml": 620_432_901,
}


@dataclass(slots=True)
class TrainingMeasurement:
    output: ModelOutput
    loss: torch.Tensor
    microbatch_seconds: list[float]
    optimizer_step_seconds: list[float]
    gradient_norms: list[float]
    hyper_gradient_norms: list[float]
    ema_update_seconds: float
    ema_natural: bool
    memory_peaks: dict[str, float]


def _build_optimizer(
    model: PolicyValueTransformer,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    optimizer_groups: list[dict[str, Any]] = []
    for group in model.parameter_groups():
        for parameters, weight_decay in (
            (group.decay, config.training.weight_decay),
            (group.no_decay, 0.0),
        ):
            if parameters:
                optimizer_groups.append(
                    {
                        "params": parameters,
                        "weight_decay": weight_decay,
                        "lr": 0.0,
                        "group_name": group.name,
                    }
                )
    return torch.optim.AdamW(
        optimizer_groups,
        lr=0.0,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        fused=True,
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    samples_seen: int,
) -> float:
    warmup = config.training.warmup_samples
    if samples_seen <= warmup:
        learning_rate = config.training.learning_rate * samples_seen / warmup
    else:
        learning_rate = config.training.learning_rate * math.sqrt(warmup / samples_seen)
    for group in optimizer.param_groups:
        multiplier = (
            config.training.hypernet.learning_rate_multiplier
            if group["group_name"] == "hypernet"
            else 1.0
        )
        group["lr"] = learning_rate * multiplier
    return learning_rate


def _targets(batch: int, device: torch.device) -> TrainingTargets:
    return TrainingTargets(
        policy=torch.full(
            (batch, ACTION_SIZE),
            1.0 / ACTION_SIZE,
            device=device,
        ),
        value=torch.zeros(batch, device=device),
        ownership=torch.zeros(batch, BOARD_AREA, device=device),
        score_margin=torch.zeros(batch, device=device),
        value_mask=torch.ones(batch, dtype=torch.bool, device=device),
        ownership_mask=torch.ones(batch, dtype=torch.bool, device=device),
        score_mask=torch.ones(batch, dtype=torch.bool, device=device),
    )


def _loss(
    model: PolicyValueTransformer,
    board: torch.Tensor,
    global_features: torch.Tensor,
    legal: torch.Tensor,
    targets: TrainingTargets,
    config: ExperimentConfig,
) -> tuple[ModelOutput, torch.Tensor]:
    output = model(board, global_features, legal)
    loss = compute_losses(output, targets, config.training).total
    return output, loss


def _clip_gradients(
    model: PolicyValueTransformer,
    config: ExperimentConfig,
) -> tuple[float, float | None]:
    norms = clip_model_gradients(
        model.parameter_groups(),
        base_max_norm=config.training.gradient_clip,
        hypernet_max_norm=config.training.hypernet.gradient_clip,
    )
    return norms.base, norms.hypernet


def _branch_gradients_are_finite(model: PolicyValueTransformer) -> tuple[bool, bool]:
    hyper_finite = not model.config.hypernet.enabled or all(
        parameter.grad is not None and parameters_are_finite((parameter.grad,))
        for parameter in model.block_plugin.parameters()
    )
    dwa_finite = not model.config.depth_mixing.enabled or all(
        parameter.grad is not None and parameters_are_finite((parameter.grad,))
        for parameter in model.depth_mixer.parameters()
    )
    return bool(hyper_finite), bool(dwa_finite)


def _measure_training(
    fast: PolicyValueTransformer,
    slow: PolicyValueTransformer,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: TrainingTargets,
    accumulation: int,
    measured_optimizer_steps: int,
) -> TrainingMeasurement:
    board, global_features, legal = inputs
    effective_batch = config.training.batch_size * accumulation
    ema_pending_samples = effective_batch
    next_ema_sample = config.training.ema_update_interval_samples
    microbatch_seconds: list[float] = []
    optimizer_step_seconds: list[float] = []
    gradient_norms: list[float] = []
    hyper_gradient_norms: list[float] = []
    ema_update_seconds: float | None = None
    ema_natural = False
    memory_peaks: dict[str, float] = {}
    output: ModelOutput | None = None
    loss: torch.Tensor | None = None
    for optimizer_step in range(2, measured_optimizer_steps + 2):
        samples_seen = optimizer_step * effective_batch
        _set_learning_rate(optimizer, config, samples_seen)
        optimizer.zero_grad(set_to_none=False)
        step_started = time.perf_counter()
        for _ in range(accumulation):
            torch.cuda.synchronize()
            microbatch_started = time.perf_counter()
            output, loss = _loss(fast, board, global_features, legal, targets, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("production smoke loss is non-finite")
            (loss / accumulation).backward()
            torch.cuda.synchronize()
            microbatch_seconds.append(time.perf_counter() - microbatch_started)
        memory_peaks["accumulation"] = torch.cuda.max_memory_reserved() / 1024**3
        gradient_norm, hyper_gradient_norm = _clip_gradients(fast, config)
        gradient_norms.append(gradient_norm)
        if hyper_gradient_norm is not None:
            hyper_gradient_norms.append(hyper_gradient_norm)
        optimizer.step()
        torch.cuda.synchronize()
        memory_peaks["optimizer_step"] = torch.cuda.max_memory_reserved() / 1024**3
        optimizer_step_seconds.append(time.perf_counter() - step_started)
        ema_pending_samples += effective_batch
        if samples_seen >= next_ema_sample:
            ema_started = time.perf_counter()
            update_slow_weights(
                slow,
                fast,
                samples=ema_pending_samples,
                half_life_samples=config.training.ema_half_life_samples,
            )
            ema_update_seconds = time.perf_counter() - ema_started
            ema_pending_samples = 0
            ema_natural = True
            interval = config.training.ema_update_interval_samples
            next_ema_sample = (samples_seen // interval + 1) * interval
    if ema_update_seconds is None:
        ema_started = time.perf_counter()
        update_slow_weights(
            slow,
            fast,
            samples=ema_pending_samples,
            half_life_samples=config.training.ema_half_life_samples,
        )
        ema_update_seconds = time.perf_counter() - ema_started
    if output is None or loss is None:
        raise RuntimeError("production smoke executed no measured optimizer steps")
    memory_peaks["ema"] = torch.cuda.max_memory_reserved() / 1024**3
    return TrainingMeasurement(
        output,
        loss,
        microbatch_seconds,
        optimizer_step_seconds,
        gradient_norms,
        hyper_gradient_norms,
        ema_update_seconds,
        ema_natural,
        memory_peaks,
    )


def run_case(
    config_path: str,
    measured_optimizer_steps: int,
    *,
    disable_compile: bool = False,
    accumulation_steps: int | None = None,
) -> dict[str, Any]:
    configure_strict_fp32()
    config = load_config(config_path)
    torch.manual_seed(config.seed)
    if config.runtime.device != "cuda" or config.runtime.ema_device != "cpu":
        raise ValueError("the production smoke test requires CUDA fast weights and CPU EMA")
    if measured_optimizer_steps <= 0:
        raise ValueError("measured_optimizer_steps must be positive")
    if accumulation_steps is not None and accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")

    compile_model = config.execution.compile_model and not disable_compile
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fast = PolicyValueTransformer(config.model, config.execution).cuda().train()
    slow = PolicyValueTransformer(config.model, config.execution).cpu().eval().requires_grad_(False)
    slow.load_state_dict(fast.state_dict())
    require_fp32_module(fast, "smoke fast model")
    require_fp32_module(slow, "smoke slow model")
    optimizer = _build_optimizer(fast, config)
    if compile_model:
        fast.compile_training_components(dynamic=False, mode=config.execution.compile_mode)

    batch = config.training.batch_size
    accumulation = accumulation_steps or config.training.accumulation_steps
    device = torch.device("cuda")
    board = torch.zeros(
        batch,
        config.model.input_planes,
        BOARD_SIZE,
        BOARD_SIZE,
        device=device,
    )
    global_features = torch.zeros(batch, config.model.global_features, device=device)
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device=device)
    targets = _targets(batch, device)

    warmup_started = time.perf_counter()
    effective_batch = batch * accumulation
    _set_learning_rate(optimizer, config, samples_seen=effective_batch)
    optimizer.zero_grad(set_to_none=False)
    warmup_output, warmup_loss = _loss(fast, board, global_features, legal, targets, config)
    (warmup_loss / accumulation).backward()
    _clip_gradients(fast, config)
    optimizer.step()
    if not parameters_are_finite(fast.parameters()):
        raise FloatingPointError("production smoke parameter is non-finite")
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    warmup_peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    post_warmup_reserved_gib = torch.cuda.memory_reserved() / 1024**3

    measurement = _measure_training(
        fast,
        slow,
        optimizer,
        config,
        (board, global_features, legal),
        targets,
        accumulation,
        measured_optimizer_steps,
    )
    output = measurement.output
    loss = measurement.loss
    hyper_gradients_finite, dwa_gradients_finite = _branch_gradients_are_finite(fast)
    if not hyper_gradients_finite or not dwa_gradients_finite:
        raise FloatingPointError("production smoke branch gradient is missing or non-finite")
    if not parameters_are_finite(fast.parameters()):
        raise FloatingPointError("production smoke parameter is non-finite")
    gradients = (parameter.grad for parameter in fast.parameters() if parameter.grad is not None)
    if not all(gradient.dtype == torch.float32 for gradient in gradients):
        raise TypeError("production smoke gradients must use float32")

    torch.cuda.synchronize()
    measurement.memory_peaks["validation"] = torch.cuda.max_memory_reserved() / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if peak_reserved > MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"peak allocated/reserved memory {peak_allocated / 1024**3:.2f}/"
            f"{peak_reserved / 1024**3:.2f} GiB exceeds 14.5 GiB reserved; "
            f"warmup peak/current: {warmup_peak_reserved_gib:.3f}/"
            f"{post_warmup_reserved_gib:.3f} GiB; phase peaks: {measurement.memory_peaks}"
        )
    parameters = sum(parameter.numel() for parameter in fast.parameters())
    expected_parameters = EXPECTED_PARAMETERS.get(config_path)
    if expected_parameters is not None and parameters != expected_parameters:
        raise RuntimeError(
            f"{config_path} has {parameters} parameters, expected {expected_parameters}"
        )
    if config.model.hypernet.enabled and parameters >= PARAMETER_LIMIT:
        raise RuntimeError(f"default model has {parameters} parameters, expected <630M")

    inference_state = slow.state_dict()
    result = {
        "config": config_path,
        "config_sha256": config.sha256,
        "effective_config": json.loads(config.canonical_json()),
        "hypernet_enabled": config.model.hypernet.enabled,
        "depth_mixing_enabled": config.model.depth_mixing.enabled,
        "parameters": parameters,
        "batch_size": batch,
        "configured_accumulation_steps": config.training.accumulation_steps,
        "smoke_accumulation_steps": accumulation,
        "compile_enabled": compile_model,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "measured_optimizer_steps": measured_optimizer_steps,
        "warmup_optimizer_steps": 1,
        "training_warmup_seconds": warmup_seconds,
        "training_warmup_peak_reserved_gib": warmup_peak_reserved_gib,
        "median_microbatch_seconds": statistics.median(measurement.microbatch_seconds),
        "median_optimizer_step_seconds": statistics.median(measurement.optimizer_step_seconds),
        "ema_update_seconds": measurement.ema_update_seconds,
        "ema_natural": measurement.ema_natural,
        "training_peak_allocated_gib": peak_allocated / 1024**3,
        "training_peak_reserved_gib": peak_reserved / 1024**3,
        "loss": loss.detach().item(),
        "base_gradient_norm": statistics.median(measurement.gradient_norms),
        "hypernet_gradient_norm": (
            statistics.median(measurement.hyper_gradient_norms)
            if measurement.hyper_gradient_norms
            else None
        ),
        "hyper_gradients_finite": hyper_gradients_finite,
        "dwa_gradients_finite": dwa_gradients_finite,
        "hyper_a_saturation": output.diagnostics.hyper_a_saturation.item(),
        "hyper_b_saturation": output.diagnostics.hyper_b_saturation.item(),
        "hyper_dynamic_rms": output.diagnostics.hyper_dynamic_rms.item(),
        "hyper_static_rms": output.diagnostics.hyper_static_rms.item(),
    }
    del output, loss, measurement, warmup_output, warmup_loss, gradients
    del optimizer, slow, fast
    del board, global_features, legal, targets
    gc.collect()
    torch.cuda.empty_cache()
    result.update(_run_inference_phase(config, inference_state, compile_model))
    return result


def _run_inference_phase(
    config: ExperimentConfig,
    state: dict[str, torch.Tensor],
    compile_model: bool,
) -> dict[str, float]:
    torch.cuda.reset_peak_memory_stats()
    model = PolicyValueTransformer(config.model, config.execution)
    model.load_state_dict(state)
    model = model.cuda().eval().requires_grad_(False)
    require_fp32_module(model, "smoke inference model")
    batch = config.selfplay.inference_batch_size
    board = torch.zeros(batch, config.model.input_planes, BOARD_SIZE, BOARD_SIZE, device="cuda")
    global_features = torch.zeros(batch, config.model.global_features, device="cuda")
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device="cuda")
    started = time.perf_counter()
    if compile_model:
        model = torch.compile(model, dynamic=False, mode=config.execution.compile_mode)
    with torch.inference_mode():
        output = model(board, global_features, legal)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tensors = (output.policy_logits, output.value, output.ownership, output.score_margin)
    if not all(tensor.dtype == torch.float32 for tensor in tensors):
        raise TypeError("production inference output must use float32")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError("production inference output is non-finite")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if peak_reserved > MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"inference peak allocated/reserved memory {peak_allocated / 1024**3:.2f}/"
            f"{peak_reserved / 1024**3:.2f} GiB exceeds 14.5 GiB reserved"
        )
    result = {
        "inference_seconds": elapsed,
        "inference_peak_allocated_gib": peak_allocated / 1024**3,
        "inference_peak_reserved_gib": peak_reserved / 1024**3,
    }
    del output, model, board, global_features, legal
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/profiles/rtx4090l_baseline.toml",
            "configs/profiles/rtx4090l.toml",
        ],
    )
    parser.add_argument("--default-optimizer-steps", type=int, default=16)
    parser.add_argument("--baseline-optimizer-steps", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--disable-compile", action="store_true")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production memory smoke test")
    results = []
    for path in arguments.configs:
        config = load_config(path)
        measured_steps = (
            arguments.default_optimizer_steps
            if config.model.hypernet.enabled
            else arguments.baseline_optimizer_steps
        )
        results.append(
            run_case(
                path,
                measured_steps,
                disable_compile=arguments.disable_compile,
                accumulation_steps=arguments.accumulation_steps,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
