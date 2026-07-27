"""E4 weight format, gradient accumulation, precision and resume (Audit 0039)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from mednorm_vi.training.phase2.artifacts import MODE_FULL, MODE_SMOKE
from mednorm_vi.training.phase2.e4_w2ner_training import (
    ATOMIC_PROJECTION_VERSION,
    DEVICE_CPU,
    DEVICE_CUDA,
    E4_INPUT_CONTRACT_VERSION,
    E4_PINNED_MODEL_REVISION,
    E4_WEIGHT_FORMAT_BIN,
    E4_WEIGHT_FORMAT_SAFETENSORS,
    FULL_RESUME_COMPATIBILITY_FIELDS,
    INITIALIZATION_FULL_RESUME,
    INITIALIZATION_PINNED_BASE,
    INITIALIZATION_PINNED_BASE_SMOKE,
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    E4TrainingContractError,
    PhoBERTWeightFormat,
    assert_compatible_full_resume,
    assert_full_checkpoint_custody,
    assert_full_initialization_source,
    assert_full_training_device,
    assert_optimizer_step_accounting,
    assert_weight_format_loadable,
    build_e4_history_row,
    build_e4_manifest,
    build_e4_resolved_config,
    build_e4_training_accounting,
    build_e4_training_state_payload,
    optimizer_signature,
    plan_gradient_accumulation,
    resolve_mixed_precision_policy,
    resolve_phobert_weight_format,
    validate_phobert_encoder_load_report,
)

REPO = Path(__file__).resolve().parents[2]
E4_NOTEBOOK = REPO / "notebooks" / "MedNorm_E4_PhoBERT_W2NER_Training.ipynb"
E4_CONFIG = REPO / "configs" / "training" / "phase2_e4_phobert_w2ner_colab.yaml"

GOVERNED_TRAIN_EXAMPLES = 33826


def _notebook_code() -> list[str]:
    payload = json.loads(E4_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in payload["cells"]
            if cell["cell_type"] == "code"]


def _config() -> dict:
    return yaml.safe_load(E4_CONFIG.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A. Pretrained weight format
# ---------------------------------------------------------------------------


def test_pinned_revision_resolves_to_the_official_bin_checkpoint() -> None:
    resolved = resolve_phobert_weight_format(
        "vinai/phobert-large", E4_PINNED_MODEL_REVISION)
    assert resolved.filename == E4_WEIGHT_FORMAT_BIN
    assert resolved.use_safetensors is False
    assert resolved.model_revision == E4_PINNED_MODEL_REVISION


def test_repository_listing_without_safetensors_resolves_to_bin() -> None:
    resolved = resolve_phobert_weight_format(
        "vinai/phobert-large", E4_PINNED_MODEL_REVISION,
        repository_files=["config.json", "pytorch_model.bin", "vocab.txt", "bpe.codes"])
    assert resolved.use_safetensors is False
    assert resolved.filename == E4_WEIGHT_FORMAT_BIN
    assert resolved.resolved_by == "repository_file_listing"


def test_repository_listing_with_safetensors_would_resolve_to_safetensors() -> None:
    resolved = resolve_phobert_weight_format(
        "some/other-model", "b" * 40,
        repository_files=["config.json", "model.safetensors"])
    assert resolved.use_safetensors is True
    assert resolved.filename == E4_WEIGHT_FORMAT_SAFETENSORS


def test_repository_with_neither_weight_file_fails_loudly() -> None:
    with pytest.raises(E4TrainingContractError, match="publishes neither"):
        resolve_phobert_weight_format(
            "some/other-model", "b" * 40, repository_files=["config.json"])


def test_official_pinned_revision_cannot_request_safetensors() -> None:
    bad = PhoBERTWeightFormat(
        filename=E4_WEIGHT_FORMAT_SAFETENSORS, use_safetensors=True,
        model_revision=E4_PINNED_MODEL_REVISION, resolved_by="manual")
    with pytest.raises(E4TrainingContractError, match="does not publish"):
        assert_weight_format_loadable(bad)


def test_weight_format_flag_must_agree_with_the_filename() -> None:
    inconsistent = PhoBERTWeightFormat(
        filename=E4_WEIGHT_FORMAT_BIN, use_safetensors=True,
        model_revision="c" * 40, resolved_by="manual")
    with pytest.raises(E4TrainingContractError, match="disagrees"):
        assert_weight_format_loadable(inconsistent)


def test_resolved_config_records_the_bin_format() -> None:
    config = build_e4_resolved_config(
        mode=MODE_FULL, model_revision=E4_PINNED_MODEL_REVISION,
        tokenizer_revision=E4_PINNED_MODEL_REVISION, seed=1,
        max_words=256, effective_batch_size=8)
    assert config["pretrained_weight_format"] == E4_WEIGHT_FORMAT_BIN
    assert config["use_safetensors"] is False
    assert config["model_revision"] == E4_PINNED_MODEL_REVISION
    assert config["tokenizer_revision"] == E4_PINNED_MODEL_REVISION


def _notebook_executable_lines() -> list[str]:
    """Notebook code with comment-only lines removed.

    The narrative comments deliberately quote the failing ``use_safetensors=True``
    call, so the assertion must look at executable code rather than prose.
    """
    lines: list[str] = []
    for source in _notebook_code():
        for line in source.splitlines():
            if line.strip().startswith("#"):
                continue
            lines.append(line)
    return lines


def test_notebook_never_requests_safetensors_on_the_active_path() -> None:
    executable = "\n".join(_notebook_executable_lines())
    assert "use_safetensors=True" not in executable
    assert "use_safetensors=WEIGHT_FORMAT.use_safetensors" in executable
    assert "resolve_phobert_weight_format(" in executable
    assert "assert_weight_format_loadable(" in executable


def test_tracked_config_pins_the_bin_weight_format() -> None:
    model = _config()["model"]
    assert model["pretrained_weight_format"] == E4_WEIGHT_FORMAT_BIN
    assert model["use_safetensors"] is False
    assert model["observed_model_revision"] == E4_PINNED_MODEL_REVISION
    assert model["observed_tokenizer_revision"] == E4_PINNED_MODEL_REVISION


def test_expected_mlm_head_mismatch_is_still_accepted() -> None:
    report = validate_phobert_encoder_load_report(
        missing_keys=(), unexpected_keys=("lm_head.dense.weight", "lm_head.bias"))
    assert report["w2ner_head_expected_from_base"] is False
    assert len(report["ignored_mlm_head_keys"]) == 2


def test_unexpected_encoder_mismatch_is_still_rejected() -> None:
    with pytest.raises(E4TrainingContractError):
        validate_phobert_encoder_load_report(
            missing_keys=("encoder.layer.0.attention.self.query.weight",),
            unexpected_keys=())
    with pytest.raises(E4TrainingContractError):
        validate_phobert_encoder_load_report(
            missing_keys=(), unexpected_keys=("some.other.head.weight",))


# ---------------------------------------------------------------------------
# C. Real gradient accumulation
# ---------------------------------------------------------------------------


def test_expected_optimizer_steps_use_ceiling_division() -> None:
    plan = plan_gradient_accumulation(
        GOVERNED_TRAIN_EXAMPLES, micro_batch_size=1, accumulation_steps=8, epochs=12)
    steps_per_epoch = -(-GOVERNED_TRAIN_EXAMPLES // 8)
    assert plan.optimizer_steps_per_epoch == steps_per_epoch
    assert plan.expected_optimizer_steps == steps_per_epoch * 12
    assert plan.expected_backward_passes == GOVERNED_TRAIN_EXAMPLES * 12
    assert plan.effective_batch_size == 8


def test_final_partial_group_is_smaller_and_scaled_by_its_real_size() -> None:
    plan = plan_gradient_accumulation(
        GOVERNED_TRAIN_EXAMPLES, micro_batch_size=1, accumulation_steps=8, epochs=1)
    assert plan.has_partial_final_group is True
    assert plan.final_partial_group_size == GOVERNED_TRAIN_EXAMPLES % 8
    last = plan.micro_batches_per_epoch - 1
    assert plan.group_size_for(last) == plan.final_partial_group_size
    # Not 1/8: dividing the trailing group by accumulation_steps would silently
    # under-scale its gradient.
    assert plan.loss_scale_for(last) == pytest.approx(1 / plan.final_partial_group_size)
    assert plan.loss_scale_for(0) == pytest.approx(1 / 8)


def test_exactly_divisible_run_has_no_partial_group() -> None:
    plan = plan_gradient_accumulation(
        64, micro_batch_size=1, accumulation_steps=8, epochs=2)
    assert plan.has_partial_final_group is False
    assert plan.final_partial_group_size == 8
    assert plan.optimizer_steps_per_epoch == 8
    assert plan.expected_optimizer_steps == 16


def test_optimizer_steps_only_at_accumulation_boundaries() -> None:
    plan = plan_gradient_accumulation(
        10, micro_batch_size=1, accumulation_steps=4, epochs=1)
    boundaries = [
        index for index in range(plan.micro_batches_per_epoch)
        if plan.is_optimizer_step_boundary(index)
    ]
    # Groups are [0..3], [4..7], [8..9]; the last is partial but still steps.
    assert boundaries == [3, 7, 9]
    assert plan.optimizer_steps_per_epoch == len(boundaries) == 3
    for index in (0, 1, 2, 4, 5, 6, 8):
        assert plan.is_optimizer_step_boundary(index) is False


def test_simulated_loop_matches_the_planned_accounting() -> None:
    """A loop-shaped simulation: steps, backwards and scales, without torch."""
    plan = plan_gradient_accumulation(
        10, micro_batch_size=1, accumulation_steps=4, epochs=3)
    optimizer_steps = 0
    backward_passes = 0
    accumulated_scale = 0.0
    per_group_scales: list[float] = []
    for _epoch in range(plan.epochs):
        group_total = 0.0
        for index in range(plan.micro_batches_per_epoch):
            backward_passes += 1
            group_total += plan.loss_scale_for(index)
            accumulated_scale += plan.loss_scale_for(index)
            if plan.is_optimizer_step_boundary(index):
                optimizer_steps += 1
                per_group_scales.append(round(group_total, 6))
                group_total = 0.0
    assert optimizer_steps == plan.expected_optimizer_steps
    assert backward_passes == plan.expected_backward_passes
    # Every group's scales sum to exactly 1.0, including the partial one.
    assert per_group_scales == [1.0] * plan.expected_optimizer_steps
    assert accumulated_scale == pytest.approx(float(plan.expected_optimizer_steps))
    assert_optimizer_step_accounting(plan, optimizer_steps)


def test_step_accounting_mismatch_is_rejected() -> None:
    plan = plan_gradient_accumulation(
        10, micro_batch_size=1, accumulation_steps=4, epochs=1)
    with pytest.raises(E4TrainingContractError, match="optimizer step accounting"):
        assert_optimizer_step_accounting(plan, plan.micro_batches_per_epoch)


def test_manifest_accounting_must_match_the_real_loop() -> None:
    plan = plan_gradient_accumulation(
        10, micro_batch_size=1, accumulation_steps=4, epochs=1)
    precision = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    weight_format = resolve_phobert_weight_format(
        "vinai/phobert-large", E4_PINNED_MODEL_REVISION)
    accounting = build_e4_training_accounting(
        accumulation=plan, precision=precision, weight_format=weight_format,
        observed_optimizer_steps=3, observed_backward_passes=10,
        observed_examples=10, max_grad_norm=1.0)
    assert accounting["observed_optimizer_steps"] == 3
    assert accounting["effective_batch_size"] == 4
    assert accounting["gradient_clipping_enabled"] is True
    assert accounting["max_grad_norm"] == 1.0
    # Relabelling the batch size without doing the work is refused.
    with pytest.raises(E4TrainingContractError, match="optimizer step accounting"):
        build_e4_training_accounting(
            accumulation=plan, precision=precision, weight_format=weight_format,
            observed_optimizer_steps=10, observed_backward_passes=10,
            observed_examples=10, max_grad_norm=1.0)
    with pytest.raises(E4TrainingContractError, match="backward-pass accounting"):
        build_e4_training_accounting(
            accumulation=plan, precision=precision, weight_format=weight_format,
            observed_optimizer_steps=3, observed_backward_passes=3,
            observed_examples=10, max_grad_norm=1.0)


def test_invalid_accumulation_configuration_is_rejected() -> None:
    for kwargs in (
        {"micro_batch_size": 0, "accumulation_steps": 8, "epochs": 1},
        {"micro_batch_size": 1, "accumulation_steps": 0, "epochs": 1},
        {"micro_batch_size": 1, "accumulation_steps": 8, "epochs": 0},
    ):
        with pytest.raises(E4TrainingContractError):
            plan_gradient_accumulation(10, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(E4TrainingContractError):
        plan_gradient_accumulation(0, micro_batch_size=1, accumulation_steps=8, epochs=1)


def test_micro_batch_index_outside_the_epoch_is_rejected() -> None:
    plan = plan_gradient_accumulation(
        4, micro_batch_size=1, accumulation_steps=2, epochs=1)
    for bad_index in (-1, 4):
        with pytest.raises(E4TrainingContractError):
            plan.group_size_for(bad_index)
        with pytest.raises(E4TrainingContractError):
            plan.is_optimizer_step_boundary(bad_index)


def test_notebook_implements_real_accumulation_with_clipping() -> None:
    joined = "\n".join(_notebook_code())
    assert "ACCUMULATION.is_optimizer_step_boundary(micro_batch_index)" in joined
    assert "ACCUMULATION.loss_scale_for(micro_batch_index)" in joined
    assert "clip_grad_norm_(trainable, MAX_GRAD_NORM)" in joined
    assert "plan_gradient_accumulation(" in joined
    # The optimizer must not step unconditionally per document any more.
    assert "loss.backward()\n            optimizer.step()" not in joined


def test_tracked_config_declares_accumulation_and_clipping() -> None:
    training = _config()["training"]
    assert training["micro_batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 8
    assert training["effective_batch_size"] == 8
    assert training["gradient_clipping_enabled"] is True
    assert training["max_grad_norm"] == 1.0
    assert training["final_partial_group_policy"] == "scale_loss_by_actual_group_size"


# ---------------------------------------------------------------------------
# D. Mixed precision and resource safety
# ---------------------------------------------------------------------------


def test_bf16_on_supported_cuda_uses_autocast_without_a_scaler() -> None:
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    assert policy.mode == PRECISION_BF16
    assert policy.autocast_enabled is True
    assert policy.use_grad_scaler is False
    assert policy.autocast_dtype_name == "torch.bfloat16"


def test_fp16_on_cuda_requires_a_grad_scaler() -> None:
    policy = resolve_mixed_precision_policy(
        PRECISION_FP16, device_type=DEVICE_CUDA, bf16_supported=True)
    assert policy.mode == PRECISION_FP16
    assert policy.use_grad_scaler is True
    assert policy.autocast_dtype_name == "torch.float16"


def test_bf16_without_hardware_support_degrades_to_fp16_with_a_scaler() -> None:
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=False)
    assert policy.mode == PRECISION_FP16
    assert policy.use_grad_scaler is True


def test_cpu_always_resolves_to_fp32() -> None:
    for requested in (PRECISION_FP32, PRECISION_FP16, PRECISION_BF16):
        policy = resolve_mixed_precision_policy(requested, device_type=DEVICE_CPU)
        assert policy.mode == PRECISION_FP32
        assert policy.autocast_enabled is False
        assert policy.use_grad_scaler is False


def test_unsupported_precision_mode_is_rejected() -> None:
    with pytest.raises(E4TrainingContractError, match="unsupported precision"):
        resolve_mixed_precision_policy("int8", device_type=DEVICE_CUDA)


def test_full_training_refuses_cpu_but_bounded_paths_allow_it() -> None:
    assert_full_training_device(DEVICE_CUDA)
    with pytest.raises(E4TrainingContractError, match="requires a CUDA device"):
        assert_full_training_device(DEVICE_CPU)


# --- GPU runtime policy: CUDA is required, a specific GPU model is not --------


def test_no_runtime_is_rejected_because_of_its_gpu_name() -> None:
    """T4 High-RAM, L4 and A100 are all acceptable; only CPU is refused."""
    for _gpu_name in ("Tesla T4", "NVIDIA L4", "NVIDIA A100-SXM4-40GB", "unknown"):
        assert_full_training_device(DEVICE_CUDA)


def test_precision_resolves_from_capability_not_from_the_device_name() -> None:
    # T4 (compute capability 7.5) has no bf16 -> fp16 with a GradScaler.
    t4 = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=False)
    assert t4.mode == PRECISION_FP16
    assert t4.use_grad_scaler is True
    assert t4.autocast_enabled is True
    # L4 / A100 support bf16 -> bf16, no scaler needed.
    for _runtime in ("L4", "A100"):
        policy = resolve_mixed_precision_policy(
            PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
        assert policy.mode == PRECISION_BF16
        assert policy.use_grad_scaler is False


def test_no_source_or_test_file_gates_on_a_gpu_model_name() -> None:
    """A100/T4/L4 may appear in prose, never in an executable comparison."""
    import re

    sources = [
        REPO / "src" / "mednorm_vi" / "training" / "phase2" / "e4_w2ner_training.py",
        REPO / "src" / "mednorm_vi" / "training" / "phase2" / "e4_runtime_io.py",
    ]
    gate = re.compile(r"(get_device_name|device_name|gpu_name)\s*\(?[^\n]*[=!]=")
    for path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0]
            assert not gate.search(code), f"{path.name}: GPU-name gate: {line!r}"


def test_notebook_reports_the_gpu_name_without_gating_on_it() -> None:
    joined = "\n".join(_notebook_code())
    assert '"gpu_name": GPU_NAME' in joined
    assert '"gpu_name_used_as_gate": False' in joined
    executable = "\n".join(_notebook_executable_lines())
    for forbidden in ('GPU_NAME == "', 'GPU_NAME != "', '"A100" in', '"T4" in'):
        assert forbidden not in executable


def test_tracked_config_declares_the_gpu_runtime_policy() -> None:
    policy = _config()["training"]["gpu_runtime_policy"]
    assert policy["requires_cuda"] is True
    assert policy["requires_specific_gpu_model"] is False
    assert policy["reject_by_device_name"] is False
    assert policy["supported_runtimes"] == ["T4 High-RAM", "L4", "A100"]
    assert policy["expected_precision_by_runtime"]["T4"] == "fp16_with_grad_scaler"
    assert policy["expected_precision_by_runtime"]["L4"] == "bf16"
    assert policy["expected_precision_by_runtime"]["A100"] == "bf16"
    assert policy["t4_vram_exhaustion_is_a_runtime_resource_limitation"] is True
    # The CUDA requirement itself is unchanged.
    assert _config()["training"]["full_training_requires_cuda"] is True


def test_resolved_config_records_the_precision_policy() -> None:
    plan = plan_gradient_accumulation(
        8, micro_batch_size=1, accumulation_steps=8, epochs=1)
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    config = build_e4_resolved_config(
        mode=MODE_FULL, model_revision=E4_PINNED_MODEL_REVISION,
        tokenizer_revision=E4_PINNED_MODEL_REVISION, seed=1, max_words=256,
        effective_batch_size=8, accumulation=plan, precision=policy)
    assert config["precision_mode"] == PRECISION_BF16
    assert config["use_grad_scaler"] is False
    assert config["expected_optimizer_steps"] == plan.expected_optimizer_steps
    assert config["quantization"] == "none"
    assert config["freeze_base_model"] is False


def test_notebook_uses_autocast_and_a_scaler_and_does_not_quantize() -> None:
    joined = "\n".join(_notebook_code())
    assert "torch.autocast(" in joined
    assert "GradScaler" in joined
    assert "scaler.unscale_(optimizer)" in joined
    assert "resolve_mixed_precision_policy(" in joined
    assert "quantiz" not in joined.lower()


# ---------------------------------------------------------------------------
# E/F. Initialization, resume and checkpoint custody
# ---------------------------------------------------------------------------


def _full_payload(**overrides):
    plan = plan_gradient_accumulation(
        8, micro_batch_size=1, accumulation_steps=8, epochs=1)
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    weight_format = resolve_phobert_weight_format(
        "vinai/phobert-large", E4_PINNED_MODEL_REVISION)
    payload = build_e4_training_state_payload(
        mode=MODE_FULL, config_sha256="0" * 64,
        model_revision=E4_PINNED_MODEL_REVISION,
        tokenizer_revision=E4_PINNED_MODEL_REVISION, parameter_count=1,
        weight_format=weight_format, accumulation=plan, precision=policy,
        optimizer_signature_value=optimizer_signature(
            name="AdamW", learning_rate=2e-5, weight_decay=0.0, max_grad_norm=1.0),
        epoch=3, optimizer_steps=3, backward_passes=24, examples_processed=24,
        best_metric=0.5, best_checkpoint_sha256="a" * 64,
        model_state={"base_model": {}, "w2ner_head": {}},
        optimizer_state={"state": {}}, scaler_state={"scale": 1.0})
    payload.update(overrides)
    return payload


def _expected_from(payload):
    return {field: payload[field] for field in FULL_RESUME_COMPATIBILITY_FIELDS}


def test_smoke_checkpoint_is_never_a_full_initializer() -> None:
    with pytest.raises(E4TrainingContractError, match="smoke checkpoint"):
        assert_full_initialization_source(
            run_full_training=True, resume_from_smoke_checkpoint=True,
            resume_from_full_checkpoint=False)


def test_initialization_sources_for_each_run_mode() -> None:
    assert assert_full_initialization_source(
        run_full_training=False, resume_from_smoke_checkpoint=False,
        resume_from_full_checkpoint=False) == INITIALIZATION_PINNED_BASE_SMOKE
    assert assert_full_initialization_source(
        run_full_training=True, resume_from_smoke_checkpoint=False,
        resume_from_full_checkpoint=False) == INITIALIZATION_PINNED_BASE
    assert assert_full_initialization_source(
        run_full_training=True, resume_from_smoke_checkpoint=False,
        resume_from_full_checkpoint=True,
        checkpoint_mode="full") == INITIALIZATION_FULL_RESUME


def test_full_resume_from_a_smoke_mode_checkpoint_is_rejected() -> None:
    with pytest.raises(E4TrainingContractError, match="full-training checkpoint"):
        assert_full_initialization_source(
            run_full_training=True, resume_from_smoke_checkpoint=False,
            resume_from_full_checkpoint=True, checkpoint_mode="smoke")


def test_full_checkpoint_carries_optimizer_and_scaler_state() -> None:
    payload = _full_payload()
    assert_full_checkpoint_custody(payload)
    assert payload["optimizer_state"] == {"state": {}}
    assert payload["scaler_state"] == {"scale": 1.0}
    assert payload["epoch"] == 3
    assert payload["optimizer_steps"] == 3
    assert payload["best_metric"] == 0.5
    assert payload["best_checkpoint_sha256"] == "a" * 64
    # No arbitrary scheduler is invented.
    assert payload["scheduler_configured"] is False
    assert payload["scheduler_state"] == {}


def test_a_checkpoint_without_resume_state_is_rejected() -> None:
    payload = _full_payload()
    del payload["optimizer_state"]
    with pytest.raises(E4TrainingContractError, match="missing resume state"):
        assert_full_checkpoint_custody(payload)


def test_a_checkpoint_without_head_state_is_rejected() -> None:
    payload = _full_payload(model_state={"base_model": {}})
    with pytest.raises(E4TrainingContractError, match="w2ner_head"):
        assert_full_checkpoint_custody(payload)


def test_compatible_full_resume_is_accepted() -> None:
    payload = _full_payload()
    assert_compatible_full_resume(payload, expected=_expected_from(payload))


def test_resume_with_changed_precision_is_rejected() -> None:
    payload = _full_payload()
    expected = _expected_from(payload) | {"precision_mode": PRECISION_FP16}
    with pytest.raises(E4TrainingContractError, match="precision_mode"):
        assert_compatible_full_resume(payload, expected=expected)


def test_resume_with_changed_accumulation_is_rejected() -> None:
    payload = _full_payload()
    expected = _expected_from(payload) | {
        "accumulation_signature": "micro1-accum16-effective16"}
    with pytest.raises(E4TrainingContractError, match="accumulation_signature"):
        assert_compatible_full_resume(payload, expected=expected)


def test_resume_with_changed_weight_format_is_rejected() -> None:
    payload = _full_payload()
    expected = _expected_from(payload) | {
        "pretrained_weight_format": E4_WEIGHT_FORMAT_SAFETENSORS}
    with pytest.raises(E4TrainingContractError, match="pretrained_weight_format"):
        assert_compatible_full_resume(payload, expected=expected)


def test_resume_with_changed_optimizer_or_config_hash_is_rejected() -> None:
    payload = _full_payload()
    for field, value in (("optimizer_signature", "AdamW-lr1e-05-wd0-clip1"),
                         ("config_sha256", "1" * 64),
                         ("model_revision", "d" * 40)):
        with pytest.raises(E4TrainingContractError, match=field):
            assert_compatible_full_resume(
                payload, expected=_expected_from(payload) | {field: value})


def test_a_smoke_mode_payload_cannot_resume_a_full_run() -> None:
    payload = _full_payload(mode=MODE_SMOKE)
    with pytest.raises(E4TrainingContractError, match="only a full-training checkpoint"):
        assert_compatible_full_resume(payload, expected=_expected_from(payload))


def test_resume_compatibility_covers_every_declared_field() -> None:
    assert set(FULL_RESUME_COMPATIBILITY_FIELDS) == {
        "e4_input_contract_version", "e4_checkpoint_schema_version",
        "atomic_projection_version", "config_sha256", "model_revision",
        "tokenizer_revision", "pretrained_weight_format", "precision_mode",
        "optimizer_signature", "accumulation_signature"}
    payload = _full_payload()
    assert payload["e4_input_contract_version"] == E4_INPUT_CONTRACT_VERSION
    assert payload["atomic_projection_version"] == ATOMIC_PROJECTION_VERSION


# ---------------------------------------------------------------------------
# G. History, manifest and governance
# ---------------------------------------------------------------------------


def test_history_row_records_the_required_per_epoch_fields() -> None:
    plan = plan_gradient_accumulation(
        8, micro_batch_size=1, accumulation_steps=8, epochs=1)
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    row = build_e4_history_row(
        epoch=1, mode=MODE_FULL, train_loss=0.25,
        validation_metrics={
            "validation_exact_precision": 0.5, "validation_exact_recall": 0.25,
            "validation_exact_f1": 0.333333},
        optimizer_steps=1, backward_passes=8, examples_processed=8,
        learning_rate=2e-5, accumulation=plan, precision=policy)
    for key in ("train_loss", "validation_exact_precision", "validation_exact_recall",
                "validation_exact_f1", "optimizer_steps", "learning_rate",
                "completed_examples", "backward_passes", "micro_batch_size",
                "accumulation_steps", "effective_batch_size", "precision_mode"):
        assert key in row
    assert row["internal_test_accessed"] is False


def test_manifest_carries_the_training_accounting() -> None:
    plan = plan_gradient_accumulation(
        8, micro_batch_size=1, accumulation_steps=8, epochs=1)
    policy = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    weight_format = resolve_phobert_weight_format(
        "vinai/phobert-large", E4_PINNED_MODEL_REVISION)
    accounting = build_e4_training_accounting(
        accumulation=plan, precision=policy, weight_format=weight_format,
        observed_optimizer_steps=1, observed_backward_passes=8,
        observed_examples=8, max_grad_norm=1.0)
    manifest = build_e4_manifest(
        mode=MODE_FULL, status="FULLY_TRAINED", run_completed=True,
        repository_commit="e" * 40,
        corpus_hashes={"train": "f" * 64}, data_hashes={"train": "f" * 64},
        resolved_config={"stage_id": "x"},
        model_revision=E4_PINNED_MODEL_REVISION,
        tokenizer_revision=E4_PINNED_MODEL_REVISION, seed=1,
        completed_epochs=1, optimizer_steps=1, effective_batch_size=8,
        parameter_count=1, checkpoint_hashes={"best": "a" * 64, "latest": "b" * 64},
        best_metric=0.0, train_split_id="t", validation_split_id="v",
        safe_to_resume=True, initialization_source=INITIALIZATION_PINNED_BASE,
        training_accounting=accounting)
    manifest.validate()
    payload = manifest.as_dict()
    assert payload["training_accounting"]["observed_optimizer_steps"] == 1
    assert payload["training_accounting"]["effective_batch_size"] == 8
    assert payload["internal_test_accessed"] is False


def test_best_criterion_remains_governed_validation_only() -> None:
    assert _config()["training"]["best_criterion"] == (
        "max_validation_exact_f1_governed_validation_only")
    assert _config()["data"]["internal_test_allowed"] is False


def test_recorded_smoke_evidence_matches_the_real_run() -> None:
    observed = _config()["observed_smoke_run"]
    assert observed["status"] == "SMOKE_EXECUTED"
    assert observed["artifact_validated"] is True
    assert observed["validation_exact_f1"] == 0.0
    assert observed["best_checkpoint_sha256"] == (
        "bd689ec9bdc824b5abb3c0fa6373a3a6461781ad88a004870a27e2b396b8bbb1")
    assert observed["latest_checkpoint_sha256"] == (
        "42265aedb53bc39e729056bd0ec073eda27f7b3b15d136c29b170280686f0999")
    assert observed["manifest_sha256"] == (
        "d2c36a1d15b395d86638fc3cdab2983689d0d3aef4c8e65ce41d3298c4c765ac")
    assert observed["validator_failures"] == []
    assert observed["internal_test_accessed"] is False
    # The smoke predates the Audit-0039 loop change, so it cannot stand in for the
    # regression smoke.
    assert observed["superseded_by_audit_0039_loop_changes"] is True


def test_regression_smoke_uses_a_fresh_output_directory() -> None:
    artifacts = _config()["artifacts"]
    assert artifacts["smoke_dir"].endswith("e4_phobert_w2ner_smoke_v2")
    assert artifacts["archived_smoke_dir"].endswith("e4_phobert_w2ner_smoke_v1")
    joined = "\n".join(_notebook_code())
    assert "e4_phobert_w2ner_smoke_v2" in joined


# ---------------------------------------------------------------------------
# I/J. Notebook operator settings and repository hygiene
# ---------------------------------------------------------------------------


def test_notebook_defaults_to_the_documented_smoke_settings() -> None:
    joined = "\n".join(_notebook_code())
    for line in ("RUN_SMOKE_TRAINING = True", "RUN_FULL_TRAINING = False",
                 'CONFIRM_FULL = ""', "RESUME_FROM_SMOKE_CHECKPOINT = False",
                 "RESUME_FROM_FULL_CHECKPOINT = False"):
        assert line in joined
    # The repository-wide notebook guard forbids the literal full-training
    # assignment anywhere in a committed notebook, so the full-mode settings are
    # documented in prose and the authorization string is referenced by name.
    assert "RUN_FULL_TRAINING = True" not in joined
    assert "E4_FULL_AUTHORIZATION" in joined
    assert "OPERATOR SETTINGS" in joined


def test_notebook_derives_the_effective_batch_size() -> None:
    joined = "\n".join(_notebook_code())
    assert "MICRO_BATCH_SIZE = 1" in joined
    assert "GRADIENT_ACCUMULATION_STEPS = 8" in joined
    assert "EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS" in joined


def test_notebook_prints_the_resolved_summary_before_model_acquisition() -> None:
    cells = _notebook_code()
    summary = next(i for i, s in enumerate(cells) if "resolved_execution_summary" in s)
    encoder = next(i for i, s in enumerate(cells) if "AutoModel.from_pretrained(" in s)
    assert summary < encoder
    source = cells[summary]
    for key in ("run_mode", "output_dir", "model_revision", "tokenizer_revision",
                "pretrained_weight_format", "micro_batch_size",
                "gradient_accumulation_steps", "effective_batch_size", "epochs",
                "expected_optimizer_steps", "mixed_precision_policy",
                "resume_source", "internal_test_accessed"):
        assert f'"{key}"' in source


def test_notebook_restores_full_resume_state() -> None:
    joined = "\n".join(_notebook_code())
    assert "assert_compatible_full_resume(" in joined
    assert 'optimizer.load_state_dict(resume_payload["optimizer_state"])' in joined
    assert "scaler.load_state_dict(resume_payload[\"scaler_state\"])" in joined
    assert 'start_epoch = int(resume_payload["epoch"]) + 1' in joined


def test_notebook_never_touches_internal_test() -> None:
    joined = "\n".join(_notebook_code())
    assert "internal_test.jsonl" not in joined
    assert joined.count('"internal_test_accessed": False') >= 5


def test_every_notebook_code_cell_parses() -> None:
    for index, source in enumerate(_notebook_code()):
        compile(source, f"e4_cell_{index}", "exec")


def test_no_model_cache_or_checkpoint_file_is_tracked_in_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip"))
        assert not line.startswith(("artifacts/", "weights/", "caches/", "checkpoint/"))


def test_notebook_does_not_embed_the_model_cache_in_the_artifact() -> None:
    joined = "\n".join(_notebook_code())
    assert 'MODEL_CACHE_DIR = DRIVE_ROOT / "model_cache" / "huggingface"' in joined
    assert 'OUTPUT_DIR / "model_cache"' not in joined
