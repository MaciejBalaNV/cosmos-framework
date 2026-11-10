# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared client and protocol probes for the Cosmos3.5 action notebooks.

Only fixture preparation/decoding imports the framework's CPU action normalizer.
No vLLM, model weights, GPU, LeRobot installation, or server launch is needed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

PROMPT_OPTIONS = {
    "action_prompt_format": "plain",
    "action_mode_description": True,
    "action_prompt_duration": False,
    "action_prompt_resolution": True,
}
MODE_SENTENCES = {
    "forward_dynamics": "Predict the future video given the image and action.",
    "inverse_dynamics": "Predict the underlying action given the video.",
    "wam": "Predict the future video and action given the image.",
    "policy_action_only": "Predict future action given the image.",
}


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric data of shape {shape}; got {array.shape}.")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} overflows float32.")
    return array


@dataclass
class Fixture:
    """All actions, history and state are already normalized, unpadded 59D rows."""

    image: Path
    video: Path
    actions: np.ndarray
    history: np.ndarray
    initial_state: np.ndarray
    valid_mask: np.ndarray
    domain_name: str
    domain_id: int
    caption: str
    fps: float
    size: tuple[int, int] = (736, 544)  # W,H; canonical 480-tier canvas.
    vision_shift: float = 3.5
    action_shift: float = 1.0
    pose_convention: str = "backward_anchored"
    caption_semantics: str | None = None
    provenance: str = "synthetic protocol probe; not a quality reference"

    def manifest(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, Path):
                data[key] = str(value.resolve())
        return {"action_space": "normalized_unified_v1", **data}

    def __post_init__(self) -> None:
        self.image, self.video = Path(self.image), Path(self.video)
        self.actions = _array(self.actions, (16, 59), "actions")
        self.history = _array(self.history, (2, 59), "history")
        self.initial_state = _array(self.initial_state, (59,), "initial_state")
        mask = np.asarray(self.valid_mask)
        if mask.shape != (59,) or mask.dtype.kind not in "bifu" or not np.isin(mask, [0, 1]).all():
            raise ValueError("valid_mask must contain 59 booleans or binary numbers.")
        self.valid_mask = mask.astype(bool)
        if not self.valid_mask.any():
            raise ValueError("valid_mask needs at least one active channel.")
        if not self.caption.strip():
            raise ValueError("A non-empty original caption is required to exercise prompt rendering.")
        for value in (self.actions, self.history, self.initial_state):
            if np.any(value[..., ~self.valid_mask]):
                raise ValueError("Fixture inactive channels must already be zero.")
        if self.pose_convention != "backward_anchored":
            raise ValueError("These probes require history and future poses anchored at the current observation.")
        if self.size != (736, 544):
            raise ValueError("These probes use the canonical 736x544 canvas; regenerate references to change it.")
        if not all(np.isfinite(v) and v > 0 for v in (self.fps, self.vision_shift, self.action_shift)):
            raise ValueError("FPS and schedule shifts must be positive and finite.")
        for path in (self.image, self.video):
            if not path.is_file():
                raise FileNotFoundError(path)


def action_normalizer(framework_root: Path, stats_path: Path | None = None):
    """Use the framework normalizer, including its exact statistics and inverse."""
    from cosmos_framework.data.generator.action.utils.action_processing import (
        load_action_normalization_stats,
        resolve_action_normalization,
    )

    if stats_path is None:
        stats_path = (
            framework_root
            / "cosmos_framework/data/generator/action/normalizers"
            / "global_asinh_v1_backward_anchored_gripper_metric.json"
        )
    return resolve_action_normalization(
        "global_asinh",
        load_action_normalization_stats("global_asinh", stats_path=stats_path, expected_dim=59),
        pose_convention="backward_anchored",
    )


def protocol_fixtures(action_root: Path, normalizer) -> tuple[Fixture, Fixture]:
    """Analytic anchored wrist/camera motion with identity rot6d and metric gripper.

    The media supplies an observation, not ground truth for the analytic actions.
    Negative-time history rows share the current observation's coordinate frame.
    """
    import torch

    def make(camera: bool) -> Fixture:
        raw = np.zeros((19, 59), dtype=np.float32)  # two history, state, sixteen future rows.
        slots = slice(0, 9) if camera else slice(9, 18)
        mask = np.zeros(59, dtype=bool)
        mask[slots] = True
        poses = np.zeros((19, 9), dtype=np.float32)
        poses[:, 2] = np.arange(-2, 17) * 0.002
        poses[:, 3:] = [1, 0, 0, 0, 1, 0]
        raw[:, slots] = poses
        if not camera:
            mask[33] = True
            raw[:, 33] = 0.04  # metric width in meters, not an 8D joint command.
        normalized = normalizer.normalize_action(torch.from_numpy(raw)).numpy()
        normalized[:, ~mask] = 0
        return Fixture(
            image=action_root / ("assets/images/av_0.jpg" if camera else "assets/images/umi.png"),
            video=action_root / ("assets/videos/av_0.mp4" if camera else "assets/videos/umi.mp4"),
            actions=normalized[3:],
            history=normalized[:2],
            initial_state=normalized[2],
            valid_mask=mask,
            domain_name="camera_pose" if camera else "umi",
            domain_id=2 if camera else 6,
            caption="Move forward." if camera else "Arrange the mouse on the table.",
            fps=10 if camera else 20,
        )

    return make(False), make(True)


def load_fixture(path: Path) -> Fixture:
    """Read a recorded fixture; media paths resolve relative to its JSON file."""
    data = json.loads(path.read_text())
    if data.pop("action_space", None) != "normalized_unified_v1":
        raise ValueError("Recorded fixtures must declare action_space=normalized_unified_v1.")
    for key in ("image", "video"):
        data[key] = path.parent / data[key]
    if "size" in data:
        data["size"] = tuple(data["size"])
    return Fixture(**data)


@dataclass
class Case:
    name: str
    mode: str
    fixture: Fixture
    extra: dict[str, Any] = field(default_factory=dict)
    form: dict[str, Any] = field(default_factory=dict)
    omit_num_frames: bool = False
    expected_error: str | None = None

    def request(self) -> tuple[dict[str, Any], dict[str, Any], Path]:
        f = self.fixture
        extra = {
            **PROMPT_OPTIONS,
            "action_mode": self.mode,
            "domain_name": f.domain_name,
            "raw_action_dim": 59,
            "action_chunk_size": 16,
            "action_valid_mask": f.valid_mask.tolist(),
        }
        if f.caption_semantics:
            extra["action_caption_semantics"] = f.caption_semantics
        if self.mode == "forward_dynamics":
            extra["action"] = f.actions.tolist()
        extra.update(copy.deepcopy(self.extra))
        # None removes a default, notably domain_name for structured mixed rows.
        extra = {key: value for key, value in extra.items() if value is not None}
        form = {
            "prompt": f.caption,
            "fps": f.fps,
            "size": f"{f.size[0]}x{f.size[1]}",
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "seed": 0,
            **self.form,
        }
        if not self.omit_num_frames and "num_frames" not in form:
            form["num_frames"] = 17
        media = f.video if self.mode == "inverse_dynamics" else f.image
        return form, extra, media


def forward_cases(robot: Fixture, camera: Fixture) -> list[Case]:
    if camera.domain_name != "camera_pose" or camera.domain_id != 2 or camera.valid_mask[9:].any():
        raise ValueError("The camera fixture must use camera_pose / 2 and only ego-pose slots [0:9].")
    mixed = np.stack([robot.actions[i] if i % 2 == 0 else camera.actions[i] for i in range(16)])
    payload = {
        "action_space": "normalized_unified_v1",
        "action": mixed.tolist(),
        "domain_names": [robot.domain_name if i % 2 == 0 else camera.domain_name for i in range(16)],
    }
    mixed_extra = {
        "domain_name": None,
        "action": payload,
        "action_valid_mask": (robot.valid_mask | camera.valid_mask).tolist(),
    }
    return [
        Case("fd_single", "forward_dynamics", robot),
        Case("fd_camera", "forward_dynamics", camera),
        Case("fd_inferred_frames", "forward_dynamics", robot, omit_num_frames=True),
        Case("fd_history", "forward_dynamics", robot, {"history_action": robot.history.tolist()}),
        Case(
            "fd_state",
            "forward_dynamics",
            robot,
            {"action_chunk_size": 17, "action": np.vstack([robot.initial_state, robot.actions]).tolist()},
        ),
        Case("fd_mixed", "forward_dynamics", robot, mixed_extra),
        Case(
            "fd_structured_single",
            "forward_dynamics",
            robot,
            {
                **mixed_extra,
                "action": {**payload, "action": robot.actions.tolist(), "domain_names": [robot.domain_name]},
            },
        ),
        Case(
            "bad_mixed_domain_count",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "domain_names": [robot.domain_name, "camera_pose"]}},
            expected_error="one name or one name per action row",
        ),
        Case(
            "bad_mixed_width",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "action": mixed[:, :58].tolist()}},
            expected_error=r"shape \[T, 59\]",
        ),
        Case(
            "bad_camera_slots",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "action": robot.actions.tolist(), "domain_names": ["camera_pose"]}},
            expected_error="zeroed non-camera slots",
        ),
        Case(
            "bad_unknown_domain",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "domain_names": ["unknown_robot"]}},
            expected_error="Unknown Cosmos3 action domain_name",
        ),
        Case(
            "bad_duplicate_domain",
            "forward_dynamics",
            robot,
            {**mixed_extra, "domain_name": robot.domain_name},
            expected_error="do not also pass domain_name",
        ),
        Case(
            "bad_mixed_history",
            "forward_dynamics",
            robot,
            {**mixed_extra, "history_action": robot.history.tolist()},
            expected_error="do not support history_action",
        ),
        Case(
            "bad_mixed_row_count",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "action": mixed[:-1].tolist()}},
            expected_error="has 15 rows, but action_chunk_size requires 16",
        ),
        Case(
            "bad_mixed_values",
            "forward_dynamics",
            robot,
            {**mixed_extra, "action": {**payload, "action": [[True] * 59] * 16}},
            expected_error="excluding booleans",
        ),
    ]


def prediction_cases(fixture: Fixture, *, inverse: bool = False) -> list[Case]:
    mode = "inverse_dynamics" if inverse else "wam"
    cases = [
        Case(f"{mode}_default", mode, fixture),
        Case(
            f"{mode}_raw_prompt_reference",
            mode,
            fixture,
            {"action_prompt_format": "raw", "action_caption_semantics": ""},
            {"prompt": expected_plain_prompt(mode, fixture)},
        ),
        Case(f"{mode}_history", mode, fixture, {"history_action": fixture.history.tolist()}),
        Case(
            f"{mode}_history_state",
            mode,
            fixture,
            {
                "action_chunk_size": 17,
                "history_action": fixture.history.tolist(),
                "initial_state": fixture.initial_state.tolist(),
            },
        ),
        Case(
            f"{mode}_state", mode, fixture, {"action_chunk_size": 17, "initial_state": fixture.initial_state.tolist()}
        ),
        Case(f"{mode}_predict_state", mode, fixture, {"action_chunk_size": 17, "predict_initial_state": True}),
        Case(
            f"{mode}_explicit_schedule",
            mode,
            fixture,
            {"independent_action_schedule": True, "action_flow_shift": fixture.action_shift},
            {"flow_shift": fixture.vision_shift},
        ),
        Case(
            f"{mode}_joint_schedule",
            mode,
            fixture,
            {"independent_action_schedule": False},
            {"flow_shift": fixture.vision_shift},
        ),
        Case(
            f"{mode}_override_schedule",
            mode,
            fixture,
            {"independent_action_schedule": True, "action_flow_shift": 2.0},
            {"flow_shift": fixture.vision_shift},
        ),
        Case(
            f"{mode}_json_prompt",
            mode,
            fixture,
            {
                "action_prompt_format": "json",
                "action_prompt_duration": True,
                "action_idle_frames": 3,
                "action_caption_semantics": "",
            },
        ),
        Case(
            f"{mode}_json_alias",
            mode,
            fixture,
            {
                "action_prompt_format": None,
                "format_prompt_as_json": True,
                "action_prompt_duration": True,
                "action_idle_frames": 3,
                "action_caption_semantics": "",
            },
        ),
        Case(
            f"{mode}_bad_state",
            mode,
            fixture,
            {"action_chunk_size": 17},
            expected_error="initial-state row on video frame 0",
        ),
        Case(
            f"{mode}_bad_history_state",
            mode,
            fixture,
            {"action_chunk_size": 17, "predict_initial_state": True, "history_action": fixture.history.tolist()},
            expected_error="does not support history actions",
        ),
        Case(
            f"{mode}_bad_schedule",
            mode,
            fixture,
            {"independent_action_schedule": False, "action_flow_shift": 1},
            expected_error="requires independent_action_schedule",
        ),
        Case(
            f"{mode}_bad_mask",
            mode,
            fixture,
            {"action_valid_mask": [True] * 58},
            expected_error="does not match raw_action_dim",
        ),
        Case(
            f"{mode}_bad_mask_values",
            mode,
            fixture,
            {"action_valid_mask": [0.5] * 59},
            expected_error="bools or 0/1 numbers",
        ),
    ]
    if not inverse:
        cases.extend(
            [
                Case("policy_alias", "policy", fixture),
                Case("wam_inferred_frames", "wam", fixture, omit_num_frames=True),
                Case("policy_action_only", "policy_action_only", fixture, omit_num_frames=True),
                Case(
                    "policy_action_only_history",
                    "policy_action_only",
                    fixture,
                    {"history_action": fixture.history.tolist()},
                    omit_num_frames=True,
                ),
                Case(
                    "policy_action_only_explicit_schedule",
                    "policy_action_only",
                    fixture,
                    {"independent_action_schedule": True, "action_flow_shift": fixture.action_shift},
                    {"flow_shift": fixture.vision_shift},
                    omit_num_frames=True,
                ),
                Case(
                    "policy_action_only_raw_prompt_reference",
                    "policy_action_only",
                    fixture,
                    {"action_prompt_format": "raw", "action_caption_semantics": ""},
                    {"prompt": expected_plain_prompt("policy_action_only", fixture)},
                    omit_num_frames=True,
                ),
                Case(
                    "bad_policy_state",
                    "policy_action_only",
                    fixture,
                    {"initial_state": fixture.initial_state.tolist()},
                    expected_error="does not take an initial_state",
                ),
                Case(
                    "bad_policy_predict_state",
                    "policy_action_only",
                    fixture,
                    {"predict_initial_state": True},
                    expected_error="defined only for",
                ),
                Case(
                    "bad_prompt_format",
                    "wam",
                    fixture,
                    {"action_prompt_format": "yaml"},
                    expected_error="action_prompt_format must be one of",
                ),
            ]
        )
    return cases


def expected_plain_prompt(mode: str, fixture: Fixture) -> str:
    """Independent golden for the duration-free 3.5 recipe, not a renderer port."""
    if mode == "inverse_dynamics":
        text = MODE_SENTENCES[mode]
    else:
        caption = fixture.caption.strip()
        if not caption.endswith((".", "!", "?")):
            caption += "."
        text = f"{caption} {MODE_SENTENCES[mode]}"
    if fixture.caption_semantics:
        text += " " + fixture.caption_semantics.strip()
    return text + " This video is of 544x736 resolution."


def validate_action(case: Case, final: dict[str, Any]) -> np.ndarray:
    """Check public HTTP values; internal-only metadata is checked if exposed."""
    _, extra, _ = case.request()
    rows = extra["action_chunk_size"]
    action = final.get("action")
    if not isinstance(action, dict) or "data" not in action:
        raise AssertionError("Completed job has no action data.")
    data = _array(action["data"], (rows, 59), "response action")
    if action.get("shape") != [rows, 59] or action.get("raw_action_dim") != 59:
        raise AssertionError(f"Incorrect action metadata: {action}")
    if action.get("action_mode") != case.mode:
        raise AssertionError(f"Expected echoed mode {case.mode}, got {action.get('action_mode')}")
    mask = np.asarray(extra["action_valid_mask"], dtype=bool)
    if np.any(data[:, ~mask]):
        raise AssertionError("Inactive action channels are not zero.")
    payload = extra.get("action")
    structured = isinstance(payload, dict)
    if action.get("domain_id") != case.fixture.domain_id:
        raise AssertionError("Response domain_id does not match the first action domain.")
    if case.mode == "forward_dynamics":
        expected = payload["action"] if structured else payload
        np.testing.assert_allclose(data, np.asarray(expected) * mask, rtol=0, atol=1e-6)
    if "initial_state" in extra:
        np.testing.assert_allclose(data[0], np.asarray(extra["initial_state"]) * mask, rtol=0, atol=1e-6)
    if "history_length" in action:
        assert action["history_length"] == len(extra.get("history_action", []))
    if "initial_state_row" in action:
        assert action["initial_state_row"] == (rows == 17 and case.mode != "policy_action_only")
    if structured and "domain_ids" in action:
        names = payload["domain_names"]
        expected_ids = [2 if name == "camera_pose" else case.fixture.domain_id for name in names]
        if len(set(expected_ids)) > 1:
            assert action["domain_ids"] == expected_ids
    return data


def validate_video(path: Path, *, frames: int, fps: float, size: tuple[int, int]) -> None:
    """Decode every frame; detect truncated/empty MP4 and incorrect geometry/FPS."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate is not None and abs(float(stream.average_rate) - fps) < 1e-3
        count = 0
        for frame in container.decode(stream):
            assert (frame.width, frame.height) == size, "Unexpected video canvas."
            count += 1
        assert count == frames, f"Expected {frames} decoded frames; got {count}."


class ActionClient:
    def __init__(self, base_url: str, output_root: Path, *, timeout_s: float = 1800, session=None):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite.")
        self.base_url = base_url.rstrip("/")
        # Never overwrite a previous run's requests or outputs.
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    @staticmethod
    def _save(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def models(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/v1/models", timeout=30)
        response.raise_for_status()
        models = response.json()
        if not models.get("data"):
            raise RuntimeError("The endpoint returned no models.")
        self._save(self.output_root / "models.json", models)
        return models

    def run(self, case: Case) -> dict[str, Any]:
        run_dir = self.output_root / case.name
        run_dir.mkdir(exist_ok=False)
        form, extra, media = case.request()
        self._save(run_dir / "fixture.json", case.fixture.manifest())
        self._save(
            run_dir / "request.json",
            {
                "form": form,
                "extra_params": extra,
                "input_reference": str(media),
                "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            },
        )
        # Snapshot comparisons below do not depend on the model exposing internal metadata.
        started = time.monotonic()
        with media.open("rb") as file:
            response = self.session.post(
                f"{self.base_url}/v1/videos",
                data={
                    **{key: str(value) for key, value in form.items()},
                    "extra_params": json.dumps(extra, allow_nan=False),
                },
                files={"input_reference": (media.name, file, mimetypes.guess_type(media.name)[0])},
                timeout=min(120, self.timeout_s),
            )
        if response.status_code >= 400:
            final = {"status": "failed", "error": response.text, "http_status": response.status_code}
        else:
            response.raise_for_status()
            initial = response.json()
            self._save(run_dir / "initial.json", initial)
            final = initial
            while final.get("status") not in {"completed", "failed", "cancelled", "canceled"}:
                remaining = self.timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Job {initial['id']} exceeded {self.timeout_s}s; see {run_dir}")
                response = self.session.get(f"{self.base_url}/v1/videos/{initial['id']}", timeout=min(30, remaining))
                response.raise_for_status()
                final = response.json()
                self._save(run_dir / "final.json", final)
                if final.get("status") not in {"completed", "failed", "cancelled", "canceled"}:
                    time.sleep(min(2, max(0, self.timeout_s - (time.monotonic() - started))))
        self._save(run_dir / "final.json", final)
        if case.expected_error:
            if final.get("status") != "failed" or not re.search(case.expected_error, json.dumps(final), re.I):
                raise AssertionError(f"Expected rejection matching {case.expected_error!r}, got {final}")
            result = {"name": case.name, "status": "expected_rejection", "run_dir": str(run_dir)}
        else:
            if final.get("status") != "completed":
                raise AssertionError(f"Job failed: {final}")
            action = validate_action(case, final)
            np.save(run_dir / "action.npy", action)
            content = self.session.get(f"{self.base_url}/v1/videos/{final['id']}/content", timeout=300)
            content.raise_for_status()
            if case.mode == "policy_action_only":
                if content.content:
                    raise AssertionError("Action-only mode returned video bytes.")
            else:
                path = run_dir / "video.mp4"
                path.write_bytes(content.content)
                validate_video(path, frames=form.get("num_frames", 17), fps=form["fps"], size=case.fixture.size)
            result = {"name": case.name, "status": "passed", "run_dir": str(run_dir)}
        result["elapsed_s"] = time.monotonic() - started
        self._save(run_dir / "checks.json", result)
        return result

    def run_suite(self, cases: list[Case]) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            print(case.name, flush=True)
            try:
                results.append(self.run(case))
            except Exception as exc:
                results.append({"name": case.name, "status": "failed", "error": str(exc)})
            self._save(self.output_root / "summary.json", results)
        failures = [result for result in results if result["status"] == "failed"]
        if failures:
            raise AssertionError(f"{len(failures)} regression cases failed; see {self.output_root / 'summary.json'}")
        return results


def compare_actions(root: Path, first: str, second: str, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Equality check for equivalent paths, not a required difference for ablations."""
    np.testing.assert_allclose(
        np.load(root / first / "action.npy"), np.load(root / second / "action.npy"), atol=atol, rtol=rtol
    )


def decode_actions(normalized: np.ndarray, fixture: Fixture, normalizer) -> np.ndarray:
    import torch

    raw = normalizer.denormalize_action(torch.as_tensor(normalized, dtype=torch.float32)).numpy()
    raw[:, ~fixture.valid_mask] = 0
    if not np.isfinite(raw).all():
        raise AssertionError("Decoded action contains non-finite physical values.")
    return raw


def compare_reference(root: Path, reference: Path, names: list[str]) -> None:
    """Compare normalized predictions with explicitly noise-aligned reference artifacts.

    A seed alone does not establish RNG parity between backends. The producer
    must record matching checkpoint/preprocessing/sampler provenance and export
    matching initial noise before declaring inputs_and_noise_aligned=true.
    """
    metadata = json.loads((reference / "reference.json").read_text())
    for key in ("checkpoint", "backend_revision", "preprocessing", "sampler"):
        if not metadata.get(key):
            raise ValueError(f"Reference provenance is missing {key}.")
    if metadata.get("inputs_and_noise_aligned") is not True:
        raise ValueError("Cross-backend references require aligned inputs and initial noise, not just the same seed.")
    atol, rtol = float(metadata["atol"]), float(metadata["rtol"])
    if not all(np.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("Reference tolerances must be finite and non-negative.")
    for name in names:
        actual_request = json.loads((root / name / "request.json").read_text())
        expected_request = json.loads((reference / name / "request.json").read_text())
        for key in ("form", "extra_params", "media_sha256"):
            if actual_request[key] != expected_request[key]:
                raise AssertionError(f"Reference input mismatch for {name}: {key}")
        actual = np.load(root / name / "action.npy", allow_pickle=False)
        expected = np.load(reference / name / "action.npy", allow_pickle=False)
        if actual.shape != expected.shape or not np.isfinite(expected).all():
            raise AssertionError(f"Reference action shape/values are invalid for {name}.")
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)


def backend_test_command(repo: Path, python: str) -> list[str]:
    """Run in the backend's own Python environment, independently of this client."""
    tests = repo / "tests/diffusion/models/cosmos3"
    return [
        python,
        "-m",
        "pytest",
        "-q",
        *[str(tests / f"test_cosmos3_action{suffix}.py") for suffix in ("", "_prompt", "_schema", "_schedule")],
        str(tests / "test_cosmos3_transformer.py"),
        str(tests / "test_cosmos3_pipeline.py"),
        str(repo / "tests/model_extras/test_model_extras.py"),
        "-k",
        "action or root_model_config or cosmos3p5_recipe or cosmos3_extra_registry",
    ]
