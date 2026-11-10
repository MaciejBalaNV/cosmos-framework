# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU checks for the actual notebook client, fixtures and response invariants."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / ".agents/skills/cookbooks/cosmos3/generator/action"
spec = importlib.util.spec_from_file_location("action_notebook_regression", ACTION / "vllm_omni_regression.py")
regression = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = regression
spec.loader.exec_module(regression)

pytestmark = [pytest.mark.CPU, pytest.mark.L0]


@pytest.fixture(scope="module")
def normalizer():
    return regression.action_normalizer(ROOT)


@pytest.fixture(scope="module")
def fixtures(normalizer):
    return regression.protocol_fixtures(ACTION, normalizer)


def final_for(case):
    _, extra, _ = case.request()
    shape = (extra["action_chunk_size"], 59)
    values = np.full(shape, 0.1, dtype=np.float32)
    if case.mode == "forward_dynamics":
        payload = extra["action"]
        values = np.asarray(payload["action"] if isinstance(payload, dict) else payload, dtype=np.float32)
    if "initial_state" in extra:
        values[0] = extra["initial_state"]
    values[:, ~np.asarray(extra["action_valid_mask"], dtype=bool)] = 0
    return {
        "id": "job-1",
        "status": "completed",
        "action": {
            "data": values.tolist(),
            "shape": list(shape),
            "raw_action_dim": 59,
            "action_mode": case.mode,
            "domain_id": case.fixture.domain_id,
        },
    }


def response(payload=None, *, status=200, content=b""):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode() if payload is not None else content
    return result


class Session:
    """Controlled HTTP responses; does not emulate the model or its validation."""

    def __init__(self, final, *, body=b"", initial=None):
        self.final = final
        self.body = body
        self.initial = response({"id": "job-1", "status": "queued"}) if initial is None else initial
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append(kwargs)
        return self.initial

    def get(self, url, **kwargs):
        if url.endswith("/content"):
            return response(content=self.body)
        return response(self.final)


def test_fixture_masks_and_physical_round_trip(fixtures, normalizer):
    robot, camera = fixtures
    assert np.flatnonzero(robot.valid_mask).tolist() == list(range(9, 18)) + [33]
    assert np.flatnonzero(camera.valid_mask).tolist() == list(range(9))
    for fixture, offset in ((robot, 9), (camera, 0)):
        raw = regression.decode_actions(fixture.actions, fixture, normalizer)
        np.testing.assert_allclose(raw[:, offset + 2], np.arange(1, 17) * 0.002, atol=1e-7)
        np.testing.assert_allclose(raw[:, offset + 3 : offset + 9], np.tile([1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-7)
        assert not raw[:, ~fixture.valid_mask].any()
        history = regression.decode_actions(fixture.history, fixture, normalizer)
        np.testing.assert_allclose(history[:, offset + 2], [-0.004, -0.002], atol=1e-7)
    np.testing.assert_allclose(regression.decode_actions(robot.actions, robot, normalizer)[:, 33], 0.04, atol=1e-7)


def test_recorded_fixture_round_trip_and_path_resolution(fixtures, tmp_path):
    robot, _ = fixtures
    payload = robot.manifest()
    payload["image"] = "observation.png"
    payload["video"] = "observation.mp4"
    (tmp_path / payload["image"]).touch()
    (tmp_path / payload["video"]).touch()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload))
    loaded = regression.load_fixture(path)
    assert loaded.image == tmp_path / "observation.png"
    np.testing.assert_array_equal(loaded.actions, robot.actions)
    payload["actions"][0][0] = 1  # inactive camera slot in a robot-only sample.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inactive channels"):
        regression.load_fixture(path)


def test_every_positive_case_checks_the_public_action_contract(fixtures):
    robot, camera = fixtures
    cases = regression.forward_cases(robot, camera) + regression.prediction_cases(robot)
    cases += regression.prediction_cases(camera, inverse=True)
    assert len({case.name for case in cases}) == len(cases)
    for case in cases:
        if case.expected_error:
            continue
        form, extra, media = case.request()
        assert "flow_shift" not in form or "schedule" in case.name
        assert "action_flow_shift" not in extra or "schedule" in case.name
        assert media == (case.fixture.video if case.mode == "inverse_dynamics" else case.fixture.image)
        assert ("num_frames" not in form) == case.omit_num_frames
        regression.validate_action(case, final_for(case))
    # Mixed rows cannot accidentally inherit the single-domain fixture's sparse mask.
    mixed = next(case for case in cases if case.name == "fd_mixed")
    _, extra, _ = mixed.request()
    assert "domain_name" not in extra
    assert extra["action_valid_mask"] == (robot.valid_mask | camera.valid_mask).tolist()
    assert extra["action"]["domain_names"] == ["umi", "camera_pose"] * 8
    assert extra["action"]["action"][1][9:] == [0.0] * 50


def test_recorded_caption_semantics_are_rendered_once(fixtures):
    robot, _ = fixtures
    robot = replace(robot, caption_semantics="This action describes the right wrist and gripper width.")
    cases = regression.prediction_cases(robot)
    for case in cases:
        if case.expected_error:
            continue
        form, extra, _ = case.request()
        if extra.get("action_prompt_format") in {"raw", "json"} or extra.get("format_prompt_as_json"):
            assert extra["action_caption_semantics"] == ""
        else:
            assert extra["action_caption_semantics"] == robot.caption_semantics
        if extra.get("action_prompt_format") == "raw":
            assert form["prompt"].count(robot.caption_semantics) == 1


@pytest.mark.parametrize(
    "broken", ["shape", "width", "history_leak", "mask", "nan", "mode", "domain", "fd_value", "state_value"]
)
def test_invalid_model_outputs_fail(fixtures, broken):
    robot, _ = fixtures
    case = regression.Case("probe", "wam", robot)
    if broken == "fd_value":
        case.mode = "forward_dynamics"
    if broken == "state_value":
        case.extra = {"action_chunk_size": 17, "initial_state": robot.initial_state.tolist()}
    final = final_for(case)
    action = final["action"]
    if broken == "shape":
        action["shape"] = [1, 16, 59]
    elif broken == "width":
        action["raw_action_dim"] = 8
    elif broken == "history_leak":
        action["data"] += action["data"][:2]
    elif broken == "mask":
        action["data"][0][0] = 1
    elif broken == "nan":
        action["data"][0][9] = float("nan")
    elif broken == "mode":
        action["action_mode"] = "policy"
    elif broken == "domain":
        action["domain_id"] = 7
    elif broken in {"fd_value", "state_value"}:
        action["data"][0][9] += 1
    with pytest.raises((AssertionError, ValueError)):
        regression.validate_action(case, final)


def make_video(*, frames=17, fps=20, size=(32, 32)):
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(np.zeros((size[1], size[0], 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize("broken", [None, "frames", "fps", "size", "empty"])
def test_video_checks_decode_actual_bytes(tmp_path, broken):
    path = tmp_path / "video.mp4"
    path.write_bytes(
        b""
        if broken == "empty"
        else make_video(
            frames=16 if broken == "frames" else 17,
            fps=15 if broken == "fps" else 20,
            size=(48, 32) if broken == "size" else (32, 32),
        )
    )
    if broken:
        with pytest.raises((AssertionError, av.error.InvalidDataError)):
            regression.validate_video(path, frames=17, fps=20, size=(32, 32))
    else:
        regression.validate_video(path, frames=17, fps=20, size=(32, 32))


def test_async_action_only_persists_the_actual_request(fixtures, tmp_path):
    robot, _ = fixtures
    case = regression.Case("only", "policy_action_only", robot, omit_num_frames=True)
    session = Session(final_for(case))
    client = regression.ActionClient("http://server/", tmp_path / "run", session=session)
    result = client.run(case)
    assert result["status"] == "passed"
    request = json.loads((client.output_root / "only/request.json").read_text())
    assert "num_frames" not in request["form"]
    assert "flow_shift" not in request["form"]
    assert len(request["media_sha256"]) == 64
    assert json.loads(session.posts[0]["data"]["extra_params"]) == request["extra_params"]
    assert (client.output_root / "only/action.npy").exists()
    assert not (client.output_root / "only/video.mp4").exists()
    with pytest.raises(FileExistsError):
        regression.ActionClient("http://server", client.output_root)


def test_action_only_rejects_unexpected_video(fixtures, tmp_path):
    robot, _ = fixtures
    case = regression.Case("only", "policy_action_only", robot)
    client = regression.ActionClient("http://server", tmp_path / "run", session=Session(final_for(case), body=b"mp4"))
    with pytest.raises(AssertionError, match="returned video"):
        client.run(case)


def test_async_wam_downloads_and_decodes_video(fixtures, tmp_path):
    robot, _ = fixtures
    case = regression.Case("wam", "wam", robot)
    body = make_video(size=robot.size, fps=int(robot.fps))
    client = regression.ActionClient("http://server", tmp_path / "run", session=Session(final_for(case), body=body))
    assert client.run(case)["status"] == "passed"
    assert (client.output_root / "wam/video.mp4").read_bytes() == body


@pytest.mark.parametrize("synchronous", [True, False])
@pytest.mark.parametrize("matching", [True, False])
def test_rejections_require_the_intended_validation_error(fixtures, tmp_path, synchronous, matching):
    robot, _ = fixtures
    case = regression.Case("bad", "wam", robot, expected_error="initial-state row")
    message = "initial-state row on video frame 0" if matching else "out of memory"
    initial = response({"error": message}, status=400) if synchronous else None
    session = Session({"status": "failed", "error": {"message": message}}, initial=initial)
    client = regression.ActionClient("http://server", tmp_path / "run", session=session)
    if matching:
        assert client.run(case)["status"] == "expected_rejection"
    else:
        with pytest.raises(AssertionError, match="Expected rejection"):
            client.run(case)


def test_async_deadline_stops_polling(fixtures, tmp_path, monkeypatch):
    robot, _ = fixtures
    session = Session({"id": "job-1", "status": "in_progress"})
    client = regression.ActionClient("http://server", tmp_path / "run", session=session, timeout_s=1)
    clock = iter([0, 2])
    monkeypatch.setattr(regression.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="exceeded"):
        client.run(regression.Case("timeout", "wam", robot))


def test_suite_saves_failures_and_continues(fixtures, tmp_path):
    robot, _ = fixtures
    case = regression.Case("bad", "policy_action_only", robot)
    bad = final_for(case)
    bad["action"]["data"][0][0] = 1
    client = regression.ActionClient("http://server", tmp_path / "run", session=Session(bad))
    with pytest.raises(AssertionError, match="2 regression cases failed"):
        client.run_suite([case, regression.Case("also_bad", "policy_action_only", robot)])
    results = json.loads((client.output_root / "summary.json").read_text())
    assert [result["name"] for result in results] == ["bad", "also_bad"]


def test_reference_requires_aligned_inputs_noise_and_provenance(tmp_path):
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    reference.mkdir()
    metadata = {
        "checkpoint": "weights-sha",
        "backend_revision": "commit",
        "preprocessing": "recipe",
        "sampler": "UniPC",
        "atol": 1e-5,
        "rtol": 1e-5,
        "inputs_and_noise_aligned": False,
    }
    (reference / "reference.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="aligned inputs and initial noise"):
        regression.compare_reference(actual, reference, ["case"])
    metadata["inputs_and_noise_aligned"] = True
    (reference / "reference.json").write_text(json.dumps(metadata))
    request = {"form": {"seed": 0}, "extra_params": {}, "media_sha256": "abc"}
    for root in (actual, reference):
        (root / "case").mkdir(parents=True)
        (root / "case/request.json").write_text(json.dumps(request))
        np.save(root / "case/action.npy", np.zeros((16, 59)))
    regression.compare_reference(actual, reference, ["case"])
    request["extra_params"]["action_flow_shift"] = 2
    (reference / "case/request.json").write_text(json.dumps(request))
    with pytest.raises(AssertionError, match="input mismatch"):
        regression.compare_reference(actual, reference, ["case"])


@pytest.mark.parametrize("suite", ["fd", "id", "policy"])
def test_notebooks_are_valid_and_setup_runs_from_root_and_notebook_directory(suite, fixtures, tmp_path, monkeypatch):
    path = ACTION / f"run_{suite}_with_vllm_omni.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        if not cell.startswith("%pip"):
            compile(cell, str(path), "exec")
    setup = next(cell for cell in cells if "FRAMEWORK_ROOT =" in cell)
    for start in (ROOT, ACTION):
        monkeypatch.chdir(start)
        monkeypatch.setenv("COSMOS3_VLLM_OUTPUT_ROOT", str(tmp_path))
        # Execute the actual bootstrap while replacing only the external readiness request.
        monkeypatch.setattr(requests.Session, "get", lambda *args, **kwargs: response({"data": [{"id": "Cosmos3.5"}]}))
        scope = {}
        exec(setup, scope)
        assert scope["FRAMEWORK_ROOT"] == ROOT
        assert scope["client"].output_root.is_dir()
        assert scope["robot"].actions.shape == (16, 59)


def test_backend_command_covers_internal_guarantees():
    command = regression.backend_test_command(Path("/backend"), "/backend/.venv/bin/python")
    assert command[0] == "/backend/.venv/bin/python"
    for suffix in (
        "action.py",
        "action_schedule.py",
        "action_prompt.py",
        "action_schema.py",
        "transformer.py",
        "pipeline.py",
    ):
        assert any(item.endswith("test_cosmos3_" + suffix) for item in command)
    assert any(item.endswith("test_model_extras.py") for item in command)
    assert "cosmos3_extra_registry" in command[-1]
