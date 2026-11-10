<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3.5 action regression notebooks

These notebooks connect to an **already deployed Cosmos3.5 action model**:

- [Forward dynamics](run_fd_with_vllm_omni.ipynb): single-domain and mixed-domain
  normalized actions, history, state rows, inferred frame counts, invalid schemas.
- [Inverse dynamics](run_id_with_vllm_omni.ipynb): video-conditioned actions,
  history/state layouts, sparse masks, schedules and prompts.
- [WAM and policy](run_policy_with_vllm_omni.ipynb): joint video/action generation,
  `policy` compatibility, and `policy_action_only` without future video.

They replace the older Cosmos3-specific vLLM notebooks. The original 8D DROID
joint-policy contract and native 9D/10D/57D inputs are not the 59D Cosmos3.5
contract. The other backend notebooks remain examples of those older recipes.

## Run against an existing endpoint

Open a notebook from anywhere inside this framework checkout. In its kernel,
install the dependencies listed in the first code cell, then run all cells.
Torch is used on the CPU for the framework's normalization helpers; neither
vLLM nor model weights are loaded by the client.

Set these environment variables before starting the kernel, or set them with
`os.environ` before the configuration cell:

| Variable | Meaning |
| --- | --- |
| `COSMOS3_VLLM_BASE_URL` | Existing endpoint, default `http://localhost:8001`. |
| `COSMOS3_VLLM_API_KEY` | Optional bearer token. Never written to output artifacts. |
| `COSMOS3_VLLM_OUTPUT_ROOT` | Default `outputs/cosmos35_action_vllm` in this checkout. |
| `COSMOS3_ACTION_TIMEOUT_S` | Per-job deadline, default 1800 seconds. |
| `COSMOS3_ACTION_STATS` | Optional canonical 59D global-asinh statistics JSON matching the checkpoint. |
| `COSMOS3_ACTION_FIXTURE` | Optional recorded robot fixture JSON, described below. |
| `COSMOS3_CAMERA_FIXTURE` | Optional recorded camera fixture JSON. |

All cases run by default, including deliberate invalid requests. They run
sequentially; the runner continues after failures and raises at the end with
the location of `summary.json`. Error cases pass only if the expected
validation message is returned, synchronously or through a failed async job.
An OOM, cancellation or unrelated server error does not count as a successful
rejection.

Each execution creates a unique `<suite>/<UTC timestamp>_<id>` directory.
Each case saves `request.json`, `fixture.json`, the initial/terminal responses,
`action.npy` in normalized space, and `video.mp4` when applicable. Requests
include a media SHA-256. The run also records the model list, normalization
offset/scale, and a machine-readable summary. Previous runs are never overwritten.

## Default fixtures and checkpoint expectations

The defaults are deterministic **protocol probes**, not quality references.
Analytic backward-anchored wrist/camera motions are normalized with the existing
framework global-asinh implementation and paired with the checked-in UMI/AV
observation media. Their action values do not describe the recorded video's
ground-truth motion. Do not use these outputs to score task success, imitation
accuracy, or real robot control.

The default profile is
`global_asinh_v1_backward_anchored_gripper_metric.json`. The wrist sample uses
slots `[9:18]` and `33`; the camera sample uses `[0:9]`. Inactive slots are zero
after normalization. Gripper width is in meters. History and future poses are
anchored at the current observation, including the negative-time history rows.
The fixture has 16 future actions, two history rows and a separate state row.

The requests use a canonical 736x544 canvas, 30 denoising steps and seed 0.
They expect a checkpoint-derived video shift of 3.5 at this resolution and an
action shift of 1.0, matching the inspected Cosmos3.5 recipe. Set fixture shift
metadata to the actual training values when testing another recipe. The default
cases omit both shifts; explicit matching requests verify the automatic path.
Joint-schedule and action-shift-2 requests are ablations, not accuracy oracles.
FD keeps all input action rows conditioned, so prediction modes are necessary
to exercise noisy-action sampling.

The default prompt options are `plain`, mode sentence enabled, resolution
enabled, duration disabled. The notebook supplies an unformatted caption;
the backend renders it. Raw-prompt equivalents independently spell out the
expected plain caption and must produce matching actions. JSON formatting and
the `format_prompt_as_json` alias are compared separately. ID must discard the
target-action caption and idle-frame annotation. Exact token-level JSON/plain
goldens are also covered by the backend checks below.

## Coverage and its limits

| vLLM-Omni commit | Notebook checks | Internal checks needed |
| --- | --- | --- |
| `a4385c7ac`, root config fallback | Loaded model exercised indirectly. | Root/transformer/runtime precedence, projector type, input bias and modality embedding. |
| `fffd39beb`, independent schedules | Default vs explicit matching shifts; joint/override ablations in ID, WAM and policy. | Separate scheduler histories and action timestep embeddings. |
| `bc34d37aa`, modes | WAM, deprecated `policy` alias, action-only policy, inferred frame count. | One observation latent and no video decode for action-only requests. |
| `d75c384d4`, row layouts | Returned row count excludes history; measured state preserved; generated state; invalid combinations. | Clean conditioning rows and mRoPE start offsets. |
| `fa590af72`, mixed domains | Structured mixed and single-domain FD payloads; width/count/domain/camera-slot errors. | Per-token domain-aware projection routing. |
| `7eab07d7b`, prompts | Plain vs raw golden; JSON vs alias; invalid format. | Exact strings/order/tokens, ID caption removal and full action-only horizon. |
| `af107599f`, sparse masks | Inactive outputs zero, including clean state; invalid masks. | Noise, clean history and velocity masked at every step. |

The current `VideoAction` HTTP schema omits `history_length`, `initial_state_row`
and per-row `domain_ids`. The runner checks these if a future server exposes
them, but does not require absent fields or mistake their absence for a model
failure. Successful HTTP results alone cannot prove the internal guarantees.
Likewise, shared-weight projectors do not prove domain-aware routing even when
a mixed-domain request succeeds.

The last notebook cell can run the existing focused tests in the backend's
own environment. Set both:

```bash
export COSMOS3_VLLM_REPO=/path/to/vllm-omni
export COSMOS3_VLLM_PYTHON=/path/to/vllm-omni/.venv/bin/python
```

This runs the action layout, prompt, schema and schedule suites, action-related
pipeline/transformer tests, and model-extras tests. The command and complete
output are saved as `backend_command.json` and `backend_tests.log`. It requires
the backend checkout's test dependencies, but does not launch a model. If these
variables are unset, the notebook explicitly reports internal checks as not run.

## Recorded fixture format

Use `Fixture.manifest()` from `vllm_omni_regression.py` as the complete template;
each run's `fixture.json` is also a valid example. Required fields are:

| Field | Contract |
| --- | --- |
| `action_space` | `normalized_unified_v1`. |
| `image`, `video` | Local observation media, paths relative to the fixture JSON or absolute. |
| `actions` | `[16,59]` finite normalized future rows. |
| `history` | `[2,59]` clean past rows in the current observation's anchor frame. |
| `initial_state` | `[59]` clean normalized state row, not included in `actions`. |
| `valid_mask` | `[59]` bool/binary mask; inactive fixture values must already be zero. |
| `domain_name`, `domain_id` | Registered domain and its trained numeric ID; camera fixture must be `camera_pose` / `2`. |
| `caption`, `fps` | Original task caption and model conditioning FPS. |

Optional fields include `size` (currently `[736,544]`), `vision_shift`,
`action_shift`, `pose_convention` (currently `backward_anchored`),
`caption_semantics`, and `provenance`. Use the matching canonical normalizer,
pose transformation and gripper calibration before supplying normalized rows;
do not pad or renormalize old Cosmos3 quantile-normalized actions. Caption
semantics should match the training family. Camera fixtures must use only
ego-pose slots so the mixed-domain case remains valid.

For visualization, predictions are inverse-normalized with the same profile,
then inactive channels are zeroed again. The displayed translations are anchored
positions, not framewise deltas to cumulatively integrate. Measured-state cases
retain their state row in the returned action tensor; history rows are removed.

## Reference artifacts

Set `COSMOS3_ACTION_REFERENCE_ROOT` to a directory with `fd/`, `id/` and/or
`policy/` subdirectories. Each selected suite needs a `reference.json`:

```json
{
  "checkpoint": "exact checkpoint identity or digest",
  "backend_revision": "reference implementation commit",
  "preprocessing": "exact action, caption and media preprocessing recipe",
  "sampler": "exact UniPC configuration and shifts",
  "inputs_and_noise_aligned": true,
  "atol": 0.00001,
  "rtol": 0.00001
}
```

For every successful case, include `<case>/request.json` and
`<case>/action.npy`. The request's form, extra parameters and media digest must
match the notebook request. Arrays must match in shape and contain finite values;
comparison uses the declared tolerances in normalized action space. Record
appropriate tolerances for the actual hardware and precision rather than
treating the example values as universally applicable.

Export references from the matching Imaginaire4 implementation only after
aligning preprocessing, sampler and **initial noise tensors**. A shared seed
alone does not establish identical random draws across implementations. Without
aligned references the notebooks report protocol coverage, not numerical parity.

## Local client tests

```bash
python -m pytest tests/action_vllm_notebook_test.py -q
```

These CPU tests validate notebook syntax/setup, canonical normalization round
trips, request construction, client failure handling and actual MP4 decoding.
They use controlled HTTP responses and do not claim to execute the deployed model.
