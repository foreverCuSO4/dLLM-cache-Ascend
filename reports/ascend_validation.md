# Ascend 910B migration validation

Validation date: 2026-07-14.  This is an inference-only validation; the
project does not contain a training migration.

## Environment and reproducibility

- Host: 8 × Ascend 910B4, 32 GiB HBM per device; devices 0–7 were visible.
- Driver: 24.1.0.3; CANN: 8.0.1; Python: 3.10.20.
- PyTorch: 2.4.0; torch-npu: 2.4.0.post2; Transformers: 4.51.3;
  Accelerate: 1.6.0; lm-eval: 0.4.12.
- Model revisions:
  - `GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`
  - `Dream-org/Dream-v0-Instruct-7B@05334cb9faaf763692dcf9d8737c642be2b2a6ae`
- The downloaded model shards were checked against the fixed Hugging Face
  revisions; the snapshot blob SHA-256 names and revision manifest are in
  `configs/model_revisions.yaml` and the Hugging Face cache.
- All commands were run after sourcing
  `/usr/local/Ascend/ascend-toolkit/set_env.sh` with
  `PYTHONNOUSERSITE=1`.

## Environment gate

`scripts/preflight_ascend.py --min-devices 8 --run-hccl` passed:

- BF16 SDPA, matmul, cosine similarity, top-k, gather and scatter passed on
  all eight devices.
- 8-rank HCCL all-reduce passed (`sum(1..8)=36`).
- `pip check` passed in the isolated environment.

The repository regression suite passed **34 tests**.  `compileall`, all shell
syntax checks and `git diff --check` also passed.

## Correctness gates

| Gate | Result | Evidence |
| --- | --- | --- |
| LLaDA baseline generation | PASS | `reports/llada_npu0_baseline_smoke.json` |
| Dream baseline generation | PASS | `reports/dream_npu0_baseline_smoke.json` |
| Full-refresh token equality | PASS | LLaDA and Dream benchmark/smoke output hashes |
| Left-padded batch and mask paths | PASS | adapter and hook regression tests; remote-code batch coverage |
| Adaptive cache branches | PASS | LLaDA/Dream single-NPU smoke and representative 8-NPU runs |
| Long-run memory-growth gate | NOT RUN | requires a separate multi-hour soak |

The new lm-eval distributed primitives (`all_gather`, `gather_object` and
`barrier`) are implemented by both model adapters.  HCCL does not expose a
native gather on this stack; torch-npu transparently uses all-gather, which
worked for the evaluation aggregation.

## Quality gates

The formal full MMLU-generative set has 14,042 samples.  It was not run in
full because one 5-shot, 256-token/256-step sample takes about 142 seconds on
this host.  Instead, the acceptance run used four different MMLU groups and
8 samples per group (32 globally), with the original Dream Instruct settings:
`max_new_tokens=256`, `diffusion_steps=256`, `temperature=0.2`, `top_p=0.95`,
5-shot, batch size 1.

| Model / run | Samples | Exact match |
| --- | ---: | ---: |
| Dream baseline (abstract algebra, business ethics, high-school government, professional law) | 23/32 | 71.875% |
| Dream adaptive (`prompt=100, gen=8, transfer=0.25`) | 24/32 | 75.000% |

The baseline and adaptive outputs contain exactly four task files × eight
rows, with matching task hashes and all 32 `(doc_id, doc_hash, target_hash)`
identities.  The adaptive delta is **+3.125 percentage points**, so the
representative quality gate (no more than 1 pp below baseline) passes.

Earlier 57-subject, `limit=1` smoke results are retained for diagnostics only:
LLaDA 70.18% → 71.93%; Dream's 3-step smoke 78.95% → 73.68% is not a formal
quality conclusion.

## Performance

Pure-generation benchmark (prompt length 128, generation 128, 128 steps,
warmup 1, repeats 3, NPU 0):

| Model | Baseline | Adaptive | Relative speed |
| --- | ---: | ---: | ---: |
| LLaDA | 15.79 token/s | 10.17 token/s | 0.644× |
| Dream | 18.49 token/s | 12.12 token/s | 0.656× |

Peak allocated HBM in those runs was about 16.1 GiB (LLaDA) and 15.3 GiB
(Dream).  These measurements exclude model loading and tokenization and are
not acceptance gates; the current cache configuration does not accelerate
this short-prompt workload.

For the representative 8-NPU MMLU run, all eight ranks loaded a full model
and processed four samples each (data parallelism, not model parallelism).
The baseline and adaptive generation phases took approximately 617 s and
577 s respectively; the adaptive wall-clock result is workload-specific and
does not contradict the short-prompt benchmark above.

## Scope and limitations

- Supported: LLaDA and Dream text inference, single-NPU use and 8-NPU data
  parallel lm-eval evaluation, with CPU/CUDA paths retained.
- Out of scope: training, LLaDA-V/MMaDA multimodal models, cross-device model
  parallelism and custom Ascend operators.
- Batch size 1 is the safe 8B configuration on this 32 GiB device.  A full
  14,042-sample MMLU run and a multi-hour memory soak should be scheduled as
  separate capacity tests.
- `fusion_result.json` is an existing user artifact.  It was not edited by
  this migration; its current four-byte file is left in place.

