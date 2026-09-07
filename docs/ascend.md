# Ascend 910B inference guide

## Supported scope

This port covers inference and `lm-eval` evaluation for LLaDA 8B and Dream
7B, with and without dLLM-Cache.  It keeps the original cache refresh and
token-selection algorithm.  CUDA and CPU remain valid backends.

The following are deliberately out of scope: model training, LLaDA-V, MMaDA,
cross-device model parallelism, and custom Ascend kernels.

## Validated platform

The pinned environment targets:

| Component | Version |
| --- | --- |
| Ascend | 8 x 910B4, 32 GiB HBM per device |
| Driver | 24.1.0.3 |
| CANN | 8.0.1 |
| Python | 3.10.20 |
| PyTorch | 2.4.0 |
| torch-npu | 2.4.0.post2 |
| lm-eval | 0.4.12 (`longbench`, `math`) |

Do not use the repository's original `install.sh` for Ascend: it installs an
unconstrained PyTorch package and does not install `torch-npu`.

## Install and preflight

Source the CANN environment before installing or running the project:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/install_ascend.sh
conda activate dllm-cache-ascend
python scripts/preflight_ascend.py --run-hccl
```

The installer creates a clean environment, disables Python user-site package
leakage, and runs `pip check`.  It does not modify the system CANN
installation.  If the official torch-npu wheel is not
available from the configured package index, set `TORCH_NPU_WHEEL` to the
approved local aarch64 wheel before invoking the installer.

`requirements-ascend.lock` pins the validated top-level packages, but is not
a hash-locked transitive dependency closure.  Keep the package index fixed
for repeatable installation and record `python -m pip freeze` in the
validation report.

By default installation validates all eight devices and runs HCCL.  On a
shared server, restrict visibility and set `PREFLIGHT_MIN_DEVICES` to the
number of devices allocated to this job; set `PREFLIGHT_RUN_HCCL=0` only when
the allocation contains a single device.

The preflight must report all eight NPUs and pass BF16 matrix multiplication,
SDPA, cosine similarity, `topk`, `gather`, and `scatter` checks.  A failure is
a platform/environment failure and must be resolved before loading a model.

## Download fixed model revisions

Model revisions are recorded in `configs/model_revisions.yaml`.  Downloads
use the modern `hf` CLI and a shared cache:

```bash
export HF_HOME=/mnt/nvme0/zhujiayi/.cache/huggingface
bash scripts/download_models_ascend.sh
```

Run this once before starting eight evaluation ranks.  Every model,
tokenizer, and config load receives the same commit SHA; the evaluation must
not silently follow a model repository's `main` branch.

## Single-NPU demos

Start with batch size one and one visible device:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python demo_LLaDA.py --device npu:0 --dtype bfloat16
python demo_Dream.py --device npu:0 --dtype bfloat16
```

The interactive `<use_cache>` and `<no_cache>` commands retain their original
meaning.  Explicitly requesting an unavailable backend fails; it never falls
back silently to CPU.

## Reproducible cache benchmark

The benchmark excludes model loading, tokenization, and decoding from pure
generation latency.  It synchronizes the backend at timing boundaries and
writes machine-readable JSON:

```bash
python benchmark_text.py \
  --model llada \
  --mode both \
  --device npu:0 \
  --dtype bfloat16 \
  --prompt "Explain diffusion language models briefly." \
  --gen-length 128 \
  --steps 128 \
  --warmup 5 \
  --repeats 20 \
  --output-json artifacts/ascend/llada-benchmark.json
```

`baseline` does not register a cache hook.  `full-refresh` uses intervals
`1/1` and transfer ratio `0`.  `adaptive` uses the requested/original cache
parameters.  No fixed speedup is an acceptance requirement; measured speedup
must still be reported.

For the four NPU anchor modes used by the performance investigation, keep
`--prompt-length 128 --gen-length 128 --steps 128 --batch-size 1
--dtype bfloat16 --warmup 10 --repeats 20` fixed. Run `baseline` and
`full-refresh` once each, then run `adaptive` twice with transfer ratios `0`
and `0.25`. Dream uses `--gen-interval-steps 8`; LLaDA uses `7`. Store every
mode in a separate JSON file so that latency percentiles, output hashes, peak
memory, and the effective cache configuration remain auditable.

## Three-step NPU profiling

`profile_npu.py` captures only the requested diffusion-step window. The
following command records steps 10--12 of the Dream selective-transfer path:

```bash
python scripts/profile_npu.py \
  --model dream \
  --mode adaptive \
  --device npu:1 \
  --dtype bfloat16 \
  --prompt-length 128 \
  --gen-length 128 \
  --steps 128 \
  --profile-start-step 10 \
  --profile-steps 3 \
  --prompt-interval-steps 100 \
  --gen-interval-steps 8 \
  --transfer-ratio 0.25 \
  --output-dir reports/profile_dream_adaptive_ratio025_npu
```

Repeat it for `baseline`, `full-refresh`, and adaptive ratio `0`. Some CANN
versions can emit host/enqueue records while leaving device duration at zero.
In that case, operator call counts and host scheduling remain useful, but host
duration must not be presented as NPU kernel duration. The completed Ascend
experiment and this limitation are summarized in
[`reports/npu_perf_root_cause.md`](../reports/npu_perf_root_cause.md).

## NPU operator microbenchmark

To separate cache bookkeeping from model execution, run the model-shaped
operator benchmark. It does not load model weights and therefore needs only a
single free NPU. The protocol enforces at least 100 warmup and 200 measured
samples, synchronizing before and after every sample:

```bash
python scripts/microbench_npu.py \
  --model dream \
  --device npu:1 \
  --dtype bfloat16 \
  --prompt-length 128 \
  --gen-length 128 \
  --transfer-ratio 0.25 \
  --warmup 100 \
  --repeats 200 \
  --output-json reports/microbench_dream_npu.json
```

Run again with `--model llada` for the LLaDA shape. The JSON report contains
p50/p90/p99 latency in microseconds per operation, current/peak HBM allocator
usage, temporary workspace deltas, output hashes, and exact tensor shapes. It
includes full and partial SDPA, cosine/top-k selection, hidden/KV
gather-scatter, concatenation, contiguous conversion, and selected/full linear
projections. These measurements are diagnostic only; final end-to-end
throughput must come from `benchmark_text.py` with model weights loaded.

## Eight-NPU evaluation

Eight-NPU execution is data parallel: each rank owns a full model and an
independent cache.  It increases evaluation throughput, not one-request
latency.

```bash
accelerate launch \
  --config_file configs/accelerate_npu_8.yaml \
  evaluation_script.py \
  --model LLaDA \
  --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,revision=0f2787f2d87eac5eed8a087d5ecd24277e6255b2,dtype=bfloat16,is_feature_cache=True,prompt_interval_steps=100,gen_interval_steps=6,transfer_ratio=0.25" \
  --tasks gsm8k \
  --batch_size 1 \
  --output_path artifacts/ascend/llada-gsm8k-cache \
  --log_samples
```

Use separate output directories for baseline and cache runs.  Dream uses
model name `dream`.  Cache parameters are checkpoint/task specific: the
Instruct scripts use `25/2/0.25` for GSM8K and `100/8/0.25` for
MMLU-generative; the Base scripts use `100/8/0.25` for GSM8K and
`100/2/0.25` for MMLU-generative.  Record the selected checkpoint and
parameters together rather than treating them as model-wide defaults.

For a single-process `lm-eval` smoke test, pass the device through the
harness CLI, for example `--device npu:0`.  Do not also put `device=...` in
`--model_args`: lm-eval 0.4.12 supplies that constructor argument itself, and
duplicating it raises a Python keyword-argument error.  Multi-process launches
use the rank device assigned by Accelerate.

When using `--limit` with eight ranks, select at least eight documents per
leaf task.  For grouped tasks such as `mmlu_generative`, `--limit 1` leaves
seven ranks with no document for each subject and lm-eval aborts by design;
use `--limit 8` or larger for an eight-rank smoke test.

## Acceptance sequence

Run gates in this order:

1. Environment and core NPU operators.
2. Single-NPU baseline generation.
3. Full-refresh hook equivalence.
4. Adaptive cache generation.
5. GSM8K and MMLU-generative baseline/cache comparison.
6. Eight-NPU sample completeness, aggregation, and throughput.

Adaptive task accuracy may decrease by at most one percentage point relative
to the corresponding Ascend baseline.  Do not compensate for a failed gate
by changing prompt length, generation length, diffusion steps, or the paper's
cache parameters.

## Troubleshooting

- `No module named tbe`: source CANN's `set_env.sh` in the same shell.
- CPU selected unexpectedly: import `torch_npu`, run the preflight, then pass
  `--device npu:0` so that an unavailable NPU becomes an explicit error.
- Out of memory: reduce batch size to one.  Do not shorten formal evaluation
  sequences.  Batch-one OOM requires a separate model-parallel project.
- Multi-rank hang: verify HCCL with the preflight and unique master port, then
  fall back to independent single-NPU task shards while diagnosing HCCL.
- Cache and baseline disagree under full refresh: stop before adaptive-cache
  testing and inspect the hook, attention mask, model revision, and dtype.

Use `reports/ascend_validation_template.md` for the final acceptance record.
