# Ascend migration validation report

## Reproducibility

- Repository commit / dirty state:
- Model ID and revision SHA:
- Weight hash verification:
- Dataset, configuration, and split:
- Command line:
- Seed / dtype / batch size:
- Prompt length / generation length / diffusion steps:
- Cache mode and `prompt/gen/cfg/transfer` parameters:

## Environment

- Server and NPU model/count/topology:
- Driver / firmware / CANN:
- Python / PyTorch / torch-npu:
- Transformers / Accelerate / lm-eval:
- Relevant environment variables:
- `pip check` result:
- Preflight and HCCL result:

## Correctness gates

| Gate | Model | Cases | Result | Evidence |
| --- | --- | ---: | --- | --- |
| Baseline generation | | | | |
| Full-refresh logits/top-1 | | | | |
| Full-refresh token equality | | | | |
| Padded batch equality | | | | |
| Adaptive cache branches | | | | |
| Stability / memory growth | | | | |

## Quality

| Model | Task | Seed | Baseline | Adaptive | Delta (pp) | Pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| | | | | | | |

Acceptance: every representative adaptive result is no more than 1.0
percentage point below its matching Ascend baseline.

## Performance

| Model/mode | Workload | Warmup/repeats | p50 | p90 | samples/s | tokens/s | Peak HBM | Speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | | | | | | | |

Record compile time separately.  State whether model loading, tokenization,
and decoding are included.  All asynchronous backends must be synchronized at
the timing boundaries.

## Eight-NPU evaluation

- Rank-to-device map:
- Expected / observed sample count:
- Missing or duplicate sample IDs:
- Single-NPU / eight-NPU aggregate metric:
- Single-NPU / eight-NPU throughput:
- Rank timing imbalance:
- Peak HBM per rank:
- Exit codes and aggregation result:

## Final decision

- Environment gate: PASS / FAIL
- Correctness gate: PASS / FAIL
- Quality gate: PASS / FAIL
- Eight-NPU gate: PASS / FAIL
- Observed performance (not a fixed acceptance gate):
- Fallbacks used:
- Known limitations and follow-up work:
