# dLLM-Cache Ascend 性能根因报告

## 结论

在 Ascend 910B4、batch=1、P=128、G=128、steps=128 下，dLLM-Cache 的
`transfer_ratio=0.25` 不是算力受限，而是小算子与动态索引受限。它减少了
部分 token 的矩阵计算，却引入大量 cosine、top-k、gather、scatter、cat、
contiguous 和小矩阵投影；省下的 SDPA 时间不足以覆盖这些开销。

- Dream 的 ratio=0.25 比 native baseline 慢 50.4%，仅为 `0.665x`；LLaDA
  慢 50.6%，仅为 `0.664x`。
- 相同缓存间隔下，ratio=0 的 Dream/LLaDA 分别达到 `3.058x` 和 `2.806x`。
  这证明缓存复用本身有效；真正昂贵的是 selective-transfer 路径。
- ratio=0.25 相对 ratio=0，Dream/LLaDA 分别慢 `4.599x` 和 `4.227x`。
- full-refresh 显示 Hook 的基础代价具有模型差异：Dream 仅慢 1.1%，LLaDA
  慢 17.5%。它是 LLaDA 的次要问题，但仍不足以解释 ratio=0.25 的整体退化。

因此，下一步不应先调大 `gen_interval`。应先压缩 selective-transfer 的算子
数量和动态内存操作，再做 prompt/gen/batch sweep。CUDA 未在本机执行，本报告
不声称 CUDA 一定不会出现同样问题。

## 实验协议

| 项目 | 固定值 |
| --- | --- |
| 硬件 | Ascend 910B4，单卡执行，32 GiB HBM |
| 精度 | BF16 |
| batch | 1 |
| prompt / generation | 128 / 128 tokens |
| diffusion steps | 128 |
| warmup / repeats | 10 / 20 |
| Dream adaptive 间隔 | prompt=100，gen=8，cfg=1 |
| LLaDA adaptive 间隔 | prompt=100，gen=7，cfg=1 |
| 计时边界 | 每次样本前后同步 NPU |

`baseline` 是未注册 Hook 的原生模型；`full-refresh` 注册 Hook，但每步完整刷新
（1/1/0）；两个 adaptive 锚点使用相同间隔，只改变 transfer ratio。模型 revision、
模式配置、原始样本、token IDs 和完整内存统计均保存在 JSON 中。

## 四模式结果

### Dream-v0-Instruct-7B

| 模式 | 有效配置 prompt/gen/ratio | mean (s) | p50 (s) | p90 (s) | tok/s | 相对 baseline | 峰值 HBM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native baseline | 1/1/0 | 6.954 | 6.953 | 6.967 | 18.406 | 1.000x | 14.411 |
| full-refresh | 1/1/0 | 7.031 | 7.022 | 7.074 | 18.206 | 0.989x | 14.530 |
| adaptive ratio=0 | 100/8/0 | 2.274 | 2.274 | 2.291 | 56.282 | 3.058x | 14.580 |
| adaptive ratio=0.25 | 100/8/0.25 | 10.460 | 10.493 | 10.585 | 12.237 | 0.665x | 14.580 |

输出 SHA-256：

- baseline：`1cedc20967034b9972f9e3ee040e90d7d7af94f6b27fd767988692ab5d34c6e2`
- full-refresh：`1cedc20967034b9972f9e3ee040e90d7d7af94f6b27fd767988692ab5d34c6e2`
- adaptive ratio=0：`3285770ca43c98916d489786ff76479b5a8d92ec804e7d41b153134ef1513089`
- adaptive ratio=0.25：`d0954fc3353ad416b1ccd182f3154d3059a956629a76f96341e0906f879d737a`

baseline 与 full-refresh 完全一致，证明完整刷新 Hook 没有改变该样本输出。两个
adaptive hash 都与 baseline 不同，因此其加速或降速不能直接解释为等质量结果，
仍需正式任务准确率验证。

### LLaDA-8B-Instruct

| 模式 | 有效配置 prompt/gen/ratio | mean (s) | p50 (s) | p90 (s) | tok/s | 相对 baseline | 峰值 HBM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native baseline | 1/1/0 | 9.786 | 9.778 | 9.904 | 13.080 | 1.000x | 15.120 |
| full-refresh | 1/1/0 | 11.498 | 11.554 | 11.620 | 11.133 | 0.851x | 15.368 |
| adaptive ratio=0 | 100/7/0 | 3.487 | 3.491 | 3.523 | 36.707 | 2.806x | 15.610 |
| adaptive ratio=0.25 | 100/7/0.25 | 14.740 | 14.837 | 15.018 | 8.684 | 0.664x | 15.610 |

四个模式的输出 SHA-256 均为：
`fa452b40791db0a69e658eac3163181bb373496a4bf604a98dd3dc342c86ff7a`。
这只证明当前固定 prompt 的 token 输出相同，不能代替数据集级质量评估。

LLaDA 表使用独占重跑的 `*_clean.json`。目录中未带 `_clean` 的早期 baseline 和
full-refresh 结果曾与其他 NPU 任务并发，不用于上述相对性能计算。

## 三步 NPU profiler

Dream 四模式均捕获 diffusion step 10--12。以下调用数来自同一个三步窗口，最能
区分 native、纯复用和 selective-transfer 三条路径：

| 调用类别 | baseline | adaptive ratio=0 | adaptive ratio=0.25 |
| --- | ---: | ---: | ---: |
| Enqueue | 3831 | 591 | 5775 |
| Linear | 591 | 24 | 591 |
| SDPA | 84 | 3 | 84 |
| Cat | 174 | 174 | 579 |
| Cosine | 0 | 0 | 81 |
| TopK | 3 | 3 | 84 |
| Gather | 0 | 0 | 162 |
| Scatter | 0 | 0 | 243 |

profiler 下每步 Host scope 均值约为 565 ms、92 ms、862 ms；其比例与关闭
profiler 后 ratio=0.25 相对 baseline 慢约 50% 的方向一致。这里不能把绝对
Host 时间当作 kernel 时间。

当前 CANN 解析结果有一个重要限制：`operator_details.csv` 的 Device Duration
均为 0，并出现 `Failed to get acl to npu flow events`。所以本报告只用 trace
证明调用数量、Host 调度和 enqueue 激增，不用它声称单个 NPU kernel 的绝对耗时。
四份原始 trace 和 metadata 均保留在 `reports/profile_dream_*_npu/`。

## 算子 microbenchmark

microbenchmark 使用与 Hook 一致的模型 shape、BF16、NPU1；每项 warmup=100、
repeats=200，并在每个样本前后同步。下表为均值，单位为微秒；JSON 同时保存
p50/p90/p99、shape、输出 hash 和显存/workspace。

| 算子 | Dream (us) | LLaDA (us) |
| --- | ---: | ---: |
| cosine similarity | 265.0 | 249.0 |
| top-k | 77.4 | 69.4 |
| hidden gather | 382.9 | 414.5 |
| KV scatter | 149.0 | 213.9 |
| hidden scatter | 205.3 | 214.2 |
| KV cat | 64.0 | 57.9 |
| hidden cat | 68.4 | 57.7 |
| contiguous | 74.5 | 69.3 |
| SDPA full，Q=256 | 141.9 | 128.0 |
| SDPA partial，Q=32 | 123.7 | 116.5 |
| Q small Linear | 77.9 | 76.2 |
| K small Linear | 67.0 | 73.5 |
| V full Linear | 74.6 | 83.5 |

把 query 从 256 减到 32，单次 SDPA 只节省约 18 us（Dream）或 12 us
（LLaDA）。仅 cosine、top-k、gather、两次 scatter 和两次 cat 的均值之和就约
1.21 ms（Dream）和 1.23 ms（LLaDA），还未包含 contiguous、小 Linear、mask、
Python/dispatcher 和 enqueue 开销。因此 FLOPs 下降但端到端时延上升并不矛盾。

## 代码级根因

多数非刷新 step 中，ratio>0 的每层仍会执行以下工作：

1. generation token 的 layer norm 和完整 V projection；
2. cosine similarity、top-k 和 hidden gather；
3. 选中 token 的 Q/K 小矩阵 projection；
4. KV/hidden cache scatter 与多次 cache cat；
5. partial attention、MLP gather/scatter、动态 q-index/mask/contiguous。

同时，第 0 层每个 step 仍强制 full refresh。结果是大矩阵被拆成大量小矩阵和
不规则访存，Linear/SDPA 的调用数没有随 transfer ratio 明显下降，反而增加了
动态算子和 enqueue。在当前 batch=1、小序列 shape 下，Ascend 很难用大核吞吐
摊薄这些固定成本。

## 优化优先级

1. **先做 NPU 专用 selective-transfer 路径**：融合 cosine+top-k+索引生成，
   预分配 cache buffer，用静态 shape/index_copy 类更新替代 scatter+cat 链。
2. **减少每层 launch**：复用 mask/index，移除重复 contiguous 和 Host 侧分支；
   评估将 Q/K/V 选择、cache 更新融合为自定义算子。
3. **再评估 attention 融合**：当前 partial SDPA 节省很小，单独融合 attention
   不是第一优先级；只有移除选择/搬运开销后才可能成为主瓶颈。
4. **最后 sweep**：在优化后的 NPU 路径上做 prompt/gen/batch sweep，再与完全
   相同的 CUDA 矩阵寻找 break-even。提前调 interval 会把根因和算法精度混在一起。

## 边界与未完成项

- 本轮没有可用 CUDA，未执行 CUDA 四模式、跨平台 sweep 或 break-even；不能仅凭
  Ascend 结果断言 CUDA 不会降速。
- 论文/README 的 `9.1x` 不能直接当成当前配置的 wall-clock 承诺：论文实验平台
  是 RTX 4090，表格配置以 batch=8 为主，headline 主要对应计算量/FLOPs 降低；
  当前实验测的是 Ascend、batch=1 的同步端到端生成时延。
- 输出 hash 是正确性锚点，不是任务质量指标。正式验收仍需 baseline/cache 使用
  同一数据集、同一解码参数并比较准确率。

## 产物

- 四模式 JSON：`reports/perf_anchor_npu/`
- microbenchmark：`reports/microbench_dream_npu.json`、
  `reports/microbench_llada_npu.json`
- 三步 profiler：`reports/profile_dream_*_npu/`
- 复现工具：`benchmark_text.py`、`scripts/profile_npu.py`、
  `scripts/microbench_npu.py`

