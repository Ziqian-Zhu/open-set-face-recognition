# 精确检索优化与全预算回归

日期：2026-09-26。对应计划 P0-2/P0-4 的一致性与预算覆盖，以及 P1-2/P2-2 的瓶颈决策。本轮没有引入 ANN/FAISS、替换模型、改变阈值、修改 UI 或读写既有人脸库。

## 1. 优化点与不变量

原 SQLite 路径已经用矩阵乘法一次计算全部模板距离，但之后逐身份调用 NumPy 的小数组聚合；1,000 身份意味着每查询约 1,000 次 Python 聚合调用。

现在在缓存 revision 更新时，按**实际模板数量**建立身份分组和模板索引。同组做二维批量聚合，再恢复身份顺序并使用原 `(distance, person_id, name)` 元组排序。没有补零/inf、截断身份或使用模板 Top-K 近似第二身份；保留全部模板、`min(k,N)` 规则和准确的 Top1/Top2。

代码：[批量聚合](../face_compare_system/face_compare/aggregation.py)、[缓存与精确检索](../face_compare_system/face_compare/vector_database.py)。标量聚合仍作为独立数值参考保留；研究选样的标量公式未改。新增索引 payload 是 O(模板数＋身份数)，模板数量种类很多时批量收益可能较小，不保证任意分布都相同倍数加速。

## 2. 同库同查询配对结果

运行：[exact-retrieval-paired-20260926-01](results/exact-retrieval-paired-20260926-01/metrics.json)，[CSV](results/exact-retrieval-paired-20260926-01/metrics.csv)。脚本为 [retrieval_benchmark.py](retrieval_benchmark.py)。

临时 SQLite，128 维合成单位向量，gallery seed=42，query seed=43，Top-3 Median；每组预热 20 次、正式 200 次，逐查询交替先运行参考/批量方法。两种实现使用同一快照、同一矩阵乘法、修订检查和排序，计时之外比较每个候选的完整得分与身份排名。这个对照不是生物识别准确率，也不是与 FAISS 比较。

环境与 P1 同机：Darwin 23.3.0 / arm64，Python 3.12.14、NumPy 2.5.3；一个 Python worker，OpenCV 请求 1 / 报告 8，BLAS 线程未限定；实际 CPU 品牌未读取成功，不称单核测试。源代码/构建哈希与全部原始计时在 JSON。

| 模板分布 | 身份数 | 标量 p50/p95 ms | 批量 p50/p95 ms | p50 比值 |
| --- | --- | --- | --- | --- |
| 每人 3 张 | 10 | 0.117 / 0.128 | 0.037 / 0.044 | 3.16× |
| 每人 3 张 | 100 | 0.956 / 1.001 | 0.068 / 0.072 | 14.03× |
| 每人 3 张 | 1000 | 9.145 / 9.304 | 0.274 / 0.300 | 33.36× |
| 每人 1/2/3/5/8 张循环 | 10 | 0.131 / 0.137 | 0.096 / 0.100 | 1.37× |
| 每人 1/2/3/5/8 张循环 | 100 | 0.960 / 0.999 | 0.116 / 0.127 | 8.30× |
| 每人 1/2/3/5/8 张循环 | 1000 | 9.240 / 9.494 | 0.333 / 0.369 | 27.74× |

总计 **1,200 次查询完整分数和排名完全相等**（不是容差内近似），同时检查 36,000 个开放集工作点/边界判定。另有单元测试覆盖其他聚合、float32/float64、不同 K、并列、阈值等号、非连续数组、不同连接增删恢复及事务回滚。这不能证明对所有可能浮点输入的数学完备性，但比只对比 Top-1 或默认准确率有更强的回归范围。

1,000 身份时额外分组索引 payload 为均匀库 32,000 字节、混合库 38,400 字节。它只统计新增 NumPy 索引数组，不是进程 RSS；向量库本身与临时工作数组另占内存。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.retrieval_benchmark \
  --gallery-sizes 10,100,1000 --iterations 200 --warmup 20 \
  --output-dir face_research/results/exact-retrieval-reproduction-001
```

无需下载模型、照片或开启摄像头。输出仅允许新目录；任何查询分数/排名不一致均拒绝发布“加速”结果。

## 3. 静态流水线复测与 ANN 决策

优化后的 [静态基准](results/static-synthetic-benchmark-optimized-20260926-01/metrics.json) 使用与 P1 相同输入/配置哈希、预热/采样和合成图库；[最新规模曲线](results/static-synthetic-benchmark-optimized-20260926-01/scaling_curve.png)。它是另一次整链路测量，不作为上表严格配对加速比的来源。

1,000 身份、每人 3 模板时，单静态图的检测至身份排序 p50/p95 为 **17.914/18.573 ms**，其中 embedding p50≈16.126 ms、matching p50≈0.383 ms；重复同图双脸布局为 **35.336/36.974 ms**。这里输入仅 120×120 / 240×120，不是真实多人人群、视频运动或相机/UI FPS，链路不含阈值决策和轨迹确认。

判断：此前大部分检索开销可通过保持精确语义的批量聚合移除；在已测的 1,000 身份负载上，SFace 重新成为主要阶段耗时。因此当前没有足够收益依据引入 ANN。**P2-2 的“只有瓶颈明确才上 ANN”条件在此规模下未满足，不安装 FAISS、不宣称做过 FAISS 实验。** 若目标规模/并发/时延预算扩大，再测候选召回、真实第二身份保持率、索引更新成本和端到端净收益。

## 4. K=5 补全，不替换原主比较

运行：[meglass-mixed-full-budget-20260926-01](results/meglass-mixed-full-budget-20260926-01/metrics.json)。使用原混合协议完全相同的六张候选 gallery、validation/test，不补图、不按结果筛图；新增 [扩展协议](protocols/meglass_mixed_budget_extension_v1.json) 明确在已看过 K≤3 结果后补 K=5，属于次要描述性分析，不能说是初始盲测注册。后续纠正了扫描准确率的未知失败计数，新 run 为 `meglass-mixed-metric-fix-20260926-01`，选样/阈值/原有识别计数不变；旧准确率字段弃用，详见 [指标更正](METRIC_CORRECTION.md)，本页性能数字不受影响。

公开矩阵现覆盖 **K=1/2/3/5 × 7 组选样/seed × 6 聚合 = 168 组合**。把原 126 组合与 `meglass-mixed-p0-regression-20260926-02` 比较，所选模板哈希、validation 阈值/margin/计数、test 识别计数、固定工程工作点及 margin 消融**全部不变**。本轮性能优化没有以减少搜索或放松阈值换取速度。

主比较仍固定 K=3、Top-3 Median：First-K 14/24、Coverage 13/24；两者可用未知误接均 0/13、Wilson 95% 上界约 22.8%。新增 K=5、Top-3 Median 下 First/Quality/Diversity/Coverage 均 14/24；Random seeds 42/43/44 分别 13/24、14/24、14/24，未知均 0/13。没有因此更换主比较或宣称自定义选样获胜。

```bash
face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.test.json \
  --protocol-spec face_research/protocols/meglass_mixed_budget_extension_v1.json \
  --budgets 1,2,3,5 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --allow-missing-session-ids --output face_research/results/meglass-full-budget-reproduction-001
```

## 5. 验证记录与未完成项

- 全量：**298 passed**。命令与根 README 相同。
- `smoke_models.py` 与 `smoke_video.py` 均通过；两者是公开图片连通性检查，显式放宽部分质量要求，不是准确率证据。
- `check_startup.py` 在正常桌面通过，临时库、未开启相机。沙盒内 Tk 缩放读数失败，换桌面环境验证，不修改应用逻辑绕过。
- 上述三个新结果目录均通过 `verify_result_directory`；历史 run 保留不覆盖。

完整目标尚未关闭：公开 1:1 验证、P2-1 模型接口/许可/对照等仍须按 [逐项核验表](../docs/audits/COMPLETION_AUDIT.md) 继续检查；不因这一性能结果把整个改造计划标成完成。
