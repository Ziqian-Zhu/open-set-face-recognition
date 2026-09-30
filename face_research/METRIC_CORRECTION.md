# 开放集准确率统计更正

日期：2026-09-26。仅修正研究报表口径；未改线上模型、质量门槛、阈值、UI 或人员库。使用已经合法取得的同一份 MeGlass 清单复跑，不需要新视频或真人采集，不是新的独立盲测。

## 错误与修正

旧 `evaluation/calibration.py` 的阈值扫描将未知人员的采集失败算成了正确拒识：

```text
旧 accuracy_all_attempts = (已知正确接受 + 全部未知 - 未知误接受) / 全部尝试
新 accuracy_all_attempts = (已知正确接受 + 可用未知 - 未知误接受) / 全部尝试
```

“没检测到脸”“质量不合格”等失败仍保留在分母，但不算成功识别。新报告同时输出正确数、总数、已知/未知采集失败数和未知成功拒识数；JSON 与 CSV 使用相同口径。这个总准确率受已知/未知样本比例影响，不能替代分别报告的 DIR、FPIR 和各自分母。

兼容字段 `unknown_rejection_all_attempts` 仍表示 **1−全部尝试 FPIR，即未接受比例**，其中包括采集失败；没有悄悄改动它的定义。新增 `unknown_correct_rejection_all_attempts` 才表示成功处理的未知脸拒识数／全部未知尝试。例如本次 full 门控 test 的未知未接受是 20/20，但成功拒识只有 13/20，另有 7 次采集失败，不能写成“未知识别准确率 100%”。

阈值选择使用验证集已知正确数、错认数、可用未知 FPIR 及预定并列规则，**从未使用 `accuracy_all_attempts` 排序**。因此本次是指标纠错，不是改算法以追逐已看过的 test 分数。

## 新旧结果与影响

| 实验 | 保留的旧 run | 更正后的新 run |
| --- | --- | --- |
| 168 组合矩阵 | `meglass-mixed-full-budget-20260926-01` | [meglass-mixed-metric-fix-20260926-01](results/meglass-mixed-metric-fix-20260926-01/metrics.json) |
| 四种质量门控 | `meglass-mixed-quality-p1-20260926-01` | [meglass-mixed-quality-metric-fix-20260926-01](results/meglass-mixed-quality-metric-fix-20260926-01/metrics.json) |

**所有使用旧公式产生的 `threshold_selection[].accuracy_all_attempts` 及同名 CSV 列均已弃用，请勿继续引用。** 这不只限于上表两个旧 run，也包括更早的矩阵/质量实验；旧文件及其校验和保留，不原地覆盖。已有公开 1:1、模型对照和性能基准不使用这个公式，不受此错误影响。

已完成逐项新旧比对：

- 矩阵 168 组合、16,552 条扫描记录，质量消融 4 组、437 条扫描记录全部核验。每条准确率减少值恰为“未知采集失败数／全部尝试”；每条 CSV 的分子/分母与 JSON 一致。
- 新增的 1,362 处识别统计均核对成功拒识和总正确数。除新增/更正指标、版本哈希、实际计时以及补充的限制说明外，所有原有字段递归比较相同，包括选样哈希、τ/m、原有验证/测试计数、条件分层及配对比较。
- 两个新 run 仍先写 validation 冻结凭据，再打开 test；矩阵 8 项、质量 4 项产物校验通过，两份旧 run 校验也通过。

手工可核对的验证集例子：First-K=3、Top-3 Median、full 门控共 44 次尝试；已知正确 17 次、可用未知 14 次、未知误接 0 次、未知采集失败 6 次。旧值 37/44（84.09%）更正为 **31/44（70.45%）**。四种门控冻结工作点的更正值分别为 none 35/44、brightness_only 34/44、sharpness_only/full 31/44。

主结论没有改变：test 已知正确 First-K 14/24、Coverage 13/24；两者可用未知误接仍为 0/13，不能宣称零总体风险。质量消融仍为 none 19/24、full 14/24。新增 test 全部尝试正确数分别为 First-K 27/44、Coverage 26/44；这些混合分母只用于口径复核，不替换原先固定的主指标。

## 回归与复现

全量 **415 项测试通过**。新增手算失败样例、空/全失败样例、未知未接受与成功拒识区分、矩阵/质量 CSV 分子分母检查；原随机穷举测试现同时检查每一条扫描记录，而不只检查最终阈值。没有把更多未知采集失败当成提高准确率的办法。

在工作区根目录执行，复现时改用新的输出名称：

```bash
face_compare_system/.venv-ui/bin/python -m pytest -q \
  face_compare_system/tests face_research/tests -p no:cacheprovider

face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.test.json \
  --protocol-spec face_research/protocols/meglass_mixed_budget_extension_v1.json \
  --budgets 1,2,3,5 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 --allow-missing-session-ids \
  --output face_research/results/meglass-mixed-metric-reproduction-001

face_compare_system/.venv-ui/bin/python -m face_research --quality-ablation \
  --validation face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.test.json \
  --budget 3 --margins 0,0.02,0.04,0.06 --target-fpir 0.05 --allow-missing-session-ids \
  --output face_research/results/meglass-mixed-quality-metric-reproduction-001
```

数据仍存在裁剪/预筛偏差、缺会话与 test 重复使用限制。更正统计不会消除这些限制，也不证明摘镜、真实运动或跨会话泛化。
