# P0 验收清单

核对日期：2026-09-26。依据根目录 `PROJECT_AUDIT.md` 的改造前基线与 `IMPROVEMENT_PLAN.md` 的 P0，**不是课程任务书完成度**。

结论：**P0 软件实现与回归验收已通过；独立真人及统计证据验收仍待数据。** 不将公开裁剪图、合成向量测试或模型连通性检查描述为真人准确率。

范围更新：用户随后取消了本轮真人采集。因此上述真人部分现为**取消/不在本轮范围**，不再阻塞公开数据与工程路线的交付；这不是将其改为已通过。下面 0/34 张及采集命令保留为历史/可选操作信息，当前不要求执行。

后续更新：全套已达 **319 passed**，公开矩阵补齐 168 组合，新增独立 1:1 校准及公开戴镜／摘镜 run；测试 88/89 可用 genuine 接受、全部 88/200，低 FAR 保持不足。见 [1:1 完整记录](VERIFICATION_RESULTS.md)。下方 256 项／126 组合为 P0 初次收尾记录，保留其历史版本；没有把新结果回填旧 run。

最新统计复核为 **415 passed**：修复扫描准确率把未知采集失败计作正确拒识的问题，旧扫描字段弃用并保留；168 组合和四种门控重新运行，全部选样/阈值/原有识别计数不变。当前应引用 [指标更正](METRIC_CORRECTION.md) 中的新 run，历史验收记录不代表旧准确率字段仍可用。

## 逐项状态

| 条目 | 已有实现及本轮补齐 | 仍受数据限制的部分 |
| --- | --- | --- |
| P0-1 协议与泄漏 | 三集合批次隔离、未知人员隔离、解码/对齐重复检查；补齐 gallery 元数据一致性、验证对→test probe 会话检查、重复/交换 pair 与同图改标签检查；主比较协议绑定、冻结凭据与测试访问日志 | 同人换匿名 ID、真实拍摄批次和知情同意无法由程序证明；现有真人索引没有照片 |
| P0-2 统一聚合与语义 | 线上与研究端共享聚合/接受公式；保留线上默认 top-3 median、阈值与 UI；实验产物明确半余弦距离、margin、FPIR/FAR、各分母 | UI 历史 `similarity` 是展示分，不能作为身份概率；不改变旧字段以维持兼容 |
| P0-3 联合标定与指标 | validation-only 的阈值/margin 搜索、1:N 指标、独立 1:1 pair 入口/ROC/AUC；补充随机手算穷举对照、边界/并列测试及每个组合的固定工程工作点；后续公开 1:1 run 已完成 | 旧 1:N run 无 pairs 时 ROC 仍是 NOT EVALUATED；新公开 1:1 缺会话，不能证明真实跨批次或低 FAR，TAR@低 FAR 为 insufficient samples |
| P0-4 核心对照 | K=1/2/3/5 × 5 类选样（Random 3 seeds）× 6 聚合，共 168 组合的合成回归；公开混合协议 K=1/2/3 共 126 组合；主比较自动单列，不从测试最高分中挑选 | 真人混合外观独立测试尚未运行；公开主比较没有胜过基线 |
| P0-5 产物与回归 | 256 项测试通过；递归代码哈希覆盖 evaluation/datasets，不扫描私有 data/results；冻结参数先落盘、结果整目录暂存后发布、文件校验、失败 run 留档、CSV 原始计数/区间、带数值坐标的阶梯阈值曲线 | 本地凭据不是第三方时间戳或防篡改注册系统，不能证明研究者从未看过旧 test |

## 本轮验证记录

- 全量：`face_compare_system/.venv-ui/bin/python -m pytest -q face_compare_system/tests face_research/tests -p no:cacheprovider` → **256 passed**。
- 完整合成矩阵及 1:1 指标：`face_research/tests/test_p0_regression.py`。固定种子重复运行结果一致；改变 test 标签会改变测试指标，但不改变模板、阈值、margin 或冻结凭据。
- 默认窗口：`face_compare_system/scripts/check_startup.py` 在正常桌面会话通过，Python 3.12.14 / Tk 9.0.4，使用临时库，未打开摄像头。沙盒内首次因桌面/Tk scaling 环境失败，不作为算法失败；在桌面权限下重试通过。
- 官方模型：`scripts/smoke_models.py` 的五点对齐、128 维特征与连通性检查通过。低清公开样例绕过画质门禁仅检查模型接线，不能用于效果结论。
- 最新公开数据回归：`results/meglass-mixed-p0-regression-20260926-02/`；早先 `...-01/` 保留，它是在图表坐标/阶梯渲染改进前的本轮回归。

公开回归的输入来自已经查看过的固定 MeGlass 混合清单，**不是新增盲测**。与此前 `meglass-mixed-v1-seed42` 对比，126 个组合的阈值、margin、所选模板及 1:N 计数无变化。主比较仍是 First-K **14/24**、Coverage **13/24**；两者未知误接受均为 **0/13 可用未知脸**，Wilson 95% 上界约 **22.8%**。44 张 test probe 中 8 张未检出脸、8 张画质拒绝。不能宣称“已提升鲁棒性”或“零误接”。

## 复现与审计

所有命令在工作区根目录执行。新运行必须换新输出名：

```bash
face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.test.json \
  --protocol-spec face_research/protocols/meglass_mixed_v1.json \
  --budgets 1,2,3 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --allow-missing-session-ids \
  --output face_research/results/meglass-mixed-new-run
```

`<run>.audit/` 先记录开始；所有 validation 参数固定后写 `frozen_validation.json`，随后才写 `test_opening.json` 并读取 test。完成写 `completed.json`，正常异常写 `failed.json`；被强制终止可能只留下已有事件。失败或中断后不覆盖旧 run，请换名并保留其测试访问历史。库函数调用方如需同样落盘审计，使用 `RunJournal` 的 `on_freeze/on_test_open` 回调；裸调用只在返回值中携带冻结凭据。

完整结果包含原有六项产物，以及 `frozen_validation.json`、`primary_comparison.json`、`artifacts.json`。无 1:1 数据时 CSV 的 FAR/FRR/TAR 留空，并给出状态/原因，不用 FPIR 代填。

校验产物：

```bash
face_compare_system/.venv-ui/bin/python -c 'from face_research.evaluation.artifacts import verify_result_directory; print(verify_result_directory("face_research/results/meglass-mixed-p0-regression-20260926-02"))'
```

## 真人数据关口

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.live_mixed status \
  --index face_research/data/live/pilot.csv --profile pilot
```

历史空索引为 **0/34 张**，本轮采集已取消，不再是当前待办。若未来重启，pilot 需要 3 名已知、2 名 validation 未知、另 2 名 test 未知，共 7 名自愿参与者；3 名已知人员各有 gallery/validation/test 三个真实不同批次，共 13 个采集批次任务。详见 [可选操作流程](README.md)。不要虚构 session ID，也不要根据模型测试结果挑图重拍。

pilot 只检查 1:N 端到端流程，不能支撑总体准确率或低 FPIR 结论，也不直接构成 1:1 跨批次验证对。后者还需另行准备有真实身份/批次标签的 reference–probe pairs；不足时保持未测。真人照片和向量不上传、不写入公开产物。
