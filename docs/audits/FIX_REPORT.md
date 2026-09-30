# Critical Fix Report

日期：2026-09-28。范围仅为旧 `select-threshold` CLI 的开放集判定边界。原始 `FINAL_AUDIT.md` 与历史实验产物均未改动。

## Root Cause

`face_compare_system/face_compare/operating_point.py:select_operating_point` 原先自行判断 `second_distance - best_distance >= margin`；主系统 `decision.py:accept_identity` 对浮点等号附近使用 `math.isclose(..., rel_tol=0, abs_tol=1e-12)`。十进制的 `0.315 - 0.275` 在二进制浮点中为 `0.03999999999999998`，所以旧 CLI 拒绝而线上和研究共享规则接受。

修复前的最小 validation 场景：一条 known `A`（best `0.275`、second `0.315`）和一条可用、明显较远的 unknown（best `0.6`、second `0.7`），margin `0.04`、目标 FPIR `0`。旧 CLI 选 `1e-8`，因为它错误地认为 known 在 `0.275` 仍不能接受；主规则及研究 `decide` 接受该 known。

## Code Change

`face_compare_system/face_compare/operating_point.py:select_operating_point` 只增加对共享 `accept_identity` 的导入，并把扫描中的一处内联 threshold＋margin 判断替换为 `accept_identity(row["distance"], second, threshold, margin)`。没有复制容差常数，没有改候选阈值集合、FPIR 排序、返回字段、异常校验、配置、metric、生产识别或研究主实验。

这是**有意的最小行为修正**：CLI 在浮点等号邻域的曲线和推荐阈值可改变；其他不在该邻域的输入仍按原规则判断。历史 run 不因源码变化被重写或追认成新版本。

## Regression Tests

在 `face_compare_system/tests/test_operating_point.py` 新增 5 个窄用例：

1. `0.275/0.315/0.04` 的十进制边界，CLI 与共享规则均接受；
2. second-best gap 比边界高 `2e-12`，继续接受；
3. gap 比边界低 `2e-12`，已越过允许容差，继续拒绝；
4. 单身份、无 second-best，不强求 margin；
5. best 距离等于阈值时接受，比阈值低一 ULP 时拒绝，并核对扫描曲线。

相关定向测试（旧阈值工具、共享规则、研究标定）：**19 passed**。测试没有删除或放宽旧断言。

## Full Test Result

命令：`env PYTHONDONTWRITEBYTECODE=1 face_compare_system/.venv-ui/bin/python -m pytest -q face_compare_system/tests face_research/tests -p no:cacheprovider`

| total | passed | failed | skipped |
| ---: | ---: | ---: | ---: |
| 454 | 454 | 0 | 0 |

原验收为 449 passed；新增 5 个用例后数量为 454，不是原有测试减少。

## Threshold Consistency

相同的两条 validation 记录，在修复后：旧 CLI `select_operating_point(..., target_fpir=0)` 选 **`0.275`**；`accept_identity(0.275, 0.315, 0.275, 0.04)` 为 **True**；研究 `selection.decide` 返回 **A**，`calibrate_open_set(..., split="validation", margins=(0.04,), target_fpir=0)` 也选 **`threshold=0.275, margin=0.04`**。修复前旧 CLI 为 `1e-8`。CLI 返回 schema 仍是原来的 9 个字段；没有修改线上默认 `τ=0.275, m=0.04` 或研究标定实现。

另用临时 validation JSON 和临时输出文件调用实际 `cli.main(..., "select-threshold", ...)`：退出码 **0**，输出 `distance_threshold=0.275`、`known_correct=1`。未创建或访问生产人员库。

一致性结论限于相同输入、同一候选规则的接受语义；旧 CLI 与研究标定的候选阈值搜索策略和输出 schema 原本不同，本轮没有强行统一整个选点算法。

## Benchmark Regression

现有 `face_research/retrieval_benchmark.py:run_retrieval_benchmark()`，默认 128 维、seed 42／查询 seed 43、20 次预热、每配置 200 次交替配对查询。1,000 **合成**身份×每身份 3 模板：

| 记录 | 标量 baseline p50 | 分组优化 p50 | 倍数 |
| --- | ---: | ---: | ---: |
| 历史参考 run | 9.144625 ms | 0.274125 ms | 33.36× |
| `FINAL_AUDIT.md` 修复前现场重跑 | 9.138979 ms | 0.274854 ms | 33.25× |
| 本轮修复后现场重跑 | 9.381062 ms | 0.289187 ms | 32.44× |

当前 p95：baseline **10.240307 ms**，优化 **0.372367 ms**。六组负载、共 **1,200** 次查询的完整身份排名及全部分数严格相等，**36,000** 次开放集决策对照一致；核心配置的 200 次全排名／分数和 6,000 次最终决策对照也全部一致。当前优化路径仍显著快于标量参考，单次计时波动不构成明显性能退化。OpenCV 请求线程数 1、环境回报 8，且未控制原生 BLAS 线程；不把毫秒波动解释成可重复的改进或回归。

该实验测量**精确检索耗时和实现等价性**，不测人脸识别准确率、真实摄像头 FPS 或整条推理链路。8 个主要历史结果目录再次通过产物 SHA 校验，没有覆盖旧 run／metrics／metadata。

## Files Changed

- `face_compare_system/face_compare/operating_point.py`：唯一生产代码修改，扫描调用共享判定函数。
- `face_compare_system/tests/test_operating_point.py`：5 个边界回归用例。
- `FIX_REPORT.md`：本修复记录。
- `FINAL_AUDIT_V2.md`：仅重新验收受影响结论。

未修改 `decision.py`、`face_research`、检索、SQLite、UI、tracking、配置、模型、历史结果或原 `FINAL_AUDIT.md`。
