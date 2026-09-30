# P1 公开数据实验与工程验证

日期：2026-09-26。真人采集已按用户要求取消，不再作为当前交付门槛。本页是可追溯的实验记录，不是现场验收或课程报告；不改变线上配置、UI 或现有人员库。

后续指标更正：旧阈值扫描的 `accuracy_all_attempts` 错把未知采集失败算成正确拒识，已弃用；更正版为 `meglass-mixed-quality-metric-fix-20260926-01`。本页已知正确、未知误接及门控得失数均经复跑确认不变，历史 run 原样保留。详见 [更正公式、完整影响与新产物](METRIC_CORRECTION.md)。

## 交付范围

| 计划项 | 本轮证据 | 边界 |
| --- | --- | --- |
| P1-1 质量/间隔消融 | 混合戴镜公开协议，四种 probe 门控；复核已有 margin 对照 | 描述性探索，不能推出现场泛化或安全保证 |
| P1-2 性能基准 | 10/100/1000 合成身份；静态单图及同图双份布局；原始计时、p50/p95、曲线 | 不是真实多人视频、相机 FPS 或准确率 |
| P1-3 时序配对评估 | 同一次导出的单帧/多帧结果；误接、覆盖、延迟、状态切换；合成轨迹回归 | 真实运动/交叉遮挡效果未测，已移出本轮必做范围 |
| P1-4 简历与面试材料 | 根目录 README、RESUME、INTERVIEW；每项主张附代码或实验依据 | 只描述已实现能力与限定实验，不写未经证实的提升 |

原始计划的“真人/真实多人视频实证”没有被实验替代或勾成通过；当前交付的是调整范围后的 P1。

## 1. 质量过滤的召回代价

运行：[meglass-mixed-quality-p1-20260926-01](results/meglass-mixed-quality-p1-20260926-01/metrics.json)。对应冻结及访问记录在同名 `.audit/` 目录。所有策略先只在 validation 定参数，再打开 test；同时额外报告统一使用 full 门控的 validation 工作点，避免把门控作用与重新标定混为一谈。

协议：MeGlass 120×120 裁剪图，12 个已知身份，每人 gallery 为黑框眼镜 3 张＋无眼镜 3 张；First-K=3、Top-3 Median；test 共 24 张已知图＋20 张未知图。所有 gallery 始终经过完整生产画质与一致性检查，本次只消融 **probe 光学画质门控**。未知 validation/test 身份互斥。种子 42，validation 目标经验 FPIR≤5%，margin 网格 0/0.02/0.04/0.06。

四组恰好都在 validation 选到 `τ=0.3071753680706024, m=0`，因此本次独立标定与共享 full 工作点的计数一致；这不是预先保证不同门控会选到同一阈值。

| probe 门控 | 已知正确 / 全部已知 | 已知比例 Wilson 95% | 未知误接 / 可用未知 | 未知误接 Wilson 95% 上界 | 画质拒绝 / 全部 44 |
| --- | --- | --- | --- | --- | --- |
| none | 19/24 | 59.5%–90.8% | 0/16 | 19.4% | 0/44 |
| brightness_only | 18/24 | 55.1%–88.0% | 0/15 | 20.4% | 2/44 |
| sharpness_only | 14/24 | 38.8%–75.5% | 0/13 | 22.8% | 8/44 |
| full | 14/24 | 38.8%–75.5% | 0/13 | 22.8% | 8/44 |

四组均有 8/44 未检出脸、0/24 已知错认。漏检与画质拒绝单列，不叫“未知正确识别”。同图配对：full 相对 none 失去 5 次已知正确、没有新增正确；brightness_only 失去 1 次。条件分层：none 的戴镜/无镜为 9/12、10/12；full 为 7/12、7/12。每名已知身份贡献两图，表中 Wilson 区间只是逐尝试描述，不能视作独立人员级置信区间。

**可以说**：这份样本中画质规则误拒了部分仍可正确匹配的 probe，且本次清晰度门控与 full 结果相同。**不能说**：取消过滤已提升实际产品准确率 20.8 个百分点、无需画质门槛、已解决摘镜问题或零误接。裁剪分辨率、身份预筛、重复查看过的 test、缺真实拍摄批次都会限制结论；三个未知分母不同，也不能直接比较零误接的安全性。线上门槛保持不变。

复现（工作区根目录，数据需按来源条款合法获取，输出必须换新 run ID）：

```bash
face_compare_system/.venv-ui/bin/python -m face_research --quality-ablation \
  --validation face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-v1-registered-seed42/mixed.test.json \
  --budget 3 --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --allow-missing-session-ids \
  --output face_research/results/meglass-mixed-quality-reproduction-001
```

## 2. Margin 没有在小样本里展现净收益

依据已有 [P0 回归结果](results/meglass-mixed-p0-regression-20260926-02/metrics.json) 的 `k3__first__s42__top3_median.margin_ablation`，同样使用完整 probe 质量门控：

- threshold-only：validation 选得 τ≈0.307175、m=0，test 已知正确 14/24、错认 0/24、可用未知误接 0/13。
- threshold＋固定 margin：独立在 validation 重新选 τ，结果仍约 0.307175，m=0.04；已知正确 13/24、错认 0/24、可用未知误接 0/13。
- 配对失去正确 1、获得正确 0。当前数据没有观察到减少误接的收益；未知分母太小，不能证明 margin 没用，更不能自动删除线上间隔。

有限模板选样的主比较仍是 First-K 14/24、Coverage 13/24，没有证据证明自定义选样优于基线。

## 3. 静态/合成负载性能

正式记录：[static-synthetic-benchmark-p1-20260926-02](results/static-synthetic-benchmark-p1-20260926-02/metrics.json)、[分阶段 CSV](results/static-synthetic-benchmark-p1-20260926-02/metrics.csv)、[检索曲线](results/static-synthetic-benchmark-p1-20260926-02/scaling_curve.png)。早先 `...-01` 保留用于审计；`...-02` 修正线程请求元数据与“缓存修订检查”的命名，不是挑选最快的一次。

环境：Darwin 23.3.0 / arm64，8 个逻辑 CPU；CPU 品牌无法读出，仅报告 `arm`，不猜具体芯片。Python 3.12.14、NumPy 2.5.3；OpenCV 版本、构建/模型/配置/代码哈希在 JSON。一个 Python 工作线程，OpenCV 请求 1、API 报告 8（未按请求生效），BLAS 未限制；**不能称为单核或确定的 8 线程测试**。

输入：固定混合 validation 清单的第一张 gallery 图片（未按 test 表现选图），120×120；双脸负载在内存并排重复同图为 240×120，**不是两个不同真人**。每场景预热 10 次、采样 100 次，按库规模升序、先单图后双图计时；每帧分别 1/2 张脸全部通过原生产质量门槛。图库是固定种子 42 的合成 128 维单位向量，每身份 3 模板；隔离查询 seed=43。

| 身份数 | 隔离缓存检索 p50/p95 ms | 单图流水线 p50/p95 ms | 重复双图流水线 p50/p95 ms |
| --- | --- | --- | --- |
| 10 | 0.113 / 0.121 | 17.521 / 18.422 | 34.750 / 37.625 |
| 100 | 0.951 / 0.968 | 18.572 / 19.853 | 36.373 / 36.842 |
| 1000 | 9.159 / 9.254 | 26.861 / 28.006 | 53.335 / 54.523 |

这里“流水线”指检测→画质→对齐→特征→身份排序，JSON 中名为 `end_to_end_ms`；不含模型加载、图像解码、摄像头、UI、threshold/margin 决策或轨迹确认。`1000/p50` 只是该静态处理负载的吞吐估计。每阶段原始 100 条样本保留；单阶段 p50 之和不必等于整链路 p50。

1,000 身份单图的 SFace p50≈16.09 ms、matching≈9.28 ms，检索已是值得剖析的部分；10 身份时 matching≈0.16 ms，特征提取占主要耗时。下一步优先剖析 Python 身份聚合/排序和批量缓存更新，再决定是否比较 ANN。当前没有 FAISS 对照，不能声称引入它一定更快或不影响第二名 margin。

向量 payload 在 1,000 身份时为 1,536,000 字节，只是一个 float32 矩阵，**不是进程总内存**。`incremental_gallery_build_ms` 包含逐身份写入及每次缓存重建；新增 100→1000 身份约 7.37 秒，不能称为从空库构建或纯 SQLite 写入耗时。`cached_revision_check_ms` 是已缓存后的修订检查，不是完整缓存重建性能。

```bash
# 使用自己合法持有的单脸静态图；复现相同数字需相同哈希输入/环境。
face_compare_system/.venv-ui/bin/python -m face_research.benchmark \
  --single-image /绝对路径/合法单脸图片.jpg --repeat-single 2 \
  --gallery-sizes 10,100,1000 --templates-per-identity 3 \
  --iterations 100 --warmup 10 --opencv-threads 1 \
  --output-dir face_research/results/static-synthetic-reproduction-001
```

## 4. 单帧与多帧：评估能力已接通

新导出 schema `video-dual-decision-v2` 为同一人脸保存 `single_frame` 原始判定和原有稳定输出。`evaluate-temporal` 共享检测、框匹配与轨迹关联，只比较时序确认＋同身份冲突抑制；不是检测器或跟踪关联算法的独立对照。

除误认/误接外，同时报告待确认造成的输出覆盖损失、首次正确确认延迟、从未确认的片段、连续已识别标签跳变，以及**包含待确认/漏检的全部状态变化**。这样不能通过一直输出“待确认”伪装成稳定。旧版导出仍可作 stable 评估，但不允许从稳定结果反推原始单帧判定。

合成回归使用真实 `MultiFaceTracker`、预设十帧两人观察结果，验证误接阻断也伴随确认延迟和正确输出损失；只证明统计与状态机接线，不代表真实视频收益。另运行真实 YuNet/SFace＋临时 SQLite＋视频解码的 12 帧重复公开图片 smoke，通过双轨迹及 known/unknown、single_frame 字段断言；该 smoke 显式放宽质量门槛，不能用于精度或性能主张。没有访问现有人员库或开启相机。

用已有、合法且完整标注的视频导出评估时（示例路径须替换）：

```bash
cd face_compare_system
.venv-ui/bin/python -m face_compare.cli evaluate-temporal \
  /绝对路径/video-v2.jsonl /绝对路径/truth.json \
  --output /绝对路径/新建-temporal-score.json
```

## 5. 回归、可复现与公开边界

```bash
face_compare_system/.venv-ui/bin/python -m pytest -q \
  face_compare_system/tests face_research/tests -p no:cacheprovider
# 263 passed
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/smoke_video.py
```

P0 矩阵、P1 质量、P1 benchmark 结果均通过 `verify_result_directory` 的文件完整性校验。运行时快照与模型、配置、代码、输入哈希保存在各自结果；不同阶段新增代码导致版本哈希不同是正常情况，旧 run 不被重写。质量 CLI 也有 freeze→test_opening→completed/failed 日志及 create-only 原子发布。

数据沿用 MeGlass 来源许可，限相应研究/教学用途。默认不上传原图、embedding、人员库或逐人可关联输出；公开展示优先使用本页聚合统计。结果哈希是复核工具，不是法律授权、第三方预注册或统计独立性的证明。
