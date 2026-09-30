# Open-set Multi-face Recognition System

## 项目简介 | Project Overview

**中文：** 这是一个面向计算机视觉工程实践的开放集多人脸识别与有限模板选样项目。`face_compare_system` 提供本地 CPU 桌面应用：YuNet 检测与五点对齐、SFace 特征、画质门禁、SQLite 精确检索、身份级多模板聚合、未知/歧义拒识和逐轨迹多帧确认。`face_research` 提供可复现的选样、阈值校准、评估与性能实验。YuNet 和 SFace 是预训练模型；本项目的贡献是系统设计、判定与检索实现及实验协议，不宣称发明新网络，也不具备活体检测或工业级身份认证保证。

**English:** An engineering project combining open-set multi-face recognition with research on template selection under a limited budget. `face_compare_system` is a local, CPU-based desktop application with YuNet detection and five-point alignment, SFace embeddings, quality gates, exact SQLite retrieval, identity-level multi-template aggregation, unknown/ambiguous-identity rejection, and per-track temporal confirmation. `face_research` provides reproducible template-selection, threshold-calibration, evaluation, and performance experiments. YuNet and SFace are pretrained models; the contributions here are the system design, decision and retrieval implementation, and evaluation protocol—not a new neural network. Liveness detection and industrial-grade identity assurance are not claimed.

## 三个核心问题 | Three Core Questions

1. **开放集身份识别 / Open-set identity recognition：** 摄像头里的人不一定在库中。最佳候选还必须通过距离阈值和 Top1/Top2 身份间隔；取最近邻并不代表可以接受。A person in view may be absent from the gallery; the nearest candidate is accepted only after the distance threshold and top-two identity margin checks.
2. **鲁棒多模板匹配 / Robust multi-template matching：** 同一人有外观变化。保留多个模板、质量门禁和按身份聚合，再做轨迹独立确认；这些机制有拒绝/延迟代价，不保证只带来收益。Multiple templates, quality gates, per-identity aggregation, and independent track confirmation handle appearance changes, but can increase rejection or confirmation delay.
3. **有限预算模板选样 / Template selection under a limited budget：** 每人 N 个合格候选，只保留 K 个时，比较 First-K、Random、Quality、Diversity 与质量－覆盖－一致性选样，并用验证集校准各自工作点。When only K of N eligible images per identity can be retained, the research module compares selection strategies and calibrates their operating points on validation data.

## Architecture

```text
Camera / Video → YuNet (boxes + 5 landmarks) → Quality gate
                                                   ↓
                                      Alignment (112×112 BGR)
                                                   ↓
                                      SFace → unit embedding
                                                   ↓
SQLite templates → cached exact distance → identity aggregation
                                                   ↓
                                      Threshold + Top1/Top2 margin
                                                   ↓
                              Geometry/appearance association + per-track votes
                                                   ↓
                              Identity / Unknown / Pending / Quality rejected
```

默认距离 `d=(1−cosine)/2`，每身份取**距离最近的**至多三个模板的中位数，不是最近拍摄的三张。共享接受条件为 `d1≤τ` 且（有第二身份时）`d2−d1≥m`。当前 SFace 默认 τ=0.275、m=0.04 是工程初值，离线标定不会自动覆盖它们。

## 当前证据

- 最新代码级验收为 **454 项回归通过**（此前 449 项，加上 5 项阈值浮点边界回归）；包含协议泄漏检查、完整 168 组合合成矩阵、冻结参数不受 test 标签影响、时序配对回归、视频采样完整性/身份macro、服务/跟踪六阶段计时、批量精确检索差分、独立 1:1 校准及可插拔模型隔离；另有采集阶段归因与人员连通分组统计。详见 [最终复验](FINAL_AUDIT_V2.md)。
- MeGlass 混合协议补齐 K=5 后完成 168 组合探索；原 126 组合未改变，原主比较仍为 Coverage 13/24、First-K 14/24，未胜过基线。质量消融 full 14/24、none 19/24，说明该样本中存在画质误拒代价，不构成现场提升结论。
- 针对逐身份聚合瓶颈，加入等模板数量分组批量计算；同机同库配对基准中，1,000 合成身份×3 模板检索 p50 为 9.145→0.274 ms，六组负载 1,200 次查询完整分数和排名一致。不是准确率或相机 FPS 提升，见 [性能与回归证据](face_research/P2_RETRIEVAL_RESULTS.md)。
- 10/100/1000 合成身份库的检索、静态单图及重复双图负载已有分阶段基准；不是实际多人运动视频或相机 FPS。
- 公开 1:1 戴镜／摘镜验证完成：验证／测试各 200 人，test 接受 88/89 可用同人配对，但按全部同人尝试为 88/200；111 组采集失败不能隐去。可用 pairs 描述性 AUC=0.99285，0/44 可用异人误接的 Wilson 上界仍约 8.03%，详见 [完整分母与协议](face_research/VERIFICATION_RESULTS.md)。
- 同批失败诊断已完成：600组配对分数/状态与原基线一致，test的111组失败细分为49组无有效检测、53组画质拒绝、9组兼有；按共享人员整组重采样，全部genuine接受率的描述性区间为37%–51%，不把零误差bootstrap当安全保证。[失败归因与分组统计](face_research/ACQUISITION_DIAGNOSTICS.md)。
- 可选 SFace / ArcFace-MBF 对照已完成：两个模型分别在 validation 校准，同一 test 全部 genuine 接受为 88/200 与 89/200；二者共有 111 次 genuine 采集失败，不足以支持切换线上模型。[接口、授权与完整对照](face_research/MODEL_COMPARISON_RESULTS.md)。
- 用户已取消真人采集，不作为当前交付门槛；真实视频泛化、真实跨会话／低 FAR 保证和活体检测均不宣称完成。
- 公开多人视频路线已取得 LTFT 作者标注并完成元数据预检，未取得并解码源视频，也未验证独立录入来源；标注人数不冒充识别成绩。见 [数据与视频评估状态](face_research/VIDEO_VALIDATION_STATUS.md)。

完整分母、置信区间、线程限制、复现命令和 run ID 见 [P1 实验记录](face_research/P1_RESULTS.md)。公开协议是裁剪图、预筛身份、缺批次元数据且重复使用的探索数据，不能声称封存盲测。未知零误接的小样本上界仍很宽。

最新统计更正：旧开放集阈值扫描误把未知采集失败计作正确拒识，已修复并复跑 168 组合＋四种质量门控；全部选样、τ/m 和原有识别计数不变。旧扫描的 `accuracy_all_attempts` 不再引用，新旧产物与影响见 [指标更正](face_research/METRIC_CORRECTION.md)。

## 运行与复现

在仓库根目录，使用 Python 3.10+ 且支持 Tk 8.6+ 的环境：

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

开发测试另安装测试依赖：

```bash
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements-dev.txt
face_compare_system/.venv-ui/bin/python -m pytest -q \
  face_compare_system/tests face_research/tests -p no:cacheprovider
```

测试不需要私人相机或真人照片。新机器安装与固定模型下载见 [应用说明](face_compare_system/README.md)；公开数据需另按来源许可获取，不随代码分发。

## 代码与阅读入口

| 范围 | 入口 |
| --- | --- |
| 检测、对齐、特征 | [deep_engine.py](face_compare_system/face_compare/deep_engine.py) |
| 业务、质量、存储 | [service.py](face_compare_system/face_compare/service.py)、[quality.py](face_compare_system/face_compare/quality.py)、[vector_database.py](face_compare_system/face_compare/vector_database.py) |
| 统一聚合/开放集公式 | [aggregation.py](face_compare_system/face_compare/aggregation.py)、[decision.py](face_compare_system/face_compare/decision.py) |
| 多人时序与评估 | [tracking.py](face_compare_system/face_compare/tracking.py)、[video_evaluation.py](face_compare_system/face_compare/video_evaluation.py) |
| 预算选样与实验 | [selection.py](face_research/selection.py)、[研究说明](face_research/README.md) |
| 简历与追问 | [RESUME.md](RESUME.md)、[INTERVIEW.md](INTERVIEW.md) |
| 改造前审计与当前计划 | [PROJECT_AUDIT.md](PROJECT_AUDIT.md)、[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) |
| 持续目标核验 | [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md)：逐项证据与尚未完成的工作 |

研究输出使用临时 SQLite，不碰现有人员库。原图、向量、人员姓名与逐帧日志不应上传公开仓库；当前归档为软删除，不是不可恢复擦除。项目不支持活体防伪，不用于高风险身份认证决策。
