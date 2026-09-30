# 人脸识别项目算法审计（代码现状）

> 本文保留 2026-09-25 的**改造前审计基线**，不代表当前待办。2026-09-26 的 P0 软件验收与尚缺的数据证据见 [P0 验收清单](face_research/P0_ACCEPTANCE.md)；后续目标仍以 [改造计划](IMPROVEMENT_PLAN.md) 为准。

审计日期：2026-09-25。范围：`face_compare_system/` 的应用、识别、数据库、静态/视频评估和测试，以及同级 `face_research/` 的选样实验与测试。本文件描述**已经实现且由代码可证实**的行为，不把工程设计或合成测试当作真实人群效果。此轮只做审计，不改变程序。

## 1. 实际数据流

```text
电脑相机 / 视频文件
  → UI 读取帧（相机预览镜像；视频不镜像；默认每 3 帧提交一次）
  → 单工作线程的最新帧队列，过期/换源结果被丢弃
  → YuNet 在最长边不超过 640 px 的图像上检测全部人脸及五点关键点
  → 原图坐标上的人脸裁剪、画质门控
      ├─ 不合格：保留画质原因，不提取特征；轨迹投票清空
      └─ 合格：五点 alignCrop → 112×112 BGR → SFace 128 维特征 → L2 归一化
                 → 与 SQLite 中每个身份的多个模板计算半余弦距离
                 → 各身份最近至多 3 个距离取中位数并排序
                 → 最优距离阈值 + 第一/第二名间隔判定 Known / Unknown
                 → 多脸轨迹关联、各轨迹独立多帧投票
                 → 稳定身份/未知、叠加画框与 UI 展示
```

证据：[`ui.py`](face_compare_system/face_compare/ui.py)、[`worker.py`](face_compare_system/face_compare/worker.py)、[`service.py`](face_compare_system/face_compare/service.py)、[`deep_engine.py`](face_compare_system/face_compare/deep_engine.py)、[`recognizer.py`](face_compare_system/face_compare/recognizer.py)、[`tracking.py`](face_compare_system/face_compare/tracking.py)。主配置是 [`config.json`](face_compare_system/config.json) 的 `sface` + `sqlite`；`Haar + U2-LBPH` 是保留的基线分支，不是当前默认路径。

| 环节 | 当前实现与边界 |
| --- | --- |
| Detection | `YuNetDetector.detect()` 用 OpenCV `FaceDetectorYN`；检测分数默认 `0.9`，按原图还原框与五点，过滤小于 `80 px` 的脸和眼距小于 `8 px` 的结果，按面积排序。不是跟踪器，也不保证遮挡/侧脸检出。 |
| Alignment | 合格脸必须带五点，调用 `FaceRecognizerSF.alignCrop()`；SFace 输入为 `112×112×3` BGR，缺关键点不能绕过对齐。 |
| Quality filter | `FaceQualityAssessor` 检查曝光、对比度、脸面积、关键点是否在画面内，以及分区清晰度；当前 `regional` 算法把脸缩放到 `128×128`，三带中的中位数抑制单个眼镜/头发高频区域的干扰，并做近似噪声补偿。阈值是工程默认值，不是经人群数据校准的“可识别概率”。画质拒绝发生在特征提取之前。CLI 的显式 `allow_low_quality` 可绕过**录入**画质门控，实验要记录是否使用。 |
| Feature extraction | SFace ONNX 提取 128 维向量；代码检查维度、有限数和非零范数，然后**显式 L2 归一化**。不能声称原模型输出已经归一化，保证来自本项目后处理。 |
| Template storage | `SQLiteVectorDatabase` 保存人员、特征、画质元数据和可选 JPEG；校验模型签名/维度，录入事务提交，有带版本的内存检索缓存；删除人员是归档/软删除，并非物理擦除。录入前还有单人脸、样本间一致性与跨身份冲突检查。 |
| Matching | 对每人所有模板做精确向量检索，取最近 `min(3,N)` 个距离的中位数形成该身份分数，再在身份间排序；不是单模板最近邻，也不是取全部模板的中位数。文件库基线走相同的“最近 K 中位数”规则。 |
| Open-set decision | 最优身份距离必须过阈值，且若存在第二名，前两名间隔须足够大；否则返回 Unknown（含“阈值超限”或“歧义”原因）。单身份库不存在第二名，margin 条件自然不生效。 |
| Temporal confirmation | `MultiFaceTracker` 给每条轨迹独立创建 `MultiFrameVoter`。默认最多保留 7 次结果、至少 4 次、赢家比例至少 `0.6`，且赢家必须与当前及上一帧一致；未稳定时 UI 显示确认中。 |

## 2. 距离、阈值、相似度的精确定义

在 SFace 路径，令归一化向量为 `x,y`，`c = x·y` 为 cosine similarity；代码使用的“距离”为：

```text
d = (1 - clip(c, -1, 1)) / 2      ∈ [0, 1]，越小越像
c = 1 - 2d
‖x-y‖²₂ = 2 - 2c = 4d        （仅对单位向量成立）
```

依据：[`deep_engine.py`](face_compare_system/face_compare/deep_engine.py) 的 `extract()` / `distance()`，以及 [`vector_database.py`](face_compare_system/face_compare/vector_database.py) 的矩阵检索。OpenCV [YuNet + SFace 官方教程](https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html) 使用 `FaceRecognizerSF_FR_COSINE` 返回**余弦相似度（越大越像）**，也展示 `FR_NORM_L2`；教程的 `0.363` 是其示例数据集的余弦相似度阈值，不能直接与本项目的 `0.275` 比大小，更不能当成此项目校准结果。

主配置的 `engine.cosine_threshold=0.45` 在 [`service.py`](face_compare_system/face_compare/service.py) 转成 `τ=(1-0.45)/2=0.275`；`engine.cosine_margin=0.08` 转成距离间隔 `m=0.04`。当前实际判定（等号接受）是：

```text
d_i = median(smallest min(3, N_i) distances to identity i)
i*  = argmin_i d_i,  d₁ = d_i*,  d₂ = second smallest identity score
Known(i*) ⇔ d₁ ≤ τ  且  (不存在 d₂ 或 d₂ - d₁ ≥ m)
否则 Unknown；阈值超限和前两名过近分别给出拒绝理由。
```

这意味着 `d=0.275` 等价于 `c=0.45`，若按单位向量欧氏距离表示则约为 `1.049`。`RecognitionResult.similarity` 的 `(1-d)×100` 只是 UI 展示分数，代入上式是 `50×(1+c)`；**不是**余弦相似度百分比，更不是身份正确概率。SFace 的 `recognition.default_distance_threshold=0.43` 和 `recognition.ambiguity_margin=0.025` 会被 `engine` 对应字段覆盖；同一配置里并存两套数字，易误读。代码尚未全局统一 `cosine_similarity`、`half_cosine_distance`、`display_score` 的命名与单位。若将来改命名，必须保留配置迁移兼容并核对所有 CLI/报告字段。

## 3. 多模板聚合：现状与待验证假设

生产库与研究版分别在 [`vector_database.py`](face_compare_system/face_compare/vector_database.py)、[`recognizer.py`](face_compare_system/face_compare/recognizer.py)、[`face_research/selection.py`](face_research/selection.py) 实现“最近 K 个距离的中位数”；二者有一致性测试，但没有一个可配置、共享的聚合策略接口。`K=3` 时是三个最近值的第二小值；`N=1` 时退化成单模板距离，`N=2` 时两数中位数等于其均值。

| 策略 | 优点 | 主要代价 / 风险 |
| --- | --- | --- |
| `min` / nearest template | 某个模板恰好覆盖眼镜、无眼镜等外观模式时召回高 | 库增大后更容易碰到偶然相近的错误模板，误接受风险随模板数变化；对噪声模板敏感。 |
| 全模板 `mean` | 利用所有证据，平滑单样本噪声 | 模板多的身份、外观多峰及低质旧样本可能被惩罚；需固定每人预算才公平。 |
| 全模板 `median` | 对少量极端模板鲁棒 | 正常的稀有外观模式可能被多数模板压制。 |
| `top-k mean` | 兼顾最近邻覆盖与平滑，可用 K 控制 | 仍受最近的异常模板影响；K 与模板预算一起改变得分分布。 |
| 当前 `top-3 median` | 减弱**一个**偶然最近距离或一个偏远模板的影响 | 两个相近的错误/重复模板足以主导；仅一张有效外观模板时可能拒绝；不存在已证实的性能最优结论。 |

因此“更鲁棒”目前是设计动机，不是已经由独立测试集证实的效果。真正的研究问题是固定相同 gallery、模板预算、训练/验证/测试协议后，比较每种聚合对已知识别、未知误接受及眼镜变化条件的影响；每换一种聚合都要在验证集重选 `τ,m`，否则分数分布不公平。

## 4. 多脸跟踪与时序：按实际代码描述的状态机

[`tracking.py`](face_compare_system/face_compare/tracking.py) 先筛选轨迹—观测边：中心距离不得超过框尺度的 `1.5` 倍；双方有特征时半余弦距离不超过 `0.25`，用 `0.75×特征距离 + 0.25×(1-IoU)/2` 排序；特征缺失时要求 `IoU≥0.3`。若轨迹端或观测端出现距离差小于 `0.025` 的近并列候选，宁可新建轨迹也不继承旧身份；其余按代价贪心一对一关联。超过 `0.9 s` 未更新会删除轨迹，时间戳倒退会重置全部轨迹；最多 32 条活跃轨迹。它是保守的 CPU 基线，不是已验证的 MOT 算法。

代码没有显式 `UNCONFIRMED → CANDIDATE → CONFIRMED → LOST` 枚举，真实状态可概括为：

| 事件 | 轨迹/投票行为 | 对外状态 |
| --- | --- | --- |
| 新检测但未关联 | 新建 track ID、空投票窗口 | 首次合格观测后 `confirming`；超容量则 `capacity`。 |
| 连续合格观测、仍未达法定票数 | 同轨迹更新框/特征/投票 | `confirming`，不能把原始单帧标签当作稳定结果。 |
| 最近窗口某身份或 Unknown 获得法定票数，且当前/前一帧一致 | 设置 `stable_recognition` | `known` 或 `unknown`。 |
| 本帧画质不合格/无特征 | 清空该轨迹投票 | `quality_rejected`，下一次重新确认。 |
| 本帧未匹配到旧轨迹 | 清空旧轨迹投票；一定时间内保留轨迹对象 | 再出现需重新攒票；超过间隔对象被删除，相当于隐式 `lost`。 |
| 两条轨迹稳定声称同一已录入身份 | 两条都撤销稳定结果并清票 | `identity_conflict`。 |
| Known↔Unknown 或身份 A↔B | 新单帧标签不会继承旧结论；赢家须重新满足窗口和最新两帧约束 | 过渡时 `confirming`，稳定后切换；近并列关联时可能直接新建轨迹。 |

对“身份闪烁”：机制能抑制单帧异常，但窗口内仍可能重获不同赢家，且摄像头拥挤、遮挡、交叉时贪心匹配可能断轨；视频评估已有轨迹切换统计，尚未用足量实拍数据证明闪烁降低幅度。UI 超过 1 秒没有新分析会清空旧结果，避免把陈旧身份挂在新画面上。

## 5. 评估与研究代码的真实覆盖

| 项目 | 已有能力 | 仍缺的算法证据 |
| --- | --- | --- |
| 静态开放集评估 | [`evaluation.py`](face_compare_system/face_compare/evaluation.py) 支持 `validation/test` 清单、临时隔离库、精确解码图/对齐裁剪去重；报告已知正确/拒绝/错认、未知误接受（全部尝试与可用脸两个分母）、Wilson 95% 区间、条件分组和粗粒度延迟。 | 没有成对 verification 样本协议、ROC/AUC/FAR/FRR/TAR@FAR；没有统一输出所需曲线与 CSV；静态评估不测多脸时序效果。 |
| 阈值选择 | [`operating_point.py`](face_compare_system/face_compare/operating_point.py) 只允许验证集、固定当前 margin 扫阈值、按经验 FPIR 目标选点，输出建议 `engine.cosine_threshold`，不直接改配置；[`face_research/experiment.py`](face_research/experiment.py) 也对每种选样方法只在验证集选 `τ`。 | 未联合选择 `τ,m`；没有独立的验证 pair ROC；“经验 FPIR 达标”不等于低误接率的统计保证。主程序 SFace 仍使用配置固定阈值，不自动应用研究标定结果。 |
| 有限模板选样 | [`face_research/selection.py`](face_research/selection.py) 已有 First-K、固定种子的 Random、质量、差异性、质量+覆盖+身份一致性五种方法；相同 SFace 特征、预算与匹配公式，按条件输出结果与配对差值。 | 单次命令只评一个 K，Random 只用一个 seed；没有 K=1/2/3/5 全矩阵与置信区间/重复实验；proposed 的常数是人工设定，优于基线尚无可引述的真实数据。 |
| 视频评估 | [`video.py`](face_compare_system/face_compare/video.py) 可导出帧/轨迹结果，[`video_evaluation.py`](face_compare_system/face_compare/video_evaluation.py) 对带真值视频报告检测召回、未匹配检测、身份切换、首次正确所需时间等。 | 尚无独立标注、多人物理场景与拍摄条件覆盖的结果；视频帧相关性使逐帧区间不能视为独立样本置信度。 |
| 性能 | 现有报告有整体单图耗时、研究选样耗时；SQLite 使用向量化精确匹配和缓存。 | 无 YuNet/对齐/SFace/检索/端到端分阶段基准，也未量化 1 人/多人及 10/100/1000 身份规模；暂不能判断 FAISS 是否必要。 |
| 回归测试 | 2026-09-25 运行 `pytest -q face_compare_system/tests face_research/tests -p no:cacheprovider`：**170 passed**。测试覆盖阈值、歧义、投票、轨迹交叉、质量、数据库事务与研究分割校验。 | 这不能替代真实人员、设备、眼镜条件下的准确率测量；聚合策略、联合标定、ROC 计算和分阶段 benchmark 尚未有对应测试。 |

### 数据隔离审计

`face_research` 把 gallery 当作录入/训练侧，在验证和测试 manifest 中要求完全相同且顺序一致；验证 probe 用于选阈值，**测试图像在各方法模板与阈值冻结后才加载**。选样中的质量、覆盖、身份一致性仅使用 gallery 的图像/特征，并未直接读测试 probe；验证/测试未知人员 `subject_id` 交集被禁止；解码图和对齐裁剪精确哈希被检查。这些是已有的反泄漏措施，不能称“完全没有 train/validation/test 隔离”。

但“train”没有独立显式 manifest/采集批次元数据；相同人的连拍、重编码或近重复、同一受试者换名，以及研究者**看过测试结果后手动修改 proposed 常数/K/质量阈值**，都不由当前精确哈希自动发现。只在 manifest 中写不同文件名不等于独立样本。未来须预先冻结协议和超参，按人员/拍摄批次审计，并记录每次测试使用；测试集不能反复用于选方案。已知人员身份在 gallery 与 probe 两侧出现是 1:N 识别实验的正常设计，关键是样本/会话不重合；未知人员的验证/测试身份应不重合。

## 6. 结论：可写进简历的事实与不能写的结论

已可据代码说明：本地 YuNet+SFace 多脸开放集识别系统、五点对齐和显式归一化、质量门控、SQLite 多模板库、距离阈值+歧义间隔、轨迹级时序确认、有限模板选样五策略基线、隔离验证/测试的初版研究框架和 170 项回归测试。

尚**不能**写“误识率降低 X%”“达到 FAR 1e-3”“眼镜条件鲁棒性显著提高”“优于 ArcFace/FAISS”或“自研最优算法”：目前没有足够的独立真人数据、对应协议或对照结果。最有价值的下一步是把现有机制变成**可复现、可反驳**的算法实验，而不是立即换模型；具体执行与风险见 [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md)。
