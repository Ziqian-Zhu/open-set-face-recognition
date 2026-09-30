# SFace / ArcFace-MBF 受控对照

日期：2026-09-26。Run：`results/sface-arcface-verification-20260926-01/`。这是复用公开 MeGlass 配对协议的**探索性 1:1 对照**，不是新盲测、官方基准、现场身份认证验证或损失函数优劣证明。主程序仍默认 SFace，未修改在线阈值、质量规则、UI 或现有数据库。

## 接口、授权与模型固定

`face_compare/embedding.py` 定义结构化 `FaceEmbedder` 接口（`align/extract/distance`、维度、标签、签名）；`FaceComparisonSystem(..., embedder=...)` 是显式注入入口。省略参数时仍构造原来的 `SFaceExtractor`，其旧签名完全保留。注入只允许 YuNet 五点路径，LBPH 不接受；配置阈值不会自动适配其他模型，故目前只从离线研究入口使用 ArcFace，不在 UI 增加切换按钮。

`face_research/arcface.py` 只接受固定版 `w600k_mbf.onnx`，来源为 [InsightFace 官方 Buffalo_SC 包](https://github.com/deepinsight/insightface/releases/tag/model-zoo)。[官方模型指南](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md#model-licenses)明确权重仅限非商业研究；不能把代码 MIT 当成模型可商用许可。本轮仅下载轻量包、提取 recognition 权重作本地研究，未引入属性识别、活体模型或外部人脸服务。

| 项目 | 固定值 |
| --- | --- |
| 官方 ZIP SHA256 | `57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72`（与发布页一致） |
| ONNX SHA256 | `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f`（从已验证包提取） |
| ONNX 大小 | 13,616,099 bytes |
| 对齐 | 标准 ArcFace 五点模板，全部五点的非反射相似变换最小二乘，112×112、双线性、黑色边界 |
| 输入 | uint8 BGR 对齐图 → RGB NCHW float32 → `(x−127.5)/127.5` |
| 输出 | 512 维 float32，检查有限非零并显式 L2 归一化 |
| 推理 | 现有 OpenCV DNN CPU，无新默认依赖、无自动下载 |

对齐目标和输入公式依据官方 [face_align.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py) 与 [arcface_onnx.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py)。本地检查固定 ONNX 的前部为 Conv/PReLU，没有嵌入 Sub/Mul 输入标准化；无需再猜预处理。对齐实现测试与独立 SVD/Procrustes 闭式解一致。未进行 ONNX Runtime 与 OpenCV 的逐元素数值对照，不宣称跨推理引擎完全相同。

ArcFace 的特征签名包含完整模型哈希及对齐／RGB／归一化版本。数据库仍检查签名和维度，**维度相同但签名不同也拒绝打开**。研究使用每模型各自的临时 SQLite；没有把 SFace 模板转换或混入 512 维空间。研究权重位于已忽略的 `data/models/`，不随代码或简历分发。

## 对照条件

同一 [公开 1:1 协议](VERIFICATION_RESULTS.md)：validation/test 各 200 个互不重叠身份，各 200 genuine＋100 impostor pairs。原始图片、YuNet、质量门控均固定；每模型保留其定义的对齐及预处理，因此比较的是**完整特征路径**，不能把差异只归因于 ArcFace/SFace 损失函数。

`model_comparison.py` 先对两个模型分别在 validation 按经验 FAR≤5% 校准，全部阈值写入冻结凭据后才访问 test 像素。阈值规则同前：最大化 genuine 接受，并列时更少 impostor 误接、更小阈值。模型／代码／配置／清单在冻结前后校验，任何变更拒绝发布。test 已在前一轮使用，此次不称首次独立盲测。

## 实际结果与不能省略的分母

| 指标 | SFace | ArcFace-MBF |
| --- | ---: | ---: |
| 维度 | 128 | 512 |
| 模型大小（bytes） | 38,696,353 | 13,616,099 |
| validation 冻结 τ | 0.3561805487 | 0.4072497189 |
| validation genuine 接受／可用 | 93/94 | 94/94 |
| validation impostor 错接／可用 | 0/46 | **2/46** |
| test genuine 接受／可用 | 88/89 | 89/89 |
| test genuine 接受／全部尝试 | **88/200** | **89/200** |
| test genuine 采集失败 | 111 | 111 |
| test impostor 错接／可用 | 0/44 | 0/44 |
| test impostor 采集失败 | 56 | 56 |
| test 可用配对描述性 AUC | 0.992850 | 1.000000 |
| 单图提取 p50 / p95（ms） | 6.072 / 7.205 | 5.521 / 6.375 |

按相同配对哈希逐项核对：ArcFace 多接受 1 个 genuine、少接受 0 个；impostor 错接变化为 0。二者都有 133 个可用 test pairs、167 个采集失败 pairs，**换特征模型没有解决这些前端失败**。SFace 的 validation/test 全部记录、分数与先前 `meglass-verification-20260926-01` 完全一致，证实默认路径没有因接口改造漂移。

两个 validation 工作点并非实际 FAR 完全相同（0/46 对 2/46），只是满足同一个经验上限；不能以 test 多 1 组就断言 ArcFace 普遍更强。test 的 0/44 误接 Wilson 95% 上界仍约 8.03%，且无会话信息，两模型的 TAR@10⁻²/10⁻³ 都是 **insufficient samples**。`AUC=1` 仅是这个小型、通过质量门控的子集观测，绝不表示总体 100% 准确。

计时环境：macOS 14.3.1 arm64，Python 3.12.14，OpenCV 4.14.0，NumPy 2.5.3，OpenCV 报告线程数 8。每模型用固定灰色合成对齐图暖机 5 次，随后记录验证＋测试共 537 个合格图像的实际特征提取；两模型按固定顺序运行，并非交替配对微基准。计时包含 blob／模型／L2，不含检测、对齐、读取、检索、UI，不可与其他 run 的阶段口径拼成加速倍数或现场 FPS。

## 复现与产物

先按上游许可自行从固定官方链接取得包；核对 ZIP SHA256 后仅提取 `w600k_mbf.onnx` 至私有目录。适配器会再次核对模型字节数和 SHA256，未找到或不匹配会报错，不会联网修复。已有数据清单的生成说明见 [1:1 协议](VERIFICATION_RESULTS.md)。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.model_comparison \
  --validation face_research/data/meglass/pairs-cross-condition-v1-seed20260926/pairs.validation.json \
  --test face_research/data/meglass/pairs-cross-condition-v1-seed20260926/pairs.test.json \
  --arcface-model face_research/data/models/w600k_mbf.onnx \
  --noncommercial-research --allow-missing-sessions --target-far 0.05 \
  --output face_research/results/sface-arcface-verification-20260926-01
```

输出目录不覆盖，重跑须换 run ID，不能借此称 test 未使用。十项产物包括配置、冻结凭据、指标 JSON/CSV、匿名哈希 pair CSV、两模型阈值扫描、原始提取时延 CSV、两张 ROC 及合并阈值曲线；`artifacts.json` 校验全部文件，`.audit/` 记录冻结和 test 访问。

**343 项全量测试通过**。新增测试覆盖五点变换、RGB/数值范围、输出维度/NaN/零向量、许可确认、哈希拒绝、签名隔离、旧 SFace 构造、两模型全部冻结后才能读 test、test 标签不改变阈值、冻结后篡改拒绝、失败分母与原子产物。默认真实 SFace 模型与视频连通性复测通过；其视频 smoke 仍是公开静态图重复帧，不算真实运动效果。

结论：可插拔接口和受限研究模型对照已落地；证据不足以切换默认模型或宣称鲁棒性提升。更值得后续诊断的是大量共同的检测／画质采集失败，以及真实多人运动视频场景（真人采集仍取消）。
