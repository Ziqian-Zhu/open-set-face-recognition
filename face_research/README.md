# 人脸模板选样实验版

这是课程程序旁边的**独立研究原型**：不改 `face_compare_system` 的 UI、默认识别策略或原有人脸库；研究端与线上端现已复用同一模板聚合、开放集接受公式。它回答一个可检验的问题：**每人只能保存 K 张脸时，怎样兼顾图像质量、不同外观的覆盖和身份一致性，使戴/摘眼镜等条件下的开集识别更稳？**

目前已经实现算法、对照实验、泄漏检查和自动化测试，并在公开 MeGlass 裁剪版上完成初步探索。**尚无独立、获同意的真人跨批次数据，也没有证据表明新选样规则提升了准确率。**本项目没有发明 YuNet、SFace 或全新的学术范式；贡献是受约束的选样规则与可复现的本地评估协议。

P0 最新状态见 [P0 验收清单](P0_ACCEPTANCE.md)：256 项回归通过；固定主比较、冻结凭据、测试访问日志与产物校验已接入矩阵 CLI。软件验收通过不等于真人效果已验证。

P1 后全套为 **263 项测试通过**。[P1 结果与限制](P1_RESULTS.md)记录混合协议的质量/间隔消融、10/100/1000 身份静态合成负载基准，以及单帧/多帧配对评估；另见根目录 [简历表述](../RESUME.md) 和 [面试 30 问](../INTERVIEW.md)。

精确检索收尾阶段为 **298 项测试通过**：公开矩阵补齐 K=5 至 168 组合，原 126 组合保持不变；新增同库配对精确检索优化（1000 合成身份×3 模板 p50 9.145→0.274 ms）。实现、复现与边界见 [精确检索与全预算回归](P2_RETRIEVAL_RESULTS.md)。P1 历史结果不覆盖；完整目标的剩余项见 [逐项核验表](../COMPLETION_AUDIT.md)。

随后公开 1:1 验证完成，全套增至 **319 项测试通过**：验证／测试各 200 人，固定元数据抽样、排除旧清单身份、validation-only 阈值校准；测试接受 88/89 可用同人配对，全部同人尝试为 88/200。缺会话与采集失败保留，不冒充低 FAR 或相机结论，见 [1:1 协议、指标与复现](VERIFICATION_RESULTS.md)。

模型接口与对照阶段，全套为 **343 项通过**。SFace / ArcFace-MBF 各自先在 validation 校准，再读取同一 test；全部 genuine 接受 88/200 与 89/200，共有 111 次 genuine 采集失败，未据此替换线上模型。可选适配器使用固定哈希、独立签名与临时库，权重仅限非商业研究。见 [模型对照及复现命令](MODEL_COMPARISON_RESULTS.md)。

截至视频评估阶段，全套 **449 项通过**：视频评估已有完整采样网格与非法字段检查、按身份等权统计、缺失会话保留，以及 LTFT 标注的严格预检；实际服务/跟踪调用的六阶段计时、预热隔离及真实模型等价检查已通过。随后增加 5 项阈值浮点边界回归，当前代码级最终验收为 **454 项通过**，见 [最终复验](../FINAL_AUDIT_V2.md)。作者标注已取得；源视频下载、跨序列身份映射与独立 gallery 仍未满足，不报告真实运动效果。见 [公开视频状态与复现](VIDEO_VALIDATION_STATUS.md)。

新增 [采集失败诊断与人员分组统计](ACQUISITION_DIAGNOSTICS.md)：不改配置/模型/清单/阈值，单次观察正式采集路径，复核600组配对结果完全一致；解释原111组genuine失败，并提供共享人员连通分组bootstrap与零误差退化保护。没有按失败结果补图或调线上门槛。

最新更正：开放集扫描准确率不再把未知采集失败当成正确拒识；旧 `accuracy_all_attempts` 弃用，新旧 168 组合＋四门控复跑保持阈值、选样和原有计数不变。未知未接受（含失败）与成功拒识使用不同字段，详见 [指标更正](METRIC_CORRECTION.md)。

**当前路线：用户已取消真人采集，不再作为交付门槛。** 本项目继续进行公开数据消融、静态/合成负载性能测量和时序评估工程验证。本文的真人采集工具保留为可选功能，不代表当前待办；不把未采集的真人数据或合成负载包装成现场识别效果。

## 使用公开 MeGlass 做戴镜双向探索实验

[MeGlass 作者仓库](https://github.com/cleardusk/MeGlass)提供 `meta.txt`、戴镜标签和原图/120×120裁剪图下载入口。作者标注的 `1` 特指**黑框眼镜**，不是所有眼镜或镜片类型。先按来源条款合法获取并**解压在私有目录**；本项目的本机私有目录已有裁剪版，但仓库不附带、复制或自动下载人脸图片。其图像来自 MegaFace，使用须遵守[来源数据的非商业研究/教学条款](https://megaface.cs.washington.edu/dataset/download.html)，不要把人脸照片或向量提交到仓库，也不能因作者代码仓库标 MIT 就推定图片可商用。

下列命令按固定随机种子和身份/源照片编号生成四份清单：黑框眼镜录入→无眼镜识别及反向协议，各自有 validation/test。两方向共用同一批已知身份，未知验证/测试身份互斥；每位已知人员每种外观需要 `K_max+1` 张录入图、1 张验证及 1 张测试。`--max-budget 3` 要求每种外观至少 6 个**不同来源照片编号**；若不足，降低预算或人数，不偷偷补同图裁剪。`--gallery-preflight` 在固定验证/测试照片后，**只对录入候选做模型与质量检查**，执行生产版单图门槛、同人一致性与跨人冲突检查。另对已选照片跨 gallery/validation/test 的同一身份执行 dHash 近重复排查，疑似重复者整人剔除；这一环节只比较图像相似性，不用模型分数、画质或正确率，但仍会改变身份分布。两种筛选都会偏向特定人群，不能当随机总体。生成的清单和协议摘要留在已忽略的 `face_research/data/`，不复制图片。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.meglass \
  --metadata face_research/data/meglass/meta.txt \
  --images face_research/data/meglass/MeGlass_120x120 \
  --image-kind cropped --gallery-preflight \
  --known-people 12 --unknown-validation-people 20 \
  --unknown-test-people 20 --max-budget 3 --seed 42 \
  --output-dir face_research/data/meglass/protocol-reproduction-001

face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/meglass/protocol-reproduction-001/glasses_to_no_glasses.validation.json \
  --test face_research/data/meglass/protocol-reproduction-001/glasses_to_no_glasses.test.json \
  --budgets 1,2,3 --target-fpir 0.05 \
  --allow-missing-session-ids \
  --output face_research/results/meglass-g2n-reproduction-001
```

再用 `no_glasses_to_glasses.validation.json` / `.test.json` 跑反方向，输出换新目录；也可对两方向分别运行 `--quality-ablation`，统计 probe 被画质门控拒绝的代价。**这些是本项目自定义的开放集协议，不是论文四组官方结果的复现**；其中黑框眼镜录入→无眼镜识别是为本项目问题新增的反向协议。清单中的 `First-K` 是固定种子的随机原始顺序基线，并非真实拍摄时间的前 K 张。MeGlass 不提供可靠拍摄批次标签，转换器故意不填 `session_id`，所以只能用 `--allow-missing-session-ids` 跑探索性评估，不能声称跨拍摄批次独立。源照片编号去重也不能证明时间/地点独立。**协议构建阶段为排查近重复而查看过测试图像像素，因此不是完全封存的盲测**；但没有用测试识别分数或正确率选择阈值。两个单向协议的 gallery 各只有一种眼镜外观，所以只能检验跨外观识别，不能直接验证“混合戴/摘镜模板覆盖”这一核心选样假设。120×120裁剪版可能被 YuNet 或质量门槛拒绝；完整检测和画质结论优先用原图，两种输入不要混成同一项结果。公开数据不能替代老师要求的真人内置相机现场演示。

本机已在裁剪版上完成固定种子 `42`、12 名已知身份、验证/测试各 20 名未知身份的初步运行。首次诊断发现 2 个身份虽来源照片编号不同，验证/测试图却几乎相同，故**首次结果不用于结论**；重新运行时按固定 dHash≤4 的规则剔除它们，最终清单的跨集近重复警告为 0。最终两方向各有 12 张已知、20 张未知测试照片；K=3、`first` 与 `coverage`、`top3_median` 的各自冻结验证阈值下，黑框眼镜→无眼镜均为 **5/12** 已知正确、**0/13** 可用未知脸误接；反向均为 **9/12**、**0/14**。这里 `0/13` 和 `0/14` 的 Wilson 95% 上界仍约 **22.8%**、**21.5%**，绝不代表“零误接”。两方向的 `coverage` 均**没有超过基线**，但单外观 gallery 不能据此否定混合外观选样方案；裁剪图在测试中分别有 13/32、9/32 张因无脸/画质而不可用。最终预筛检查了 216 张录入候选、拒绝 74 张，在 19 名候选身份中选得 12 名，另有 2 名因近重复被剔除；这构成明显选择偏差。完整计数、各 K/方法和配置快照见私有结果目录 `face_research/results/meglass-g2n-leak-screened-seed42-k3/`、`face_research/results/meglass-n2g-leak-screened-seed42-k3/`。画质消融显示黑框眼镜→无眼镜在不施加光学门槛时由 5/12 变成 8/12，但仅是小样本描述，不能据此放宽线上采集或识别门槛。

## 固定的戴镜＋摘镜混合录入协议

[固定协议](protocols/meglass_mixed_v1.json)在打开混合协议测试结果前写明：12 名已知人员每人录入黑框眼镜 3 张、无眼镜 3 张，验证和测试各留两种外观各 1 张；两组未知人员各 20 名、每组两种外观各占一半。录入图必须通过生产版单张画质、六张组内一致性及跨人员冲突检查。预定主比较为 `K=3`、`coverage` 对 `first`、`top3_median`；验证集目标经验 FPIR 为 5%，测试主要看**所有已知尝试**中的正确识别数，并同时列未知误接及区间。其余预算、方法和聚合方式只作探索性描述，不挑测试集最高值冒充主结果。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.meglass_mixed \
  --metadata face_research/data/meglass/meta.txt \
  --images face_research/data/meglass/MeGlass_120x120 \
  --image-kind cropped --known-people 12 \
  --unknown-validation-people 20 --unknown-test-people 20 \
  --gallery-per-condition 3 --max-budget 3 --seed 42 \
  --output-dir face_research/data/meglass/protocol-mixed-reproduction-001

face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/meglass/protocol-mixed-reproduction-001/mixed.validation.json \
  --test face_research/data/meglass/protocol-mixed-reproduction-001/mixed.test.json \
  --protocol-spec face_research/protocols/meglass_mixed_v1.json \
  --budgets 1,2,3 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --allow-missing-session-ids \
  --output face_research/results/meglass-mixed-reproduction-001
```

已完成的一次运行见 `face_research/results/meglass-mixed-v1-seed42/`：主比较的 `first` 为 **14/24** 已知正确，`coverage` 为 **13/24**，同图配对为新增正确 0、失去正确 1；两者在 **13 张可用未知测试脸中均为 0 次误接**，但其 Wilson 95% 上界仍约 **22.8%**。`coverage` 所选 36 张模板中黑框/无眼镜分别为 17/19，`first` 为 21/15；更均衡的外观覆盖**没有**在这批图上换来更高识别率。测试共 44 张 probe，8 张未检出脸、8 张画质拒绝。结果必须附带这些分母；不能宣传为“已提升鲁棒性”。此协议仍存在裁剪图、缺失拍摄批次、录入身份筛选以及 dHash 查看测试像素的限制，**不是完全封存的独立盲测**。

## 独立真人内置相机验证：本轮取消，工具保留

[真人采集固定协议](protocols/live_camera_mixed_v1.json)和 [本地工具](datasets/live_mixed.py)保留备用，**本轮不执行；尚未得出真人结果**。若未来重新启用，可先做 `pilot`：3 名已知人员和验证/测试各 2 名**不同**未知人员，共 7 人。每名已知人员在一次 gallery 批次拍戴镜 3 张＋摘镜 3 张；在另外两个真实、不同的拍摄批次各拍验证和测试的戴/摘镜各 1 张。尽量分不同日期、光照或背景，绝不能从一段连续视频中抽帧并分别命名为三个批次。每组未知人员各有 1 名戴镜、1 名无眼镜，各拍 1 张。`pilot` 仅检查流程，不能宣传准确率或低 FPIR；扩展版要求 12 名已知＋20/20 名不同未知人员。

在工作区根目录创建私有采集索引。先征求每人的真实知情同意，说明本地保存、研究用途和删除方式；CSV 中的 `yes` **不能替代同意本身**。照片留在已忽略的 `face_research/data/`，不要提交或上传。采集命令只会使用电脑内置相机：预览为镜像方便构图，按空格保存**未镜像、未裁剪的原始帧**，Esc 停止；不会运行模型筛选测试帧，也不会覆盖已有文件。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.live_mixed template \
  --profile pilot --output face_research/data/live/pilot.csv

face_compare_system/.venv-ui/bin/python -m face_research.datasets.live_mixed capture \
  --index face_research/data/live/pilot.csv \
  --subject-id known_001 --split gallery \
  --session-id known001-day1 --consent-confirmed
```

按 CSV 提示逐人完成其他批次；`known_001` 的 validation/test 必须分别使用**不同于 gallery 且彼此不同**的真实 session ID。也可以用电脑相机应用自行拍摄并把照片放到 CSV 指定的路径，再填写 `session_id` 和 `consent_confirmed=yes`。所有图像到齐后执行：

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.live_mixed build \
  --profile pilot --index face_research/data/live/pilot.csv \
  --output-dir face_research/data/live/pilot-manifests-001

face_compare_system/.venv-ui/bin/python -m face_research --matrix \
  --validation face_research/data/live/pilot-manifests-001/mixed.validation.json \
  --test face_research/data/live/pilot-manifests-001/mixed.test.json \
  --protocol-spec face_research/protocols/live_camera_mixed_v1.json \
  --budgets 1,2,3 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --output face_research/results/live-pilot-001
```

清单生成器只验证匿名编号、人数、同意勾选、路径和批次元数据，不打开验证/测试图像像素；实验入口再检查重复图像和输入有效性，且只在 gallery 选样与 validation 定阈值后读取 test。程序无法证明 CSV 填的确实是独立拍摄批次，也无法替代参与者同意记录。如果真人测试失败，应如实记录检测/画质失败和分母，不能看过测试结果后换帧重测并继续称其为首次盲测。

随时检查缺哪些照片、同意确认和批次，不会打开相机或读取照片像素：

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.live_mixed status \
  --index face_research/data/live/pilot.csv --profile pilot
```

## 方法与对照

所有方法使用相同的 YuNet 检测、五点对齐和 SFace 向量。旧命令固定预算 K、最近 3 张样本距离的中位数与现有第二名身份间隔；新 `--matrix` 命令同时比较 K、聚合策略，并仅在验证集联合选择距离阈值和间隔。距离为 `d=(1−cosine)/2`。

| 名称 | 选样规则 |
| --- | --- |
| `first` | 清单中的前 K 张，模拟无策略录入 |
| `random` | 固定种子的 K 张 |
| `quality` | 手工图像质量代理分最高的 K 张 |
| `diversity` | 从最高质量开始，之后选离已选模板最远的样本 |
| `coverage` | 本实验方法：最高质量起步，贪心选能覆盖更多录入外观且较可靠的模板 |

`coverage` 对每个合格样本计算质量分 `q`（清晰度、对比度、人脸面积、曝光），再按向量距离计算局部相似度 `s=max(0,1−d/0.25)`。身份一致性软分 `c=clip(1−median(d_同人其他样本)/(2×录入一致性距离),0.25,1)`，并以 `q×c/(0.2+局部密度)` 加权。每轮增加覆盖收益，再加一个固定的 `0.05×q×c` 质量偏好。课程程序原有的画质门禁与同一人一致性检查依然先执行；不合格样本不会被“新算法”偷偷纳入。上述常数是预先固定的工程默认值，并非测试集优化结果。

这种做法的假设是：同一人“戴眼镜”和“不戴眼镜”等有效外观变化，不能被大量近重复样本挤掉；但罕见且模糊或身份可疑的样本也不应仅因罕见就入选。假设是否成立，必须由以下独立测试验证。

## 数据与实验协议

1. 征得拍摄者同意。建议至少 3 名已知人员，每人 5–10 张录入图，包含戴/摘眼镜、角度和光照变化；另采集更多**未录入的真实人员**。3 人仅是程序运行下限，远不足以支撑普遍性能结论。
2. 录入图统一放入两份清单的 `gallery`，**像素内容、标签、顺序完全相同**。每人至少有 `max(K)+1` 张才有完整的选样对比意义，默认矩阵包含 K=5，所以每人至少 6 张。录入、验证、测试使用不同时间/地点的拍摄批次，不使用同一段视频的连续帧跨集合；新矩阵实验默认要求每张图有非空 `session_id`，并拒绝同一个人的批次跨 gallery/validation/test 复用。
3. `validation` 的 `probes` 用于确定每种方法的距离阈值；`test` 的 `probes` 仅用于最后一次评估。两组未知人员必须是不同的人，用 `subject_id`（匿名编号）验证。未知人脸的 `label` 写 `null`；已知人员 `label` 必须与录入姓名一致。
4. `condition` 建议明确写 `glasses`、`no_glasses`、`low_light`、`normal` 等，结果会逐条件给出。不要把同一照片调亮、缩放、重新压缩后分别放入验证/测试；程序会硬性拒绝完全相同的解码图像/对齐裁剪，并用 64-bit dHash **提示**相似整帧供人工复核，但不能自动识别所有近重复，也不能拿 dHash 判断是否同一人。

可复制 [验证清单示例](manifest.validation.example.json) 和 [测试清单示例](manifest.test.example.json)，把其中**所有示意路径**换成同意使用的真实照片。图片路径相对于各自清单所在目录；两份示例本身不含数据，直接运行会报“无法读取测试图片”。请把真人照片放在 `face_research/data/`（已忽略）或工作区外的私有目录，不要上传原始人脸照片/向量/结果 JSON 到公开仓库。

在工作区根目录执行：

```bash
face_compare_system/.venv-ui/bin/python -m face_research \
  --validation face_research/manifest.validation.json \
  --test face_research/manifest.test.json \
  --budget 3 --target-fpir 0.05 --seed 42 \
  --output face_research/results/run-001.json
```

`--output` 原子创建新文件，避免覆盖旧实验或留下半份结果。运行会用**临时 SQLite** 进行录入一致性和跨人员冲突检查，不会读写课程程序已有的人脸库；退出后删除临时库。照片逐张解码，并在推理前复核内容哈希，不会把整套照片常驻内存。结果保存配置、模型、研究代码和课程代码哈希、特征签名、各方法所选图片哈希、验证集阈值、测试集总体与条件分组指标、相对 `first` 的同图配对得失，以及 Wilson 95% 区间。

阈值只根据验证集选择：在可用未知人脸的经验误接受率不超过目标值时，最大化已知正确数；平局优先更少误认、误接受和更低阈值。报告还附同一个课程固定阈值下的测试结果。**5% 经验目标不是部署风险保证**：例如验证集只有 10 张未知人脸，即使 0 次误接受，置信区间上界仍然很宽。

该选样实验范围为**静态单人图像**；实时多人跟踪、摄像头端到端延迟、活体检测不在这份实验里。对简历可以描述“设计并实现质量-覆盖-一致性约束选样与开集评估框架”。只有在未用于开发的独立测试协议上得到可复核收益，才按真实数据来源、场景与分母写提升；不一定要求自采，但不能由公开图外推现场。当前没有该收益证据，不调完测试集后仍称其为独立测试。

## 新的预算 × 选样 × 聚合矩阵

`--matrix` 会对 First-K、Random（默认 3 个种子）、Quality、Diversity、Coverage，及 Nearest / 全量 Mean / 全量 Median / Top-2 Mean / Top-3 Mean / Top-3 Median 做对照。默认预算为 `1,2,3,5`；所有组合在读取 test 清单前先完成 gallery 选样与 validation 联合标定。每个组合还独立比较“只用 threshold”与“threshold + 固定 Top1/Top2 margin”，两组各自在验证集重新选 threshold；若某组达不到目标 FPIR，则标为不可比较。测试集报告 Top-1、已知正确接受、已知拒绝/错认、未知 FPIR 等，并保留分母与 95% Wilson 区间。大量组合属于探索性分析，不应在看过测试结果后只挑最好的一行宣称为预注册主结果。

示例清单仍只是路径格式示意；运行矩阵时给每张 gallery、validation、test 图加 `session_id`，同一人跨集合须换拍摄批次。若确实没有批次元数据，可用 `--allow-missing-session-ids` 做**探索性连通测试**，但结果不能声称会话独立。现有示例每人仅 4 张，可先运行 `--budgets 1,2,3`；要测 K=5，需增加每人至少 2 张候选录入图。

```bash
face_compare_system/.venv-ui/bin/python -m face_research \
  --matrix \
  --validation face_research/manifest.validation.json \
  --test face_research/manifest.test.json \
  --budgets 1,2,3,5 --random-seeds 42,43,44 \
  --margins 0,0.02,0.04,0.06 --target-fpir 0.05 \
  --output face_research/results/matrix-001
```

输出目录只允许新建，包含 `experiment_config.json`、`metrics.json`、`metrics.csv`、`threshold_selection.csv`、`roc_curve.png`、`threshold_curve.png`。`threshold_selection.csv` 的 `FAR/FRR/TAR` 在只有 1:N probes 时留空；若提供独立的 validation 验证对，才填入对应阈值下的 1:1 指标。`metrics.csv` 在提供 test 验证对时报告**验证集冻结阈值**下的 test FAR/FRR/TAR。没有单独 1:1 测试对时，`roc_curve.png` 会醒目标注 **NOT EVALUATED**，不是伪造的曲线。

矩阵 CLI 还创建相邻 `<run>.audit/`：先记录开始，再将所有模板选择、threshold/margin 与版本哈希写入 `frozen_validation.json`，随后才记录并打开 test。失败记录保留，重试换新 run ID，不覆盖或删除失败历史。最终产物暂存在临时目录，全部生成后一起发布，另含 `frozen_validation.json`、`primary_comparison.json` 和文件校验表 `artifacts.json`。每个变体的 JSON 保留固定工程阈值下的补充对照；CSV 补充原始分子/分母、Wilson 区间和缺失 verification 指标的原因。冻结记录不存原始图像、向量或姓名，但仍应按研究数据保管。

使用 `--protocol-spec` 时，会在读取 test 前核对主比较、预算/种子/间隔网格（若协议声明）及 FPIR 目标；未指定时明确记为探索性结果。已有两份混合协议均可使用，不能用它们绑定不相符的实验。这些本地记录不是第三方预注册，不能证明研究者没看过旧 test，也不能把本次已重复使用的公开数据说成首次盲测。

若要计算 1:1 verification ROC/AUC，可另提供 `--verification-validation-pairs face_research/manifest.verification.validation.json` 与 `--verification-test-pairs face_research/manifest.verification.test.json`；每对需 `left_image`、`right_image`、两侧匿名 `subject_id`、两侧 `session_id` 和布尔 `genuine`。验证对用于报告验证阈值扫描的 FAR/FRR/TAR；本版 1:N 工作点仍由 validation 的 FPIR/已知正确率选择，而不是偷看测试对。模板选样与阈值/margin 冻结后才读取 test 验证对。低 FAR 的 TAR 只有在唯一 impostor 尝试标识与样本量足够时才给描述性估计，否则为 `insufficient samples`；标识唯一仍不能自动证明统计独立。1:1 FAR 与 1:N FPIR 是不同指标；验证对质量/检测失败单独统计，不算作正确拒绝。测试对可以描述 ROC，但不能反过来选择线上阈值。

独立于 1:N 的 `python -m face_research.verification` 入口现已完成公开 1:1 ROC 探索；与上面的矩阵补充指标不同，它专门在 validation pairs 校准 τ。严格要求 session ID 是默认行为；公开缺会话探索须显式 `--allow-missing-sessions`，清单也须声明缺失；不能据此给出低 FAR 保证。协议构建、完整分母与复现命令见 [1:1 记录](VERIFICATION_RESULTS.md)。质量消融与静态/合成负载计时见 [P1 记录](P1_RESULTS.md)，可选 SFace/ArcFace 对照见 [模型记录](MODEL_COMPARISON_RESULTS.md)，均不等于真实多人视频实测。真人采集不再是本轮门槛，也不能把取消写成通过。

## 画质门控消融：量化误拒代价

`--quality-ablation` 固定同一批**经过线上完整画质与录入一致性检查**的 gallery、First-K 模板和 Top-3 Median 聚合；对同一原始验证/测试 probe 池分别运行 `none`、`brightness_only`、`sharpness_only`、`full` 四种光学画质门控。每种门控都独立只在 validation 选择阈值与 margin，**全部冻结后才打开 test**。检测失败、多脸、无效关键点、对齐/特征提取失败仍单列，`none` 也不把这些失败当成成功识别。结果同时列出门控拒绝数、全部尝试和可用脸分母、条件分层及同图配对得失；缺少可用的验证集已知/未知脸时明确标记 `unavailable`，不借用 test 调参。

```bash
face_compare_system/.venv-ui/bin/python -m face_research \
  --quality-ablation \
  --validation face_research/manifest.validation.json \
  --test face_research/manifest.test.json \
  --budget 3 --margins 0,0.02,0.04,0.06 \
  --target-fpir 0.05 \
  --output face_research/results/quality-001
```

输出为不覆盖旧结果的目录，包含 `metrics.json`、`metrics.csv`、`threshold_selection.csv`、冻结凭据和 `artifacts.json` 校验表；同名 `.audit/` 记录冻结→test 访问→成功/失败。JSON 还包含 `test_at_shared_full_gate_validation_point`，把 full 的 validation 工作点固定给各门控作补充对照。这项实验只回答**识别 probe 的画质门控**取舍，不能用来声称放宽**录入**门槛安全或有益。`FPIR` 是 1:N 开集未知人员误接，不是 1:1 verification `FAR`；CSV 的 `FAR/FRR/TAR` 留空是因为此入口没有成对验证协议。

## 分阶段性能基准

已完成静态/合成负载测量，见 [P1 记录](P1_RESULTS.md)；**未测真实多人运动视频**。可提供单脸图及真实静态多脸图，或用 `--repeat-single 2` 创建明确标注的重复同图负载，二者不能同时指定：

```bash
face_compare_system/.venv-ui/bin/python -m face_research.benchmark \
  --single-image /path/to/consented-single.jpg \
  --multiple-image /path/to/consented-multiple.jpg \
  --gallery-sizes 10,100,1000 --templates-per-identity 3 \
  --iterations 30 --warmup 5 \
  --opencv-threads 1 \
  --output-dir face_research/results/benchmark-001
```

它分别记录 YuNet、画质、对齐、SFace、缓存精确检索与检测至身份排序链路的 p50/p95；保留每次原始计时、机器/构建信息、输入/代码/模型哈希。输出 JSON、CSV、检索规模曲线和校验表，旧 `--output 新文件.json` 仍兼容。FPS 是一个 Python 工作线程的静态处理吞吐估计，不含相机/UI/解码/时序确认；OpenCV 请求线程数可能不生效，必须看 `machine.opencv_thread_control`，BLAS 线程另列。图库由合成单位向量填充，**仅用于测检索规模与耗时，不可用于识别准确率**。向量 payload 不是进程总内存；逐次入库耗时包含缓存重建，之后的 refresh 仅为修订检查。

## 开发检查

代码按边界拆分：`manifest.py` 负责数据清单和泄漏检查，`selection.py` 只做向量选样与匹配，`experiment.py` 负责临时数据库、验证集校准与盲测编排，`_baseline.py` 集中管理对课程项目的依赖。阈值搜索按距离事件扫描，模板向量对每种方法只归一化一次；配对结果按图片哈希核对。

```bash
face_compare_system/.venv-ui/bin/python -m pytest -q face_research/tests -p no:cacheprovider
face_compare_system/.venv-ui/bin/python -m face_research --help
```

## 相关工作与边界

质量感知识别、聚类模板选样和多原型人脸表示已有研究，不宜声称本项目是该方向的“首次发明”。可参考 [早期模板选样研究](https://www.sciencedirect.com/science/article/abs/pii/S0031320305004188)、[Multi-Prototype Networks](https://arxiv.org/abs/1902.04755)、[MagFace](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html) 和 [AdaFace](https://arxiv.org/abs/2204.00964)。后续值得做的是：在足够样本的情况下预注册消融设计，比较是否增加跨人员可分性项，再把经过验证的策略接入实时端；不要在缺少数据时先宣称鲁棒性收益。
