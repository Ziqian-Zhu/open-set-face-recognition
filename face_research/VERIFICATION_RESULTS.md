# 公开戴镜／摘镜 1:1 验证（探索性）

日期：2026-09-26。Run：`results/meglass-verification-20260926-01/`；同名 `.audit/` 保留开始、validation 冻结、test 访问及完成凭据。没有改线上阈值、质量规则、UI 或人员库，也没有打开摄像头。

## 固定协议

- 数据：本地已有的 MeGlass 120×120 裁剪图，沿用来源的研究／教学／非商业使用限制，不重新下载、不上传原图或向量。
- `datasets/meglass_pairs.py` 仅读取元数据和文件是否存在，不做像素、画质或识别分数预筛。seed=`20260926`；排除此前 28 份 validation/test 清单涉及的 255 个身份及源照片编号。
- validation/test 各 200 人、400 张图，身份与源照片编号互斥；每人固定一张黑框眼镜和一张无眼镜图。每组 200 个 genuine pairs；相邻且互不复用的两个身份产生一个 impostor pair，共 100 个。每个身份只出现在一个 impostor pair 中，但图像同时参与 genuine，因此两类统计相关。
- 不伪造拍摄会话：源照片 ID 不是 session ID。显式 `--allow-missing-sessions` 配合清单 `session_metadata=unavailable`；严格模式仍为默认值，公开模式不生成独立 attempt ID，低 FAR 结果强制保留样本不足。
- validation 只在 `[0,1]` 的距离事件上选 τ，经验 FAR≤5% 时最大化 true accepts；并列先更少 false accepts、再更小 τ。冻结凭据写入后才读取 test 像素；test 清单哈希已提前绑定。没有在测试集重选工作点。
- 保留生产检测和完整画质门控，失败单独列出；精确解码／对齐重复、同图改身份／会话、重复或交换 pair、身份泄漏、运行期间文件变动均拒绝。

这不是 MeGlass 官方基准、独立跨会话实验或第三方封存盲测。只排除了既有清单中的身份，不能证明历史预筛阶段从未查看过其他候选；也无法排除预训练重叠、近重复、源标签同人多 ID。当前协议生成时未查看本轮测试识别结果；重跑后仍须声明 test 已使用。

## 完整分母与结果

冻结半余弦距离 τ=`0.3561805486679077`。它是本次 1:1 协议工作点，不可直接替代多人 1:N 的线上 τ=`0.275`。

| 统计 | validation | test |
| --- | ---: | ---: |
| genuine 总尝试 | 200 | 200 |
| genuine 可用 | 94 | 89 |
| genuine 检测／画质等采集失败 | 106 | 111 |
| genuine 接受 | 93 | 88 |
| genuine 可用但拒绝 | 1 | 1 |
| impostor 总尝试 | 100 | 100 |
| impostor 可用 | 46 | 44 |
| impostor 采集失败 | 54 | 56 |
| impostor 错误接受 | 0 | 0 |
| TAR／可用 genuine | 93/94（98.94%） | 88/89（98.88%） |
| 接受数／全部 genuine 尝试 | 93/200（46.5%） | 88/200（44.0%） |
| 描述性 ROC AUC／可用 pairs | 0.999306 | 0.992850 |

test 的 0/44 可用 impostor 误接，其 Wilson 95% 上界仍约 **8.03%**；即便把独立性当近似也不足以证明低风险。TAR 的 Wilson 描述性区间约 [93.91%, 99.80%]，同样不覆盖数据偏差与相关性。`TAR@FAR=10⁻²/10⁻³` 均为 **insufficient samples**，不能用 ROC 插值冒充足量证据。

固定工程阈值 τ=0.275 的补充对照为 78/89 可用 genuine 接受（全部 78/200）、0/44 可用 impostor 错接。校准后本批多接受 10 组 genuine；这不是模型变强，也不说明全人群误接风险保持不变，更不是 1:N 鲁棒性提升。

关键发现：本协议有 111/200 genuine 无法进入可用配对评价，**条件式高 AUC 不能替代端到端表现**。本页原始产物将检测／画质／提取失败合并计数。后续已按固定事后诊断协议保留同一批图片和原阈值，逐项核对600组分数/状态不变：test的111组失败为49组只有无有效检测、53组只有画质拒绝、9组兼有；并补上共享人员连通分组统计。见 [完整失败归因与复现](ACQUISITION_DIAGNOSTICS.md)。原run没有覆盖，不换图追高分。

## 工程交付与复现

新增 `verification.py`、`evaluation/verification_calibration.py`、`datasets/meglass_pairs.py`；扩展 `evaluation/pairs.py`，不改变原严格调用方。输出八项文件及 `artifacts.json`：配置、冻结凭据、指标 JSON/CSV、匿名哈希配对 CSV、阈值扫描 CSV、ROC 与阈值曲线。无原图、embedding、源姓名或本地图片路径。

```bash
# 创建新协议目录；输出同名会报错，不覆盖已有协议。
face_compare_system/.venv-ui/bin/python -m face_research.datasets.meglass_pairs \
  --metadata face_research/data/meglass/meta.txt \
  --images face_research/data/meglass/MeGlass_120x120 \
  --output face_research/data/meglass/pairs-cross-condition-v1-seed20260926 \
  --subjects-per-split 200 --seed 20260926 \
  --exclude-manifests face_research/data/meglass/protocol-*/*.validation.json \
  face_research/data/meglass/protocol-*/*.test.json

# 已生成协议则直接使用；重跑需新的 run ID，并承认 test 已使用。
face_compare_system/.venv-ui/bin/python -m face_research.verification \
  --validation face_research/data/meglass/pairs-cross-condition-v1-seed20260926/pairs.validation.json \
  --test face_research/data/meglass/pairs-cross-condition-v1-seed20260926/pairs.test.json \
  --output face_research/results/meglass-verification-20260926-01 \
  --target-far 0.05 --allow-missing-sessions
```

首次 run 的 validation 清单 SHA256：`42ee6d2d4b1ae8df796448c8986a857e704d9c719a41f8feadf7b06da3f57994`；test：`2e38ceb7977f79c786414543013415cc70e8575abcf895d4b6f28f2acea39143`。新增既有排除清单会改变未来重建结果，应以记录的 28 份清单哈希为准，不能把不同版本结果混合。

全量 **319 项测试通过**；覆盖 strict 默认、缺会话显式声明、低 FAR 禁止推断、身份／图像隔离、输入变更、validation-only 校准、test 标签扰动、协议目标绑定、失败分母、原子发布与篡改检测。此次八项产物校验通过，两张曲线已查看。真实摄像头验证与实际多人运动视频依旧未测。
