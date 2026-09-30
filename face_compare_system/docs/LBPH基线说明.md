# 旧版LBPH基线说明（历史文档）

本文件仅记录升级前实现。当前启动、录入和准确性评估请阅读项目根目录README.md；不要按本文的旧界面和自动校准描述操作深度模式。

这是一个面向“智能科学综合实践”的完整、离线友好型课程项目。系统通过外置摄像头采集画面，以 OpenCV 自带的 Haar 级联检测人脸，使用纯 NumPy 实现的多尺度 LBPH 提取可解释纹理特征，完成“已知人员 / 未知人员”的实时判定。

系统不会下载模型，也不依赖 `opencv-contrib`。项目仓库不包含任何真实人脸、特征或运行日志。

> 重要边界：LBPH 适合教学演示、低风险现场比对，不具备活体检测能力，也不应作为门禁、支付、执法等高风险场景的唯一身份依据。

## 功能完成情况

- 外置摄像头扫描、选择、打开、实时采集与主动释放；
- 正脸 + 左右侧脸 Haar 检测，重叠框抑制，CLAHE 暗光增强；
- 录入新人员或为已有人员追加样本，标准库无需改代码即可扩容；
- 多尺度、分网格 uniform-LBP（U2）特征，卡方距离和百分制相似度；
- 标准库内正/负样本对自动校准判定阈值；
- 距离阈值 + 前两名间隔双重拒绝未知/歧义人员；
- 亮度、曝光、对比度、清晰度、人脸尺寸质量门控；
- 7 帧滑动窗口多数投票，降低单帧抖动；
- 深色现代 Tkinter UI，实时框选、结果、质量与数据库状态；
- JSONL 事件日志，不记录原始视频帧或特征；
- 静态图片批量录入/识别、标准库查看、阈值校准 CLI；
- 配置校验、特征/数据库/校准/投票等非摄像头自动化测试。

## 快速运行

要求 Python 3.10 或更高版本。首次安装：

```bash
cd face_compare_system
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

macOS 第一次启动摄像头时，请允许 Terminal、Python 或 Codex 使用相机。外置摄像头通常为索引 `1`，内置摄像头通常为索引 `0`；实际顺序取决于操作系统。界面会扫描索引 0–5，并优先选中 `config.json` 中的 `preferred_index`。

详细演示步骤见 [旧版操作手册.md](旧版操作手册.md)。

## 建议验收流程

1. 连接 USB 摄像头，点击“刷新”，选择标记为“外置优先”的设备并启动。
2. 依次录入至少 3 名人员，每人采集 5 张：正脸、轻微左转、轻微右转、两个自然表情。
3. 点击“保存入库”。每次更新库后系统会自动重算阈值。
4. 让已录入人员进入画面，观察姓名、相似度、距离、阈值和多帧投票结果。
5. 让未录入人员进入画面，观察“未知人员 / 比对不通过”。
6. 演示暗光、过曝、模糊或距离过远时的可解释质量提示。
7. 查看 `logs/events.jsonl`，说明每条 JSON 记录可复现实验时间和判定依据，但不保存视频帧。

## 算法原理

### 检测和预处理

输入 BGR 帧先转为灰度，再经 CLAHE（限制对比度自适应直方图均衡）改善局部暗光。系统分别运行正脸与侧脸 Haar 分类器；右侧脸通过水平翻转复用侧脸分类器，最终以 IoU 非极大值抑制去重。裁剪时保留 14% 周边上下文。

### 多尺度 LBPH

对中心像素与八个方向邻域逐一比较，大于等于中心记为 1，否则记为 0，得到 8 位局部二值码：

```text
LBP(x, y) = Σ s(neighbour_k - centre) × 2^k
```

系统把循环位串跳变次数不超过 2 的编码视为 uniform（U2）模式。8 邻域共有 58 种 U2 模式，其余原始编码合并到第 59 个“非均匀”桶；这比保留 256 个稀疏桶更紧凑、抗噪。系统在半径 1、2 上各计算一次 LBP，并将 128×128 人脸划为 8×8 网格。每格统计 59 维 L1 归一化直方图，再拼接为 `2×8×8×59 = 7552` 维固定向量。相比直接比较像素，LBPH 对单调亮度变化、小幅表情和局部纹理更稳定，而且每个维度都有明确含义。

两特征使用平均卡方距离：

```text
d(H1,H2) = (1 / cells) × 1/2 × Σ ((H1-H2)^2 / (H1+H2+ε))
similarity = 100 × (1-d)
```

距离越小越像。系统对每个身份取最近 `K=3` 个样本距离的中位数，避免单个异常样本主导结果。

### 已知/未知判定与阈值校准

当最佳身份距离不超过阈值，且与第二名的距离差不小于 `ambiguity_margin` 时才判为已知；否则明确返回未知。阈值校准枚举库内同人正样本对和不同人负样本对，以 `2 × FAR + FRR` 最小为目标，强调降低误接收；样本不足时使用配置默认值 0.43。阈值始终限制在 0.20–0.62 之间。

这只是“库内校准”，不是严格的独立测试集评估。课程报告中应另外准备不参与录入的测试图片，计算混淆矩阵、准确率、FAR、FRR和平均耗时。

### 鲁棒性优化

- CLAHE：缓解局部过暗/不均匀曝光；
- 正脸与双向侧脸检测：扩大轻微姿态覆盖范围；
- 质量门控：低质量图像不进入标准库，也不做武断匹配；
- 多样本中位数：减弱单一异常样本；
- 前两名间隔拒绝：减少相似人员误判；
- 多帧投票：只有滑动窗口达到 60% 多数才稳定输出。

## 项目结构

```text
face_compare_system/
├── main.py                    # GUI 入口
├── config.json                # 所有可调参数
├── requirements.txt
├── 旧版操作手册.md
├── face_compare/
│   ├── camera.py              # 摄像头发现与生命周期
│   ├── config.py              # 强类型配置与校验
│   ├── database.py            # 动态标准库与阈值校准
│   ├── detector.py            # Haar 检测、侧脸与 NMS
│   ├── event_log.py           # JSONL 日志
│   ├── features.py            # 纯 NumPy 多尺度 uniform-LBP（U2）
│   ├── models.py              # 模块间数据结构
│   ├── quality.py             # 质量门控
│   ├── recognizer.py          # 判定与多帧投票
│   ├── service.py             # 业务编排
│   ├── cli.py                 # 无界面命令
│   └── ui.py                  # Tkinter UI
├── data/                      # 运行后产生，默认被 Git 忽略
├── logs/                      # 运行后产生，默认被 Git 忽略
└── tests/                     # 非摄像头逻辑测试
```

## 命令行用法

所有命令都应在项目目录执行：

```bash
# 批量录入（每张图只应有一张人脸）
python -m face_compare.cli enroll --name "张三" photos/zhangsan_*.jpg

# 识别一张或多张图
python -m face_compare.cli recognize test/known.jpg test/unknown.jpg
python -m face_compare.cli recognize --json test/*.jpg

# 查看标准库并重新校准
python -m face_compare.cli list
python -m face_compare.cli calibrate

# 检查摄像头索引
python -m face_compare.cli cameras
```

CLI 录入默认执行与 UI 相同的质量门控。只有在诊断时才建议使用 `--allow-low-quality`。

## 配置说明

常用参数位于 `config.json`：

- `camera.preferred_index`：优先的外置摄像头索引；
- `detection.min_face_pixels`：检测框最小边长；
- `quality.*`：亮度、清晰度、人脸占比门槛；
- `recognition.default_distance_threshold`：未校准时的距离阈值；
- `recognition.ambiguity_margin`：第一、第二名的最小距离差；
- `recognition.vote_*`：多帧投票窗口与多数比例；
- `storage.save_face_images`：设为 `false` 后仅保存特征，不保存裁剪图；
- `ui.enrollment_target_samples`：UI 建议采集数量。

调阈值时，减小距离阈值会减少误接收但增加拒真，增大则相反。应固定测试集后再比较，不要根据单张失败图片随意修改。

## 数据与日志

首次运行会创建：

- `data/people.json`：身份显示名、样本索引、校准信息；
- `data/features/<person-id>/*.npy`：LBPH 特征；
- `data/samples/<person-id>/*.jpg`：裁剪人脸（可配置关闭）；
- `logs/events.jsonl`：录入、校准和稳定识别事件。

这些路径已在 `.gitignore` 中排除。采集前应取得本人同意；实验结束后可手工删除 `data/` 下的运行数据。日志不含原图与特征，但姓名本身仍可能属于个人信息。

## 自动化测试

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

测试不访问摄像头，覆盖配置校验、U2 映射与直方图归一化、数据库动态扩容、质量门控、阈值校准、未知/歧义拒绝、多帧投票和纯人脸裁剪业务路径。真实摄像头、光照和不同人员的性能仍需按操作手册现场测试。

## 已知限制与后续改进

- Haar + LBPH 对大角度侧脸、严重遮挡和跨日期外观变化不如 ArcFace 等深度模型；
- 系统没有活体检测，照片/屏幕翻拍可能被接受；
- 摄像头“外置/内置”无法由 OpenCV 跨平台可靠获取名称，所以以可选索引和优先索引表达；
- 库内校准数据较少时容易乐观，正式评估应拆分录入集与测试集；
- 多人同框会逐脸给出单帧结果，但 UI 的主状态卡只跟踪面积最大的人脸；
- 第一次运行依赖本机安装 Python 包，但运行时无需联网或模型下载。
