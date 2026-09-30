# Open-set Multi-face Recognition System

本地多人脸开放集识别应用。整体架构、有限模板选样与面向算法岗的证据入口见 [项目首页](../README.md)；最新 [P1 实验](../face_research/P1_RESULTS.md) 区分公开图效果、静态合成负载性能与时序工程回归。真人采集已取消，不作为本轮待办；线上默认阈值、画质和 UI 不因研究结果自动修改。

新增显式 `FaceEmbedder` 注入接口，默认仍是原 SFace 构造和特征签名；可选 ArcFace-MBF 仅从独立研究入口运行，使用固定权重校验、独立特征空间及临时库。公开 1:1 对照与非商业研究许可边界见 [模型实验记录](../face_research/MODEL_COMPARISON_RESULTS.md)，不在 UI 自动换模型，也不复用 SFace 阈值宣称 ArcFace 已完成生产标定。

## 2026-09-19：多人视频与向量库

本轮加入多人轨迹独立确认、本地视频回放与逐帧导出、SQLite事务向量存储、验证集阈值选择和带标注的视频评测。操作与实验流程见 [多人视频与开放集指南](多人视频与开放集指南.md)。尚未加入活体检测，也未完成真人独立准确率实验。

默认SFace配置现使用 `storage.backend=sqlite`。下次启动会把同目录旧 `people.json` 所引用的有效记录复制进 `vectors.sqlite3`，保留原文件，不需重新录入。先关闭旧程序再启动；迁移后不要让旧版程序继续写JSON库。SQLite归档是软删除恢复索引，不能复制为 `people.json`。

默认采用 **YuNet 检测 → 五点对齐 → SFace 128维单位特征 → 多样本距离聚合 → 未知/歧义拒识 → 连续帧确认**。支持电脑内置或USB相机，CPU本地推理。

旧版Haar＋Uniform-LBP保留为对比基线。升级版使用 `data_sface/`，旧版 `data/` 保留原样；两套特征不能混用，升级后需要重新录入。

## 启动

本机已安装依赖，并下载、校验两个官方模型：

```bash
cd face_compare_system  # 从仓库根目录执行
.venv-ui/bin/python main.py
```

新电脑建议Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

首次下载约39MB，来源为OpenCV官方仓库，固定SHA256验证后才使用，运行时无需联网。许可证在 `models/LICENSE-*`。Linux需安装Tkinter。Mac启动入口会避开旧Tk8.5，设备按真实名称优先选择内置相机，不试开手机镜头。当前本机使用`.venv-ui`。

## 创新点及实现依据

这里的创新属于课程项目工程设计与方法组合；YuNet和SFace是已有算法，不宣称原创网络。

| 创新点 | 实际实现 | 验证方式 |
|---|---|---|
| 五点对齐与双路线对比 | 眼睛、鼻尖、嘴角对齐后提取128维特征；保留7552维LBP | 同一独立清单分别运行两种配置 |
| 录入质量和身份一致性联合检查 | 姓名锁定、质量门禁、最小采集间隔、重复图检查、组内身份一致性、已有身份冲突检查、缩略图和拍摄提示 | 改姓名、连点、重复图、换人等测试 |
| 连续性约束的多帧确认 | 框重叠、特征距离、时间间隔联合检查；当前帧必须支持赢家；多人/低质量/无人脸清空历史 | 同位置换人、旧票占优但当前未知等回归测试 |
| 有界异步推理 | 后台运算，仅保留最新等待帧；设备切换或结果超时后丢弃旧结果 | 队列积压与推理异常恢复测试 |
| 可审计开放集评估 | 完全重复图去重、认错/拒真/未知误接受/检测失败分开统计、分场景统计、Wilson95%区间 | 独立清单evaluate，结果保存为JSON |

姿态提示是操作引导，不是自动姿态合格判定；连续性检查不是完整多人跟踪。没有加入未经验证的活体检测。

## 判定规则

SFace特征L2归一化，距离 `d=(1-cosine)/2`。每个人员取最近至多3张模板距离的中位数。默认要求余弦相似度至少0.45（距离≤0.275），第一、第二名余弦分数间隔至少0.08（距离间隔0.04）。这些是待现场验证的初始参数，不是准确率保证。

OpenCV教程给出LFW上的余弦参考阈值0.363；不能直接当成电脑摄像头场景的最佳阈值。当前初值更严格，需用独立validation集调整，再用独立test集评估。深度模式不根据刚录入的照片自动调阈值。界面匹配分只是距离的显示变换，不是身份概率。

多帧窗口7帧，至少4帧、60%支持且最近两帧支持当前赢家才确认。每条轨迹独立投票，外观与位置联合关联；歧义关联、漏检、超时与低质量会重新确认。两条轨迹同时声明同一已录入身份时暂不确认。多人可同时识别，录入仍严格单人。

## 录入、换人和删除重录

输入姓名，画面保持单人且质量合格，每次采一张。首张成功后姓名锁定，建议采5张略有不同的照片，检查缩略图后保存。新人员最少3张，同名追加最少1张。

SFace正式配置使用 `quality.focus_method=regional`：人脸统一到128×128，先通过局部对角残差的稳健中位数估计噪声，扣除其对一阶/二阶导数的预计能量贡献，再计算三区水平/垂直相对细节，取较弱方向和三区中位数。日志版本为 `regional-v2-noise-corrected`，包含 `noise_estimate`。它减少原始尺寸、整体对比度、单区强边缘和近似白噪声的影响，没有增强或改写入库照片；对相关噪声、压缩伪影、姿态仍有限制。默认细节门槛0.30、最小边缘能量16是待真人验证的工程初值，不是概率或经认证的画质模型。亮度、对比度、脸大小、关键点完整性仍独立检查。LBPH保留legacy原始Laplacian方差基线；regional不使用旧的55方差阈值作决定，但保留原值用于诊断。

平时戴眼镜的人员应使用同一姓名覆盖摘戴两种状态，建议各采3–5张，并用新画面测试。补录和识别共用最近至多3张模板距离的中位数；每个新样本都与提交前的库比较，组内一致性仍检查，追加还须排除其他身份的歧义。已匹配已有身份的样本不能通过换姓名另存；可疑样本不自动合并或放宽阈值。SFace权重、匹配阈值和未知拒识判定未改变；规则测试不能替代跨眼镜状态的真人实测。

界面动效与状态绑定：连接/查找/确认时显示活动环，结果变化时短暂脉冲，采集进度平滑过渡，保存成功后显示非阻塞文字与淡出背景。动画不是活体扫描或虚构的百分比。底部“减少动画”立即停止动态效果，文字/颜色和最终采集数量不受影响；本次窗口设置不自动落盘，可用 `ui.reduce_motion: true` 设置下次启动默认值。动画共用一个约30Hz定时器，空闲后停止、关闭窗口时取消。

可运行 `.venv-ui/bin/python scripts/smoke_quality.py` 复现公开样例尺寸、对比度、失焦和方向拖影的回归检查。它不打开相机、不写人员库，也不是多人真人眼镜识别准确率测试。算子依据见 [OpenCV导数说明](https://docs.opencv.org/4.12.0/d5/db5/tutorial_laplace_operator.html)；分区组合与阈值属于本项目的启发式设计，并非OpenCV推荐工作点。

放弃未保存照片：点“清空未保存样本”。删除已保存人员：点右上角“管理 / 删除已保存人员”，选择姓名后确认。SQLite中照片和向量仍保留，删除只是停用身份；`archives/`中保存恢复索引，必须与数据库一起保留，不是独立完整备份或永久擦除。

## 独立测试与基线对比

复制 `evaluation.example.json` 并替换图片路径。路径相对于清单文件所在目录；若移动清单，也要调整路径。gallery是录入集，probes是独立测试集，未知人员用 `label: null`。两组应来自不同拍摄批次。

```bash
python -m face_compare.cli evaluate evaluation.example.json --output evaluation_results/sface-test-01.json
python -m face_compare.cli --config config_lbph.json evaluate evaluation.example.json --output evaluation_results/lbph-test-01.json
```

评估使用临时库，不改变当前人员库，不在测试图上训练或调阈值。输出包含逐图结果、各场景、已知正确率、认错率、拒绝率、未知误接受率、检测/质量失败和静态链路延迟。没有未知样本时相应指标为null。无人脸和质量失败不算“未知正确”。

完全相同的解码图像或对齐裁剪会被拒绝；近重复连拍仍需人为避免。调参用validation，最终评估另采test；不要反复观察test调参数。

其他操作：

```bash
python -m face_compare.cli enroll --name "人员A" a1.jpg a2.jpg a3.jpg
python -m face_compare.cli recognize --json test1.jpg test2.jpg
python -m face_compare.cli list
python -m face_compare.cli --config config_lbph.json gui
```

## 验证及限制

完整结果见 `验证记录.md`。原16项测试已重新执行，并新增存储、录入、换人、后台处理和评估回归测试。官方模型实际加载、128维归一化和黑帧无人脸测试通过；实际Tk窗口录入/保存/复位测试使用临时数据通过，未打开相机。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/check_ui.py
```

两张OpenCV公开图片做了真实模型连通性检查。低清样例被质量门禁拒绝；单独的模型检查绕过门禁并明确记录，不能据此推导准确率。真人准确率和相机端到端延迟仍需独立采集验证。没有活体检测，照片/屏幕可能通过。

SQLite后端采用事务、模型签名/维数校验、同名约束和版本化精确检索缓存；新增、停用、恢复会使其他新版本连接刷新缓存。旧files后端保留原子JSON索引机制，仍建议单实例运行。人脸照片和向量未加密，采集须经同意，运行数据、日志、归档不应公开提交。

## 源码与来源

`deep_engine.py`是深度检测/对齐/特征，`enrollment.py`是录入和连续性，`worker.py`负责后台队列，`evaluation.py`负责独立评估；其余配置、数据、界面、业务、质量、日志职责仍分离。历史文档保留在docs/，当前操作以本README及根目录操作手册为准。

- [OpenCV检测与识别教程](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [YuNet官方模型](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SFace官方模型](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
