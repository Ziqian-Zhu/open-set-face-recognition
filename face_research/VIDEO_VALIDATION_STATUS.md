# 公开多人视频：工程补强与数据预检

更新：2026-09-26。真人采集已取消；本页只讨论公开数据。**尚未得到真实运动视频识别结果，不能据此验收 P1-2 的真实视频端到端或 P1-3 的场景效果。**

最新用户确认：目前也没有已获取的视频数据，先完成**不依赖视频数据的部分**。因此暂不继续下载或请求真人采集；下面的数据缺口保留为待验证，不改成“已通过”。最新全套 **449 项测试通过**；静态公开数据的 [采集失败诊断](ACQUISITION_DIAGNOSTICS.md) 已补齐，另见此前 [指标更正](METRIC_CORRECTION.md)。

## 新增：实际调用路径的分阶段视频性能入口

`video_benchmark.py` 使用独立临时 SQLite 和明确标注的合成单位向量库，直接调用正式 `FaceComparisonSystem.analyze_frame` 与 `MultiFaceTracker.update`。`profiling.py` 只在该研究实例上临时计时，结束/异常后恢复实例方法，不修改类、UI、默认模型、质量或阈值，也不复制一套识别算法来测。

现在可记录检测、裁剪＋质量、对齐、SFace、检索＋身份判定、关联＋时序投票各阶段，以及处理总耗时、读取耗时与完整解码循环耗时。输出原始帧记录、阶段调用次数、p50/p95、处理吞吐率、环境/线程状态与数据/代码/模型/配置哈希。FPS 的分母明确是分析帧的处理总秒数，不是源视频或相机/GUI FPS。

预热使用首帧，随后**新建 tracker**，清除预热身份票和轨迹；首帧未触发的阶段明确标为未预热。视频元数据必须有效，实际解码数须等于声明数或明确指定的上限，不自动用25fps补齐，也不接受解码中断的半份性能成绩。库规模顺序测量，不据此声称新旧算法配对加速。按检测数量分组的零/单/多脸负载不是按真值分组，漏检仍可能落到零检测组。

将来有合法本地视频时可运行以下命令（当前未运行真实运动数据）：

```bash
face_compare_system/.venv-ui/bin/python -m face_research.video_benchmark \
  --video /合法本地数据/sequence.avi \
  --gallery-sizes 10,100,1000 --templates-per-identity 3 \
  --every 3 --warmup 5 --opencv-threads 1 \
  --output face_research/results/video-benchmark-new-001
```

它只测性能，**不生成已知/未知准确率**，不能替代独立录入库与人工真值的识别实验。线程请求与实际返回数都保留；本机仍是请求1、OpenCV报告8，不写成“单线程实测”。

无需新数据即可复现的工程检查：

```bash
face_compare_system/.venv-ui/bin/python -m face_research.smoke_video_benchmark \
  --output face_research/results/video-profile-wiring-reproduction-001
```

本机通过的 run 是 `results/video-profile-wiring-20260926-03/`。真实 YuNet/SFace 的12帧原始/计时调用逐项等价，包括框、质量、向量、识别和轨迹状态；随后10/100合成身份各测24帧，零脸、单脸、重复同脸布局各8帧，六个阶段均在预热中触发。三项产物哈希校验通过。**这仍是重复公开静态图＋空白画面的连通性检查，不是两名不同的人或真实运动。**

首次 `video-profile-wiring-20260926-01` 的整张图片拼接未覆盖单脸检测分组，自检失败，`.audit/failed.json`保留原因；当时没有发布时延数值，不重构或伪造它。随后只将固定测试人脸区域按可复现规则放大到布局中，使用新的run ID，默认检测/质量配置保持不变。后续自检已改为先保存实际测量及覆盖检查，再报告覆盖失败，避免遗失失败证据。新增24项回归验证调用/判定不变、零脸/质量拒绝、异常恢复、预热隔离、截断视频拒绝、临时库关闭、哈希变更拒绝及产物只建不覆盖。

## 本轮确实完成的工作

版本留档补充：`video-profile-wiring-20260926-02`已通过布局和真实模型检查，随后补上自检专用`RunJournal`，最终版本以新run ID `...-03`复测通过。自检失败也会留下审计目录并拒绝复用该run ID；这是合成连通性检查，不把它记成validation/test标定实验。既有01失败记录和02成功产物均保留。

`face_compare_system/face_compare/video_evaluation.py` 增加以下保护，默认 UI、跟踪算法、门槛及人员库均未改变：

- 预测帧号必须是非负整数，帧号/视频时间严格递增，耗时/距离必须有限且非负；同帧轨迹 ID 不得重复。没有真值脸的帧也检查所有预测框，没有检测结果的帧也检查所有真值框。
- `known` 必须有身份名称；未知真值必须显式写 `label: null`，遗漏标签报错，避免把错误输入统计成成功拒识。
- 新版导出的 `fps`、`every`、`decoded_frames`、`analyzed_frames` 与每条记录逐一核对。即使同时从预测和真值中删掉困难帧，也会因采样网格不完整被拒绝。旧的无元数据文件仍兼容，但无法提供同等级完整性检查。它们都不是防篡改认证，无法证明人工没改过原始文件。
- 增加每个 subject 的可见/可用/正确/误接/错认帧数、片段数与切换次数；给按身份等权的 macro 比例。出现 10 帧且全部正确的人与出现 1 帧但全漏检的人，逐帧正确率是 10/11，身份等权是 1/2，二者均保留。
- `session_id` 只接受清单提供的明确值，缺失保持 `null`；提供值也仅标为未独立核验。不会把视角、短暂重现或同一连续采集的人为切片当作新会话。身份级比例不给假设独立的置信区间。

新增严格标注解析器 `datasets/ltft.py`：保留无人帧、低检测置信度及 `face=0` 记录；拒绝漏行、重复帧/身份、非法框或字段。仅预检元数据，不生成 gallery，不调用识别器，不自动制造验证/测试真值。全套 **386 项测试通过**，其中本轮新增 43 项。

## 已取得的公开标注

来源是 [LTFT 作者仓库](https://github.com/hertasecurity/LTFT)，本机读取 `annotations_IJCB/choke1.txt` 与 `choke2.txt`。底层视频来自 [NICTA ChokePoint 官方数据页](https://arma.sourceforge.net/chokepoint/)。ChokePoint 的非商业研究等许可有用途及归属条件；LTFT 仓库未附通用 `LICENSE`，公开可访问不等于允许商用或任意再分发。标注/原始数据只保留在忽略的 `data/`，统计产物不附原图、向量或姓名。

预检 run：`results/chokepoint-annotation-audit-20260926-01/`，两个产物哈希已校验。**表中数据属于来源标注，不是本系统检测结果。**

| 序列 | 帧数 | `face=1` 身份 | `face=1` 框 | `face=0` 框 | 多个 `face=1` 的帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choke1 | 2526 | 24 | 7881 | 83 | 1768 |
| Choke2 | 2139 | 26 | 8509 | 201 | 1678 |

来源声明分辨率 800×600、30 FPS，尚未由视频解码复核。两份文件分别有 169、187 个 `face=1` 框超出图像边界，可能包括边缘部分人脸，预检如实保留，不擅自裁切。源仓库归档 SHA256 为 `63cc140d1b553f9a73deede2872f521df41691b201cc7031f1303b6a452130dc`；原始标注的 SHA256 记录在 `metadata.json`。

```bash
face_compare_system/.venv-ui/bin/python -m face_research.datasets.ltft \
  --choke1 face_research/data/chokepoint/LTFT-master/annotations_IJCB/choke1.txt \
  --choke2 face_research/data/chokepoint/LTFT-master/annotations_IJCB/choke2.txt \
  --output face_research/results/chokepoint-annotation-audit-reproduction-001
```

该命令只读取已有本地标注，不自动下载。输出目录须使用新名字；实际依赖源标注的研究使用与再分发须尊重原作者条件。

## 仍不能直接做识别实验的原因

1. **源视频未取得。** 官方下载位于 [Zenodo 815657](https://zenodo.org/records/815657)，两段拥挤序列压缩包约 180.5 MB 和 143.7 MB。本机对标准 records/record 文件地址、API content 地址及 www 入口均遇到 `SSL_ERROR_SYSCALL`，Python 路径遇到代理 503；浏览读取返回限流/文件过大。这些只是本次环境的访问失败，不代表数据已下架。没有关闭 TLS、绕过安全配置或上传本地人脸。GitHub 标注归档下载成功，不能冒充视频下载成功。
2. **视角不等于会话。** 作者按指定顺序拼接三路视角；不得用同时拍摄的另一路相机当作独立采集的 test。跨 Choke1/2 的数字身份映射尚未确认，不能直接按同号建库，也不能用本系统预测反推身份真值。
3. **标注不等于全部人脸真值。** [作者论文](https://arxiv.org/html/2107.13273v1)说明检测候选经人工核验。漏掉的候选可能仍不在标注内；`face=0` 同时表示假检或框内多脸，缺少二者区分。尚不能把它们全部作为未知人或忽略区而声称得到完整检测/误接指标。
4. **开放集还缺独立 gallery 与固定协议。** 需核对合法录入来源、跨文件身份对应、采集事件隔离、已知/未知定义和视频切换边界，再冻结抽帧、τ/m、时序规则。不能先挑本模型认得出的脸再报告总体结果。

后续可在上述来源问题解决后补充真正运动数据的分阶段耗时、逐身份/片段确认、误接、切换与延迟；缺的数据继续标未测。重复静态图的视频 smoke、合成轨迹回归与本次标注统计，均不能填入这些识别效果指标。无需恢复真人招募或开启摄像头。
