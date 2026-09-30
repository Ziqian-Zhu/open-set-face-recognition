# 保守式重构审计

日期：2026-09-26。状态：**仅审计，未实施重构，等待用户审核。**

本轮以用户最新任务文件末节为范围：只新增本文件及 `REFACTOR_PLAN.md`；测试审计和疑似清理项合并在本文件，不另建 `TEST_AUDIT.md`、`POSSIBLE_CLEANUP.md`。没有修改生产代码、测试、配置、启动入口或既有结果，没有开启摄像头或操作已有人员库。

这里的 P0/P1/P2 表示**保守重构的收益/风险优先级**，不是 `IMPROVEMENT_PLAN.md` 的算法实验阶段，也不是缺陷严重度。

## 1. 结论与简历定位

**已经具备写进简历的内容，适合定位为“计算机视觉工程＋可复现实验项目”。不需要为了简历继续堆功能。** 这是一项基于本地实现与证据的判断，不是对招聘结果、工业部署或论文创新性的保证。

- 工程贡献：完整检测、五点对齐、特征、身份级聚合、开放集拒识、多人轨迹确认、SQLite 持久化及异步桌面链路。
- 可量化贡献：精确检索分组批量化，在限定同库配对负载中有可复核时延收益，并检查完整排名/距离不变。
- 算法实验贡献：有限预算选样、validation-only 标定、质量/间隔消融、失败归因、人员分组统计和结果留档。
- 边界：Coverage 主比较为 13/24，First-K 为 14/24，不能写“原创选样已提升准确率”；公开裁剪图不能证明真实相机摘镜鲁棒性、真实多人运动收益或低 FAR。第三方模型和协作完成部分应如实归属。

目前更值得做的是小范围维护整理，**不是重写 matching、换模型或引入新框架**。现有 [RESUME.md](../career/RESUME.md) 已有证据受限的表述；未发现必须先完成大规模重构才能用于简历的理由。

## 2. 阅读范围与验证基线

### 2.1 范围

已逐文件审阅当前应用包、`main.py`、六个应用脚本、研究包（含 evaluation/datasets）、两套测试；并核对根目录审计/计划/简历/面试文档、两侧 README、操作指南/历史说明、两套配置、依赖、启动脚本、忽略规则、示例清单和四份固定协议。通过全项目引用搜索与 AST 检查补充调用关系、重复函数及规模统计。

活动代码清单：

- 应用包 28 个模块：`__init__`、`aggregation`、`camera`、`cli`、`config`、`database`、`decision`、`deep_engine`、`detector`、`embedding`、`enrollment`、`evaluation`、`event_log`、`features`、`gui_runtime`、`macos_devices`、`models`、`motion`、`operating_point`、`quality`、`recognizer`、`service`、`tracking`、`ui`、`vector_database`、`video`、`video_evaluation`、`worker`，另含 `main.py`。
- 应用脚本：`check_startup`、`check_ui`、`download_models`、`smoke_models`、`smoke_quality`、`smoke_video`。
- 研究包：顶层 `__init__`、`__main__`、`_baseline`、`acquisition`、`arcface`、`benchmark`、`experiment`、`manifest`、`matrix`、`model_comparison`、`profiling`、`quality_ablation`、`retrieval_benchmark`、`selection`、`smoke_video_benchmark`、`verification`、`verification_diagnostics`、`video_benchmark`；evaluation 下全部 9 个模块；datasets 下全部 6 个模块。
- 两侧测试共 46 个 Python 文件，分类见第 8 节。

没有把虚拟环境、缓存、模型二进制、原始人脸/私人数据库、所有生成结果逐字阅读算成“代码审计”。`.docx_work/` 的四个历史报告工具仅盘点，不属于当前应用/研究源码统计或重构对象；没有生成或改写课程报告。历史实验选择关键 run 做只读完整性校验，而非重新运行或覆盖结果。

### 2.2 可重复的源码规模口径

Python 文件按上述活动目录统计；研究端排除 `data/results/__pycache__`。物理 LOC 包含注释和空行，非空行也包含注释。函数数包括方法和嵌套函数，类数来自 AST；不是圈复杂度或代码质量评分。

| 范围 | Python 文件 | 物理 LOC | 非空行 | 函数 | 类 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 应用包＋main | 29 | 4,841 | 4,270 | 216 | 40 |
| 应用 scripts | 6 | 483 | 436 | 8 | 0 |
| 研究包，不含测试 | 33 | 6,852 | 6,199 | 189 | 20 |
| 两侧测试 | 46 | 5,290 | 4,441 | 428 | 21 |
| 总计 | **114** | **17,466** | **15,346** | **841** | **81** |

AST 扫描在非测试函数中去除首个 docstring、忽略位置信息，比较函数体，且只纳入跨度至少 5 行的函数：发现 **2 组完全相同函数体、4 处实现**，分别是文件 SHA256 和嵌套 dHash 缓存读取。它不是全项目“语义重复率”，也不证明其他相似代码可合并。

跨度至少 100 行的函数 **16 个**，列表见第 7 节；不以数量下降作为验收目标。

### 2.3 本轮实际验证

```bash
face_compare_system/.venv-ui/bin/python -m pytest -q \
  face_compare_system/tests face_research/tests -p no:cacheprovider
```

本轮末次结果：**449 passed in 4.84s**（同轮此前一次为 5.40s）。这不是视觉准确率、测试覆盖率百分比或 GUI 像素验收。未重新运行桌面/相机 smoke；已有相关记录只能称历史证据。

默认 `config.json` 的 SHA256：`4e1edc89a162460848ce459989c85fc05b22a53b3a5495c6cdd3bd6319d5acaf`。

只读校验的 8 个结果目录均通过 `verify_result_directory`：

| run，均位于 face_research/results/ | 校验文件数，不含 artifacts.json |
| --- | ---: |
| `exact-retrieval-paired-20260926-01` | 2 |
| `static-synthetic-benchmark-optimized-20260926-01` | 3 |
| `meglass-mixed-metric-fix-20260926-01` | 8 |
| `meglass-mixed-quality-metric-fix-20260926-01` | 4 |
| `meglass-verification-20260926-01` | 8 |
| `sface-arcface-verification-20260926-01` | 10 |
| `meglass-acquisition-diagnostics-20260926-01` | 5 |
| `video-profile-wiring-20260926-03` | 3 |

完整性通过不等于许可、数据独立性、盲测或结论正确的证明；相应限制仍以原实验记录为准。

## 3. 模块职责审计

| 模块 | 现在承担的职责 | 审计判断 |
| --- | --- | --- |
| `ui.py` | Tk 布局、设备/视频、结果时效、采集交互、人员管理、动效 | 长但主要是交互；没有在 UI 内复制 embedding/聚合算法。保留主流程，不拆 MVC。 |
| `service.py` | 组装后端；检测→质量→提取→匹配；录入一致性、事务编排、日志 | 业务编排集中是合理的。整帧/已裁剪脸入口具有不同输入约束，不宜合成一个大量布尔开关的函数。 |
| `deep_engine.py` | 模型校验、YuNet 坐标还原、SFace 对齐与向量契约 | 边界清楚；模型输入、维度和签名属于模型契约。 |
| `quality.py` | 曝光、对比度、面积、关键点、regional/legacy 清晰度 | 不做身份判定；与选样代理分用途不同，保留。 |
| `vector_database.py` | SQLite 事务、签名、旧库导入、归档、revision 缓存、精确排名 | 没有自行决定 Known/Unknown；身份聚合已复用核心函数。向量化排名留在库侧，避免退回逐身份循环。 |
| `database.py` | 旧文件库、兼容元数据、LBPH 库内校准 | 校准与持久化存在历史职责交叠，但仍是可达兼容功能，非本轮拆分重点。 |
| `worker.py` | 有界最新帧队列、代次/库版本复位、后台调用、错误传回 | 调用识别/跟踪，不另写决策公式；没有操作 Tk。小模块不应再引入 Handler/Manager。 |
| `tracking.py` | 几何/外观关联、每轨迹投票、超时、容量、身份冲突 | 没有直接访问 UI/数据库；关联与身份接受是不同规则，不能合并阈值。 |
| `face_research` | 清单/协议、选样、标定、实验编排、统计、产物与基准 | 方向正确；主要维护债是跨模块私有 helper 和长编排。重复校准入口需要兼容，而非一律删除旧入口。 |

应用包中未发现 `face_research` 导入；研究包按需依赖应用稳定算法。维持这一方向。`_baseline.py` 的“唯一导入边界”描述已经不准确，但不能为了兑现旧注释把所有核心纯函数绕回庞大的 service。

## 4. Matching 逐层检查及重复分析

当前合理的调用关系是：

```text
SFace/可选研究 embedder → 特征契约
  → 生产 rank_identity_candidates / 研究 rank_with_index
  → SQLite 批量或研究标量 template distances
  → aggregation 的标量/批量身份聚合
  → 候选排序 → decision.accept_identity → known / unknown
  → 生产端各轨迹独立时序确认
```

| 主题与代码位置 | 相似内容/实际差异 | 是否统一；风险 |
| --- | --- | --- |
| `deep_engine.py:SFaceExtractor.extract`、`arcface.py`、`selection.py:_matrix`、`vector_database.py:_validate_vector/refresh_cache` | 均检查向量，部分做 L2；维度、dtype、错误信息和存储前后语义不同。SQLite `_validate_vector` 并不等价于“总是返回单位向量”。 | **保持**。盲目统一可能改变存储字节、模型签名关联和浮点结果。 |
| `deep_engine.py:distance`、`selection.py:rank_with_index`、`vector_database.py:rank_candidates` | 同为半余弦距离；标量 norm/dot 与预归一化矩阵乘法的舍入路径不同。 | 不新增万能距离器。数学等价不代表逐位相同；先保留已有数值 oracle。 |
| `aggregation.py` 标量与 batch | 同一策略的两种执行形态，batch 按真实模板数分组，标量用作独立参考。 | **已共享算法定义，保留双实现**。合成一个 Python 循环会丢失已测性能。 |
| `recognizer.py:rank_identity_candidates` 与 `selection.py:rank_with_index` | 生产含 person_id、库后端；研究输入离线 GalleryIndex，输出没有 person_id。并列键分别为距离/ID/姓名和距离/姓名。 | 不强行统一返回类型/并列排序；可解释契约，不能假定同分身份必然同序。 |
| `decision.py:accept_identity`、研究 `selection.py:decide` | 生产和主要研究路径已共享接受规则。 | 保持，无需再造 `decide_identity` 类或重复包装。 |
| `operating_point.py:select_operating_point` | 旧 CLI 工具直接写 `gap >= margin`；共享函数还容忍 1e-12 等号舍入。阈值扫描范围/候选也不同。 | **存在真实边界差异**，见 R10。不能当无行为影响的替换。 |
| `enrollment.py:validate_identity_set`、`service.py:_enroll_samples` | 组内所有同人距离的中位数、逐新样本与现有身份最近 K 聚合、跨身份冲突门槛分别对应不同业务问题。 | 保留；不套用统一识别 threshold/margin 替代录入约束。 |
| `quality.py` 与 `selection.py:quality_proxy` | 前者决定能否使用；后者是 gallery 内的排序代理分，融合清晰度/面积等。 | 非重复，代理分不应反向决定线上质量门禁。 |
| `evaluation.py:summarize` 与研究 `metrics.py` | 部分相同计数，但输入、兼容字段和 1:N/1:1 分母不同。 | 不大合并。保留采集失败、未接受、成功拒识三者区别。 |
| `experiment.py:calibrate_threshold` 与 `evaluation/calibration.py:calibrate_open_set` | 距离事件扫描相似；前者旧固定 margin API，后者联合搜索/冻结对象/扩展统计。 | P2 评估兼容适配，不替换后直接宣称等价。 |
| `config.py:load_config` 与研究各 CLI | 核心配置已有一个读取/校验入口；研究参数是协议/基准参数，不全属于线上配置。 | 保持；不做跨入口总配置管理器。 |
| `experiment.py:280` 与 `datasets/meglass.py:50` | 完全相同的分块文件 SHA256 函数。 | **值得统一，R02**。不合并解码图/对齐图哈希等不同定义。 |
| `datasets/meglass.py:187` 与 `datasets/meglass_mixed.py:49` | 两个嵌套 `dhash` 完全相同：按 filename 缓存、读取、报错、计算。外围跨集合循环不同。 | **可小范围共享缓存读取，R05**；不统一外围抽样/筛选协议。 |
| `experiment.py:write_report` 与 `evaluation/artifacts.py:write_json_once` | 都是临时文件→flush/fsync→create-only link→清理；路径解析、建父目录、返回值和临时名不同。 | P1 共享最小写入机制，保留包装契约，R06。 |
| `profiling.py` 与 `acquisition.py` | 都临时替换实例方法再恢复；一个计时且含 tracker，一个记录终止阶段/异常且含 crop。 | 保留少量重复，勿引入通用拦截框架。 |

### 已复现的边界差异

```text
0.315 - 0.275 = 0.03999999999999998
旧工具直接比较 >= 0.04：False
accept_identity(0.275, 0.315, 0.275, 0.04)：True
```

这只证明旧阈值工具的等号边界可能与主链路不同，**不表示已发现当前真实人员错认，也不意味着所有研究标定均有此问题**。本轮不修正，后续需明确批准为兼容性缺陷修复。

## 5. 配置、命名和数据结构

| 项目 | 当前含义 | 本轮建议 |
| --- | --- | --- |
| `engine.cosine_threshold=0.45` | SFace 相似度门槛，换算为距离 τ=0.275 | 保留键和值，注释说明换算；不与 LBPH 0.43 合并。 |
| `engine.cosine_margin=0.08` | 距离间隔为 0.04 | 同上，绝不直接把 0.08 当距离 margin。 |
| `RecognitionResult.similarity` | `100*(1-distance)` 的展示分，不是正确概率，也非 cosine 百分比 | 保留字段，说明语义；CLI 文案修改另审，见 R10。 |
| `nearest_samples=3` | 最近的至多三模板，不是最近采集三张 | 已配置，不再复制另一份研究默认管理层。 |
| 投票 7/4/0.6、tracking 参数、质量参数 | 已由 dataclass/config 统一管理 | 不改默认；已有参数不必重复放到常量模块。 |
| 112×112 / 128维 / ArcFace512维 | 模型输入/输出契约 | 留在模型实现，不做任意可调配置。 |
| `service.py:250/252` 的 `.65` | 跨身份高相似冲突检查的 cosine 门槛 | 可同模块命名常量，保持 `(1-c)/2` 运算顺序，R03。 |
| UI 三处 1 秒 | 分别是已返回分析时效、无新分析超时、待采集原始帧时效 | 可用明确的局部/模块常量名，但不能因数值相同就共用同一个业务参数。视频卡顿重设期限又是另一含义。 |
| 研究选样的 0.25/0.2/0.05 | 当前固定算法定义/超参 | 不在重构中调值或顺手增加搜索空间。 |

`BoundingBox`、`QualityReport`、`RecognitionResult`、`FaceObservation`、`PreparedSample`、`Track`、`Template`、冻结参数等已有适当结构。worker 输入四元组/输出五元组可以先写明字段和两个时钟，研究报告的 dict 是可序列化 schema；没有必要全部改为 dataclass。认定类型注解过时，不等于必须引入 Repository/Adapter 层。

## 6. 按优先级的可审核建议

下列“建议实施”均指**用户审核后的下一轮**，不是本轮授权。当前所有项均未实施。

### R01 · P0：修正过时说明，补齐接口语义

- 位置：`face_compare/__init__.py:1`、`recognizer.py:33`、`config.py:24`、`camera.py:1`、`ui.py:897/989`；研究 `_baseline.py:1`、`selection.py:217`、`evaluation/pairs.py:22` 及 README“开发检查”。路径前缀分别为 `face_compare_system/` 与 `face_research/`。
- 问题：默认深度路线仍被描述为 LBPH；“唯一导入边界”“只支持 SFace”“仅 nearest-k median”等描述与当前可注入/多策略能力不完全匹配。
- 建议/收益：只改注释和 docstring，说明默认值、后端支持、距离单位、排名返回字段；减少读码误解。
- 风险/行为：低；不改类名、参数、`__version__`、公开字段和算法。类型重构不与本项捆绑。
- 验证：审阅纯说明 diff，相关导入/配置/识别测试后跑全量；源码哈希会正常变化，不能伪装旧版本。
- 建议实施：**是，第一项**。历史实验数字、旧版本文档事实不直接回填成新版本。

### R02 · P0：合并两处完全相同的文件 SHA256

- 位置：`face_research/experiment.py:280`、`face_research/datasets/meglass.py:50`。
- 问题：同一流式文件摘要实现有两个维护位置，其他研究模块还跨私有名导入。
- 建议/收益：在现有 `evaluation/artifacts.py` 提供一个明确的文件摘要函数；旧位置保留兼容名，分块大小、输入契约、异常不变。无需新增 UtilsManager。
- 风险/行为：低，但须检查导入环及 monkeypatch 查找位置；文件摘要应逐字相同。不要顺手改图片内容哈希、模型校验或把大文件改成一次读取。
- 验证：空文件、跨 1 MiB 文件、非法路径及旧导入；artifacts、experiment、MeGlass、matrix、verification、benchmark 相关测试，再全量。
- 建议实施：**是**，与 R01 分成独立小批次。

### R03 · P0：给固定数值和传输字段补语义，不扩大配置

- 位置：`service.py:_enroll_samples`、`ui.py:_tick/_capture_enrollment_sample`、`worker.py:submit/_run/poll`。
- 问题：`.65` 的单位、三个 `1.0` 的不同用途、worker 元组位置和 captured/tracking_time 易混淆。
- 建议/收益：同模块命名 `.65`，分别注明三个时效含义；为元组及墙钟/媒体时钟写契约。首轮不改变元组结构，不把相同数字强行绑定到一个配置。
- 风险/行为：低；仅名称替换仍需保持精确算式/比较符/时间来源。UI 时效如无清晰收益可只加说明。
- 验证：enrollment_matching、refinements、worker、tracking、video/motion；相关后全量。若实际改 UI 表达式，补临时库桌面 smoke。
- 建议实施：**是，仅限语义整理**，不新增可调参数或改阈值。

### R04 · P1：公开少量已跨模块使用的稳定纯函数

- 位置：`detector.py:91` 的 IoU，被 tracking/enrollment/video_evaluation 调用；研究 `experiment.py` 的摘要/准备/记录 helper，`evaluation/reporting.py` 的 `_csv/_write_json` 等。
- 问题：跨模块长期调用下划线方法，实际共享契约没有被明确表达。
- 建议/收益：先处理 IoU 或 CSV 其中一个主题；使用简单公共函数/兼容别名，保留旧入口。复杂实验编排 helper 暂不整体迁移。
- 风险/行为：中；猴子补丁位置、循环依赖、CSV 列序和错误信息都可能受影响。IoU 的数值函数可共享，各调用方门槛不可统一。
- 验证：NMS、跟踪、连续性、视频匹配；或 reporting/产物原子性测试；再全量。比较同一输入的输出与排序。
- 建议实施：**可选，一次一个 helper**；不要求所有下划线函数改公共 API。

### R05 · P1：仅共享 dHash 的缓存读取片段

- 位置：`datasets/meglass.py:_has_cross_split_near_duplicate`、`datasets/meglass_mixed.py` 同名函数。
- 问题：两个嵌套函数完全相同，但外围单向/混合协议的样本组合并不同。
- 建议/收益：抽出接收 `row, hash_cache` 的普通 helper，保留缓存键、读图时机、错误信息；外围筛选和迭代顺序保持原样。
- 风险/行为：中；读取/短路顺序改变可能影响首次报错和协议筛选。不要顺手更改 dHash 阈值、种子或 held-out 检查。
- 验证：两套 MeGlass 合成 fixture，缓存命中不重复读取、相同错误、完全相同清单；不得重采样旧 test 或覆盖协议。
- 建议实施：**可选**，收益小于 R02，不作为首批必做。

### R06 · P1：共享 create-only JSON 的最小机制

- 位置：`experiment.py:492`、`evaluation/artifacts.py:15`。
- 问题：原子文件发布机制重复。
- 建议/收益：保留 `write_report` 的路径 resolve、建父目录和 Path 返回值，以及 `write_json_once` 的原契约；只复用序列化/写入机制。
- 风险/行为：中；NaN 拒绝时机、文件已存在、失败清理、异常类型、临时文件可见性不可无意改变。staging 内 `_write_json` 允许写入私有暂存文件，与公开 create-only 写入**不等价**。
- 验证：原子性测试外，增加写入/fsync/link 失败注入、无半成品、Unicode/末尾换行、已存在目标不变及旧返回值测试；再全量。
- 建议实施：**可选**，不应为二十余行重复引入通用存储层。

### R07 · P1：补齐异常路径上的资源关闭

- 位置：`face_compare_system/face_compare/cli.py:80`（restore 与最终业务分支）；`face_research/datasets/meglass.py:269`。
- 问题：CLI 部分分支依赖进程退出释放数据库；restore 只在成功路径显式关闭。MeGlass 三个系统在进入 try/finally 前逐个构造，第二/第三个构造失败时前面连接没有显式关闭保证。
- 建议/收益：用小范围 try/finally，或按构造成功次序注册 ExitStack 清理；不改三份库、构造顺序、事务边界。
- 风险/行为：中；正常算法结果不应变，但异常资源行为会变。关闭异常不能掩盖原始错误，须覆盖无 close 的 files 后端。
- 验证：成功/异常均恰好清理、部分构造失败、原错误/退出码保留、临时库可释放；再全量。
- 建议实施：**建议单独批准后做**，不将它称为纯注释整理。

### R08 · P1：减少异常文本耦合，保留正确的容错

- 位置：`matrix.py:run_matrix_experiment`、`quality_ablation.py:run_quality_ablation/policy_accepts`、`scripts/check_ui.py:179`。
- 问题：部分“不可比较/不可用”分支依赖 ValueError 中文片段；质量消融依赖原因字符串；UI smoke 清理有宽泛 `except Exception: pass`。
- 建议/收益：先为当前分类补回归；若收益明确，再用单个明确异常类型/错误码表达不可标定，保留旧消息与 ValueError 兼容。smoke 仅缩窄已知 Tk 销毁异常。质量原因 schema 暂不改。
- 风险/行为：中；错误从 unavailable 变为终止或反向变化均是实验语义变化，不能忽略。不要把 worker 的异常传 UI 当作吞错。
- 验证：缺已知/未知、FPIR 不可达、意外 ValueError 传播、原状态/文案；GUI 清理成功/失败；相关后全量。
- 建议实施：**先测试/说明，后续择一处理**，不重建整个异常体系。

### R09 · P1：局部改善测试组织，保留独立 oracle

- 位置：`face_research/tests/test_model_comparison.py:9` 导入另一测试文件 `_fixture`；`face_compare_system/tests/test_tracking.py` 的多事件循环断言。
- 问题：跨测试模块取 fixture 的边界不清；一条循环测试失败不易定位具体事件。
- 建议/收益：仅抽取真正共享的 fixture 到相邻测试辅助文件/适当 conftest；独立事件可参数化，案例、数值、断言不变。
- 风险/行为：低至中；fixture scope、随机数消费和隔离方式改变会影响测试含义，不能为了减少行数改数据。
- 验证：修改前后案例映射、各文件单独运行、全量；允许参数化后测试项数增加，不要求永远恰好 449。
- 建议实施：**可选**，不是删测试任务。

### R10 · P1：决策边界及 CLI 展示语义，单列行为修正

- 位置：`operating_point.py:40` 附近的直接 gap 比较；`cli.py:201` 的“相似度 %”。
- 问题：旧工具与主判定浮点等号不一致；CLI 易将展示分理解成概率/余弦百分比。
- 建议/收益：先补刻画测试并记录兼容差异；另行批准后，仅旧工具接受条件复用共享函数，保留其扫描范围和选择规则；CLI 可改“匹配分（非概率）”，JSON `similarity` 键不变。
- 风险/行为：**确定可能改变边界模拟结果/用户可见文案**，不属于本轮 behavior-preserving 实施；不能改共享函数来迎合旧工具，更不能改测试期望掩盖差异。
- 验证：精确等号、one-ULP 邻居、单身份、非有限值、validation-only、旧报告兼容；新旧曲线逐项解释差异。
- 建议实施：**本次保守整理不做，需明确批准缺陷修复**。

### R11 · P2：大范围类型化/返回结构更换

- 位置：worker 四/五元组、research 报告 dict、`FaceRecognizer.__init__` 的历史类型注解。
- 问题：有可读性空间，但已有消费者按位置或 JSON schema 使用。
- 建议/收益：优先 R01/R03 契约说明；仅实际有收益时补局部 type alias 或准确的结构类型。
- 风险/行为：中至高；NamedTuple/dataclass 的序列化、反射和消费者兼容不能假定。无须新增整套存储接口层。
- 验证：旧解包、JSON 字段/值/顺序契约、两后端及注入测试，全量。
- 建议实施：**否**，不为 typing 制造大 diff。

### R12 · P2：长编排函数、校准入口和通用 pipeline 合并

- 位置：`matrix.py:79`、`experiment.py:336/152`、`quality_ablation.py:161`、三个 MeGlass 构建器、`ui.py`。
- 问题：长函数提高阅读成本，但很多长度来自必须显式呈现的协议和布局。
- 建议/收益：如以后维护确实困难，可先抽纯结果格式化/冻结凭据组装；保留验证→全部冻结→test 顺序可见。旧校准入口若适配必须保留输出契约。
- 风险/行为：高；易改变 RNG 消费、图像访问顺序、失败记录、冻结时机和分母。不要抽万能 ExperimentRunner。
- 验证：冻结先于任何 test 访问、test 标签扰动、相同 seed 清单/模板、逐字段计数/阈值、失败留档；全量及固定输入回归。
- 建议实施：**否，首轮保持**。超 100 行不是缺陷本身。

### R13 · P2：清理疑似未使用项、统一格式和压缩文档

- 位置：第 10 节列表、研究 README 的历史进度段、历史操作文档链接。
- 问题：少数辅助入口仅见定义，进度文档有重复信息；缺少引用不代表外部调用不存在。
- 建议/收益：只记录证据；若后续获批，先修明确失效链接/过时当前状态说明，保留历史 run、实验限制和兼容入口。仅格式化实际修改文件。
- 风险/行为：删除风险高于收益；旧脚本/导入/实验复现可能依赖。历史资料不要被新结果覆盖。
- 验证：全项目引用、CLI 帮助/导入、链接、历史命令、全量；不能只看静态未引用就删。
- 建议实施：**本轮不删、不全仓格式化、不搬目录**。

## 7. 长函数、异常及日志补充

### 长函数不是自动拆分名单

以下行数为 AST 起止位置的包含式跨度，含函数体内空行/注释：

| 文件/函数 | 起始行 | 行数 | 处理 |
| --- | ---: | ---: | --- |
| app `cli.main` | 80 | 139 | 先关注 R07 清理，不改分发框架 |
| app `ui._build_layout` | 177 | 129 | 保留布局顺序 |
| app `video_evaluation._validate_video_inputs` | 27 | 106 | 保留逐类完整性校验 |
| app `video_evaluation._evaluate_video_rows` | 198 | 157 | 状态统计先保持 |
| scripts `check_ui.main` | 30 | 151 | 连贯场景测试保留 |
| research `benchmark.run_benchmark` | 107 | 142 | 保护计时边界 |
| datasets `meglass.build_meglass_manifests` | 309 | 264 | 保护抽样/筛选顺序 |
| datasets `meglass_mixed.build_meglass_mixed_manifests` | 69 | 253 | 同上 |
| datasets `meglass_pairs.build_pair_protocol` | 16 | 111 | 保持仅元数据构建 |
| evaluation `grouped.grouped_verification` | 14 | 122 | 保持整连通组采样与分母 |
| evaluation `pairs.score_pair_manifest` | 22 | 159 | 保持严格/显式缺会话两种契约 |
| research `experiment.run_experiment` | 336 | 154 | 旧 CLI 仍使用 |
| research `matrix.run_matrix_experiment` | 79 | 392 | 最长；R12 只列可选局部提取 |
| research `quality_ablation.run_quality_ablation` | 161 | 187 | 保持四门控独立标定/统一工作点 |
| research `selection.select_templates` | 91 | 103 | 五种短策略分支，不建策略类层次 |
| tests `test_experiment.test_end_to_end_orchestration_freezes_all_methods_before_test` | 287 | 102 | 不以行数拆断时序断言 |

异常审计没有发现“所有 except/pass 都该删”的依据：worker 的 `queue.Empty/Full` 是有界最新帧语义；`poll()->None` 是暂无结果；UI 设备错误展示给用户；数据库 BaseException 回滚和实例方法恢复属于必要清理；CPU 元信息探测失败有平台信息降级；Tk theme/window cleanup 有容错需要。明确候选已列 R07/R08。

CLI、下载器和研究脚本的 `print` 是命令输出，不应整体换 logging。生产事件已有 `EventLogger`，不输出每帧 embedding；日志失败不会把成功入库伪装成失败。当前不引入日志框架、不增加高频帧日志。已有包含姓名的运行日志仍应视为私人数据。

## 8. 测试审计

五类是重叠标签，不是互斥删除清单。**449 项原有测试均保留；本轮删除 0 项、修改预期 0 项。**

### 8.1 必须保留

| 保护的契约 | 主要测试模块 |
| --- | --- |
| 聚合/距离/拒识等号、非法向量、展示分 | app `test_aggregation`、`test_decision`、`test_features`、`test_recognizer`；research `test_selection`、`test_calibration`、`test_metrics` |
| JSON/SQLite、签名/迁移/归档、回滚/多连接、录入混人 | app `test_database`、`test_vector_database`、`test_upgrade`、`test_refinements`、`test_enrollment_matching`、`test_service_crops` |
| 质量门禁/分区抗干扰及配置 | app `test_quality`、`test_regional_quality`、`test_config` |
| 轨迹换人/时间倒退/漏检/身份冲突、后台代次/异常 | app `test_tracking`、`test_worker`、`test_video`、`test_video_evaluation` |
| 设备路由、运行时兼容与动画调度 | app `test_camera`、`test_gui_runtime`、`test_motion` |
| 隔离/泄漏、所有组合冻结后读 test、扰动标签不改参数 | research `test_protocol`、`test_pairs`、`test_experiment`、`test_matrix`、`test_p0_regression`、`test_meglass`、`test_meglass_mixed`、`test_meglass_pairs`、`test_live_mixed` |
| 失败分母、质量消融、1:1/模型对照和诊断不改结果 | research `test_quality_ablation`、`test_verification`、`test_arcface`、`test_model_comparison`、`test_acquisition_diagnostics`、`test_grouped` |
| 产物不覆盖/失败不发布/篡改拒绝、时延口径、视频完整性 | research `test_artifacts`、`test_reporting`、`test_benchmark`、`test_retrieval_benchmark`、`test_video_benchmark`、`test_ltft` |

两侧 `conftest`/测试支持代码也属于阅读和计数范围。名称相近不构成重复：单位公式、持久化集成、全链路编排分别保护不同层次。

### 8.2 可以考虑参数化

- `test_tracking` 中质量失败/漏检/超时/时间回退的独立事件循环，可拆为具名参数案例，保持每次全新 tracker，便于看到哪个事件失败（R09）。
- `test_selection` 的独立非法 budget/feature 输入可局部参数化；如果牺牲错误定位或需要复杂 fixture，则维持现状。
- 已有 `test_config`、retrieval 参数校验等参数化正常，没必要再套一层生成器。

### 8.3 疑似重复

- 两套 MeGlass 测试的合成数据构造结构相近，但样本数/种子不同（例如 19 与 127），不可直接换成同一随机 fixture。
- `test_model_comparison` 借用 `test_verification._fixture` 是测试支持复用的位置问题，不是功能测试重复。
- 聚合标量对照、手算指标、旧 reference_rank 看似重复生产计算，实为**独立 oracle**。不能让 expected 与 actual 调同一优化函数后删掉对照。
- artifacts 单元测试、reporting 文件发布、CLI existing-output 测试分别检验机制、完整产物、执行前阻断，应同时保留。

### 8.4 实现细节耦合较高

- SQLite 测试直接操作 `_conn`、检查 `_matrix/_ranges/_aggregation_groups`：用于事务故障、缓存失效和精确向量化，具有必要性；如内部结构改变，先保留外部行为 oracle，不能只删除断言。
- experiment/matrix monkeypatch `_source_hashes/_sha256_file/_prepare_gallery/_read_probes`：避免接触真实数据并断言 test 访问顺序；R02/R04 必须保留 patch 点或明确验证等价。
- 服务用 `__new__` 绕过昂贵模型构造：适合隔离单元测试，不要求为此重写生产构造/引入 DI 框架。
- 校准调用次数/实例方法恢复等断言对内部实现敏感，但前者保护全部冻结顺序，后者本身就是契约；先识别目的再动。

### 8.5 暂时不要动

- worker 并发与有界超时测试：可能受繁忙机器影响，当前通过。不要改为随意 sleep 或降低断言来“稳定”。
- regional 质量、模型哈希/预处理/签名测试：规则数据不是精度证据，但仍是防止行为漂移的关键。
- GUI 实际控件 smoke 与单元测试互补，pytest 通过不代表两种窗口尺寸、姓名锁定/删除确认的桌面流程已重新验收。
- 公开结果完整性/输入漂移/失败日志测试，不因“太保守”删除。

优先补的刻画测试是文件摘要边界、资源部分构造失败、旧阈值工具舍入差异以及原子写入失败；补测不授权修改算法或放宽预期。

## 9. Keep As Is

1. `aggregation.py` 的标量和批量实现并存；SQLite 按真实模板数分组、全身份精确排名，保留第二身份用于 margin。
2. `decision.accept_identity` 与主要生产/研究调用共享；单身份、等号、非有限值语义不改。
3. SFace 的半余弦单位、模型校验/签名、旧 LBPH/files 分支；不能让相同维度但不同特征空间共用库。
4. SQLite 事务、write_guard、revision、软删除/恢复和旧 JSON 变动检查；不引入 ORM 或 Repository。
5. 录入“逐样本与提交前库比较”和组内一致性；不能用新样本互相担保或为摘镜直接放松门槛。
6. Track 各自投票、关联歧义/漏检/时间异常清票、容量和身份冲突；不用新状态机框架重写稳定代码。
7. worker 单后台线程＋最新帧队列，UI 的代次和时效保护；无追求完整处理所有相机帧的队列改造。
8. UI 主操作、名字锁定、归档确认、单定时器动效与减少动画；保留无 track_id 时的 voter/continuity 回退，它仍被调用。
9. `RunJournal` 冻结/访问顺序、create-only 产物、失败留档、原始计时、计数/分母、配置/模型/源码哈希。
10. 单向、混合、1:1 元数据协议各自编排；旧 test 不因重构变回盲测，保持 RNG/筛选/读取顺序。
11. 研究 instrumentation 在隔离实例上安装并恢复；不把观察器加到正式默认路径。
12. 当前明确写出的负结果、统计不足、无活体/无真实运动验证、私人数据不公开的限制。

## 10. Possible Cleanup, Do Not Delete Yet

以下引用结论来自活动源码/测试和工作区 Python 文本搜索，不证明工作区外无人使用。**没有任何项获准删除。** 总体处置对应 R13（P2）；本节为逐目标证据。

| 候选/位置 | 引用证据、保留原因 | 若以后处理：风险与验证 |
| --- | --- | --- |
| `features.py:145 chi_square_distance` | 本地仅见定义；LBPHExtractor.distance 有相近公式，但 dtype/截断/参数契约不同。 | 可能为外部教学 API；先加刻画/导入测试并确认兼容，再讨论弃用。本轮不改行为。 |
| `database.py:383 iter_feature_vectors` | 本地仅见定义；属于文件库公开辅助 API。 | 不能排除手工脚本；检查所有入口/外部使用并做 files 回归，收益有限。 |
| `recognizer.py:67` 附近再次 sort | 内置 rank 路径已排序，但自定义后端可能返回未排序候选；稳定 sort 还保留并列顺序。 | 删掉可能改变契约；先覆盖自定义后端和同分，不建议本轮删除。 |
| `detector.py:115 largest_box` | **不是死代码**：service.py:141 正在调用。 | 保留单人采集入口，不能因多人支持而删除。 |
| `experiment.run_experiment/write_report` | `__main__.py` 非 matrix 路径、benchmark 旧 JSON 输出仍调用，测试也覆盖。 | 保留旧 CLI/输出兼容，不因有新矩阵入口删除。 |
| `reference_rank` 与测试的标量距离/穷举实现 | 性能/数值独立参照，明确被基准及测试调用。 | 删除会削弱等价性验证；Keep As Is。 |
| `TrackContinuity` 与 UI 单人 voter | UI 仍有无跟踪结果的回退路径。 | 不以“已支持多人”为理由删除；需特定兼容场景测试。 |
| LBPH 配置/旧库校准/历史 docs | 非默认，但有明确配置、CLI 与回归。 | 移除将改变功能范围和数据兼容；不建议。历史 LBPH 说明的相对“操作手册.md”链接需修正，不是删文档。 |
| `datasets/live_mixed.py` 与真人协议 | 采集被用户取消，不等于工具无用或获准删除。 | 保留备用，不启动相机；不把保留误写为已完成真人验证。 |
| 可选 `arcface.py` 与模型对照 | 已有受限研究功能及历史结果，不是本轮新增项。 | 不新增模型，也不删除现有复现能力或改默认；保留许可/签名限制。 |
| `.docx_work/` 四个历史报告脚本 | 盘点到独立报告用途，不属于当前识别运行路径。 | 用户本轮不做报告，保留、不运行、不批量清理。 |

## 11. 重构前性能基线与保护要求

以下是**既有实验值，本轮只读复核产物**，不是此次审计新跑出的优化结果。

| 负载/口径 | reference p50/p95 ms | 当前 batch p50/p95 ms |
| --- | ---: | ---: |
| 1,000 合成身份，各 3 模板；同 SQLite/同查询交替配对 | 9.144625 / 9.303508 | 0.274125 / 0.299821 |
| 1,000 合成身份，1/2/3/5/8 模板循环；同库配对 | 9.239521 / 9.493700 | 0.333104 / 0.368622 |

六组规模/模板分布共 1,200 次查询完整排名和距离逐项相同；36,000 个开放集边界判定检查。详见 [P2_RETRIEVAL_RESULTS.md](../../face_research/P2_RETRIEVAL_RESULTS.md) 及对应 metrics.json。

当前静态基准：1,000 合成身份、单个 120×120 输入的检测至身份排序 p50/p95 **17.914/18.573 ms**；embedding p50 **16.126 ms**；matching p50 **0.383 ms**；重复同图双份布局 **35.336/36.974 ms**。JSON 名称 `end_to_end_ms` 不包括相机、UI、解码、阈值判定和时序确认；不得冒充真实应用端到端 FPS。

已有实际服务/跟踪六阶段视频 profiler smoke 可保护接线，但其重复静态图/空白布局仍不是实际运动数据。环境为本地 macOS/arm64，记录 OpenCV 请求 1、返回 8，不能称单核实测。

后续重构需要同输入/配置/环境复测；数值/排名/判定必须保持，时延允许正常测量波动，出现可重复的明显退化应保留旧实现。不得以修改基准范围、删慢样本或降低质量门槛换“改善”。**本轮没有重构后数据；所有 before→after 的 after 栏应保持未测。**

## 12. 本轮交付与停机点

交付仅为本文件与 [REFACTOR_PLAN.md](../planning/REFACTOR_PLAN.md)。没有删代码/测试，没有改阈值/质量/跟踪/存储，没有新增模型/框架，没有声称重构完成或性能提升。

建议用户先审批 R01→R02→R03；每项单独完成相关测试和全量回归后再进行下一项。R07、R10 需要独立审视异常/边界行为，其他 P1 可择需，P2 默认保留。完成两份文档后停止，等待审核。
