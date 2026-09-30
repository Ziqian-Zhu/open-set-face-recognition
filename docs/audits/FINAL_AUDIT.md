# Final Project Audit

日期：2026-09-28。范围：当前 `face_compare_system`、`face_research`、配置、测试、现存实验产物及 `README.md`／`RESUME.md`。本轮只做只读检查、临时库验证和本报告；没有修改功能代码、配置、测试、人员库或既有实验结果。以下“通过”不等于真实相机、真实多人运动或工业级生物识别认证。

## 1. Executive Summary

**最终判定：NEEDS FIX。** 主应用的 YuNet→质量门禁→五点对齐→SFace 128 维→身份聚合→开放集阈值／Top2 间隔→独立轨迹投票→UI 链路仍在；449 项测试全通过，公开图像及合成视频的模型连通性检查通过，精确检索性能与排名等价性在当前代码上重新得到验证。研究框架的 168 个组合、验证集冻结／测试集评估顺序和现存结果校验也有直接证据。项目已有可讲清的工程与实验贡献，简历中的性能数字没有冒充准确率。

但旧命令行 `select-threshold` 的 margin 比较与线上／主要研究路径使用的共享判定函数不一致；一个可复现的浮点边界案例会让它推荐完全不同的阈值。鉴于本次验收特别要求**算法语义一致**，不能仅凭测试全绿签发“整个项目最终验收通过”。这不表示已经发现真实人脸错认，也不影响默认 SFace 固定阈值或 168 组合的主标定路径。

另一个必须分清的事实：`REFACTOR_AUDIT.md` 和 `REFACTOR_PLAN.md` 明写“仅审计、待批准”，计划中的 R01–R13 **没有实施**。因此不能声称“精简化＋工程化＋保守重构已经完成”或“重构前后行为不变”；能证实的是**现有版本**功能与基准状态。历史实验产物完整，但当前研究源码与当时记录的源码哈希已有差异；不能把历史 run 称为当前源码的逐字重跑。

桌面启动脚本在本次运行环境中未完成：`check_startup.py` 于 `tk.Tk()` 报 Tk 9 资源／显示缩放初始化错误，尚未进入应用逻辑；这不是已证实的 UI 代码缺陷，也不能记为 GUI 验收通过。本轮未打开真实摄像头或使用现有私人库。

## 2. Critical Issues

1. **旧阈值选择命令与实际开放集判定不一致。** `face_compare_system/face_compare/operating_point.py:select_operating_point` 用 `second - distance >= margin` 直接模拟；主链路 `face_compare_system/face_compare/decision.py:accept_identity` 对等号附近采用绝对容差 `1e-12`。实测 `0.315 - 0.275` 为 `0.03999999999999998`：旧工具在 margin `0.04` 下拒绝，主链路接受。构造一个可用已知样本（最佳距离 `0.275`、第二距离 `0.315`）和一个较远的可用未知样本，旧工具在目标 FPIR=0 时选出 `1e-8`，而按实际接受规则该已知样本在 `0.275` 可通过。影响范围是 CLI `select-threshold` 推荐结果，不是默认线上判定或 `face_research/evaluation/calibration.py` 的共享路径。**最小修正建议**：仅在旧扫描中复用 `accept_identity` 的 margin 判定，保留原候选阈值集合、返回 schema 和“不自动写配置”行为；先补十进制边界及 one-ULP 邻居回归，再检查旧工具选点、主识别、研究标定的一致性并跑全量测试。此为行为修正，不应伪称无行为变化，也不应改写旧实验文件。

## 3. Major Issues

1. **无法签发当前桌面环境的启动／摄像头验收。** `face_compare_system/scripts/check_startup.py` 本次在创建 Tk 根窗口时失败：当前 `.venv-ui` 所用 Tk 9 报缺少可用 `tk.tcl`，内部 `tk scaling` 出现 NaN。公开图像 `smoke_models.py`、`smoke_quality.py` 和 `smoke_video.py` 均通过，但只验证模型、质量信号和静态图重复写成视频的双脸链路。**最小建议**：在支持桌面显示、Tk 资源完整的目标 Mac 环境用临时 SQLite 库运行现有 `check_startup.py`／`check_ui.py`，随后人工检查内置相机打开、录入、识别、删除和视频播放；记录 Python/Tk 与结果，不要把此处环境失败直接归因于 UI 算法。
2. **历史研究 run 不等于当前源码的完全可复现重跑。** `meglass-mixed-metric-fix-20260926-01` 的 `course_code_sha256` 与当前应用包一致，但 `research_code_sha256` 有 4 个模块差异（`acquisition.py`、`evaluation/artifacts.py`、`evaluation/grouped.py`、`verification_diagnostics.py`）。历史检索 run 的源码也与当前版本不同，故本轮专门在当前版本重跑检索。8 组主要结果目录的文件校验均通过，168 组合冻结 receipt 与结果一致；校验和证明产物未变，不提供旧源码本身。**最小建议**：未来发布 run 时同时保留可定位的源码版本／归档及依赖锁定，既有 run 保持历史版本身份；若要声称“当前代码重现 168 数字”，需合法数据可用时新建 run ID、完整重跑且承认 test 已被重复查看。

## 4. Minor Cleanup

- 说明与代码有局部错位：`face_compare_system/face_compare/__init__.py` 和 `recognizer.py:FaceRecognizer` 仍偏向“LBPH 是默认”；`ui.py:_save_enrollment` 的 docstring 写“calibrate”，SFace 实际不根据录入库自动校准；研究 `_baseline.py` 的“唯一导入边界”说法已不准确。这些是 R01 的待批准说明纠偏，不是本轮已完成项。
- `service.py:_enroll_samples` 的跨身份高相似度常数 `.65`、UI 多处不同含义的 1 秒时效值可分别命名；研究中的选样常数已具名。当前有效配置并无冲突：发布的 `config.json` 为 SFace／SQLite，`cosine_threshold=0.45` 换成半余弦距离 `τ=0.275`，`cosine_margin=0.08` 换成距离 `m=0.04`；`RecognitionConfig` 的 `0.43/0.025` 是兼容 LBPH 的类默认，不是当前 SFace 值。
- `ui.py` 和矩阵实验编排较长，跨层阅读需要跳转，但 UI 没复制核心识别公式，研究端没有改写生产库；未见仅为“设计模式”而增加的 Manager／Factory／Adapter 链。少量文件 SHA256、dHash、旧／新校准入口的重复有历史接口和协议差异，不宜为降 LOC 一次性合并。`worker.py` 的四／五元组及两个时钟可以补契约说明，而无需新抽象层。
- 精确异常扫描显示：`worker.py` 的 `except Exception` 会把错误交给 UI 并清除轨迹；相机扫描、UI 校准会显式显示错误；队列 `pass` 只处理预期的空／满；模型哈希或数据库失败不会伪装成成功。CLI 个别异常路径的数据库关闭、预筛中途失败的资源释放仍可按 R07 单独改善；当前不是算法错误。

## 5. Core Algorithm Verification

实际主链路由 `main.py`／`ui.py` 的相机或 `video.py` 的视频输入，交给 `service.py:analyze_frame`，再由 `worker.py`＋`tracking.py` 输出稳定结果。研究包从应用包导入共享聚合和接受规则；应用包未导入研究包。已检查的调用路径未出现循环导入。

| 环节 | 结论 | 实际代码与语义 |
| --- | --- | --- |
| Detection | PASS | `deep_engine.py:YuNetDetector` 校验模型尺寸／SHA，按 `detection_score` 创建 OpenCV YuNet；缩放后使用实际宽高比 `sx/sy` 分别还原框和五点坐标，不只取第一张脸。空帧返回空列表，过小框和无效关键点被过滤。 |
| Alignment | PASS | `SFaceExtractor.align` 将 `[x,y,w,h,10 个 landmark 数值,confidence]` 传给 `alignCrop`；应用保留 BGR，输出由 `extract` 强制检查为 `112×112×3`。公开图像模型 smoke 已穿过该路径。 |
| Embedding | PASS WITH NOTES | `SFaceExtractor.extract` 得到 `float32`、128 维并在提取边界 L2 归一化一次。SQLite 缓存重建和研究 `GalleryIndex` 对入库／输入向量又做**防御性单位化**，查询也单位化；这不是再次调用模型或更换特征空间，标量／批量数值已对照，但“全链路只归一化一次”若按字面说法并不成立。单张对齐脸推理是现有接口；未单独证明一个不存在的批量 SFace 推理 API 与单张 API 等价。 |
| Matching | PASS | 生产和研究都采用半余弦距离 `d=clip((1−cos)/2,0,1)`，越小越近；显示 `similarity=100×(1−d)` 只是匹配分、不是身份概率。`aggregation.py` 共享 nearest／mean／median／top-k mean／top-k median，默认最近 `min(3,N)` 张的距离中位数；SQLite 用真实模板数分组做批量归约，不丢模板或第二身份，完整排序保留 `(distance, person_id, name)` 旧并列规则。研究无 person_id，并列按名称，不能把两个不同 API 的完全同分身份顺序说成相同。 |
| Open-set Decision | NEEDS FIX | 主应用 `FaceRecognizer.match` 与研究 `selection.decide` 共享 `accept_identity`：最佳距离 `≤τ`，有第二身份时 `d₂−d₁≥m`（含浮点等号容差）；空库／单身份／超阈值／歧义均有处理。只有旧 CLI 阈值模拟不复用该规则，见 Critical Issue。 |
| Tracking | PASS WITH NOTES | `tracking.py` 按位置／IoU／外观关联，每轨迹独立 `MultiFrameVoter`；近并列关联新建轨迹，丢帧、质量失败、时间回退、库 revision 变化会清票，容量有界，同一身份被两轨迹同时确认会清除冲突。UI 多人时展示每轨迹状态；真实交叉、遮挡和持续运动尚无真值评估。 |
| Template Selection | PASS WITH NOTES | `face_research/selection.py` 的 coverage 贪心目标由质量代理分、稀有外观覆盖增益和同人一致性软约束组成；只读已验证 gallery 模板，不读 test query。5 类方法含 Random 三个预定 seed；覆盖法是工程假设，不是训练得到的新模型。主比较 13/24，低于 First-K 14/24，文档未声称正向增益。 |
| Evaluation | PASS WITH NOTES | `matrix.py` 先对全部变体做 validation 选样／标定并形成冻结 receipt，再打开 test；`metrics.py` 分开 1:N FPIR 与 1:1 FAR、采集失败与可用脸拒识；`video_evaluation.py` 强制完整逐帧标注并保留无人帧。公开数据缺真实 session 元数据，MeGlass 预筛／多轮探索也不是封存盲测；真实视频源未取得。 |

模块职责复核：`main.py` 仅入口（PASS）；`ui.py` 为展示、采集与设备交互（PASS WITH NOTES：较长且本次桌面启动未完成）；`service.py` 编排业务而不绘制 UI（PASS）；`deep_engine.py`、`quality.py`、`recognizer.py` 分别负责模型、质量、身份决策（PASS，旧 docstring 除外）；`vector_database.py` 负责事务／缓存／距离排名，不自行作 Known 决策（PASS）；`database.py` 保留 LBPH／文件库兼容（PASS WITH NOTES）；`worker.py` 管理最新帧及错误传递，`tracking.py` 管理独立时序状态（PASS）；`face_research` 管理实验／评测／产物（PASS WITH NOTES：历史源码版本及公开协议限制）。

## 6. Test Verification

执行：`env PYTHONDONTWRITEBYTECODE=1 face_compare_system/.venv-ui/bin/python -m pytest -q face_compare_system/tests face_research/tests -p no:cacheprovider`。结果：**total 449、passed 449、failed 0、skipped 0**（6.47 s）。与此前审计记录的 449 项一致；不能把 449 当成覆盖率百分比或真人准确率。代码级测试之外，三个临时库 smoke 通过；`check_startup.py` 的 Tk 环境失败已在 Major Issues 单列。

关键边界逐项核对：空库／单身份／全部超阈值（`test_recognizer.py`、`test_decision.py`）；每身份一张、模板不足 K、距离相同、Top1/Top2 并列、阈值与 margin 边界（`test_aggregation.py`、`test_vector_database.py`、`test_decision.py`）；NaN／零向量／维度错误（`test_vector_database.py`、`test_upgrade.py`）；空检测和无效五点／图像（`test_upgrade.py`、`test_refinements.py`、服务输入检查）；同时双脸、丢轨、切人、同身份冲突、时间倒退（`test_tracking.py`、`test_recognizer.py`、`smoke_video.py`）；文件库缺失特征、SQLite 事务回滚／模型签名冲突（`test_upgrade.py`、`test_vector_database.py`）；模型缺失或不完整在 `verified_model` 明确报错。**建议补窄测试而非凑数量**：旧 CLI 浮点边界是现有覆盖缺口；SQLite 缓存读取被人为损坏的 vector blob、SFace 模型文件缺失可各有一个明确回归，以强化失败闭合证据。

## 7. Benchmark Verification

本轮直接调用现有 `face_research/retrieval_benchmark.py:run_retrieval_benchmark`，保持默认 seed=42、查询 seed=43、128 维、200 次查询／配置、20 次预热、单 Python worker。主要目标 `uniform_3`、1,000 **合成身份×每人 3 模板**：标量 baseline p50 **9.138979 ms**，分组 batch p50 **0.274854 ms**，**33.25×**。历史产物同条件为 9.144625→0.274125 ms，33.36×；合理波动，不要求逐位相同。混合 1/2/3/5/8 模板、1,000 身份本轮为 9.183041→0.332646 ms，27.61×。

方法公平性：两路使用同一临时 SQLite 库、同一缓存矩阵和相同查询；先分别预热，查询时交替先后顺序，计时都包含 revision 检查、矩阵乘法、聚合和全身份排序，均不包含磁盘建库／YuNet／SFace／UI／摄像头 I/O。6 种负载共 **1,200 查询**，每次完整排名和分数严格相等，并作 **36,000 次**开放集判定比对。另用独立临时库、独立标量 NumPy oracle 对 1,000 身份×3 模板的 **128 个新随机 query** 逐条校验完整排名／身份，分数差 ≤`1e-7`，且 **1,920 次**阈值／margin 判定一致。该验证不是仅比较 Top1。

限制：OpenCV 请求线程数为 1，但环境回报 8；两路同进程仍具可比性，跨机器绝对毫秒与原生 BLAS 线程可变。没有并发或整体摄像头 FPS 测量；**此 benchmark 衡量精确 retrieval 性能和实现等价性，不是人脸识别准确率**，也非 ANN／FAISS 比较。

## 8. Experiment Verification

- **168 可追溯**：`matrix.py` 对每个 K∈{1,2,3,5} 运行 First、Quality、Diversity、Coverage 各一个 seed，加 Random 三个 seed，共每 K 七组选样；乘 6 种聚合，得到 `4×7×6=168`。当前修正产物 `meglass-mixed-metric-fix-20260926-01/metrics.json` 的 `variants`、`frozen_validation.variants` 均为 168；四字段组合去重后仍为 168，168 条都有 test identification、validation 阈值和所选模板摘要，无失败条目混充成功。K=5 为事后预算扩展，不能冒充初始预注册主比较。
- **冻结与隔离**：`matrix.py` 在读取 test 清单前计算全部 validation 工作点和模板，并提供 `on_freeze`／`on_test_open` 审计钩子；冻结 SHA 与 `metrics.json`、旁侧 `.audit/frozen_validation.json` 一致，168 条 threshold/margin 与 test 结果引用值一致。重复图、对齐图哈希、身份和可用 session 交叉有检查；原公开裁剪图没有真实 session 元数据，本 run 显式 `require_session_ids=false`，所以**不能证明跨会话独立**。对 test 的预筛与反复查看意味着该数据集是探索性评估，不是完全封存盲测；代码级“单次 run 不拿 test 调参”可证，研究者层面的盲性不可证。
- **数值与分母**：固定主比较 First-K 为 **14/24**、Coverage 为 **13/24**，没有性能提升主张。质量消融 full **14/24**、none **19/24** 是此样本的误拒线索，不证明线上安全放松。公开 1:1 验证 validation/test 各 200 个身份，test 可用同人配对接受 **88/89**，全部同人尝试 **88/200**；111 组同人采集失败；可用异人 0/44 误接受的 Wilson 95% 上界约 **8.03%**，低 FAR 的 TAR 状态为 `insufficient samples`。这些数与 1:N 的 FPIR 不混算。
- **复现凭据**：产物记录协议、manifest/config/模型与源码 SHA、seed、预算、策略、聚合、τ/m、失败状态和完整统计；8 个主要结果目录通过 `verify_result_directory`，但 SHA 不是历史源码可恢复性的替代物。合成检索显式标 `synthetic_performance_and_equivalence_only`。不重新使用旧 run ID，不把旧 test 再跑一遍称为独立新测试。公开原图、私人向量与人员库不在报告中复制。

## 9. Resume Claim Verification

逐条对应 `RESUME.md` 正文的四条：

1. **SUPPORTED** — `service.py`／`deep_engine.py`／`quality.py`／`vector_database.py`／`tracking.py` 及公开双脸 smoke 证明本地 YuNet＋SFace、128 维、SQLite 精确排名、开放集和分轨确认。只支持功能链路，不支持真实多人运动识别率。
2. **SUPPORTED** — `selection.py` 与 `matrix.py` 及 168 条产物证明设计、5 类选样、6 聚合、K=1/2/3/5 的探索；文案如实写 K=5 后补、主方法未胜过基线。它不是“算法提高准确率”的证据。
3. **PARTIALLY SUPPORTED** — 主研究路径确实 validation-only 冻结，1:N／1:1 分母、身份隔离 1:1、失败归因和 449 回归可核查；但“独立校准”只可理解为身份与 validation/test 数据划分，不能延伸为跨 session 或封存盲测。旧 CLI `select-threshold` 另有 Critical Issue，故不宜把“全项目阈值流程完全一致”作为面试表述。
4. **SUPPORTED** — 历史产物直接支持 **9.145→0.274 ms、约 33×** 的限定条件；本轮当前代码重跑为 9.139→0.275 ms、33.25×，完整排名／分数等价。文案已经限定为同机同库、200 次交替配对、合成身份、检索 p50，未谎称整机 33× 或精度提升。

没有在根 README／上述四条中发现缺乏来源却宣称的“真实相机准确率”“自训模型”或“低 FAR 保证”。`README.md` 所列 449、168、阈值换算、性能和 1:1 分母均找到代码／测试／产物依据；其中历史数字必须带 run ID 和协议限制。

## 10. Remaining Limitations

真人相机独立采集已按用户决定取消；当前缺真实运动多人视频真值、跨 session 或跨设备验证和遮挡／换人压力评估。公开数据经过裁剪和预筛，只有有限数量未知样本；低 FAR、人口级风险、公平性与戴镜／摘镜泛化均不能保证。未加入活体检测，系统不适合高风险身份认证。合成 gallery 检索不等于人脸精度、摄像头 FPS 或并发 SLA。模型为第三方预训练权重；自有贡献在系统集成、质量／模板预算实验、精确检索优化与诚实评估。软删除可恢复，并非合规意义的不可恢复擦除。当前 Tk 环境与真实摄像头端到端启动仍需目标机器复验。

## 11. Final Acceptance

**NEEDS FIX。** 必须先修复第 2 节唯一的共享接受规则／旧阈值扫描不一致，并以边界回归、449 项全量测试和阈值选点对照验证。不要为了此修复重构目录、切换模型或覆盖历史产物。第 3 节的桌面运行复验和历史源码版本化也应在对外声称“完整可运行／完全复现当前版本”前完成；它们不是更换算法的理由。修复后可以重新审视当前版本是否达到 `ACCEPT`；本轮的 `NEEDS FIX` 不否定现有代码、实验与受限措辞的简历价值，但明确否定“保守重构已完成且最终验收无条件通过”的说法。
