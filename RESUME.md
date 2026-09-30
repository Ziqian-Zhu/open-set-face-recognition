# 简历项目表述：可核验版本

日期：2026-09-26。根据当前实现与公开数据探索撰写；真人采集已取消。使用前应能亲自解释并复现下面对应代码，不把第三方模型或协作完成的部分说成独立原创。

## 简历正文

**开放集多人脸识别与有限模板选样系统**  
Python · OpenCV · NumPy · SQLite · Open-set Recognition

- 实现 YuNet＋SFace 本地多人脸识别流水线，集成五点对齐、质量门控、128 维归一化特征、SQLite 模板管理及缓存精确检索，结合距离阈值、身份歧义间隔与轨迹级多帧确认处理未知身份。
- 针对有限录入模板预算，设计质量－外观覆盖－身份一致性约束的贪心选样策略，建立 5 类选样、6 种聚合与 K=1/2/3/5 对照；在 MeGlass 自定义混合戴镜协议完成 168 组合探索（K=5 为后补分析），保留未优于基线的结果与失败分析。
- 建立 validation-only 阈值/margin 标定及开集评价流程，区分 1:N FPIR 与 1:1 FAR；在身份隔离的公开戴镜／摘镜协议完成 1:1 独立校准，补充采集阶段失败归因与共享人员分组统计，单列低 FAR 样本不足；实现泄漏检查、冻结凭据和产物校验，累计 449 项回归测试通过。
- 定位逐身份聚合瓶颈并实现分组批量精确检索；在同机、同库、200 次交替配对查询下，将 1,000 合成身份×3 模板的检索 p50 从 9.145 ms 降至 0.274 ms，跨规模/分布 1,200 次查询完整分数和排名一致；不将微基准扩张为相机 FPS 或准确率结论。

版面较紧时优先用第一、第三、第四条，把选样实验作为面试展开内容。不要增加“准确率 99%”“摘镜鲁棒性提升 X%”“原创 SFace”“百万级实时检索”等未经验证描述。

## 60 秒介绍

我做的是一个本地开放集多人脸识别系统，但研究重点不是换一个识别网络，而是固定 YuNet/SFace 后，分析模板有限、外观变化和陌生人拒识的取舍。我把身份级聚合与 threshold＋margin 判定做成共享接口，在研究端建立模板预算、选样和聚合矩阵，只在 validation 定参数，并记录 test 访问与冻结证据。

我自己的选样规则结合质量、外观覆盖和同人一致性，但公开混合戴镜协议的主比较是 13/24，对照是 14/24，没胜出。我没有把结果包装成提升；反而通过质量消融发现这批裁剪图里，完整门控挡住了 5 张仍能正确识别的 probe。性能方面，我定位到逐身份小数组聚合开销，用分组批量计算保持精确排名，在 1,000 合成身份×3 模板同库配对基准上把检索 p50 从约 9.1 ms 降到 0.27 ms。工程上有多人轨迹、SQLite、异步 UI 和 449 项回归；还补上了不改识别结果的失败归因和人员分组统计。真实多人运动、独立低 FAR 和活体检测还没有验证。

## 证据索引与追问边界

| 简历主张 | 可以打开的证据 | 不应延伸为 |
| --- | --- | --- |
| 完整流水线与本地存储 | `face_compare_system/face_compare/{deep_engine,service,vector_database,tracking}.py` | 自己训练/发明 YuNet 或 SFace |
| 自定义有限预算选样 | `face_research/selection.py`；P0 公共矩阵的固定主比较 | 已超过 First-K 或学术 SOTA |
| 168 组合公开探索 | `meglass-mixed-metric-fix-20260926-01`；`METRIC_CORRECTION.md` | 全数据集官方基准；K=5 初始盲测注册；真实相机测试 |
| 质量误拒分析 | `meglass-mixed-quality-metric-fix-20260926-01` | 取消过滤安全；已完成相机摘镜验证 |
| 公开 1:1 受控校准 | `meglass-verification-20260926-01`；`VERIFICATION_RESULTS.md` | 真实跨会话验证；只报 88/89 而隐去全部 88/200；低 FAR 保证 |
| 失败归因与人员分组统计 | `meglass-acquisition-diagnostics-20260926-01`；`ACQUISITION_DIAGNOSTICS.md` | 已修好质量门控；眼镜因果效应；bootstrap消除了会话偏差；新的盲测 |
| 可选模型接口与对照 | `sface-arcface-verification-20260926-01`；`MODEL_COMPARISON_RESULTS.md` | 发明 ArcFace；总体100%准确；权重可商用；已切换线上模型 |
| 检索规模测试 | `static-synthetic-benchmark-p1-20260926-02` | 真实身份分布精度或并发在线延迟 SLA |
| 精确检索性能优化 | `exact-retrieval-paired-20260926-01`；`P2_RETRIEVAL_RESULTS.md` | 准确率提升；整个识别应用快 33 倍；已做 FAISS 比较 |
| 时序配对能力 | `video_evaluation.py`、`test_video_evaluation.py` | 已证明真实交叉遮挡场景收益 |

run ID 均位于 `face_research/results/`。完整结果与分母见 [P1_RESULTS.md](face_research/P1_RESULTS.md)，逐题答辩见 [INTERVIEW.md](INTERVIEW.md)。输出展示以聚合统计为主，面试时不要直接提供原图、embedding 或私人数据库。
