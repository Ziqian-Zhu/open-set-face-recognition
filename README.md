# 开放集多人脸识别系统

[![简体中文（当前）](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E5%BD%93%E5%89%8D-0F766E?style=for-the-badge)](./README.md)
[![Switch to English](https://img.shields.io/badge/English-Read-52677A?style=for-the-badge)](./README.en.md)

## 项目简介

这是一个基于 Python 和 OpenCV 的本地开放集人脸识别项目，面向摄像头实时画面与本地视频。它不仅查找“最像谁”，还要判断画面中的人是否已录入、当前画质是否足够，以及识别结果是否需要继续观察几帧才能确认。画面中出现多个人时，系统会分别处理每张脸和对应的轨迹。

项目将**人脸检测 → 关键点对齐 → 质量检查 → 特征提取 → 向量检索 → 开放集判定 → 多帧确认**串成可操作的桌面应用。`face_compare_system` 负责人员录入、实时识别、视频回放和本地数据管理；`face_research` 用于研究有限样本下的模板选择、阈值校准与性能取舍。YuNet 和 SFace 是预训练模型，本项目的重点是围绕它们构建完整、可解释、可测试的识别系统。

## 项目亮点

- **从画面到特征的完整链路：** YuNet 在一帧中检测多张人脸并给出五点关键点；通过人脸对齐和画质检查后，SFace 提取可用于比较的特征向量。检测、画质不合格的情况会单独处理，不直接当作“陌生人”。
- **开放集身份判定：** 匹配不仅看最近的样本，还按人员聚合多个模板的距离，并同时检查最佳身份的距离与前两名身份的差距。库中没有这个人，或两个身份过于接近时，系统可以拒识或继续确认。
- **多人独立跟踪与多帧确认：** 将位置和外观信息结合，为画面中的人脸维持各自轨迹；每条轨迹独立累计识别结果，再由连续帧决定是否显示稳定身份，避免仅凭单帧波动直接确认。
- **兼顾外观变化的录入流程：** 同一人可保存多张样本，也可混合录入戴镜、摘镜状态。录入时检查模糊、亮度、重复采集及身份冲突，避免把明显不合适的样本直接写入人员库。
- **面向实际使用的桌面交互：** 提供摄像头选择、实时预览、采集进度、人员管理及本地视频回放。界面中的“未知”“待确认”和画质提示对应不同处理状态，不把所有未识别结果混为一类。
- **本地存储与检索优化：** SQLite 保存人员和人脸向量；缓存与批量化精确检索减少多身份匹配时的重复计算。推理任务优先处理最新画面，整个识别流程可在本机 CPU 运行。
- **独立的研究与评估模块：** `face_research` 比较不同模板选样方法，并提供阈值校准、模型对照和性能测试入口。应用功能与实验流程分开，便于复现、分析和继续扩展。

## 启动方式

需要 Python 3.10+ 和 Tk 8.6+；摄像头演示还需要可用相机。在仓库根目录运行（macOS / Linux）：

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

首次运行会下载并校验模型文件；之后识别可离线进行。启动后打开摄像头，先录入人员，再进行识别。更多安装与操作说明见 [应用指南](face_compare_system/README.md)。
