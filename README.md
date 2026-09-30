# 开放集多人脸识别系统

[![简体中文（当前）](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E5%BD%93%E5%89%8D-0F766E?style=for-the-badge)](./README.md)
[![Switch to English](https://img.shields.io/badge/English-Read-52677A?style=for-the-badge)](./README.en.md)

## 项目简介

一个基于 Python、OpenCV 的本地人脸识别项目，支持摄像头和视频中的多人识别，并能拒绝未录入或身份不明确的人。项目由桌面应用 `face_compare_system` 和实验模块 `face_research` 组成；使用预训练的 YuNet、SFace 模型，重点在识别流程、开放集判定和工程实现。

## 项目亮点

- **完整视觉链路：** YuNet 检测人脸并定位五点关键点，经过对齐、画质检查后，用 SFace 提取特征并匹配身份。
- **开放集与多人识别：** 结合距离阈值、前两名身份间隔及逐轨迹多帧确认；对未知人员和模糊匹配保持拒识或待确认状态。
- **更稳健的录入：** 每人保存多张样本，检查模糊、曝光和重复采集；支持同一人混合录入戴镜、摘镜照片。
- **本地向量检索：** 使用 SQLite 管理人脸特征，通过缓存和批量化精确检索降低多身份匹配开销；识别在本机 CPU 完成。
- **可复现实验：** `face_research` 提供模板选样、阈值校准、模型对照与性能评估，方便分析不同方案的取舍。

## 启动方式

需要 Python 3.10+ 和 Tk 8.6+；摄像头演示还需要可用相机。在仓库根目录运行（macOS / Linux）：

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

首次运行会下载并校验模型文件；之后识别可离线进行。启动后打开摄像头，先录入人员，再进行识别。更多安装与操作说明见 [应用指南](face_compare_system/README.md)。
