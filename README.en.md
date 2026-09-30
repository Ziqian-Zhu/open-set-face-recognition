# Open-Set Multi-Face Recognition System

[![Read in Chinese](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E9%98%85%E8%AF%BB-52677A?style=for-the-badge)](./README.md)
[![English (current)](https://img.shields.io/badge/English-Current-0F766E?style=for-the-badge)](./README.en.md)

## Overview

A local face-recognition project built with Python and OpenCV. It recognizes multiple people in camera or video footage and can reject unknown or ambiguous identities. The repository contains a desktop application, `face_compare_system`, and an experiment module, `face_research`. It uses pretrained YuNet and SFace models; the focus is on the recognition pipeline, open-set decisions, and engineering implementation.

## Highlights

- **End-to-end vision pipeline:** YuNet detects faces and five landmarks; alignment and quality checks precede SFace feature extraction and identity matching.
- **Open-set, multi-face recognition:** A distance threshold, top-two identity margin, and per-track multi-frame confirmation keep unknown or ambiguous faces rejected or pending.
- **More robust enrollment:** Multiple samples per person, blur/exposure and duplicate checks, and support for enrolling the same person both with and without glasses.
- **Local vector retrieval:** SQLite stores face embeddings; caching and batched exact search reduce matching overhead across many identities. Recognition runs locally on the CPU.
- **Reproducible experiments:** `face_research` supports template selection, threshold calibration, model comparisons, and performance evaluation to study design trade-offs.

## Quick Start

Requires Python 3.10+ and Tk 8.6+; a camera is needed for the live demo. From the repository root (macOS / Linux):

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

The first setup downloads and verifies the model files; recognition can then run offline. Open the camera, enroll people in the application, and then try recognition. For more installation and usage details, see the [application guide](face_compare_system/README.md) (Chinese).
