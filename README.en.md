# Open-Set Multi-Face Recognition System

[![Read in Chinese](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E9%98%85%E8%AF%BB-52677A?style=for-the-badge)](./README.md)
[![English (current)](https://img.shields.io/badge/English-Current-0F766E?style=for-the-badge)](./README.en.md)

## Overview

This is a local, open-set face-recognition project built with Python and OpenCV for live camera feeds and local video. Rather than simply returning the closest face in a gallery, it asks whether that person is enrolled, whether the image quality is sufficient, and whether the result needs more frames before confirmation. When several people appear together, each face and its track are handled independently.

The project connects **face detection → landmark alignment → quality checks → feature extraction → vector retrieval → open-set decision → multi-frame confirmation** in an interactive desktop application. `face_compare_system` handles enrollment, live recognition, video playback, and local data management. `face_research` studies template selection under limited sample budgets, threshold calibration, and performance trade-offs. YuNet and SFace are pretrained models; the contribution here is the complete, interpretable, and testable system built around them.

## Highlights

- **Complete path from frame to embedding:** YuNet detects multiple faces and five landmarks in each frame. After alignment and quality checks, SFace extracts embeddings for comparison. Detection and quality failures are handled separately instead of being mislabeled as unknown identities.
- **Open-set decisions beyond nearest neighbor:** Distances from multiple templates are aggregated by identity. The best identity must pass both a distance threshold and a margin against the runner-up; missing or ambiguous identities can remain rejected or pending.
- **Independent tracks and temporal confirmation:** Position and appearance cues associate faces across frames. Each track accumulates its own recognition evidence, so a single unstable frame does not immediately confirm an identity.
- **Enrollment designed for appearance changes:** A person can have multiple templates, including samples with and without glasses. Blur, brightness, duplicate captures, and identity conflicts are checked before samples enter the gallery.
- **Practical desktop workflow:** The application provides camera selection, live preview, collection progress, person management, and local video playback. Unknown, pending, and low-quality states are presented separately instead of collapsing all failures into one label.
- **Local storage and retrieval optimization:** SQLite stores people and embeddings; caching and batched exact search avoid repeated work as the gallery grows. Inference prioritizes recent frames, and the recognition pipeline can run locally on a CPU.
- **Separate research and evaluation module:** `face_research` compares template-selection strategies and provides entry points for threshold calibration, model comparison, and performance tests. Keeping experiments separate from the application makes the system easier to reproduce, analyze, and extend.

## Quick Start

Requires Python 3.10+ and Tk 8.6+; a camera is needed for the live demo. From the repository root (macOS / Linux):

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

The first setup downloads and verifies the model files; recognition can then run offline. Open the camera, enroll people in the application, and then try recognition. For more installation and usage details, see the [application guide](face_compare_system/README.md) (Chinese).
