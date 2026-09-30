# Open-Set Multi-Face Recognition System

[![Read in Chinese](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E9%98%85%E8%AF%BB-52677A?style=for-the-badge)](./README.md)
[![English (current)](https://img.shields.io/badge/English-Current-0F766E?style=for-the-badge)](./README.en.md)
[![Tests](https://github.com/Ziqian-Zhu/open-set-face-recognition/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Ziqian-Zhu/open-set-face-recognition/actions/workflows/tests.yml)

![Face recognition workbench preview](docs/assets/app-preview.png)

*Actual application UI with a temporary empty gallery and the camera disabled; no real face, name, or embedding is included.*

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

### Quantified Comparisons

Compared with a single-image, nearest-neighbor demo, this project adds quality gates, unknown-identity rejection, independent multi-face tracks, and multi-frame confirmation. Those are **functional design differences**, not a measured accuracy ranking against third-party systems. The numbers below compare implementations and operating points within this project.

| Comparison | Result | Scope |
| --- | --- | --- |
| Earlier per-identity scalar aggregation → grouped batched exact retrieval | With 1,000 synthetic identities × 3 templates each, retrieval p50 fell from **9.381 to 0.289 ms (32.44× faster)**. Full scores and rankings matched across 1,200 queries, as did 36,000 final decisions. [Final acceptance review](docs/audits/FINAL_AUDIT_V2.md) (Chinese) | Retrieval latency only—not camera FPS or recognition-accuracy improvement. |
| Earlier Laplacian blur threshold → regional, noise-corrected quality measure | On the same public sample after contrast reduction, the old variance **42.62 < 55** caused rejection; the new detail score **0.676 > 0.30** passed, while a simulated blurred image still failed at **0.103 < 0.30**. [Quality regression record](face_compare_system/验证记录.md) (Chinese) | A public-image and synthetic-degradation regression case, not proof of a higher live-person enrollment rate. |
| Fixed engineering threshold → validation-only calibrated threshold | In an exploratory public glasses/no-glasses **1:1** test, accepted usable genuine pairs increased from **78/89 to 88/89**; across all genuine attempts, the result was **88/200**. [Pair verification](face_research/VERIFICATION_RESULTS.md) (Chinese) | Not evidence of improved production 1:N recognition. Only 44 usable impostor pairs were available, too few to establish low false-accept risk. |

The engineering regression suite passed **454/454 tests**, including open-set boundaries, multi-face tracking, database behavior, and retrieval equivalence. See the [acceptance review](docs/audits/FINAL_AUDIT_V2.md) (Chinese) for scope. Limited-budget template selection remains a research comparison: the custom Coverage strategy did **not** beat the First-K baseline (13/24 versus 14/24), so it is not presented as an accuracy gain. See [selection results](face_research/P2_RETRIEVAL_RESULTS.md) (Chinese).

## Quick Start

Requires Python 3.10+ and Tk 8.6+; a camera is needed for the live demo. From the repository root (macOS / Linux):

```bash
make bootstrap
make run
```

`make bootstrap` creates an isolated environment, installs dependencies, and downloads and verifies the models; recognition can then run offline. Open the camera, enroll people in the application, and then try recognition. See the [application guide](face_compare_system/README.md) (Chinese) for full commands, Windows setup, and usage, or browse the [documentation index](docs/README.md).
