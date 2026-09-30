# Open-Set Multi-Face Recognition and Template Selection

[![Read in Chinese](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-%E9%98%85%E8%AF%BB-52677A?style=for-the-badge)](./README.md)
[![English (current)](https://img.shields.io/badge/English-Current-0F766E?style=for-the-badge)](./README.en.md)

## Project Overview

An engineering project combining open-set multi-face recognition with research on template selection under a limited budget. `face_compare_system` is a local, CPU-based desktop application with YuNet detection and five-point alignment, SFace embeddings, quality gates, exact SQLite retrieval, identity-level multi-template aggregation, unknown/ambiguous-identity rejection, and per-track temporal confirmation. `face_research` provides reproducible template-selection, threshold-calibration, evaluation, and performance experiments.

YuNet and SFace are pretrained models. The contributions here are the system design, decision and retrieval implementation, and evaluation protocol—not a new neural network. Liveness detection and industrial-grade identity assurance are not claimed.

## Three Core Questions

1. **Open-set identity recognition:** A person in view may be absent from the gallery. The nearest candidate is accepted only after distance-threshold and top-two identity-margin checks.
2. **Robust multi-template matching:** Appearance varies for the same person. Multiple templates, quality gates, per-identity aggregation, and independent track confirmation help handle that variation, but can increase rejection or confirmation delay.
3. **Template selection under a limited budget:** When only K of N eligible images per identity can be retained, the research module compares First-K, Random, Quality, Diversity, and quality–coverage–consistency selection, calibrating each operating point on validation data.

## Architecture

```text
Camera / Video → YuNet (boxes + 5 landmarks) → Quality gate
                                                   ↓
                                      Alignment (112×112 BGR)
                                                   ↓
                                      SFace → unit embedding
                                                   ↓
SQLite templates → cached exact distance → identity aggregation
                                                   ↓
                                      Threshold + Top1/Top2 margin
                                                   ↓
                              Geometry/appearance association + per-track votes
                                                   ↓
                              Identity / Unknown / Pending / Quality rejected
```

The default distance is `d=(1−cosine)/2`. For each identity, the system takes the median distance of up to its **three nearest templates**, not its three most recently captured images. The shared acceptance rule is `d1≤τ` and, when a second identity exists, `d2−d1≥m`. The current SFace defaults, τ=0.275 and m=0.04, are engineering starting points; offline calibration does not automatically overwrite them.

## Evidence and Boundaries

- The latest code-level acceptance run passed **454 regression tests**: 449 earlier tests plus five floating-point boundary regressions. Coverage includes leakage checks, a 168-combination synthetic matrix, frozen validation parameters, paired temporal evaluation, video sampling completeness, subject-macro reporting, six-stage service/tracking profiling, batched exact-retrieval equivalence, independent 1:1 calibration, and optional model isolation. See the [final acceptance review](FINAL_AUDIT_V2.md) (Chinese).
- The MeGlass mixed-appearance protocol was explored across 168 combinations after extending the budget to K=5. In the original primary comparison, Coverage scored 13/24 known probes and First-K scored 14/24; the proposed method **did not beat the baseline**. A quality-gate ablation scored 14/24 with the full gate and 19/24 without it, showing a rejection cost in this sample—not proof that relaxing the production gate improves real-world performance.
- Batched grouping by template count addresses per-identity retrieval aggregation. In a paired benchmark with 1,000 synthetic identities × 3 templates, median retrieval latency was 9.145→0.274 ms, with identical full scores and rankings across 1,200 queries in six workloads. This is **retrieval latency**, not a 33× gain in recognition accuracy or camera FPS. See [retrieval evidence](face_research/P2_RETRIEVAL_RESULTS.md) (Chinese).
- Staged benchmarks cover retrieval and static one-face/repeated two-face loads at 10, 100, and 1,000 synthetic identities. They do not measure real moving-face video or camera FPS.
- Public glasses/no-glasses 1:1 verification used 200 identities in each of validation and test. On test, 88/89 usable genuine pairs were accepted, but the all-attempt denominator is 88/200: 111 genuine pairs failed acquisition. Descriptive AUC on usable pairs was 0.99285; zero false accepts among 44 usable impostor pairs still gives a Wilson upper bound of about 8.03%. See the [protocol and denominators](face_research/VERIFICATION_RESULTS.md) (Chinese).
- Acquisition diagnostics preserved all 600 pair scores/statuses. The 111 failed test genuine pairs comprise 49 with no valid detection, 53 quality rejections, and 9 with both. A subject-grouped descriptive interval for all-attempt genuine acceptance was 37%–51%. A zero-error bootstrap is not presented as a safety guarantee. See [diagnostics](face_research/ACQUISITION_DIAGNOSTICS.md) (Chinese).
- An optional SFace/ArcFace-MBF comparison calibrated each model separately on validation. All-attempt genuine acceptance on the same test was 88/200 and 89/200, with 111 acquisition failures shared by both; this does not justify changing the production model. See [model comparison](face_research/MODEL_COMPARISON_RESULTS.md) (Chinese).
- Live-person collection was removed from the current delivery criteria at the user's request. Real moving-video generalization, cross-session/low-FAR assurance, and liveness detection remain unverified or unimplemented. LTFT author annotations were obtained and metadata preflight checks completed, but source video was neither downloaded nor decoded, and an independent enrollment source was not verified. Annotation counts are not recognition results. See [video-validation status](face_research/VIDEO_VALIDATION_STATUS.md) (Chinese).

Full denominators, confidence intervals, thread limits, reproduction commands, and run IDs are documented in the [P1 experiment record](face_research/P1_RESULTS.md) (Chinese). The public exploratory protocol uses cropped images and pre-screened identities, lacks session metadata, and has been reused; it is not a sealed blind test. A zero observed false-accept count with a small denominator still has a wide uncertainty bound.

**Metric correction:** An older open-set threshold scan incorrectly counted unknown acquisition failures as successful rejections. The metric was corrected and the 168-combination matrix plus four quality-gate settings were rerun; template choices, τ/m, and original recognition counts were unchanged. The old `accuracy_all_attempts` field should not be cited. See [metric correction](face_research/METRIC_CORRECTION.md) (Chinese).

## Run and Reproduce

From the repository root, use Python 3.10+ with Tk 8.6+ support:

```bash
python3 -m venv face_compare_system/.venv-ui
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements.txt
face_compare_system/.venv-ui/bin/python face_compare_system/scripts/download_models.py
face_compare_system/.venv-ui/bin/python face_compare_system/main.py
```

Install the development dependencies separately before running the tests:

```bash
face_compare_system/.venv-ui/bin/python -m pip install -r face_compare_system/requirements-dev.txt
face_compare_system/.venv-ui/bin/python -m pytest -q \
  face_compare_system/tests face_research/tests -p no:cacheprovider
```

The tests do not require a personal camera or live-person images. See the [application guide](face_compare_system/README.md) (Chinese) for installation and pinned-model download details. Public datasets must be obtained separately under their source terms; they are not distributed with this repository.

## Code and Reading Guide

| Area | Entry point |
| --- | --- |
| Detection, alignment, embeddings | [deep_engine.py](face_compare_system/face_compare/deep_engine.py) |
| Service, quality, storage | [service.py](face_compare_system/face_compare/service.py), [quality.py](face_compare_system/face_compare/quality.py), [vector_database.py](face_compare_system/face_compare/vector_database.py) |
| Shared aggregation and open-set decision | [aggregation.py](face_compare_system/face_compare/aggregation.py), [decision.py](face_compare_system/face_compare/decision.py) |
| Multi-face tracking and evaluation | [tracking.py](face_compare_system/face_compare/tracking.py), [video_evaluation.py](face_compare_system/face_compare/video_evaluation.py) |
| Budgeted selection and experiments | [selection.py](face_research/selection.py), [research guide](face_research/README.md) (Chinese) |
| Résumé and interview preparation | [RESUME.md](RESUME.md), [INTERVIEW.md](INTERVIEW.md) (Chinese) |
| Earlier audit and improvement plan | [PROJECT_AUDIT.md](PROJECT_AUDIT.md), [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) (Chinese) |
| Goal-by-goal completion audit | [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md) (Chinese) |

Research runs use a temporary SQLite database and do not modify the enrolled-person gallery. Raw face images, embeddings, names, and frame-level logs must not be uploaded to a public repository. Gallery removal is a recoverable archive operation, not irreversible erasure. This project has no liveness defense and must not be used for high-stakes identity decisions.
