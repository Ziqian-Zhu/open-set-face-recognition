# Changelog

All notable public milestones are documented here. The project follows semantic versioning for tagged releases.

## [1.0.0] - 2026-09-30

### Added

- Local multi-face recognition pipeline using YuNet detection, five-point alignment, and SFace embeddings.
- Open-set identity decisions with per-identity multi-template aggregation, distance rejection, and ambiguity margins.
- Independent multi-face tracking and temporal confirmation for live camera and local video inputs.
- Enrollment quality checks, SQLite vector storage, local person management, and privacy-preserving offline operation.
- Separate `face_research` package for template selection, threshold calibration, model comparison, diagnostics, and reproducible benchmarks.
- Bilingual project documentation, a privacy-reviewed application preview, and continuous integration on Python 3.10 and 3.12.

### Verified

- 454 regression tests covering decision boundaries, retrieval equivalence, tracking, persistence, and research protocols.
- Exact batched retrieval reduced p50 latency from 9.381 ms to 0.289 ms on the documented 1,000-identity × 3-template synthetic workload, with ranking, score, and final-decision equivalence checks.

### Limitations

- Results from public cropped images and synthetic workloads do not establish production accuracy, low-FAR performance, liveness detection, or cross-device robustness.
- YuNet and SFace model files are downloaded separately and retain their upstream licenses; private face data and embeddings are not included.

[1.0.0]: https://github.com/Ziqian-Zhu/open-set-face-recognition/releases/tag/v1.0.0
