# Versioning & PyPI Guidance

This document defines package naming, SemVer rules, and a future path to publish to PyPI.

## 1) Package name
- Distribution name (PyPI): `crom-efficientllm` (lowercase, hyphen-separated)
- Import name (module): `crom_efficientllm` (PEP 8 underscore)

> **Tip**: Keep both names consistent to avoid confusion in docs.

### Check name availability on PyPI
- Visit: https://pypi.org/project/crom-efficientllm/ (404 → available)
- If taken, consider: `crom-efficient-llm`, `crom-llm-efficient`, `crom-ctx-pack`
- Reserve on TestPyPI first: use `test.pypi.org` to validate metadata & upload

## 2) Semantic Versioning (SemVer)
We follow **MAJOR.MINOR.PATCH**.

- **MAJOR**: Backward-incompatible API changes
  - e.g., rename function signatures (`budget_pack`), move/rename modules, change return schemas
- **MINOR**: Backward-compatible features
  - new functions/flags (e.g., `pack_summary`, CLI subcommands), performance improvements
- **PATCH**: Backward-compatible bug fixes
  - logic corrections, docs/CI fixes, dependency pin updates without API changes

### Pre-releases
Use suffixes: `-a.1`, `-b.1`, `-rc.1` (alpha/beta/release-candidate)
- Example: `0.3.0-rc.1`

### Deprecation Policy
- Mark deprecated APIs in `CHANGELOG.md` and docstrings
- Provide at least **one MINOR release** with warnings before removal

### Public API Surface
We commit compatibility for:
- `crom_efficientllm.budget_packer.packer`: `Chunk`, `budget_pack`, `pack_summary`
- `crom_efficientllm.rerank_engine.rerank`: `hybrid_rerank`
- `crom_efficientllm.drift_estimator.estimator`: `DriftEstimator`, `DriftMode`
- CLI entrypoints: `crom-demo`, `crom-bench` and their documented flags

## 3) Release Flow (GitHub → PyPI later)
- Tag: `vX.Y.Z` → GitHub Actions builds & creates a Release (artifacts attached)
- Keep `CHANGELOG.md` updated per release
- After API stabilizes, enable **PyPI publish** using a separate workflow with `PYPI_API_TOKEN` secret

### (Future) PyPI publishing steps
1. Create a PyPI account & project
2. Add `PYPI_API_TOKEN` to repo `Settings → Secrets and variables → Actions`
3. Add `release-pypi.yml` workflow to upload on tag
4. Verify install: `pip install crom-efficientllm` and import `crom_efficientllm`

---

_Last updated: 2025-09-02_
