# Changelog

## [0.2.1] - 2025-09-02
### Added
- CLI `--save-plots` option for `sweep` and `dp-curve`; saves PNG charts to `benchmarks/out/` (or `--out-dir`).
- README Quick Examples mention of plotting flag.
- This CHANGELOG.

### Changed
- Dev tooling: recommend `matplotlib` via dev extra for plotting.

## [0.2.0] - 2025-09-02
### Added
- GitHub Actions CI (3.9–3.12), pre-commit(ruff/black).
- `crom-bench` CLI: `e2e`, `sweep`, `scale`, `dp-curve`, `haystack-compare`.
- Plugins: FlashRank/LLMLingua/Evidently (optional extras).
- Example corpus & queries (JSONL).

## [0.1.0] - 2025-09-02
- Initial packaging; budget packer, hybrid rerank, drift estimator, demo & metrics.
