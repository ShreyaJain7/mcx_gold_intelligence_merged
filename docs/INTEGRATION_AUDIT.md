# Integration audit

## Summary

The two supplied archives were combined into one repository. The newer `mcx_gold_intel` implementation is the primary application because it has the more complete walk-forward engine, dashboard, configuration, cost model, lifecycle calendar, and dedicated tests. The older `mcx_intelligence-main` implementation and its bundled Parquet files are retained under `legacy/` to avoid overwriting similarly named modules.

## Checks performed

- Inspected both archive file trees and Python entry points.
- Ran the primary project's automated tests in the supplied runtime: **14 passed**.
- Confirmed the primary project declares runtime and test dependencies in `requirements.txt`.
- Kept the two implementations isolated because they have different input schemas and different P&L/strategy assumptions.

## Findings / things not silently assumed

1. **Input schemas differ.** The primary engine expects `trade_date`, `symbol`, `expiry_date`, `close`, `volume`, and `open_interest`. The legacy loader expects CSV-style symbol/date/close columns and creates a different daily frame. The old Parquet exports are preserved but are not automatically fed into the primary engine.
2. **Parquet inspection limitation in the audit runtime.** The runtime used for this audit did not have `pyarrow` or `fastparquet` installed, so the bundled legacy Parquet schemas/content could not be independently read here. The root requirements include `pyarrow`; inspect those files locally after installing dependencies before relying on them.
3. **Legacy entry point uses relative paths.** Run it from `legacy/` (`cd legacy` then `python main.py`) so its `data/` paths resolve to the preserved legacy data. It is not the default entry point.
4. **Market assumptions require human verification.** The contract specs, purity assumptions, transaction charges, tick sizes and tender-window approximations are marked indicative in the primary implementation. Verify current MCX and broker rules.
5. **Real-data validation remains necessary.** The tests use synthetic data. A passing test suite validates software behavior, not profitability or correctness against live MCX data.
6. **Repository hygiene.** Generated root outputs and raw CSVs are ignored. The two legacy Parquet files are retained because they were included in the upload; check their provenance/size and whether they should be tracked or moved to Git LFS before pushing.

## Suggested pre-commit checklist

```bash
python -m pip install -r requirements.txt
pytest -q
python -m src.synthetic_data
python -m src.pipeline
streamlit run app/main.py
git status --short
```

Review generated outputs and any real market data for licensing, privacy, size and provenance before committing.
