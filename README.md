# MCX Gold Intelligence

Unified repository for MCX gold-contract relative-value research and dashboarding.

## Primary implementation

The root project (`src/`, `app/`) is the primary implementation. It provides:
- MCX bhavcopy validation and price normalisation to ₹/gram at 999 purity
- active-contract pairing, walk-forward detrending and z-score signals
- event-driven backtesting with transaction-cost and liquidity gates
- attribution, contract lifecycle views, batch exports, and a Streamlit dashboard
- synthetic data for development only (not market data)

### Quick start

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.synthetic_data
python -m src.pipeline
pytest -q
streamlit run app/main.py
```

The synthetic data command writes `data/processed/normalized_bhavcopy.parquet`. Use real, validated MCX bhavcopy data before interpreting results. Input path can be passed to the pipeline with `--input`; see `python -m src.pipeline --help`.

## Repository layout

```text
.
├── app/                     Streamlit dashboard and Plotly chart builders
├── data/
│   ├── processed/           Input placeholder for normalized_bhavcopy.parquet
│   └── outputs/             Generated batch outputs (ignored by Git)
├── docs/INTEGRATION_AUDIT.md
├── legacy/                  Previous implementation preserved without import collisions
├── src/                     Primary quant engine and pipeline
├── tests/                   Automated tests for the primary implementation
└── requirements.txt
```

## Legacy implementation and bundled data

The previous `mcx_intelligence-main` code and its two Parquet exports are preserved under `legacy/`. They use a different input schema and a different strategy/metrics implementation from the primary engine. They are intentionally not imported by the primary app, because placing both implementations in the root `src/` would create module-name collisions and silently mix unlike P&L assumptions.

Legacy entry point (run from its own directory so relative data paths resolve correctly):

```bash
cd legacy
python main.py
```

It reads `legacy/data/raw/*.csv` or `legacy/data/processed/master_gold_contracts_12m.parquet`. It requires the dependencies listed in the root `requirements.txt`, including a Parquet engine.

## Important before committing or presenting results

- Contract specifications, purity, tender-window rules, tick sizes and charges in `src/config.py` are indicative defaults. Verify them against current MCX documentation and your broker's charges.
- Synthetic data is only for software testing. Do not report synthetic backtest performance as a real trading result.
- Review the bundled `legacy/data/processed/*.parquet` files before deciding whether they belong in the Git repo; they are included here because they were part of the supplied project. If they are generated outputs, consider removing them from version control or using Git LFS.
- See `docs/INTEGRATION_AUDIT.md` for the integration findings and remaining checks.
