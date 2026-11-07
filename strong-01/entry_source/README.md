# House Edge

Team ID: Not specified; select the correct team on the Form
Round: main
Training Python: 3.13.14

Python 3.11+; standard library only; CPU only; no packages or network required. Deterministic; no random seed.

Reproduce from the ZIP root:
python predict.py --predict --data /data --out /output/submission.csv

Inference reads model.json, config.json, metadata.json, and test queries/history only. It does not read training data, validation answers, condition labels, or a saved prediction table.

Training priors use this round's train histories and labels. Each machine's earlier outcomes supply Bayesian evidence for single-lag and parity-of-two-lag models, using actual lags 1..8. Recent, long-memory, and chunked evidence are combined. Win probability and positive payout amounts are estimated separately. Missing ticks remain unknown. No data at/after cutoff is used.

Main settings were developed using tuning episodes. The audit half is scored after settings are fixed. Full validation includes tuning episodes and is not a new independent test. See training_report.json for settings, scores, splits, and hashes.

Refit on a new released round: python predict.py --data /data --out /tmp/new-run --search
Use a fresh output folder. Training may use validation answers for tuning; prediction does not.

Forecast SHA-256: b158b5b8298d20062af545acef03c3f35ec0f3886789a8b3285cefb8448f2d93
