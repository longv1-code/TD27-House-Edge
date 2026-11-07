# House Edge next model

Team: Select the registered team on the submission form
Round: main
Season: 5fffdc4d0c700bbf61c3f911

Python 3.11+. CPU only. Deterministic; no random seed. Install the pinned requirements before the offline run. No network calls during inference.

Reproduce: python predict.py --predict --data /data --out /output/submission.csv

Prediction reads checkpoint.json, metadata.json, and test queries/history only. It uses no forecast lookup table and no training/validation files. IDs only group histories and identify output rows. Missing ticks are unknown; future ticks are rejected.

Training discovers win and payout-size lag patterns from training histories. Population odds and the residual readout are fitted on all released validation targets after model selection. The raw validation fit score is optimistic. Use the episode-grouped CV score in training_report.json for comparison. That CV was used in selection and is not an untouched final test.

Whole episodes stay together in each CV fold. Centers, positive-payout templates, amount-pattern priors, and the residual readout are refitted outside each fold. No private test answers are used.

To retrain from the participant kit, use the separate source bundle and: python house_edge_next.py --data public --out new-run --cv
