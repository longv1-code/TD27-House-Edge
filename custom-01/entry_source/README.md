# House Edge entry

Team ID: Not specified
Round: main
Selected model: lag7_h24
Training Python: 3.13.14

Requires Python 3.11+ and its standard library only. CPU only. Deterministic; no random seed.

Recreate the forecast from the ZIP root:
python predict.py --data /data --out /output/submission.csv

Inference loads model.json and config.json and reads only metadata.json and test queries/history.
Training uses train queries, histories, and observed labels. Settings are selected using half the validation episodes per condition; the remaining episodes audit the fixed choice. The audit does not choose settings. Missing ticks stay unknown; features use earlier ticks only.

Settings, scores, episode splits, and input hashes are in experiments.json.
To repeat training against a released kit: python my_experiment.py --data /data --out /tmp/retrained
