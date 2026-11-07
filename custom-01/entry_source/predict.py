import argparse
import json
from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.transition import TransitionForecaster

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
if any(metadata.get(k) != config[k] for k in ("round", "season_id")):
    raise SystemExit("Wrong round or season for this checkpoint. Retrain on the correct kit.")
base = FrequencyForecaster.load(root / "model.json")
model = base
if config["family"] == "transition":
    model = TransitionForecaster(**config["parameters"])
    model.base = base
queries, history = load_inputs(args.data / "test")
write_submission(args.out, model.predict(queries, history))
