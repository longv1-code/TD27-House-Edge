"""Save beside run_starter.py, then run: python my_experiment.py --out custom-01."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import run_experiments
from slot_forecasting.io import (
    load_inputs, read_labels, read_submission, write_json,
)
from slot_forecasting.schema import ValidationError
from slot_forecasting.submission_contract import extract_archive, package_code
from slot_forecasting.transition import TransitionForecaster


def candidate_specs():
    """Edit these settings for later experiments; keep the audit separate."""
    specs = {}
    for method in ("global", "context", "empirical", "recent", "pooled", "pooled_recent"):
        specs[method] = {"family": "frequency", "parameters": {"method": method}}

    # Smaller half-life: forget old spins faster.
    # Larger prior strength: trust similar machines more than this machine's history.
    for half_life in (12, 48, 96):
        for strength in (5, 20, 60):
            specs[f"recent_h{half_life}_s{strength}"] = {
                "family": "frequency",
                "parameters": {"method": "pooled_recent", "half_life": half_life,
                               "prior_strength": strength},
            }

    # Try individual delays of 1 through 8 actual ticks.
    for lag in range(1, 9):
        for half_life in (24, 128):
            specs[f"lag{lag}_h{half_life}"] = {
                "family": "transition",
                "parameters": {"lags": [lag], "half_life": half_life, "strength": 8},
            }

    # Also try a few combinations of earlier payouts.
    for lags in ((1, 2), (1, 4), (2, 4), (4, 8)):
        specs[f"pair{lags[0]}_{lags[1]}"] = {
            "family": "transition",
            "parameters": {"lags": list(lags), "half_life": 64, "strength": 12},
        }
    return specs


def make_model(spec):
    factory = FrequencyForecaster if spec["family"] == "frequency" else TransitionForecaster
    return factory(**spec["parameters"])


# This becomes the standalone prediction script inside the submission ZIP.
PREDICT_CODE = '''import argparse
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
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "public")
    parser.add_argument("--out", type=Path, default=Path("custom-01"))
    parser.add_argument("--team-id", default="Not specified")
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if out.exists():
        raise ValidationError("Output folder already exists. Choose a new --out name.")
    if out == data or data in out.parents:
        raise ValidationError("Choose an output folder outside public/.")

    specs = candidate_specs()
    # Bind each spec to its own factory, rather than sharing the last loop value.
    candidates = [(name, lambda spec=spec: make_model(spec)) for name, spec in specs.items()]
    print(f"Comparing {len(candidates)} models. Please wait...", flush=True)

    # The kit selects on half the validation episodes, then audits the winner once.
    # Training uses train/ only. No hidden test answers are read.
    report = run_experiments(data, out, candidates=candidates)
    winner = report["chosen"]
    spec = specs[winner]
    report["selected_configuration"] = spec
    report["environment"] = {
        "python": platform.python_version(),
        "dependencies": "standard library only",
        "randomness": "none; deterministic",
    }
    write_json(out / "experiments.json", report)

    print("\nModel comparison: tuning points, higher is better")
    ranked = sorted(
        report["comparison"].items(),
        key=lambda item: (-item[1]["tuning"]["technical_points"], item[0]),
    )
    for name, row in ranked:
        print(f"{name:<24} {row['tuning']['technical_points']:.4f}")
    print(f"\nSelected: {winner}")
    print(f"Held-out audit: {report['audit']['technical_points']:.4f} / {report['max_points']}")
    print(f"Pooled reference: {report['max_points'] / 2:.4f}")

    # Refit the selected configuration deterministically, then save its checkpoint.
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train/labels.csv", queries)
    model = make_model(spec).fit(queries, history, labels)
    source = out / "entry_source"
    source.mkdir()
    base = model if spec["family"] == "frequency" else model.base
    base.save(source / "model.json")
    write_json(source / "config.json", {
        **spec, "round": report["round"], "season_id": report["season_id"],
    })
    (source / "predict.py").write_text(PREDICT_CODE, encoding="utf-8")
    shutil.copyfile(__file__, source / "my_experiment.py")
    shutil.copyfile(out / "experiments.json", source / "experiments.json")

    module_source = Path(__file__).resolve().parent / "slot_forecasting"
    module_target = source / "slot_forecasting"
    module_target.mkdir()
    for path in module_source.glob("*.py"):
        shutil.copyfile(path, module_target / path.name)

    write_json(source / "reproduce.json", {
        "image": "python",
        "command": [
            "python", "predict.py",
            "--data", "/data",
            "--out", "/output/submission.csv",
        ],
    })
    (source / "README.md").write_text(
        f"# House Edge entry\n\nTeam ID: {args.team_id}\nRound: {report['round']}\n"
        f"Selected model: {winner}\nTraining Python: {platform.python_version()}\n\n"
        "Requires Python 3.11+ and its standard library only. CPU only. Deterministic; no random seed.\n\n"
        "Recreate the forecast from the ZIP root:\n"
        "python predict.py --data /data --out /output/submission.csv\n\n"
        "Inference loads model.json and config.json and reads only metadata.json and test queries/history.\n"
        "Training uses train queries, histories, and observed labels. Settings are selected using half the "
        "validation episodes per condition; the remaining episodes audit the fixed choice. "
        "The audit does not choose settings. Missing ticks stay unknown; features use earlier ticks only.\n\n"
        "Settings, scores, episode splits, and input hashes are in experiments.json.\n"
        "To repeat training against a released kit: python my_experiment.py --data /data --out /tmp/retrained\n",
        encoding="utf-8",
    )
    archive = package_code(source, out / "entry.zip")
    if archive.stat().st_size > 100_000_000 or (out / "submission.csv").stat().st_size > 10_000_000:
        raise ValidationError("Entry exceeds the kit's upload size limits.")

    # Test the actual ZIP with only test inputs available to the prediction command.
    print("\nChecking predictions from a fresh extraction of entry.zip...", flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        extracted = temporary / "code"
        extract_archive(archive.read_bytes(), extracted)
        isolated = temporary / "data"
        (isolated / "test").mkdir(parents=True)
        shutil.copyfile(data / "metadata.json", isolated / "metadata.json")
        for name in ("queries.csv", "history.csv"):
            shutil.copyfile(data / "test" / name, isolated / "test" / name)

        reproduced = temporary / "submission.csv"
        subprocess.run(
            [
                sys.executable, "predict.py",
                "--data", str(isolated),
                "--out", str(reproduced),
            ],
            cwd=extracted,
            check=True,
            timeout=1800,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        expected = read_submission(out / "submission.csv")
        actual = read_submission(reproduced, set(expected))
        difference = max(
            abs(a - b)
            for key in expected
            for a, b in zip(expected[key], actual[key])
        )
        if difference > 1e-8:
            raise ValidationError(f"ZIP reproduction mismatch: {difference}")

    write_json(out / "verification.json", {
        "passed": True,
        "max_probability_difference": difference,
    })
    print(f"ZIP reproduction passed. Largest probability difference: {difference:.3g}")
    print(f"\nForecast: {out / 'submission.csv'}\nMatching code ZIP: {archive}")
    print(f"Scores and settings: {out / 'experiments.json'}")
    print("Nothing has been uploaded.")


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"ERROR: {error}\nUse a fresh --out folder when retrying.")