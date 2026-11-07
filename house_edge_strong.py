"""House Edge: machine-specific Bayesian lag patterns, with no extra packages.

Save beside run_starter.py. Run: python house_edge_strong.py --out strong-01
For new-round model selection: add --search and point --data at the new public/.
"""

import argparse
from itertools import combinations, product
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import (
    load_inputs, read_labels, read_rows, read_submission, read_targets,
    require_ids, write_json, write_submission,
)
from slot_forecasting.rounds import round_metadata, score_round
from slot_forecasting.schema import PAYOUTS, ValidationError, index_history
from slot_forecasting.submission_contract import extract_archive, package_code


# Chosen using main-round tuning episodes, before evaluating the held-out half.
DEFAULT_CONFIG = {
    "half_life": 12, "alpha": 0.5, "temperature": 0.25,
    "win_strength": 0.25, "positive_strength": 64,
    "long_weight": 0.25, "chunk_weight": 1.0,
    "max_order": 2, "power": 1.3, "prior_blend": 0.1,
}


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class PatternEngine:
    """Learn each machine's patterns from observations strictly before its cutoff."""

    def __init__(self, queries, history, prior_model):
        self.queries = tuple(queries)
        indexed = index_history(self.queries, history)
        # Null, 8 individual lags, and all 28 pairs. Each pattern has two states.
        self.patterns = [()] + [(i,) for i in range(8)] + list(combinations(range(8), 2))
        self.encoded = []
        self.count_cache = {}
        self.evidence_cache = {}
        for query in self.queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            prior = prior_model.context_priors.get(
                prior_model._key(query), prior_model.global_prior,
            )

            def contexts(tick):
                # Missing observations stay unknown. A lag is an actual tick offset.
                bits = [None if tick-k not in sequence else int(sequence[tick-k] > 0)
                        for k in range(1, 9)]
                codes = []
                for pattern in self.patterns:
                    if any(bits[k] is None for k in pattern):
                        codes.append(None)
                    else:
                        # Pair state: did exactly one of the two earlier spins pay?
                        codes.append(sum(bits[k] for k in pattern) % 2)
                return tuple(codes)

            observations = [
                (query.cutoff - 1 - row.tick, PAYOUTS.index(row.payout), contexts(row.tick))
                for row in rows
            ]
            self.encoded.append((tuple(prior), observations, contexts(query.cutoff)))

    def counts(self, half_life, chunk=0):
        key = (half_life, chunk)
        if key not in self.count_cache:
            tables = []
            for prior, rows, wanted in self.encoded:
                bins = 1 + max((age // chunk for age, _, _ in rows), default=0) if chunk else 1
                table = [[[[0.0] * 5 for _ in range(2)]
                          for _ in self.patterns] for _ in range(bins)]
                for age, outcome, codes in rows:
                    weight = 2.0 ** (-age / half_life)
                    group = table[age // chunk if chunk else 0]
                    for j, code in enumerate(codes):
                        if code is not None:
                            group[j][code][outcome] += weight
                tables.append(table)
            self.count_cache[key] = tables
        return self.count_cache[key]

    def evidence(self, half_life, alpha, chunk=0):
        key = (half_life, alpha, chunk)
        if key not in self.evidence_cache:
            result = []
            for (prior, _, _), table in zip(self.encoded, self.counts(half_life, chunk)):
                a, b = alpha * prior[0], alpha * (1.0 - prior[0])
                constant = log_beta(a, b)
                scores = [0.0] * len(self.patterns)
                for group in table:
                    for j, (first, second) in enumerate(group):
                        lose0, lose1 = first[0], second[0]
                        win0, win1 = math.fsum(first[1:]), math.fsum(second[1:])
                        scores[j] += (
                            log_beta(a + lose0, b + win0)
                            + log_beta(a + lose1, b + win1)
                            - log_beta(a + lose0 + lose1, b + win0 + win1)
                            - constant
                        )
                result.append(scores)
            self.evidence_cache[key] = result
        return self.evidence_cache[key]

    def predict(self, config):
        half, alpha = config["half_life"], config["alpha"]
        counts = self.counts(half)
        recent = self.evidence(half, alpha)
        long = self.evidence(96, alpha) if config["long_weight"] else None
        # Separate 32-tick windows help identify patterns whose direction changes.
        chunks = self.evidence(128, alpha, 32) if config["chunk_weight"] else None
        predictions = {}
        for i, query in enumerate(self.queries):
            prior, _, wanted = self.encoded[i]
            unconditional = counts[i][0][0][0]
            total = math.fsum(unconditional) + alpha
            base = [(c + alpha*p) / total for c, p in zip(unconditional, prior)]
            positive_prior = [p / math.fsum(prior[1:]) for p in prior[1:]]
            logits = []
            for j, pattern in enumerate(self.patterns):
                score = recent[i][j]
                if long is not None:
                    score += config["long_weight"] * long[i][j]
                if chunks is not None:
                    score += config["chunk_weight"] * chunks[i][j]
                logits.append(score / config["temperature"] - 0.5 * len(pattern))
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            weight_total = math.fsum(weights)
            probability = [0.0] * 5
            for j, weight in enumerate(weights):
                observed = counts[i][0][j][wanted[j]] if wanted[j] is not None else [0.0]*5
                wins = math.fsum(observed[1:])
                win = (wins + config["win_strength"]*(1.0-base[0])) / (
                    math.fsum(observed) + config["win_strength"]
                )
                mass = weight / weight_total
                probability[0] += mass * (1.0-win)
                for k in range(1, 5):
                    amount = (observed[k] + config["positive_strength"]*positive_prior[k-1]) / (
                        wins + config["positive_strength"]
                    )
                    probability[k] += mass * win * amount
            # Conservative calibration, selected on tuning episodes only.
            win = 1.0 - probability[0]
            positive = win ** config["power"]
            adjusted = positive / (positive + (1.0-win) ** config["power"])
            calibrated = [1.0-adjusted] + [p*adjusted/win for p in probability[1:]]
            blend = config["prior_blend"]
            final = [(1.0-blend)*p + blend*r for p, r in zip(calibrated, prior)]
            denominator = math.fsum(final)
            predictions[query.query_id] = tuple(p/denominator for p in final)
        return predictions


def search_configs():
    # A bounded refit search for a new round. The audit never chooses a setting.
    yield dict(DEFAULT_CONFIG)
    for half, long, chunk, power, blend in product(
        (6, 12, 24, 48), (0.0, 0.25), (0.0, 1.0), (1.0, 1.3), (0.02, 0.1),
    ):
        candidate = {**DEFAULT_CONFIG, "half_life": half, "long_weight": long,
                     "chunk_weight": chunk, "power": power, "prior_blend": blend}
        if candidate != DEFAULT_CONFIG:
            yield candidate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_saved(data, output, root):
    saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if any(metadata.get(k) != saved[k] for k in ("round", "season_id")):
        raise ValidationError("Checkpoint round/season does not match the supplied data. Retrain on that kit.")
    base = FrequencyForecaster.load(root / "model.json")
    queries, history = load_inputs(data / "test")
    prediction = PatternEngine(queries, history, base).predict(saved["parameters"])
    write_submission(output, prediction)


def train(args):
    started = perf_counter()
    data, out = args.data.resolve(), args.out.resolve()
    if out.exists():
        raise ValidationError("Choose a new --out folder; existing runs are preserved.")
    if out == data or data in out.parents:
        raise ValidationError("Output must be outside public/.")
    metadata = round_metadata(data)
    tq, th = load_inputs(data / "train")
    labels = read_labels(data / "train/labels.csv", tq)
    vq, vh = load_inputs(data / "validation")
    if {q.episode_id for q in tq} & {q.episode_id for q in vq}:
        raise ValidationError("Training and validation episodes overlap.")
    base = FrequencyForecaster("pooled_recent").fit(tq, th, labels)
    targets = read_targets(data / "validation/targets.csv")
    require_ids({t.query_id for t in targets}, {q.query_id for q in vq})
    conditions = read_rows(data / "validation/conditions.csv", ("query_id", "condition"))
    groups = {row["query_id"]: row["condition"] for row in conditions}
    if len(groups) != len(conditions):
        raise ValidationError("Duplicate validation condition IDs.")
    reference = read_submission(data / "validation/reference.csv", {q.query_id for q in vq})
    tuning, audit = episode_folds(targets, groups)

    def score(predictions, episodes):
        return subset_score(predictions, targets, groups, reference, episodes, metadata["max_points"])

    print("Building machine-specific lag evidence...", flush=True)
    engine = PatternEngine(vq, vh, base)
    candidates = list(search_configs()) if args.search else [dict(DEFAULT_CONFIG)]
    comparison = []
    best_key, best_config, validation = None, None, None
    for i, candidate in enumerate(candidates, 1):
        prediction = engine.predict(candidate)
        result = score(prediction, tuning)
        comparison.append({"parameters": candidate, "tuning": result})
        key = (result["mean_loss_ratio"], json.dumps(candidate, sort_keys=True))
        if best_key is None or key < best_key:
            best_key, best_config, validation = key, candidate, prediction
        print(f"[{i}/{len(candidates)}] tuning: {result['technical_points']:.6f}", flush=True)

    # Freeze settings above, before calculating the audit result below.
    held_out = score(validation, audit)
    full = score(validation, {q.episode_id for q in vq})
    print(f"Selected settings: {json.dumps(best_config, sort_keys=True)}")
    print(f"Held-out audit: {held_out['technical_points']:.6f} / {metadata['max_points']}")
    print(f"Full validation: {full['technical_points']:.6f} / {metadata['max_points']}")

    test_q, test_h = load_inputs(data / "test")
    if {q.episode_id for q in test_q} & {q.episode_id for q in tq + vq}:
        raise ValidationError("Test episodes overlap training/validation.")
    print("Predicting test queries...", flush=True)
    test = PatternEngine(test_q, test_h, base).predict(best_config)
    out.mkdir(parents=True)
    write_submission(out / "validation_predictions.csv", validation)
    write_submission(out / "submission.csv", test)
    read_submission(out / "submission.csv", {q.query_id for q in test_q})
    checked = score_round(out / "validation_predictions.csv", data)
    if abs(checked["technical_points"]-full["technical_points"]) > 1e-9:
        raise ValidationError("Saved validation predictions do not match the calculated score.")

    input_names = (
        "metadata.json", "train/queries.csv", "train/history.csv", "train/labels.csv",
        "validation/queries.csv", "validation/history.csv", "validation/targets.csv",
        "validation/conditions.csv", "validation/reference.csv", "test/queries.csv", "test/history.csv",
    )
    report = {
        "round": metadata["round"],
        "season_id": metadata["season_id"],
        "parameters": best_config,
        "search": args.search,
        "comparison": comparison,
        "tuning_episodes": sorted(tuning),
        "audit_episodes": sorted(audit),
        "tuning": score(validation, tuning),
        "audit": held_out,
        "full_validation": full,
        "selection": (
            "Main defaults developed on tuning episodes; --search selects only "
            "on tuning episodes. Audit never selects settings."
        ),
        "environment": {
            "python": platform.python_version(),
            "dependencies": "standard library only",
            "seed": "none; deterministic",
        },
        "input_sha256": {name: sha256(data / name) for name in input_names},
        "script_sha256": sha256(__file__),
        "submission_sha256": sha256(out / "submission.csv"),
    }
    write_json(out / "results.json", report)

    source = out / "entry_source"
    source.mkdir()
    base.save(source / "model.json")
    write_json(source / "config.json", {
        "round": metadata["round"],
        "season_id": metadata["season_id"],
        "parameters": best_config,
    })
    shutil.copyfile(__file__, source / "predict.py")
    shutil.copyfile(out / "results.json", source / "training_report.json")

    modules = source / "slot_forecasting"
    modules.mkdir()
    for path in (Path(__file__).resolve().parent / "slot_forecasting").glob("*.py"):
        shutil.copyfile(path, modules / path.name)

    write_json(source / "reproduce.json", {
        "image": "python",
        "command": [
            "python", "predict.py", "--predict",
            "--data", "/data",
            "--out", "/output/submission.csv",
        ],
    })
    (source / "README.md").write_text(
        f"# House Edge\n\nTeam ID: {args.team_id}\nRound: {metadata['round']}\n"
        f"Training Python: {platform.python_version()}\n\n"
        "Python 3.11+; standard library only; CPU only; no packages or network required. "
        "Deterministic; no random seed.\n\n"
        "Reproduce from the ZIP root:\n"
        "python predict.py --predict --data /data --out /output/submission.csv\n\n"
        "Inference reads model.json, config.json, metadata.json, and test queries/history only. "
        "It does not read training data, validation answers, condition labels, or a saved prediction table.\n\n"
        "Training priors use this round's train histories and labels. Each machine's earlier outcomes "
        "supply Bayesian evidence for single-lag and parity-of-two-lag models, using actual lags 1..8. "
        "Recent, long-memory, and chunked evidence are combined. Win probability and positive payout "
        "amounts are estimated separately. Missing ticks remain unknown. No data at/after cutoff is used.\n\n"
        "Main settings were developed using tuning episodes. The audit half is scored after settings "
        "are fixed. Full validation includes tuning episodes and is not a new independent test. "
        "See training_report.json for settings, scores, splits, and hashes.\n\n"
        "Refit on a new released round: python predict.py --data /data --out /tmp/new-run --search\n"
        "Use a fresh output folder. Training may use validation answers for tuning; prediction does not.\n"
        f"\nForecast SHA-256: {report['submission_sha256']}\n",
        encoding="utf-8",
    )
    archive = package_code(source, out / "entry.zip")
    if archive.stat().st_size > 100_000_000 or (out / "submission.csv").stat().st_size > 10_000_000:
        raise ValidationError("Submission exceeds kit upload limits.")

    print("Rechecking the ZIP with no installed third-party packages...", flush=True)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        extracted = root / "source"
        extract_archive(archive.read_bytes(), extracted)
        isolated = root / "data"
        (isolated / "test").mkdir(parents=True)
        shutil.copyfile(data / "metadata.json", isolated / "metadata.json")
        for name in ("queries.csv", "history.csv"):
            shutil.copyfile(data / "test" / name, isolated / "test" / name)

        reproduced = root / "reproduced.csv"
        run = subprocess.run(
            [
                sys.executable, "-S", "predict.py", "--predict",
                "--data", str(isolated),
                "--out", str(reproduced),
            ],
            cwd=extracted,
            capture_output=True,
            text=True,
            timeout=1800,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if run.returncode:
            raise ValidationError(f"ZIP reproduction failed: {run.stdout}\n{run.stderr}")

        actual = read_submission(reproduced, {q.query_id for q in test_q})
        expected = read_submission(out / "submission.csv", {q.query_id for q in test_q})
        difference = max(
            abs(a-b)
            for key in expected
            for a, b in zip(expected[key], actual[key])
        )
        if difference > 1e-8:
            raise ValidationError(f"ZIP reproduction mismatch: {difference}")

    report["verification"] = {
        "passed": True,
        "max_probability_difference": difference,
        "no_third_party_packages": True,
        "inference_inputs": "metadata and test queries/history only",
    }
    report["entry_zip_sha256"] = sha256(archive)
    report["seconds"] = perf_counter() - started
    write_json(out / "results.json", report)

    print(f"ZIP reproduction PASSED. Max probability difference: {difference:.3g}")
    print(f"Runtime: {report['seconds']:.1f} seconds")
    print(f"Forecast: {out / 'submission.csv'}\nMatching code: {archive}")
    print("Nothing was uploaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "public")
    parser.add_argument("--out", type=Path, default=Path("strong-01"))
    parser.add_argument("--team-id", default="Not specified; select the correct team on the Form")
    parser.add_argument("--search", action="store_true", help="Select settings on this round's tuning episodes")
    parser.add_argument("--predict", action="store_true", help="Recreate test CSV from a packaged checkpoint")
    args = parser.parse_args()

    if args.predict:
        predict_saved(args.data.resolve(), args.out.resolve(), Path(__file__).resolve().parent)
    else:
        train(args)


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(
            f"ERROR: {error}\n"
            "If training created a partial folder, retry with a new --out name."
        )