# House Edge next model — runnable source

Extract this ZIP's contents into your fully extracted participant kit folder. Keep `house_edge_next.py`, `next_model/`, `requirements_next.txt`, `public/`, and the kit's `slot_forecasting/` together. Open that folder in VS Code. Keep prior output folders.

Requires Python 3.11+ and NumPy/SciPy. No GPU or scikit-learn is needed.

Run one command at a time:

```powershell
python -m pip install -r requirements_next.txt
python house_edge_next.py --out strong-03 --cv
python check_submission.py strong-03/cv_predictions.csv --split validation
python check_submission.py strong-03/submission.csv
```

Wait for training and `ZIP reproduction PASSED` before the checker commands. CPU performance varies; this may take several minutes. `strong-03` must be a new folder.

Submit the matching `strong-03/submission.csv` and `strong-03/entry.zip`. The source ZIP you extracted is not the submission ZIP. Nothing uploads automatically.

## Which score to use

`cv_predictions.csv` contains four-fold predictions from models that excluded the corresponding complete episodes. Each fold refits population centers, payout templates, amount-pattern priors and the residual readout outside that fold. `results.json` records the cross-validation points. These folds were used for model selection and are not an untouched final test.

After selection, the final checkpoint fits **all released validation targets**. Consequently `validation_predictions.csv` is a fit diagnostic and gives an optimistic score. Compare models using the CV score, not the fit score. No private test answers are accessed.

## Model

Training histories identify recurring delayed patterns. Win probabilities average over hidden orientation and uncertain baseline odds. Positive payouts use two learned distributions per machine type and evidence about previous payout sizes, including 2x-or-larger payouts. A small regularized residual correction and causal episode-level filter refine win estimates.

Missing ticks are unknown; observations at/after a machine's cutoff are rejected. Related machines supply only earlier outcomes. IDs are used for grouping and output, never as a forecast lookup table.

The generated entry pins the installed NumPy/SciPy versions. Its offline inference reads only checkpoint.json, metadata.json and test queries/history. It does not need fitting data, validation targets, conditions, internet or the uploaded forecast CSV.

## Verified main-round results

On the unmodified main kit for season `5fffdc4d0c700bbf61c3f911`, four-fold episode-grouped CV scored **57.863653 / 60**. The previous research pipeline, refitted under the same folds, scored **55.547594 / 60**. These folds were used for model selection.

Final refitting on all released validation targets produced a **58.096948 / 60 fit score**. That number includes fitting data. It is not a hidden-test score or an independent validation result. Both CSV formats and offline code reproduction passed; reproduction had zero probability difference.

The methods draw on Bayesian sequential state estimation and dynamic model averaging. The implementation uses a custom finite-grid hidden-state filter and an episode-level logistic update. Primary background: [McCormick et al., Dynamic Logistic Regression and Dynamic Model Averaging for Binary Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC3831847/), and [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742).

## Adaptation

Use a separate copy of the newly released adaptation kit, copy these source files into it, check that its metadata says adaptation, and repeat the commands with a fresh output folder. The program refits on that kit and guards checkpoint round/season during prediction. Main-round settings should be revalidated; no adaptation score is promised before its data releases.
