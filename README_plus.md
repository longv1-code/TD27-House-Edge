# House Edge plus — complete runnable source

This version improves the previous next model while keeping its source and output folders intact. Use the fully extracted participant kit containing `public/`, `slot_forecasting/`, `run_starter.py` and `check_submission.py`.

## Install and run in Windows VS Code

Save `install_house_edge_plus.py` in Downloads. In the VS Code terminal already open in your TD27-House-Edge project folder, run these commands one at a time:

```powershell
python "$env:USERPROFILE\Downloads\install_house_edge_plus.py"
python house_edge_plus.py --out strong-04 --cv
python check_submission.py strong-04/cv_predictions.csv --split validation
python check_submission.py strong-04/submission.csv
```

The installer contains all plus-model source files. It installs them into the terminal's current project folder and checks imports. Existing plus source is backed up before replacement. It preserves `next_model/`, `house_edge_next.py`, public data and previous output folders.

NumPy and SciPy are required; your working next-model installation already has them. If the installer reports a missing dependency, run `python -m pip install -r requirements_plus.txt` and retry training. No GPU, PyTorch, pandas or scikit-learn is required. Python 3.11+ is supported; verified here with Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0.

Wait for training and `ZIP reproduction PASSED` before running the checker commands. It can take several minutes on a laptop. `strong-04` must be a new output folder; choose another fresh name consistently if it already exists.

Alternatively, extract the complete `house_edge_plus_source.zip` into the kit folder, keeping `house_edge_plus.py`, `plus_model/` and `requirements_plus.txt` together.

## Verified comparison

On the unmodified main kit, season `5fffdc4d0c700bbf61c3f911`:

| Episode-grouped evaluation | Previous next model | Plus model |
| --- | ---: | ---: |
| Original four-fold CV | 57.863653 / 60 | 58.376322 / 60 |
| Second four-fold partition after settings freeze | 57.844219 / 60 | 58.361919 / 60 |

The original CV comparison gains 0.512670 points and reduces the average condition-normalized squared error by 24.664954%. All four conditions improve:

| Condition | Previous loss ratio | Plus loss ratio |
| --- | ---: | ---: |
| stable | 0.025277 | 0.020925 |
| short_history | 0.069273 | 0.049963 |
| drift | 0.024245 | 0.021183 |
| shared_change | 0.028887 | 0.019185 |

Lower loss ratios are better. Both CV partitions use the same released validation targets. The original partition was used for model selection; the second partition is a robustness check, not an independent dataset or private-test estimate. No further settings were chosen using the second partition.

## What changed

Training histories support a strong link between each machine's win lag pattern and payout-size lag pattern. An EM-fitted joint prior lets the more plentiful win observations help identify the payout-size pattern. Both component filters remain conditional on their own past observations. Missing payout contexts are marginalized using smoothed empirical rates calculated strictly from earlier observations.

The model retains finite-grid hidden-state filtering over uncertain win odds, two positive-payout templates per machine type, a regularized residual correction, and a causal episode-level logistic filter. Identifiers group histories and label output rows; they do not retrieve saved forecasts. Missing ticks remain unknown, and input observations at/after a machine's cutoff are rejected.

Whole episodes stay together in each CV fold. Each fold refits population centers, positive-payout templates, amount-pattern priors, the shared lag prior and the residual readout outside that fold. Pattern discovery uses training histories only. The shared prior also uses training-history likelihoods, with population parameters learned outside each validation fold.

## Which files and score to use

`strong-04/cv_predictions.csv` contains the episode-grouped out-of-fold forecasts. Use its **58.376322 / 60 CV score** to compare with the previous model.

The final checkpoint then fits all released validation targets. `validation_predictions.csv` gives a **58.607266 / 60 fit score**, which includes fitting data and is optimistic. It is not a hidden-test score. The actual event score remains unknown until the organizers evaluate the test predictions.

After the local checks pass, submit the matching **`strong-04/submission.csv` and `strong-04/entry.zip`** according to the event's submission instructions. The installer and source ZIP are for running the model, not the competition entry ZIP. Nothing uploads automatically.

## Reproduction and correctness

The generated entry pins the installed NumPy/SciPy versions. Offline inference reads only its checkpoint, metadata and test queries/history. It needs no training/validation files, forecast CSV, conditions, internet or private test answers.

Both CSV formats pass the official local checker. The actual generated ZIP was extracted and run with only test inputs: all 1,536 predictions matched with maximum probability difference 0. The runnable CV agrees with the independently cached research predictions to 6.7e-16. Additional checks passed for batch and row-order invariance, empty histories, all-loss histories, sparse histories, future-tick rejection and causal missing-context calculation.

The methods draw on Bayesian sequential state estimation, dynamic model averaging and latent-variable EM. This is a custom implementation. Primary background: [McCormick et al., Dynamic Logistic Regression and Dynamic Model Averaging for Binary Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC3831847/) and [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742).

## Adaptation

Revalidate these settings when the adaptation kit releases. Use a separate extracted adaptation kit, install this source there and choose a fresh output folder. The runner respects the kit's round/season metadata and rejects prediction with a checkpoint from another round or season. No adaptation performance is promised before its data releases.
