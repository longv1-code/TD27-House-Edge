# House Edge — submission README template

Copy this into your submission as `README.md`. Replace the bracketed fields with your actual details. Keep it short; this is a setup guide, not a graded writeup.

## Entry

- Team ID: 3xko
- Round: Main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted_main/mycheck_blend.csv
- Forecast SHA-256: 162e0272b9de07444b5e92bdbb805e6131779a07dacfd5971eee86deede50b1b
- Code version: submitted_main/entry_blend_main.zip
- Environment: `python` image; numpy==2.5.4 and scipy==1.18.1

## Setup

- Python/runtime version: Python 3.13.14
- OS tested: Windows 11 development environment; organizer reproduction uses the Python image
- Dependencies: `requirements.txt` with numpy==2.5.4 and scipy==1.18.1
- Required hardware: CPU only; no GPU required
- Put the released data at: `public/`

The organizer rerun mounts the current round at `/data`, your code at `/submission` (read-only), and writable scratch/output at `/tmp` and `/output`. Adaptation also mounts main-round public data at `/main_data`. Your `reproduce.json` command must write `/output/submission.csv` without network access. Include this README and `reproduce.json` at the code ZIP root. Organizers prepare the pinned dependencies in an isolated environment before running predictions offline.

## Recreate the forecast

Run from: the code ZIP root

```sh
python predict_blend.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: `checkpoint.json`, `gen_params.npz`, `plus_model/`, and the packaged prediction modules
- Random seeds and settings: deterministic; no randomness
- What the command does: runs the packaged plus model and generative model, then averages their predictions 50/50
- Measured runtime and hardware: CPU-only; runtime not recorded
- Training command, if separate: not applicable; trained checkpoints and parameters are included in the archive
- Main-round data/model reused in adaptation, if any: packaged models fitted using released train and validation data; no private test answers

If reruns are not deterministic, describe the source of variation and how the submitted predictions were produced. Include the saved checkpoint/settings needed to reproduce them.

## Local check

```sh
python check_submission.py submitted_main/mycheck_blend.csv --data public
```

- Checker result: valid format; 1536 queries; main/test; season `5fffdc4d0c700bbf61c3f911`
- Fresh-folder reproduction checked: no; organizer reproduction should use the included archive and pinned dependencies

Local validation does not submit an entry. Upload the forecast CSV (up to 10 MB) and code ZIP (up to 100 MB) through the correct round's Google Form, click Submit, and wait for confirmation. Keep the exact files. The latest response before cutoff counts, even if invalid; organizers validate files and check prize contenders' code afterward. The rerun budget is 30 minutes, two CPU cores, and 16 GiB RAM.
