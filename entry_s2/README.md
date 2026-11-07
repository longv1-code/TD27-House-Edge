# House Edge entry (main round, solution s2)

- Environment: python image, numpy==2.4.4 (see requirements.txt). No network, no other packages. Tested on Python 3.14.0 and 3.12.3.
- Recreate: python predict.py --data /data --out /output/submission.csv --solution s2
- The command loads model.npz (trained weights) and runs a deterministic forward pass on /data/test. Runtime is under 1 minute on 2 cores.
- Solution s2: 5-seed MLP ensemble on per-machine lag features, trained on train+validation.
- Features only read ticks strictly before the target tick. No outside data.
