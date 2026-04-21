# Synthetic Evaluation Data

These files are designed to match the synthetic training monitor under `train_data/`.

- The evaluation bundle is framed as a stratified holdout benchmark, not a raw class-imbalanced dump.
- Validation metrics sit slightly above test metrics.
- Severity uses a more balanced benchmark mix so the label-balance chart is readable.
- Dataset-level strengths still follow the known roles of Forta, eth-labels, EtherScamDB, PTXPhish, and Raven.