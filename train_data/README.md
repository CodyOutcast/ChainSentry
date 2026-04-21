# Synthetic Monitoring Data

These files describe a realistic but intentionally synthetic training run.

- 15 epochs are used consistently across every training plot.
- Curves are shaped by dataset scale and task coverage, not by the current checkpoint.
- PTXPhish-driven heads stay noisier than Forta/EtherScamDB/Raven-driven heads.
- The validation endpoint is aligned with the evaluation payload stored under `evaluation_data/`.