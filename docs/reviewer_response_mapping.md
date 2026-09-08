# Reviewer Response Mapping

| Reviewer request | Repository evidence |
|---|---|
| Split before augmentation | `src/dataset/` + `data/splits/` |
| Source separation | `source_id` in metadata and split files |
| Four canonical classes | `src/config.py`, `configs/dataset.yaml` |
| Sensor-only / image-only / multimodal / shuffled controls | `configs/experiments.yaml`, `src/train.py` |
| Joint sensor-vector shuffling | `src/dataset/augmentation.py` |
| Checkpoint rule | `configs/training.yaml`, `docs/reproduction_protocol.md` |
| Exact correct/incorrect counts | `results/tables/table4_results.csv` and epoch histories after rerun |
| Prediction-file evaluation | `src/evaluate.py`, `src/utils/metrics.py` |
| Source-level aggregation | `src/utils/metrics.py` |
| Sensor shortcut diagnostics | `results/sensor_diagnostics/summary.csv` |
| Reproducibility notebook | `notebooks/Reproduce_Section4.ipynb` |
| Paper and response letter | `paper/` |

## Important artifact note

The binary checkpoints and exact validation prediction CSVs generated in the completed Colab runtime are not contained in the source archive used to reconstruct this GitHub repository. Their recorded SHA-256 values are preserved where available. Before claiming complete public reproducibility, add the exact binaries and prediction files from the completed run, preferably through Git LFS or GitHub Releases.
