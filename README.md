# Context-Aware Multimodal Classification of Coffee Leaf Diseases Using Vision Transformer and Agro-Environmental Sensor Data

Research repository for the IJIES manuscript **“Context-Aware Multimodal Classification of Coffee Leaf Diseases Using Vision Transformer and Agro-Environmental Sensor Data.”**

## Overview

This project evaluates coffee leaf disease classification using visual information and three agro-environmental variables: relative humidity, soil moisture, and soil pH. The proposed model combines a pretrained Vision Transformer (ViT-Base) with a sensor MLP through feature-level fusion.

The final protocol contains **840 original paired observations**, split before augmentation into **672 source-separated training observations** and **168 fixed validation observations**. Training-only sensor augmentation produces **2,016 effective training entries**.

## Experimental configurations

1. Sensor-only MLP
2. EfficientNet-B0
3. ViT-Base
4. MaxViT-Tiny
5. ViT + Sensor
6. ViT + Shuffled Sensor

The shuffled-sensor control jointly permutes the complete three-variable sensor vector using seed 4242, breaking the original image-sensor correspondence while preserving the sensor-vector structure.

## Final reported results

| Configuration | Validation accuracy | Macro F1 | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|
| Sensor-only MLP | 71.43% | 72.95% | 74.43% | 0.618 |
| EfficientNet-B0 | 100.00% | 100.00% | 100.00% | 1.000 |
| ViT-Base | 100.00% | 100.00% | 100.00% | 1.000 |
| MaxViT-Tiny | 100.00% | 100.00% | 100.00% | 1.000 |
| **ViT + Sensor** | **100.00%** | **100.00%** | **100.00%** | **1.000** |
| ViT + Shuffled Sensor | 100.00% | 100.00% | 100.00% | 1.000 |

The equal performance of ViT + Sensor, ViT-only, and ViT + Shuffled Sensor means the present in-domain validation does **not** demonstrate an incremental accuracy benefit from correct image-sensor correspondence. The sensor-only result nevertheless shows that the environmental variables contain predictive information.

## Repository structure

```text
coffee-leaf-vit-multimodal/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── models/
│   ├── dataset/
│   ├── engine/
│   └── utils/
├── baselines/
├── experiments/
├── data/
├── results/
├── notebooks/
├── docs/
└── paper/
```

## Reproduction

Raw images are not distributed in this repository. Place the authorized image collection under `data/images/` while preserving filenames referenced by `data/metadata/image_manifest_reviewer_v2.csv`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --experiment sensor_only --images-dir data/images
python -m src.train --experiment vit_only --images-dir data/images
python -m src.train --experiment vit_sensor --images-dir data/images
python -m src.train --experiment shuffled_sensor --images-dir data/images
python -m src.train --experiment efficientnet_b0 --images-dir data/images
python -m src.train --experiment maxvit_tiny --images-dir data/images
```

For the reviewer rerun, the Colab notebook `notebooks/Reproduce_Section4.ipynb` preserves the completed experimental commands and their recorded outputs.

## Checkpoints and exact prediction files

The completed Colab run recorded SHA-256 values for the core checkpoints, but the binary checkpoint files and exact validation prediction CSVs were **not present in the source archive used to reconstruct this repository**. They are therefore not fabricated here. See `results/checkpoints/README.md` and `results/predictions/README.md`.

Before making a public claim of full binary-level reproducibility, copy the exact completed `.pt` files and prediction CSVs from the finished runtime into the repository or, preferably, publish them through GitHub Releases/Git LFS and update the artifact manifest.

## Data and licensing note

The repository includes metadata, source IDs, split definitions, and sensor records needed to document the experimental protocol. Raw images should only be redistributed if the data owner permits redistribution.

## Manuscript

The revised manuscript and point-by-point reviewer response are included under `paper/` for traceability.

## Citation

Use `CITATION.cff` or the manuscript citation once the journal metadata and complete author list are finalized.
