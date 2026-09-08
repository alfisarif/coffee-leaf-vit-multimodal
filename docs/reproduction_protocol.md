# Reproduction Protocol

## 1. Dataset
- 840 original paired observations.
- Four canonical classes: Leaf spot, Coffee rust, Sooty mold, Healthy.
- Source grouping uses `source_id`.
- Source-separated stratified split: 672 original training observations and 168 original validation observations.
- Validation remains original and unaugmented.

## 2. Training augmentation
- Each training observation is retained plus two sensor-perturbed variants, yielding 2,016 effective training entries.
- Relative humidity and soil moisture: Gaussian noise SD 1.2.
- Soil pH: Gaussian noise SD 0.08.
- Values are clipped to plausible ranges defined in the dataset-generation protocol.
- Visual augmentation during training: horizontal/vertical flip, rotation up to 20 degrees, brightness 0.15 and contrast 0.15.

## 3. Sensor scaling
`StandardScaler` is fitted only on the 672 original training observations, then applied to augmented training entries and the fixed validation set.

## 4. Models
1. Sensor-only MLP
2. EfficientNet-B0
3. ViT-Base
4. MaxViT-Tiny
5. ViT + Sensor
6. ViT + Shuffled Sensor

The shuffled-sensor control jointly permutes the complete three-variable sensor vector across observations using seed 4242, breaking image-sensor correspondence while preserving vector structure.

## 5. Optimization
AdamW, learning rate 2e-5, weight decay 0.01, batch size 8, maximum 10 epochs, seed 42, NVIDIA Tesla T4. The ViT uses `google/vit-base-patch16-224` with all ViT parameters trainable.

## 6. Checkpoint selection
Minimum validation cross-entropy loss is the primary criterion. If tied within 1e-12, higher validation accuracy is selected; if still tied, the earlier epoch is retained. No early stopping is used.

## 7. Evaluation
The selected checkpoint is restored and a single validation inference pass produces prediction probabilities. Metrics are generated from the saved prediction file. Source-level predictions are obtained by averaging class probabilities across records sharing the same `source_id`.

## 8. Commands
```bash
pip install -r requirements.txt
python -m src.train --experiment sensor_only --images-dir data/images
python -m src.train --experiment vit_only --images-dir data/images
python -m src.train --experiment vit_sensor --images-dir data/images
python -m src.train --experiment shuffled_sensor --images-dir data/images
python -m src.train --experiment efficientnet_b0 --images-dir data/images
python -m src.train --experiment maxvit_tiny --images-dir data/images
```

For a full reviewer reproduction, also run the Colab notebook in `notebooks/Reproduce_Section4.ipynb` and compare outputs with `results/tables/` and `results/reported_final_results.csv`.
