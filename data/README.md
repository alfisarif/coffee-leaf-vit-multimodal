# Data

This repository distributes metadata, source identifiers, split files, and sensor records required to document the final experimental protocol. Raw coffee leaf images are **not** included.

The final protocol uses 840 original paired observations, split before augmentation into 672 source-separated training observations and 168 fixed validation observations. Training-only sensor augmentation produces 2,016 effective training entries. Sensor scaling is fit only on the 672 original training observations.

Place the raw image collection under `data/images/` and preserve the filenames referenced by `metadata/image_manifest_reviewer_v2.csv`.
