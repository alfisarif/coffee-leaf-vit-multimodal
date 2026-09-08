# Checkpoints

The completed Colab run reported SHA-256 hashes for the four core checkpoints and generated additional EfficientNet-B0 and MaxViT-Tiny checkpoints. The source archive available for this repository reconstruction does **not** contain the binary `.pt` files themselves, so this folder intentionally does not contain fabricated or empty checkpoint binaries.

Recorded hashes from the completed run:

| Experiment | Selected epoch | SHA-256 |
|---|---:|---|
| Sensor-only | 10 | `ff805645d0da70a00acee56f217bb1e5fa1e2a12b0611314995399ad2a55c537` |
| ViT-only | 10 | `616a84247300e2826c2a7ac8b57eaadf745a603c755565371ea9a0ca0ecd48f9` |
| ViT + Sensor | 9 | `d3c3d9a814f9950aab6197b36319fab2a04fe1db2c386029ace199588dde6196` |
| ViT + Shuffled Sensor | 9 | `63351ab9d5a070a10547bc503f9dc8a71b99842290190e6f0e2162016386eb90` |
| EfficientNet-B0 | 9 | See the completed run artifact if available |
| MaxViT-Tiny | 3 | `e1d6cd2e3bfbb2d52439c32dbf7c251fe2506caaad96105026bd92c2e6039f71` |

For public release, upload the exact binary checkpoints through Git LFS or GitHub Releases and update this file with the actual download locations and hashes.
