# Notices and third-party material

This repository contains multiple categories of material:

1. Original Quad4FHE experiment code and notebooks.
2. A packaged local Python library archive: `quad4fhe-0.1.0-new.tar.gz`.
3. A SEAL-compatible third-party source tree under `FHE_CKKS/thirdparty/SEAL-4.1-bs/`.
4. Dataset files and cached/generated experiment summaries.
5. References to external pretrained models and Hugging Face datasets.

The repository-level `LICENSE` applies to original Quad4FHE code unless a file
or subdirectory states otherwise. Third-party code, datasets, and pretrained
models retain their own licenses and terms.

## SEAL / NEXUS-derived third-party code

`FHE_CKKS/thirdparty/SEAL-4.1-bs/` is retained for reproducibility and should be
reviewed together with its upstream license and notices before redistribution.
The paper experiments use no bootstrapping, but this directory may originate
from a bootstrapping-friendly SEAL variant.

## Datasets

Dataset files included in the repository are provided only to reproduce the
experiments. They retain their original dataset licenses and terms of use.
Before redistributing the repository publicly, confirm that each included data
file may be redistributed under its source license.

## Hugging Face models and datasets

DINOv2/Qwen3 experiments depend on external models and datasets. Users are
responsible for complying with their respective model and dataset licenses,
including any Hugging Face terms, access controls, or attribution requirements.
