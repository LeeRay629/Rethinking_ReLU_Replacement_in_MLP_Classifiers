# Quad4FHE Code Artifact

This repository contains the code artifact for **Rethinking ReLU Replacement in MLP Classifiers: From Approximation to Decision-Preserving**, a decision-aware post-training replacement method for ReLU activations in single-hidden-layer MLP heads and encrypted CKKS inference experiments.

The artifact has two main parts:

1. **Plaintext experiments** for fitting and evaluating quadratic ReLU replacements, classical polynomial baselines, calibration/test agreement diagnostics, and HE artifact export.
2. **CKKS experiments** for encrypted inference on selected MLP heads using Microsoft SEAL 4.1-compatible CKKS.

The DINOv2 and Qwen3 experiments evaluate **encrypted classification-head inference on encrypted feature vectors**. They do **not** evaluate the full DINOv2 or Qwen3 backbone under FHE. In the intended frozen-backbone workflow, the client computes features locally, encrypts the feature vector, and sends the encrypted feature vector to the server-side CKKS evaluator.

---

## Repository layout

A typical checkout is organized as follows.

```text
Quad4FHE_Code/
  README.md
  LICENSE
  CITATION.cff
  environment.yml
  quad4fhe-0.1.0-new.tar.gz

  Single_Layer/
    AG_News(NLP_multi-class)/
    Breast Cancer Wisconsin(binary)/
    CIFAR-10(CV_multi-class)/
    CIFAR-100(CV_multi-class)/
    Diabetes(binary)/
    Otto Group(multi-class)/
    Shuttle(multi-class)/
    SST-5(NLP_multi-class)/
    DINOv2 / Qwen3 experiment notebooks, depending on your checkout
    results/

  FHE_CKKS/
    CMakeLists.txt
    run_exp_otto_all.sh
    run_exp_fgvc_all.sh
    run_exp_massive_all.sh
    src/
      run_he_inference_otto.cpp
      run_he_inference_fgvcaircraft.cpp
      run_he_inference_massive.cpp
    thirdparty/
      SEAL-4.1-bs/
```

The `quad4fhe-0.1.0-new.tar.gz` archive is included so that users can reproduce the plaintext experiments without separately packaging the local `quad4fhe` library.

The `FHE_CKKS/thirdparty/SEAL-4.1-bs/` directory is kept as a convenience copy of the SEAL-compatible library used in the original experiments. If you already have a compatible Microsoft SEAL 4.1 installation, you may use that instead by setting `SEAL_DIR` when configuring CMake.

---

## Hardware used for the reported experiments

The plaintext experiments require a CUDA-capable GPU for the larger DINOv2 and Qwen3 runs.

The CKKS experiments reported in the paper were run on:

```text
CPU: INTEL(R) XEON(R) GOLD 6530
Instruction set: AVX-512 supported
CKKS full-sweep threads: 32
CKKS sanity-check threads: 8
```

Different CPUs, compiler flags, SEAL builds, and thread counts can change latency numbers.

---

## 1. Create the Python environment

The recommended environment is a CUDA-enabled Conda environment.

```bash
conda env create -f environment.yml
conda activate quad4fhe
```

Then install the local Quad4FHE package:

```bash
python -m pip install --force-reinstall --no-cache-dir ./quad4fhe-0.1.0-new.tar.gz
```

Verify the installation:

```bash
python - <<'PY'
import quad4fhe
print("quad4fhe version:", getattr(quad4fhe, "__version__", "unknown"))
print("quad4fhe path:", quad4fhe.__file__)
PY
```

If you run the notebooks from Jupyter, restart the kernel after installing or reinstalling `quad4fhe`.

---

## 2. Data and model preparation

### Single-hidden-layer tabular/text experiments

Some datasets are included in this repository for convenience. If your local checkout does not contain them, place the expected files in the same folder as the corresponding notebook.

Examples:

```text
Single_Layer/AG_News(NLP_multi-class)/train.csv
Single_Layer/Diabetes(binary)/diabetes_dataset.csv
Single_Layer/Otto Group(multi-class)/train.csv
Single_Layer/Shuttle(multi-class)/shuttle.trn
Single_Layer/Shuttle(multi-class)/shuttle.tst
Single_Layer/SST-5(NLP_multi-class)/sst5_train.csv
Single_Layer/SST-5(NLP_multi-class)/sst5_validation.csv
Single_Layer/SST-5(NLP_multi-class)/sst5_test.csv
```

### CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 are **not automatically downloaded by the repository setup**. Download them manually and place the dataset files in the same folder as the corresponding notebook:

```text
Single_Layer/CIFAR-10(CV_multi-class)/
Single_Layer/CIFAR-100(CV_multi-class)/
```

The notebooks use official CIFAR train/test partitions and then create internal validation/calibration splits where needed.

### DINOv2 experiments

For DINOv2 experiments, first check that the DINOv2 model and image data are available locally. The results JSON files expect paths similar to:

```text
./dinov2-base
./data/dinov2_base_cifar100_cls_features.npz
./data/dinov2_base_tinyimagenet_cls_features.npz
./data/dinov2_base_stanfordcars_cls_features.npz
./data/dinov2_base_fgvcaircraft_cls_features.npz
```

If the cached feature file does not exist, the notebook will compute features using the DINOv2 backbone. This can take time and requires a GPU.

### Qwen3 embedding experiments

For Qwen3 experiments, make sure the Hugging Face datasets and model are accessible before launching long runs. The notebooks use models/datasets such as:

```text
Qwen/Qwen3-Embedding-0.6B
mteb/banking77
mteb/amazon_massive_intent
Davlan/sib200
```

A quick connectivity check is:

```bash
python - <<'PY'
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

print(load_dataset("mteb/banking77", split="test[:5]"))
print(AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B").__class__)
print(AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True).__class__)
PY
```

If you use a local mirror or pre-downloaded model, update the paths in the notebooks accordingly.

---

## 3. Running plaintext experiments

Each dataset folder contains autosave notebooks. Run the full-train and small-pool notebooks from their own directory. For example:

```bash
cd "Single_Layer/Diabetes(binary)"
jupyter lab Diabetes_fulltrain_autosave.ipynb
jupyter lab Diabetes_smallpool_autosave.ipynb
```

The notebooks automatically save logs and structured outputs under:

```text
../results/<DATASET_NAME>/<EXPERIMENT_NAME>/
```

For example:

```text
Single_Layer/results/Diabetes/fulltrain/Diabetes_fulltrain_results.json
Single_Layer/results/Diabetes/fulltrain/Diabetes_fulltrain_summary.csv
Single_Layer/results/Diabetes/smallpool/Diabetes_smallpool_results.json
```

The combined dataset-level JSON is saved as:

```text
Single_Layer/results/<DATASET_NAME>/<DATASET_NAME>_results.json
```

The most important diagnostic fields are:

```text
constraint_version
method_used
hard_feasible
calib_n
calib_agreement
calib_mismatch_count
exact_preserved_on_calib
test_n
test_agreement
test_mismatch_count
calib_test_agreement_gap
num_pairwise_constraints
min_pairwise_margin
normalized_min_pairwise_margin
slack_positive_count
sum_slack
max_slack
selected_C
soft_C_grid
soft_trace
```

For binary RCH runs, `mu_grid`, `mu_p`, and `mu_n` are also reported.

---

## 4. HE artifact export for CKKS

The plaintext notebooks export HE artifact folders containing MLP weights, polynomial coefficients, plaintext reference logits, and test features needed by the C++ CKKS programs.

The CKKS scripts expect the relevant artifact folders to be under `FHE_CKKS/` unless you edit the shell scripts.

For the main CKKS experiments, copy or move these folders into `FHE_CKKS/`:

```text
he_artifacts_otto_h256
he_artifacts_dinov2_fgvcaircraft_h256
he_artifacts_qwen3_massive_intent_h256
```

For small-pool encrypted experiments or custom scripts, you may also need these folders:

```text
he_artifacts_qwen3_massive_intent_pool_01
he_artifacts_qwen3_massive_intent_pool_05
he_artifacts_qwen3_massive_intent_pool_10
he_artifacts_qwen3_massive_intent_pool_20

he_artifacts_dinov2_fgvcaircraft_pool_01
he_artifacts_dinov2_fgvcaircraft_pool_05
he_artifacts_dinov2_fgvcaircraft_pool_10
he_artifacts_dinov2_fgvcaircraft_pool_20
```

You can either copy them:

```bash
cp -r path/to/he_artifacts_otto_h256 FHE_CKKS/
cp -r path/to/he_artifacts_dinov2_fgvcaircraft_h256 FHE_CKKS/
cp -r path/to/he_artifacts_qwen3_massive_intent_h256 FHE_CKKS/
```

or edit `run_exp_otto_all.sh`, `run_exp_fgvc_all.sh`, and `run_exp_massive_all.sh` to point to the artifact folders in their original locations.

---

## 5. Build SEAL 4.1

The repository keeps a local SEAL-compatible tree at:

```text
FHE_CKKS/thirdparty/SEAL-4.1-bs/
```

Build and install it locally:

```bash
cd FHE_CKKS/thirdparty/SEAL-4.1-bs
rm -rf build install
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSEAL_BUILD_DEPS=ON \
  -DSEAL_BUILD_EXAMPLES=OFF \
  -DSEAL_BUILD_TESTS=OFF
cmake --build build -j32
cmake --install build --prefix "$(pwd)/install"
```

Find the generated SEAL CMake config:

```bash
find "$(pwd)/install" -name '*SEAL*Config.cmake' -o -name 'SEALConfig.cmake'
```

The resulting directory is passed to `FHE_CKKS` as `SEAL_DIR`. Depending on the exact SEAL build, it is usually one of:

```text
FHE_CKKS/thirdparty/SEAL-4.1-bs/install/lib/cmake/SEAL-4.1
FHE_CKKS/thirdparty/SEAL-4.1-bs/install/lib/cmake/SEAL
```

If you already have Microsoft SEAL 4.1 installed elsewhere, you can skip this step and use your existing `SEAL_DIR`.

---

## 6. Build the CKKS executables

From `FHE_CKKS/`:

```bash
cd FHE_CKKS
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_NATIVE_OPT=ON \
  -DSEAL_DIR="$(pwd)/thirdparty/SEAL-4.1-bs/install/lib/cmake/SEAL-4.1"
cmake --build build -j32
```

If CMake cannot find SEAL, replace `SEAL_DIR` with the path found in the previous step.

The build should generate:

```text
FHE_CKKS/build/bin/run_he_inference_otto
FHE_CKKS/build/bin/run_he_inference_fgvcaircraft
FHE_CKKS/build/bin/run_he_inference_massive
```

---

## 7. Run CKKS experiments

From `FHE_CKKS/`, run:

```bash
./run_exp_otto_all.sh 2>&1 | tee otto_outputs.log
./run_exp_fgvc_all.sh 2>&1 | tee fgvc_outputs.log
./run_exp_massive_all.sh 2>&1 | tee massive_outputs.log
```

The scripts first run a 200-sample sanity check with 8 threads and then run the full sweeps. The full-sweep scripts use 32 threads in the reported configuration.

The C++ programs print:

```text
requested security = tc128
SEAL security level = tc128
MaxBitCount(tc128)
Plain<->HE agreement
PH-mis
max/mean logit error
CKKS operation counts
latency breakdown
```

The paper reports no bootstrapping for these selected CKKS experiments.

---

## 8. Notes on reproducibility

- Plaintext experiments can be sensitive to GPU type, PyTorch/CUDA version, and deterministic settings.
- CKKS latency depends strongly on CPU, compiler, SEAL build, `-march=native`, and thread count.
- The full experiments use fixed seeds in the notebooks where specified, but the artifact does not claim multi-seed statistical confidence intervals.
- DINOv2 and Qwen3 experiments depend on external model/data availability and may require prior Hugging Face downloads.
- The CKKS experiments cover encrypted MLP-head inference only, not full encrypted image/text backbones.

---

## 9. Licensing and third-party material

This repository includes original code, third-party code, raw/processed datasets, and references to external pretrained models.

- The repository-level `LICENSE` applies to the original Quad4FHE code unless a file or subdirectory states otherwise.
- `FHE_CKKS/thirdparty/SEAL-4.1-bs/` retains its own upstream license and notices.
- Dataset files retain their original dataset licenses and terms of use.
- Hugging Face models and datasets retain their own licenses and terms.

Before redistributing this repository, check that retaining the bundled datasets and third-party source tree is compatible with your intended distribution channel.

---

## 10. Citation

If you use this code, please cite the associated paper and this code artifact. A `CITATION.cff` file is included for GitHub citation support.
