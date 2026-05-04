#!/usr/bin/env bash
set -e
cd ~/HE_Test/FHE_CKKS

BINARY=./build/bin/run_he_inference_otto
ARTIFACT=he_artifacts_otto_h256
N_THREADS=32

###############################################################
# Phase 1: Sanity test on Otto full-train (200 samples)
###############################################################
echo
echo "=== Phase 1: SANITY (Otto, 200 samples) ==="
mkdir -p logs
"${BINARY}" "${ARTIFACT}" 200 8 40 \
    "60,40,40,40,40,40,60" "otto_sanity" 16384 \
    2>&1 | tee logs/run_otto_sanity.log
echo "If above ran successfully, press Enter to continue with full experiments"
echo "(or Ctrl-C to abort)"
read

###############################################################
# Phase 2: Exp 2 — precision sweep on Otto full-train
###############################################################
echo
echo "=== Phase 2: Exp 2 — Precision sweep (Otto) ==="
OUT_DIR=logs
declare -a EXP2_CONFIGS=(
    "25:35,25,25,25,25,25,35"
    "30:50,30,30,30,30,30,50"
    "35:55,35,35,35,35,35,55"
    "40:60,40,40,40,40,40,60"
    "45:60,45,45,45,45,45,60"
)
for cfg in "${EXP2_CONFIGS[@]}"; do
    SCALE="${cfg%%:*}"
    BITS="${cfg##*:}"
    TAG="s${SCALE}"
    LOG="logs//run_otto_${TAG}.log"
    echo "  scale=2^${SCALE}  bits=${BITS}  tag=${TAG}"
    "${BINARY}" "${ARTIFACT}" 0 "${N_THREADS}" \
        "${SCALE}" "${BITS}" "${TAG}" 16384 \
        2>&1 | tee "${LOG}" || echo "[!] ${TAG} failed"
done

###############################################################
# Phase 3: Exp 3 — config feasibility on Otto full-train
###############################################################
echo
echo "=== Phase 3: Exp 3 — Config feasibility (Otto) ==="
mkdir -p "${OUT_DIR}"
declare -a EXP3_CONFIGS=(
    "N16K_d4 : 40 : 60,40,40,40,40,60        : 16384"
    "N16K_d5 : 40 : 60,40,40,40,40,40,60     : 16384"
    "N32K_d5 : 40 : 60,40,40,40,40,40,60     : 32768"
    "N32K_d6 : 40 : 60,40,40,40,40,40,40,60  : 32768"
)
for cfg in "${EXP3_CONFIGS[@]}"; do
    cfg_clean=$(echo "$cfg" | tr -d ' ')
    TAG=$(  echo "$cfg_clean" | cut -d: -f1)
    SCALE=$(echo "$cfg_clean" | cut -d: -f2)
    BITS=$( echo "$cfg_clean" | cut -d: -f3)
    N=$(    echo "$cfg_clean" | cut -d: -f4)
    LOG="logs/run_otto_${TAG}.log"
    echo "  tag=${TAG}  N=${N}  scale=2^${SCALE}  bits=${BITS}"
    "${BINARY}" "${ARTIFACT}" 0 "${N_THREADS}" \
        "${SCALE}" "${BITS}" "${TAG}" "${N}" \
        2>&1 | tee "${LOG}" || echo "[!] ${TAG} failed"
done

###############################################################
# Phase 4: Exp 4 — data efficiency on Otto small-pools
###############################################################
echo
echo "=== Phase 4: Exp 4 — Data efficiency (Otto small-pools) ==="
mkdir -p "${OUT_DIR}"
declare -a POOLS=(
    "he_artifacts_otto_pool_01"
    "he_artifacts_otto_pool_05"
    "he_artifacts_otto_pool_10"
    "he_artifacts_otto_pool_20"
)
SCALE=40
BITS="60,40,40,40,40,40,60"
for ART in "${POOLS[@]}"; do
    [ -d "${ART}" ] || { echo "[!] missing ${ART}"; continue; }
    TAG="${ART#he_artifacts_otto_}"
    LOG="logs/run_otto_${TAG}.log"
    echo "  artifact=${ART}  tag=${TAG}"
    "${BINARY}" "${ART}" 0 "${N_THREADS}" "${SCALE}" "${BITS}" "${TAG}" 16384 \
        2>&1 | tee "${LOG}" || echo "[!] ${TAG} failed"
done

echo
echo "============================================"
echo "All Otto experiments complete."
echo "============================================"
