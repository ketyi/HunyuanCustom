#!/bin/bash

# --- Configuration ---
JOBS_DIR=$(dirname $(dirname "$0"))
INPUT_IMAGES_DIR="/localscratch/istvan/HunyuanCustom/inference_photos/"
MODEL_BASE="${JOBS_DIR}/models"
CHECKPOINT_PATH="${MODEL_BASE}/hunyuancustom_audio_720P/mp_rank_00_model_states.pt"
MODEL_NAME='Tencent_HunyuanCustom_Audio_720P'

# --- Environment Variables ---
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE
export NCCL_DEBUG=OFF # Consider setting to 'INFO' or 'WARN' for debugging if needed

# --- Torchrun Command Base Arguments ---
TORCHRUN_CMD_BASE=(
    torchrun
    --nnodes=1
    --nproc_per_node=8
    --master_port 29605
    hymm_sp/sample_batch.py
    --input-audio 'alex.wav'
    --audio-strength 0.8
    --audio-condition
    --pos-prompt "High-quality, realistic video of a person talking. Person is standing still."
    --neg-prompt "Two people, two persons, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
    --ckpt "${CHECKPOINT_PATH}"
    --seed 1982
    --video-size 720 1280
    --sample-n-frames 193
    --cfg-scale 7.5
    --infer-steps 30
    --use-deepcache 1
    --flow-shift-eval-video 13.0
)

# --- Process Each Image ---
mkdir -p ./results/${MODEL_NAME} # Ensure base results directory exists

for image_path in "${INPUT_IMAGES_DIR}"/*.png; do
    if [[ -f "$image_path" ]]; then
        IMAGE_FILENAME=$(basename "$image_path")
        IMAGE_NAME_WITHOUT_EXT="${IMAGE_FILENAME%.*}"
        CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
        OUTPUT_BASEPATH="./results/${MODEL_NAME}/${IMAGE_NAME_WITHOUT_EXT}_${CURRENT_TIME}"

        echo "Processing image: ${image_path}"
        echo "Saving results to: ${OUTPUT_BASEPATH}"

        "${TORCHRUN_CMD_BASE[@]}" \
            --ref-image "$image_path" \
            --save-path "${OUTPUT_BASEPATH}"
        echo "----------------------------------------------------"
    fi
done

echo "All images processed."