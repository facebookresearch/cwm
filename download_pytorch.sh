#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the CWM License Agreement.

####
## NOTE: For downloading CWM please refer to https://github.com/facebookresearch/cwm/tree/main/release?tab=readme-ov-file#model-download
####

set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <SIGNED_URL> [MODEL_TYPES]"
    echo "  SIGNED_URL: The signed URL with wildcard (*) received in your email"
    echo "  MODEL_TYPES: Optional comma-separated list of models to download (cwm,cwm-sft,cwm-pretrain)"
    echo "               If not specified, all models will be downloaded"
    echo ""
    echo "Examples:"
    echo "  $0 \"\$URL\"                           # Download all models"
    echo "  $0 \"\$URL\" cwm                       # Download only cwm model"
    echo "  $0 \"\$URL\" cwm,cwm-sft               # Download cwm and cwm-sft models"
    exit 1
fi

SIGNED_URL="$1"
MODEL_TYPES_INPUT="$2"

# Check if the URL contains the wildcard
if [[ "$SIGNED_URL" != *"*"* ]]; then
    echo "Error: The signed URL must contain a wildcard (*) to replace with file paths"
    exit 1
fi

TARGET_FOLDER="model_weights"
mkdir -p "${TARGET_FOLDER}"

# Determine which models to download
if [[ -n "$MODEL_TYPES_INPUT" ]]; then
    IFS=',' read -ra MODEL_TYPES <<< "$MODEL_TYPES_INPUT"
else
    MODEL_TYPES=("cwm" "cwm-sft" "cwm-pretrain")
fi

echo "Downloading README.md..."
README_URL="${SIGNED_URL/'*'/README.md}"
wget --continue "$README_URL" -O "${TARGET_FOLDER}/README.md"

# Process each model type
for MODEL_FOLDER_PATH in "${MODEL_TYPES[@]}"; do
    case $MODEL_FOLDER_PATH in
        "cwm"|"cwm-sft"|"cwm-pretrain")
            TOTAL_SHARDS=8
            ;;
        *)
            echo "Warning: Unknown model type '$MODEL_FOLDER_PATH'. Skipping..."
            continue
            ;;
    esac

    echo ""
    echo "Processing CWM $MODEL_FOLDER_PATH model ($TOTAL_SHARDS shards)"
    mkdir -p "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}"

    echo "Downloading metadata file for $MODEL_FOLDER_PATH..."
    METADATA_URL="${SIGNED_URL/'*'/$MODEL_FOLDER_PATH/.metadata}"
    wget --continue "$METADATA_URL" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/.metadata"

    echo "Downloading configuration file for $MODEL_FOLDER_PATH..."
    PARAMS_URL="${SIGNED_URL/'*'/$MODEL_FOLDER_PATH/params.json}"
    wget --continue "$PARAMS_URL" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/params.json"

    echo "Downloading $TOTAL_SHARDS checkpoint shards for $MODEL_FOLDER_PATH..."
    for SHARD in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="__${SHARD}_0.distcp"
        echo "Downloading $MODEL_FOLDER_PATH shard ${SHARD}/$((TOTAL_SHARDS - 1)): ${SHARD_FILE}"
        SHARD_URL="${SIGNED_URL/'*'/$MODEL_FOLDER_PATH/$SHARD_FILE}"
        wget --continue "$SHARD_URL" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/${SHARD_FILE}"
    done

    echo "Downloading checksum file for $MODEL_FOLDER_PATH..."
    CHECKSUM_URL="${SIGNED_URL/'*'/$MODEL_FOLDER_PATH/checklist.chk}"
    wget --continue "$CHECKSUM_URL" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/checklist.chk"

    echo "Downloading tokenizer for $MODEL_FOLDER_PATH..."
    TOKENIZER_URL="${SIGNED_URL/'*'/$MODEL_FOLDER_PATH/tokenizer.model}"
    wget --continue "$TOKENIZER_URL" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/tokenizer.model"

    echo "Checking checksums for $MODEL_FOLDER_PATH..."
    (cd "${TARGET_FOLDER}" && md5sum -c "${MODEL_FOLDER_PATH}/checklist.chk")

    echo "Successfully downloaded $MODEL_FOLDER_PATH model with $TOTAL_SHARDS checkpoint shards."
done

echo ""
echo "CWM model download completed successfully!"
