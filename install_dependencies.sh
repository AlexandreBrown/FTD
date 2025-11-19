#!/bin/bash
pip install -r ./baselines/ftd/requirements.txt
# Original implementation uses mobile sam : pip install ./baselines/ftd/src/mobile_sam

# We use EfficientViT SAM for faster inference.
# Use the WEIGHTS_FOLDER environment variable if set, otherwise default to "weights"
: "${WEIGHTS_FOLDER:=weights}"

if [ ! -d "$WEIGHTS_FOLDER" ]; then
    mkdir -p "$WEIGHTS_FOLDER"
    echo "Created folder: $WEIGHTS_FOLDER"
fi

# Install EfficientViT-SAM
pip install git+https://github.com/mit-han-lab/efficientvit.git@20317cb7240c81e9ded74501a523846597021133
# Download EfficientViT-SAM weights
SAM_WEIGHTS_FILE="efficientvit_sam_l0.pt"
SAM_WEIGHTS_FILE_URL="https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/$SAM_WEIGHTS_FILE"
if [ ! -f "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" ]; then
    echo "Downloading $SAM_WEIGHTS_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" "$SAM_WEIGHTS_FILE_URL"
    echo "Download complete: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
else
    echo "EfficientViT-SAM weights already exists: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
fi