#!/bin/bash
echo "Ejecutando test..."
export LR=$3
export MODEL_TYPE=$4
export BS=$5
export LORA=$6
export FPN=$7
export BACKBONE=$8
export NO_ALBUMENTATIONS_UPDATE=1

python test.py --config config.yaml