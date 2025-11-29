#!/bin/bash
echo "Ejecutando entrenamiento..."
export LR=$3
export MODEL_TYPE=$4
export BS=$5
export LORA=$6
export FPN=$7
export BACKBONE=$8
export NO_ALBUMENTATIONS_UPDATE=1

rm -rf ../../checkpoints/*
python train.py --config config.yaml
