#!/bin/bash
echo "Ejecutando entrenamiento..."

rm -rf ../../checkpoints/*
python train.py --config config.yaml
