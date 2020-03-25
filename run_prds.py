#!/usr/bin/env bash

while IFS= read -r exp_name; do
    echo "Experiment: $exp_name"
    echo "Sourcing conda.sh"
    source "/anaconda3/etc/profile.d/conda.sh"
    echo "Activating conda environment"
    conda activate pytorch
    
    python train_InfoGAN_general_mnist.py \
            --config_name=${exp_name}  \
            --train=0 \
            --eval=0 \
            --compute_prd=1 \

done < prd_models.txt
