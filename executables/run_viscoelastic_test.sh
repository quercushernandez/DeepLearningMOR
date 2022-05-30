#!/bin/bash
python main.py --sys_name viscoelastic --train_SAE False --train_SPNN False \
    --layer_vec_SAE 400, 160, 160, 10 \
    --hidden_vec_SPNN 24, 24, 24, 24, 24 \
    --save_plots True
