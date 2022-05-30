#!/bin/bash
python main.py --sys_name viscoelastic \
    --train_SAE True --max_epoch_SAE 10000 --miles_SAE 5000 --gamma_SAE 1e-1 \
    --lr_SAE 1e-3 --lambda_r_SAE 1e-4 --layer_vec_SAE 400, 160, 160, 10 \
    --train_SPNN True --max_epoch_SPNN 5000 \
    --lr_SPNN 1e-3 --lambda_r_SPNN 1e-3 --lambda_d_SPNN 5e2 --hidden_vec_SPNN 24, 24, 24, 24, 24 \
    --save_plots True
