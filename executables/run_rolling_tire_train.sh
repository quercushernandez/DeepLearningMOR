#!/bin/bash
python main.py --sys_name rolling_tire \
    --train_SAE True --max_epoch_SAE 4000 --miles_SAE 2000 --gamma_SAE 1e-1 \
    --lr_SAE 1e-3 --lambda_r_SAE 1e-2 --layer_vec_SAE_q 12420, 40, 40, 10 --layer_vec_SAE_v 12420, 40, 40, 10 --layer_vec_SAE_sigma 24840, 80, 80, 20 \
    --train_SPNN True --max_epoch_SPNN 12000 --miles_SPNN 3000, 6000 --gamma_SPNN 5e-1 \
    --lr_SPNN 1e-3 --lambda_r_SPNN 1e-4 --lambda_d_SPNN 5e2 --hidden_vec_SPNN 198, 198, 198, 198, 198 \
    --save_plots True
