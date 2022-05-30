#!/bin/bash
python main.py --sys_name rolling_tire --train_SAE False --train_SPNN False \
    --layer_vec_SAE_q 12420, 40, 40, 10 --layer_vec_SAE_v 12420, 40, 40, 10 --layer_vec_SAE_sigma 24840, 80, 80, 20 \
    --hidden_vec_SPNN 198, 198, 198, 198, 198 \
    --save_plots True
