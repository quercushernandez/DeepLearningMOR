"""main.py"""

import argparse

import numpy as np
import torch

from utilities.utils import str2bool
from SAE_solver import SAE_Solver
from SPNN_solver import SPNN_Solver


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sparse Autoencoder
    SAE_solver = SAE_Solver(args)
    if args.train_SAE:
        SAE_solver.train()
    SAE_solver.test()

    # Detect latent dimensionality 
    x_trunc, latent_idx = SAE_solver.detect_dimensionality()

    # Structure-Preserving Neural Network
    SPNN_solver = SPNN_Solver(args, x_trunc)
    if args.train_SPNN:
        SPNN_solver.train()
    SPNN_solver.test(SAE_solver.SAE, latent_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')

    # Study Case
    parser.add_argument('--sys_name', default='viscoelastic', type=str, help='physic system name')
    parser.add_argument('--train_SAE', default=False, type=str2bool, help='SAE train or test')
    parser.add_argument('--train_SPNN', default=False, type=str2bool, help='SPNN train or test')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--seed', default=2, type=int, help='random seed')

    # Save options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--save_plots', default=False, type=str2bool, help='save results in png file')

    ## Sparse Autoencoder
    # Net Parameters
    parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4, 40*4, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
    parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 40*2, 10*2], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
    parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')

    # Training Parameters
    parser.add_argument('--max_epoch_SAE', default=1e4, type=float, help='maximum training iterations SAE')
    parser.add_argument('--lr_SAE', default=1e-3, type=float, help='learning rate SAE')
    parser.add_argument('--lambda_r_SAE', default=1e-3, type=float, help='sparsity (regularization) weight SAE')
    parser.add_argument('--miles_SAE', default=[1e9], nargs='+', type=int, help='learning rate scheduler milestones SAE')
    parser.add_argument('--gamma_SAE', default=1e-1, type=float, help='learning rate milestone decay SAE')

    ## Structure-Preserving Neural Network
    # Net Parameters
    parser.add_argument('--hidden_vec_SPNN', default=5*[24], nargs='+', type=int, help='layer vector of hidden layers SPNN (excluding input and output layers)')
    parser.add_argument('--activation_SPNN', default='relu', type=str, help='activation function')
    parser.add_argument('--init_SPNN', default='kaiming_uniform', type=str, help='initialization SPNN')

    # Training Parameters  
    parser.add_argument('--max_epoch_SPNN', default=1e4, type=float, help='maximum training iterations')
    parser.add_argument('--lr_SPNN', default=1e-3, type=float, help='learning rate SPNN')
    parser.add_argument('--lambda_r_SPNN', default=1e-4, type=float, help='weight decay SPNN')
    parser.add_argument('--lambda_d_SPNN', default=5e2, type=float, help='data weight SPNN')
    parser.add_argument('--miles_SPNN', default=9e4, nargs='+', type=int, help='learning rate scheduler milestones SAE')
    parser.add_argument('--gamma_SPNN', default=1e-1, type=float, help='learning rate milestone decay SAE')

    args = parser.parse_args()

    main(args)
