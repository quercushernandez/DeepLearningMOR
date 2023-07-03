"""solver.py"""

import os

import torch
import torch.optim as optim
import numpy as np

from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire
from utilities.utils import print_mse, truncate_latent


class SAE_Solver(object):
    def __init__(self, args):
        # Study Case
        self.sys_name = args.sys_name

        # Dataset Parameters
        self.dset_dir = args.dset_dir
        self.dataset = load_dataset(args)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t

        self.train_snaps, self.test_snaps = split_dataset(self.dim_t)

        # Training Parameters
        self.max_epoch = args.max_epoch_SAE
        self.lambda_r = args.lambda_r_SAE

        # Net Parameters
        if self.sys_name == 'viscoelastic':
            self.SAE = SparseAutoEncoder(args.layer_vec_SAE, args.activation_SAE).float() 
        elif self.sys_name == 'rolling_tire':
            self.SAE = StackedSparseAutoEncoder(args.layer_vec_SAE_q, args.layer_vec_SAE_v, args.layer_vec_SAE_sigma, args.activation_SAE).float()
        self.optim = optim.Adam(self.SAE.parameters(), lr=args.lr_SAE, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles_SAE, gamma=args.gamma_SAE)

        # Load/Save options
        if (args.train_SAE == False):
            # Load pretrained nets
            load_name = 'SAE_' + self.sys_name + '.pt'
            load_path = os.path.join(self.dset_dir, load_name)
            self.SAE.load_state_dict(torch.load(load_path))
            
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots = args.save_plots
        

    # Train SAE Algorithm
    def train(self):

        print("\n[SAE Training Started]\n")

        # Training data
        z_gt = self.dataset.z[self.train_snaps,:]
        z_gt_norm = self.SAE.normalize(z_gt)

        epoch = 1      
        # Main training loop
        while (epoch <= self.max_epoch):
            # Forward pass            
            z_sae_norm, x = self.SAE(z_gt_norm)

            # Loss function
            loss_reconst = torch.mean((z_sae_norm - z_gt_norm)**2)
            loss_sparsity = torch.mean(torch.abs(x))
            loss = loss_reconst + self.lambda_r * loss_sparsity

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            loss_reconst_mean = loss_reconst.item()/len(self.train_snaps)
            loss_sparsity_mean = loss_sparsity.item()/len(self.train_snaps)
            print ("Epoch [{}/{}], Reconst Loss: {:1.2e} (Train), Sparsity Loss: {:1.2e} (Train)" 
                .format(epoch, int(self.max_epoch), loss_reconst_mean, loss_sparsity_mean)) 

            epoch += 1

        print("\n[SAE Training Finished]\n")
        print("[Train Set Evaluation]\n")

        # Denormalize
        z_sae = self.SAE.denormalize(z_sae_norm)

        # Compute MSE
        print_mse(z_sae, z_gt, self.sys_name)

        # Save net
        file_name = 'SAE_' + self.sys_name + '.pt'
        save_dir = os.path.join(self.output_dir, file_name)
        torch.save(self.SAE.state_dict(), save_dir)
        

    # Test SAE Algorithm
    def test(self):
        print("\n[SAE Testing Started]\n")

        # Load data
        z_gt = self.dataset.z
        z_gt_norm = self.SAE.normalize(z_gt)

        # Forward pass
        z_sae_norm, _ = self.SAE(z_gt_norm)
        z_sae = self.SAE.denormalize(z_sae_norm)

        # Compute MSE
        print_mse(z_sae, z_gt, self.sys_name)

        if (self.save_plots):
            plot_name = 'SAE Reduction Only'
            plot_results(z_sae, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)

        print("\n[SAE Testing Finished]\n")


    # Latent dimensionality detection
    def detect_dimensionality(self):
        # Load data
        z = self.dataset.z
        z_norm = self.SAE.normalize(z)
        # Forward pass
        _, x = self.SAE(z_norm)

        if self.sys_name == 'viscoelastic':
            # Detect latent dimensionality 
            x_trunc, latent_idx = truncate_latent(x)
            print('Latent Dimensionality: {}'.format(len(latent_idx)))

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[Viscoelastic] SAE Latent Variables'
                plot_latent_visco(x, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == 'rolling_tire':
            x_q, x_v, x_sigma = self.SAE.split_latent(x)

            # Detect latent dimensionality
            x_q_trunc, latent_idx_q = truncate_latent(x_q)
            x_v_trunc, latent_idx_v = truncate_latent(x_v)
            x_sigma_trunc, latent_idx_sigma = truncate_latent(x_sigma)
            print('Latent Dimensionality:')
            print('  Position: {}'.format(len(latent_idx_q)))
            print('  Velocity: {}'.format(len(latent_idx_v)))
            print('  Stress Tensor: {}'.format(len(latent_idx_sigma)))

            x_trunc = torch.cat((x_q_trunc, x_v_trunc, x_sigma_trunc), 1)
            latent_idx_v = latent_idx_v + self.SAE.dim_latent_q
            latent_idx_sigma = latent_idx_sigma + self.SAE.dim_latent_q + self.SAE.dim_latent_v
            latent_idx = list(latent_idx_q) + list(latent_idx_v) + list(latent_idx_sigma)

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[Rolling Tire] SAE Latent Variables'
                plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        return x_trunc, latent_idx


if __name__ == '__main__':
    pass


