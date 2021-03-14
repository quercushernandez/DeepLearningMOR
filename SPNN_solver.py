"""solver.py"""

import os

import torch
import torch.optim as optim
import numpy as np

from model import StructurePreservingNN
from dataset import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent
from utilities.utils import print_mse


class SPNN_Solver(object):
    def __init__(self, args, x_trunc):
        self.x_trunc = x_trunc.detach()

        # Study Case
        self.sys_name = args.sys_name

        # Dataset Parameters
        self.dataset = load_dataset(args)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
     
        self.train_snaps, self.test_snaps = split_dataset(self.dim_t-1)

        # Training Parameters
        self.max_epoch = args.max_epoch_SPNN
        self.lambda_d = args.lambda_d_SPNN

        # Net Parameters
        _, self.dim_in = x_trunc.shape
        self.dim_out = int(self.dim_in * (self.dim_in + 2))
        self.SPNN = StructurePreservingNN(self.dim_in, self.dim_out, args.hidden_vec_SPNN, args.activation_SPNN).float() 

        if (args.train_SPNN == False):
            # Load pretrained nets
            load_name = 'SPNN_' + self.sys_name + '.pt'
            load_path = os.path.join(args.dset_dir, load_name)
            self.SPNN.load_state_dict(torch.load(load_path))
        else:
            self.SPNN.weight_init(args.init_SPNN)
        self.optim = optim.Adam(self.SPNN.parameters(), lr=args.lr_SPNN, weight_decay=args.lambda_r_SPNN)  
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles_SPNN, gamma=args.gamma_SPNN) 
        
        # Load/Save options
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots = args.save_plots

    # Train SPNN Algorithm
    def train(self):

        print("\n[SPNN Training Started]\n")

        # Training data
        x_gt, x1_gt = self.x_trunc[self.train_snaps,:], self.x_trunc[self.train_snaps + 1,:]

        epoch = 1        
        # Main training loop
        while (epoch <= self.max_epoch):
            # Net forward pass
            x1_net, deg_E, deg_S = self.SPNN(x_gt, self.dt)

            # Compute loss
            loss_data = ((x1_gt - x1_net)**2).mean()
            loss_degeneracy = (deg_E**2).mean() + (deg_S**2).mean()
            loss = self.lambda_d*loss_data + loss_degeneracy

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            # Print epoch error
            loss_data_mean = loss_data.item() / len(self.train_snaps)
            loss_degeneracy_mean = loss_degeneracy.item() / len(self.train_snaps)
            print ("Epoch [{}/{}], Data Loss: {:1.2e} (Train), Degeneracy Loss: {:1.2e} (Train)" 
                .format(epoch, int(self.max_epoch), loss_data_mean, loss_degeneracy_mean)) 

            epoch += 1

        print("\n[SPNN Training Finished]\n")

        # Save net
        file_name = 'SPNN_' + self.sys_name + '.pt'
        save_dir = os.path.join(self.output_dir, file_name)
        torch.save(self.SPNN.state_dict(), save_dir)


    # Test SPNN Algorithm
    def test(self, SAE, latent_idx):
        print("\n[SPNN Testing Started]\n")

        x_net = torch.zeros(self.x_trunc.shape)
        dEdt_net, dSdt_net = torch.zeros(self.dim_t), torch.zeros(self.dim_t)

        # Initial conditions z(t=0)
        x_net[0,:] = self.x_trunc[0,:]
        x = self.x_trunc[0,:]

        for snapshot in range(self.dim_t - 1):
            # Structure-Preserving Neural Network
            x1_net, _, _ = self.SPNN(x, self.dt)
            dEdt, dSdt = self.SPNN.get_thermodynamics(x)

            # Save results and Time update
            x_net[snapshot + 1,:] = x1_net
            dEdt_net[snapshot] = dEdt
            dSdt_net[snapshot] = dSdt
            x = x1_net
        
        # Detruncate
        x_spnn = torch.zeros([self.dim_t, SAE.dim_latent])
        x_spnn[:, latent_idx] = x_net

        # Decode latent vector
        z_spnn_norm = SAE.decode(x_spnn)
        z_spnn = SAE.denormalize(z_spnn_norm)

        # Load Ground Truth and Compute MSE
        z_gt = self.dataset.z
        print_mse(z_spnn, z_gt, self.sys_name)

        # Plot results
        if (self.save_plots):
            plot_name = 'SPNN Full Integration (Latent)'
            plot_latent(x_net, self.x_trunc, dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)
            plot_name = 'SPNN Full Integration'
            plot_results(z_spnn, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)

        print("\n[SPNN Testing Finished]\n")


if __name__ == '__main__':
    pass


