"""utils.py"""

import torch
import numpy as np

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_variables(z, sys_name):

    if (sys_name == 'viscoelastic'):
        n_nodes = 100

        # MSE Error
        q = z[:,n_nodes*0:n_nodes*1]
        v = z[:,n_nodes*1:n_nodes*2]
        e = z[:,n_nodes*2:n_nodes*3]
        tau = z[:,n_nodes*3:n_nodes*4]

        return q, v, e, tau

    elif (sys_name == 'rolling_tire'):
        n_nodes = 4140

        # Initialize vectors
        q = torch.zeros([3, z.size(0), n_nodes])
        v = torch.zeros([3, z.size(0), n_nodes])
        sigma = torch.zeros([6, z.size(0), n_nodes])

        # Position
        q[0] = z[:,n_nodes*0:n_nodes*1]
        q[1] = z[:,n_nodes*1:n_nodes*2]
        q[2] = z[:,n_nodes*2:n_nodes*3]
        # Velocity
        v[0] = z[:,n_nodes*3:n_nodes*4]
        v[1] = z[:,n_nodes*4:n_nodes*5]
        v[2] = z[:,n_nodes*5:n_nodes*6]
        # Stress
        sigma[0] = z[:,n_nodes*6:n_nodes*7]
        sigma[1] = z[:,n_nodes*7:n_nodes*8]
        sigma[2] = z[:,n_nodes*8:n_nodes*9]
        sigma[3] = z[:,n_nodes*9:n_nodes*10]
        sigma[4] = z[:,n_nodes*10:n_nodes*11]
        sigma[5] = z[:,n_nodes*11:n_nodes*12]

        return q, v, sigma


def print_mse(z_net, z_gt, sys_name):

    if (sys_name == 'viscoelastic'):
        # Get variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)

        # MSE Error
        q_mse = torch.mean((q_net - q_gt)**2)
        v_mse = torch.mean((v_net - v_gt)**2) 
        e_mse = torch.mean((e_net - e_gt)**2) 
        tau_mse = torch.mean((tau_net - tau_gt)**2)

        # Print MSE
        print('Position MSE = {:1.2e}'.format(q_mse))
        print('Velocity MSE = {:1.2e}'.format(v_mse))
        print('Energy MSE = {:1.2e}'.format(e_mse))
        print('Conformation Tensor MSE = {:1.2e}'.format(tau_mse))

    elif (sys_name == 'rolling_tire'):

        # Get Variables
        q_net, v_net, sigma_net = get_variables(z_net, sys_name)
        q_gt, v_gt, sigma_gt = get_variables(z_gt, sys_name)

        # Compute MSE
        q1_mse = torch.mean((q_gt[0] - q_net[0])**2)
        q2_mse = torch.mean((q_gt[1] - q_net[1])**2)
        q3_mse = torch.mean((q_gt[2] - q_net[2])**2)

        v1_mse = torch.mean((v_gt[0] - v_net[0])**2)
        v2_mse = torch.mean((v_gt[1] - v_net[1])**2)
        v3_mse = torch.mean((v_gt[2] - v_net[2])**2)

        s11_mse = torch.mean((sigma_gt[0] - sigma_net[0])**2)
        s22_mse = torch.mean((sigma_gt[1] - sigma_net[1])**2)
        s33_mse = torch.mean((sigma_gt[2] - sigma_net[2])**2)
        s12_mse = torch.mean((sigma_gt[3] - sigma_net[3])**2)
        s13_mse = torch.mean((sigma_gt[4] - sigma_net[4])**2)
        s23_mse = torch.mean((sigma_gt[5] - sigma_net[5])**2)

        # Print MSE
        print('Position "X" MSE = {:1.2e}'.format(q1_mse))
        print('Position "Y" MSE = {:1.2e}'.format(q2_mse))
        print('Position "Z" MSE = {:1.2e}\n'.format(q3_mse))

        print('Velocity "X" MSE = {:1.2e}'.format(v1_mse))
        print('Velocity "Y" MSE = {:1.2e}'.format(v2_mse))
        print('Velocity "Z" MSE = {:1.2e}\n'.format(v3_mse))

        print('Stress Tensor "XX" MSE = {:1.2e}'.format(s11_mse))
        print('Stress Tensor "YY" MSE = {:1.2e}'.format(s22_mse))
        print('Stress Tensor "ZZ" MSE = {:1.2e}'.format(s33_mse))
        print('Stress Tensor "XY" MSE = {:1.2e}'.format(s12_mse))
        print('Stress Tensor "XZ" MSE = {:1.2e}'.format(s13_mse))
        print('Stress Tensor "YZ" MSE = {:1.2e}'.format(s23_mse))


def truncate_latent(x):
    # Sort latent vector by L2 norm
    latent = np.sum(x.detach().numpy()**2, axis = 0)**0.5
    latent_val = np.sort(latent)
    latent_idx = np.argsort(latent)

    # Select the most energetic modes
    rel_importance = latent_val/np.max(latent_val)
    latent_dim_trunc = sum(1 for i in rel_importance if i > 0.1) 

    # Get the relevant latent variables (truncation)
    _, full_shape = x.shape
    latent_idx_trunc = latent_idx[full_shape-latent_dim_trunc:full_shape]

    return x[:,latent_idx_trunc], latent_idx_trunc

