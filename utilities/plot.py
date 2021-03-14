"""utils.py"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from utilities.utils import get_variables
import os


def plot_latent_visco(x, dt, plot_name, output_dir):
    plt.clf()
    N = x.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    fig, axes = plt.subplots(1,1, figsize=(5, 5))
    fig.suptitle(plot_name)

    axes.plot(t_vec, x.detach())
    axes.set_ylabel('$x$ [-]')
    axes.set_xlabel('$t$ [s]')
    axes.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()


def plot_latent_tire(x_q, x_v, x_sigma, dt, plot_name, output_dir):
    plt.clf()
    N = x_q.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    ax1, ax2, ax3 = axes.flatten()
    fig.suptitle(plot_name)

    ax1.plot(t_vec, x_q.detach())
    ax1.set_ylabel('$x_q$ [-]')
    ax1.set_xlabel('$t$ [s]')
    ax1.grid()

    ax2.plot(t_vec, x_v.detach())
    ax2.set_ylabel('$x_v$ [-]')
    ax2.set_xlabel('$t$ [s]')
    ax2.grid()

    ax3.plot(t_vec, x_sigma.detach())
    ax3.set_ylabel('$x_\sigma$ [-]')
    ax3.set_xlabel('$t$ [s]')
    ax3.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()


def plot_latent(x_spnn, x_trunc, dEdt, dSdt, dt, plot_name, output_dir, sys_name):
    plt.clf()
    N = x_spnn.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    if (sys_name == 'viscoelastic'): plot_name = '[Viscoelastic] ' + plot_name
    elif (sys_name == 'rolling_tire'): plot_name = '[Rolling Tire] ' + plot_name

    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    ax1, ax2 = axes.flatten()
    fig.suptitle(plot_name)
 
    ax1.plot(t_vec, x_spnn.detach(),'b')  
    ax1.plot(t_vec, x_trunc.detach(),'k--')
    l1, = ax1.plot([],[],'k--')
    l2, = ax1.plot([],[],'b')
    ax1.legend((l1, l2), ('GT','Net'))
    ax1.set_ylabel('$x_{latent}$ [-]')
    ax1.set_xlabel('$t$ [s]')    
    ax1.set_ylim([1.1*torch.min(x_trunc).item(),1.1*torch.max(x_trunc).item()])
    ax1.grid()

    ax2.plot(t_vec, dEdt.detach(),'r')  
    ax2.plot(t_vec, dSdt.detach(),'b')
    l1, = ax2.plot([],[],'r')
    l2, = ax2.plot([],[],'b')
    ax2.legend((l1, l2), ('dEdt','dSdt'))
    ax2.set_ylabel('$dEdt, dSdt$ [-]')
    ax2.set_xlabel('$t$ [s]')    
    ax2.grid()  

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()


def plot_results(z_net, z_gt, dt, name, output_dir, sys_name):
    plt.clf()
    N = z_gt.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    if (sys_name == 'viscoelastic'):

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        nodes = [20-1, 40-1, 60-1, 80-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[Viscoelastic] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach(),'b')
        ax2.plot(t_vec, v_gt[:,nodes].detach(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$v$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach(),'b')
        ax3.plot(t_vec, e_gt[:,nodes].detach(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$e$ [-]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
       
        ax4.plot(t_vec, tau_net[:,nodes].detach(),'b')
        ax4.plot(t_vec, tau_gt[:,nodes].detach(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$\tau$ [-]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)

    elif (sys_name == 'rolling_tire'):

        # Only 4 Nodes to plot
        nodes = [1000-1, 2000-1, 3000-1, 4000-1]

        # Get Variables
        q_net, v_net, sigma_net = get_variables(z_net, sys_name)
        q_gt, v_gt, sigma_gt = get_variables(z_gt, sys_name)

        # Position and Velocity Figure
        fig, axes = plt.subplots(2,3, figsize=(20, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        plot_name = '[Rolling Tire] ' + name + ' (Position and Velocity)'
        fig.suptitle(plot_name)
     
        ax1.plot(t_vec, q_net[0,:,nodes].detach(),'b')
        ax1.plot(t_vec, q_gt[0,:,nodes].detach(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q_1$ [m]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()

        ax2.plot(t_vec, q_net[1,:,nodes].detach(),'b')
        ax2.plot(t_vec, q_gt[1,:,nodes].detach(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$q_2$ [m]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()

        ax3.plot(t_vec, q_net[2,:,nodes].detach(),'b')
        ax3.plot(t_vec, q_gt[2,:,nodes].detach(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$q_3$ [m]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()

        ax4.plot(t_vec, v_net[0,:,nodes].detach(),'b')
        ax4.plot(t_vec, v_gt[0,:,nodes].detach(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$v_1$ [m/s]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        ax5.plot(t_vec, v_net[1,:,nodes].detach(),'b')
        ax5.plot(t_vec, v_gt[1,:,nodes].detach(),'k--')
        l1, = ax5.plot([],[],'k--')
        l2, = ax5.plot([],[],'b')
        ax5.legend((l1, l2), ('GT','Net'))
        ax5.set_ylabel('$v_2$ [m/s]')
        ax5.set_xlabel('$t$ [s]')
        ax5.grid()

        ax6.plot(t_vec, v_net[2,:,nodes].detach(),'b')
        ax6.plot(t_vec, v_gt[2,:,nodes].detach(),'k--')
        l1, = ax6.plot([],[],'k--')
        l2, = ax6.plot([],[],'b')
        ax6.legend((l1, l2), ('GT','Net'))
        ax6.set_ylabel('$v_3$ [m/s]')
        ax6.set_xlabel('$t$ [s]')
        ax6.grid()

        save_dir = os.path.join(output_dir, plot_name)
        plt.savefig(save_dir)
        plt.clf()

        # Stress Tensor Figure
        fig, axes = plt.subplots(2,3, figsize=(20, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        plot_name = '[Rolling Tire] ' + name + ' (Stress Tensor)'
        fig.suptitle(plot_name)
        
        ax1.plot(t_vec, sigma_net[0,:,nodes].detach(),'b')
        ax1.plot(t_vec, sigma_gt[0,:,nodes].detach(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$\\sigma_{11}$ [MPa]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
        
        ax2.plot(t_vec, sigma_net[1,:,nodes].detach(),'b')
        ax2.plot(t_vec, sigma_gt[1,:,nodes].detach(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$\\sigma_{22}$ [MPa]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
       
        ax3.plot(t_vec, sigma_net[2,:,nodes].detach(),'b')
        ax3.plot(t_vec, sigma_gt[2,:,nodes].detach(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$\\sigma_{33}$ [MPa]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
        
        ax4.plot(t_vec, sigma_net[3,:,nodes].detach(),'b')
        ax4.plot(t_vec, sigma_gt[3,:,nodes].detach(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$\\sigma_{12}$ [MPa]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()
        
        ax5.plot(t_vec, sigma_net[4,:,nodes].detach(),'b')
        ax5.plot(t_vec, sigma_gt[4,:,nodes].detach(),'k--')
        l1, = ax5.plot([],[],'k--')
        l2, = ax5.plot([],[],'b')
        ax5.legend((l1, l2), ('GT','Net'))
        ax5.set_ylabel('$\\sigma_{13}$ [MPa]')
        ax5.set_xlabel('$t$ [s]')
        ax5.grid()
        
        ax6.plot(t_vec, sigma_net[5,:,nodes].detach(),'b')
        ax6.plot(t_vec, sigma_gt[5,:,nodes].detach(),'k--')
        l1, = ax6.plot([],[],'k--')
        l2, = ax6.plot([],[],'b')
        ax6.legend((l1, l2), ('GT','Net'))
        ax6.set_ylabel('$\\sigma_{23}$ [MPa]')
        ax6.set_xlabel('$t$ [s]')
        ax6.grid()

        save_dir = os.path.join(output_dir, plot_name)

    plt.savefig(save_dir)
    plt.clf()
