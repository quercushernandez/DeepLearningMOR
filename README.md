
<div align="center">  
  
# Deep learning of thermodynamics-aware reduced-order models from data

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/quercus-hernandez/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2007.03758.pdf)
[![CMAME](https://img.shields.io/badge/CMAME-2021-green)](https://arxiv.org/abs/2007.03758)

</div>

## Abstract

We present an algorithm to learn the relevant latent variables of a large-scale discretized physical system and predict its time evolution using thermodynamically-consistent deep neural networks. Our method relies on sparse autoencoders, which reduce the dimensionality of the full order model to a set of sparse latent variables with no prior knowledge of the coded space dimensionality. Then, a second neural network is trained to learn the metriplectic structure of those reduced physical variables and predict its time evolution with a so-called structure-preserving neural network. This data-based integrator is guaranteed to conserve the total energy of the system and the entropy inequality, and can be applied to both conservative and dissipative systems. The integrated paths can then be decoded to the original full-dimensional manifold and be compared to the ground truth solution. This method is tested with two examples applied to fluid and solid mechanics.

For more information, please refer to the following:

- Hernández, Quercus and Badías, Alberto and González, David and Chinesta, Francisco and Cueto, Elías. "[Deep learning of thermodynamics-aware reduced-order models from data](https://arxiv.org/abs/2007.03758)." Computer Methods in Applied Mechanics and Engineering (2021).

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/quercushernandez/DeepLearningMOR.git
cd DeepLearningMOR
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.7_.

```bash
# install dependencies
pip install scipy numpy matplotlib pytorch
 ```

## How to run the code  

### Test pretrained nets

The results of the paper (Viscolastic Fluid and Hyperelastic Rolling Tire) can be reproduced with the following executables, found in the `executables/` folder. The `data/` folder includes the database and the pretrained parameters of the networks.

```bash
sh executables/run_viscolastic_test.sh
sh executables/run_rolling_tire_test.sh
```

The resulting time evolution of the latent variables (reduced with the SAE) and the thermodynamics-aware integrator (SPNN) are plotted and saved in .png format in a generated `outputs/` folder.

|           Viscolastic Fluid               |             Rolling Tire              | 
|-------------------------------------------| --------------------------------------|
|<div align="center"> <img src="/data/viscoelastic.png" width="500"></div>|<div align="center"> <img src="/data/tire.png" width="500"></div>|

### Train a custom net

You can also run your own experiments for the implemented datasets by setting parameters manually. Several training examples can be found in the `executables/` folder. Both the SAE and the SPNN can be trained individually, and the trained parameters and output plots are saved in the `outputs/` folder. 

```bash
e.g.
python main.py --sys_name viscoelastic --train_SAE True --lr_SAE 1e-5 --lambda_r 1e-3 --train_SPNN True --lr_SPNN 1e-5 ...
```

General Arguments:

|     Argument              |             Description                           |           Options                |
|---------------------------| ------------------------------------------------- |----------------------------------|
| `--sys_name`              | Study case                                        | `viscolastic`, `rolling_tire`    |
| `--train_SAE`             | Train or test mode for SAE                        | `True`, `False`                  |
| `--train_SPNN`            | Train or test mode for SPNN                       | `True`, `False`                  |
| `--dset_dir`              | Dataset and pretrained nets directory             | Default: `data`                  |
| `--train_percent`         | Train porcentage of the full database             | Default: `0.8`                   |
| `--output_dir`            | Output data directory                             | Default: `output`                |
| `--save_plots`            | Save plots of latent and state variables          | `True`, `False`                  |

Sparse Autoencoder (SAE) Arguments:

|     Argument              |             Description                           |           Options                   |
|---------------------------| ------------------------------------------------- |-------------------------------------|
| `--layer_vec_SAE`         | Layer vector (viscoelastic)                       | Default: `400, 160, 160, 10`        |
| `--layer_vec_q_SAE`       | Layer vector (rolling_tire, position)             | Default: `12420, 40, 40, 10`        |
| `--layer_vec_v_SAE`       | Layer vector (rolling_tire, velocity)             | Default: `12420, 40, 40, 10`        |
| `--layer_vec_sigma_SAE`   | Layer vector (rolling_tire, stress tensor)        | Default: `24840, 80, 80, 20`        |
| `--activation_SAE`        | Activation functions of the hidden layers         | Default: `2e3`                      |
| `--max_epoch_SAE`         | Maximum number of training epochs                 | Default: `10e3`                     |
| `--lr_SAE`                | Learning rate                                     | Default: `1e-4`                     |
| `--weight_decay_SAE`      | Weight decay regularizer                          | Default: `0`                        |
| `--lambda_r`              | Sparse regularizer                                | Default: `1e-3`                     |

Structure-Preserving Neural Network (SPNN) Arguments:

|     Argument              |             Description                           |                Options                 |
|---------------------------| ------------------------------------------------- |----------------------------------------|
| `--hidden_vec_SPNN`       | Hidden layers vector                              | Default: `24, 24, 24, 24`              |
| `--activation_SPNN`       | Activation functions of the hidden layers         | `linear`, `sigmoid`, `relu`, `tanh`    |
| `--init_SPNN`             | Net initialization method                         | `kaiming_normal`, `xavier_normal`      |
| `--max_epoch_SPNN`        | Maximum number of training epochs                 | Default: `1e4`                         |
| `--lr_SPNN`               | Learning rate                                     | Default: `1e-3`                        |
| `--lambda_d`              | Data weight                                       | Default: `5e2`                         |
| `--miles`                 | Learning rate scheduler milestones                | Default: `9e4`                         |
| `--gamma`                 | Learning rate scheduler decay                     | Default: `1e-1`                        |

## Citation

If you found this code useful please cite our work as:

```
@article{hernandez2020deep,
  title={Deep learning of thermodynamics-aware reduced-order models from data},
  author={Hern{\'a}ndez, Quercus and Bad{\'\i}as, Alberto and Gonz{\'a}lez, David and Chinesta, Francisco and Cueto, El{\'\i}as},
  journal={arXiv preprint arXiv:2007.03758},
  year={2020}
}
```
