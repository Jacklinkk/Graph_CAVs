## GRL_CAVs

GRL_CAVs is the source code for our [paper](https://arxiv.org/abs/2211.03005): **Graph Reinforcement 
Learning Application to 
Co-operative Decision-Making in Mixed Autonomy 
Traffic: Framework, Survey, and Challenges**.

GRL_CAVs is an all-round improvement and optimization source code based on our previously repository TorchGRL. GRL_CAVs is a modular simulation framework that integrates different GRL algorithms, flow interface and SUMO simulation platform to realize the simulation of multi-agents decision-making algorithms for connected and autonomous vehicles (CAVs) in mixed autonomy traffic. You can design your own traffic scenarios, adjust the implemented GRL algorithm or do your mprovements for a particular module according to your needs.

-------------------------------------
* [Preparation](#preparation)
* [Installation](#installation)
* [Instruction](#instruction)
* [Tutorial](#tutorial)


## Preparation
Before starting to carry out some relevant 
works on our framework, 
some preparations are required to be done.

### Hardware
Our framework is developed based on a laptop, and the specific configuration is as follows:
- Operating system: Ubuntu 20.04
- RAM: 32 GB
- CPU: Intel (R) Core (TM) i9-10980HK CPU @ 2.40GHz
- GPU: RTX 2070

We suggest that our program should be reproduced under the Ubuntu 20.04 operating system, and we strongly recommend using GPU for training.

### Development Environment
Before compiling the code of our framework,
you need to install the following 
development environment:
- Ubuntu 20.04 with latest GPU driver
- Pycharm
- Anaconda
- CUDA 11.3
- cudnn-11.3, 8.2.0.53


## Installation
Please download our GRL framework 
repository first through git or directly download the 
compressed files:
```
git clone https://github.com/Jacklinkk/GRL_CAVs.git
```

Then enter the root directory of GRL_CAVs:
```
cd GRL_CAVs
```

and **please be sure to run the 
below commands from /path/to/GRL_CAVs.**

### Installation of FLOW
The [FLOW](https://flow-project.github.io/usingFlow.html)
library will be firstly installed.

Firstly, enter the flow directory:
```
cd flow
```

Then, create a conda environment from flow library.

The name of the conda environment is defined in "environment.yml"
in the flow folder, you can change the name accordingly. Here, we 
choose **GraphRL** as the name of our environment.
```
conda env create -f environment.yml
```

Activate conda environment:
```
conda activate GraphRL
```

Install flow from source code:
```
python setup.py develop
```

### Installation of SUMO
[SUMO](https://www.eclipse.org/sumo/) simulation platform will be installed. 
**Please make sure to run the below commands 
in the "GraphRL" virtual environment.**

Install via pip, here we choose the 1.12.0 version of SUMO.
```
pip install eclipse-sumo==1.12.0
```

Setting in Pycharm:

In order to adopt SUMO correctly, 
you need to define the environment variable 
of SUMO_HOME in Pycharm. 
The specific directory is:
```
/home/…/.conda/envs/GRL_CAVs/lib/python3.7/site-packages/sumo
```
You can define the pycharm environment variable through 
the following steps.
Click "Run" in the menu bar, then click "Edit Configurations";
find the installation path of SUMO, and add the path in "Environment->Environment variables".

Setting in Ubuntu:

At first, run:
```
gedit ~/.bashrc
```

then copy the path name 
of SUMO_HOME to “~/.bashrc”:
```
export SUMO_HOME=“/home/…/.conda/envs/GraphRL/lib/python3.7/site-packages/sumo”
```

Finally, run:
```
source ~/.bashrc
```

### Installation of Pytorch and related libraries
**Please make sure to run the below commands 
in the "GraphRL" virtual environment.**

Installation of [Pytorch](https://pytorch.org/):

We use Pytorch version 1.11.0 for development
under a specific version of CUDA and cudnn.
```
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Installation of [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/#):

Pytorch geometric is a graph neural network (GNN) library upon Pytorch.
You need to install the corresponding version of pytorch geometric according to the pytorch version you have installed.
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## Instruction
### flow folder

The flow folder is the root directory of 
the library after the 
FLOW library is installed through 
source code, including interface-related 
programs between the developed GRL algorithms
and SUMO platform.

### Flow_Test folder 

The Flow_Test folder includes the related programs
of the test environment configuration.
This folder includes test code for ring network, highway network, etc.. 
If this programs runs successfully, 
the environment configuration of the source code successful.

### GRL_Envs folder 
The programs in the GRL_Envs folder are used to 
define the environment configuration for mixed 
autonomy traffic constructed in our modular framework. Here we have constructed 
two mixed traffic scenarios, highway ramping scenario
and Figure-Eight scenario.

A scenario is constructed by several python files described
as follows:
- XX_base.py: The base reinforcement learning environment for
the designed traffic scenario. It provides the interface for interacting 
with various aspects of the traffic simulation.
- XX_network.py: Contains road network configuration and 
generates xml files.
- XX_specific.py: The core code of the mixed autonomy traffic. 
It is the specific definition of the mixed autonomy traffic. The graph representation, reward function
and other relative function are all defined in this python file.
- XX_router.py: Defines the global path information of the vehicles in the environment. 
If the optional paths of the vehicles are unique, 
this file is not needed to be defined (e.g. Figure-Eight scenario).

You can construct your own traffic scenarios in GRL_Envs folder 
by referring to the above file structure.

### GRL_Experiment folder 
The GRL_Experiment folder contains the programs for simulation
configuration. Relative parameters for simulation are 
defined in each python file. 

This folder contains two sub-folders: Exp_FigureEight folder and
Exp_HighwayRamps folder. Each folder contains simulation files for different 
GRL algorithms for this traffic scenario. 
If you design a new scenario, 
or a new GRL algorithm, 
you need to create a new python program 
in this folder for simulation configuration.

It should be noted that in each python file, 
the model save path and load path need to be set 
and create the corresponding folders in advance.

### GRL_Library folder 
The GRL folder is the core of the modular framework,
which includes different GRL algorithms. It consists of 
two sub-folders: agent folder and common folder.
#### ---agent folder---
The agent folder contains several GRL agents for solving
multi-agent decision-making problem in mixed autonomy traffic.
We have divided the GRL algorithms into continuous and discrete algorithms, 
depending on the type of action space that 
the developed GRL algorithm can handle. 

Each folder contains programs 
for several GRL algorithms with detailed parameter descriptions and comments.
You can find detailed descriptions in the python files of each 
GRL algorithm for easy code reproduction
and secondary development.
#### ---common folder---
The common folder contains several generic programs of
different GRL algorithms. The python files are described
as follows:
- explorer_continuous.py: Programs of some exploration strategies
for continuous action spaces.
- explorer_discrete.py: Programs of some exploration strategies
for discrete action spaces.
- replay_buffer.py: Programs of replay buffer for reinforcement learning.
- prioritized_replay_buffer.py: Programs of prioritized replay buffer.

You can design your own GRL algorithms in this folder as required.

### GRL_Net folder 
The GRLNet folder contains the GRL neural network 
built in the pytorch environment. The networks are divided
into continuous network and discrete network according to the
categories of action space. The sub-folders are illustrated
as follows:
- Model_Continuous: Network models for continuous action spaces.
- Model_Discrete: Network models for discrete action space.
- NoisyNet: Noisy network that can add noise to the above network model.

You can modify the source code as needed 
or add your own neural network.

### GRL_utils folder 
The GRL_utils folder contains basic functions 
such as model training and testing, 
data storage, and curve drawing. The files are illustrated
as follows:
- Data_Plot_Train.py: This file is used to plot the training
curve of each implemented GRL algorithms.
- Data_Process_Train_FE.py: This file is used to further
process the training data of the Figure-Eight scenario.
- Data_Process_Train_HR.py: This file is used to further
process the training data of the highway ramping scenario.
- Train_and_Test_XX.py: Training and testing programs for different
GRL algorithms.

Before using these functions, please set the path for 
saving and reading the relevant data and curves. In addition,
You need to select the corresponding "Train_and_Test" program 
according to the GRL algorithm to be verified.

### GRL_Simulation folder 
The GRL_Simulation folder contains the 
main program to run the simulation of different traffic scenarios.

## Tutorial
You can simply run python files in "/GRL_Simulation/main" in Pycharm to 
simulate the GRL algorithm, and observe the 
simulation process in SUMO platform. You can
generate training plot such as reward curve.

### Verification of other algorithms
If you want to verify other algorithms, 
you can develop the source code as needed 
under the GRL_Library folder. Don't 
forget to change the imported python script 
in "main.py", and define your own experiment file in 
the GRL_Experiment folder. In addition, you can
also construct your own network in GRL_Net folder.

### Verification of other traffic scenario
If you want to verify other traffic scenario, 
you can define a new scenario in GRL_Envs folder. 
You can refer to the documentation of 
SUMO and FLOW for more details.

## Citation
To cite our publications, please cite our paper currently on arxiv, 
the library on which Graph_CAVs is based:
```
@article{liu2022graph,
  title={Graph Reinforcement Learning Application to Co-operative Decision-Making in Mixed Autonomy Traffic: Framework, Survey, and Challenges},
  author={Liu, Qi and Li, Xueyuan and Li, Zirui and Wu, Jingda and Du, Guodong and Gao, Xin and Yang, Fan and Yuan, Shihua},
  journal={arXiv preprint arXiv:2211.03005},
  year={2022}
}
```
