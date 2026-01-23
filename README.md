# Roboslice
## Installation
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f environment.yml
```

python=3.10

pytorch later than 2.6

cuda later than 12.4

you can use conda as well:
```console
$ conda env create -f environment.yml
```
Activate conda environment and login to [wandb](https://wandb.ai/).
```console
$ conda activate roboninja
$ wandb login
```


## Simulation and Trajectory Optimization

### Quick Example
[simulation_example.ipynb](simulation_example.ipynb) provides a quick example of the simulation. It first create a scene and render an image. It then runs a forward pass a backward pass using the initial action trajectory. Finally, it executes an optimized action trajectory.

If you get an error related to rendering, here are some potential solutions:
- make sure [vulkan](https://www.vulkan.org/) is installed
- `TI_VISIBLE_DEVICE` is not correctly set in [roboninja/env/tc_env.py](roboninja/env/tc_env.py) (L17). The reason is that vukan device index is not alighed with cuda device index, and that's the reason I have the function called `get_vulkan_offset()`. Change this function implementation based on your setup.
### Trajectory Optimization via Differentiable Simulation
```console
$ python rl/policy.py 
```

we aims to use the new cutloss that force the knife to reach the ground by shortest distance