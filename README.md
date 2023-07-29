# DDPG Cart-Pole
This is my implementation of the DDPG algorithm in a classic, custom cart-pole environment.
Some pre-trained weights are to be found here, which can be used.
Important to note, there are some "flags" to be found in the "util" module, with which certain logs can be controled (also the ones added subsequently, if necessary), as well as rendering and recording.

## Required packages
This is not a comprehensive list of all the packages used in this project, however, the packages listed are the minimum required to run the project.
### TensorFlow GPU
    Use conda (miniconda3) to install TensorFlow and all its dependencies - as per the official instructions found on TensorFlow website.
### tf-agents
    Use pip to install tf-agents WITHIN the conda environment where TensorFlow is set up.
### gymnasium (optional: used for starting the example)
    Use pip to install gymnasium - can be a global install (not specific to the conda environment).
### tkinter
    Use pip to install tkinter - can be a global install (not specific to the conda environment).
### pygame
    Use pip to install pygame - can be a global install (not specific to the conda environment).
### numpy
    Use pip to install numpy - can be a global install (not specific to the conda environment).

## Instructions
The project is structured so that its components are grouped into modules.
The main module is the "ddpg", where the DDPG class and the main implementation of the algorithm in the custom Cart-Pole environment are stored, in "ddpg.py" and "cartpole.py" respectively.
Therefore, the recommended way of running the main cartpole.py is:

python3 -m ddpg.cartpole

*(In my case, for Debian based distro, the "python3" command is used - otherwise for RPM, Arch based distros use "python")*

Similarly, running any other file from the project follows the same scheme e.g. **python3 -m example.pendulum**.


