# Proximal Accelerated Nerual ODEs  

## Installation

A list of all necessary packages is under `requirements.txt` . You may also install all required packages using the following command.

```bash
pip install -r .\requirements.txt
```

This also uses packages from [FFJORD](https://github.com/rtqichen/ffjord) and [GRAND](https://github.com/twitter-research/graph-neural-pde). To run each of the models, please download and manually install the packages from each of these sources. The following is the recommended manual installation.

- You may rename the [GRAND](https://github.com/twitter-research/graph-neural-pde) directory `\src` to `\grand` and place it as a subdirectory in the `\lib` directory (e.g. `\lib\grand\*.py`.

- For the [FFJORD](https://github.com/rtqichen/ffjord) library, you may directly integrate their `\lib` directory with proximal `\lib` directory.

## Main Methods

**All commands should be run with `proxNode` as the main level directory.**

The following commands offer full output reconstruction. To run them ensure that you have a bash shell with pthon.exe on the PATH.

**Full Data Reconstruction**
```bash
.\bin\examples.sh
```
For additional options type `--h`.

**Model Reconstruction**
To reconstruct the models please refer to each of the `jupyter-notebook` files in the `\models` subdirectory.

## Proximal Directory Structure

The directory `\lib\prox` holds all of the methods used by the proximal algorithm.

```bash
──prox
  │   adjoint.py
  │   misc.py
  │   odeprox.py
  │   optimizers.py
  │   proximals.py
  │   solvers.py
  └───
```
Recall that the proximal algorithm solves and ode by generating a optimization method based on the gradient of the proximal method. With this in mind the general structure is 

`odeprox.py` - Contains the main `odeint()` method similar to that of `torchdiffeq`

`solvers.py` - Contains several *outer loop* solver classes for various proximal type algorithms [multiscale, multistep, singlestep]

`optimizers.py` - Contains gradient based optimization classes with callable the `minimize()` *inner step* method

`proximals.py` - Contains proximal gradient methods with callable `step_grad()` (source of NFEs)

`adjoint.py` - Contains the adjoint process for backpropogation of a neural differential equation (*heavily borrowed from [torchdiffeq](https://github.com/rtqichen/torchdiffeq)*)

`misc.py` - Contains several miscelaneous subroutines used by the proximal algorithms.