# Corridor Navigation

The project is built as a package for convenience. Functions, classes and constant must be declared in /src as part of a module (e.g. Obstacle is a class part of the corridor_opt module, declared in the obstacle.py file). The actual scripts to be runned must be placed in the /scripts folder.

## Installation

## Clone this repository in recursive mode
```bash
git clone --recurse-submodules https://github.com/stiefen1/corridor-nav.git
```

## Conda
Setup your own environment with python:
```bash
conda init
conda env create -f env.yml 
conda activate corridor-nav-env
```

## PSO
Install PSO and its dependencies by running:
```bash
pip install -e submodules/pso
```

## SeaCharts
Install SeaCharts by running:
```bash
pip install -e submodules/seacharts_ais
```
## Install corridor-nav
Finally, install corridor-nav as an editable package for ease of development:
```bash
pip install -e .
```

## Test
Run ```examples/test_installation.py```