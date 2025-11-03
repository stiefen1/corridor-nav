# Corridor Navigation

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
```bash
pip install -e .
```

## Test
Run ```examples/test_installation.py```