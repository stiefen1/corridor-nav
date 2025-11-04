# Corridor Navigation

The project is built as a package for convenience. Functions, classes and constant must be declared in ```/src``` as part of a module (e.g. Obstacle is a class part of the corridor_opt module, declared in the ```obstacle.py``` file). The actual scripts to be runned must be placed in the ```/Scripts``` folder. The installatation procedure was only tested on Windows using miniconda environments. 

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

## Bathymetric Data
To run the examples provided in ```/Scripts```, bathymetric data must be available for Trøndelag and Møre og Romsdal counties (Norway). These FGDB files can be downloaded from the [Norwegian Mapping Authority](https://kartkatalog.geonorge.no/?organization=Norwegian%20Mapping%20Authority) using the ```EUREF89 UTM sone 33, 2d``` projection and ```FGDB 10.0``` format. Once download is complete, put the .gdb folders in the ```corridor-nav/data/db``` folder (to be created). You should have a folder structure similar to:

```
corridor-nav
│   README.md
│   .gitignore
|   .gitmodules
|   env.yml
|   setup.py    
│
└───data 
│   └───db
│       │   Trondelag_utm33.gdb
│       │   More_og_Romsdal_utm33.gdb
│   
└───Scripts
|    │   ...
...
```

Note that the actual name of the .gdb files will usually be slightly different, but it is not an issue at all and you do not need to change it.

## Test
Run ```Scripts/test_installation.py```

## Troubleshoot

***SeaCharts returns an empty map ?***

Delete the ```/data/shapefiles``` folder and run your script again. SeaCharts creates this folder to store the shapefiles, but it sometimes fail to update it automatically when the region changes. 

