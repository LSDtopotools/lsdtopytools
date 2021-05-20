# lsdtopytools

[![DOI](https://zenodo.org/badge/235968220.svg)](https://zenodo.org/badge/latestdoi/235968220)

Repository for `lsdtopytools`, a python wrapper on LSDTopoTools. It binds the c++ code with `lsdtt-xtensor` and expose it to `python` via `numpy`. It allows us to take advantage of both world: `c++` speed and python flexibility. `lsdtopytools` suite can be done using Docker or Anaconda.

This repository host the static version of the code available for installation, it is updated at each release (usually when we are publishing a new manuscript utilising new features). Tutorial and example notebooks can be found in [this dedicated repository](https://github.com/LSDtopotools/lsdtt_notebooks/tree/master/lsdtopytools).

# Features

- Raster i/o and basic operations
- Preprocessing depressions (carving, filling, nodata, ...)
- River extraction (drainage area threshold)
- ksn from segmented profiles (Mudd et al. 2014)
- concavity index (Mudd et al., 2018, Gailleton et al., 2021 (preprint))
- polyfit metrics
- ...

# Installation using docker

See https://github.com/LSDtopotools/lsdtt_pytools_docker for intruction on how to install the Docker container.

# Installation using `conda`

## Installation through `conda` and `pip`

You need first to have a valid installation of `anaconda` on your computer. I really reccomend `miniforge` [see readme here](https://github.com/conda-forge/miniforge) which guarentee a full open-source and open-license installation.

Installation from `conda` has been tested for windows10, MacOS and different linuxes (ubuntu, Debian, Redhat, WSL). Other OSes will need to build from source (explanation bellow).

First create a `conda` environment `with python <= 3.8` (3.9 will come shortly) and activate it:

```
conda create -n lsdtopytools python=3.8

conda activate lsdtopytools
```

**RECOMENDED STEP IF YOU HAVE ANOTHER DISTRIBUTION THAN MINIFORGE**: you need to fix an annoying tendency of gdal to rely on messed up dependencies:
```
conda config --prepend channels defaults
conda config --prepend channels conda-forge
conda config --set channel_priority strict
```

Then, you can install by simply running:

```
conda install lsdtopytools
```

## Something went wrong

In that case you need to check several things:
- You are on a previously created conda environment and trying to install lsdtt-xtensor-python in it, check that you DO have `python=3.7.X`
- You have a `32 bits` OS/processor, You need to install from source (also it's been like 15 years all the computers are 64 bits, this is the reason we do not guarentee compatibility)
- You are on an antiquated computer unable to get recent `libc`, `libc++` or `libstdc++`, there is nothing we can do unfortunately although I'd be surprised if it happens. We need `c++14` for `lsdtt-xtensor` and that is a hard-pass unfortunately.

## Installation from source

You need to get the code (so far from the development repository under `src/driver_analysis_BG/)`, and you need a `c++14`-ready compiler (tested with `gcc 5+` and `MSVS` so far, should work with `clang`, theoretically `icc` as well). Make sure all the dependencies (see above) are installed, especially `xtensor` and `xtensor-python`. There are different ways to install a python package from source. From far the safest is to generate a wheel, i.e. a python binary.

For `lsdtt-xtensor-python`:

```
cd lsdtt-xtensor-python
python setup.py bdist_wheel
pip install dist/XXX.whl
```

where XXX is the name of the created wheel file.
Then for `lsdtopytools`:

```
cd ../lsdtopytools
pip install .
```
Done.

# Troubleshoots

- **[LINUX] GLIBC error**: I am working on it, but basically your linux is older than the one with which I installed it and does not have as recent glibc. Good news is that there is a solution, bad news is that It will take me a bit of time to implement and you need to install from source. I need to build `manylinux2010` wheels which involve using docker, which is not compatible with any of my hardware without accessing the BIOS and I cannot.

So far in early development stage, all of that code will evolve very rapidly and probably drastically. Contact `boris.gailleton@gfz-potsdam.de` for questions.


