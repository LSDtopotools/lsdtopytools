# lsdtopytools

Repository for `lsdtopytools`. So far, it only contains instructions to install and wheels (python installers). Will evolve soon. Source code in the `LSDTT_development` repository. `lsdtt-xtensor-python` is a direct wrapper to the `c++` code and `lsdtopytools` a nicer `python` interface easier to use.

# Installation

## Installation through `conda` and `pip`

Best installation is from `conda`, it has been tested for windows and different linuxes. Other OSes will need to build from source (explanation bellow).

First create a `conda` environment and activate it:

```
conda create lsdtopytools python=3.7

conda activate lsdtopytools
```

Then install the required dependencies:

```
conda install -c conda-forge gdal rasterio geopandas matplotlib=3.1 numpy scipy pytables numba feather-format pandas pip pybind11 xtensor xtensor-python
```

We are now ready to install the wheels:

```
cd wheels/lsdtt-xtensor-python
pip install lsdtt_xtensor_python-YYY-cp37-cp37m-XXX.whl
```
where YYY is the latest version of lsdtt-xtensor-python, e.g. `0.0.3`, and XXX your architecture, e.g. `linux_x86_64` for linux 64 bits. Finally:

```
cd ../lsdtopytools
pip install lsdtopytools-XXX-py2.py3-none-any.whl
```

where XXX is the latest version available. And everything is installed!

## No compatible wheels for lsdtt-xtensor-python?

In that case you need to check several things:
- You are on iOS, I do not own a mac and cannot generate the wheel, you need to generate it from source.
- You are on a previously created conda environment and trying to install lsdtt-xtensor-python in it, check that you DO have `python=3.7.X`
- You have a `32 bits` OS/processor, You need to install from source (also it's been like 15 years all the computers are 64 bits)
- You are on an antiquated computer unable to get recent `libc`, `libc++` or `libstdc++`, there is nothing we can do unfortunately although I'd be surprised if it happens.

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

# Updating the code

So far I will post updated wheel in that repository, in the future I will create a more sofisticated and simpler installing method. To update the `lsdtt-xtensor-python`:

```
pip uninstall -y lsdtt-xtensor-python
cd wheels/lsdtt-xtensor-python
pip install XXX
```

where XXX is the newest wheel. To update `lsdtopytools`:

```
cd wheels/lsdtopytools
pip install --force-reinstall --no-deps XXX
```
where XXX is the newest wheel.

# Quick start

I will put tutorial and example scripts and jupyter notebooks in the future.

# Troubleshoots

- **[LINUX] GLIBC error**: I am working on it, but basically your linux is older than the one with which I installed it and does not have as recent glibc. Good news is that there is a solution, bad news is that It will take me a bit of time to implement and you need to install from source. I need to build `manylinux2010` wheels which involve using docker, which is not compatible with any of my hardware without accessing the BIOS and I cannot.

So far in early development stage, all of that code will evolve very rapidly and probably drastically. Contact `b.gailleton@sms.ed.ac.uk` for questions.


