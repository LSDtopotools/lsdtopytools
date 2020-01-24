# lsdtopytools

Repository for lsdtopytools. So far, it only contains instructions to install and wheels (python installers). Will evolve soon. Source code in the LSDTT_development repository.


## Installation through `conda`

Best installation is from `conda`, it has been tested for windows and different linuxes. Other OSes will need to build from source (explanation bellow).

First create a `conda` environment and activate it:

```
conda create lsdtopytools python=3.7

conda activate lsdtopytools
```

Then install the required dependencies:

```
conda install -c conda-forge gdal rasterio geopandas matplotlib numpy scipy pytables numba feather-format pandas pip pybind11 xtensor xtensor-python
```



