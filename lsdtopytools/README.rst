============
lsdtopytools
============

High-level interface to *LSDTopoTools* in python. We aim to include as many tools as we can port in python, for the rest you can use the python interfaces to LSDTopoTools command line tools using *lsdttparamselector* and *lsdviztools*. This package is still in beta development but a certain amount of features are already available. This package offers (i) a cross platform use of LSDTopoTools, (ii) A scripting language making easy creation of python code using LSDTopoTools fast c++ routines, (iii) easy command-line tools

* Free software: GNU General Public License v3

Features
--------

* TODO

Release Notes
-------------
0.0.3.10 - Adding tools to deal with shapefiles and running windows, sorting some bugs and a segmentation fault in `lsdtt_xtensor-python` core linked to no data and points outside rasters.
0.0.3.9 - Minor bug fixes, Adding command line tools to preprocess the raster and adding useful functions in lsdtopytools: Extract raster to the extent of a basin, get uncertainties on concavity determination, raster saving options.
0.0.3.8 - Many bug fixes, adding comments and few random tools that are uploaded for testing.
0.0.3.6 - Bug fixes in the disorder metrics and addition of comments. It highlights the need to test units!
Pre-0.0.3.6 - Mostly random bug fixes and additions

Credits
-------

LSDTopoTools is a software package for analysing topography. Applications of these analyses span hydrology, geomorphology, soil science, ecology, and cognate fields. The software began within the Land Surface Dynamics group at the University of Edinburgh, and now has developers and users around the world. 

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
