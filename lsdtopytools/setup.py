#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = ['Click>=6.0','numpy','pandas','rasterio',"scipy", 'tables', 'geopandas','matplotlib' ,'gdal','numba','lsdtt-xtensor-python']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Boris Gailleton",
    author_email='b.gailleton@sms.ed.ac.uk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English'
    ],
    description="""
    High-level interface to LSDTopoTools in python. Include as much tools as we can port in python, for the rest see our docker solutions. This package is using lsdtt-xtensor-python as
    """,
    entry_points={
        'console_scripts': [
            'lsdtt-concavity-tools=scripts_for_lsdtopytools.lsdtt_concavity_tools:main_concavity',
            'lsdtt-hypsometry=scripts_for_lsdtopytools.lsdtt_hypsometry:hypsometry_tools_general',
            'lsdtt-depressions=scripts_for_lsdtopytools.lsdtt_basic_operations:PreProcess',
            'lsdtt-polyfits=scripts_for_lsdtopytools.lsdtt_basic_operations:Polyfit_Metrics',
            'lsdtt-chi-tools=scripts_for_lsdtopytools.lsdtt_chi_ksn_knickoint_tools:chi_mapping_tools',
            'lsdtt-burn2csv=scripts_for_lsdtopytools.lsdtt_csv_operations:burn_rast_val_to_csv',
            'lsdtt-topomap=scripts_for_lsdtopytools.lsdtt_basic_operations:topomap',
            'lsdtt-plotalltifs=scripts_for_lsdtopytools.lsdtt_autoplot_tools:plot_all_tif_of_folder',
            'lsdtt-remove-seas=scripts_for_lsdtopytools.lsdtt_basic_operations:remove_seas',
            'lsdtt-extract-single-river=scripts_for_lsdtopytools.lsdtt_basic_operations:extract_single_river_from_source',
            'lsdtt-concFFS=scripts_for_lsdtopytools.lsdtt_concavity_tools:temp_concavity_FFS_all',
            'lsdtt-concFFS-single=scripts_for_lsdtopytools.lsdtt_concavity_tools:concavity_single_basin',
            'lsdtt-concFFS-multiple=scripts_for_lsdtopytools.lsdtt_concavity_tools:concavity_multiple_basin',
            'lsdtt-concFFS-spawn-outlets=scripts_for_lsdtopytools.lsdtt_concavity_tools:spawn_XY_outlet',
            'lsdtt-concavity-down-to-top=scripts_for_lsdtopytools.lsdtt_concavity_tools:concavity_FFS_down_to_top',
            'lsdtt-concFFS-spawn-subbasins=scripts_for_lsdtopytools.lsdtt_concavity_tools:spawn_XY_outlet_subbasins',
            'lsdtt-csv-conversions=scripts_for_lsdtopytools.lsdtt_csv_operations:csv_conversions'
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    python_requires='>=3.7',
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lsdtopytools',
    name='lsdtopytools',
    packages=find_packages(include=['lsdtopytools','scripts_for_lsdtopytools']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/bgailleton/lsdtopytools',
    version='0.0.4.6',
    zip_safe=False,
    extras_require = {
        'shapefile_tools':  ["fiona","shapely","geopandas"],
        'fastscapelib_ultimate_binding_lsdtt': ["fastscapelib-fortran"]
        }
)
