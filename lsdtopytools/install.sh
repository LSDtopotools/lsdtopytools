#!/bin/bash
pip uninstall -y lsdtopytools 
python setup.py sdist bdist_wheel 
pip install dist/lsdtopytools-0.0.1-py2.py3-none-any.whl 