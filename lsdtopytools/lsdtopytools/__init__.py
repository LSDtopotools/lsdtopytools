"""
__innit__ for lsdtopytools. That's wher you have to add your new modules if you create new.
Contact B.G. for info. 
"""

__author__ = """Boris Gailleton"""
__email__ = 'b.gailleton@sms.ed.ac.uk'
__version__ = '0.0.1'
import sys
# Dealing with the imports: making it compatible with python 3 and 2
if(sys.version[0] == 2):
	from raster_loader import *
	from geoconvtools import *
	from lsdtopytools_utilities import *
	from lsdtopytools import LSDDEM
	# from LSDTribBas import *
	from quickplot import *
	from quickplot_utilities import *
	from quickplot_movern import *
	from quickplot_ksn_knickpoints import *
	from argparser_debug import *
	# from Muddpyle import *
	from numba_tools import *
	from concavity_automator import *
else:
	from .raster_loader import *
	from .geoconvtools import *
	from .lsdtopytools_utilities import *
	from .lsdtopytools import LSDDEM
	# from .LSDTribBas import *
	from .quickplot import *
	from .quickplot_utilities import *
	from .quickplot_movern import *
	from .quickplot_ksn_knickpoints import *
	from .argparser_debug import *
	# from .Muddpyle import *
	from .numba_tools import *
	from .concavity_automator import *

	try:
		from .shapefile_tools import *
	except ImportError:
		shp_not_installed = True
	# try:
	# 	from .fastscapelib_ultimate_binding_lsdtt import *
	# except ImportError:
	# 	shp_not_installed = True