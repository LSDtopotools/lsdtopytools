//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
//
// LSDChiTools
// Land Surface Dynamics ChiTools object
//
// An object within the University
//  of Edinburgh Land Surface Dynamics group topographic toolbox
//  for performing various analyses in chi space
//
//
// Developed by:
//  Simon M. Mudd
//  Martin D. Hurst
//  David T. Milodowski
//  Stuart W.D. Grieve
//  Declan A. Valters
//  Fiona Clubb
//  Boris Gailleton
//
// Copyright (C) 2016 Simon M. Mudd 2016
//
// Developer can be contacted by simon.m.mudd _at_ ed.ac.uk
//
//    Simon Mudd
//    University of Edinburgh
//    School of GeoSciences
//    Drummond Street
//    Edinburgh, EH8 9XP
//    Scotland
//    United Kingdom
//
// This program is free software;
// you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation;
// either version 2 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY;
// without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the
// GNU General Public License along with this program;
// if not, write to:
// Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor,
// Boston, MA 02110-1301
// USA
//
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// LSDChiTools.cpp
// LSDChiTools object
// LSD stands for Land Surface Dynamics
//
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// This object is written by
// Simon M. Mudd, University of Edinburgh
//
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//-----------------------------------------------------------------
//DOCUMENTATION URL: http://www.geos.ed.ac.uk/~s0675405/LSD_Docs/
//-----------------------------------------------------------------



#ifndef LSD_xtensor_convtools_CPP
#define LSD_xtensor_convtools_CPP

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include "TNT/tnt.h"
// #include "LSDMostLikelyPartitionsFinder.cpp"
// #include "LSDIndexRaster.cpp"
// #include "LSDRaster.cpp"
// #include "LSDRasterInfo.cpp"
// #include "LSDFlowInfo.cpp"
// #include "LSDJunctionNetwork.cpp"
// #include "LSDIndexChannel.cpp"
// #include "LSDChannel.cpp"
// #include "LSDIndexChannelTree.cpp"
// #include "LSDStatsTools.cpp"
// #include "LSDShapeTools.cpp"
// #include "LSDChiNetwork.cpp"
// #include "LSDBasin.cpp"
// #include "LSDParticle.cpp"
// #include "LSDChiTools.cpp"
// #include "LSDParameterParser.cpp"
// #include "LSDSpatialCSVReader.cpp"
// #include "LSDCRNParameters.cpp"
// #include "LSDRasterMaker.cpp"


#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor-python/pytensor.hpp"

#include "LSD_xtensor_convtools.hpp"

#include <iostream>
#include <numeric>
// #include <cmath>

namespace conv
{

	std::map<std::string, xt::pytensor<float,2> > map_string_2Dvec_to_py(std::map<std::string, std::vector<std::vector<float> > >& input )
	{

		std::map<std::string, xt::pytensor<float,2> > output;

		for (std::map<std::string, std::vector<std::vector<float> > >::iterator monique = input.begin(); monique!=input.end(); ++monique)
		{
			std::string butter = monique->first;
			std::vector<std::vector<float> > galabru = monique->second;
			size_t nrows = galabru.size();
			size_t ncols = galabru[0].size();

			std::array<size_t, 2> shape = { nrows, ncols };
			xt::xtensor<float, 2, xt::layout_type::row_major> awah(shape);

			for(size_t i=0; i<nrows;i++)
			for(size_t j=0; j<ncols;j++)
			{
				awah(i,j) = galabru[i][j];
			}

			output[butter] = awah;
		}

		return output;



	}


}










#endif
