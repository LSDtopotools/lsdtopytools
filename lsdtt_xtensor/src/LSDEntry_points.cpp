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



#ifndef LSDEntry_points_CPP
#define LSDEntry_points_CPP
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include "LSDStatsTools.hpp"
#include "LSDChiNetwork.hpp"
#include "LSDRaster.hpp"
#include "LSDRasterInfo.hpp"
#include "LSDIndexRaster.hpp"
#include "LSDFlowInfo.hpp"
#include "LSDJunctionNetwork.hpp"
#include "LSDIndexChannelTree.hpp"
#include "LSDBasin.hpp"
#include "LSDChiTools.hpp"
#include "LSDParameterParser.hpp"
#include "LSDSpatialCSVReader.hpp"
#include "LSDShapeTools.hpp"
#include "LSDRasterMaker.hpp"

#include <omp.h>

#include "LSDEntry_points.hpp"

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"

#include "LSDStatsTools.hpp"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include <iostream>
#include <numeric>
#include <cmath>

namespace EPPy
{
	//################################################################################################################################################
	// Function that calculates drainage density by custom areas expressed as a raster: e.g. by river basins or by lithology
	// I need a preprocessed raster (e.g., filled or breached or both or whatever) where every cells drains to at least one neightbor or is a baselevel
	//
	// B.G. 11/11/2018
	//################################################################################################################################################
	std::map<int, float> get_drainage_density_from_sources(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data,  xt::pytensor<int,2>& comparative_data, std::vector<float>& x_sources, std::vector<float>& y_sources)
	{
		// output
		std::map<int,float> output;

		TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
		// Trying to generate a ghost TNT array that points to the real values of my xtensor
		// std::cout << "DEBUG::INITIALIZING THE DATA" << std::endl;
		for(size_t i=0;i<nrows;i++)
		for(size_t j=0;j<ncols;j++)
			data_pointerast[i][j] = data(i,j); // You forgot the semicolon you twat!

		// First step is to load the raster into LSDTT
		// std::cout << "DEBUG::RATER" << std::endl;
		LSDRaster PP_raster(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
		std::vector<std::string> bound = {"n","n","n","n"};
		// std::cout << "DEBUG::FLOWINFO" << std::endl;
		LSDFlowInfo FlowInfo(bound,PP_raster);
			// std::cout << "DEBUG::1.1" << std::endl;
		
		LSDIndexRaster FlowAcc = FlowInfo.write_NContributingNodes_to_LSDIndexRaster();
			// std::cout << "DEBUG::1.2" << std::endl;

		LSDRaster DrainageArea = FlowInfo.write_DrainageArea_to_LSDRaster();
			// std::cout << "DEBUG::1.3" << std::endl;
		LSDRaster DistanceFromOutlet = FlowInfo.distance_from_outlet();
			// std::cout << "DEBUG::1.4" << std::endl;
		std::vector<int> sources;
			// std::cout << "DEBUG::1.5" << std::endl;
		sources = FlowInfo.Ingest_Channel_Heads(x_sources,y_sources);
		// now get the junction network
			// std::cout << "DEBUG::1.6" << std::endl;
  		LSDJunctionNetwork JunctionNetwork(sources, FlowInfo);
			// std::cout << "DEBUG::1.7" << std::endl;
		std::map<int,bool> iznodechan;
			// std::cout << "DEBUG::1.8" << std::endl;
  		iznodechan = JunctionNetwork.GetMapOfChannelNodes(FlowInfo);

		// std::cout << "DEBUG::DONE WITH LSDTT CONV" << std::endl;
		// I can now calculate the drainage density

		std::map<int,float> distance_p_val, area_p_val;
		std::vector<int> list_of_ID;
		for(size_t i=0;i<nrows;i++)
		for(size_t j=0;j<ncols;j++)
		{
			// std::cout << "DEBUG::I: "<< i << " || J: " << j << "VAL:" <<PP_raster.get_data_element(i,j) << std::endl;
			// std::cout << "DEBUG::1" << std::endl;
			int this_ID = comparative_data(i,j);
			// If the area is to be ignored I shall continue the loop with no bother
			if(this_ID == ndv)
				continue;
			// Initializing the data only once
			// std::cout << "DEBUG::1" << std::endl;
			if(distance_p_val.count(this_ID) ==0)
			{
				//New value
				distance_p_val[this_ID] = 0;
				area_p_val[this_ID] = 0;
				list_of_ID.push_back(this_ID);
			}
			area_p_val[this_ID] = area_p_val[this_ID] + std::pow(cellsize,2);


			// std::cout << "DEBUG::2" << std::endl;


			// get node index
			int node = FlowInfo.retrieve_node_from_row_and_column(i,j);
			int recnode; // dga->GARG

			// Check if my node is in the river network
			if(iznodechan.count(node) == 0)
				continue; // Stopping the entire loop if the node is not in my river network

			// get receiver
			FlowInfo.retrieve_receiver_information(node,recnode);
			// std::cout << "DEBUG::3" << std::endl;

			// check if flkow
			if(node == recnode)
				continue;
			// Getting the flow distance and the receiver row/col
			int rrow,rcol;
			FlowInfo.retrieve_current_row_and_col(recnode,rrow,rcol);
			int recID = comparative_data(rrow,rcol);
			float dist = FlowInfo.get_Euclidian_distance(node,recnode);
			// std::cout << "DEBUG::4" << std::endl;

			if(this_ID == recID)
				// if my nodes are in the same area then I am just adding this shit
				distance_p_val[this_ID] = distance_p_val[this_ID] + dist;
			else
			{
				// Initializing the data only once if required
				if(distance_p_val.count(recID) ==0)
				{
					//New value
					distance_p_val[recID] = 0;
					area_p_val[recID] = 0;
					list_of_ID.push_back(recID);

				}
				// Only adding half of the distance to each domains
				distance_p_val[this_ID] = distance_p_val[this_ID] + dist/2;
				distance_p_val[recID] = distance_p_val[recID] + dist/2;
			}
			// Finally adding the DA
			
			// std::cout << "DEBUG::5" << std::endl;


		}

		// Calculating the Dd
		// std::cout << "DEBUG::RETRIBUTING THE ZULU" << std::endl;

		for(std::vector<int>::iterator awah = list_of_ID.begin(); awah != list_of_ID.end(); awah++)
		{
			int this_ID = *awah;
			output[this_ID] = distance_p_val[this_ID]/area_p_val[this_ID];
		}

		// I am done Aye!
		return output;
	}
	//################################################################################################################################################//
	//############################################################### END ############################################################################//
	//################################################################################################################################################//


	//################################################################################################################################################//
	// Takes a mask raster and xy data, return map of vectors with the same dimensions than xy with some basics metrics
	// it checks a radius around each xy and father all the pixels and calculates the metrics on that part.
	// Parameters: 
	// Authors (last edit): B.G (13/11/2018)
	//################################################################################################################################################//

	std::map<std::string, xt::pytensor<float,1> > get_circular_windowed_stats(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_coord, xt::pytensor<float,1>& y_coord , float radius)
	{
		// Warning cpt
		int n_tot_ndt_for_WARNING = 0;
		// The first step is to load the raster
		TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
		// Trying to generate a ghost TNT array that points to the real values of my xtensor
		// std::cout << "DEBUG::INITIALIZING THE DATA" << std::endl;
		for(size_t i=0;i<nrows;i++)
		for(size_t j=0;j<ncols;j++)
			data_pointerast[i][j] = data(i,j); // You forgot the semicolon you twat!

		// First step is to load the raster into LSDTT
		// std::cout << "DEBUG::RATER" << std::endl;
		LSDRaster RASR(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);

		// Getting the index_radius ie the radius in term of pixel length
		int index_radius = int(round(radius/cellsize));
		if(index_radius % 2 != 0)
		{
			index_radius--;			
		}
		std::cout << "DEBUG:: Index radius: " << index_radius << std::endl;
		// Let's initialize all the variable I need for the loop
		std::array<size_t, 1> shape = { x_coord.size() };
		xt::xtensor<float,1> gogmedian(shape);
		xt::xtensor<float,1> gogmin(shape);
		xt::xtensor<float,1> gogmax(shape);
		xt::xtensor<float,1> gogmean(shape);
		xt::xtensor<float,1> gogstd_dev(shape);
		xt::xtensor<float,1> gogstd_err(shape);
		xt::xtensor<float,1> gogfiQ(shape);
		xt::xtensor<float,1> gogtiQ(shape);
		xt::xtensor<float,1> gogn(shape);
		xt::xtensor<float,1> gogn_ndv(shape);

		#pragma parallel
		{	
			#pragma for schedule(static,1) num_threads(4)
			for(int i=0; i< x_coord.size(); i++)
			{
				// Local variablessssss
				float X = x_coord[i];
				float Y = y_coord[i];
				int row = 0, col = 0;
				float tmin=0, tmax=0, tmean=0,tstd_dev=0,tstd_err=0,tn=0,tn_ndv=0;

				// Getting row_col 
				RASR.get_row_and_col_of_a_point(X,Y,row,col);

				//extract all row/col values in the radius
				std::vector<float> das_values;
				for(int tr = -1*index_radius/2; tr <= index_radius/2; tr++)
				for(int tc = -1*index_radius/2; tc <= index_radius/2; tc++)
				{
					int ttr = row+tr, ttc = col+ tc;	
					//Check if in raster
					if(ttr<0 || ttr>= nrows || ttc < 0 || ttc >= ncols)
						continue;
					// Now checks if still in the radius
					if(std::sqrt( std::pow(ttr-row,2) + std::pow(ttc-col,2) ) > index_radius)
						continue;
					// we are in the radius, let's gather the data ʕ•͡ᴥ • ʔ
					float val = RASR.get_data_element(ttr,ttc);
					// std::cout << "DEBUG::val: " << val << std::endl;
					if(val!=ndv)
						{das_values.push_back(RASR.get_data_element(ttr,ttc));}
					else
						{tn_ndv = tn_ndv + 1;}
				}

				// I am just checking here if the amount of nodata matches
				// std::cout << "DEBUG::VALSIZE : " << das_values.size() << std::endl;
				// std::cout << "DEBUG::tn_ndv: " << tn_ndv << std::endl;
				if(das_values.size() == tn_ndv)
				{
					n_tot_ndt_for_WARNING++;
					gogmedian[i] = 0;
					gogmin[i] = 0;
					gogmax[i] = 0;
					gogmean[i] = 0;
					gogstd_dev[i] = 0;
					gogstd_err[i] = 0;
					gogn[i] = 0;
					gogn_ndv[i] = tn_ndv;
					gogfiQ[i] = 0;
					gogtiQ[i] = 0;
				}
				else
				{
					gogmedian[i] = get_median(das_values);
					gogmin[i] = Get_Minimum(das_values, ndv);
					gogmax[i] = Get_Maximum(das_values, ndv);
					gogmean[i] = get_mean(das_values);
					gogstd_dev[i] = get_standard_deviation(das_values,gogmean[i]);
					gogstd_err[i] = get_standard_error(das_values, gogstd_dev[i]);
					gogn[i] = float(das_values.size());
					gogn_ndv[i] = float(das_values.size()) + tn_ndv;
					gogfiQ[i] = get_percentile(das_values, 25);
					gogtiQ[i] = get_percentile(das_values, 75);
				}

			}
		}
		std::map<std::string, xt::pytensor<float,1> > output;

		output["median"] = gogmedian;
		output["min"] = gogmin;
		output["max"] = gogmax;
		output["mean"] = gogmean;
		output["std_dev"] = gogstd_dev;
		output["std_err"] = gogstd_err;
		output["N"] = gogn;
		output["N_tot"] = gogn_ndv;
		output["first_quartile"] = gogfiQ;
		output["third_quartile"] = gogtiQ;

		// Done basically??
		if(n_tot_ndt_for_WARNING>0)
			std::cout << "lsdtt_xtensor::IMPORTANT WARNING, I am done with the circular window test, however I found " << n_tot_ndt_for_WARNING << " points in a total circle of no data. That might be normal in your case, I prefer to check with you though. I mean, I've been told to do that, I cannot judge if it is relevant in your case, just sayin." << std::endl;


		return output;

	}



	std::map<std::string, std::vector<std::vector<float> > > get_polyfit_rasters(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<int,1> selecao, float window_radius)
	{
		// output
		std::map<std::string, std::vector<std::vector<float> > > output;
		// The first step is to load the raster
		TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
		// Trying to generate a ghost TNT array that points to the real values of my xtensor
		// std::cout << "DEBUG::INITIALIZING THE DATA" << std::endl;
		for(size_t i=0;i<nrows;i++)
		for(size_t j=0;j<ncols;j++)
			data_pointerast[i][j] = data(i,j); // You forgot the semicolon you twat!

		// First step is to load the raster into LSDTT
		// std::cout << "DEBUG::RATER" << std::endl;
		LSDRaster RASR(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);

		std::vector<LSDRaster> polyf;
		// 0 -> Elevation (smoothed by surface fitting)
		// 1 -> Slope
		// 2 -> Aspect
		// 3 -> Curvature
		// 4 -> Planform Curvature
		// 5 -> Profile Curvature
		// 6 -> Tangential Curvature
		// 7 -> Stationary point classification (1=peak, 2=depression, 3=saddle)

		// raster selection
		vector<int> pol_selecao(8);
		for(size_t i=0; i<8; i++)
			pol_selecao[i] = selecao[i];
		polyf = RASR.calculate_polyfit_surface_metrics(window_radius, pol_selecao);
		if(selecao[8]==1)
		{
			std::cout <<"THIS "<< std::endl;
			LSDRaster tR = RASR.hillshade();
			polyf.push_back(tR);
			std::cout <<"THIS polyf is fucking  "<< polyf.size() << std::endl;
		}
		else 
		{
			TNT::Array2D<float> void_array(1,1,ndv);
  			LSDRaster ovdhga(1,1,ndv,ndv,ndv,ndv,void_array);
			polyf.push_back(ovdhga);
		}

		std::vector<string> vec_of_string = {"elevation", "slope","aspect","curvature","planform_curvature","profile_curvature","tangential_curvature","stationary_point","hillshade"};

		for(int s=0;s<9;s++)
		{
			std::cout << s << std::endl;
			if(selecao[s]==1)
			{
				std::cout <<"THIS " << s << std::endl;
				std::vector<std::vector<float> > this_rast(nrows, std::vector<float>(ncols)); // Defaults to zero initial value
				for (size_t tr=0; tr<nrows;tr++)
				for (size_t tc=0; tc<ncols;tc++)
				{
					this_rast[tr][tc] = polyf[s].get_data_element(tr,tc);
				}
				std::cout << "BITE " << std::endl;

				std::string dart = vec_of_string[s];
				std::cout << "BITE1 " << std::endl;

				output[dart] = this_rast;

				std::cout << "BITE2 " << std::endl;

			}
		}
		std::cout << "BITE3 " << std::endl;


		return output;


	}















}










#endif
