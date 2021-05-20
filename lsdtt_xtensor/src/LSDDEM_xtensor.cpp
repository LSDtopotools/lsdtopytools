//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
//
// lsdtopytools - LSDDEM_xtensor 
// Land Surface Dynamics python binding
//
// An object within the University
//  of Edinburgh Land Surface Dynamics group topographic toolbox
//  This is the main object to get the bindings. It manages a "pythonisation"
//  of the c++ logic and the automatisation of I/O routines from python 
//
// Developed by:
//  Boris Gailleton
//  Simon M. Mudd
//
// Copyright (C) 2018 Simon M. Mudd 2018
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

#ifndef LSDDEM_xtensor_CPP
#define LSDDEM_xtensor_CPP
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <ctime>
#include <fstream>
#include <functional>
#include <queue>
#include <limits>
#include <chrono>
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

#include "LSDDEM_xtensor.hpp"

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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>


// This empty constructor is just there to have a default one.
void LSDDEM_xtensor::create()
{
  // Initialising empty objects (throw an error if no default constructors in these classes)
  BaseRaster = return_fake_raster();
  PPRaster = return_fake_raster();
  DrainageArea = return_fake_raster();
  DistanceFromOutlet = return_fake_raster();
  chi_coordinates = return_fake_raster();
  FlowAcc = return_fake_indexraster();
  BasinArrays = return_fake_indexraster();
  FlowInfo = return_fake_flowinfo();
  JunctionNetwork = return_fake_junctionnetwork();
  ChiTool = return_fake_chitool();
  // std::exit(EXIT_FAILURE);
}

// Real constructor for LSDDEM_xtensor object.
// It takes te following arguments, easily accessible when loading a raster with GDAL-like programs
//  tnrows(int): number of row in the raster array
//  tncols(int): number of columns in the raster array
//  txmin(float): minimum X coordinate
//  tymin(float): minimum Y coordinate
//  tcellsize (float): resolution (presumably in metres)
//  tndv(float): No data value
//  data(pytensor, 2D array): the raster array
// Returns: a LSDDEM object (Does not return it but build it)
// Authors: B.G.
void LSDDEM_xtensor::create(int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& data)
{
  // Loading the attributes of the main raster, Relatively straightforward to guess their use
  nrows = tnrows;
  ncols = tncols;
  xmin = txmin;
  ymin = tymin;
  cellsize = tcellsize;
  NoDataValue = tndv;
  is_preprocessed = false;
  verbose = true; // I'll add it later, I want it to remain True at the moment as there are crucial debugging and warning messages

  // Generating the LSDraster object of the base raster.
  BaseRaster = LSDRaster(nrows, ncols, xmin, ymin, cellsize, NoDataValue, xt_to_TNT(data,size_t(nrows),size_t(ncols)), false);

  // Initialising empty objects (throw an error if no default constructors in these classes)
  // I think these are not required anymore but I am keeping it at the moment
  // std::cout << "BUK" << std::endl;
  // PPRaster = return_fake_raster();
  // std::cout << "BUK2" << std::endl;
  // DrainageArea = return_fake_raster();
  // std::cout << "BUK34" << std::endl;
  // DistanceFromOutlet = return_fake_raster();
  // std::cout << "BUK5" << std::endl;
  // chi_coordinates = return_fake_raster();
  // std::cout << "BUK6" << std::endl;
  // FlowAcc = return_fake_indexraster();
  // std::cout << "BUK7" << std::endl;
  // BasinArrays = return_fake_indexraster();
  // std::cout << "BUK" << std::endl;
  // FlowInfo = return_fake_flowinfo();
  // std::cout << "BBAGU:K" << std::endl;
  
  // Dealing here with some error messages
  std::cout << std::endl;
  JunctionNetwork = return_fake_junctionnetwork();
  std::cout << "Yes, yes, we know. Ignore the above depressing debugging text, that's perfectly normal to feel empty sometimes." << std::endl;

  // And finally the ChiTools
  ChiTool = return_fake_chitool();

  // Initialising the boundary conditions
  boundary_conditions = {"n","n","n","n"};
  default_boundary_condition = true; // This is the default and recommended boundary conditions.

  // Setting a default value to the number of nodes to visit below a base level.
  n_nodes_to_visit_downstream = 50;

  //Done! Object constructed.
}


// The PreProcessing routines wrap raster filling, depression breaching and (TODO) eventual filtering essential to most of the routines
// Parameters:
//    carve(bool): Wanna carve?
//    fill(cool): Wanna fill?
//    min_slope_for_fill(float): If you are filling, it requires a minimum slope to induce to the raster.
// Returns: Nothing, but implement the preprocessed raster in the object
// Authors: B.G.
void LSDDEM_xtensor::PreProcessing(bool carve, bool fill, float min_slope_for_fill)
{
  // Carve the dem first
  if(carve)
  {
    PPRaster = BaseRaster.Breaching_Lindsay2016();
  }
  // If you want to fill on the top of carving (recommended, the carving does not impose a slope on flat surface)
  if(carve && fill)
  {
    PPRaster = PPRaster.fill(min_slope_for_fill);
  }
  // If only filling is required case
  else if (fill){PPRaster = BaseRaster.fill(min_slope_for_fill);}
  // Nothing, then PPRaster is base raster
  else if(carve == false && fill == false){PPRaster = BaseRaster;}

  // // I don't need the base raster anymore
  // BaseRaster = LSDRaster();
}

// Calculates the FlowInfo object with D8 method
void LSDDEM_xtensor::calculate_FlowInfo()
{
  // Alright that's all I need, let's get the attributes now
  //# FlowInfo contains node ordering and all you need about navigatinfg through donor/receiver through your raster
  FlowInfo = LSDFlowInfo(boundary_conditions, PPRaster);
  //# Flow accumulation represent the number of pixels draining in each pixels
  FlowAcc = FlowInfo.write_NContributingNodes_to_LSDIndexRaster();
  //# FlowAcc can be converted to drainage area if we know the res. Which we do ofc
  DrainageArea = FlowInfo.write_DrainageArea_to_LSDRaster();
  //# I'll need the distance from outet for a lot of application
  DistanceFromOutlet = FlowInfo.distance_from_outlet();
}

// Calculates the FlowInfo object with D-infinity Dinf method
void LSDDEM_xtensor::calculate_FlowInfo_Dinf()
{
  std::cout << "IMPORTANT-WARNING: I havent tested the Dinf approach yet, carful with that." << std::endl;
  // I just need that before
  vector<string> temp_bc(4);
  boundary_conditions = {"n","n","n","n"};
  // Alright that's all I need, let's get the attributes now
  std::vector<LSDRaster> DINFRAST = PPRaster.D_inf_flowacc_DA();
  //# FlowInfo contains node ordering and all you need about navigatinfg through donor/receiver through your raster
  FlowInfo = LSDFlowInfo(boundary_conditions, PPRaster);
  //# Flow accumulation represent the number of pixels draining in each pixels
  FlowAcc = DINFRAST[0];
  //# FlowAcc can be converted to drainage area if we know the res. Which we do ofc
  DrainageArea = DINFRAST[1];
  //# I'll need the distance from outet for a lot of application
  DistanceFromOutlet = FlowInfo.distance_from_outlet();
}

// Calculates the sources node index once for all
// Different wil be available when I'll get the SpectralRaster to work with python
void LSDDEM_xtensor::calculate_channel_heads(std::string method, int min_contributing_pixels)
{
  // Extraction of river heads and subsequent channel network

  //# Simplest method: area threshold: how many accumulated pixels do you want to initiate a channel.
  //# Easily constraniable and suitable for cases where the exact location of channel heads  does not matter (+/-500 m for example)
  sources = std::vector<int>();
  if(method == "min_contributing_pixels")
  {
    // Sources contains the node index of each channel heads
    sources = FlowInfo.get_sources_index_threshold(FlowAcc, min_contributing_pixels);
    // I am saving the min_contributing pixel in a global way: I need it for some reasons
    min_cont = min_contributing_pixels;
    if(sources.size()==0){std::cout << "I did not find any sources" << std::endl; exit(EXIT_FAILURE);}
  }
  else
  {
    // Here later will lie the other methods
    std::cout << "FATAL ERROR::Method " << method << " not implemented!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  
  sources_clean = sources;
}


// This ingests a lost of channel heads from external sources
// For example calculated with other LDSTopoTools algotirhms
void LSDDEM_xtensor::ingest_channel_head(xt::pytensor<int,1>& these_sources)
{
  // Need to convert the sources to vector
  sources = std::vector<int>(these_sources.size());

  // Initialising the minimum to the maximum possible
  int minimum_there = std::numeric_limits<int>::max(); 
  for(size_t i=0; i< these_sources.size(); i++)
  {
    // Current node
    int this_node = these_sources[i];
    sources[i]= this_node;
    // Getting the location of that node
    int row,col;
    // getting the minimum DA for other purposes
    FlowInfo.retrieve_current_row_and_col(this_node,row,col);
    if(DrainageArea.get_data_element(row,col)<minimum_there)
      min_cont = DrainageArea.get_data_element(row,col);
  }// done
}


// Junction Network is an object that helps navigating through the river network junction by junction
void LSDDEM_xtensor::calculate_juctionnetwork(){JunctionNetwork = LSDJunctionNetwork(sources, FlowInfo);}

// A method to get outlet of basins (and subsequent drainage network informations) from a list of xy outlets.
void LSDDEM_xtensor::calculate_outlets_locations_from_xy(xt::pytensor<float,1>& tx_coord_BL, xt::pytensor<float,1>& ty_coord_BL, int search_radius_nodes,int threshold_stream_order, bool test_drainage_boundary, bool only_take_largest_basin)
{
  // Internal calculations needd a bunch of temp raster
  std::vector<float> x_coord_BL, y_coord_BL;
  for(size_t i=0; i<tx_coord_BL.size(); i++)
  {
    x_coord_BL.push_back(tx_coord_BL(i));
    y_coord_BL.push_back(ty_coord_BL(i));
  }
  std::vector<int> valid_cosmo_points;         // a vector to hold the valid nodes
  std::vector<int> snapped_node_indices;       // a vector to hold the valid node indices
  std::vector<int> snapped_junction_indices;   // a vector to hold the valid junction indices
  // ACtual localisation of required outlets
  JunctionNetwork.snap_point_locations_to_channels(x_coord_BL, y_coord_BL, search_radius_nodes, threshold_stream_order, FlowInfo, valid_cosmo_points, snapped_node_indices, snapped_junction_indices);
  // std::cout << snapped_junction_indices.size() << std::endl;
  baselevel_junctions = snapped_junction_indices;// saving

  if(baselevel_junctions.size() ==0)
  {
    std::cout << "I DID NOT FIND ANY BASINS, ADAPT YO COORDINATES" << std::endl;
    exit(EXIT_FAILURE);
  }
  // if I found several basins, I am checking if they are nested. Unfortunately we don't support nested basins yet
  if(baselevel_junctions.size() >1)
    baselevel_junctions = JunctionNetwork.Prune_Junctions_If_Nested(baselevel_junctions,FlowInfo, FlowAcc);
  // Do you want to chek if drainage boundaries are within the raster (strongly adviced)
  if(test_drainage_boundary){baselevel_junctions = JunctionNetwork.Prune_Junctions_Edge_Ignore_Outlet_Reach(baselevel_junctions, FlowInfo, PPRaster);}
  // Sometimes you jsut want your largest basin
  if(only_take_largest_basin){baselevel_junctions = JunctionNetwork.Prune_Junctions_Largest(baselevel_junctions, FlowInfo, FlowAcc);}
  // clean the river network
  JunctionNetwork.get_overlapping_channels_to_downstream_outlets(FlowInfo, baselevel_junctions, DistanceFromOutlet, sources_clean, outlet_nodes,baselevel_nodes,n_nodes_to_visit_downstream);
  // this function calculate a vector containing the node index of each investigated basins. Node equivalent of baselevel_junctions
  this->calculate_baselevel_nodes_unique();
}


// A method to get outlet of basins (and subsequent drainage network informations) from a list of xy outlets.
void LSDDEM_xtensor::calculate_outlets_locations_from_xy_v2(xt::pytensor<float,1>& tx_coord_BL, xt::pytensor<float,1>& ty_coord_BL, int n_pixels, bool check_edges)
{
  // First I need the node indices of the XY points
  std::vector<int> target_nodes;
  // Reserving best-case size
  target_nodes.reserve(tx_coord_BL.size());
  // Iterating through all coordinates in order to create the target nodes
  for(size_t i = 0; i<tx_coord_BL.size(); i++)
  {
    // GEtting row and column of the XY, the true at the end snap the node to the closest DEM point if outside
    int this_row, this_col; this->XY_to_rowcol(tx_coord_BL[i], ty_coord_BL[i], this_row, this_col,true);
    // Getting the node index
    int this_node = FlowInfo.get_NodeIndex_from_row_col(this_row,this_col);
    // emplace_back is the most efficient way to assign value to a vector
    target_nodes.emplace_back(this_node);
  }

  // Just making sure my vector is alright
  target_nodes.shrink_to_fit();

  // Preformatting the numpy array later needed
  xt::pytensor<int,1> TTG = xt::adapt(target_nodes);
  // Calculating from the lower-level method
  this->calculate_outlets_locations_from_nodes_v2(TTG, n_pixels,check_edges);
}

// A method to get outlet of basins (and subsequent drainage network informations) from a list of outlet nodes.
void LSDDEM_xtensor::calculate_outlets_locations_from_nodes_v2(xt::pytensor<int,1>& target_nodes, int n_pixels, bool check_edges)
{
  // Formatting my numpy array into something LSDTT can understand: a std::vector
  std::vector<int> vec_TG;
  vec_TG.reserve(target_nodes.size());
  // Range-based loop are great addition of c++11 standard!!
  for(auto v : target_nodes)
    vec_TG.emplace_back(v);

  // If required, first check a radius around the inputted coordinates and snap to the most probable point for the outlet
  // THe most probable point being the one with the largest drainage area (or discharge)
  if(n_pixels > 0)
  {
    JunctionNetwork.basin_from_node_snap_to_largest_surrounding_DA( vec_TG, sources_clean, baselevel_nodes, baselevel_junctions, outlet_nodes, FlowInfo, DistanceFromOutlet, DrainageArea, n_pixels,check_edges);
  }
  // Otherwise, I assume the function is called with the EXACT location of the outlet. For example precalculated by another function just above
  // a junction and surely does not want to snap to anything unwanted
  else
  {
    JunctionNetwork.select_basin_from_nodes(vec_TG, sources_clean, baselevel_nodes, baselevel_junctions, outlet_nodes, FlowInfo, DistanceFromOutlet,check_edges);
  }

  // Calculating a vector of unique values of base levels
  this->calculate_baselevel_nodes_unique();
}

// Calculate all the drainage sub-basins draining into a mother one
std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::calculate_outlets_min_max_draining_to_baselevel(float X, float Y, double min_DA, double max_DA, int n_pixels_to_chek)
{

  // First getting the row and col indices of the mother basins
  int this_row, this_col; this->XY_to_rowcol(X,Y, this_row, this_col,true);
  if(this_row == NoDataValue || this_col == NoDataValue)
  {
    std::cout << "FATALERROR::CANNOT CONVERT COORDINATES" << X << "||" << Y << std::endl;
    std::map<std::string, xt::pytensor<float,1> > empty_output;
    return empty_output;
  }

  // Getting the node index
  int this_node= FlowInfo.get_NodeIndex_from_row_col(this_row,this_col);
  
  // Starting to snap
  int target_row, target_col;
  if(n_pixels_to_chek > 0)
  {
    DrainageArea.snap_to_row_col_with_greatest_value_in_window(this_row, this_col, target_row, target_col, n_pixels_to_chek);
    int this_node = FlowInfo.get_NodeIndex_from_row_col(target_row, target_col);
  }
  // Actually extracting the basin
  vector<int> out_nodes =  JunctionNetwork.basin_from_node_all_minimum_DA_for_one_watershed(this_node, sources_clean, baselevel_nodes, baselevel_junctions, outlet_nodes, FlowInfo, DistanceFromOutlet, DrainageArea, min_DA, max_DA);

  // -#-#-#-#-#-#--#-#-#-#-#-#-#-#--#
  // Formatting the output
  std::map<std::string, xt::pytensor<float,1> > output;
  std::vector<float> these_X,these_Y;
  for(size_t i=0; i< out_nodes.size(); i++)
  {
    float tX,tY; int this_node = out_nodes[i];
    FlowInfo.get_x_and_y_from_current_node(this_node, tX, tY);
    these_X.push_back(tX);
    these_Y.push_back(tY);
  }
  xt::xtensor<float,1> tX = xt::adapt(these_X);
  xt::xtensor<float,1> tY = xt::adapt(these_Y);
  output["X"] = tX;
  output["Y"] = tY;

  // #-#-#-#-#-#-#-#-#-#-#
  // Done, just preprocessing the unique base levels
  this->calculate_baselevel_nodes_unique();

  return output;
}



// Extract all basin greater than a minimum Drainage Area
std::map<std::string, std::vector<float> > LSDDEM_xtensor::calculate_outlets_locations_from_minimum_size(double minimum_DA, bool test_drainage_boundary, bool prune_to_largest)
{

  std::vector<int> nodes_used;
  nodes_used = JunctionNetwork.basin_from_node_minimum_DA(sources_clean, baselevel_nodes, baselevel_junctions,outlet_nodes, FlowInfo, DistanceFromOutlet, DrainageArea, minimum_DA,test_drainage_boundary);
  std::map<std::string, std::vector<float> > output;
  output["X"] = std::vector<float>(nodes_used.size());
  output["Y"] = std::vector<float>(nodes_used.size());
  size_t LouisVI_le_gros = 0;
  for(auto v:nodes_used)
  {
    float tx,ty;FlowInfo.get_x_and_y_from_current_node(v, tx, ty);
    output["X"][LouisVI_le_gros] = tx;
    output["Y"][LouisVI_le_gros] = ty;
    LouisVI_le_gros ++;

  }
  this->calculate_baselevel_nodes_unique();
  return output;
}

// Extract basin with oulet location
std::map<std::string, std::vector<float> > LSDDEM_xtensor::calculate_outlets_locations_from_range_of_DA(double minimum_DA, double max_DA, bool check_edges)
{

  std::vector<int> nodes_used;
  nodes_used = JunctionNetwork.basin_from_node_range_DA(sources_clean, baselevel_nodes, baselevel_junctions,outlet_nodes, FlowInfo, DistanceFromOutlet, DrainageArea, minimum_DA, max_DA,check_edges);
  std::map<std::string, std::vector<float> > output;
  output["X"] = std::vector<float>(nodes_used.size());
  output["Y"] = std::vector<float>(nodes_used.size());
  size_t LouisVI_le_gros = 0;
  for(auto v:nodes_used)
  {
    float tx,ty;FlowInfo.get_x_and_y_from_current_node(v, tx, ty);
    output["X"][LouisVI_le_gros] = tx;
    output["Y"][LouisVI_le_gros] = ty;
    LouisVI_le_gros ++;

  }

  this->calculate_baselevel_nodes_unique();
  return output;
}

// Extract the location of the main drainage basin
std::map<std::string, float> LSDDEM_xtensor::calculate_outlet_location_of_main_basin(bool test_drainage_boundary)
{
  //Get baselevel junction nodes from the whole network
  std::vector<int> BaseLevelJunctions_Initial = JunctionNetwork.get_BaseLevelJunctions();

  if(test_drainage_boundary == false){baselevel_junctions = JunctionNetwork.Prune_Junctions_Largest(BaseLevelJunctions_Initial, FlowInfo, FlowAcc);}
  else
  {
      vector<int> these_junctions = JunctionNetwork.Prune_Junctions_Edge_Ignore_Outlet_Reach(BaseLevelJunctions_Initial,FlowInfo, PPRaster);
      baselevel_junctions = JunctionNetwork.Prune_Junctions_Largest(these_junctions, FlowInfo, FlowAcc);
  }
  JunctionNetwork.get_overlapping_channels_to_downstream_outlets(FlowInfo, baselevel_junctions, DistanceFromOutlet, sources_clean, outlet_nodes,baselevel_nodes,n_nodes_to_visit_downstream);

  std::map<std::string, float> output;
  int node = JunctionNetwork.get_Node_of_Junction(baselevel_junctions[0]);
  float tx,ty;FlowInfo.get_x_and_y_from_current_node(node, tx, ty);
  output["X"] = tx;
  output["Y"] = ty;
  calculate_baselevel_nodes_unique();
  return output;

}


// Extract the location of ALL watershed
void LSDDEM_xtensor::force_all_outlets(bool test_drainage_boundary)
{
  baselevel_junctions = JunctionNetwork.get_BaseLevelJunctions();
  if(test_drainage_boundary)
    baselevel_junctions = JunctionNetwork.Prune_Junctions_Largest(baselevel_junctions, FlowInfo, FlowAcc);
  calculate_baselevel_nodes_unique();
}

// Extract watersheds by which drains to a river/line defined by xy coordinates
std::map<std::string,xt::pytensor<float,1> > LSDDEM_xtensor::extract_basin_draining_to_coordinates(xt::pytensor<float,1> input_X,xt::pytensor<float,1> input_Y, double min_DA)
{
  // Initialising the node vector
  std::vector<int> target_nodes, outnodes;
  std::vector<float> X_coord,Y_coord;
  // converting coordinates into node indices
  for(size_t i=0;i<input_X.size(); i++)
  {
    int this_row, this_col; DrainageArea.get_row_and_col_of_a_point(input_X[i], input_Y[i], this_row, this_col);
    int this_node= FlowInfo.get_NodeIndex_from_row_col(this_row,this_col);
    target_nodes.push_back(this_node);
    
    // std::cout << this_row << "," << this_col << std::endl;
  }
  // std::cout << "1" << std::endl;
  // That's it basically, running the code now
  outnodes = JunctionNetwork.basin_from_node_minimum_DA_draining_to_list_of_nodes(target_nodes, sources_clean, baselevel_nodes, 
 baselevel_junctions, outlet_nodes, FlowInfo, DistanceFromOutlet, DrainageArea, min_DA);

  for(auto v : outnodes)
  {
    int this_node = v;
    float tX,tY; FlowInfo.get_x_and_y_from_current_node(this_node, tX, tY);
    X_coord.push_back(tX);
    Y_coord.push_back(tY);
  }


  // Formating the output
  std::map<std::string,xt::pytensor<float,1> > output;
  xt::xtensor<float,1> ttX = xt::adapt(X_coord);
  xt::xtensor<float,1> ttY = xt::adapt(Y_coord);
  output["X"] = ttX;
  output["Y"] = ttY;
  calculate_baselevel_nodes_unique();

  return output;

}

void LSDDEM_xtensor::calculate_baselevel_nodes_unique()
{
  std::map<int,bool> test_in_there;
  baselevel_nodes_unique.clear();
  baselevel_nodes_unique.shrink_to_fit();
  baselevel_nodes_unique.reserve(baselevel_nodes.size());
  for(auto v: baselevel_nodes)
  {
    if(test_in_there.count(v) == 0)
    {
      test_in_there[v] = true;
      baselevel_nodes_unique.emplace_back(v);
    }
  }
  baselevel_nodes_unique.shrink_to_fit();
}

std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::get_baselevels_XY()
{
  std::map<std::string, xt::pytensor<float,1> > output;
  output["X"] = xt::zeros<float>({baselevel_nodes_unique.size()});
  output["Y"] = xt::zeros<float>({baselevel_nodes_unique.size()});
  for(size_t i=0; i<baselevel_nodes_unique.size(); i++)
  {
    int this_node = baselevel_nodes_unique[i];
    int row,col; FlowInfo.retrieve_current_row_and_col(this_node,row,col);
    float X,Y; FlowInfo.get_x_and_y_locations(row,col,X,Y);
    output["X"][i] = X;
    output["Y"][i] = Y;
  }

  return output;
}


// Calculate the chi coordinate for extracted basins for a given concavity theta and reference drainage area A_0
void LSDDEM_xtensor::generate_chi(float theta, float A_0)
{
  ChiTool = LSDChiTools(FlowInfo);
  chi_coordinates = FlowInfo.get_upslope_chi_from_multiple_starting_nodes_custom(baselevel_nodes_unique,theta,A_0, DrainageArea);
  

  ChiTool.chi_map_automator_chi_only(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes, PPRaster, DistanceFromOutlet, DrainageArea, chi_coordinates);
  std::map<int,int> temp_corresp;


  // STUFF TO DO JUST HERE!!!
  // giving the basin informations to the object
  std::vector<int> this_ordered_BL = ChiTool.get_ordered_baselevel();
  for(size_t i=0;i<this_ordered_BL.size();i++)
  {  
    baselevel_nodes_to_BK[baselevel_nodes_unique[i]] = int(i);
    BK_to_baselevel_nodes[int(i)] = baselevel_nodes_unique[i];
    int dat_junk = baselevel_junctions[i];
    BLjunctions_to_BK[dat_junk] = int(i);
    BK_to_BLjunctions[int(i)] = dat_junk;
    // std::cout << dat_junk << " ----->" << int(i) << " || baselevelnode: " <<  baselevel_nodes[i] << std::endl;    
    // BLjunctions_to_BK[this_ordered_BL[i]] = int(i);
    // BK_to_BLjunctions[int(i)] = this_ordered_BL[i];
    // std::cout << this_ordered_BL[i] << " -----." << int(i) << std::endl;    
  }
}

// Calculate ksn
void LSDDEM_xtensor::generate_ksn(int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma,int nthreads)
{
  // JunctionNetwork.get_overlapping_channels_to_downstream_outlets(FlowInfo, baselevel_junctions, DistanceFromOutlet, sources, outlet_nodes, baselevel_nodes,30);
  // if(nthreads == 0) // WARNING:: Switch that ast number back to one after debugging!!!
  // {
    ChiTool.chi_map_automator(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes,
      PPRaster, DistanceFromOutlet, 
      DrainageArea, chi_coordinates, 
      target_nodes, n_iterations, skip, minimum_segment_length, sigma);
  // }
  // else
  // {
  //   ChiTool.chi_map_automator(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes,PPRaster, DistanceFromOutlet, DrainageArea, chi_coordinates, target_nodes, n_iterations, skip, minimum_segment_length, sigma, nthreads);
  // }

  ChiTool.segment_counter(FlowInfo, 10000000000);
}

// Calculate channel gradient using Mudd et al., 2014 algorithm
std::pair<std::map<std::string, xt::pytensor<float,1> > , std::map<std::string, xt::pytensor<int,1> > > LSDDEM_xtensor::get_channel_gradient_muddetal2014(int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma,int nthreads)
{
  LSDChiTools this_Chitools(FlowInfo);
  this_Chitools.chi_map_automator_chi_only(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes, PPRaster, DistanceFromOutlet, DrainageArea, DistanceFromOutlet);

  // JunctionNetwork.get_overlapping_channels_to_downstream_outlets(FlowInfo, baselevel_junctions, DistanceFromOutlet, sources, outlet_nodes, baselevel_nodes,30);
  if(nthreads == 0) // WARNING:: Switch that ast number back to one after debugging!!!
    this_Chitools.chi_map_automator(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes,PPRaster, DistanceFromOutlet, DrainageArea, DistanceFromOutlet, target_nodes, n_iterations, skip, minimum_segment_length, sigma);
  // else
  //   this_Chitools.chi_map_automator(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes,PPRaster, DistanceFromOutlet, DrainageArea, DistanceFromOutlet, target_nodes, n_iterations, skip, minimum_segment_length, sigma, nthreads);

  this_Chitools.segment_counter(FlowInfo, 10000000000);
  // Getting the data
  std::map<std::string, xt::pytensor<float,1> > datFromCT_flaot = this->get_float_ksn_data_ext_ChiTools(this_Chitools);
  std::map<std::string, xt::pytensor<int,1> > datFromCT_int = this->get_int_ksn_data_ext_ChiTools(this_Chitools);

  return std::make_pair(datFromCT_flaot,datFromCT_int);

}

// This function calculate ksn by direct regression of chi-z steepness
std::pair<std::map<std::string, xt::pytensor<float,1> > , std::map<std::string, xt::pytensor<int,1> > > LSDDEM_xtensor::get_ksn_first_order_chi()
{
  // This requires Chitools to already be calculated as it generates the node ordering!
  // Here we collect our data maps, mchi and bchi will be 0 of course.
  std::map<std::string,xt::pytensor<int,1> > tint = this->get_int_ksn_data();
  std::map<std::string,xt::pytensor<float,1> > tflo = this->get_float_ksn_data();
  tflo.erase("m_chi");
  size_t sizla = tint["nodeID"].size();
  std::array<size_t,1> shape = {sizla};
  xt::xtensor<float,1> ksn_FO(shape);
  // iterating through my nodes
  for(size_t i=0; i<sizla; i++)
  {
    int this_node = tint["nodeID"][i];
    int row,col;
    FlowInfo.retrieve_current_row_and_col(this_node,row,col);
    int recnode; FlowInfo.retrieve_receiver_information(this_node,recnode);
    int rrow, rcol;
    FlowInfo.retrieve_current_row_and_col(recnode, rrow,rcol);
    // Calculating dz/dchi
    ksn_FO[i] = (PPRaster.get_data_element(row,col) - PPRaster.get_data_element(rrow,rcol)) / (PPRaster.get_data_element(row,col) - PPRaster.get_data_element(rrow,rcol)); 

  }

  tflo["ksn"] = ksn_FO;
  return std::make_pair(tflo,tint);
}


// Extract the knickpoint using Gailleton et al., 2019
void LSDDEM_xtensor::detect_knickpoint_locations( float MZS_th, float lambda_TVD,int stepped_combining_window,int window_stepped, float n_std_dev, int kp_node_search)
{
  ChiTool.ksn_knickpoint_automator_no_file(FlowInfo, MZS_th, lambda_TVD,stepped_combining_window,window_stepped, n_std_dev, kp_node_search);
}

TNT::Array2D<float> LSDDEM_xtensor::xt_to_TNT(xt::pytensor<float,2>& myarray, size_t tnrows, size_t tncols)
{
  std::cout << "Nrows:" << tnrows << " and ncols:" << tncols << std::endl;
  TNT::Array2D<float> out(size_t(tnrows),size_t(tncols),myarray.data());
  std::cout << "out Nrows:" << out.dim1() << " and out ncols:" << out.dim2() << std::endl;
  return out;
}


// I/O: Get chi raster
xt::pytensor<float,2> LSDDEM_xtensor::get_chi_raster()
{
  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<float,2> output(shape);

  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {output(i,j) = chi_coordinates.get_data_element(i,j);}

  return output;
}

// I/O: Get chi raster
xt::pytensor<float,2> LSDDEM_xtensor::get_chi_raster_all(float m_over_n, float A_0, float area_threshold)
{
  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<float,2> output(shape);

  LSDRaster all_chi = FlowInfo.get_upslope_chi_from_all_baselevel_nodes(m_over_n, A_0, area_threshold);

  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {output(i,j) = all_chi.get_data_element(i,j);}

  return output;
}

// I/O: Get basin raster
xt::pytensor<int,2> LSDDEM_xtensor::get_chi_basin()
{
  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<int,2> output(shape);
  LSDIndexRaster garg = ChiTool.get_basin_raster(FlowInfo, JunctionNetwork, baselevel_junctions);
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {
    int val = garg.get_data_element(i,j), val2 = -9999;
    if(val != -9999)
      val2 = BLjunctions_to_BK[val];
    output(i,j) = val2;
  }


  return output;
}

// I/O: Get basin raster
xt::pytensor<float,2> LSDDEM_xtensor::get_flow_distance_raster()
{
  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<float,2> output(shape);
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {
    float val = DistanceFromOutlet.get_data_element(i,j);
    output(i,j) = val;
  }


  return output;
}




// Let the code know that it's already preprocessed
void LSDDEM_xtensor::already_preprocessed(){is_preprocessed = true;  Array2D<float>* datptr =  BaseRaster.get_RasterDataPtr() ; PPRaster = LSDRaster(nrows, ncols, xmin, ymin, cellsize, NoDataValue, *datptr );}

// void LSDDEM_xtensor::already_preprocessed(){is_preprocessed = true; std::unique_ptr<LSDRaster> PPRaster = std::make_unique<LSDRaster>(BaseRaster);}
std::map<std::string, xt::pytensor<int,1> > LSDDEM_xtensor::get_int_ksn_data()
{
  std::map<std::string, std::vector<int> > datFromCT = ChiTool.get_integer_vecdata_for_m_chi(FlowInfo);
  std::map<std::string, xt::pytensor<int,1> > output;

  for(std::map<std::string, std::vector<int> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<int> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<int,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

std::map<std::string, xt::pytensor<int,1> > LSDDEM_xtensor::get_int_ksn_data_ext_ChiTools(LSDChiTools& this_Chitools)
{
  std::map<std::string, std::vector<int> > datFromCT = this_Chitools.get_integer_vecdata_for_m_chi(FlowInfo);
  std::map<std::string, xt::pytensor<int,1> > output;

  for(std::map<std::string, std::vector<int> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<int> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<int,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

std::map<std::string, xt::pytensor<int,1> > LSDDEM_xtensor::get_int_knickpoint_data()
{
  std::map<std::string, std::vector<int> > datFromCT = ChiTool.get_integer_vecdata_for_knickpoint_analysis(FlowInfo);
  std::map<std::string, xt::pytensor<int,1> > output;

  for(std::map<std::string, std::vector<int> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<int> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<int,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::get_float_ksn_data()
{
  std::map<std::string, std::vector<float> > datFromCT = ChiTool.get_float_vecdata_for_m_chi(FlowInfo);
  std::map<std::string, xt::pytensor<float,1> > output;

  for(std::map<std::string, std::vector<float> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<float> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<float,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::get_float_ksn_data_ext_ChiTools(LSDChiTools& this_Chitools)
{
  std::map<std::string, std::vector<float> > datFromCT = this_Chitools.get_float_vecdata_for_m_chi(FlowInfo);
  std::map<std::string, xt::pytensor<float,1> > output;

  for(std::map<std::string, std::vector<float> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<float> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<float,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::get_float_knickpoint_data()
{
  std::map<std::string, std::vector<float> > datFromCT = ChiTool.get_float_vecdata_for_knickpoint_analysis(FlowInfo);
  std::map<std::string, xt::pytensor<float,1> > output;

  for(std::map<std::string, std::vector<float> >::iterator it = datFromCT.begin(); it!= datFromCT.end(); it++)
  {
    std::string denom = it->first;
    std::vector<float> denomnomnom = it->second;

    std::array<size_t,1> siz = {denomnomnom.size()};
    xt::xtensor<float,1> this_denomnomnom(siz);
    this_denomnomnom = xt::adapt(denomnomnom,siz);
    output[denom] = this_denomnomnom;
  }

  return output;
}

// Empty LSDTT constructors
LSDRaster LSDDEM_xtensor::return_fake_raster()
{
  TNT::Array2D<float> tempTNT(2,2,-9999);
  return LSDRaster(2,2,0,0,1,-9999,tempTNT);
}
LSDIndexRaster LSDDEM_xtensor::return_fake_indexraster()
{
  TNT::Array2D<int> tempTNT(2,2,-9999);
  return LSDIndexRaster(2,2,0,0,1,-9999,tempTNT);
}
LSDFlowInfo LSDDEM_xtensor::return_fake_flowinfo()
{
  return LSDFlowInfo(BaseRaster);
}
  LSDChiTools LSDDEM_xtensor::return_fake_chitool()
{
  return LSDChiTools(FlowInfo);
}
LSDJunctionNetwork LSDDEM_xtensor::return_fake_junctionnetwork()
{
  return LSDJunctionNetwork();
}

xt::pytensor<float,2> LSDDEM_xtensor::get_hillshade(float hs_altitude, float hs_azimuth, float hs_z_factor)
{
  LSDRaster hs_raster;
  if(is_preprocessed)
    hs_raster = PPRaster.hillshade(hs_altitude,hs_azimuth,hs_z_factor);
  else
    hs_raster = BaseRaster.hillshade(hs_altitude,hs_azimuth,hs_z_factor);

  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<int,2> output(shape);
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {output(i,j) = hs_raster.get_data_element(i,j);}

  return output;
}

// Version of hillshading for custom raster
xt::pytensor<float,2> LSDDEM_xtensor::get_hillshade_custom(xt::pytensor<float,2>& datrast, float hs_altitude, float hs_azimuth, float hs_z_factor)
{

  LSDRaster hs_raster, tohillshade = LSDRaster(nrows, ncols, xmin, ymin, cellsize, NoDataValue, xt_to_TNT(datrast,size_t(nrows),size_t(ncols)));
  hs_raster = tohillshade.hillshade(hs_altitude,hs_azimuth,hs_z_factor);
  std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};
  xt::xtensor<int,2> output(shape);
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
  {output(i,j) = hs_raster.get_data_element(i,j);}

  return output;
}


std::vector<xt::pytensor<float,2> > LSDDEM_xtensor::get_polyfit_on_topo(float window, std::vector<int> selecao)
{
  //        0 -> Elevation (smoothed by surface fitting)
//        1 -> Slope
//        2 -> Aspect
//        3 -> Curvature
//        4 -> Planform Curvature
//        5 -> Profile Curvature
//        6 -> Tangential Curvature
//        7 -> Stationary point classification (1=peak, 2=depression, 3=saddle)
  vector<LSDRaster> preoutput;
  if(is_preprocessed)
  {
    preoutput = PPRaster.calculate_polyfit_surface_metrics(window, selecao);
  }
  else
  {
    preoutput = BaseRaster.calculate_polyfit_surface_metrics(window, selecao);
  }

  std::vector<xt::pytensor<float,2> > output;
  for(size_t t=0; t<preoutput.size();t++)
  {
    if(selecao[t]==1)
    {
      std::array<size_t,2> size = {size_t(nrows),size_t(ncols)};
      xt::xtensor<float,2> temparray(size);
      for(size_t i=0;i<nrows;i++)
      for(size_t j=0;j<ncols;j++)
      {temparray(i,j) = preoutput[t].get_data_element(i,j);}
      output.push_back(temparray);
      preoutput[t] = return_fake_raster();
    }
    else
    {
      std::array<size_t,2> size = {0,0};
      xt::xtensor<float,2> temparray(size);
      output.push_back(temparray);
    }
  }
  return output;
}

// This function extract the perimeter for each basin previously extraced.
// return a map with node-x-y-basin_key
std::map<int, std::map<std::string,xt::pytensor<float, 1> > > LSDDEM_xtensor::get_catchment_perimeter()
{
  // initialising the output
  std::map<int,std::map<std::string,xt::pytensor<float, 1> > > output;
  // now iterating through my basins
  int cpt = 0;
  for(std::vector<int>::iterator it = baselevel_junctions.begin(); it != baselevel_junctions.end(); it++)
  {
    LSDBasin thisBasin(*it,FlowInfo, JunctionNetwork,baselevel_nodes_unique[cpt]);
    thisBasin.set_Perimeter(FlowInfo);
    std::vector<int> these_node = thisBasin.get_Perimeter_nodes();
    std::array<size_t,1> shape = {these_node.size()};
    xt::xtensor<int,1> perinode(shape), perirow(shape), pericol(shape);
    xt::xtensor<float,1> perix(shape), periy(shape), perielev(shape), perikey(shape);
    int cpt2 = 0;
    for (std::vector<int>::iterator it2 = these_node.begin(); it2 != these_node.end(); it2++)
    {
      int this_node = *it2;
      perinode[cpt2] = this_node;
      int row,col;
      FlowInfo.retrieve_current_row_and_col(this_node,row,col);
      perirow[cpt2] = row;
      pericol[cpt2] = col;
      perikey[cpt2] = float(cpt); // basin_key
      float tx,ty;
      FlowInfo.get_x_and_y_from_current_node(this_node,tx,ty);
      perix[cpt2] = tx;
      periy[cpt2] = ty;
      perielev[cpt2] = PPRaster.get_data_element(row,col);
      cpt2++;
    }

    std::map<std::string,xt::pytensor<float, 1> > temp_ou;
    // temp_ou["node"] = perinode;
    // temp_ou["row"] = perirow;
    // temp_ou["col"] = pericol;
    temp_ou["x"] = perix;
    temp_ou["y"] = periy;
    temp_ou["elevation"] = perielev;
    temp_ou["basin_key"] = perikey;
    output[perikey[0]] = temp_ou;
    std::cout << perikey[0] << std::endl;

    cpt++;
  }
  return output;

}

void LSDDEM_xtensor::calculate_movern_disorder(float start_movern, float delta_movern, float n_movern, float A_0, float area_threshold, int n_rivers_by_combination)
{

  // A bit of timing here to check whence the code is too slow
  float time_all = 0, time_loop = 0;
  std::chrono::steady_clock::time_point begin_all = std::chrono::steady_clock::now();

  // Switch used for debugging purposes, ignore at the moment.
  float checker_A_0 = A_0;
  if(checker_A_0 <= 0)
    A_0 = 1;

  // Creating a temporary chitool 
  LSDChiTools ChiTool_disorder(FlowInfo);
  // Calculating the temporary chitool param (does not take uch time but could be optimised)
  ChiTool_disorder.chi_map_automator_chi_only(FlowInfo, sources_clean, outlet_nodes, baselevel_nodes,
                        PPRaster, DistanceFromOutlet, DrainageArea, chi_coordinates);

  // this map will store the uncertainties results
  std::map<int, std::vector<float> > best_fit_movern_for_basins;
  std::map<int, std::vector<float> > lowest_disorder_for_basins;

  //Getting the ordered base level
  std::vector<int> this_ordered_BL = ChiTool_disorder.get_ordered_baselevel();

  // creating the chi disorder vector of values: all the movern which will be tested
  std::vector<float> movern;
  for(int i=0; i<n_movern; i++)
    movern.push_back(start_movern + float(i) * delta_movern);

  // Here I am copying the info into my attribute (small vector, overcost is minimal)
  associated_movern_disorder = movern;// saving this information

  // std::map<int, std::map<float, xt::pytensor<float,2> > > disorder_all_values_by_movern;

  // Initialising the movern by BK to empty vectors (cannot presize it unfortunately) 
  std::vector<std::vector<float> > movern_per_BK(this_ordered_BL.size());
  for(size_t j=0;j<movern_per_BK.size();j++)
  {
    std::vector<float> temp;
    movern_per_BK[int(j)] = temp;

       
  }


  // std::map<int, std::map<float, std::vector<float> > > disorder_all_values_by_movern_xt;
  // std::map<int, std::map<int, std::vector<float> > > best_fits_movern_per_BK_per_SK

  // Before running the tests, I need to preprocess my basin to gather some informations:
  // First about each basins: map<int,int>& sources_are_keys, map<int,int>& comboindex_are_keys and vector< vector<int> >& combo_vecvec
  std::map<int, std::map<int,int> > sources_are_keys_FE_BK;
  std::map<int, std::map<int,int> > comboindex_are_keys_FE_BK;
  std::map<int, std::vector<std::vector<int> > > combo_vecvec_FE_BK;
  std::map<int, std::vector<int> > node_per_basin_FE_BK;
  std::map<int, std::vector<int> > SK_per_basin_FE_BK;
  std::map<int, std::vector<float> > elevation_per_basin_FE_BK;

  normalised_disorder_n_pixel_by_combinations = std::vector<std::vector<int> >(int(this_ordered_BL.size()));

  for( int BK =0; BK< int(this_ordered_BL.size()); BK++)
  {
    std::map<int,int> temp_sources_are_keys;
    std::map<int,int> temp_comboindex_are_keys;
    std::vector<std::vector<int> > temp_combo_vecvec;
    std::vector<int> temp_nodes, temp_nodes_sorted;
    std::vector<float> temp_elev, temp_elev_sorted;
    std::vector<int> temp_SK, temp_SK_sorted;


    std::map<float,std::vector<float> > temp2;
    for(auto jug : movern)
    {
      // std::cout << jug << endl;
      std::vector<float> temp3;
      temp2[jug] = temp3;
    }
    // std::cout << temp2.size()  << "dfsjlksdjf" << std::endl;

    disorder_all_values_by_movern[BK] = temp2;

    // std::map<float, xt::pytensor<float,1> > temp3;


    // disorder_all_values_by_movern[BK] = temp3;

    // Feeding the maps
    ChiTool_disorder.precombine_sources_for_disorder_with_uncert_opti(FlowInfo, BK, temp_sources_are_keys, temp_comboindex_are_keys, 
      temp_combo_vecvec, temp_nodes, temp_SK, n_rivers_by_combination);


    temp_elev.resize(temp_nodes.size());
    // Now dealing with node sorting for elevation: getting the elevation.
    ChiTool_disorder.update_elevation_vector_for_opti_disorder_with_uncert(temp_nodes, temp_elev);
    std::cout << "size bas = " << temp_nodes.size() << std::endl;

    // Now sorting once for all by ascending elevation
    std::vector<size_t> index_yolo;
    matlab_float_sort(temp_elev, temp_elev_sorted, index_yolo);
    matlab_int_reorder(temp_nodes, index_yolo, temp_nodes_sorted);
    matlab_int_reorder(temp_SK, index_yolo, temp_SK_sorted);

    normalised_disorder_n_pixel_by_combinations[BK] = std::vector<int>(temp_combo_vecvec.size());

    //Integrating them to the above-scoped map
    sources_are_keys_FE_BK[BK] = temp_sources_are_keys;
    comboindex_are_keys_FE_BK[BK] = temp_comboindex_are_keys;
    combo_vecvec_FE_BK[BK] = temp_combo_vecvec;
    node_per_basin_FE_BK[BK] = temp_nodes_sorted;
    elevation_per_basin_FE_BK[BK] = temp_elev_sorted;
    SK_per_basin_FE_BK[BK] = temp_SK_sorted;
  }

  // Preotpimisation finished

  // KEEP FOR DEBUGGING PURPOSES
  // for(auto i:movern)
  //   raw_disorder_per_SK[i] = {};

  // Now checking the best fits for each movern values to check
  for(size_t i=0; i<n_movern; i++)
  {  
    if(verbose==true)
      std::cout << "Disorder movern for theta=" << movern[i] << " ..." << std::endl;

    // recalculating chi for this movern
    LSDRaster this_chi = FlowInfo.get_upslope_chi_from_all_baselevel_nodes(movern[i], A_0, area_threshold);
    ChiTool_disorder.update_chi_data_map(FlowInfo, this_chi); // updating the chitool object


    // Potential multithreading here:
    // Checking each basins independently for that value of movern
    for(size_t BK =0; BK< this_ordered_BL.size(); BK++)
    {
      // First step is to extract the collinearity for the entire basin
      // This takes the map of vector and for each watershed push back one float corresponding to the disoder value for 
      // the movern tested in movern[i]
      movern_per_BK[BK].push_back(ChiTool_disorder.test_collinearity_by_basin_disorder(FlowInfo, BK));


      // updating the chi vector associated with the uncert
      std::vector<float> this_chi_basin(node_per_basin_FE_BK[BK].size()); ChiTool_disorder.update_chi_vector_for_opti_disorder_with_uncert(node_per_basin_FE_BK[BK], this_chi_basin);
      
      // I have the disorder value for the overall basin


      // Now I want an idea of how spreaded the disorder value is within the basin
      // See Mudd et al 2018 for detail about teh algorithm


      //First I am initialising a map with: vector of source key tested -> vector of disorder values
      std::map< std::vector<int>, std::vector<float> > map_of_disorder_stats;
      std::vector<float> disorder_vals;

      std::chrono::steady_clock::time_point begin_loop = std::chrono::steady_clock::now();

      // This is the main function: calling the algorithm calculating the movern
      // The checker is kept for the sake of compatibility test: older version of that algorithm was a wee buggy
      // and we want to be able to check some of the results calculated with the former algorithm
      // UPDATE: the extent of the former bug was minimal and reserved for big basins. 
      if(checker_A_0 == -2)
      {
        // std::cout << "USING a bit old FUNCTION FOR TEST PURPOSES it works at least!!!!" << std::endl;

        // Last version of the algorithm
        disorder_vals = ChiTool_disorder.test_collinearity_by_basin_disorder_with_uncert(FlowInfo, BK);
      }
      else if (checker_A_0 == -1)
      {
        // std::cout << "USING OLD FUNCTION FOR TEST PURPOSES, DONT DO IT!!!!" << std::endl;
        // new version of the algorithm
        map_of_disorder_stats = ChiTool_disorder.TEST_FUNCTION_OLD_DISORDER_DO_NOT_USE_KEY(FlowInfo, BK);
      }
      else if (checker_A_0 > 0)
      {
        // std::cout << "USING MOST RECENT VERSION" << std::endl;

        std::map<int,int>& temp_sources_are_keys = sources_are_keys_FE_BK[BK];
        std::map<int,int>& temp_comboindex_are_keys = comboindex_are_keys_FE_BK[BK];
        std::vector<std::vector<int> >& temp_combo_vecvec = combo_vecvec_FE_BK[BK];
        std::vector<int>& temp_nodes =  node_per_basin_FE_BK[BK];
        std::vector<float>& temp_elev = elevation_per_basin_FE_BK[BK];
        std::vector<int>& temp_SK = SK_per_basin_FE_BK[BK];
        std::vector<int>& temp_n_pix_comb =  normalised_disorder_n_pixel_by_combinations[BK];
        // map_of_disorder_stats = ChiTool_disorder.opti_collinearity_by_basin_disorder_with_uncert_retain_key(FlowInfo, BK,
        //  sources_are_keys_FE_BK[BK], comboindex_are_keys_FE_BK[BK], this_chi_basin, elevation_per_basin_FE_BK[BK], 
        //  SK_per_basin_FE_BK[BK], node_per_basin_FE_BK[BK], combo_vecvec_FE_BK[BK]);
        disorder_vals = ChiTool_disorder.opti_collinearity_by_basin_disorder_with_uncert_retain_key(FlowInfo,
          BK, temp_sources_are_keys, temp_comboindex_are_keys, this_chi_basin,
          temp_elev,temp_SK , temp_nodes, temp_combo_vecvec, temp_n_pix_comb);
        // std::cout << "DONE WITH TEH MAIN LOOP, got " <<disorder_vals.size() << " out of " << temp_combo_vecvec.size()  << std::endl;


      }

      // DEBUG TO REMOVE AT AOME POINTS
      // for(auto it : map_of_disorder_stats)
      // {
      //   DEBUG_get_all_dis_SK_SK.push_back(it.first);
      //   DEBUG_get_all_dis_SK_val.push_back(it.second);
      // }

      std::chrono::steady_clock::time_point end_loop = std::chrono::steady_clock::now();

      time_loop += float(std::chrono::duration_cast<std::chrono::microseconds>(end_loop - begin_loop).count()) / 1000000;

      
      // determinng the number of values
      size_t nval = disorder_vals.size();
      int tempID = 0;
      // for(std::map<std::vector<int>, std::vector<float> >::iterator olive = map_of_disorder_stats.begin(); olive != map_of_disorder_stats.end(); olive++)
      // {
      //   nval += olive->second.size();
      //   for(auto& v:olive->second)
      //     all_disorders.push_back(v);        

      //   tempID++;
      // }

     


      // initialising vector with the right size
      std::vector<float>& disorder_stats = disorder_vals;
      // std::vector<std::vector<int> > associated_key(nval);
      // size_t ti =0 ;
      // for(std::map<std::vector<int>, std::vector<float> >::iterator olive = map_of_disorder_stats.begin(); olive != map_of_disorder_stats.end(); olive++)
      // {
      //   for( size_t gaspaccio = 0; gaspaccio< olive->second.size(); gaspaccio++)
      //   {
      //     disorder_stats[ti] = olive->second[gaspaccio];
      //     associated_key[ti] = olive->first;
      //     ti++;
      //     // std::cout << olive->first << std::endl;
      //   }

      // }


      // if this is the first m over n value, then initiate the vectors for this basin key
      if (i == 0)
      {
        lowest_disorder_for_basins[BK] = disorder_stats;
        
        int n_combos_this_basin = int(disorder_stats.size());

        std::vector<float> best_fit_movern( disorder_stats.size() );
        for(int bf = 0; bf < n_combos_this_basin; bf++)
        {
          best_fit_movern[bf] = movern[i];
        }
        best_fit_movern_for_basins[BK] = best_fit_movern;
      }
      else
      {
        // loop through all the combos and get the best fit movern
        std::vector<float> existing_lowest_disorder = lowest_disorder_for_basins[BK];
        std::vector<float> existing_best_fit_movern = best_fit_movern_for_basins[BK];
        int n_combos_this_basin = int(disorder_stats.size());
        
        for(int bf = 0; bf < n_combos_this_basin; bf++)
        {
          if (existing_lowest_disorder[bf] > disorder_stats[bf] )
          {
            existing_lowest_disorder[bf] = disorder_stats[bf];
            existing_best_fit_movern[bf] = movern[i];
          }
        }
        lowest_disorder_for_basins[BK] = existing_lowest_disorder;
        best_fit_movern_for_basins[BK] = existing_best_fit_movern;
      }

      // IKNOW WHAST IS WRONG WITH YOU
      // YOU REINITIALISE THE LIST EACH NK
      // Updating the map for each source keys now, this is not the most optimal way to do it but the performance loss is extremely low and this is the fastest way to code it from waht has
      // been done before anyway
      // std::map<int, std::vector<float> > temp_best_fits_movern_per_BK_per_SK;
      // for(size_t trolloc = 0; trolloc < best_fit_movern_for_basins[BK].size();trolloc++)
      // {
      //   std::vector<int> this_associated_key = associated_key[trolloc];
      //   for(size_t tti =0 ; tti<this_associated_key.size();tti++)
      //   {
      //     int ti = this_associated_key[tti];
      //     if(temp_best_fits_movern_per_BK_per_SK.find( ti ) == temp_best_fits_movern_per_BK_per_SK.end())
      //     {
      //       temp_best_fits_movern_per_BK_per_SK[ti] = {best_fit_movern_for_basins[BK][trolloc]};
      //     }
      //     else
      //     {
      //       temp_best_fits_movern_per_BK_per_SK[ti].push_back(best_fit_movern_for_basins[BK][trolloc]);
      //     }
      //   }
      // }

      // best_fits_movern_per_BK_per_SK[BK] = temp_best_fits_movern_per_BK_per_SK;
      // std::cout << "flub" << std::endl;
      // xt::xtensor<float,1> temp_pyt = xt::adapt(disorder_vals);
      // std::cout << "flub1.5" << std::endl;
      disorder_all_values_by_movern[BK][movern[i]] = disorder_vals;

      // std::cout << "flub2" << std::endl;

    }

    

    if(verbose==true)
      std::cout << "OK" << std::endl;
  }

  for(size_t BK =0; BK< this_ordered_BL.size(); BK++)
  {
    disorder_movern_per_BK[BK] = movern_per_BK[BK];
  }

  best_fits_movern_per_BK = best_fit_movern_for_basins;

  std::map<float,int> movern_to_id2d;
  for(size_t i=0; i<movern.size();i++)
  {
    movern_to_id2d[movern[i]] = int(i);
  }
    // std::cout << "T1" << std::endl;

  normalised_disorder_val =  std::vector< std::vector<std::vector<float> > >(this_ordered_BL.size());
    // std::cout << "T2" << std::endl;

  for(auto i1 = disorder_all_values_by_movern.begin(); i1 != disorder_all_values_by_movern.end(); i1++)
  {

    // std::cout << "T3" << std::endl;
    std::vector<std::vector<float> > temp(i1->second[movern[0]].size());
    size_t o=0;
    for(auto v: i1->second[movern[0]])
    {
      std::vector<float> temp2(movern.size());
      temp[o] = temp2;
      o++;
    }

    // std::cout << "T4" << std::endl;

    for(auto i2 = i1->second.begin(); i2 !=i1->second.end(); i2++ )
    {
      for(size_t i3=0; i3<i2->second.size();i3++)
        temp[i3][movern_to_id2d[i2->first]] = i2->second[i3];
    }
    normalised_disorder_val[i1->first] = temp;
    // std::cout << "T5" << std::endl;

  }


  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();

  time_all += float(std::chrono::duration_cast<std::chrono::microseconds>(end_all - begin_all).count()) / 1000000;
  std::cout << "TIME TOT: " << time_all << " incl. TIME LOOP: " << time_loop << std::endl;


 
  // done
}


//####################################################################################
// Take two arrays of X and Y coordinates and return the corresponding raster values #
//####################################################################################
xt::pytensor<float,1> LSDDEM_xtensor::burn_rast_val_to_xy(xt::pytensor<float,1> X_coord,xt::pytensor<float,1> Y_coord)
{
  // Initial check, return nothing if XY dont match
  if(X_coord.size()!=Y_coord.size())
  {
    std::cout << "WARNING, CAnnot burn values as XY sizes do not match. (Match, burn ... get it?)" << std::endl;
    xt::pytensor<float,1> fake_return;
    // returning empty array which closes the function
    return fake_return;
  } 

  // Initialising the array of burned data
  std::array<size_t,1> shape = {X_coord.size()};
  xt::xtensor<float,1> output(shape);
  // Preparing the iteration
  int row,col;
  int n_point_out_of_raster = 0;
  // Iterating throught the points
  for(size_t i=0; i< X_coord.size(); i++)
  {
    // Convert the coordinates to XY, no snaping if the point is out
    this->XY_to_rowcol(X_coord[i], Y_coord[i], row, col, false);
    // Assigning no data by default
    float val = -9999;
    // Check if the point is in the raser
    if(row >= 0 && col >= 0 && row < nrows && col < ncols)
    {
      // and assign the value if it is
      val = PPRaster.get_data_element(row,col);
    }
    // warning statement which should not be tooooooo annoying
    else{n_point_out_of_raster++; std::cout << n_point_out_of_raster << " out of the DEM"; }
    // Assigning the value
    output[i] = val;
  }
  // Done, returning the output
  return output;
}

// Test function to return the perimeter of each basin, the "ridgeline"
std::map<int, std::map<std::string, xt::pytensor<float,1> > > LSDDEM_xtensor::extract_perimeter_of_basins()
{
  std::map<int, std::map<std::string, xt::pytensor<float,1> > > output;

  size_t n_bas = baselevel_junctions.size();
  for(size_t samp = 0; samp<n_bas; samp++)
  {
    
    std::cout << "Analysing catchment ..." << std::endl;
    LSDBasin thisBasin(baselevel_junctions[samp],FlowInfo, JunctionNetwork, baselevel_nodes_unique[samp]);
    std::cout << "Getting the perimeter ..." << std::endl;
    thisBasin.set_Perimeter(FlowInfo);
    std::vector<int> permineter_nodes = thisBasin.get_Perimeter_nodes();
    std::cout << "Got it, let me format the data" << std::endl;
    size_t n_node_in_this_perimeter = permineter_nodes.size();
    std::array<size_t,1> sadfkhkjogjtaoehgoaisdfgoih = {n_node_in_this_perimeter};
    xt::xtensor<float,1> X(sadfkhkjogjtaoehgoaisdfgoih), Y(sadfkhkjogjtaoehgoaisdfgoih), Z(sadfkhkjogjtaoehgoaisdfgoih);
    int this_row, this_col; float this_X, this_Y, this_elev;
    for(size_t huile_de_coude = 0; huile_de_coude < n_node_in_this_perimeter; huile_de_coude ++)
    {
      int this_node = permineter_nodes[huile_de_coude];
      FlowInfo.retrieve_current_row_and_col(this_node,this_row,this_col);
      FlowInfo.get_x_and_y_from_current_node(this_node,this_X,this_Y);
      X[huile_de_coude] = this_X;Y[huile_de_coude] = this_Y;Z[huile_de_coude]= PPRaster.get_data_element(this_row,this_col); 
    }

    std::map<std::string, xt::pytensor<float,1> > temp_ou;
    temp_ou["X"] = X;
    temp_ou["Y"] = Y;
    temp_ou["Z"] = Z;
    output[int(samp)] = temp_ou;
  }

  return output;
}


std::map<std::string, xt::pytensor<float,1> > LSDDEM_xtensor::get_sources_full()
{
  std::array<size_t,1> sizla = { sources.size() };
  xt::xtensor<float,1> outX(sizla), outY(sizla), outNodes(sizla);
  for(size_t i=0;i<sources.size();i++)
  {
    int this_node = sources[i];
    if(this_node <0)
    {  
          outX[i] = -9999;
          outY[i]=- 9999;
          outNodes[i] = -9999 ;
          continue;
    }
    outNodes[i] = float(this_node);
    float this_X,this_Y;
    FlowInfo.get_x_and_y_from_current_node(this_node,this_X,this_Y);
    outX[i] = this_X;
    outY[i]=this_Y;
  }
  std::map<std::string, xt::pytensor<float,1> > output;
  output["X"] = outX;
  output["Y"] = outY;
  output["nodeID"] = outNodes;
  return output;
}


void LSDDEM_xtensor::calculate_discharge_from_precipitation(int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& precdata, bool accumulate_current_node)
{

  // First step is to convert back the precipitation raster into LSDRaster
  std::cout << "I am ingesting the precipitation raster. I am drinking it I guess? Right? haha " << std::endl;
  LSDRaster precipitation_raster_original = LSDRaster(tnrows, tncols, txmin, tymin, tcellsize, tndv, xt_to_TNT(precdata, tnrows, tncols));
  std::cout << "Got it" << std::endl;

  // Potential recasting if extents are different
  if(precipitation_raster_original.get_NRows()!= nrows || precipitation_raster_original.get_NCols()!= ncols)
  {
    std::cout << "I now need to recast it to the right extent. I am just resampling vertically, if you need a more fancy method just do it yourself and ingest a raster to the right extent yo!" << std::endl;
    float goulg = 0;
    TNT::Array2D<float> GORIS(size_t(nrows),size_t(ncols),goulg);
    LSDRaster new_precipitation = LSDRaster(nrows, ncols, xmin, ymin, cellsize, NoDataValue, GORIS);; // copying the original raster
    for(size_t i=0; i<new_precipitation.get_NRows(); i++)
    for(size_t j=0; j<new_precipitation.get_NCols(); j++)
    {
      float x,y;
      new_precipitation.get_x_and_y_locations(i,j,x,y);
      float new_val = precipitation_raster_original.get_value_of_point(x,y);
      new_precipitation.set_data_element(i,j,new_val);
    }
    std::cout << "RECASTED!" << std::endl;
    precipitation_raster_original = new_precipitation;
  }

  std::cout << "Alright let's calculate the Discharge, it will take time" << std::endl;
  float dx = precipitation_raster_original.get_DataResolution();

  // volume precipitation per time precipitation times the cell areas
  precipitation_raster_original.raster_multiplier(dx*dx);

  // discharge accumulates this precipitation
  DrainageArea = FlowInfo.upslope_variable_accumulator_v2(precipitation_raster_original, accumulate_current_node);
  std::cout << "Done with that calculation! Note that I replace the drainage area raster!" << std::endl;

}

// Simple wrapper that returns xy coordinates from row and column
std::pair<xt::pytensor<float,1>,xt::pytensor<float,1> >LSDDEM_xtensor::query_xy_from_rowcol(xt::pytensor<int,1>& row, xt::pytensor<int,1>& col)
{
  // first simply initialise the output
  std::array<size_t,1> sizla = {row.size()};
  xt::xtensor<float,1> Xs(sizla), Ys(sizla);

  // then query my friend
  for(size_t i=0; i< row.size(); i++)
  {
    float tx,ty;
    PPRaster.get_x_and_y_locations(row[i],col[i],tx,ty);
    Xs[i] = tx; Ys[i] = ty;
  }

  std::pair<xt::pytensor<float,1>,xt::pytensor<float,1> > output =  std::make_pair(Xs,Ys);  
  return output;
}

std::map<std::string ,std::map<std::string, xt::pytensor<float,1> > > LSDDEM_xtensor::query_xy_for_each_basin()
{
  // std::cout << "N_basin_junction: " << baselevel_junctions.size() << " || N outlet_nodes " << outlet_nodes.size()  << " || baseevel nodes " << baselevel_nodes.size() << std::endl;
  
  // exit(EXIT_FAILURE);

  std::map<std::string ,std::map<std::string, xt::pytensor<float,1> > > output;

  // Going through the base_level nodes
  int this_BL = 0;
  // initialising a set to check for unique vlues
  std::set<int> mahset;
  for(auto BLnode : outlet_nodes)
  {
    auto checker = mahset.insert(BLnode);
    // Checker tells if the value has been succesfully inserted or if it already existed
    if(checker.second == false)
      continue;

    // std::cout << "processing the baselevel node : " << BLnode << std::endl;
    // preparing the output
    std::map<std::string, xt::pytensor<float,1> > temp;

    std::vector<int> all_nodes = FlowInfo.get_upslope_nodes_include_outlet(BLnode), these_X, these_Y;
    for(auto this_node : all_nodes)
    {
      int row,col; FlowInfo.retrieve_current_row_and_col(this_node, row,col);
      float tX, tY; PPRaster.get_x_and_y_locations(row,col, tX,tY);
      these_X.push_back(tX);
      these_Y.push_back(tY);
    }

    temp["X"] = xt::adapt(these_X);
    temp["Y"] = xt::adapt(these_Y);
    output[std::to_string(this_BL)] = temp;
    this_BL++;
  
  }


  return output; 
}

// Simple wrapper that returns xy coordinates from row and column
std::pair<xt::pytensor<int,1>,xt::pytensor<int,1> > LSDDEM_xtensor::query_rowcol_from_xy(xt::pytensor<float,1>& X, xt::pytensor<float,1>& Y)
{
  // first simply initialise the output
  std::array<size_t,1> sizla = {X.size()};
  xt::xtensor<int,1> row(sizla), col(sizla);

  // then query my friend
  for(size_t i=0; i< row.size(); i++)
  {
    int trow,tcol;
    this->XY_to_rowcol(X[i],Y[i],trow, tcol,true);
    row[i] = trow; col[i] = tcol;
  }

  std::pair<xt::pytensor<int,1>,xt::pytensor<int,1> > output =  std::make_pair(row,col);  
  return output;
}


std::map<std::string, xt::pytensor<float, 1> > LSDDEM_xtensor::get_FO_Mchi()
{
  std::map<std::string,  std::vector<int>  > out =  ChiTool.get_integer_vecdata_for_m_chi(FlowInfo);
  std::map<std::string,  xt::pytensor<float, 1>  > recasted_output;
  xt::pytensor<float, 1>  temp_vec_X,temp_vec_Y,temp_vec_FOMC;
  // first I am creating the arrays
  std::array<size_t,1> sizla = {out["nodeID"].size()};
  xt::xtensor<float,1> this_X(sizla),this_Y(sizla),this_FOMC(sizla), this_elev(sizla), this_DA(sizla), this_flow(sizla);
  for(size_t j = 0; j< out["nodeID"].size(); j++)
  {
    int this_node = out["nodeID"][j], row,col,recrow,reccol,recnode; FlowInfo.retrieve_current_row_and_col(this_node,row,col); FlowInfo.retrieve_receiver_information(this_node, recnode,recrow,reccol);
    float dz,dchi,tx,ty, TFOCM;
    if(this_node != recnode)
    {
      dchi = chi_coordinates.get_data_element(row,col) - chi_coordinates.get_data_element(recrow,reccol);
      dz = PPRaster.get_data_element(row,col) - PPRaster.get_data_element(recrow,reccol);
      this_FOMC[j] = dz/dchi;
    }
    else
    {
      this_FOMC[j] = 0;
    }        
    FlowInfo.get_x_and_y_locations(row,col,tx,ty);
    this_X[j] = tx;this_Y[j]=ty;
    this_elev[j] = PPRaster.get_data_element(row,col);
    this_DA[j] = DrainageArea.get_data_element(row,col);
    this_flow[j] = DistanceFromOutlet.get_data_element(row,col);
    temp_vec_X = this_X;
    temp_vec_Y = this_Y;
    temp_vec_FOMC = this_FOMC;
  }



  recasted_output["X"] = temp_vec_X;
  recasted_output["Y"] = temp_vec_Y;
  recasted_output["fo_m_chi"] = temp_vec_FOMC;
  recasted_output["DA"] = this_DA;
  recasted_output["elevation"] = this_elev;
  recasted_output["flow_distance"] = this_flow;

  return recasted_output;
}

void LSDDEM_xtensor::mask_topo(float value, xt::pytensor<float,2> masker)
{

  for(size_t i = 0; i<nrows; i++)
  {
    for(size_t j = 0; j<ncols; j++)
    {
      if(masker(i,j) == value)
        PPRaster.set_data_element(i,j,value);
    }
  }

}


std::tuple<std::vector<xt::pytensor<float,2> >, std::vector<std::map<std::string, float> > > LSDDEM_xtensor::get_individual_basin_raster()
{
  // formatting the outputs
  std::vector<xt::pytensor<float,2> > vec_1;
  std::vector<std::map<std::string, float> > vec_2;
  // getting the base levels
  for(size_t BK =0; BK< baselevel_nodes_unique.size(); BK++)
  {
    std::cout << "DEBUG::PROC BAS " << BK << " FOR EXTRACTION -> node: " << baselevel_nodes_unique[BK] << std::endl; 
    // First getting the basin
    LSDBasin that_basin(baselevel_junctions[BK], FlowInfo,  JunctionNetwork, baselevel_nodes_unique[BK]);
    // getting the array of topo values
    LSDRaster trimmed_rast = FlowInfo.get_raster_draining_to_node(baselevel_nodes_unique[BK], PPRaster);
    // Trimming the no data values
    // LSDRaster trimmed_rast = this_rast.RasterTrimmerPadded(30);
    // pushing in the output
    vec_1.push_back(this->LSDRaster_to_xt(trimmed_rast));
    // the saving method needs the following attribute: Z,x_min,x_max,y_min,y_max,res
    std::vector<float> datvec =  trimmed_rast.get_XY_MinMax();
    vec_2.push_back({{"x_min", datvec[0]}, {"y_min", datvec[1]}, {"x_max", datvec[2]}, {"y_max", datvec[3]}, {"res", trimmed_rast.get_DataResolution()} });
  }

  return {vec_1,vec_2};
}

xt::pytensor<float,2> LSDDEM_xtensor::LSDRaster_to_xt(LSDRaster& datrast)
{

  xt::pytensor<float,2> output = xt::zeros<float>({datrast.get_NRows(),datrast.get_NCols()});

  for(size_t i=0; i< datrast.get_NRows(); i++)
  {
    for(size_t j=0; j<datrast.get_NCols() ; j++)
    {
      output(i,j) = datrast.get_data_element(i,j);
    }
  }
  return output;
}


// This function requires the flow routines to be calculated. Then it uses the flow infos to extract a single river from a pre-defined source (X/Y coordinates)
// It returns a tuple of maps (dictionnaries in python) with the int and float info of each node:
// node ID, row, col, X, Y, elevation and drainage area
std::tuple<std::map<std::string,xt::pytensor<int,1> >, std::map<std::string,xt::pytensor<float,1> > > LSDDEM_xtensor::get_single_river_from_top_to_outlet(float upX, float upY)
{

  // getting the flow path
  std::vector<int> node_river = FlowInfo.get_flow_path(upX,upY);
  // How many nodes do I have
  int n_nodes_in_river = int( node_river.size());
  // Initialisng the outputs
  std::map<std::string,xt::pytensor<int,1> > mapint;
  std::map<std::string,xt::pytensor<float,1> > mapfloat;
  mapint["row"] = xt::zeros<int>({n_nodes_in_river});
  mapint["col"] = xt::zeros<int>({n_nodes_in_river});
  mapint["nodeID"] = xt::zeros<int>({n_nodes_in_river});
  mapfloat["X"] = xt::zeros<float>({n_nodes_in_river});
  mapfloat["Y"] = xt::zeros<float>({n_nodes_in_river});
  mapfloat["elevation"] = xt::zeros<float>({n_nodes_in_river});
  mapfloat["drainage_area"] = xt::zeros<float>({n_nodes_in_river});
  mapfloat["flow_distance"] = xt::zeros<float>({n_nodes_in_river});

  // Feeding the output
  for(size_t i =0; i<n_nodes_in_river; i++)
  {
    // local outputs
    int this_nodeID,this_row,this_col;
    float this_X,this_Y,this_elevation,this_drainage_area;

    //Gettign the nodeID
    this_nodeID = node_river[i];
    // row and col from node_ID
    FlowInfo.retrieve_current_row_and_col(this_nodeID,this_row,this_col);
    // And getting the rest
    FlowInfo.get_x_and_y_from_current_node(this_nodeID,this_X,this_Y);
    this_elevation = PPRaster.get_data_element(this_row,this_col);
    this_drainage_area = DrainageArea.get_data_element(this_row,this_col);
    float this_flow_distance = DistanceFromOutlet.get_data_element(this_row,this_col);
    mapint["row"][i] = this_row;
    mapint["col"][i] = this_col;
    mapint["nodeID"][i] = this_nodeID;
    mapfloat["X"][i] = this_X;
    mapfloat["Y"][i] = this_Y;
    mapfloat["elevation"][i] = this_elevation;
    mapfloat["drainage_area"][i] = this_drainage_area;
    mapfloat["flow_distance"][i] = this_flow_distance;

    // next iterations
  }
  return {mapint,mapfloat};
}


// THis function is designed to extract the D8 receiver informations from a list of X and Y coordinate
std::map<std::string ,xt::pytensor<float,1> > LSDDEM_xtensor::get_receiver_data(xt::pytensor<float,1>& X, xt::pytensor<float,1>& Y)
{

  // formatting the output
  xt::xtensor<float,1> rec_flowdist = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_elev = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_chi = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_DA = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_X = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_Y = xt::zeros<float>({X.size()});
  xt::xtensor<float,1> rec_node = xt::zeros<float>({X.size()});

  // Noot Noooooot
  for(size_t i=0; i<X.size(); i++)
  {
    // Getting X and Y coordinates
    float tX = X[i], tY = Y[i];
    int row, col; XY_to_rowcol(tX,tY,row,col,false);
    int recrow,reccol,node_ID = FlowInfo.get_NodeIndex_from_row_col(row, col), recnode;
    FlowInfo.retrieve_receiver_information(node_ID,recnode,recrow,reccol);
    rec_flowdist[i] = DistanceFromOutlet.get_data_element(recrow,reccol);
    rec_elev[i] = PPRaster.get_data_element(recrow,reccol);
    rec_chi[i] = chi_coordinates.get_data_element(recrow,reccol);
    rec_DA[i] = DrainageArea.get_data_element(recrow,reccol);
    float recX,recY; FlowInfo.get_x_and_y_from_current_node(recnode,recX, recY);
    rec_X[i] = recX;
    rec_Y[i] = recY;
    rec_node[i] = recnode;

  }
  std::map<std::string ,xt::pytensor<float,1> > output;
  output["receiver_flow_distance"] = rec_flowdist;
  output["receiver_elevation"] = rec_elev;
  output["receiver_drainage_area"] = rec_DA;
  output["receiver_chi"] = rec_chi;
  output["receiver_X"] = rec_X;
  output["receiver_Y"] = rec_Y;
  output["receiver_node"] = rec_node;


  return output;


}


std::tuple<std::map<std::string, xt::pytensor<int,1> >, std::map<std::string, xt::pytensor<float,1> > > LSDDEM_xtensor::get_SA_from_vertical_interval(float vertical_interval)
{
  
  // Chi and flowinfo objects need to have been calculated before
  std::vector<int> midpoint_nodes;
  std::vector<float> SA_slopes;
  // Getting the slope
  ChiTool.get_slope_area_data(FlowInfo,  vertical_interval, midpoint_nodes,  SA_slopes);

  // Getting curent keys
  auto mahmaps =  ChiTool.get_current_map_basin_source_key();
  auto SKs = mahmaps[0];
  auto BKs = mahmaps[1];

  // Fromatting the output
  std::map<std::string, xt::pytensor<int,1> > map1 = { {"nodeID", xt::zeros<int>({SA_slopes.size()})} , {"row",xt::zeros<int>({SA_slopes.size()})}, {"col",xt::zeros<int>({SA_slopes.size()})}, {"source_key",xt::zeros<int>({SA_slopes.size()})},{"basin_key",xt::zeros<int>({SA_slopes.size()})}};
  std::map<std::string, xt::pytensor<float,1> >map2 = { {"elevation",xt::zeros<float>({SA_slopes.size()})}, {"x",xt::zeros<float>({SA_slopes.size()})}, {"y",xt::zeros<float>({SA_slopes.size()})}, {"slope",xt::zeros<float>({SA_slopes.size()})}, {"drainage_area",xt::zeros<float>({SA_slopes.size()})}};

  for( size_t i =0; i< midpoint_nodes.size(); i++ )
  {
    int this_node = midpoint_nodes[i];
    int row,col; FlowInfo.retrieve_current_row_and_col(this_node,row,col);
    float x,y; FlowInfo.get_x_and_y_from_current_node(this_node,x,y);
    map1["nodeID"][i] = this_node;
    map1["row"][i] = row;
    map1["col"][i] = col;
    map1["source_key"][i] = SKs[this_node];
    map1["basin_key"][i] = BKs[this_node];
    map2["elevation"][i] = PPRaster.get_data_element(row,col);
    map2["x"][i] = x;
    map2["y"][i] = y;
    map2["slope"][i] = SA_slopes[i];
    map2["drainage_area"][i] = DrainageArea.get_data_element(row,col);

  }


  return {map1,map2};

}

xt::pytensor<bool,1> LSDDEM_xtensor::trim_nodes_draining_to_baselevel(int base_level_node, xt::pytensor<int,1> test_nodes)
{
  // Initialising the output
  xt::xtensor<bool,1> out = xt::zeros<bool>({test_nodes.size()});
  // Getting all the nodes draining at some points to the one
  std::vector<int> nodebas = FlowInfo.get_upslope_nodes(base_level_node);

  // mapping them:
  std::map<int,bool> menodes;
  for(size_t i=0; i<test_nodes.size(); i++)
    menodes[test_nodes[i]] = false;
  for(size_t i=0; i<nodebas.size(); i++)
     menodes[nodebas[i]] = true;

  //FLUB!
  for (size_t i =0; i < test_nodes.size();i++)
  {
    if(menodes[test_nodes[i]] == true)
      out[i] = true;
    else
      out[i] = false;
  }

  return out; 
   
}

//####################################################################
// Converts XY coordinates to their row/col equivalent.
// Possibility to snap to the closest raster edge if the coordinate 
// falls outside of the DEM (usefull with approximative coord).
// Authors: B.G.
//#####################################################################
void LSDDEM_xtensor::XY_to_rowcol(float X_coord, float Y_coord, int& row, int& col, bool snap_to_closest_point_on_raster)
{
  // I need to keep in mind if I recasted the coordinates
  bool is_X_recasted = false, is_Y_recasted = false;
  // recasteing point x
  // ## First I change the referential if my minimum coordinates are <0
  if(xmin<0)
    {X_coord = X_coord + -1 * xmin; xmin = 0;}
  // Checking the coordinate
  float corrected_X = X_coord - xmin;
  // std::cout << "X_corr is " << corrected_X << std::endl;

  // Checking if the point is W of the DEM
  if(corrected_X<0)
  {
    if(snap_to_closest_point_on_raster)
    {
      // not in the raster -> recasting to col 0
      is_X_recasted = true;
      std::cout << "WARNING::X coordinate is West to the X minimum by " << corrected_X << ", I am recasting to column 0" << std::endl;
      col = 0;
    }
    else
    {
      // I crash. I need to find alternative for that, Jupyter really does not like it.
      std::cout << "FATALERROR::Point offset from the raster by " << corrected_X << " unit West." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Now checking if the point is E of the DEM
  else if(corrected_X > (ncols+1) * cellsize)
  {
    if(snap_to_closest_point_on_raster)
    {
      // Recasting to last col if needed
      is_X_recasted = true;
      std::cout << "WARNING::X coordinate is East to the X minimum by " << corrected_X - ((ncols+1) * cellsize) << ", I am recasting to column " << ncols - 1 << std::endl;
      col = ncols - 1;
    }
    else
    {
      std::cout << "FATALERROR::Point offset from the raster by " << corrected_X - ((ncols+1) * cellsize) << " unit East." << std::endl;
      exit(EXIT_FAILURE);
    }
  }


  // Exact same process with the Y coordinate, apart from the conversion which needs to be inverted as Y and row are going in opposite directions
  if(ymin<0)
    {Y_coord = Y_coord + -1 * ymin; ymin = 0;}
  // Correcting the coord
  float corrected_Y = Y_coord - ymin;
  // std::cout << "Y_corr is " << corrected_Y << std::endl;

  // Checking if the point is South of the DEM
  if(corrected_Y < 0)
  {
    if(snap_to_closest_point_on_raster)
    {
      is_Y_recasted = true;
      std::cout << "WARNING::Y coordinate is South to the Y minimum by " << - corrected_Y << ", I am recasting to row " << nrows - 1 << std::endl;
      row = nrows - 1;
    }
    else
    {
      std::cout << "FATALERROR::Point offset from the raster by " << - corrected_Y << " unit West." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Checking if the point is North of the DEM
  else if(corrected_Y > (nrows+1) * cellsize)
  {
    if(snap_to_closest_point_on_raster)
    {
      is_Y_recasted = true;
      std::cout << "WARNING::X coordinate is North to the Y maximum by " <<  (nrows+1) * cellsize - corrected_Y << ", I am recasting to row " << 0 << std::endl;
      row = 0;
    }
    else
    {
      std::cout << "FATALERROR::Point offset from the raster by " <<   ((nrows+1) * cellsize) - corrected_Y << " unit East." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Done if I recasted my coordinates
  if(is_Y_recasted && is_X_recasted)
    return;

  // Otherwise I need to convert them to row col
  if(is_X_recasted == false)
  {
    col = floor(corrected_X/cellsize);
  }
  if(is_Y_recasted == false)
  {
    row = (nrows - 1) - floor(corrected_Y/cellsize);
  }

  // No need to return anything: the row/col are passed as adresses
}










#endif
