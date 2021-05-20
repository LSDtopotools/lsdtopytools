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
#ifndef LSDDEM_xtensor_HPP
#define LSDDEM_xtensor_HPP

// STL imports
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include <queue>
#include <iostream>
#include <numeric>
#include <cmath>

// All the LSDTopoTools header used in the python bindings
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
#include "LSDStatsTools.hpp"
#include "TNT/tnt.h"

// Manages the multithreaded parts of the code
#include <omp.h>

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same


///@brief The LSDDEM_xtensor is a direct binder to LSDTopoTools by provinding a OO class
///@brief directly callable from python via in-memory numpy arrays.
///@authors B.G 
///@Date 2018 - 2020
class LSDDEM_xtensor
{

  // Public functions callable from the outer world
	public:

  //~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#//
  // Functions to ingest the base raster and translate it to LSDTopoTools - I/O  #~#~#~#~#~#//
  //~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#//

	///@brief Default constructor, only used if integrated into other objects. Would probably not be usefull otherwise.
  ///@authors B.G 
  ///@Date At some points in 2018
	LSDDEM_xtensor() { create(); }

  ///@brief This constructor takes few geographical informations about the DEM and ingest it into LSDTopoTools object
  ///@param tnrows (int): number of rows
  ///@param tncols (int): number of cols
  ///@param txmin (float): minimum X coordinate (left side)
  ///@param txmax (float): maximum X coordinate (right side)
  ///@param tymin (float): minimum y coordinate (top side (I tend to be confused with that ...) )
  ///@param tymax (float): maximum y coordinate (bottom side (I tend to be confused with that ...))
  ///@param tndv (float): NoData Value, Needs to be recasted to one ndv value beforehand (I could craft some solution for that if needed but so far it is managed in python)
  ///@param data (numpy 2D array of float32 or 64): the 2D array of values
  ///@returns an LSDDEM object
  ///@authors B.G 
  ///@Date At some same point in 2018
  LSDDEM_xtensor(int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& data){ create(tnrows, tncols, txmin, tymin, tcellsize, tndv,data);}

  ///@brief Manually switches the DEM to preprocessed mode, as the DEM deadlocks fluvial routine if not preprocessed explicitely
  ///@authors B.G
  ///@date November 2018
  void already_preprocessed();

  ///@brief Quite obscur to me at the moment, see LSDTT documentation for details.
  ///@details TO DO: explore how to force periodic boundaries when investigating LEMs output!
  ///@authors B.G
  ///@date 2018
  void set_boundary_conditions(std::vector<std::string> flab) {boundary_conditions = flab;};

  // Conversion functions
  // TODO: find a way to convert xtensors to TNT arrays pointing to the same underlying data
  // TODO: Also find a way to create a xtensor from TNT data without having to copy it 
  
  ///@brief Convert an xtensor to a TNT array
  ///@brief OLD VERSION, ONGOING WORK TO AVOID COPY.
  ///@date 27/12/2019
  ///@authors B.G
  TNT::Array2D<float> xt_to_TNT(xt::pytensor<float,2>& myarray, size_t tnrows, size_t tncols);
  
  ///@brief Convert an TNT array to xtensor usable by numpy
  ///@brief OLD VERSION, ONGOING WORK TO AVOID COPY.
  ///@date 27/12/2019
  ///@authors B.G
  xt::pytensor<float,2> LSDRaster_to_xt(LSDRaster& datrast);





  //~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#//
  // Binding functions to LSDTopotools Preprocessing and flow routines  ~#~#~#~#~#~#~#~#~#~#//
  //~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#//


  ///@brief Preprocess the raster to ensure flow routine with various options
  ///@detail So far, uses carve and/or fill algorithms (in that order) to ensure flow routing to DEM edges.
  ///@detail Future implementation will add an hybrid carving directly imposing a minimal slope (significant speedup) 
  ///@detail and algorithms from Schwagarht et al., 2017 to process bumps in river profiles.
  ///@detail Filling algorithm will be replaced by Barnes' priority flood at some points
  ///@param carve (boolean): if True, will use Lindsay (2016) to carve the depressions using an implementation from RichDEM
  ///@param fill (boolean): if True, will use Wang and Liu (2006) to impose a minimum slope draining to the edge.
  ///@authors B.G
  ///@date 2018
  void PreProcessing(bool carve, bool fill, float min_slope_for_fill);

  ///@brief Calculate the FlowInfo object (containing DA, flow direction, stack, node indices anipulations, ...)
  ///@Details Basic wrapper: D8 flow direction and drainage area from topography (O'Callaghan)
  ///@authors B.G
  ///@date 2018
  void calculate_FlowInfo();

  ///@brief Calculate the FlowInfo object with a Dinf method (containing DA, flow direction, stack, node indices anipulations, ...)
  ///@Details Basic wrapper: Dinf drainage area from topography, rest is from D8 
  ///@authors B.G
  ///@date 2018
  void calculate_FlowInfo_Dinf();

  ///@brief Calculate the source location based on different methods, this step calculates the channel head for the WHOLE landscape prior to
  ///@brief selecting the watershed. Different methods will be implemented, so far only area threshold is implemented (Others requires obscure libraries)
  ///@param method (string): name of teh targetted method, so far "area_threshold"
  ///@param min_contributing_pixels (float): for the area threshold method, determine the number of draining pixels required to initiate a source
  ///@returns Nothing, initiate the source and source_clean vector attributes
  ///@authors B.G
  ///@date 2018
  void calculate_channel_heads(std::string method ,int min_contributing_pixels);
  
  ///@brief Ingest sources/channel heads calculated from other means. 
  ///@brief Takes a numpy array of node index defining the source location, typiicaly from the "node"
  ///@brief column in a csv file calculated with LSDTopoTOols (e.g. DreiCH method)
  ///@brief REQUIREMENT: needs the raster to be preprocessed (or flagged as preprocessed) to have correct flow routines
  ///@brief REQUIREMENT: Needs the FlowInfo to be calculated
  ///@param these_sources (numpy array of integers): numpy array of sources node indices 
  ///@returns Nothing, but get the attribute right
  ///@authors B.G
  ///@date 2018
  void ingest_channel_head(xt::pytensor<int,1>& these_sources);

  ///@brief Initiate the JunctionNetwork Object, the LSDTopoTools object dealing with river network
  ///@brief REQUIREMENT: needs the source vectors to be initialised with its own requirements
  ///@returns Nothing, initialise the JunctionNetwork Attribute
  ///@authors B.G
  ///@date 2018
  void calculate_juctionnetwork();

  ///@brief Isolate the river network to a selection of watershed based on the xy location of the outlet (snap to junction).
  ///@brief DEPRECATED!! I keep it for backward compatibility, see _v2 for more accurate version
  ///@param Not explicated to force the use of the new version.
  ///@authors B.G 
  ///@date 2018
  void calculate_outlets_locations_from_xy(xt::pytensor<float,1>& x_coord_BL, xt::pytensor<float,1>& y_coord_BL, int search_radius_nodes,int threshold_stream_order, bool test_drainage_boundary, bool only_take_largest_basin);
  

  ///@brief Uses Mudd et al 2014's profile segmentation algorithm to find the best-fit segmenst of flow-distance elevation profiles. 
  ///@brief It's clean method to get channel slope and determine slope patches sensu Royden 2013.
  ///@param target_nodes (int): Determine, along with n_skip, the length of the created segments
  ///@param n_iterations (int): Number of MC iterations
  ///@param skip (int): influence, along with the target nodes, the size of te created segments
  ///@param minimum_segment_length (int): minimum number of nodes to create a segment
  ///@param sigma (float): see paper for full explanation
  ///@param nthreads (int): Experimental feature for multithreading. Usually segfault, sometimes work, there is probably a race issue somewhere. Leave to 1 to avoid.
  ///@return: a pair of map/dictionnaries with information about rivers: DA, S, row col, source key, basin key, elevation, fow distance ...
  ///@authors B.G  
  std::pair<std::map<std::string, xt::pytensor<float,1> > , std::map<std::string, xt::pytensor<int,1> > > get_channel_gradient_muddetal2014(int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma,int nthreads);

  ///@brief Original Mudd et al., 2014 ksn calculation method.
  ///@brief See above for details, segment chi-Elevation profile rather than long profiles
  ///@return: Nothing but calculate two map/dictionnary with river attribute, including chi, ksn called m_chi and segmented elevation
  ///@authors: B.G
  void generate_ksn(int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma, int nthreads);

  ///@brief Calculates chi coordinate for selected watersheds.
  ///@param: theta (float) the reference concavity index
  ///@param: A0, integration constant, keep proportionality between chi--elevation profiles and ksn if = 1
  ///@return: Nothing, but chi is calculated after that and ready to be used/extracted by other functions
  ///@Authors: B.G.
  void generate_chi(float theta, float A_0);
  
  ///@brief Calculate asimpler version of k_sn by direct derivation of chi--elevation profiles node to node.
  ///@brief Recommended for simulated landscapes, as results are very noisy on stndards DEMs
  ///@returns: two maps/dictionnaries with river information, including ksn
  ///@authors: B.G
  std::pair<std::map<std::string, xt::pytensor<float,1> > , std::map<std::string, xt::pytensor<int,1> > > get_ksn_first_order_chi();

  ///@brief Determine investigated watersheds by selecting them from their minimum size
  ///@param minimum_DA (double): the minimum Drainage area required to generate a basin
  ///@param test_drainage_boundary (bool): if true, will ignore any basin  potentially influenced by No Data and therefore beheaded. Deactivate if you trust your dem
  ///@param prune_to_largest (bool): only keep the largest basin
  ///@return: nothing. but the code now knows which watershed to investigate
  ///@authors: B.G
  std::map<std::string, std::vector<float> > calculate_outlets_locations_from_minimum_size(double minimum_DA, bool test_drainage_boundary, bool prune_to_largest);
  
  ///@brief Similar function than above calculate_outlets_locations_from_minimum_size. But from a range of drainage area
  ///@param See above + max_DA (double): the maximum drainage area to define a basin
  ///@retunrs: nothing but get the code ready to work on specific watersheds
  ///@authors: B.G
  std::map<std::string, std::vector<float> > calculate_outlets_locations_from_range_of_DA(double minimum_DA, double max_DA, bool check_edges);
  
  ///@brief Determine the study area by selecting the largest watershed.
  ///@param test_drainage_boundary (bool): only select the largest watershed not affected by nodata
  ///@retunrs: nothing but get the code ready to work on specific watersheds
  ///@authors: B.G
  std::map<std::string, float> calculate_outlet_location_of_main_basin(bool test_drainage_boundary);
  
  ///@brief Run knickpoint detection algorithm from Gailleton et al., 2019
  ///@brief REQUIREMENT: having calculated chi and ksn
  ///@param MZS_th (float): DEPRECATED, just put random value
  ///@param lambda_TVD (float): regulatory parameter to run the Total Variation Denoising form Condat 2013.
  ///@param stepped_combining_window (int): DEPRECATED
  ///@param window_stepped (int): size of the window used to detect stepped knickpoints
  ///@param n_std_dev (float): sensibility, in number of std deviation, of the window detecting stepped knickpoint
  ///@param kp_node_search (int): number of node that will combine adjacent knickpoints together
  ///@returns Nothing, but knickpoint database is generated
  ///@authors: B.G.
  void detect_knickpoint_locations( float MZS_th, float lambda_TVD,int stepped_combining_window,int window_stepped, float n_std_dev, int kp_node_search);
  
  ///@brief Main function running the concavity analysis with Disorder method from Mudd et al., 2018 and the improved version from Gailleton et al., in prep
  ///@param start_movern (float): lowest theta value to try, typically 0.05
  ///@param delta_movern (float): step between analysis, typically 0.025
  ///@param n_movern (float): number of steps
  ///@param area_threshold: DEPRECATED
  ///@return: Nothing but results for movern are calculated
  ///@authors: B.G.
  void calculate_movern_disorder(float start_movern, float delta_movern, float n_movern, float A_0, float area_threshold, int n_rivers_by_combination);
  
  ///@brief Brute-force function to select all the outlets fr analysis, conditionless.
  ///@brief Only recomended for debugging
  ///@authors B.G.
  void force_all_outlets(bool test_drainage_boundary);

  ///@brief Accessor for the chi raster
  xt::pytensor<float,2> get_chi_raster();

  ///@brief Accessor for the basin raster
  xt::pytensor<int,2> get_chi_basin();

  ///@brief Accessor for the basin raster
  xt::pytensor<float,2> get_flow_distance_raster();

  ///@brief Accessor for ksn_data (integers)
  std::map<std::string, xt::pytensor<int,1> > get_int_ksn_data();
  
  ///@brief Accessor for ksn_data (float)
  std::map<std::string, xt::pytensor<float,1> > get_float_ksn_data();

  ///@brief Accessor for knickpoint data (integers)
  std::map<std::string, xt::pytensor<int,1> > get_int_knickpoint_data();
  
  ///@brief Accessor for knickpoint data (float)
  std::map<std::string, xt::pytensor<float,1> > get_float_knickpoint_data();

  ///@brief Generates a hillshade for visualisation purposes. Simulated lighting on the DEM
  ///@param hs_altitude (float): altitude of the light
  ///@param hs_azimuth (float): azimuth of the light
  ///@param hs_z_factor (float): exageration factor
  ///@returns: a numpy 2D array to the extent of the base DEM, with hillshaded values
  ///@authors: B.G.
  xt::pytensor<float,2> get_hillshade(float hs_altitude, float hs_azimuth, float hs_z_factor);
  
  ///@brief: Similar function, but with custom array having the same extents than the base DEM.
  xt::pytensor<float,2> get_hillshade_custom(xt::pytensor<float,2>& datrast, float hs_altitude, float hs_azimuth, float hs_z_factor);

  ///@brief Calculate polyfit window metrics (Hurst et al., 2014)
  ///@param window (float): the size (in map units) of the window used to calculate the polyfits
  ///@param vector selecao: the selection of metrics: 
  ///        0 -> Elevation (smoothed by surface fitting)
  ///        1 -> Slope
  ///        2 -> Aspect
  ///        3 -> Curvature
  ///        4 -> Planform Curvature
  ///        5 -> Profile Curvature
  ///        6 -> Tangential Curvature
  ///        7 -> Stationary point classification (1=peak, 2=depression, 3=saddle)
  ///@returns: a vector of selected raster
  ///@authors: B.G.
  std::vector<xt::pytensor<float,2> > get_polyfit_on_topo(float window, std::vector<int> selecao);
  
  ///@brief Get ksn metrics calculted from a temporary/external chi tools. Mostly for internal use.
  ///@authors B.G.
  std::map<std::string, xt::pytensor<int,1> > get_int_ksn_data_ext_ChiTools(LSDChiTools& this_Chitools);
  std::map<std::string, xt::pytensor<float,1> > get_float_ksn_data_ext_ChiTools(LSDChiTools& this_Chitools);


  ///@brief Extract the perimeter of previously defined catchments.
  ///@returns A map (key: basin key) of map (key: attribute name) of perimeter attribute (X,Y,Z,...)
  ///@authors B.G.
  std::map<int, std::map<std::string,xt::pytensor<float, 1> > > get_catchment_perimeter();

  // Accessor to the different raster tpe data
  xt::pytensor<float,2> get_base_raster(){std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};xt::xtensor<float,2> output(shape);for(size_t i=0;i<size_t(nrows);i++)for(size_t j=0;j<size_t(ncols);j++){output(i,j) = BaseRaster.get_data_element(i,j);}return output;}
  xt::pytensor<float,2> get_PP_raster(){std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};xt::xtensor<float,2> output(shape);for(size_t i=0;i<size_t(nrows);i++)for(size_t j=0;j<size_t(ncols);j++){output(i,j) = PPRaster.get_data_element(i,j);}return output;}
  xt::pytensor<float,2> get_DA_raster(){std::array<size_t,2> shape = {size_t(nrows),size_t(ncols)};xt::xtensor<float,2> output(shape);for(size_t i=0;i<size_t(nrows);i++)for(size_t j=0;j<size_t(ncols);j++){output(i,j) = DrainageArea.get_data_element(i,j);}return output;}

  // setter and modifier of the different datasets
  // This one sets a new drainage area raster that needs to be EACTLY the same dimensions that the topo one.
  xt::pytensor<float,2> set_DA_raster(xt::pytensor<float,2> newDA){TNT::Array2D<float> tdata = xt_to_TNT( newDA, nrows, ncols); LSDRaster temp = BaseRaster; temp.set_data_array(tdata); DrainageArea = temp;}

  // Accessor to more simple data (direct access)
  std::map<int,vector<float> > get_disorder_dict() {return disorder_movern_per_BK;};
  std::map<int, std::map<int,vector<float> > > get_disorder_dict_SK() {return best_fits_movern_per_BK_per_SK;};

  std::vector<float> get_disorder_vec_of_tested_movern() {return associated_movern_disorder;};


  // Burning and external data routines
  xt::pytensor<float,1> burn_rast_val_to_xy(xt::pytensor<float,1> X_coord,xt::pytensor<float,1> Y_coord);



  std::map<std::string, xt::pytensor<float,1> > get_sources_full();



  LSDRaster return_fake_raster();
  LSDIndexRaster return_fake_indexraster();
  LSDFlowInfo return_fake_flowinfo();
  LSDChiTools return_fake_chitool();
  LSDJunctionNetwork return_fake_junctionnetwork();

  std::map<int, std::map<std::string, xt::pytensor<float,1> > > extract_perimeter_of_basins();
  void calculate_discharge_from_precipitation(int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& precdata, bool accumulate_current_node);


  ///@brief This function takes a numpy array of row and col and returns the corresponding X and Y coordinates in the base raster system.
  ///@detail Such a function is really useful when doing some calculation in python on the base raster that may require to be translated into XY
  ///@param row (numpy array or xtensor): the row indices
  ///@param col (numpy array or xtensor): the col indices
  ///@return a pair of xy coordinates array
  ///@authors B.G.
  std::pair<xt::pytensor<float,1>,xt::pytensor<float,1> > query_xy_from_rowcol(xt::pytensor<int,1>& row, xt::pytensor<int,1>& col);

  std::pair<xt::pytensor<int,1>,xt::pytensor<int,1> > query_rowcol_from_xy(xt::pytensor<float,1>& X, xt::pytensor<float,1>& Y);



  ///@brief This function returns a map/dictionnary of the stack order and inverted stack order sensus Braun and Willett 2013
  ///@brief All the remianing blathering is jsut the conversion of data type from vectors to numpy
  ///@authors B.G.
  std::map<std::string, std::vector< xt::pytensor<int, 1> > > get_fastscape_ordering()
  {
    std::map<std::string, std::vector< std::vector<int> > > out =  FlowInfo.get_map_of_vectors();
    std::map<std::string, std::vector< xt::pytensor<int, 1> > > recasted_output;
    for(std::map<std::string, std::vector< std::vector<int> > >::iterator GOOOOORIS = out.begin(); GOOOOORIS != out.end(); GOOOOORIS++)
    {
      std::string tis_entry = GOOOOORIS->first;std::vector< std::vector<int> > vectors_of_nodes = GOOOOORIS->second;
      std::vector<xt::pytensor<int,1> > temp_array;
      for(size_t j = 0; j<vectors_of_nodes.size(); j++)
      {
        // std::array<size_t,1> sizla = {vectors_of_nodes[j].size()};
        // xt::xtensor<int,1> tout(sizla);
        auto tout = xt::adapt(vectors_of_nodes[j]);
        temp_array.push_back(tout);
      }
      recasted_output[tis_entry] = temp_array;

    }
    return recasted_output;
  }

  std::map<std::string, xt::pytensor<float, 1> > get_FO_Mchi();

  void mask_topo(float value, xt::pytensor<float,2> masker);
  
  std::map<int, std::vector<float> > get_best_fits_movern_per_BK() {return  best_fits_movern_per_BK;};

  std::tuple<std::vector<xt::pytensor<float,2> >, std::vector<std::map<std::string, float> > > get_individual_basin_raster();


  // Set a new value for the n_nodes to visit downstream
  void set_n_nodes_to_visit_downstream(int val){n_nodes_to_visit_downstream = val;}

  ///@brief  
  std::tuple<std::map<std::string,xt::pytensor<int,1> >, std::map<std::string,xt::pytensor<float,1> > > get_single_river_from_top_to_outlet(float upX, float upY);


  std::map<std::string ,xt::pytensor<float,1> > get_receiver_data(xt::pytensor<float,1>& X, xt::pytensor<float,1>& Y);
  xt::pytensor<float,2> get_chi_raster_all(float m_over_n, float A_0, float area_threshold);

  std::tuple<std::map<std::string, xt::pytensor<int,1> >, std::map<std::string, xt::pytensor<float,1> > > get_SA_from_vertical_interval(float vertical_interval);

  xt::pytensor<bool,1> trim_nodes_draining_to_baselevel(int base_level_node, xt::pytensor<int,1> test_nodes);

  void calculate_outlets_locations_from_xy_v2(xt::pytensor<float,1>& tx_coord_BL, xt::pytensor<float,1>& ty_coord_BL, int n_pixels, bool check_edges);

  void calculate_outlets_locations_from_nodes_v2(xt::pytensor<int,1>& target_nodes, int n_pixels, bool check_edges);

  std::map<std::string,xt::pytensor<float,1> > extract_basin_draining_to_coordinates(xt::pytensor<float,1> input_X,xt::pytensor<float,1> input_Y, double min_DA);

  std::map<std::string, xt::pytensor<float,1> > calculate_outlets_min_max_draining_to_baselevel(float X, float Y, double min_DA, double max_DA, int n_pixels_to_chek);

  std::map<std::string ,std::map<std::string, xt::pytensor<float,1> > > query_xy_for_each_basin();

  std::tuple<std::vector<std::vector<int> >, std::vector<std::vector<float> > > get_DEBUG_all_dis_SK(){return {DEBUG_get_all_dis_SK_SK,DEBUG_get_all_dis_SK_val};};

  std::map< float, std::vector<float> > get_raw_disorder_per_SK(){return raw_disorder_per_SK;};

  std::vector< std::vector<std::vector<float> > > get_all_disorder_values(){return normalised_disorder_val;}
  std::vector< std::vector<int> > get_n_pixels_by_combinations(){ return normalised_disorder_n_pixel_by_combinations;};

  void calculate_baselevel_nodes_unique();

  std::map<std::string, xt::pytensor<float,1> > get_baselevels_XY();


  ///@brief Translate XY coordinates to row, col
  ///@brief Somehow works eratically with previous versions so I made mine
  ///@param X_coord (float): the x coordinate
  ///@param Y_coord (float): the y coordinate
  ///@param row (int): will get the value of the row
  ///@param col (int): will get the value of the col
  ///@param snap_to_closest_point_on_raster: if true and if the points is outside the raster, it snaps it t the closest boundary
  void XY_to_rowcol(float X_coord, float Y_coord, int& row, int& col, bool snap_to_closest_point_on_raster);




  protected:
  // Attributes are going there

   
  // Scalar attributes 
  //### Base raster attribute
  float NoDataValue;
  float cellsize;
  int nrows;
  int ncols;
  float xmin;
  float ymin;



  // Boundary conditions: {N,E,S,W}
  std::vector<std::string> boundary_conditions;
  bool default_boundary_condition;

  //### Minimum threshold for source
  float min_cont;

  // Internal chacker
  bool is_preprocessed;
  bool verbose;


  // 2D arrays and vectors
  //### This is the base raster
  LSDRaster BaseRaster;
  //### This is the Raster after preprocessing: fill, breach, filters, ...
  LSDRaster PPRaster;
  //### Flow accumulation raster
  LSDIndexRaster FlowAcc;
  //### Drainage Area 
  LSDRaster DrainageArea;
  //### Flowinfo
  LSDFlowInfo FlowInfo;
  //### Distance from outlet
  LSDRaster DistanceFromOutlet;
  //### Basin Junctions
  LSDIndexRaster BasinArrays;
  //### ChiTool object
  LSDChiTools ChiTool;
  //### Chi values as a raster
  LSDRaster chi_coordinates;

  // LSD specific objects
  //### Raw river network (not trimmed to basins)
  LSDJunctionNetwork JunctionNetwork;


  // 1D arrays and vectors
  //### contains the node index (calculated by LSDFlowInfo) of each channel heads
  std::vector<int> sources; // contains the node index (calculated by LSDFlowInfo) of each channel heads
  std::vector<int> sources_clean; // contains the node index (calculated by LSDFlowInfo) of each channel heads
  //### contains the node index (calculated by LSDFlowInfo) of each baselevels in CHitools
  std::vector<int> baselevel_junctions;
  std::vector<int> baselevel_nodes;
  std::vector<int> baselevel_nodes_unique;
  std::vector<int> outlet_nodes;



  // Fastscape_node_ordering
  xt::xtensor<int,1> FastScape_Node_Ordering_receivers;
  xt::xtensor<int,1> FastScape_Node_Ordering_di;
  xt::xtensor<int,1> FastScape_Node_Ordering_delta_i;  
  std::vector<int> FastScape_Node_Ordering_base_levels;
  int FastScape_Node_Ordering_nthreads;
  std::vector<std::vector<int> > FastScape_Node_Ordering_stack;


  // Other parameters linked to river extraction
  // The chi map starts with the longest channel in each basin and then works in tributaries in sequence. 
  //This parameter determines how many nodes onto a receiver channel you want to go when calculating the chi slope at the bottom of a tributary.
  int n_nodes_to_visit_downstream;


  // Cordonnier-processing
  // # basin-raw-ID -> bool
  std::map<int,bool> is_boundary_basin;
  // # node -> basin raw ID
  std::map<int,int> Cordonnier_FO_ID;
  // Initialised with the basin that drains outside
  std::map<int,std::vector<int> > Cordonnier_to_concat;

  //stack_adder: it remember for each basin which stack it has been added to in the Cordonnier_to_concat
  // std::map<int,int> stack_adder_for_kriskal;
  // # basin raw ID -> link
  // std::map<int,link> Cordonnier_links;
  // This stores the fastscape node ordering
  std::map<std::string, std::vector<std::vector<int> > > raw_FS;

  // Map of informations per basin key
  //### map of best_fit movern with disorder method per BK. key: Basin key, value: vector of disorder metric; associated witht the vector of tested values
  std::map<int, std::vector<float> > disorder_movern_per_BK; std::vector<float> associated_movern_disorder;
  std::map<int, std::vector<float> > best_fits_movern_per_BK; std::map<int, float > bet_fits_movern_per_BK;
  std::map< float, std::vector<float> > raw_disorder_per_SK;

  std::vector< std::vector<std::vector<float> > > normalised_disorder_val;
  std::vector< std::vector<int> > normalised_disorder_n_pixel_by_combinations;

  std::map<int, std::map<float, std::vector<float> > > disorder_all_values_by_movern;

  std::map<int, std::map<int, std::vector<float> > > best_fits_movern_per_BK_per_SK;
  //### map of baselevel node to basin_key and its opposite
  std::map<int,int> baselevel_nodes_to_BK, BK_to_baselevel_nodes, BLjunctions_to_BK, BK_to_BLjunctions;

  // DEBUGGING TEMP ATTRIBUTES
  std::vector< std::vector<int> > DEBUG_get_all_dis_SK_SK; std::vector<std::vector<float> > DEBUG_get_all_dis_SK_val;

  // Barnes flat resolver variables
  xt::xtensor<short,2> Barnes_flow_dir;
  // That map contains all the flat nodes with their respective label
  xt::xtensor<short,2> Barnes_label_flat;
  // Get the flat_mask thing
  xt::xtensor<short,2> Barnes_flat_mask;

	private:



  // Create functions there so far as well
  void create();
  void create(int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& data);

};


#endif