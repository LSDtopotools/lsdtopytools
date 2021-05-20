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
#ifndef LSDTribBasModel_HPP
#define LSDTribBasModel_HPP
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

// #ifdef __linux__
// #include "LSDRasterModel.hpp"
// #endif

#include <omp.h>

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

namespace TriBas
{

	//#########################################################	
	//Internal functions
	//#########################################################
	void calculate_m_chi_basic_inplace(xt::xtensor<float,1>& elevation, xt::xtensor<float,1>& chi, xt::xtensor<float,1>& m_chi, xt::xtensor<int,1>& node_index, std::map<int,int>& node_to_rec);
    std::map<std::string, float > calculate_comparative_metrics(xt::xtensor<float,1>& elevation_1, xt::xtensor<float,1>& elevation_2, xt::xtensor<float, 1>& common_x, float x_bin, float tolerance_delta);
	//#########################################################
	//#########################################################


	//#########################################################
	// Entry_point form python: 
	//#########################################################
	std::map<std::string, xt::pytensor<float,1> > setting_up_TriBAs(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, 
    float m, float n, float A0, double Northing, double Easting, int search_radius_nodes,
    int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma);
	std::map<std::string, xt::pytensor<float,1> > run_model(float meq, float neq, int n_timestep, int timestep, int save_step, xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, std::map<std::string, xt::pytensor<float,1> >& prebuilt);
	std::map<string,xt::pytensor<float,1> > burn_external_to_map(std::map<string,xt::pytensor<float,1> >& prebuilt, int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, std::string name_of_burned_column);
    std::map<std::string, xt::pytensor<float,1> > run_to_steady_state(float meq, float neq, int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, std::map<std::string, xt::pytensor<float,1> >& prebuilt, int max_timestep);
	
	std::map<std::string, xt::pytensor<float,1> > generate_MTO(std::map<std::string, xt::pytensor<float,1> >& prebuilt);
    //#########################################################
	//#########################################################
	
};



class LSDTribBas
{

	public:
	// Default constructor 
	LSDTribBas() { create(); }

	// Full constructor
	LSDTribBas(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, 
    float m, float n, float A0, double Northing, double Easting, int search_radius_nodes,
    int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma){ create(nrows, ncols, xmin, ymin, cellsize, ndv, 
    data,  x_sources,  y_sources,  m,  n,  A0,  Northing,  Easting,  search_radius_nodes,
    target_nodes, n_iterations, skip, minimum_segment_length,  sigma);}

  	void generate_MTO();
	float newton_rahpson_solver(float keq, float meq, float neq, float tDA, float dt, float dx, float this_elevation, float receiving_elevation ,float tol);
	void ingest_external_data_from_xy(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, std::string name_of_burned_column);
  	std::map<std::string,xt::pytensor<float,1> > run_model(int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, int max_timestep );
	std::map<std::string,xt::pytensor<float,1> > run_model_parallel(int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, int max_timestep );


	xt::pytensor<float,1> first_order_m_chi_from_custarray(xt::pytensor<float,1>& this_chi, xt::pytensor<float,1>& this_elevation);
	// std::map<std::string, float > calculate_comparative_metrics(xt::xtensor<float,1>& elevation_1, xt::xtensor<float,1>& elevation_2, xt::xtensor<float, 1>& common_x, float x_bin, float tolerance_delta);



	// Getter/setter
	float get_baselevel_elevation(){return baselevel_elevation;}
	void set_baselevel_elevation(float& new_baselevel_elevation){baselevel_elevation = new_baselevel_elevation;}
	float get_meq(){return meq;}
	float get_neq(){return neq;}

	xt::pytensor<int,1> get_node_ID(){return node_ID;}
	xt::pytensor<int,1> get_source_key(){return source_key;}
	xt::pytensor<int,1> get_receivers(){return receivers;}
	xt::pytensor<int,1> get_raster_row(){return raster_row;}
	xt::pytensor<int,1> get_raster_col(){return raster_col;}

	xt::pytensor<float,1> get_chi(){return chi;}
	xt::pytensor<float,1> get_elevation(){return elevation;}
	xt::pytensor<float,1> get_flow_distance(){return flow_distance;}
	xt::pytensor<float,1> get_drainage_area(){return drainage_area;}
	xt::pytensor<float,1> get_x_coord(){return x_coord;}
	xt::pytensor<float,1> get_y_coord(){return y_coord;}
	xt::pytensor<float,1> get_m_chi(){return m_chi;}
	xt::pytensor<float,1> get_last_elevation_modelled(){return last_elevation_modelled;}
	xt::pytensor<float,1> get_last_U_modelled(){return last_U_modelled;}
	xt::pytensor<float,1> get_last_K_modelled(){return last_K_modelled;}

	std::map<std::string, xt::pytensor<float,1> > get_external_data(){return external_data;}

	xt::pytensor<int,1> get_MTO_at_node(){return MTO_at_node;}
	int get_max_MTO(){return max_MTO;}
	std::vector<std::vector<std::pair<int,int> > > get_MTO_global_index(){return MTO_global_index;}


	void simplification_by_median(float flow_distance_bin);

	void set_m_and_n(float m, float n){meq = m; neq = n;}

	/// @brief this function segment the profile with a basic n_node_per_segment_method
	/// @brief to generate a cross-checking routine that will check the median m_chi value of the 
	/// @brief modelled version of the segment compare to the real one.
	/// @param n_node_max_per_segment (int): the number of node per segment. Max because we don't want segments in 2 rivers
	/// @returns Nothing, void. But get the cross-validation structure ready and call the basic-segmented-original sta function I havent created yet
	/// @authors B.G. 19/02/2019. At the writing retreat yo.
	void generate_cross_checking_segmentation_basic(int n_node_max_per_segments);

	///@brief Compute a bunch of stats on a given global array based on the segmented indexing described before
	///@param py or x tensor of values to compute that MUST be in the same indexing than the global arrays
	///@return a map of statistical metrics: keys are "median", "first_quartile" and "third_quartile" and indices in the array are the segment ID yo
	///@authors B.G
	///@date 19/02/2019
	std::map<std::string, xt::pytensor<float, 1> > calculate_segmented_basic_metrics(xt::pytensor<float, 1>& this_array_to_compute);

	///@brief This important function (also every function is important, do not accuse me of functionnism!!) check the global gathering of segmented median
	///@brief median past data and check which one are in steady state for a sufficient time to be validated
	///@param tolarance_interval: tolarance used to determin steady_state for a segment. All the m_chi values would nee to be within the same tolerance to be consdered OK
	///@param min_time_checked (int): the minimum of time a segment has to be processed before actually check for steady-state. Hopefully less time than the number of time Guillaume has been checked for STDs lol
	///@returns a vector of boolean telling which segments need to be adapted. The actual adaptation and data extraction is in another function.
	///@authors B.G. 19/02/2019
	std::vector<bool> cross_checker_of_steady_state(int min_time_checked, float tolerance_interval);

	///@brief This function ingest a previously calculated checker and save/resample the segments that needs to be
	///@param vector of bool with the same dimension as the number of cross-checking segments
	///@returns Nothing, but adapt the internal uplift and erodibility fields
	///@authors B.G.
	///@date 19/02/2019
	void resample_and_save_matches_on_segments_that_need_to_be(vector<bool> checker, float m_chi_tolerance_for_match);

	///@brief Comment to come. Run my version of a MC analysis to find the best k fit on the model
	///@brief bear grylls because the model actually adapts, survives, and overcomes.
	///@brief Bad pun. SNS.
	///@param int timestep: the timestep for the model
	///@param save_step: model will save an output every save_step time
	///@param base_uplift_field: starting uplift field
	///@param base_erodibility_field: starting erodibility field
	///@param int n_initial_run: You probably want to run the model a certain amount of time before actually starting the experiment to avoid checking close states
	///@param int min_run_before_assignment: after each change of k/U values, this determines the number of  run to wait before assessing potential steady-state
	///@param int max_number_of_run: the number of run to achieve
	///@param float SS_tolerance: tolerance of m_chi median per segment to be considered a steady-state
	///@param float match_tolerance: tolerance for delta m_chi model - original to consider a match
	///@returns a map of node-wide statistics about the matchs
	///@authors B.G
	///@date 19/02/2019
	std::map<std::string, xt::pytensor<float,1> > segmented_bear_grylls(int timestep, int save_step, xt::pytensor<float,1>& base_uplift_field, 
	xt::pytensor<float,1> base_erodibility_field, int n_initial_run, int min_run_before_segment_assessment, 
	int max_number_of_run, float SS_tolerance, float match_tolerance);


	std::map<std::string,xt::pytensor<float,1> > get_bear_results();
	std::map<std::string, std::map<int,xt::pytensor<float,1> > > get_distribeartion(int n_bins);


	void set_range_of_uplift_and_K_for_Bear_gryll(std::vector<float>& these_new_K, std::vector<float>& these_new_U);

	std::vector<std::vector<std::pair<float,float> > > get_best_fits_gatherer_for_the_segments(){ return best_fits_gatherer_for_the_segments; };

	///@brief This function update the working elevation (the one that will be modelled) with a new one
	///@returns Nothing, Update the elevation field
	///@authors B.G
	void update_to_last_simulation() {elevation = last_elevation_modelled;};

	///@brief This function update the working elevation (the one that will be modelled) with a custom one
	///@param new_elev: The new elevation
	///@returns Nothing, Update the elevation field
	///@authors B.G
	void update_current_elevation(xt::pytensor<float,1>& new_elev) {elevation = new_elev;};




  protected:

  ///@brief elevation of the outlet node
  float baselevel_elevation;
  float meq;
  float neq;

  // General arrays for building the model 
	xt::pytensor<int,1> node_ID;
	xt::pytensor<float,1> chi;
	xt::pytensor<float,1> elevation;
	xt::pytensor<float,1> flow_distance;
	xt::pytensor<int,1> source_key;
	xt::pytensor<float,1> drainage_area;
	xt::pytensor<int,1> receivers;
	xt::pytensor<int,1> raster_row;
	xt::pytensor<int,1> raster_col;
	xt::pytensor<float,1> x_coord;
	xt::pytensor<float,1> y_coord;
	xt::pytensor<float,1> m_chi;

	// Saving savestates after each run in case
	xt::pytensor<float,1> last_elevation_modelled;
	xt::pytensor<float,1> last_U_modelled;
	xt::pytensor<float,1> last_K_modelled;

	// Cross-checking attributes
	xt::pytensor<int,1> cross_checking_ID;
	std::map<int,int> map_of_size_for_each_cross_checking_ID; // this should be explicit enough
	size_t n_segment_for_cross_checking; // keep track of how many segments ID I have.
	std::vector<std::vector<float> > keep_track_of_the_last_medians; // this is an important bit : it keeps track of the last medians until SS is reached for each segments.
	std::vector<int> keep_track_of_how_many_time_it_has_been_processed_for_SS; //  This shit keeps track if a segment has been prossed enough time
	std::vector<std::vector<std::pair<float,float> > > best_fits_gatherer_for_the_segments; // Gather the best fit combination of K and U for each segments when founded
	xt::pytensor<float,1> original_segment_median_checker; // contains the original segments median, to be compared when potentially adapting the model
	xt::pytensor<float,1> segment_median_chi; // contains the original segments median, to be compared when potentially adapting the model
	xt::pytensor<float,1> segment_median_elev; // contains the original segments median, to be compared when potentially adapting the model
	xt::pytensor<float,1> segment_flow_dist; // contains the original segments median, to be compared when potentially adapting the model
	int change_counter;
	int match_counter;
	

	// MC-ish analysis
	// These 2 attribute are gathering the possibility to check in the MC analysis to find the best fit
	std::vector<float> ranges_of_uplift;
	std::vector<float> ranges_of_erodibility;
	xt::pytensor<float,1> uplift_to_adapt; // This array simply is the uplift/erodibility field of the rivers. However, these are the one that will constantly be adapted
	xt::pytensor<float,1> erodibility_to_adapt; // This array simply is the uplift/erodibility field of the rivers. However, these are the one that will constantly be adapted

	// adding data from raster or other to the model
	std::map<std::string, xt::pytensor<float,1> > external_data;

	// MTO structure (experimental)
	xt::pytensor<int,1> MTO_at_node;
	int max_MTO;
	std::vector<std::vector<std::pair<int,int> > > MTO_global_index;

	// Simplified model
	xt::pytensor<float,1> simplified_elevation;
	xt::pytensor<float,1> simplified_flow_distance;
	xt::pytensor<int,1> simplified_node_ID;
	xt::pytensor<int,1> simplified_receivers;

	//Utilities
	std::map<int,int> recnode_global;



  private:

  void create();
  void create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, 
    float m, float n, float A0, double Northing, double Easting, int search_radius_nodes,
    int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma);






};

// #ifdef __linux__

// class muddpyle
// {

// 	public:
// 	// Default constructor 
// 	muddpyle() { create(); }
// 	muddpyle(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& initial_topography, float m, float n, int dt, int Umod, float save_step, float default_D, float default_Sc, xt::pytensor<float,2>& K_field, xt::pytensor<float,2>& U_field, bool hillslope_diffusion, bool use_adaptive_timestep, float max_dt, std::string OUT_DIR, std::string OUTID, std::vector<std::string> bc )
// 	{create( nrows,  ncols,  xmin,  ymin,  cellsize,  ndv,  initial_topography, m,  n,  dt,  Umod,  save_step,  default_D,  default_Sc,  K_field,  U_field,  hillslope_diffusion, use_adaptive_timestep,  max_dt,  OUT_DIR,  OUTID, bc );}
//  	muddpyle(bool tquiet){create(tquiet);}

//  	/// @brief Feed the model with real topography
//  	/// @param nrows (int): number of row in the dem
//  	/// @param ncols (int): number of col in the dem
//  	/// @param xmin (float): minimum x coordinate
//  	/// @param ymin (float): minimum y coordinate
//  	/// @param cellsize (float): resolution
//  	/// @param xmin (float): No Data Value
//  	/// @param data (numpy 2D array or xtensor of floats) The topography array 
//  	void initialise_model_with_dem(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data);
//  	void initialise_model_kimberlite(int nrows, int ncols, float resolution, int diamond_square_feature_order, float diamond_square_relief, float parabola_relief, float roughness_relief, int prediff_step);


//  	void set_uplift_raster(xt::pytensor<float,2>&data){  TNT::Array2D<float> data_pointerast(base_nrows,base_ncols,base_ndv);for(size_t i=0;i<base_nrows;i++){for(size_t j=0;j<base_ncols;j++){data_pointerast[i][j] = data(i,j);}}; LSDRaster tp(base_nrows, base_ncols, base_xmin, base_ymin, base_resolution, base_ndv, data_pointerast); UpliftRaster = tp;}
//  	void set_K_raster(xt::pytensor<float,2>&data){  TNT::Array2D<float> data_pointerast(base_nrows,base_ncols,base_ndv);for(size_t i=0;i<base_nrows;i++){for(size_t j=0;j<base_ncols;j++){data_pointerast[i][j] = data(i,j);}}; LSDRaster tp(base_nrows, base_ncols, base_xmin, base_ymin, base_resolution, base_ndv, data_pointerast); KRaster = tp;}
//  	void set_D_raster(xt::pytensor<float,2>&data){  TNT::Array2D<float> data_pointerast(base_nrows,base_ncols,base_ndv);for(size_t i=0;i<base_nrows;i++){for(size_t j=0;j<base_ncols;j++){data_pointerast[i][j] = data(i,j);}}; LSDRaster tp(base_nrows, base_ncols, base_xmin, base_ymin, base_resolution, base_ndv, data_pointerast); DRaster = tp;}
//  	void set_Sc_raster(xt::pytensor<float,2>&data){  TNT::Array2D<float> data_pointerast(base_nrows,base_ncols,base_ndv);for(size_t i=0;i<base_nrows;i++){for(size_t j=0;j<base_ncols;j++){data_pointerast[i][j] = data(i,j);}}; LSDRaster tp(base_nrows, base_ncols, base_xmin, base_ymin, base_resolution, base_ndv, data_pointerast); ScRaster = tp;}
//  	/// @brief sets output path and prefix
//  	void set_save_path_name(std::string OUT_DIR, std::string OUT_ID){mymod.add_path_to_names(OUT_DIR); mymod.set_name(OUT_DIR+OUT_ID);}
//  	/// @brief change values for m and n exponents
//  	void set_SPL_exponents(float m, float n){mymod.set_m(m);mymod.set_n(n);}

//  	/// The mode of uplift. Options are:
//   /// 0 == block uplift, 1 == tilt block, 2 == gaussian, 3 == quadratic, 4 == periodic
//   void set_uplift_mode(short Umod){mymod.set_uplift_mode(Umod);}

//   /// Deals with timing: sets dt, max_dt in case of adaptative timestep, endtime and endtime mode
//   /// TODO: understand and explain what the end time options are
//   void deal_with_time(float dt, float max_dt, float endtime,short entime_mode ){mymod.set_timeStep(dt);mymod.set_maxtimeStep(max_dt);mymod.set_endTime(endtime);mymod.set_endTime_mode(entime_mode);}

//   /// Set the time interval in year between each output printing
//   void set_print_interval_in_year(float pi) {mymod.set_float_print_interval(pi);}

//   /// It is mandatory to give to the model default values for erodibility, diffusion and critical slope. These will be overwritten if KRaster or DRaster or ScRaster are given
//   void set_global_default_param(float tD, float tsc, float erod){ mymod.set_D(tD);mymod.set_S_c(tsc);mymod.set_K(erod);}

//   /// This activate or deactivate the hillslope diffusion process
//   void set_hillslope_diffusion_switch(bool swip){mymod.set_hillslope(swip);}
//   /// If Hillslope_diffusion is activated, you amy want to shift it to non linear
//   void set_nonlinear_switch_for_hillslope_diffusion(bool swip){mymod.set_nonlinear(swip);}

//   /// This activate the fluvial component of the model (atm fastscape)
//   void set_fluvial_erosion(bool swip){mymod.set_fluvial(swip);}

//   /// I need to dig in the code to play with that. Leave default might be a good idea so far
//   void set_boundary_conditions(std::vector<std::string> bc){mymod.set_boundary_conditions(bc);}

//   /// @brief Run the model
//   void run_model(bool fluvial, bool hillslope, bool spatially_variable_UK, int save_frame, bool use_adaptative_timestep);

//  protected:

//  	/// the Muddpile object
//   LSDRasterModel mymod;
//   int base_nrows;
//   int base_ncols;
//   float base_xmin;
//   float base_ymin;
//   float base_resolution;
//   float base_ndv;


//   /// Runtime attributes
//   /// Quiet or not
//   bool verbose;
  
//   /// Raster of erodibility values
//   LSDRaster KRaster;
//   /// Raster of Uplift Values
//   LSDRaster UpliftRaster;
//   /// Raster of Uplift Values
//   LSDRaster DRaster;
//   /// Raster of Uplift Values
//   LSDRaster ScRaster;

		


//   private:

//   void create();
//   void create(bool tquiet);
//   void create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, float m, 
//   float n, int dt, int Umod, float save_step, float default_D, float default_Sc, xt::pytensor<float,2>& K_field, xt::pytensor<float,2>& U_field, bool hillslope_diffusion,
//   bool use_adaptive_timestep, float max_dt, std::string OUT_DIR, std::string OUT_ID, std::vector<std::string> bc );


// };

// #endif // end of linux rule

// Carpythians aims to be a lightweight Landscape Evolusion Model porting sediment transport to the game while keeping access to LSDTopoTools
// "Why are you not integrating that into MuddPile?" -> I struggle to integrate new stuff in Muddpile for some reason.
// I am not sure how far it will go, this is mostly for a specific paper
class Carpythians
{

  public:
	// Default constructor 
	Carpythians() { create(); }
   	Carpythians(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<short,2>& boundary_conditions, int n_threads) 
   {create( nrows,  ncols,  xmin,  ymin,  cellsize,  ndv,  data, boundary_conditions, n_threads);};
	


	// MATH ARE FUN
	
	// Solve node to node equation for a general simple SPL
	float newton_rahpson_solver(float keq, float meq, float neq, float tDA, float dt, float dx, float this_elevation, float receiving_elevation ,float tol);
	xt::pytensor<float,2> run_SPL(float dt, int n_dt, xt::pytensor<float,2>& uplift, xt::pytensor<float,2>& erodibility, int num_threads, float meq, float neq);
	xt::pytensor<float,2> test_run_SSPL(float dt, float n_dt, float Kb, float Ks, float Gb, float Gs, float U_range, float U_foreland, float meq, float neq);
	float local_newton_rhapson_solver_for_SSPL(float a, float b, float c, float d, float neq, float htp1k, float toleritude);
	void cumulate_for_STSPL_eq23_righthand_side(TNT::Array2D<float>& cumularray, TNT::Array2D<float> elevatk, vector<int>& custom_inverted_stack, xt::pytensor<float,2>& uplift, float dt , bool all_nodes, bool reinitialise);
	xt::pytensor<float,2> run_STSPL(float dt, float n_dt, xt::pytensor<float,2>& Kb, xt::pytensor<float,2>& sediment_thickness, float Ks, float Gb, float Gs, xt::pytensor<float,2> uplift,  float meq, float neq, bool lake);
	void reprocess_FlowInfo();





  protected:

   // Initial conditions
   LSDRaster current_topography;
   LSDFlowInfo current_FI;
   std::map<string, std::vector< std::vector<int> > > current_node_orders;
   xt::pytensor<float,2> uplift, erodibility_K;


   // SPL things
   float m_equ, n_equ;


   // Other
   float initial_time;
   float current_time;
   std::vector<std::string> BoCo;
   int FATAL_WARNINGS_NRITERATIONS, FATAL_WARNINGS_GSITERATIONS;


   // MP
   int max_n_threads;

  private:

  void create();
  void create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<short,2>& boundary_conditions, int n_threads);

};




#endif