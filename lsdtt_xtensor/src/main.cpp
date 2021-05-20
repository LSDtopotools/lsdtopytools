//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
//
// lsdtopytools - main 
// Land Surface Dynamics python binding
//
// An object within the University
//  of Edinburgh Land Surface Dynamics group topographic toolbox
//  for Using the LSDTopoTools framwork from python without having to compile manually the code
//  and to script it. Cross platform, this requires a bit of cpp-python gymnastic using pybind11 and xtensor.
//  The original version of that file was created from a template from python-binding cookiecutter, from xtensor git-hub. 
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

// These includes are for binding with python
#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

// Common includes
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <map>

// Includes for openMP (experimental use, you can ignore)
#include <omp.h>  //Used for OpenMP run-time functions
#ifndef _OPENMP
  #define omp_get_thread_num()  0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif

// Other lsdtopytools includes
#include "LSD_xtensor_utils.hpp"
#include "LSDStatsTools.hpp"
#include "LSDEntry_points.hpp"
#include "LSD_xtensor_convtools.hpp"
// #include "LSDTribBasModel.hpp" // Moved to Carpythians
#include "LSDDEM_xtensor.hpp"
namespace py = pybind11;



// ###########################################################################################################
// ###########################################################################################################
// ############################### First attempts to bind, ignore these inline functions #####################
// ################################## I keep them for back-compatibility purposes  ###########################
// ###########################################################################################################
// ###########################################################################################################

inline std::map<int, std::map<std::string,float> > comparison_stats_from_2darrays(xt::pytensor<int, 2>& arr1, xt::pytensor<float, 2>& arr2, float ignore_value, int nRows, int nCols)
{py::gil_scoped_release release; std::map<int, std::map<std::string,float> > output = xtlsd::_comparison_stats_from_2darrays(arr1,arr2,ignore_value,nRows,nCols); return output;}

inline std::map<int, std::vector<float> > get_groupped_values(xt::pytensor<int, 2>& arr1, xt::pytensor<float, 2>& arr2, float ignore_value, size_t nRows, size_t nCols)
{py::gil_scoped_release release; std::map<int, std::vector<float> > output = xtlsd::_get_groupped_values(arr1, arr2, ignore_value, nRows, nCols); return output;}


inline std::vector<float> KDE_gaussian(xt::pytensor<float,1>& x_val, xt::pytensor<float,1>& y_val , float h)
{py::gil_scoped_release release; std::vector<float> out = xtlsd::_gaussian_KDE(x_val, y_val , h); return out;}


inline std::map<int, std::vector< std::vector<float> > > growing_window_stat(xt::pytensor<double,2>& base_array, size_t nRows, size_t nCols, int min_window, int step, int nstep)
{py::gil_scoped_acquire acquire; py::gil_scoped_release release; std::map<int, std::vector< std::vector<float> > > out = xtlsd::_growing_window_stat(base_array, nRows, nCols, min_window, step, nstep); return out;}


// inline std::map<std::string, std::vector<float> > get_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<float, 1>& Y, float interval, int nthread)
// {py::gil_scoped_acquire acquire; py::gil_scoped_release release; std::map<std::string, std::vector<float> > output = xtlsd::_get_median_profile(X, Y, interval, nthread); return output;}

inline  std::map<int, float> get_drainage_density_from_sources(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<int,2>& comparative_data, std::vector<float>& x_sources, std::vector<float>& y_sources)
{py::gil_scoped_release release; std::map<int,float> output = EPPy::get_drainage_density_from_sources( nrows,  ncols,  xmin,  ymin,  cellsize,  ndv,  data, comparative_data, x_sources, y_sources); return output;}

inline std::map<int, std::vector<float> > proportion_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<int, 1>& Y, float interval, int nthread)
{py::gil_scoped_release release; std::map<int, std::vector<float> > output = xtlsd::_proportion_median_profile(X, Y,  interval,  nthread); return output;}

inline std::map<std::string, xt::pytensor<float,2> > get_polyfit_rasters(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<int,1> selecao, float window_radius)
{std::map<std::string, std::vector<std::vector<float> > > preoutput = EPPy::get_polyfit_rasters(nrows, ncols, xmin, ymin, cellsize, ndv, data, selecao, window_radius); return conv::map_string_2Dvec_to_py(preoutput);}

inline std::map<std::string, xt::pytensor<float,1> > get_circular_windowed_stats(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_coord, xt::pytensor<float,1>& y_coord , float radius)
{ return EPPy::get_circular_windowed_stats(nrows, ncols, xmin, ymin, cellsize, ndv, data, x_coord, y_coord , radius);}

// The follwing function is just there for debugging purpose
inline std::vector<float> compt_debug(xt::pytensor<float,1>& x_val)
{
  size_t s = x_val.size();
  std::vector<float> v(s), v1(s);

  // py::gil_scoped_release release;
  std::cout << "Looping through pyarray" << std::endl;
  #pragma omp parallel
  {
    #pragma omp for
    for(int i=1;i<(s-1);i++)
    {
      int id = omp_get_thread_num();
      std::cout << "thread_ID: " << id << std::endl;
      for(int u=0; u< 500000; u++)
      v[i] = (x_val(i-1)+x_val(i+1))/std::pow(x_val(i),2);
    }
  }
  std::cout << "Done" << std::endl;

  std::cout << "Looping through vector+copy()" << std::endl;
  
  for(int i=0;i<(s);i++)
  {
    v1[i]=x_val[i];
  }
  #pragma omp parallel
  {
    #pragma omp for
    for(int i=1;i<(s-1);i++)
    {
      int id = omp_get_thread_num();
      std::cout << "thread_ID: " << id << std::endl;
      for(int u=0; u< 500000; u++)
      v[i] = (v1[i-1]+v1[i+1])/std::pow(v1[i],2);
    }
    std::cout << "Done" << std::endl;
  }

  return v;

}
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//######################################################################## Find below the actual bindings ##################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################
//##########################################################################################################################################################################################################


// Python Module and Docstrings
PYBIND11_MODULE(lsdtt_xtensor_python, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        lsdtt_xtensor_python:
        Python bindings for LSDTopoTools

        .. currentmodule:: lsdtt_xtensor_python

        .. autosummary::
           :toctree: _generate  
           
    )pbdoc";

   
    // py::bind_map<std::map<std::string, double>>(m, "comparison_stats_from_2darrays");
    m.def("comparison_stats_from_2darrays", comparison_stats_from_2darrays, "AB45FF FF78FC 45452F DDD5D5 EF456666 66664466 41DF4A");
    m.def("get_groupped_values", get_groupped_values, "Extract all the values from array 2 for each individual values of array 1 (integer).");
    m.def("KDE_gaussian", KDE_gaussian, "Run a gaussian KDE on the 1D array passed to the data.");
    m.def("compt_debug", compt_debug, "Debugging test, ignore",py::call_guard<py::gil_scoped_release>());
    m.def("growing_window_stat", growing_window_stat, "TESTTOGO");
    m.def("get_median_profile", &xtlsd::_get_median_profile, "This function takes X and Y numpy 1Darrays as an input (sorted by X values) and generates a min-med-3rd quartile profile out of that. The only application I can think of so far is median river profile (long, chi , steepnes,...).");
    m.def("get_drainage_density_from_sources", get_drainage_density_from_sources, "Gets drainage density.");
    m.def("proportion_median_profile",proportion_median_profile, "Do stuff.");
    m.def("get_polyfit_rasters",get_polyfit_rasters, "Gets polyfit rasters with parameters that I will develop later. SNS.");
    m.def("get_circular_windowed_stats",get_circular_windowed_stats, "Gets polyfit rasters with parameters that I will develop later. SNS.",py::call_guard<py::gil_scoped_release>());

    m.def("segment_data_Mudd14", &mudd14partitioner::segment_data);

    py::class_<LSDDEM_xtensor>(m, "LSDDEM_cpp",py::dynamic_attr())
      .def(py::init<>())
      //.def(py::init([](/*param*/){return std::unique_ptr<LSDDEM_xtensor>(new LSDDEM_xtensor(/*param without identifier*/)); })) // <- template for new constructors
      .def(py::init([](int tnrows, int tncols, float txmin, float tymin, float tcellsize, float tndv, xt::pytensor<float,2>& data){return std::unique_ptr<LSDDEM_xtensor>(new LSDDEM_xtensor(tnrows, tncols, txmin, tymin, tcellsize, tndv, data)); }))
      .def("PreProcessing",&LSDDEM_xtensor::PreProcessing)
      .def("calculate_FlowInfo",&LSDDEM_xtensor::calculate_FlowInfo)
      .def("calculate_channel_heads",&LSDDEM_xtensor::calculate_channel_heads)
      .def("calculate_juctionnetwork",&LSDDEM_xtensor::calculate_juctionnetwork)
      .def("calculate_outlets_locations_from_xy",&LSDDEM_xtensor::calculate_outlets_locations_from_xy)
      .def("calculate_outlets_locations_from_minimum_size",&LSDDEM_xtensor::calculate_outlets_locations_from_minimum_size)
      .def("calculate_outlet_location_of_main_basin",&LSDDEM_xtensor::calculate_outlet_location_of_main_basin)
      .def("generate_chi",&LSDDEM_xtensor::generate_chi)
      .def("generate_ksn",&LSDDEM_xtensor::generate_ksn,py::call_guard<py::gil_scoped_release>())
      .def("detect_knickpoint_locations",&LSDDEM_xtensor::detect_knickpoint_locations)
      .def("set_boundary_conditions", &LSDDEM_xtensor::set_boundary_conditions)
      .def("get_chi_raster", &LSDDEM_xtensor::get_chi_raster)
      .def("get_chi_basin", &LSDDEM_xtensor::get_chi_basin)
      .def("get_int_ksn_data", &LSDDEM_xtensor::get_int_ksn_data)
      .def("get_float_ksn_data", &LSDDEM_xtensor::get_float_ksn_data)
      .def("get_int_knickpoint_data", &LSDDEM_xtensor::get_int_knickpoint_data)
      .def("get_float_knickpoint_data", &LSDDEM_xtensor::get_float_knickpoint_data)
      .def("get_hillshade", &LSDDEM_xtensor::get_hillshade)
      .def("get_polyfit_on_topo",&LSDDEM_xtensor::get_polyfit_on_topo)
      .def("is_already_preprocessed",&LSDDEM_xtensor::already_preprocessed)
      .def("get_PP_raster", &LSDDEM_xtensor::get_PP_raster)
      .def("get_catchment_perimeter", &LSDDEM_xtensor::get_catchment_perimeter)
      .def("get_base_raster", &LSDDEM_xtensor::get_base_raster)
      .def("calculate_movern_disorder", &LSDDEM_xtensor::calculate_movern_disorder, py::call_guard<py::gil_scoped_release>())
      .def("get_disorder_dict", &LSDDEM_xtensor::get_disorder_dict)
      .def("get_disorder_vec_of_tested_movern", &LSDDEM_xtensor::get_disorder_vec_of_tested_movern)
      .def("burn_rast_val_to_xy", &LSDDEM_xtensor::burn_rast_val_to_xy)
      .def("calculate_FlowInfo_Dinf", &LSDDEM_xtensor::calculate_FlowInfo_Dinf)
      .def("get_channel_gradient_muddetal2014", &LSDDEM_xtensor::get_channel_gradient_muddetal2014)
      .def("get_ksn_first_order_chi", &LSDDEM_xtensor::get_ksn_first_order_chi)
      .def("ingest_channel_head", &LSDDEM_xtensor::ingest_channel_head)
      .def("get_DA_raster", &LSDDEM_xtensor::get_DA_raster)
      .def("extract_perimeter_of_basins", &LSDDEM_xtensor::extract_perimeter_of_basins)
      .def("get_sources_full",&LSDDEM_xtensor::get_sources_full)
      .def("calculate_discharge_from_precipitation", &LSDDEM_xtensor::calculate_discharge_from_precipitation, py::call_guard<py::gil_scoped_release>())
      .def("query_xy_from_rowcol", &LSDDEM_xtensor::query_xy_from_rowcol)
      .def("get_fastscape_ordering", &LSDDEM_xtensor::get_fastscape_ordering)
      .def("get_hillshade_custom", &LSDDEM_xtensor::get_hillshade_custom)
      .def("get_FO_Mchi",&LSDDEM_xtensor::get_FO_Mchi)
      .def("force_all_outlets", &LSDDEM_xtensor::force_all_outlets)
      .def("mask_topo", &LSDDEM_xtensor::mask_topo)
      .def("get_individual_basin_raster", &LSDDEM_xtensor::get_individual_basin_raster)
      .def("get_best_fits_movern_per_BK", &LSDDEM_xtensor::get_best_fits_movern_per_BK)
      .def("set_n_nodes_to_visit_downstream", &LSDDEM_xtensor::set_n_nodes_to_visit_downstream)
      .def("get_single_river_from_top_to_outlet", &LSDDEM_xtensor::get_single_river_from_top_to_outlet)
      .def("query_rowcol_from_xy", &LSDDEM_xtensor::query_rowcol_from_xy)
      .def("get_receiver_data", &LSDDEM_xtensor::get_receiver_data)
      .def("get_disorder_dict_SK", &LSDDEM_xtensor::get_disorder_dict_SK)
      .def("get_SA_from_vertical_interval",&LSDDEM_xtensor::get_SA_from_vertical_interval)
      .def("trim_nodes_draining_to_baselevel", &LSDDEM_xtensor::trim_nodes_draining_to_baselevel)
      .def("calculate_outlets_locations_from_xy_v2", &LSDDEM_xtensor::calculate_outlets_locations_from_xy_v2)
      .def("extract_basin_draining_to_coordinates", &LSDDEM_xtensor::extract_basin_draining_to_coordinates)
      .def("calculate_outlets_min_max_draining_to_baselevel", &LSDDEM_xtensor::calculate_outlets_min_max_draining_to_baselevel)
      .def("query_xy_for_each_basin", &LSDDEM_xtensor::query_xy_for_each_basin)
      .def("get_DEBUG_all_dis_SK",&LSDDEM_xtensor::get_DEBUG_all_dis_SK)
      .def("get_all_disorder_values",&LSDDEM_xtensor::get_all_disorder_values)
      .def("get_n_pixels_by_combinations", &LSDDEM_xtensor::get_n_pixels_by_combinations)
      .def("calculate_outlet_location_of_main_basin", &LSDDEM_xtensor::calculate_outlet_location_of_main_basin)
      .def("calculate_outlets_locations_from_range_of_DA", &LSDDEM_xtensor::calculate_outlets_locations_from_range_of_DA)
      .def("get_baselevels_XY", &LSDDEM_xtensor::get_baselevels_XY)
      .def("get_chi_raster_all", &LSDDEM_xtensor::get_chi_raster_all)
      .def("get_flow_distance_raster",&LSDDEM_xtensor::get_flow_distance_raster)
  
      ;// end of class LSDDEM (semicolon important)

    // TribBas
    // m.def("prebuild_TribBas", prebuild_TribBas, "prebuild the tribas object");
    // m.def("run_TribBas", run_TribBas, "run the model with custom parameters from a prebuilt dictionnary");
    // m.def("burn_external_to_prebuilt",burn_external_to_prebuilt,"All in the title");
    // m.def("run_TribBas_to_steady_state",run_TribBas_to_steady_state,"Run the model until Steady State (within a given tolerance).");
    // m.def("inverse_weighted_distance", &xtlsd::inverse_weighted_distance, "Get the Inverse Weighted Distance spatial interpolation. Quite basic but should be relatively a quick way to get something nice.", py::call_guard<py::gil_scoped_release>());


    // // TribBas experimental object binding

    // py::class_<LSDTribBas>(m, "LSDTribBas_cpp")
    //   .def(py::init<>())
    //   // .def(py::init<int , int , float , float , float , float ,xt::pytensor<float,2>& , xt::pytensor<float,1>& , xt::pytensor<float,1>& , float , float , float , double , double , int ,int , int , int , int , int>(&LSDTribBas::create))
    //   .def(py::init([](int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, float m, float n, float A0, double Northing, double Easting, int search_radius_nodes, int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma){return std::unique_ptr<LSDTribBas>(new LSDTribBas(nrows, ncols, xmin, ymin, cellsize, ndv,  data,  x_sources, y_sources, m, n, A0, Northing, Easting, search_radius_nodes, target_nodes, n_iterations, skip, minimum_segment_length, sigma)); }))
    //   .def("baselevel_elevation", &LSDTribBas::get_baselevel_elevation)
    //   .def("meq", &LSDTribBas::get_meq)
    //   .def("neq", &LSDTribBas::get_neq)
    //   .def("node_ID",&LSDTribBas::get_node_ID)
    //   .def("chi",&LSDTribBas::get_chi)
    //   .def("elevation",&LSDTribBas::get_elevation)
    //   .def("flow_distance",&LSDTribBas::get_flow_distance)
    //   .def("source_key",&LSDTribBas::get_source_key)
    //   .def("drainage_area",&LSDTribBas::get_drainage_area)
    //   .def("receivers",&LSDTribBas::get_receivers)
    //   .def("raster_row",&LSDTribBas::get_raster_row)
    //   .def("raster_col",&LSDTribBas::get_raster_col)
    //   .def("x_coord",&LSDTribBas::get_x_coord)
    //   .def("y_coord",&LSDTribBas::get_y_coord)
    //   .def("m_chi",&LSDTribBas::get_m_chi)
    //   .def("last_elevation_modelled",&LSDTribBas::get_last_elevation_modelled)
    //   .def("last_U_modelled",&LSDTribBas::get_last_U_modelled)
    //   .def("last_K_modelled",&LSDTribBas::get_last_K_modelled)
    //   .def("external_data",&LSDTribBas::get_external_data)
    //   .def("MTO_at_node",&LSDTribBas::get_MTO_at_node)
    //   .def("max_MTO",&LSDTribBas::get_max_MTO)
    //   .def("MTO_global_index",&LSDTribBas::get_MTO_global_index)
    //   .def("generate_MTO",&LSDTribBas::generate_MTO)
    //   .def("newton_rahpson_solver",&LSDTribBas::newton_rahpson_solver)
    //   .def("ingest_external_data_from_xy",&LSDTribBas::ingest_external_data_from_xy)
    //   .def("run_model",&LSDTribBas::run_model)
    //   .def("set_m_and_n", &LSDTribBas::set_m_and_n)
    //   .def("first_order_m_chi_from_custarray", &LSDTribBas::first_order_m_chi_from_custarray)
    //   .def("generate_cross_checking_segmentation_basic", &LSDTribBas::generate_cross_checking_segmentation_basic)
    //   .def("calculate_segmented_basic_metrics", &LSDTribBas::calculate_segmented_basic_metrics)
    //   .def("cross_checker_of_steady_state", &LSDTribBas::cross_checker_of_steady_state)
    //   .def("resample_and_save_matches_on_segments_that_need_to_be", &LSDTribBas::resample_and_save_matches_on_segments_that_need_to_be)
    //   .def("segmented_bear_grylls", &LSDTribBas::segmented_bear_grylls)
    //   .def("get_bear_results", &LSDTribBas::get_bear_results)
    //   .def("get_distribeartion", &LSDTribBas::get_distribeartion)
    //   .def("set_range_of_uplift_and_K_for_Bear_gryll", &LSDTribBas::set_range_of_uplift_and_K_for_Bear_gryll)
    //   .def("get_best_fits_gatherer_for_the_segments", &LSDTribBas::get_best_fits_gatherer_for_the_segments)
    //   .def("update_to_last_simulation", &LSDTribBas::update_to_last_simulation)
    //   .def("update_current_elevation", &LSDTribBas::update_current_elevation)


    //   ; // end of class LSDTribBas

    // #ifdef __linux__  
    // py::class_<muddpyle>(m, "muddpyle_cpp")
    //   .def(py::init<>())
    //   .def(py::init([](int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& initial_topography, float m, float n, int dt, int Umod, float save_step, float default_D, float default_Sc, xt::pytensor<float,2>& K_field, xt::pytensor<float,2>& U_field, bool hillslope_diffusion, bool use_adaptive_timestep, float max_dt, std::string OUT_DIR, std::string OUTID, std::vector<std::string> bc ){return std::unique_ptr<muddpyle>(new muddpyle( nrows,  ncols,  xmin,  ymin,  cellsize,  ndv,  initial_topography, m,  n,  dt,  Umod,  save_step,  default_D,  default_Sc,  K_field,  U_field,  hillslope_diffusion, use_adaptive_timestep,  max_dt,  OUT_DIR,  OUTID, bc)); }))
    //   .def(py::init([](bool tquiet){return std::unique_ptr<muddpyle>(new muddpyle(tquiet));}))
    //   .def("initialise_model_with_dem",&muddpyle::initialise_model_with_dem)
    //   .def("initialise_model_kimberlite",&muddpyle::initialise_model_kimberlite)
    //   .def("set_uplift_raster",&muddpyle::set_uplift_raster)
    //   .def("set_K_raster",&muddpyle::set_K_raster)
    //   .def("set_D_raster",&muddpyle::set_D_raster)
    //   .def("set_Sc_raster",&muddpyle::set_Sc_raster)
    //   .def("set_save_path_name",&muddpyle::set_save_path_name)
    //   .def("set_SPL_exponents",&muddpyle::set_SPL_exponents)
    //   .def("set_uplift_mode",&muddpyle::set_uplift_mode)
    //   .def("deal_with_time",&muddpyle::deal_with_time)
    //   .def("set_print_interval_in_year",&muddpyle::set_print_interval_in_year)
    //   .def("set_global_default_param",&muddpyle::set_global_default_param)
    //   .def("set_hillslope_diffusion_switch",&muddpyle::set_hillslope_diffusion_switch)
    //   .def("set_nonlinear_switch_for_hillslope_diffusion",&muddpyle::set_nonlinear_switch_for_hillslope_diffusion)
    //   .def("set_fluvial_erosion",&muddpyle::set_fluvial_erosion)
    //   .def("set_boundary_conditions",&muddpyle::set_boundary_conditions)
    //   .def("run_model",&muddpyle::run_model)      
    // ;
    // #endif

    // py::class_<Carpythians>(m, "Carpythians_cpp")
    //   .def(py::init<>())
    //   // .def(py::init<int , int , float , float , float , float ,xt::pytensor<float,2>& , xt::pytensor<float,1>& , xt::pytensor<float,1>& , float , float , float , double , double , int ,int , int , int , int , int>(&LSDTribBas::create))
    //   .def(py::init([](int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<short,2>& boundary_conditions, int n_threads){return std::unique_ptr<Carpythians>(new Carpythians( nrows,  ncols,  xmin,  ymin,  cellsize,  ndv, data, boundary_conditions, n_threads)); }))
    //   .def("run_SPL", &Carpythians::run_SPL,py::call_guard<py::gil_scoped_release>())
    //   .def("test_run_SSPL", &Carpythians::test_run_SSPL, py::call_guard<py::gil_scoped_release>())
    //   .def("run_STSPL", &Carpythians::run_STSPL, py::call_guard<py::gil_scoped_release>())
      
      // ; // end of class LSDTribBas

}

// ###################################################################################################################################################
// ########################################################### Binding c++ class LSDTribBas ##########################################################
// ###################################################################################################################################################

