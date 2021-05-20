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

#ifndef LSDTribBasModel_CPP
#define LSDTribBasModel_CPP
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include <queue>
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

#include "LSDTribBasModel.hpp"

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

// #include <Eigen/Sparse>


// // Dealing with muddpile stuff
// #ifdef __linux__ 
// #include "LSDRasterModel.hpp"
// #endif




// All the functions with the spacename TriBas are used in this model
namespace TriBas
{


  //#######################################################################################################
  // DEFINING STRUCT
  //#######################################################################################################
  // Probably a temporary structure. I kind of struggle to get the node ordered in the right way I need for
  // running bottom to top my model.
  // sortation structure attempts to solve that in a hacky way where I am using a priority queue 
  // to sort my rivers bottom to top.
  struct sortation
  {
    // Vector of river node
    std::vector<int> Node_of_river;
    // Elevation at eh base of the river
    float base_elevation;
  };

  // These are the operator used to sort the river per grater/lower elevation in the priority queue
  bool operator>( const sortation& lhs, const sortation& rhs )
  {
    return lhs.base_elevation > rhs.base_elevation;
  }
  bool operator<( const sortation& lhs, const sortation& rhs )
  {
    return lhs.base_elevation < rhs.base_elevation;
  }
  //#######################################################################################################
  //#######################################################################################################



	// Alright let's begin that. B.G. 21/11/2018 at 4 AM in Edinburgh's airport

  //#######################################################################################################
  //#######################################################################################################
  // MAIN FUNCTIONS


  ///@brief Functions that sets up the models from a range of parameters.
  ///@details The model needs a specific structure to be build once. This function etract the river network and all the information the models need to know to run (e.g., node IDs and their receivers, elevqtions, ...).
  ///@details It requires Information to understance the raster and on what you want to calculate mchi which basin and which channel heads you want.
  ///@details It then returns a map of all these data ready to be ran in the run function(s)
  ///@param nrows (int): Number of rows of your raster
  ///@param ncols (int): Number of cols of your raster
  ///@param xmin (float): the minimum x of your raster
  ///@param ymin (float): the minimum y of your raster
  ///@param ndv (float): No Data Value (Which value I will ignore)
  ///@param data (2Dtensor float): the raster array
  ///@param x_sources (1Dtensor double): x coordinates of our cannel heads (Same coordinate system than our raster!!)
  ///@param y_sources (1Dtensor double): y coordinates of our cannel heads (Same coordinate system than our raster!!)
  ///@param Northing: basin outlet y coordinate (Same coordinate system than our raster!!)
  ///@param Easting: basin outlet x coordinate (Same coordinate system than our raster!!)
  ///@param search_node_radius (int): How many nodes should I check around your basin outlet input for the nearest consequent Junction
  ///@param target_nodes (int)/n_iteration (int)/ skip (int)/ minimum_segment_length/ sigma (int)/ m (float)/ n (float)/ A0 (float): parameters linked to mchi extraction (see Mudd et al., 2014 and associated documentation for details).
  ///@return A map of vectors (Node, Elevation,X,Y...)
  ///@authors B.G.
  ///@date 25/11/2018
  std::map<std::string, xt::pytensor<float,1> > setting_up_TriBAs(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, 
    float m, float n, float A0, double Northing, double Easting, int search_radius_nodes,
    int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma)
  {

    std::cout << "I am loading the data into LSDTT" << std::endl;
    // This will host the data. At the moment I need to copy the data from pytensors to TNT array. Not ideal but No other solution unfortunately.
    // Memory leak can become a problem for realy big rasters, however, the time loss is not important (few milliseconds).
    TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
    for(size_t i=0;i<nrows;i++)
    for(size_t j=0;j<ncols;j++)
      data_pointerast[i][j] = data(i,j);

    // Data is copied! I am ready to feed a LSDRaster object
    // Here we go: 
    LSDRaster PP_raster(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);

    // Now I need all the flow informations
    std::vector<std::string> bound = {"n","n","n","n"};
    LSDFlowInfo FlowInfo(bound,PP_raster);
    // OK I got it
    
    // Calculating the flow accumulation and the ...
    LSDIndexRaster FlowAcc = FlowInfo.write_NContributingNodes_to_LSDIndexRaster();
    // Drainage area
    LSDRaster DrainageArea = FlowInfo.write_DrainageArea_to_LSDRaster();
    // DOne

    // Getting the flow distance
    LSDRaster DistanceFromOutlet = FlowInfo.distance_from_outlet();

    // Initialising the channel heads
    std::vector<int> sources; //  sources will contain the node index (unique identifer for each node) of me sources
    // # I first need to copy the xy cordinates into vector for lsdtt
    std::vector<float> x_sources_v, y_sources_v;
    for (size_t yuio = 0; yuio<x_sources.size(); yuio ++)
      {x_sources_v.push_back(x_sources(yuio));y_sources_v.push_back(y_sources(yuio));}
    // ~ DOne
    sources = FlowInfo.Ingest_Channel_Heads(x_sources_v,y_sources_v);
    // I got me channels head

    // Lets extract my channels
    // now get the junction network
    LSDJunctionNetwork JunctionNetwork(sources, FlowInfo);
    // Ok, great, I preprocessed all my raw rivers

    // Now let's get the basin outlet
    // # A bit of hacky way to get the data into the right format
    std::vector<float> fUTM_easting; fUTM_easting.push_back(Easting);
    std::vector<float> fUTM_northing; fUTM_northing.push_back(Northing);
    int threshold_stream_order = 2;
    std::vector<int> valid_cosmo_points;         // a vector to hold the valid nodes
    std::vector<int> snapped_node_indices;       // a vector to hold the valid node indices
    std::vector<int> snapped_junction_indices;   // a vector to hold the valid junction indices
    // Sending the coordinates to the outlet tracker
    JunctionNetwork.snap_point_locations_to_channels(fUTM_easting, fUTM_northing, 
                search_radius_nodes, threshold_stream_order, FlowInfo, 
                valid_cosmo_points, snapped_node_indices, snapped_junction_indices);
    std::vector<int> BaseLevelJunctions = snapped_junction_indices;
    // Done

    // Results check
    if(BaseLevelJunctions.size() == 0)
    {
      std::cout << "ERROR: I couldn't find any basin with your criteri. Increase search_radius_nodes maybe??" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Let me tell you outside of the code just in case you are wondering what is happenning
    std::cout << "Successfully detected the outlet of a basin! Hopefully the one you had in mind" << std::endl;

    // Ignore but keep that in case
    // BaseLevelJunctions = JunctionNetwork.Prune_Junctions_If_Nested(BaseLevelJunctions,FlowInfo, FlowAcc);
    // Small but required cleaning
    std::vector<int> outlet_nodes, baselevel_node_of_each_basin;
    JunctionNetwork.get_overlapping_channels(FlowInfo, BaseLevelJunctions, DistanceFromOutlet,
                                    sources, outlet_nodes,baselevel_node_of_each_basin,10);

    // The steps below are getting (i) mchi and related values and (ii) the node structure of our river network
    // Now let's get the chitools N stuffs
    LSDChiTools ChiTool(FlowInfo);
    // Getting chi because we kind of need it
    LSDRaster chi_coordinate = FlowInfo.get_upslope_chi_from_all_baselevel_nodes(m/n,A0,1);
    // getting mchi
    ChiTool.chi_map_automator(FlowInfo, sources, outlet_nodes, baselevel_node_of_each_basin,
                          PP_raster, DistanceFromOutlet,
                          DrainageArea, chi_coordinate, target_nodes,
                          n_iterations, skip, minimum_segment_length, sigma);
    const int n_dfdg = 10000;
    ChiTool.segment_counter(FlowInfo, n_dfdg);

    // Now getting the ndoe sequence and the data maps
    // The node_sequence correspond to the ordered fastscape node index
    // The maps contains elevation, flow distance ... for each of these node indexes
    std::map<std::string, std::map<int,float> > mahmaps =  ChiTool.get_data_maps();
    std::vector<int> node_sequence =  ChiTool.get_vectors_of_node();

    // Generating the data structure for the output
    std::map<std::string,xt::pytensor<float,1> > output;
    // # THat's the way you create a xtensor array
    std::array<size_t, 1> shape = { node_sequence.size() };
    xt::xtensor<float,1> elev(shape);
    xt::xtensor<float,1> chi(shape);
    xt::xtensor<float,1> flowdis(shape);
    xt::xtensor<float,1> node_seq(shape);
    xt::xtensor<float,1> rec(shape);
    xt::xtensor<float,1> SK(shape);
    xt::xtensor<float,1> DA(shape);
    xt::xtensor<int,1> row(shape);
    xt::xtensor<int,1> col(shape);
    xt::xtensor<float,1> these_x(shape);
    xt::xtensor<float,1> these_y(shape);
    xt::xtensor<float,1> mchi(shape);

    // OK now I am dealing with node ordering. I did not manage to extrct it correctly within LSDTT for some reason
    // So here is a quick and dirty way to order my nodes in a modelling friendly way
    // # First I am creating a priority queue with a greater logic 
    std::priority_queue< sortation, std::vector<sortation>, std::greater<sortation> > myp;
    std::vector<int> temp_node, totnode;// temp variables
    // Sorting my rivers:
    float last_elevation_for_sorting = mahmaps["elevation"][node_sequence[0]];
    int this_SK = mahmaps["SK"][node_sequence[0]]; int last_SK = this_SK;
    for (size_t i=0;i<node_sequence.size(); i++)
    {
      // the idea is to separate each of my rivers and get the lower ones at the bottom of my queue
      // Getting the current cource key
      this_SK = mahmaps["SK"][node_sequence[i]];
      if(this_SK != last_SK)
      {
        // If I changed e source key (my river chunk) I ewant to save it into my queue
        // Nodes are sorted top to bottom per rivers -> here is the bottom to top one
        std::reverse(temp_node.begin(), temp_node.end());
        // Creating my node structure here, see description above
        sortation temp_sortation;
        temp_sortation.Node_of_river = temp_node; // vectore of nodes
        temp_sortation.base_elevation = last_elevation_for_sorting; // elevation of the lowest cell
        myp.push(temp_sortation); // feeding my queue (highly questionable sentence is translated in french)
        temp_node.clear(); // Getting it ready for my next loop run 
      }

      // I do that all the time anyway
      temp_node.push_back(node_sequence[i]); // adding my node to the vector
      // Saving the last elevation in case next one is a change
      last_elevation_for_sorting = mahmaps["elevation"][node_sequence[i]];
      // saving current sk to check if change
      last_SK = this_SK;
    }
    // I need to do it a last time to save the last river
    std::reverse(temp_node.begin(), temp_node.end());
    // Creating my node structure here, see description above
    sortation temp_sortation;
    temp_sortation.Node_of_river = temp_node; // vectore of nodes
    temp_sortation.base_elevation = last_elevation_for_sorting; // elevation of the lowest cell
    myp.push(temp_sortation); // feeding my queue (highly questionable sentence is translated in french)
    temp_node.clear(); // Getting it ready for my next loop run 
    // DOne
    // Now I kind of need to recompile my data into one single tensor
    while(myp.size()>0) // While my priority queue is not empty
    {
      // Getting my vector of lowest river nodes
      std::vector<int> this_veconde;
      sortation hugh = myp.top(); // this gives the elements at the top of the queue
      this_veconde = hugh.Node_of_river;
      myp.pop(); //  this removes the abovementioned river from my stack
      // Now simply pushing the node into me global node vector
      for (size_t u =0 ; u < this_veconde.size() ; u++) {totnode.push_back(this_veconde[u]);}
    }
    // Nodes should be ordered now

    // Last but not least I am feeding my output map. Not detailing here as it is simple copy.
    for(size_t i=0; i<totnode.size(); i++)
    {
      int this_node = totnode[i]; int tr; int tc;
      FlowInfo.retrieve_current_row_and_col(this_node,tr,tc);
      node_seq[i] = this_node;
      int recnode =0;
      FlowInfo.retrieve_receiver_information(this_node,recnode, tr,tc);
      rec[i] = recnode;
      chi[i] = mahmaps["chi"][this_node];
      flowdis[i] = mahmaps["flow_distance"][this_node];
      SK[i] = mahmaps["SK"][this_node];
      DA[i] = mahmaps["DA"][this_node];
      row[i] = tr;
      col[i] = tc;
      float tx,ty;
      FlowInfo.get_x_and_y_locations(tr,tc,tx,ty);
      these_x[i] = tx;
      these_y[i] = ty;
      elev[i] = mahmaps["elevation"][this_node];
      mchi[i] = mahmaps["m_chi"][this_node];

      // if(elev[i] == -9999 || mahmaps["elevation"].count(this_node) == 0)
      //   {std::cout << "GARGGGGGGG: " << elev[i] << std::endl;}
      // if(elev[i] == -9999 || mahmaps["elevation"].count(recnode) == 0)
      //   {std::cout << "GARGGGGGGG_recnode: " << elev[i] << std::endl;}
      if(mahmaps["elevation"].count(recnode) == 0)
      {
        bool keep_checking = true;
        int noronode = 0;
        while(keep_checking)
        {
          FlowInfo.retrieve_receiver_information(recnode,noronode,tr,tc);
          if(mahmaps["elevation"].count(noronode)==0 || noronode == -9999)
          {
            keep_checking = false;
          }
          else
          {
            recnode = noronode;
          }
        }
        rec[i] = noronode;
      }
    }

    output["nodeID"] = node_seq;
    output["chi"] = chi;
    output["elevation"] = elev;
    output["flow_distance"] = flowdis;
    output["source_key"] = SK;
    output["drainage_area"] = DA;
    output["receivers"] = rec;
    output["row"] = row;
    output["col"] = col;
    output["x"] = these_x;
    output["y"] = these_y;
    output["m_chi"] = mchi;

    // Ready to return!!
    return output;
  }


  ///@brief this is the basic run of the model: it runs on a prebuilt model for a given time, a spatial uplift field and a spatial erodibility and given m and n.
  ///@details requires a prebuilt model to run. It calculate and save for given time steps the resulting elevation at the end of the run.
  ///@param meq, neq (flaot): the SPL constants
  ///@param n_dt (int): the nuber of time step to run 
  ///@param timestep (int): The value in years of each time step
  ///@param save_step (int): sve the elevation numpy array every X time step. Beware of your memory for large run!! I'll implement a disk I/O option if it becomes a problem
  ///@param uplift (xtensor<float,1>): the spatially variable uplift field (Doesn't have to be variable though, it has to be an array with the same dimension then the input)
  ///@param erodability_K (xtensor<float,1>): the spatially variable erodibility (Doesn't have to be variable though, it has to be an array with the same dimension then the input)
  ///@param prebuilt (map<string,xtensor<float,1>>): the prebuilt model, ouput from the building functions. It contains useful informationand the initial condition of the model.
  ///@return A map/dictionnary(in python) of xtensor/array(in python) of the elevation at each saved time steps.
  ///@authors B.G.
  ///@date 27/11/2018
  std::map<std::string, xt::pytensor<float,1> > run_model(float meq, float neq, int n_timestep, int timestep, int save_step, xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, std::map<std::string, xt::pytensor<float,1> >& prebuilt )
  {
    
    // Creating the final output to fill
    std::map<std::string, xt::pytensor<float,1> > output;

    //global size of my arrays
    size_t siz = prebuilt["elevation"].size();

    // Setting arrays to store the new and last elevation
    std::array<size_t, 1> shape = { siz };
    xt::xtensor<float,1> last_elev(shape), this_new_elev(shape);
    // Setting base variables I need
    float baselevel = prebuilt["elevation"][0]; // elevation of the very first element. So far has to be fixed
    float last_h = baselevel, this_h=0; // navigating through the different elevtion layers
    float tol = 1e-3;// Tolerance for the newton-rhapson itarative solver for the stream power law
    int this_node,last_node;// Navigating through the node and its receivers
    last_elev = prebuilt["elevation"]; // initializing our previous elevation to the first one

    // Alright let's run the model
    // for(int tTt = 0; tTt < 1; tTt++) // Ingore that it's when I need to only run one model to check some stuff
    // Real run here
    for(int tTt = 0; tTt < n_timestep; tTt++)
    {
      // These maps are storing the elevation and flow distance for each node IDs. 
      // this idea is to rn the model bottom up, store informations for current nodes and 
      // therefore get the receiver elevation/flow distance of previously processed node.
      std::map<int,float> rec_to_elev, rec_to_fd;
      // How far are we in the model. Let me stop my work here, there are pretty intense turbulences in my flights.
      // OK seems better now, f*ck these were impressive
      // So I was saying that we want to know which timestep we are at.
      std::cout << "Processing: " << tTt+1 << "||" << n_timestep << "\r"; // the last sign means "Return to the beginning of my line and flush"
      // Running through the river network:
      for(size_t i=0; i<siz;i++)
      {
        // We are in the first element, it is a bit different so we need a special case:
        // Saving elevation and stuff as baselevel, and giving the first receiver information
        if(i==0)
          {this_new_elev[i]=baselevel;last_h = baselevel; rec_to_elev[prebuilt["nodeID"][i]] = baselevel;rec_to_fd[prebuilt["nodeID"][i]] = prebuilt["flow_distance"][i];continue;}
        // Done, the continue statement at the end of last if makes sure it stop this loop run here if i == 0

        // I am getting my receiver and current nodes (available in the prebuilt model)
        last_node = prebuilt["receivers"][i];
        this_node = prebuilt["nodeID"][i];
        // Getting the length between these nodes (dx)
        float length = prebuilt["flow_distance"][i] - rec_to_fd[last_node];
        // getting the flow distance ready when next nodes will query this one as receiver
        rec_to_fd[this_node] = prebuilt["flow_distance"][i];
        // Getting the drainage area (from prebuilt model)
        float tDA = prebuilt["drainage_area"][i];
        // Getting the erodibility from input
        float keq = erodility_K[i];
        // Newton-rahpson method to solve non linear equation
        float epsilon;     // in newton's method, z_n+1 = z_n - f(z_n)/f'(z_n)
                         // and here epsilon =   f(z_n)/f'(z_n)
                         // f(z_n) = -z_n + z_old - dt*K*A^m*( (z_n-z_r)/dx )^n
                         // We differentiate the above equation to get f'(z_n)
                         // the resulting equation f(z_n)/f'(z_n) is seen below
        // A bit of factorisation to clarify the equation
        float streamPowerFactor = keq * pow(tDA, meq) * timestep;
        float slope; // Slope.
        float dx = length; // Embarrassing renaming of variables because I am lazy and got confused
        float new_zeta = last_elev[i]; float old_zeta = new_zeta; // zeta = z = elevation
        // iterate until you converge on a solution. Uses Newton's method.
        int iter_count = 0;
        do
        {
          // Get the slope
          slope = (new_zeta - rec_to_elev[last_node]) / dx;
          // Check backslope or no slope ie no erosion
          if(slope <= 0)
          {
            epsilon = 0;
          }
          else
          {
            // Applying the newton's method
            epsilon = (new_zeta - old_zeta + streamPowerFactor * std::pow(slope, neq)) /
                 (1 + streamPowerFactor * (neq/dx) * std::pow(slope, neq-1));
          }
          // iterate the result
          new_zeta -= epsilon;

          // This limits the number of iterations, it is a safety check to avoid infinite loop
          // Thsi will begin to split some  nan or inf if it diverges
          iter_count++;
          if(iter_count > 100)
          {
            // std::cout << "Too many iterations! epsilon is: " << std::abs(epsilon) << std::endl;
            epsilon = 0.5e-6;
          }
          // I want it to run while it can still have an effect on the elevation
        } while (abs(epsilon) > tol);

        // Avioding inversion there!!
        if(new_zeta < rec_to_elev[last_node])
          new_zeta = rec_to_elev[last_node];

        // embarassing renaming here as well
        float hnew = new_zeta;

        // Applying the new elevation field
        this_new_elev[i] = hnew;
        rec_to_elev[this_node] = hnew;
      }
      // Done with the elevation stuff

      // computing uplift
      for(size_t i=1; i<siz;i++)
      {
        this_new_elev[i] = this_new_elev[i] + uplift[i]*timestep;
      }

      // Saving this step if required
      // and adding to the global map if required
      if(tTt != 0)
      {
        if((tTt%save_step) == 0)
        {
          std::string gh = itoa(tTt * timestep); // Calculate the real time
          output[gh] = this_new_elev;
        }
      }
      // IMPoORTANT: giving the next elevation the new one
      last_elev = this_new_elev;
    }

    //We are basically done
    return output;


  }

  // Brief description before real doc: This burns to any map <string,float> ontaining x and y key the underlying raster info
  // No security check, make sure your rasters are good
  std::map<string,xt::pytensor<float,1> > burn_external_to_map(std::map<string,xt::pytensor<float,1> >& prebuilt, int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
    xt::pytensor<float,2>& data, std::string name_of_burned_column)
  {
    std::cout << "I am loading the data into LSDTT" << std::endl;
    // This will host the data. At the moment I need to copy the data from pytensors to TNT array. Not ideal but No other solution unfortunately.
    // Memory leak can become a problem for realy big rasters, however, the time loss is not important (few milliseconds).
    TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
    for(size_t i=0;i<nrows;i++)
    for(size_t j=0;j<ncols;j++)
      data_pointerast[i][j] = data(i,j);

    // Data is copied! I am ready to feed a LSDRaster object
    // Here we go: 
    LSDRaster PP_raster(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
    std::cout << "Extracting the xy data, assigning nodata if point outside of the raster." << std::endl;

    // creating the array for output
    size_t siz = prebuilt["x"].size();
    std::array<size_t, 1> shape = {siz};
    xt::xtensor<float,1> new_col(shape);

    for(size_t t = 0; t < prebuilt["x"].size(); t++)
    {
      float tx = prebuilt["x"][t];
      float ty = prebuilt["y"][t];
      if(PP_raster.check_if_point_is_in_raster(tx,ty)){new_col[t] = PP_raster.get_value_of_point(tx,ty);}
      else {new_col[t] = ndv;}
    }

    // Done if in place stuff worked. I didn't, I am returning the shit
    prebuilt[name_of_burned_column] = new_col;
    std::cout << "Entry " << name_of_burned_column << " burned to the dictionnary/map" << std::endl; 

    return prebuilt;
  }




  std::map<std::string, xt::pytensor<float,1> > run_to_steady_state(float meq, float neq, int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, std::map<std::string, xt::pytensor<float,1> >& prebuilt, int max_timestep )
  {

    // Creating the final output to fill
    std::map<std::string, xt::pytensor<float,1> > output;

    //global size of my arrays
    size_t siz = prebuilt["elevation"].size();

    // Setting arrays to store the new and last elevation
    std::array<size_t, 1> shape = { siz };
    xt::xtensor<float,1> last_elev(shape), this_new_elev(shape);
    // Setting base variables I need
    float baselevel = prebuilt["elevation"][0]; // elevation of the very first element. So far has to be fixed
    float last_h = baselevel, this_h=0; // navigating through the different elevtion layers
    float tol = 1e-3;// Tolerance for the newton-rhapson itarative solver for the stream power law
    int this_node,last_node;// Navigating through the node and its receivers
    last_elev = prebuilt["elevation"]; // initializing our previous elevation to the first one

    // Alright let's run the model
    // for(int tTt = 0; tTt < 1; tTt++) // Ingore that it's when I need to only run one model to check some stuff
    // Real run here
    int tTt = 0;
    bool not_steady_state = true;
    float ratio_tol = 0;
    while(not_steady_state && tTt < max_timestep)
    {

      tTt++;  // just a simple counter

      // These maps are storing the elevation and flow distance for each node IDs. 
      // this idea is to rn the model bottom up, store informations for current nodes and 
      // therefore get the receiver elevation/flow distance of previously processed node.
      std::map<int,float> rec_to_elev, rec_to_fd;
      // How far are we in the model. Let me stop my work here, there are pretty intense turbulences in my flights.
      // OK seems better now, f*ck these were impressive
      // So I was saying that we want to know which timestep we are at.
      // std::cout << "Processing time step: " << tTt+1 << " and tolerance ratio is " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"
      // Running through the river network:
      for(size_t i=0; i<siz;i++)
      {
        // We are in the first element, it is a bit different so we need a special case:
        // Saving elevation and stuff as baselevel, and giving the first receiver information
        if(i==0)
          {this_new_elev[i]=baselevel;last_h = baselevel; rec_to_elev[prebuilt["nodeID"][i]] = baselevel;rec_to_fd[prebuilt["nodeID"][i]] = prebuilt["flow_distance"][i];continue;}
        // Done, the continue statement at the end of last if makes sure it stop this loop run here if i == 0

        // I am getting my receiver and current nodes (available in the prebuilt model)
        last_node = prebuilt["receivers"][i];
        this_node = prebuilt["nodeID"][i];
        // Getting the length between these nodes (dx)
        float length = prebuilt["flow_distance"][i] - rec_to_fd[last_node];
        // getting the flow distance ready when next nodes will query this one as receiver
        rec_to_fd[this_node] = prebuilt["flow_distance"][i];
        // Getting the drainage area (from prebuilt model)
        float tDA = prebuilt["drainage_area"][i];
        // Getting the erodibility from input
        float keq = erodility_K[i];
        // Newton-rahpson method to solve non linear equation
        float epsilon;     // in newton's method, z_n+1 = z_n - f(z_n)/f'(z_n)
                         // and here epsilon =   f(z_n)/f'(z_n)
                         // f(z_n) = -z_n + z_old - dt*K*A^m*( (z_n-z_r)/dx )^n
                         // We differentiate the above equation to get f'(z_n)
                         // the resulting equation f(z_n)/f'(z_n) is seen below
        // A bit of factorisation to clarify the equation
        float streamPowerFactor = keq * pow(tDA, meq) * timestep;
        float slope; // Slope.
        float dx = length; // Embarrassing renaming of variables because I am lazy and got confused
        float new_zeta = last_elev[i]; float old_zeta = new_zeta; // zeta = z = elevation
        // iterate until you converge on a solution. Uses Newton's method.
        int iter_count = 0;
        do
        {
          // Get the slope
          slope = (new_zeta - rec_to_elev[last_node]) / dx;
          // Check backslope or no slope ie no erosion
          if(slope <= 0)
          {
            epsilon = 0;
          }
          else
          {
            // Applying the newton's method
            epsilon = (new_zeta - old_zeta + streamPowerFactor * std::pow(slope, neq)) /
                 (1 + streamPowerFactor * (neq/dx) * std::pow(slope, neq-1));
          }
          // iterate the result
          new_zeta -= epsilon;

          // This limits the number of iterations, it is a safety check to avoid infinite loop
          // Thsi will begin to split some  nan or inf if it diverges
          iter_count++;
          if(iter_count > 100)
          {
            // std::cout << "Too many iterations! epsilon is: " << std::abs(epsilon) << std::endl;
            epsilon = 0.5e-6;
          }
          // I want it to run while it can still have an effect on the elevation
        } while (abs(epsilon) > tol);

        // Avioding inversion there!!
        if(new_zeta < rec_to_elev[last_node])
          new_zeta = rec_to_elev[last_node];

        // embarassing renaming here as well
        float hnew = new_zeta;

        // Applying the new elevation field
        this_new_elev[i] = hnew;
        rec_to_elev[this_node] = hnew;
      }
      // Done with the elevation stuff

      // computing uplift
      for(size_t i=1; i<siz;i++)
      {
        this_new_elev[i] = this_new_elev[i] + uplift[i]*timestep;
      }

      // Saving this step if required
      // and adding to the global map if required
      if(tTt != 0)
      {
        if((tTt%save_step) == 0)
        {
          std::string gh = itoa(tTt * timestep); // Calculate the real time
          output[gh] = this_new_elev;
        }
      }


      // Now checking if we keep on eroding
      // Map has these entry:
      // delta_median
      // delta_mean
      // delta_stddev
      // delta_stderr
      // delta_FQ
      // delta_TQ
      // n_above
      // n_below
      // n_tolerated
      // n_untolerated
          // Setting arrays to store the new and last elevation
      xt::xtensor<float,1> flowdist(shape);
      flowdist = prebuilt["flow_distance"];
      std::map<std::string, float > compstat; compstat = calculate_comparative_metrics(last_elev, this_new_elev, flowdist, 50, delta_tolerance);
      ratio_tol = compstat["n_tolerated"]/siz;
      std::cout << "TS: " << tTt+1 << " tolrat: " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"

      // Let's try this simple approach:
      // Steady-state = when most of my points doesn't evolve?
      if(ratio_tol>ratio_of_tolerance && tTt > min_iterations)
      {
        not_steady_state = false;
      }

      // IMPoORTANT: giving the 
      // next elevation the new one
      last_elev = this_new_elev;
    }

    // Just saving the last time_step here
    std::string gh = itoa(tTt * timestep); // Calculate the real time
    output[gh] = this_new_elev;


    //We are basically done
    return output;

  }


  ///@brief Internal building function to generate Multi Threading Order (MTO) for my river system. 
  ///@details Multithreading is organised in the following way: first my main river is processed with a single thread (MTO=1)
  ///@details then all the channels draining into this first one can be processed parallely (MTO=2)
  ///@details This process can be repeated until the highest MTO is reached
  ///@param prebuilt: a classic prebuilt of TribBass
  ///@return I don't know yet actually...
  ///@author B.G.
  ///@Date 02/12/2018
  std::map<std::string, xt::pytensor<float,1> > generate_MTO(std::map<std::string, xt::pytensor<float,1> >& prebuilt)
  {

    size_t siz = prebuilt["elevation"].size();
    std::array<size_t,1> shape = {siz};
    xt::xtensor<float,1> MTO(shape);
    std::map<int, int> node_to_MTO;
    int this_MTO = 1, this_SK = prebuilt["elevation"][0]; int last_SK = this_SK;
    for(size_t i=0; i<prebuilt["elevation"].size(); i++)
    {
      int this_node = prebuilt["nodeID"][i];
      int receiver_node = prebuilt["receivers"][i];
      this_SK = prebuilt["source_key"][i];
      if(last_SK != this_SK)
      {
        this_MTO = node_to_MTO[receiver_node]+1;
      }

      node_to_MTO[i] = this_MTO;
      MTO[i] = this_MTO;
    }

    prebuilt["MTO"] = MTO;

    // OK Now I have my MTO, I need to gather the index info for each node
    std::pair<int,int> this_first_to_last;
    std::vector<std::pair<int,int> > these_pair_of_ID ; // This will store all pair of first-to-last element
    std::map<int, std::vector<std::pair<int,int> > > MTO_to_idx;
    for(size_t i=0; i< siz; i++)
    {

    }


  }



  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################
  //###################################################################################################




  //###################################################################################################
  //###################################################################################################
  //Internal_functions
  //###################################################################################################


  ///@brief Calculates basic m_chi (dz/dchi) inplace on given xtensors
  ///@param chi,elevation and m_chi xtensor<float,1>.
  ///@param node_index (xtensor<int,1>): contains the ID of each node
  ///@param node_to_rec (map<int,int>): contains the receiver of each node ID
  ///@author B.G.
  ///@date 27/11/2018
  void calculate_m_chi_basic_inplace(xt::xtensor<float,1>& elevation, xt::xtensor<float,1>& chi, xt::xtensor<float,1>& m_chi, xt::xtensor<int,1>& node_index, std::map<int,int>& node_to_rec)
  {

    // Few variables I need first
    std::map<int,float> rec_to_elev,rec_to_chi; // This will contain the lasts chi and elevation for receivers. Thanks to node ordering it can be filled in time.
    int this_node, last_node;
    //Let's loop
    for(size_t i =0; i< elevation.size();i++)
    {
      //First run = simple
      if(i==0){this_node=node_index[i];rec_to_elev[this_node]=elevation[i];rec_to_chi[this_node]=chi[i];m_chi[i]=0;continue;}

      //Others:
      this_node=node_index[i];
      float dz = elevation[i]-rec_to_elev[this_node]; rec_to_elev[this_node] = elevation[i];
      float dchi = chi[i]-rec_to_chi[this_node]; rec_to_chi[this_node]=chi[i];
      m_chi[i]=dz/dchi;
    }
  }

  ///@brief Evaluate a series of metrics to compare 2 1D y profiles having a common x.
  ///@brief This is to help detecting steady-state.
  ///@param profile 1 and profile 2 xtensor<float,1> and their x
  ///@param x_bin (float): the binning for the median profile NOT READY YET, NOT SURE IF I WILL NEED IR
  ///@param tolerance_detla (float): simply counts how many points are within a tolerance compare to their brother 
  ///@return comparative_metrics (map<string,xtensor<float,1> >)
  ///@return -> So far it has only the global comparison: 
  ///@return -> delta_median, delta_mean, delta_stddev, delta_stderr, delta_FQ, delta_TQ, n_above, n_below, n_tolerated, n_untolerated
  ///@autors B.G.
  ///@date 27/11/2018
  std::map<std::string, float > calculate_comparative_metrics(xt::xtensor<float,1>& elevation_1, xt::xtensor<float,1>& elevation_2, xt::xtensor<float, 1>& common_x, float x_bin, float tolerance_delta)
  {
    std::map<std::string, float > output;

    size_t sizla = elevation_1.size();
    // std::array<size_t,1> shape = {sizla};
    // xt::xtensor<float,1> temp_output(shape);
    std::vector<float> temp_delta; // I need a temp vector to use the different statstools in LSDTT 
    int n_below=0, n_above=0, n_within_tol = 0, n_outside_tol = 0; // stores how many values  are above or below
    // First I want to get the delta and associated stats
    for (size_t i=0; i<sizla; i++)
    {
      float this_delta = elevation_1[i]-elevation_2[i];
      temp_delta.push_back(this_delta);
      if(this_delta>=0){n_above++;}else{n_below++;}
      if(abs(this_delta)>tolerance_delta){n_outside_tol++;}else{n_within_tol++;}
    }
    // Alright let's get basic stats out of that now
    output["delta_median"] = get_median(temp_delta);
    float mean = get_mean(temp_delta);
    output["delta_mean"] = mean;
    output["delta_stddev"] = get_standard_deviation(temp_delta, mean);
    output["delta_stderr"] = get_standard_error(temp_delta, output["delta_stddev"]);
    output["delta_FQ"] = get_percentile(temp_delta, 25);
    output["delta_TQ"] = get_percentile(temp_delta, 75);
    output["n_above"] = float(n_above); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
    output["n_below"] = float(n_below); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
    output["n_tolerated"] = float(n_within_tol); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
    output["n_untolerated"] = float(n_outside_tol); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it


    return output;
  }


};

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////// Builing a new Class here, cleaner approach now that first tests are promising ///////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////



void LSDTribBas::create()
{ 
  std::cout << "I need information to be created!" << std::endl; 
  std::exit(EXIT_FAILURE);
}

void LSDTribBas::create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
  xt::pytensor<float,2>& data, xt::pytensor<float,1>& x_sources, xt::pytensor<float,1>& y_sources, 
  float m, float n, float A0, double Northing, double Easting, int search_radius_nodes,
  int target_nodes, int n_iterations, int skip, int minimum_segment_length, int sigma)
{
  std::cout << "I am loading the data into LSDTT" << std::endl;
  // This will host the data. At the moment I need to copy the data from pytensors to TNT array. Not ideal but No other solution unfortunately.
  // Memory leak can become a problem for realy big rasters, however, the time loss is not important (few milliseconds, yes I benchmarked it because of my coding OCD).
  TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
    data_pointerast[i][j] = data(i,j);
  std::cout << "Loaded" << std::endl;
  // Global variable set: I want my m and n to be constant!! or eventually to really specify which one we want
  meq = m;
  neq = n;

  // Random counter to keep track of stuff ARG I SAID STUFF AGAIN.
  change_counter = 0;
  match_counter = 0;

  // Data is copied! I am ready to feed a LSDRaster object
  // Here we go: 
  LSDRaster PP_raster(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);

  // Now I need all the flow informations
  std::vector<std::string> bound = {"n","n","n","n"};
  LSDFlowInfo FlowInfo(bound,PP_raster);
  // OK I got it
  
  // Calculating the flow accumulation and the ...
  LSDIndexRaster FlowAcc = FlowInfo.write_NContributingNodes_to_LSDIndexRaster();
  // Drainage area
  LSDRaster DrainageArea = FlowInfo.write_DrainageArea_to_LSDRaster();
  // DOne

  // Getting the flow distance
  LSDRaster DistanceFromOutlet = FlowInfo.distance_from_outlet();

  // Initialising the channel heads
  std::vector<int> sources; //  sources will contain the node index (unique identifer for each node) of me sources
  // # I first need to copy the xy cordinates into vector for lsdtt
  std::vector<float> x_sources_v, y_sources_v;
  for (size_t yuio = 0; yuio<x_sources.size(); yuio ++)
    {x_sources_v.push_back(x_sources(yuio));y_sources_v.push_back(y_sources(yuio));}
  // ~ DOne
  sources = FlowInfo.Ingest_Channel_Heads(x_sources_v,y_sources_v);
  // I got me channels head

  // Lets extract my channels
  // now get the junction network
  LSDJunctionNetwork JunctionNetwork(sources, FlowInfo);
  // Ok, great, I preprocessed all my raw rivers

  // Now let's get the basin outlet
  // # A bit of hacky way to get the data into the right format
  std::vector<float> fUTM_easting; fUTM_easting.push_back(Easting);
  std::vector<float> fUTM_northing; fUTM_northing.push_back(Northing);
  int threshold_stream_order = 2;
  std::vector<int> valid_cosmo_points;         // a vector to hold the valid nodes
  std::vector<int> snapped_node_indices;       // a vector to hold the valid node indices
  std::vector<int> snapped_junction_indices;   // a vector to hold the valid junction indices
  // Sending the coordinates to the outlet tracker
  JunctionNetwork.snap_point_locations_to_channels(fUTM_easting, fUTM_northing, 
              search_radius_nodes, threshold_stream_order, FlowInfo, 
              valid_cosmo_points, snapped_node_indices, snapped_junction_indices);
  std::vector<int> BaseLevelJunctions = snapped_junction_indices;
  // Done

  // Results check
  if(BaseLevelJunctions.size() == 0)
  {
    std::cout << "ERROR: I couldn't find any basin with your criteri. Increase search_radius_nodes maybe??" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Let me tell you outside of the code just in case you are wondering what is happenning
  std::cout << "Successfully detected the outlet of a basin! Hopefully the one you had in mind" << std::endl;

  // Ignore but keep that in case
  // BaseLevelJunctions = JunctionNetwork.Prune_Junctions_If_Nested(BaseLevelJunctions,FlowInfo, FlowAcc);
  // Small but required cleaning
  std::vector<int> outlet_nodes, baselevel_node_of_each_basin;
  JunctionNetwork.get_overlapping_channels(FlowInfo, BaseLevelJunctions, DistanceFromOutlet,
                                  sources, outlet_nodes,baselevel_node_of_each_basin,10);

  // The steps below are getting (i) mchi and related values and (ii) the node structure of our river network
  // Now let's get the chitools N stuffs
  LSDChiTools ChiTool(FlowInfo);
  // Getting chi because we kind of need it
  LSDRaster chi_coordinate = FlowInfo.get_upslope_chi_from_all_baselevel_nodes(m/n,A0,1);
  // getting mchi
  ChiTool.chi_map_automator(FlowInfo, sources, outlet_nodes, baselevel_node_of_each_basin,
                        PP_raster, DistanceFromOutlet,
                        DrainageArea, chi_coordinate, target_nodes,
                        n_iterations, skip, minimum_segment_length, sigma);
  ChiTool.segment_counter(FlowInfo, 1000000);

  // Now getting the ndoe sequence and the data maps
  // The node_sequence correspond to the ordered fastscape node index
  // The maps contains elevation, flow distance ... for each of these node indexes
  std::map<std::string, std::map<int,float> > mahmaps =  ChiTool.get_data_maps();
  std::vector<int> node_sequence =  ChiTool.get_vectors_of_node();


  // # THat's the way you create a xtensor array
  std::array<size_t, 1> shape = { node_sequence.size() };
  xt::xtensor<float,1> TT_elev(shape);
  xt::xtensor<float,1> TT_chi(shape);
  xt::xtensor<float,1> TT_flowdis(shape);
  xt::xtensor<float,1> TT_node_seq(shape);
  xt::xtensor<float,1> TT_rec(shape);
  xt::xtensor<float,1> TT_SK(shape);
  xt::xtensor<float,1> TT_DA(shape);
  xt::xtensor<int,1> TT_row(shape);
  xt::xtensor<int,1> TT_col(shape);
  xt::xtensor<float,1> TT_these_x(shape);
  xt::xtensor<float,1> TT_these_y(shape);
  xt::xtensor<float,1> TT_mchi(shape);

  // OK now I am dealing with node ordering. I did not manage to extrct it correctly within LSDTT for some reason
  // So here is a quick and dirty way to order my nodes in a modelling friendly way
  // # First I am creating a priority queue with a greater logic 
  std::priority_queue< TriBas::sortation, std::vector<TriBas::sortation>, std::greater<TriBas::sortation> > myp;
  std::vector<int> temp_node, totnode;// temp variables
  // Sorting my rivers:
  float last_elevation_for_sorting = mahmaps["elevation"][node_sequence[0]];
  int this_SK = mahmaps["SK"][node_sequence[0]]; int last_SK = this_SK;
  for (size_t i=0;i<node_sequence.size(); i++)
  {
    // the idea is to separate each of my rivers and get the lower ones at the bottom of my queue
    // Getting the current cource key
    this_SK = mahmaps["SK"][node_sequence[i]];
    if(this_SK != last_SK)
    {
      // If I changed e source key (my river chunk) I ewant to save it into my queue
      // Nodes are sorted top to bottom per rivers -> here is the bottom to top one
      std::reverse(temp_node.begin(), temp_node.end());
      // Creating my node structure here, see description above
      TriBas::sortation temp_sortation;
      temp_sortation.Node_of_river = temp_node; // vectore of nodes
      temp_sortation.base_elevation = last_elevation_for_sorting; // elevation of the lowest cell
      myp.push(temp_sortation); // feeding my queue (highly questionable sentence is translated in french)
      temp_node.clear(); // Getting it ready for my next loop run 
    }

    // I do that all the time anyway
    temp_node.push_back(node_sequence[i]); // adding my node to the vector
    // Saving the last elevation in case next one is a change
    last_elevation_for_sorting = mahmaps["elevation"][node_sequence[i]];
    // saving current sk to check if change
    last_SK = this_SK;
  }
  // I need to do it a last time to save the last river
  std::reverse(temp_node.begin(), temp_node.end());
  // Creating my node structure here, see description above
  TriBas::sortation temp_sortation;
  temp_sortation.Node_of_river = temp_node; // vectore of nodes
  temp_sortation.base_elevation = last_elevation_for_sorting; // elevation of the lowest cell
  myp.push(temp_sortation); // feeding my queue (highly questionable sentence is translated in french)
  temp_node.clear(); // Getting it ready for my next loop run 
  // DOne
  // Now I kind of need to recompile my data into one single tensor
  while(myp.size()>0) // While my priority queue is not empty
  {
    // Getting my vector of lowest river nodes
    std::vector<int> this_veconde;
    TriBas::sortation hugh = myp.top(); // this gives the elements at the top of the queue
    this_veconde = hugh.Node_of_river;
    myp.pop(); //  this removes the abovementioned river from my stack
    // Now simply pushing the node into me global node vector
    for (size_t u =0 ; u < this_veconde.size() ; u++) {totnode.push_back(this_veconde[u]);}
  }
  // Nodes should be ordered now

  // Last but not least I am feeding my output map. Not detailing here as it is simple copy.
  for(size_t i=0; i<totnode.size(); i++)
  {
    int this_node = totnode[i]; int tr; int tc;
    FlowInfo.retrieve_current_row_and_col(this_node,tr,tc);
    TT_node_seq[i] = this_node;
    int recnode =0;
    FlowInfo.retrieve_receiver_information(this_node,recnode, tr,tc);
    TT_rec[i] = recnode;
    // std::cout << recnode << "||" << std::endl;
    recnode_global[this_node] = recnode;
    TT_chi[i] = mahmaps["chi"][this_node];
    TT_flowdis[i] = mahmaps["flow_distance"][this_node];
    TT_SK[i] = mahmaps["SK"][this_node];
    TT_DA[i] = mahmaps["DA"][this_node];
    TT_row[i] = tr;
    TT_col[i] = tc;
    float tx,ty;
    FlowInfo.get_x_and_y_locations(tr,tc,tx,ty);
    TT_these_x[i] = tx;
    TT_these_y[i] = ty;
    TT_elev[i] = mahmaps["elevation"][this_node];
    TT_mchi[i] = mahmaps["m_chi"][this_node];

    // if(elev[i] == -9999 || mahmaps["elevation"].count(this_node) == 0)
    //   {std::cout << "GARGGGGGGG: " << elev[i] << std::endl;}
    // if(elev[i] == -9999 || mahmaps["elevation"].count(recnode) == 0)
    //   {std::cout << "GARGGGGGGG_recnode: " << elev[i] << std::endl;}
    if(mahmaps["elevation"].count(recnode) == 0)
    {
      bool keep_checking = true;
      int noronode = 0;
      while(keep_checking)
      {
        FlowInfo.retrieve_receiver_information(recnode,noronode,tr,tc);
        if(mahmaps["elevation"].count(noronode)==0 || noronode == -9999)
        {
          keep_checking = false;
        }
        else
        {
          recnode = noronode;
        }
      }
      TT_rec[i] = recnode;
    }
  }

  // transferring to the global variables
  node_ID = TT_node_seq;
  chi = TT_chi;
  elevation = TT_elev;
  flow_distance = TT_flowdis;
  source_key = TT_SK;
  drainage_area = TT_DA;
  receivers = TT_rec;
  raster_row = TT_row;
  raster_col = TT_col;
  x_coord = TT_these_x;
  y_coord = TT_these_y;
  m_chi = TT_mchi;

  // MTO generation
  generate_MTO();

}

struct rivIndexByLength
{
  float length;
  std::pair<int,int> indexes_bounds;    
};
// These are the operator used to sort the river per grater/lower length in the priority queue
bool operator>( const rivIndexByLength& lhs, const rivIndexByLength& rhs )
{
  return lhs.length > rhs.length;
};
bool operator<( const rivIndexByLength& lhs, const rivIndexByLength& rhs )
{
  return lhs.length < rhs.length;
};

///@brief Internal building function to generate Multi Threading Order (MTO) for my river system. 
///@details Multithreading is organised in the following way: first my main river is processed with a single thread (MTO=1)
///@details then all the channels draining into this first one can be processed parallely (MTO=2)
///@details This process can be repeated until the highest MTO is reached
///@param prebuilt: a classic prebuilt of TribBass
///@return I don't know yet actually...
///@author B.G.
///@Date 02/12/2018
void LSDTribBas::generate_MTO()
{

  // Creating a structure only valid for this function: 
  // I need to sort my indexes within each MTO to get the longest river processed firts
  // using a priority queue with struct to do that

  size_t siz = elevation.size();
  std::array<size_t,1> shape = {siz};
  xt::xtensor<float,1> MTO(shape);
  std::map<int, int> node_to_MTO;
  int this_MTO = 1, this_SK = source_key[0]; int last_SK = this_SK;

  max_MTO = 0;
  for(size_t i=0; i<siz; i++)
  {
    int this_node = node_ID[i];
    int receiver_node = receivers[i];
    this_SK = source_key[i];
    if(last_SK != this_SK)
    {
      this_MTO = node_to_MTO[receiver_node]+1;
      if(this_MTO > max_MTO){max_MTO=this_MTO;}
    }

    node_to_MTO[this_node] = this_MTO;
    MTO[i] = this_MTO;
    last_SK = this_SK;
  }

  MTO_at_node = MTO;
  // max_MTO=this_MTO+1;
  // OK Now I have my MTO, I need to gather the index info for each node
  std::pair<int,int> this_first_to_last;
  std::vector<std::vector<std::pair<int,int> > > TGH(max_MTO);
  std::vector<std::pair<int,int> > these_pair_of_ID ; // This will store all pair of first-to-last element
  for(size_t t=0; t<TGH.size();t++){TGH[t]=these_pair_of_ID;}
  MTO_global_index = TGH;

  int first_ID = 0; int last_ID = 0; this_MTO = MTO_at_node[0]; int last_MTO = MTO_at_node[0];

  for(size_t i=0; i< siz; i++)
  {

    this_MTO = MTO_at_node[i];

    if(this_MTO != last_MTO)
    {

      this_first_to_last = std::make_pair(first_ID, i-1);

      first_ID = int(i);
      if((this_MTO-1)>= MTO_global_index.size()){std::cout << "DEBUG:: You should definitely not see this message... Node ordering can be fucked up: MTO = " << this_MTO << std::endl;}
      else{MTO_global_index[this_MTO-1].push_back(this_first_to_last);}


    }
    if(i == siz-1)
    {
      this_first_to_last = std::make_pair(first_ID, i);
      MTO_global_index[this_MTO-1].push_back(this_first_to_last);
    }
    last_MTO = this_MTO;
  }

  // Now last step is to sort my nodes into the priority queue
  for(size_t i=0; i<MTO_global_index.size(); i++)
  {
    // std::Lower because I need the biggest river to be first 
    std::priority_queue< rivIndexByLength, std::vector<rivIndexByLength>, std::less<rivIndexByLength> > datQ;
    std::vector<std::pair<int,int> > datVec = MTO_global_index[i];
    // Feeding the Queue
    for (size_t j=0; j<datVec.size(); j++)
    {
      // Getting all the data I need
      std::pair<int,int> datPair = datVec[j];
      int datFirstIndex = datPair.first;
      int datLastIndex = datPair.second;
      rivIndexByLength datRiv; datRiv.length = flow_distance[datFirstIndex] - flow_distance[datFirstIndex]; datRiv.indexes_bounds = datVec[j];
      // implementing
      datQ.push(datRiv);
    }

    //OK now I have my rivers by size
    // I need to refeed my MTO_gobal_index
    size_t cpt = 0;
    while(datQ.size()>0) // While my priority queue is not empty
    {
      // Getting my vector of lowest river nodes
      std::pair<int,int> datPair;
      rivIndexByLength hugh = datQ.top(); // this gives the elements at the top of the queue
      datPair = hugh.indexes_bounds;
      datQ.pop(); //  this removes the abovementioned river from my stack
      // Now simply correcting the 
      MTO_global_index[i][cpt] = datPair;
    }

  }


  // SHould be Done now


}




float LSDTribBas::newton_rahpson_solver(float keq, float meq, float neq, float tDA, float dt, float dx, float this_elevation, float receiving_elevation ,float tol)
{
  // Newton-rahpson method to solve non linear equation, see Braun and Willett 2013
  float epsilon;     // in newton's method, z_n+1 = z_n - f(z_n)/f'(z_n)
                   // and here epsilon =   f(z_n)/f'(z_n)
                   // f(z_n) = -z_n + z_old - dt*K*A^m*( (z_n-z_r)/dx )^n
                   // We differentiate the above equation to get f'(z_n)
                   // the resulting equation f(z_n)/f'(z_n) is seen below
  // A bit of factorisation to clarify the equation
  float streamPowerFactor = keq * pow(tDA, meq) * dt;
  float slope; // Slope.
  float new_zeta = this_elevation; float old_zeta = new_zeta; // zeta = z = elevation
  // iterate until you converge on a solution. Uses Newton's method.
  int iter_count = 0;
  do
  {
    // Get the slope
    slope = (new_zeta - receiving_elevation) / dx;
    // Check backslope or no slope ie no erosion
    if(slope <= 0)
    {
      epsilon = 0;
    }
    else
    {
      // Applying the newton's method
      epsilon = (new_zeta - old_zeta + streamPowerFactor * std::pow(slope, neq)) /
           (1 + streamPowerFactor * (neq/dx) * std::pow(slope, neq-1));
    }
    // iterate the result
    new_zeta -= epsilon;

    // This limits the number of iterations, it is a safety check to avoid infinite loop
    // Thsi will begin to split some  nan or inf if it diverges
    iter_count++;
    if(iter_count > 100)
    {
      std::cout << "Too many iterations! epsilon is: " << std::abs(epsilon) << std::endl;
      epsilon = 0.5e-6;
    }
    // I want it to run while it can still have an effect on the elevation
  } while (abs(epsilon) > tol);

  // Avioding inversion there!!
  if(new_zeta < receiving_elevation)
    new_zeta = receiving_elevation;

  // std::cout << receiving_elevation<< "||" <<this_elevation <<"||"<< new_zeta << std::endl;

  return new_zeta;
}

// Brief description before real doc: This burns to any map <string,float> ontaining x and y key the underlying raster info
// No security check, make sure your rasters are good
void LSDTribBas::ingest_external_data_from_xy(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, 
  xt::pytensor<float,2>& data, std::string name_of_burned_column)
{
  std::cout << "I am loading the data into LSDTT" << std::endl;
  // This will host the data. At the moment I need to copy the data from pytensors to TNT array. Not ideal but No other solution unfortunately.
  // Memory leak can become a problem for realy big rasters, however, the time loss is not important (few milliseconds).
  TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers
  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
    data_pointerast[i][j] = data(i,j);

  // Data is copied! I am ready to feed a LSDRaster object
  // Here we go: 
  LSDRaster PP_raster(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
  std::cout << "Extracting the xy data, assigning nodata if point outside of the raster." << std::endl;

  // creating the array for output
  size_t siz = elevation.size();
  std::array<size_t, 1> shape = {siz};
  xt::xtensor<float,1> new_col(shape);

  for(size_t t = 0; t < elevation.size(); t++)
  {
    float tx = x_coord[t];
    float ty = y_coord[t];
    if(PP_raster.check_if_point_is_in_raster(tx,ty)){new_col[t] = PP_raster.get_value_of_point(tx,ty);}
    else {new_col[t] = ndv;}
  }

  // Done if in place stuff worked. I didn't, I am returning the shit
  external_data[name_of_burned_column] = new_col;
  std::cout << "Entry " << name_of_burned_column << " burned to the LSDObject" << std::endl; 

}


std::map<std::string,xt::pytensor<float,1> > LSDTribBas::run_model(int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, int max_timestep )
{

  // Creating the final output to fill
  std::map<std::string, xt::pytensor<float,1> > output;

  //global size of my arrays (elevation is an attribute)
  size_t siz = elevation.size();

  // Setting arrays to store the new and last elevation
  std::array<size_t, 1> shape = { siz };
  xt::xtensor<float,1> last_elev(shape), this_new_elev(shape);
  // Setting base variables I need
  float baselevel = elevation[0]; // elevation of the very first element. So far has to be fixed
  float last_h = baselevel, this_h=0; // navigating through the different elevtion layers
  float tol = 1e-3;// Tolerance for the newton-rhapson itarative solver for the stream power law
  int this_node,last_node;// Navigating through the node and its receivers
  last_elev = elevation; // initializing our previous elevation to the first one

  // Alright let's run the model
  // for(int tTt = 0; tTt < 1; tTt++) // Ingore that it's when I need to only run one model to check some stuff
  // Real run here
  int tTt = 0;
  bool not_steady_state = true;
  float ratio_tol = 0;
  while(not_steady_state && tTt < max_timestep)
  {

    tTt++;  // just a simple counter

    // These maps are storing the elevation and flow distance for each node IDs. 
    // this idea is to rn the model bottom up, store informations for current nodes and 
    // therefore get the receiver elevation/flow distance of previously processed node.
    std::map<int,float> rec_to_elev, rec_to_fd;
    // How far are we in the model. Let me stop my work here, there are pretty intense turbulences in my flights.
    // OK seems better now, f*ck these were impressive
    // So I was saying that we want to know which timestep we are at.
    // std::cout << "Processing time step: " << tTt+1 << " and tolerance ratio is " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"
    // Running through the river network:
    for(size_t i=0; i<siz;i++)
    {
      // We are in the first element, it is a bit different so we need a special case:
      // Saving elevation and stuff as baselevel, and giving the first receiver information
      if(i==0)
        {this_new_elev[i]=baselevel;last_h = baselevel; rec_to_elev[node_ID[i]] = baselevel;rec_to_fd[node_ID[i]] = flow_distance[i];continue;}
      // Done, the continue statement at the end of last if makes sure it stop this loop run here if i == 0

      // I am getting my receiver and current nodes (available in the prebuilt model)
      last_node = receivers[i];
      // std::cout << "VAGUL" << last_node << std::endl;
      this_node = node_ID[i];
      // Getting the length between these nodes (dx)
      float length = flow_distance[i] - rec_to_fd[last_node];
      // getting the flow distance ready when next nodes will query this one as receiver
      rec_to_fd[this_node] = flow_distance[i];
      // Getting the drainage area (from prebuilt model)
      float tDA = drainage_area[i];
      // Getting the erodibility from input
      float keq = erodility_K[i];

      float hnew = newton_rahpson_solver(keq, meq, neq,  tDA,  timestep,  length, last_elev[i], rec_to_elev[last_node] ,tol);
       

      // Applying the new elevation field
      this_new_elev[i] = hnew;
      rec_to_elev[this_node] = hnew;
    }
    // Done with the elevation stuff

    // computing uplift
    for(size_t i=1; i<siz;i++)
    {
      this_new_elev[i] = this_new_elev[i] + uplift[i]*timestep;
    }

    // Saving this step if required
    // and adding to the global map if required
    if(tTt != 0)
    {
      if((tTt%save_step) == 0)
      {
        std::string gh = itoa(tTt * timestep); // Calculate the real time
        output[gh] = this_new_elev;
      }
    }


    // Now checking if we keep on eroding
    // Map has these entry:
    // delta_median
    // delta_mean
    // delta_stddev
    // delta_stderr
    // delta_FQ
    // delta_TQ
    // n_above
    // n_below
    // n_tolerated
    // n_untolerated
    // Setting arrays to store the new and last elevation
    xt::xtensor<float,1> flowdist(shape);
    flowdist = flow_distance;
    std::map<std::string, float > compstat; compstat = TriBas::calculate_comparative_metrics(last_elev, this_new_elev, flowdist, 50, delta_tolerance);
    ratio_tol = compstat["n_tolerated"]/siz;
    // std::cout << "TS: " << tTt+1 << " tolrat: " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"

    // Let's try this simple approach:
    // Steady-state = when most of my points doesn't evolve?
    if(ratio_tol>ratio_of_tolerance && tTt > min_iterations)
    {
      not_steady_state = false;
    }

    // IMPoORTANT: giving the 
    // next elevation the new one
    last_elev = this_new_elev;
  }

  // Just saving the last time_step here
  std::string gh = itoa(tTt * timestep); // Calculate the real time
  output[gh] = this_new_elev;

  // Saving my last stage because you might need it at some point
  last_elevation_modelled = this_new_elev;
  last_U_modelled = uplift;
  last_K_modelled = erodility_K;


  //We are basically done
  return output;

}

// Derive chi-elevation profile from custom chi and elevation array, but with the same receiver-node indexing
// this is suitable for modelled data
xt::pytensor<float,1> LSDTribBas::first_order_m_chi_from_custarray(xt::pytensor<float,1>& this_chi, xt::pytensor<float,1>& this_elevation)
{

  // Alright let's do it
  // output initialization
  size_t shape = chi.size();
  std::array<size_t,1> sizla = {shape};
  xt::xtensor<float,1> c_m_chi(sizla); c_m_chi[0] = 0;
  std::map<int,float> chi_receiver,elev_receiver;
  chi_receiver[node_ID[0]] = this_chi[0]; elev_receiver[node_ID[0]] = this_elevation[0];

  for(size_t i=0; i<shape; i++)
  {
    int this_node = node_ID[i]; int rec_node = receivers[i];
    float dz = this_elevation[i] - elev_receiver[rec_node];
    float dchi = this_chi[i] - chi_receiver[rec_node];
    chi_receiver[this_node] = this_chi[i];
    elev_receiver[this_node] = this_elevation[i];
    // Let's get mchi
    c_m_chi[i] = dz/dchi;

    // DEBUG
    // std::cout << c_m_chi[i] << std::endl;
    
    // Alright done
  }

  return c_m_chi;
}


///@brief Run the model, Multithreading version (I want to keep those separated, small catchment will be faster in serial).
///@brief Work in progress. Unstable.
std::map<std::string,xt::pytensor<float,1> > LSDTribBas::run_model_parallel(int timestep, int save_step, float ratio_of_tolerance, float delta_tolerance, int min_iterations , xt::pytensor<float,1>& uplift, xt::pytensor<float,1>& erodility_K, int max_timestep )
{
  
  // Creating the final output to fill
  std::map<std::string, xt::pytensor<float,1> > output;

  //global size of my arrays (elevation is an attribute)
  size_t siz = elevation.size();

  // Setting arrays to store the new and last elevation
  std::array<size_t, 1> shape = { siz };
  xt::xtensor<float,1> last_elev(shape), this_new_elev(shape);
  // Setting base variables I need
  float baselevel = elevation[0]; // elevation of the very first element. So far has to be fixed
  float last_h = baselevel, this_h=0; // navigating through the different elevtion layers
  float tol = 1e-3;// Tolerance for the newton-rhapson itarative solver for the stream power law
  int this_node,last_node;// Navigating through the node and its receivers
  last_elev = elevation; // initializing our previous elevation to the first one

  // Alright let's run the model
  // for(int tTt = 0; tTt < 1; tTt++) // Ingore that it's when I need to only run one model to check some stuff
  // Real run here
  int tTt = 0;
  bool not_steady_state = true;
  float ratio_tol = 0;
  while(not_steady_state && tTt < max_timestep)
  {

    tTt++;  // just a simple counter

    // These maps are storing the elevation and flow distance for each node IDs. 
    // this idea is to rn the model bottom up, store informations for current nodes and 
    // therefore get the receiver elevation/flow distance of previously processed node.
    std::map<int,float> rec_to_elev, rec_to_fd;
    // How far are we in the model. Let me stop my work here, there are pretty intense turbulences in my flights.
    // OK seems better now, f*ck these were impressive
    // So I was saying that we want to know which timestep we are at.
    // std::cout << "Processing time step: " << tTt+1 << " and tolerance ratio is " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"
    // Running through the river network:
    for(size_t i=0; i<siz;i++)
    {
      // We are in the first element, it is a bit different so we need a special case:
      // Saving elevation and stuff as baselevel, and giving the first receiver information
      if(i==0)
        {this_new_elev[i]=baselevel;last_h = baselevel; rec_to_elev[node_ID[i]] = baselevel;rec_to_fd[node_ID[i]] = flow_distance[i];continue;}
      // Done, the continue statement at the end of last if makes sure it stop this loop run here if i == 0

      // I am getting my receiver and current nodes (available in the prebuilt model)
      last_node = receivers[i];
      this_node = node_ID[i];
      // Getting the length between these nodes (dx)
      float length = flow_distance[i] - rec_to_fd[last_node];
      // getting the flow distance ready when next nodes will query this one as receiver
      rec_to_fd[this_node] = flow_distance[i];
      // Getting the drainage area (from prebuilt model)
      float tDA = drainage_area[i];
      // Getting the erodibility from input
      float keq = erodility_K[i];

      float hnew = newton_rahpson_solver(keq, meq, neq,  tDA,  timestep,  length, last_elev[i], rec_to_elev[node_ID[i]] ,tol);
       

      // Applying the new elevation field
      this_new_elev[i] = hnew;
      rec_to_elev[this_node] = hnew;
    }
    // Done with the elevation stuff

    // computing uplift
    for(size_t i=1; i<siz;i++)
    {
      this_new_elev[i] = this_new_elev[i] + uplift[i]*timestep;
    }

    // Saving this step if required
    // and adding to the global map if required
    if(tTt != 0)
    {
      if((tTt%save_step) == 0)
      {
        std::string gh = itoa(tTt * timestep); // Calculate the real time
        output[gh] = this_new_elev;
      }
    }


    // Now checking if we keep on eroding
    // Map has these entry:
    // delta_median
    // delta_mean
    // delta_stddev
    // delta_stderr
    // delta_FQ
    // delta_TQ
    // n_above
    // n_below
    // n_tolerated
    // n_untolerated
    // Setting arrays to store the new and last elevation
    xt::xtensor<float,1> flowdist(shape);
    flowdist = flow_distance;
    std::map<std::string, float > compstat; compstat = TriBas::calculate_comparative_metrics(last_elev, this_new_elev, flowdist, 50, delta_tolerance);
    ratio_tol = compstat["n_tolerated"]/siz;
    // std::cout << "TS: " << tTt+1 << " tolrat: " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"

    // Let's try this simple approach:
    // Steady-state = when most of my points doesn't evolve?
    if(ratio_tol>ratio_of_tolerance && tTt > min_iterations)
    {
      not_steady_state = false;
    }

    // IMPoORTANT: giving the 
    // next elevation the new one
    last_elev = this_new_elev;
  }

  // Just saving the last time_step here
  std::string gh = itoa(tTt * timestep); // Calculate the real time
  output[gh] = this_new_elev;

  // Saving my last stage because you might need it at some point
  last_elevation_modelled = this_new_elev;
  last_U_modelled = uplift;
  last_K_modelled = erodility_K;


  //We are basically done
  return output;

}



// ///@brief Evaluate a series of metrics to compare 2 1D y profiles having a common x.
// ///@brief This is to help detecting steady-state.
// ///@param profile 1 and profile 2 xtensor<float,1> and their x
// ///@param x_bin (float): the binning for the median profile NOT READY YET, NOT SURE IF I WILL NEED IR
// ///@param tolerance_detla (float): simply counts how many points are within a tolerance compare to their brother 
// ///@return comparative_metrics (map<string,xtensor<float,1> >)
// ///@return -> So far it has only the global comparison: 
// ///@return -> delta_median, delta_mean, delta_stddev, delta_stderr, delta_FQ, delta_TQ, n_above, n_below, n_tolerated, n_untolerated
// ///@autors B.G.
// ///@date 27/11/2018
// std::map<std::string, float > LSDTribBas::calculate_comparative_metrics(xt::xtensor<float,1>& elevation_1, xt::xtensor<float,1>& elevation_2, xt::xtensor<float, 1>& common_x, float x_bin, float tolerance_delta)
// {
//   std::map<std::string, float > output;

//   size_t sizla = elevation_1.size();
//   // std::array<size_t,1> shape = {sizla};
//   // xt::xtensor<float,1> temp_output(shape);
//   std::vector<float> temp_delta; // I need a temp vector to use the different statstools in LSDTT 
//   int n_below=0, n_above=0, n_within_tol = 0, n_outside_tol = 0; // stores how many values  are above or below
//   // First I want to get the delta and associated stats
//   for (size_t i=0; i<sizla; i++)
//   {
//     float this_delta = elevation_1[i]-elevation_2[i];
//     temp_delta.push_back(this_delta);
//     if(this_delta>=0){n_above++;}else{n_below++;}
//     if(abs(this_delta)>tolerance_delta){n_outside_tol++;}else{n_within_tol++;}
//   }
//   // Alright let's get basic stats out of that now
//   output["delta_median"] = get_median(temp_delta);
//   float mean = get_mean(temp_delta);
//   output["delta_mean"] = mean;
//   output["delta_stddev"] = get_standard_deviation(temp_delta, mean);
//   output["delta_stderr"] = get_standard_error(temp_delta, output["delta_stddev"]);
//   output["delta_FQ"] = get_percentile(temp_delta, 25);
//   output["delta_TQ"] = get_percentile(temp_delta, 75);
//   output["n_above"] = float(n_above); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
//   output["n_below"] = float(n_below); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
//   output["n_tolerated"] = float(n_within_tol); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it
//   output["n_untolerated"] = float(n_outside_tol); // Bad recasting here but I am lazy... class approach would solve that. I'll think about it


//   return output;
// }








void LSDTribBas::simplification_by_median(float flow_distance_bin)
{
  // This function should take the prebuit model and downgrade to smaller size. The idea is to optimise it by finding the right states for simpler profile before going to the full version of it
  // I Will use the MTO order to generate the simplification
  std::vector<int> node_simID_temp;
  for (int tmto = 0; tmto< max_MTO; tmto++)
  {
    // first step is to get each river bounds on 
    std::vector<std::pair<int,int> > these_river = MTO_global_index[tmto];
    for(size_t it1=0; it1<these_river.size();it1++)

    {
      size_t first_index = these_river[it1].first;
      size_t last_index = these_river[it1].second;
    
      // If my river is too small, I am not even bothering with it
      float size_of_this_river = flow_distance[last_index] - flow_distance[last_index];
      if(size_of_this_river <= flow_distance_bin)
      {std::cout << "getting there but on hold for now" << std::endl;}
    }

  }

}


void LSDTribBas::generate_cross_checking_segmentation_basic(int n_node_max_per_segments)
{
  // The idea of that function is to generate a node structure to crosscheck the m_chi values.
  // I am here using a simple method to assign to each cors-checking segment an ID and nodes and they will just be used to compare their state as well as their
  // m_chi to adapt the model while running using a MonteVideo scheme. Nah I am just kidding it is a Monte-Carlo scheme. Lolz

  // First I am creating an xtensor with the right size
  std::array<size_t,1> shave = {node_ID.size()};
  xt::xtensor<int,1> this_cross_checking_ID(shave);

  // Alright, let's just initialise some temp variable here:
  bool still_same_channel = true; // just checking here in which channel I am
  int incrementing_ID = 0; // icrementing ID. Wait I will rename that incrementing_ID it will make more sense (to future boris, you named that temp_ID first and that could be confusing yo).
  int last_Sk = source_key[0]; // the SK to test at each iteration
  int n_node_in_segment = 0; // keep track of the nodes in the segments. As we don't want more than a certain aount of node in each segments. yo.
  // Alright here we go, I am beginning the iteration and for each of the loop turn I will:
  for(size_t i = 0; i< shave[0]; i++)
  {
    // first getting the source key
    int this_source_key = source_key[i];
    // checking if it has changed
    if(this_source_key != last_Sk)
      // Well we are not in the same channel anymore aren't we
      still_same_channel = false;
    else
      // we are still in the same channel aren't we
      still_same_channel = true;

    // Alright if we are NOT in the same channel
    if(still_same_channel == false)
    {
      // increment a new segment and save the last one
      // -> saving last segment information
      map_of_size_for_each_cross_checking_ID[incrementing_ID] = n_node_in_segment;
      // reinitialise the n_node_segment stuff
      n_node_in_segment = 1;
      // Icrementing the segment ID
      incrementing_ID ++;
      // saving the segment ID
      this_cross_checking_ID[i] = incrementing_ID;
    }
    else
    {
      // Alright we are still in the same channel, let's first check if we actually need to increment the segment size
      if(n_node_in_segment > n_node_max_per_segments)
      {
        // we NEED a new segment yo
        // But first let me get a seflie. No. hate that song and it is stuck in my head now. Well I don't HATE it. I jsut don't care and don't particularly enjoy it
        // So first let's save the last segment
        map_of_size_for_each_cross_checking_ID[incrementing_ID] = n_node_in_segment;
        // Now incrementing the idea
        incrementing_ID++;
        // reinitialise the number of node per segments
        n_node_in_segment = 1;
        // Finally let's implement the new segment
        this_cross_checking_ID[i] = incrementing_ID;
      }
      else
      {
        // last pssible case:
        // THe most boring one actually... just incrementing the n_node and feeding the segment ID thing 
        this_cross_checking_ID[i] = incrementing_ID;
        n_node_in_segment ++;

      }
      // DOne
    }

    // let's just not forget about saving last source key
    last_Sk = this_source_key;
    // ready for the last iteration
  }

  // Last thing to do is to adapt the node indexery to the new state of stuff. I cannot speak english.
  cross_checking_ID = this_cross_checking_ID;
  // Saving the total number of segments I have in order to be able to loop through it.
  // I need to add one to the previous number because it starts as 0. As any incrementation should to be fair. Array indexing strating at 1 is harmful for society. F*ck MATLAB.
  n_segment_for_cross_checking = incrementing_ID+1;

  // Generating the original stats
  std::map<std::string, xt::pytensor<float, 1> > res_from_stat;

  // Actually calculating the results
  res_from_stat = this->calculate_segmented_basic_metrics(m_chi);

  // Only saving the median bits for the moment
  original_segment_median_checker = res_from_stat["median"];

    // Actually calculating the results
  res_from_stat = this->calculate_segmented_basic_metrics(chi);

  // Only saving the median bits for the moment
  segment_median_chi = res_from_stat["median"];

    // Actually calculating the results
  res_from_stat = this->calculate_segmented_basic_metrics(elevation);

  // Only saving the median bits for the moment
  segment_median_elev = res_from_stat["median"];

  // Actually calculating the results
  res_from_stat = this->calculate_segmented_basic_metrics(flow_distance);

  // Only saving the median bits for the moment
  segment_flow_dist = res_from_stat["median"];
  

  // for(size_t i =0; i< original_segment_median_checker.size(); i++)
  //   std:: cout << original_segment_median_checker[i] << std:: endl;
  // exit(EXIT_FAILURE);


  // Initialising the median tracker or whatever I am supposed to call it. I mean, who excatly gives name of stuff right?
  for(size_t i = 0; i< n_segment_for_cross_checking; i++)
  {
    // just making sure that my tracker is initialised to empty vectors as it will need to ingest significant amount of other data while the model is running
    std::vector<float> tempvectal;
    keep_track_of_the_last_medians.push_back(tempvectal);
    // this one simply needs to begin with 0;
    keep_track_of_how_many_time_it_has_been_processed_for_SS.push_back(0);
    // As easy as installing python right ?
    // ok now a bit more tricky
    std::vector<std::pair<float,float> > this_vecblade;
    best_fits_gatherer_for_the_segments.push_back(this_vecblade);
    // actually not that bad
  }

}


///@brief Compute a bunch of stats on a given global array based on the segmented indexing described before
///@param py or x tensor of values to compute that MUST be in the same indexing than the global arrays
///@return a map of statistical metrics: keys are "median", "first_quartile" and "third_quartile" and indices in the array are the segment ID yo
///@authors B.G
///@date 19/02/2019
std::map<std::string, xt::pytensor<float, 1> > LSDTribBas::calculate_segmented_basic_metrics(xt::pytensor<float, 1>& this_array_to_compute)
{
  // Let's begin the function by checking if the segmentation has been initialised. If not throw error.
  if(n_segment_for_cross_checking<1)
    {std::cout << "FATALERROR::calculate_segmented_basic_metrics is trying to run but the identification of segments has not been initialised yet." << std::endl;exit(EXIT_FAILURE);}
  // Now let's create the different arrays we need
  std::array<size_t,1> shlab = {n_segment_for_cross_checking};
  xt::xtensor<float, 1> tmedian(shlab), tthird_quartile(shlab),tfirst_quartile(shlab), tmin(shlab), tmax(shlab), tstddev(shlab);

  // We do have all our array
  // let's compute the data then right?
  // need few variables before doing that:
  std::vector<float> temp_values;
  int last_segID = cross_checking_ID[0]; // initialise the cross checking validation at te first index whic should be 0 I guess but let's keep that like that jsut to make sur ok?


  for (size_t i=0; i<node_ID.size();i++)
  {
    // Gathering current check ID
    int this_segID = cross_checking_ID[i];
    // Checking if it changed
    if(this_segID == last_segID && i < node_ID.size() - 1)
    {
      // it hasn't change, let's append data
      temp_values.push_back(this_array_to_compute[i]);
      // AAlright that's it for me 
    }
    else
    {
      // First of all, Just checking if this is the last element of the array. if it is I need to take account of it before saving the stats.
      if(i == node_ID.size()-1)
        temp_values.push_back(this_array_to_compute[i]);

      // REALLY IMPORTANT THERE
      // I AM SKIPPING THE FIRST NODE OF THE SEGMENT BECAUSE FUCK IT
      // JOKE APART IT SI MOST OF THE TIME AN OUTLIER AS IT DESCRIBE AN ARTIFICIAL
      // JUMP FROM A SEGMENT TO ANOTHER!!!!!!
      // WHY AM I SCREAMIIIIIIING
      // Actually aI am not screaming but reading capital letters make me think I am sceaming
      // anyway
      // std::cout <<"reach here?" << std:: endl;
      temp_values.erase(temp_values.begin());
      // std::cout <<"and here?" << std:: endl;



      if(temp_values.size()>0)
      {
        // More interesting, we changed segment, first I need to compute what has been done so far
        float this_mean, this_med,this_3qt,this_1qt, this_min, this_max, this_stddev;
        // Getting hte previously initiated metrics
        this_med = get_median(temp_values);
        this_mean = get_mean(temp_values);
        this_stddev = get_standard_deviation(temp_values, this_mean);
        this_3qt = get_percentile(temp_values,25);
        this_1qt = get_percentile(temp_values,75);
        this_min = Get_Minimum(temp_values, -9999);
        this_max = Get_Maximum(temp_values, -9999);
        // I got those metrics, let's feed the global dataset with these
        tmedian[last_segID] = this_med;
        tthird_quartile[last_segID] = this_3qt;
        tfirst_quartile[last_segID] = this_1qt;
        tmin [last_segID] = this_min;
        tmax [last_segID] = this_max;
        tstddev[last_segID] = this_stddev;

        // Feeded!
        // Reinitialisation of the gathering vector and initialisation of the next round
        // clearing the vector values
        temp_values.clear();
      }
      else
      {
        tmedian[last_segID] = 0;
        tthird_quartile[last_segID] = 0;
        tfirst_quartile[last_segID] = 0;
        tmin [last_segID] = 0;
        tmax [last_segID] = 0;
        tstddev[last_segID] = 0;
      }
      // Initialising with the first next value. No worries if this is the last element, this won't impact much yo.
      temp_values.push_back(this_array_to_compute[i]);

    }
    // std::cout <<"finally here?" << std:: endl;


    // getiing ready for the next loop tuna
    last_segID = this_segID;
  }
  // std::cout <<"maybe here?" << std:: endl;


  // DOne with the computation, let's format the output:
  std::map<std::string, xt::pytensor<float,1> > output;
  output["median"] = tmedian;
  output["first_quartile"] = tfirst_quartile;
  output["third_quartile"] = tthird_quartile;
  output["minimum"] = tmin;
  output["maximum"] = tmax;
  output["standard_deviation"] = tstddev;
  // Done let's just send that back
  return output;
}

///@brief This important function (also every function is important, do not accuse me of functionnism!!) check the global gathering of segmented median
///@brief median past data and check which one are in steady state for a sufficient time to be validated
///@param tolarance_interval: tolarance used to determin steady_state for a segment. All the m_chi values would nee to be within the same tolerance to be consdered OK
///@param min_time_checked (int): the minimum of time a segment has to be processed before actually check for steady-state. Hopefully less time than the number of time Guillaume has been checked for STDs lol
///@returns a vector of boolean telling which segments need to be adapted. The actual adaptation and data extraction is in another function.
///@authors B.G. 19/02/2019
std::vector<bool> LSDTribBas::cross_checker_of_steady_state(int min_time_checked, float tolerance_interval)
{
  // generate output
  std::vector<bool> output;
  // First thing to do is to loop through my segment and check which one CAN be EVENTUALLY assessed
  for(size_t i =0; i < n_segment_for_cross_checking ; i++)
  {
    // I can assess this segment ony if it has been processed enough time since the last check
    if(keep_track_of_how_many_time_it_has_been_processed_for_SS[i] >= min_time_checked)
    {
      // Yaaaay I can be tested (probably not what GUillaume said though honhon)
      vector<float> this_vec_of_median = keep_track_of_the_last_medians[i];
      // IMPORTANT: I am using the percentile here to avoid extreme values as the end node of the segment will always be an outlier
      float this_min = get_percentile(this_vec_of_median, 10);
      float this_max = get_percentile(this_vec_of_median, 90);
      // Checking my range of medians
      if(this_max - this_min < tolerance_interval)
      {
        // Therefore it is actually a steady segment and I can save it.
        output.push_back(true);
        change_counter++;

      }
      else
      {
        // Not ready to be extracted/change
        output.push_back(false);
      }

    }
    else
    {
      // Not ready as well to be extracted
      output.push_back(false);
    }
  }
  // keep_track_of_the_last_medians
  // keep_track_of_how_many_time_it_has_been_processed_for_SS
  return output;
}

///@brief This function ingest a previously calculated checker and save/resample the segments that needs to be
///@param vector of bool with the same dimension as the number of cross-checking segments
///@param float m_chi_tolerance_for_match: determine if the mchi is close enough to the original one
///@returns Nothing, but adapt the internal uplift and erodibility fields
///@authors B.G.
///@date 19/02/2019
void LSDTribBas::resample_and_save_matches_on_segments_that_need_to_be(vector<bool> checker, float m_chi_tolerance_for_match)
{
  // Alright the checker has been determined by the cross_checker_of_steady_state function for example
  // It detemermines which segment need to be resampled and therefore save
  // So let's go through all the node and check with their ID if the checker needs to save them
  bool need_changes = false;
  for (size_t i=0; i<node_ID.size();i++)
  {
    // std::cout << "10.1" << std::endl;
    int this_segID = cross_checking_ID[i];
    // std::cout << "10.2" << std::endl;
    // check if that segment needs to be adapted
    if(checker[this_segID])
    {
      need_changes = true;
    }

    // Checking if it does need changes
    if(need_changes)
    {
      // std::cout << "10.3" << std::endl;

      // Changes need to be made apparently... 
      if( abs(original_segment_median_checker[this_segID] - get_median(keep_track_of_the_last_medians[this_segID])) < m_chi_tolerance_for_match )
      {    
        // first, let's save the match if we can :)
        match_counter++;
        std::pair<float,float> this_match = std::make_pair(uplift_to_adapt[i],erodibility_to_adapt[i]);
        // std::cout << "Match on seg # " << this_segID << std::endl;
        best_fits_gatherer_for_the_segments[this_segID].push_back(this_match);
      }
      // std::cout << "10.4" << std::endl;
      // std::cout << "Changing seg # " << this_segID << " because " << original_segment_median_checker[this_segID] << " and " << get_median(keep_track_of_the_last_medians[this_segID]) << " and " << keep_track_of_the_last_medians[this_segID].size() << std::endl;


      // REALLY IMPORTANT YOU FORGOT THAT TWAT!
      // Reinitialisation of that segment data gathering
      // -> Forget about your medians
      keep_track_of_the_last_medians[this_segID].clear();
      // std::cout << "10.5" << std::endl;

      // -> Forget How many times you got checked guillaume! Brand new start
      // -> I should call that vector Guillaume but (i) it would be extremely confusing, (ii) Nobody won't actually see the joke sooooo.
      // -> Also it's not very nice but funny
      // -> anyway
      keep_track_of_how_many_time_it_has_been_processed_for_SS[this_segID] = 0;
      // std::cout << "10.6" << std::endl;


      // Let's now randomly pick new values of Uplift and Erodibility:
      // Credits to : Saksham answer on https://stackoverflow.com/questions/18803254/how-to-pick-a-random-element-from-a-vector-filled-with-txt-file-names-c
      int randomIndexUplift = rand() % ranges_of_uplift.size();
      // std::cout << randomIndexUplift << std::endl;

      int randomIndexErodibility = rand() % ranges_of_erodibility.size();
      // std::cout << randomIndexErodibility << std::endl;

      float new_uplift = ranges_of_uplift[randomIndexUplift];
      float new_erodibility = ranges_of_erodibility[randomIndexErodibility];
      // std::cout << "10.7" << std::endl;
      // std::cout << "new_U = " <<new_uplift << std::endl;
      // std::cout << "new_K = " <<new_erodibility << std::endl;


      // Yaay I have it, no let's implement my new values
      // Implemented for the first indice would  be the act of a fool, I need to go through the rest of the segment now
      while(cross_checking_ID[i] == this_segID && i < node_ID.size() )
      {
        uplift_to_adapt[i] = new_uplift;
        erodibility_to_adapt[i] = new_erodibility;
        i++;
      }

      // Embarrassing hacky way here
      if(i < node_ID.size())
        i = i - 1;
      // What's happening is that if I am not at the end of my global node indices
      // the for loop will increment i
      // But i is already at the beginning of the next segment
      // Therefore I am decrementing it until the for loop actually gets back on track
      // It would have been clearear and faster to hardcode than going through these explanations
      // I'll keep it though as performances won't be impacted and anyway fuck it yo.

    }
    // Might work better with that innit cunt!
    need_changes = false;

  }
  // Uplift and erofibility adapted, we are ready to roll now 
}

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
std::map<std::string, xt::pytensor<float,1> > LSDTribBas::segmented_bear_grylls(int timestep, int save_step, xt::pytensor<float,1>& base_uplift_field, 
xt::pytensor<float,1> base_erodibility_field, int n_initial_run, int min_run_before_segment_assessment, 
int max_number_of_run, float SS_tolerance, float match_tolerance)
{

  // WHile I think about it, let's implement the initial uplift/erodibility pattern
  // std::cout << "1" << std::endl;
  uplift_to_adapt = base_uplift_field;
  erodibility_to_adapt = base_erodibility_field;
  // Creating the final output to fill
  std::map<std::string, xt::pytensor<float,1> > output;

  //global size of my arrays (elevation is an attribute)
  size_t siz = elevation.size();

  // Setting arrays to store the new and last elevation
  std::array<size_t, 1> shape = { siz };
  xt::xtensor<float,1> last_elev(shape), this_new_elev(shape);
  // Setting base variables I need
  float baselevel = elevation[0]; // elevation of the very first element. So far has to be fixed
  float last_h = baselevel, this_h=0; // navigating through the different elevtion layers
  float tol = 1e-3;// Tolerance for the newton-rhapson itarative solver for the stream power law
  int this_node,last_node;// Navigating through the node and its receivers
  last_elev = elevation; // initializing our previous elevation to the first one

  // Alright let's run the model
  // for(int tTt = 0; tTt < 1; tTt++) // Ingore that it's when I need to only run one model to check some stuff
  // Real run here
  int tTt = 0;
  bool not_steady_state = true;
  float ratio_tol = 0;
  // std::cout << "2" << std::endl;
  // The model will run until the max numer of run is achieved
  while(tTt < max_number_of_run)
  {

    // just a simple counter, Well No offense though. I mean "Just a simple counter" already is something.
    tTt++; 
    // std::cout << tTt << std::endl;
    if(tTt % 100 == 0)
      std::cout << std::flush << "Processing timestep # " << tTt << " || adjustments: " << change_counter << " || matches: " << match_counter << "\r"; 

    // These maps are storing the elevation and flow distance for each node IDs. 
    // this idea is to rn the model bottom up, store informations for current nodes and 
    // therefore get the receiver elevation/flow distance of previously processed node.
    std::map<int,float> rec_to_elev, rec_to_fd;
    // How far are we in the model. Let me stop my work here, there are pretty intense turbulences in my flights.
    // OK seems better now, f*ck these were impressive
    // So I was saying that we want to know which timestep we are at.
    // std::cout << "Processing time step: " << tTt+1 << " and tolerance ratio is " << ratio_tol << "\r"; // the last sign means "Return to the beginning of my line and flush"
    // Running through the river network:
    for(size_t i=0; i<siz;i++)
    {
      // We are in the first element, it is a bit different so we need a special case:
      // Saving elevation and stuff as baselevel, and giving the first receiver information
      if(i==0)
        {this_new_elev[i]=baselevel;last_h = baselevel; rec_to_elev[node_ID[i]] = baselevel;rec_to_fd[node_ID[i]] = flow_distance[i];continue;}
      // Done, the continue statement at the end of last if makes sure it stop this loop run here if i == 0

      // I am getting my receiver and current nodes (available in the prebuilt model)
      last_node = receivers[i];
      // Old COUT statement below there. I was inspired apparently
      // std::cout << "VAGUL" << last_node << std::endl;
      this_node = node_ID[i];
      // Getting the length between these nodes (dx)
      float length = flow_distance[i] - rec_to_fd[last_node];
      // getting the flow distance ready when next nodes will query this one as receiver
      rec_to_fd[this_node] = flow_distance[i];
      // Getting the drainage area (from prebuilt model)
      float tDA = drainage_area[i];
      // Getting the erodibility from input
      float keq = erodibility_to_adapt[i];
      // std::cout << "3" << std::endl;

      float hnew = newton_rahpson_solver(keq, meq, neq,  tDA,  timestep,  length, last_elev[i], rec_to_elev[last_node] ,tol);
      // std::cout << "4" << std::endl;
       

      // Applying the new elevation field
      this_new_elev[i] = hnew;
      rec_to_elev[this_node] = hnew;
    }
    // Done with the elevation stuff

    // computing uplift
    for(size_t i=1; i<siz;i++)
    {
      this_new_elev[i] = this_new_elev[i] + uplift_to_adapt[i]*timestep;
    }
    // std::cout << "5" << std::endl;

    // Saving this step if required
    // and adding to the global map if required
    if(tTt != 0)
    {
      if((tTt%save_step) == 0)
      {
        std::string gh = itoa(tTt * timestep); // Calculate the real time
        output[gh] = this_new_elev;
      }
    }

    // Alright y model ran and that's fun and all
    // Let's now just get to the shit
    // First I need to check if the model ran long enough to have actually started. You don't want to compare the outputs to yourself do you?
    if(tTt > n_initial_run)
    {
      // Right here we are
      // first I o need the new m_chi to compare with the rest of the data
      // because the elevation is now modelled, it is much much much cleaner and can be calculated more directly
      xt::pytensor<float,1> ttNZ = this_new_elev;
      // std::cout << "6" << std::endl;
      xt::pytensor<float,1> current_m_chi = this->first_order_m_chi_from_custarray(chi, ttNZ);
      // std::cout << "7" << std::endl;

      // Got my new m_chi
      // Now I need to get tthe segmented stats:
      std::map<std::string, xt::pytensor<float, 1> > segstatmap = this->calculate_segmented_basic_metrics(current_m_chi);
      // std::cout << "8" << std::endl;
      // Got it so let me feed the model with the segmented median
      for(size_t g=0;g<segstatmap["median"].size();g++)
      {
        // Here I am implementing the median 
        keep_track_of_the_last_medians[g].push_back(segstatmap["median"][g]);
        // and here the count 
        keep_track_of_how_many_time_it_has_been_processed_for_SS[g]++;
        // to the globa variables
      }
      // std::cout << "9" << std::endl;
      std::vector<bool> checker = this->cross_checker_of_steady_state(min_run_before_segment_assessment, SS_tolerance);
      // std::cout << "10" << std::endl;

      this->resample_and_save_matches_on_segments_that_need_to_be(checker, match_tolerance);
      // std::cout << "11" << std::endl;
// 

    }



    // IMPoORTANT: giving the 
    // next elevation the new one
    last_elev = this_new_elev;
  }


  return output;
}

// To work on: getting an acceptable output format

std::map<std::string,xt::pytensor<float,1> > LSDTribBas::get_bear_results()
{
  // The aim of that function is to get the results of the best fits from the segmented bear grylls segmentation
  // I'll generate first some stats about it
  // The other solution would be to output all the results but it might be annoying for many reasons
  std::map<std::string,xt::pytensor<float,1> > output;

  // temp map that format the result in term of 
  std::map<int,std::map<std::string, float> > stat_for_each_csegment;

  // I am first feeding these temp maps
  int counter_NODMAR = 0;
  for(size_t i=0;i<n_segment_for_cross_checking; i++)
  {

    std::vector<std::pair<float,float> > temp_vec = best_fits_gatherer_for_the_segments[i];
    std::vector<float> these_U, these_K;
    for(size_t j=0; j<temp_vec.size();j++)
    {
      these_U.push_back(temp_vec[j].first);
      these_K.push_back(temp_vec[j].second);
    }
    bool NODOTAFOUND = false;

    // just checking if my U/K is OK    
    if(these_K.size()<1)
    {
      these_U.push_back(-9999);
      these_K.push_back(-9999);
      NODOTAFOUND = true;
      counter_NODMAR++;
      
    }



    // Alright I got the best fits for that part
    // let's get the stats out of that

    stat_for_each_csegment[i]["uplift_median"] = get_median(these_U);
    stat_for_each_csegment[i]["uplift_mean"] = get_mean(these_U);
    stat_for_each_csegment[i]["uplift_stddev"] = get_standard_deviation(these_U, stat_for_each_csegment[i]["uplift_mean"]);
    stat_for_each_csegment[i]["uplift_first_quartile"] = get_percentile(these_U,25);
    stat_for_each_csegment[i]["uplift_third_quartile"] = get_percentile(these_U,75);
    stat_for_each_csegment[i]["uplift_min"] = Get_Minimum(these_U, -9999);
    stat_for_each_csegment[i]["uplift_max"] = Get_Maximum(these_U, -9999);

    stat_for_each_csegment[i]["erodibility_median"] = get_median(these_K);
    stat_for_each_csegment[i]["erodibility_mean"] = get_mean(these_K);
    stat_for_each_csegment[i]["erodibility_stddev"] = get_standard_deviation(these_K, stat_for_each_csegment[i]["erodibility_mean"]);
    stat_for_each_csegment[i]["erodibility_first_quartile"] = get_percentile(these_K,25);
    stat_for_each_csegment[i]["erodibility_third_quartile"] = get_percentile(these_K,75);
    stat_for_each_csegment[i]["erodibility_min"] = Get_Minimum(these_K, -9999);
    stat_for_each_csegment[i]["erodibility_max"] = Get_Maximum(these_K, -9999);
    if(NODOTAFOUND)
      stat_for_each_csegment[i]["n_element"] = 0;
    else
      stat_for_each_csegment[i]["n_element"] = float(these_U.size());

    // DOne for each segments

  }

  std::cout << "RESULTS: " << counter_NODMAR << " SEGMENTS WITH NO MATCH out of " << n_segment_for_cross_checking << std::endl;

  // Should be good with generating the stats. let's now format the output
  std::array<size_t, 1> dapf = {node_ID.size()};
  xt::xtensor<float,1> crossID(dapf),uplift_median(dapf),uplift_mean(dapf),uplift_stddev(dapf),uplift_first_quartile(dapf),uplift_third_quartile(dapf),uplift_min(dapf),uplift_max(dapf),erodibility_median(dapf),erodibility_mean(dapf),erodibility_stddev(dapf),erodibility_first_quartile(dapf),erodibility_third_quartile(dapf),erodibility_min(dapf),erodibility_max(dapf), n_element(dapf);

  for(size_t i=0;i<dapf[0];i++)
  {
    int this_segID = cross_checking_ID[i];

    uplift_median[i] = stat_for_each_csegment[this_segID]["uplift_median"];
    uplift_mean[i] = stat_for_each_csegment[this_segID]["uplift_mean"];
    uplift_stddev[i] = stat_for_each_csegment[this_segID]["uplift_stddev"];
    uplift_first_quartile[i] = stat_for_each_csegment[this_segID]["uplift_first_quartile"];
    uplift_third_quartile[i] = stat_for_each_csegment[this_segID]["uplift_third_quartile"];
    uplift_min[i] = stat_for_each_csegment[this_segID]["uplift_min"];
    uplift_max[i] = stat_for_each_csegment[this_segID]["uplift_max"];
    erodibility_median[i] = stat_for_each_csegment[this_segID]["erodibility_median"];
    erodibility_mean[i] = stat_for_each_csegment[this_segID]["erodibility_mean"];
    erodibility_stddev[i] = stat_for_each_csegment[this_segID]["erodibility_stddev"];
    erodibility_first_quartile[i] = stat_for_each_csegment[this_segID]["erodibility_first_quartile"];
    erodibility_third_quartile[i] = stat_for_each_csegment[this_segID]["erodibility_third_quartile"];
    erodibility_min[i] = stat_for_each_csegment[this_segID]["erodibility_min"];
    erodibility_max[i] = stat_for_each_csegment[this_segID]["erodibility_max"];
    n_element[i] = stat_for_each_csegment[this_segID]["n_element"];
    crossID[i] = this_segID;
  }

  // should have gathered all the data I need, let's retrun it now

  output["uplift_median"] = uplift_median;
  output["uplift_mean"] = uplift_mean;
  output["uplift_stddev"] = uplift_stddev;
  output["uplift_first_quartile"] = uplift_first_quartile;
  output["uplift_third_quartile"] = uplift_third_quartile;
  output["uplift_min"] = uplift_min;
  output["uplift_max"] = uplift_max;
  output["erodibility_median"] = erodibility_median;
  output["erodibility_mean"] = erodibility_mean;
  output["erodibility_stddev"] = erodibility_stddev;
  output["erodibility_first_quartile"] = erodibility_first_quartile;
  output["erodibility_third_quartile"] = erodibility_third_quartile;
  output["erodibility_min"] = erodibility_min;
  output["erodibility_max"] = erodibility_max;
  output["n_element"] = n_element;
  output["segment_ID"] = crossID;


  // for(size_t fabrice = 0 ; fabrice < uplift_median.size(); fabrice ++)
  //   std::cout << uplift_median[fabrice] << std::endl;


  return output;

}

// std::map<std::string,xt::pytensor<float,1> > LSDTribBas::get_full_bear()
// {
//   // The aim of that function is to get the results of the best fits from the segmented bear grylls segmentation
//   // I'll generate first some stats about it
//   // The other solution would be to output all the results but it might be annoying for many reasons
//   std::map<std::string,xt::pytensor<float,1> > output;

//   // temp map that format the result in term of 
//   std::map<int,std::map<std::string, float> > stat_for_each_csegment;

//   // I am first feeding these temp maps
//   int counter_NODMAR = 0;
//   for(size_t i=0;i<n_segment_for_cross_checking; i++)
//   {

//     std::vector<std::pair<float,float> > temp_vec = best_fits_gatherer_for_the_segments[i];
//     std::vector<float> these_U, these_K;
//     for(size_t j=0; j<temp_vec.size();j++)
//     {
//       these_U.push_back(temp_vec[j].first);
//       these_K.push_back(temp_vec[j].second);
//     }
//     bool NODOTAFOUND = false;

//     // just checking if my U/K is OK    
//     if(these_K.size()<1)
//     {
//       these_U.push_back(-9999);
//       these_K.push_back(-9999);
//       NODOTAFOUND = true;
//       counter_NODMAR++;
      
//     }
//   }
// }


// The following function respond to the need of a distribution visualisation for LSDTribBas!
// It returns a map of bins for each segments. Values are in log space for Erodibility and normal for the Uplift
std::map<std::string, std::map<int,xt::pytensor<float,1> > > LSDTribBas::get_distribeartion(int n_bins)
{
  // The aim of that function is to get a histogram distribution for each segments
  // Intermediate alternative to outputting everything!

  // Fromaging the output
  std::map<std::string, std::map<int,xt::pytensor<float,1> > > output;

  // Determining the bindaries (boundary of each bins. That's a pun hon hon hon. bin boundaries -> bindaries. Get it?)
  // Boundaries are all superior boundaries. 
  std::vector<float> upper_bounds_uplift(n_bins), upper_bounds_erodibility(n_bins);

  float min_U = Get_Minimum(ranges_of_uplift, -9999);
  float max_U = Get_Maximum(ranges_of_uplift, -9999);
  std::vector<float> logofer;
  // Just need to log10 the thing
  for(size_t it=0;it<ranges_of_erodibility.size();it++)
    logofer.push_back(log10(ranges_of_erodibility[it]));
  // Bounds of stuff yo
  float min_er = Get_Minimum(logofer, -9999);
  float max_er = Get_Maximum(logofer, -9999);
  // steps
  float step_U = (max_U - min_U)/n_bins;
  float step_er = (max_er - min_er)/n_bins;

  // gettin' the bounds
  for(size_t it=1;it<=size_t(n_bins);it++)
  {
    size_t gig = it-1;
    upper_bounds_uplift[gig] = min_U + it*step_U;
    upper_bounds_erodibility[gig] = min_er + it*step_er;
  }

  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------
  // Yaaay lunch time------------------------------------------------

  // I am first feeding these temp maps
  int counter_NODMAR = 0;
  std::array<size_t, 1> shjazcy = { size_t(n_bins) };

  for(size_t i=0;i<n_segment_for_cross_checking; i++)
  {

      // Creating that specific distribution
    // std::array<size_t, 1> shjazcy = { size_t(n_bins) };
    xt::xtensor<float, 1> this_distribU(shjazcy), this_distribK(shjazcy);

    for(size_t j=0; j<shjazcy[0]; j++)
    {
      this_distribK[j]=0;
      this_distribU[j]=0;
    }

    std::vector<std::pair<float,float> > temp_vec = best_fits_gatherer_for_the_segments[i];
    std::vector<float> these_U, these_K;
    for(size_t j=0; j<temp_vec.size();j++)
    {
      these_U.push_back(temp_vec[j].first);
      these_K.push_back(log10(temp_vec[j].second));
    }

    for (size_t iU=0; iU<these_U.size();iU++)
    {
  
      bool foundit_U_edition = false;
      float this_U_to_test = these_U[iU];
      // float last_U = 0;
      for (size_t TBTT = 0; foundit_U_edition == false; TBTT++)
      {
        if(this_U_to_test <= upper_bounds_uplift[TBTT] )
        {
          this_distribU[TBTT] = this_distribU[TBTT] + 1;
          foundit_U_edition = true;
        }
      }
      // Same shit with K
      bool foundit_K_edition = false;
      float this_K_to_test = these_K[iU];

      for (size_t TBTT = 0; foundit_K_edition == false; TBTT++)
      {
        if(this_K_to_test <= upper_bounds_erodibility[TBTT] )
        {
          this_distribK[TBTT] = this_distribK[TBTT] + 1;
          foundit_K_edition = true;
        }
    
      }
    
    }


    output["uplift"][i] = this_distribU;
    output["erodibility"][i] = this_distribK;
  }
  xt::xtensor<float, 1> Upper_bound_U(shjazcy), Upper_bound_K(shjazcy);
  Upper_bound_U = xt::adapt(upper_bounds_uplift);
  Upper_bound_K = xt::adapt(upper_bounds_erodibility);
  output["uplift_upper_X"][0] = Upper_bound_U;
  output["erodibility_upper_X"][0] = Upper_bound_K;
  output["segment_median_chi"][0] = segment_median_chi;
  output["segment_median_elev"][0] = segment_median_elev;
  output["segment_flow_dist"][0] = segment_flow_dist;

  // Should be done
  return output;

}



void LSDTribBas::set_range_of_uplift_and_K_for_Bear_gryll(std::vector<float>& these_new_K, std::vector<float>& these_new_U)
{
  ranges_of_uplift = these_new_U;
  ranges_of_erodibility = these_new_K;
}

























////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////// Bindings for Muddpile thereafter //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// #ifdef __linux__

// void muddpyle::create()
// {
//   // Empty creator
//   std::cout << "I am an empty creator. You may need me for something but it might trigger a bug. Who knows? Anyway, have a good day." << std::endl;

// }

// void muddpyle::create(bool tquiet)
// {
//   if(tquiet == false)
//   {
//     std::cout << "Welcome to MuddPyle, python binding for Muddpile. You can find documentation here:" << std::endl << "https://lsdtopotools.github.io/LSDTT_documentation/LSDTT_MuddPILE.html" << std::endl;
//   }
//   // Initialiseing Muddpile
//   mymod = LSDRasterModel();
//   // set level of quietness
//   mymod.set_quiet(tquiet);
//   verbose = tquiet;

//   // Parameters specific to the python binding
//   mymod.set_print_hillshade(false);
//   // Done
// }

// // initialise the model with an already existing topography
// void muddpyle::create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, float m, 
//   float n, int dt, int Umod, float save_step, float default_D, float default_Sc, xt::pytensor<float,2>& K_field, xt::pytensor<float,2>& U_field, bool hillslope_diffusion,
//   bool use_adaptive_timestep, float max_dt, std::string OUT_DIR, std::string OUT_ID, std::vector<std::string> bc )
// {
//   TNT::Array2D<float> data_pointerast(nrows,ncols,ndv), tKdat(nrows,ncols,ndv), tUdat(nrows,ncols,ndv); // Trying here to create a raster of pointers

//   for(size_t i=0;i<nrows;i++)
//   for(size_t j=0;j<ncols;j++)
//     {data_pointerast[i][j] = data(i,j);tKdat[i][j] = K_field(i,j);tUdat[i][j] = U_field(i,j);}

//   // Data is copied! I am ready to feed a LSDRaster object
//   // Here we go: 
//   LSDRaster temprast(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
//   LSDRaster this_KRaster(nrows, ncols, xmin, ymin, cellsize, ndv, tKdat);
//   LSDRaster this_URaster(nrows, ncols, xmin, ymin, cellsize, ndv, tUdat);

//   LSDRasterModel mod(temprast);
//   mod.add_path_to_names(OUT_DIR);
//   mod.set_name(OUT_DIR+OUT_ID);

//   // m and n are the exponent of spls
//   mod.set_m(m);
//   mod.set_n(n);
//   // The mode of uplift. Options are:
//   // default == block uplift
//   // 1 == tilt block
//   // 2 == gaussian
//   // 3 == quadratic
//   // 4 == periodic
//   mod.set_uplift_mode(Umod);
//   // time step in years and max time
//   mod.set_timeStep(dt);
//   // mod.set_maxtimeStep(max_dt);
//   mod.set_endTime(max_dt);
//   mod.set_endTime_mode(1);
//   // save step in number of timestep (not in year!)
//   mod.set_float_print_interval(save_step);
//   // Default diffusion and critical slope param for hillslope diffusion
//   mod.set_D( default_D);
//   mod.set_S_c(default_Sc);
//   // Default Erodibility param for river system
//   mod.set_K(this_KRaster.get_data_element(0,0));

//   // Remove edge artefacts
//   mod.initialise_taper_edges_and_raise_raster(1);

//   // ACtivate (or not the hillslope diffusion)
//   mod.set_hillslope(hillslope_diffusion);
//   // need to tell the model the first time to print (I guess???????)
//   mod.set_next_printing_time(0);
//   mod.set_current_frame(0);
//   mod.set_print_hillshade(false);
//   mod.set_boundary_conditions(bc);

//   mod.raise_and_fill_raster();



//   mod.run_components_combined(this_URaster,this_KRaster,use_adaptive_timestep);
//   // Done basically???

// }

// // this function initialises the model with real topography
// void muddpyle::initialise_model_with_dem(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data)
// {
//   // transferring the array
//   TNT::Array2D<float> data_pointerast(nrows,ncols,ndv);
//   for(size_t i=0;i<nrows;i++)
//   for(size_t j=0;j<ncols;j++)
//     {data_pointerast[i][j] = data(i,j);}

//   LSDRaster topo(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
//   mymod = LSDRasterModel(topo);
//   base_nrows = nrows;
//   base_ncols = ncols;
//   base_xmin = xmin;
//   base_ymin = ymin;
//   base_resolution = cellsize;
//   base_ndv = ndv;

//   if(verbose)
//     std::cout<< "I have feeded the model with real topography, I just need few internal routines to make it ready for muddpilisations." << std::endl;

//   mymod.raise_and_fill_raster();
//   mymod.set_print_hillshade(false);


// }

// // This function initialise (or at least attempt to initialise) the model with a diamond square method crafted by simon.
// // It generates nice initial topography, detail in the doc
// // Diamonds -> mineral -> main deposit type -> kimberlite. Funny innit?. I don't care I find it funny anyway and noone will read that yo.
// // B.G. 2019
// void muddpyle::initialise_model_kimberlite(int nrows, int ncols, float resolution, int diamond_square_feature_order, float diamond_square_relief, float parabola_relief, float roughness_relief, int prediff_step)
// {
  
//   if(this->verbose)
//     std::cout << "Looking for kimberlites ..." << std::endl;
//   // reshaping the model.
//   mymod.resize_and_reset(nrows,ncols,resolution);
//   base_nrows = nrows;
//   base_ncols = ncols;
//   base_xmin = 0;
//   base_ymin = 0;
//   base_resolution = resolution;
//   base_ndv = -9999;
//   // Spotting the smallest/largest (possible) dimension mate
//   int smallest_dim = 0; if(nrows>=ncols){smallest_dim = ncols;}else{smallest_dim=nrows;} 
//   float lps = floor( log2( float(smallest_dim) ) ); int largest_possible_scale = int(lps);
//   // You don't want to create larger figures than possible
//   if(diamond_square_feature_order>largest_possible_scale)
//     diamond_square_feature_order = largest_possible_scale;
//   // let's goooooooooooooooooooooooooooo
//   mymod.intialise_diamond_square_fractal_surface(diamond_square_feature_order, diamond_square_relief);
//   // Wanna add parabola relief on that?
//   if(parabola_relief>0)
//     mymod.superimpose_parabolic_surface(parabola_relief);
//   // A bit of noise never hurts right?
//   if(roughness_relief > 0)
//   {
//     mymod.set_noise(roughness_relief);
//     mymod.random_surface_noise();
//   }

//   if(prediff_step>0)
//   {
//     if(verbose)
//       cout << "I am going to diffuse the initial surface for you." << endl;

//     float tpts = mymod.get_timeStep();
//     mymod.set_timeStep( 0.5 );
//     for (int i = 0; i< prediff_step; i++)
//     {
//       mymod.MuddPILE_nl_soil_diffusion_nouplift();
//     }
//     mymod.set_timeStep(tpts);
//   }

//   mymod.initialise_taper_edges_and_raise_raster(1);

//   // NOT SURE IF I NEED THAT
//   mymod.raise_and_fill_raster();
//   // printing the outputs
//   if(verbose)
//     std::cout << "I am printing the initial surface, if nothing happens, you forgot to tell me the output loc and prefix!!" << std::endl; 
//   mymod.print_rasters_and_csv(-1);
//   if(verbose)
//     std::cout << "Awrite mate I found the kimberlite." << std::endl; 

// }

// void muddpyle::run_model(bool fluvial, bool hillslope, bool spatially_variable_UK, int save_frame, bool use_adaptative_timestep)
// {
//   mymod.set_next_printing_time(0);
//   mymod.set_current_frame(save_frame);
//   mymod.set_print_hillshade(false);
//   mymod.set_hillslope(hillslope);
//   mymod.set_fluvial(fluvial);

//   // mymod.raise_and_fill_raster();
//   if(spatially_variable_UK)
//     mymod.run_components_combined(UpliftRaster,KRaster,use_adaptative_timestep);
//   else
//     mymod.run_components_combined();
// }

// // end of the only-linux part 
// #endif




















////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////// Builing a new Class here, cleaner approach now that first tests are promising ///////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

void Carpythians::create()
{
  std::cout << "Carpythians constructed by default, hopefully you know what you are doing." << std::endl;
}

void Carpythians::create(int nrows, int ncols, float xmin, float ymin, float cellsize, float ndv, xt::pytensor<float,2>& data, xt::pytensor<short,2>& boundary_conditions, int n_threads)
{

  // Setting up the maximum number of threads:
  max_n_threads = n_threads;

  // Converting the pyarray to array2D
  TNT::Array2D<float> data_pointerast(nrows,ncols,ndv); // Trying here to create a raster of pointers

  for(size_t i=0;i<nrows;i++)
  for(size_t j=0;j<ncols;j++)
    {data_pointerast[i][j] = data(i,j);}

  // Data is copied! I am ready to feed a LSDRaster object
  // Here we go: 
  std::cout << "Initialising the model structure" << endl;
  LSDRaster temprast(nrows, ncols, xmin, ymin, cellsize, ndv, data_pointerast);
  current_topography = temprast;

  // I just need that before
  vector<string> temp_bc(4);
  BoCo = {"n","n","n","n"};
  // Alright that's all I need, let's get the attributes now
  //# FlowInfo contains node ordering and all you need about navigatinfg through donor/receiver through your raster
  current_FI = LSDFlowInfo(BoCo, current_topography);
  current_node_orders = current_FI.get_map_of_vectors();
  // Contains the following:
  // stack_order
  // inverted_stack_order
  // rows
  // cols
  // inverted_rows
  // inverted_cols
  // tracer

  FATAL_WARNINGS_NRITERATIONS = 0;
  FATAL_WARNINGS_GSITERATIONS = 0;


  initial_time = 0;
  current_time = 0;

  std::cout << "Model initialised. Carpythians (muddpile-inspired) ready to roll." << std::endl;



}


xt::pytensor<float,2> Carpythians::run_SPL(float dt, int n_dt, xt::pytensor<float,2>& uplift, xt::pytensor<float,2>& erodibility, int num_threads, float meq, float neq)
{

  // Setting up the new elevation
  LSDRaster new_elevation = current_topography;
  bool once_upon_a_time = false;

  // AAAAAALLLLLLRIGHT
  for(int dat_t = 1; dat_t <= n_dt; dat_t++)
  {
    // first I need to check if I have to reprocess me FI
    if(once_upon_a_time)
    {
      current_FI = LSDFlowInfo(BoCo, current_topography);
      current_node_orders = current_FI.get_map_of_vectors();
    }
    once_upon_a_time = true;

    #pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<current_node_orders["stack_order"].size();i++)
    {
      for(size_t j=0; j<current_node_orders["stack_order"][i].size(); j++)
      {
        int this_node = current_node_orders["stack_order"][i][j];
        int this_row = current_node_orders["rows"][i][j];
        int this_col = current_node_orders["cols"][i][j];
        int receiving_node, receiving_col, receiving_row;current_FI.retrieve_receiver_information(this_node, receiving_node,receiving_row,receiving_col);
        if(this_row != 0 && this_row != current_topography.get_NRows()-1 && this_col != 0 && this_col != current_topography.get_NCols()-1)
          current_topography.set_data_element(this_row,this_col, current_topography.get_data_element(this_row,this_col)+uplift(this_row,this_col)*dt); // Incrementing the uplift
        float nel = this->newton_rahpson_solver(erodibility(this_row,this_col), meq, neq,current_FI.get_DrainageArea_square_m(this_node), 
dt, current_topography.get_DataResolution(),current_topography.get_data_element(this_row,this_col), current_topography.get_data_element(receiving_row,receiving_col), 1e-4);
        new_elevation.set_data_element(this_row,this_col,nel); 
      }        
    }
    current_topography = new_elevation;
  }

  std::array<size_t,2> sizla = {current_topography.get_NRows(),current_topography.get_NCols()};
  xt::xtensor<float,2> output(sizla);
  for(size_t i=0;i<current_topography.get_NRows();i++)
  {
    for(size_t j=0;j<current_topography.get_NCols();j++)
    {
      output(i,j) = current_topography.get_data_element(i,j);
    }
  }

  return output;
}

float Carpythians::newton_rahpson_solver(float keq, float meq, float neq, float tDA, float dt, float dx, float this_elevation, float receiving_elevation ,float tol)
{
  // Newton-rahpson method to solve non linear equation, see Braun and Willett 2013
  float epsilon;   // in newton's method, z_n+1 = z_n - f(z_n)/f'(z_n)
                   // and here epsilon =   f(z_n)/f'(z_n)
                   // f(z_n) = -z_n + z_old - dt*K*A^m*( (z_n-z_r)/dx )^n
                   // We differentiate the above equation to get f'(z_n)
                   // the resulting equation f(z_n)/f'(z_n) is seen below
  // A bit of factorisation to clarify the equation
  float streamPowerFactor = keq * pow(tDA, meq) * dt;
  float slope; // Slope.
  float new_zeta = this_elevation; float old_zeta = new_zeta; // zeta = z = elevation
  // iterate until you converge on a solution. Uses Newton's method.
  int iter_count = 0;
  do
  {
    // Get the slope
    slope = (new_zeta - receiving_elevation) / dx;
    // Check backslope or no slope ie no erosion
    if(slope <= 0)
    {
      epsilon = 0;
    }
    else
    {
      // Applying the newton's method
      epsilon = (new_zeta - old_zeta + streamPowerFactor * std::pow(slope, neq)) /
           (1 + streamPowerFactor * (neq/dx) * std::pow(slope, neq-1));
    }
    // iterate the result
    new_zeta -= epsilon;

    // This limits the number of iterations, it is a safety check to avoid infinite loop
    // Thsi will begin to split some  nan or inf if it diverges
    iter_count++;
    if(iter_count > 100)
    {
      std::cout << "Too many iterations! epsilon is: " << std::abs(epsilon) << std::endl;
      epsilon = 0.5e-6;
    }
    // I want it to run while it can still have an effect on the elevation
  } while (abs(epsilon) > tol);

  // Avioding inversion there!!
  if(new_zeta < receiving_elevation)
    new_zeta = receiving_elevation;

  // std::cout << receiving_elevation<< "||" <<this_elevation <<"||"<< new_zeta << std::endl;

  return new_zeta;
}


// trying to implement here the Sediment-transport version of the SPL (Yuan et al.,2019)
xt::pytensor<float,2> Carpythians::test_run_SSPL(float dt, float n_dt, float Kb, float Ks, float Gb, float Gs, float U_range, float U_foreland, float meq, float neq)
{

  if(neq != 1)
  {
    std::cout << "ERROR: only support n=1 as I am still learning. SNS." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Alright let dx be clearer
  float dx = current_topography.get_DataResolution(), dxsquared = dx*dx;
  // Setting up the new elevation
  LSDRaster new_elevation = current_topography;
  bool once_upon_a_time = false;

  // AAAAAALLLLLLRIGHT
  for(int dat_t = 1; dat_t <= n_dt; dat_t++)
  {
    // first I need to check if I have to reprocess me FI
    if(once_upon_a_time)
    {
      current_FI = LSDFlowInfo(BoCo, current_topography);
      current_node_orders = current_FI.get_map_of_vectors();
      if(current_node_orders["stack_order"].size() == 0)
      {
       std::exit(EXIT_FAILURE);
      }
    }
     // I only need not to process it for the first timestep then I need to reprocess it all the time so:
    // once_upon_a_time = true;

    // I need a cumulated topography raster to feed the b vec of solutions!
    // Basically using the fastscape inverted ordering to accumulate current node elevation to receiver node without including it for oneself
    // this is basically representing the sum part of the bterm!
    float goulg = 0;
    TNT::Array2D<float> cumulated_array_of_upstream_dist(current_topography.get_NRows(), current_topography.get_NCols(), goulg);
    for(size_t i=0;i<current_node_orders["stack_order"].size();i++)
    {
      for(size_t j=0;j<current_node_orders["stack_order"][i].size();j++)
      {
        int this_node = current_node_orders["inverted_stack_order"][i][j];
        int receiving_row, receiving_col, receiving_node; current_FI.retrieve_receiver_information(this_node,receiving_node,receiving_row,receiving_col);
        // Only cumulating if this is not the baselevel
        if(this_node!=receiving_node)
        {
          cumulated_array_of_upstream_dist[receiving_row][receiving_col] += (cumulated_array_of_upstream_dist[current_node_orders["inverted_rows"][i][j]][current_node_orders["inverted_cols"][i][j]] + current_topography.get_data_element(current_node_orders["inverted_rows"][i][j],current_node_orders["inverted_cols"][i][j]));
          // std::cout << "ACCUMUCHECK: row||col||val :-> " << current_node_orders["inverted_rows"][i][j] << "||" << current_node_orders["inverted_cols"][i][j] << "||" << cumulated_array_of_upstream_dist[current_node_orders["inverted_rows"][i][j]][current_node_orders["inverted_cols"][i][j]] << endl;
        }
      }
    }

    // to get tild-Ai
    // current_FI.retrieve_ndonors_to_node
    // Iterating through my baselevels
    // I am myself quite pragmatic
    #pragma omp parallel for num_threads(4)
    for(int i=0;i<current_node_orders["stack_order"].size();i++)
    {

      // For some obscur reasons, I cannae stop flowinfo to catch empty basins!! so here is a safety check
      if (current_node_orders["stack_order"][i].size() == 0)
        continue; // jum to te next iteration


      // Building my matrix! PROBABLY DEPRECATED LETS WAIT FOR N!=1 TO GET RID OF IT
      int n_elements = int(current_node_orders["stack_order"][i].size());
      // My matrix has has many row col as unknowns: so as many as my number of nodes in that stack
      // -> this is the general initialisation of my matrix, let's be clear and call my matrix xirtam.
      // Eigen::SparseMatrix<float> xirtam(n_elements,n_elements);
      // According to https://eigen.tuxfamily.org/dox/group__TutorialSparse.html best way to create me matrix is to use Eigen triplets,
      // where each triplet is Eigen::Triplet<float>(row,col,val)
      // std::vector<Eigen::Triplet<float> > belleville;

      // Initialising the matrix of equations as well
      std::vector<float> b_t(current_node_orders["stack_order"][i].size());

      // At that time, I think the most efficient matrix row tracker would be a map
      std::map<int,int> node_ID_to_row,node_ID_to_recrow;
      int cpt_row = 0;

      //Will need that later: ordered vectors of intermediate values
      std::vector<float> dat_new_elev(current_node_orders["stack_order"][i].size()), stack_ordered_Fi(current_node_orders["stack_order"][i].size());
      
      // building the matrix!
      for(size_t j=0; j<current_node_orders["stack_order"][i].size(); j++)
      {
        // Getting my working node
        int this_node = current_node_orders["stack_order"][i][j];
        // This node will go into the matrix, The row idince correspond to my unknow and therefore place in the stackorder
        node_ID_to_row[this_node] = cpt_row;
        int this_row = current_node_orders["rows"][i][j];
        int this_col = current_node_orders["cols"][i][j];
        int receiving_node, receiving_col, receiving_row;current_FI.retrieve_receiver_information(this_node, receiving_node,receiving_row,receiving_col);
        float dl_i = current_FI.get_Euclidian_distance(this_node, receiving_node);
        int tildAi =  current_FI.retrieve_ndonors_to_node(this_node);
        float dat_zeta = current_topography.get_data_element(this_row,this_col);
        // getting the row index of the receiver:
        node_ID_to_recrow[this_node] = node_ID_to_row[receiving_node];

        //Will need that later
        dat_new_elev[j] = dat_zeta;

        // Computing the b part of the system.
        // note that I have computed the cumulated elevation earlier on!
        // note also that the (tildAi) * (U_range*dt) is a factorisation of the last part of the sum for bt on equation 16

        float second_part_of_bt_eq_to_check = 0;
        if(tildAi>0) // if there is no node upstream, there is no distance to add -> 0
         second_part_of_bt_eq_to_check = (Gb/tildAi) * (cumulated_array_of_upstream_dist[this_row][this_col] + (tildAi * U_range*dt) );
        float dat_bti = dat_zeta + U_range*dt + second_part_of_bt_eq_to_check; 

        // Getting the b matrix of solutions
        b_t[j] = dat_bti;
        // std::cout << "bt = " << dat_bti << " Atild = " << tildAi << " cumcum = " << cumulated_array_of_upstream_dist[this_row][this_col] << std::endl;


        // int matrix_row = int(j); // the row correspond to the unknown
        // then I need the to implement that actual row
        // need to get Fi from equation (16)
        float Fi = 0;
        if(j!=0)
        {
          // FI gets weird if I am my own receiver!
          if(dl_i == 0)
          {
            // std::cout << "FATAL WARNING: dl_i = 0 for some reasons!! and j is " << j << std::endl;
            // std::exit(EXIT_FAILURE);
            Fi = 0;
          }
          // Classic case otherwise
          else
            Fi = (Kb * std::pow(current_FI.get_DrainageArea_square_m(this_node), meq) * dt) / dl_i;
        }

        // SAving my Fi to recover it rapidly
        stack_ordered_Fi[j] = Fi;

        // First triplet is the actual row/col combination:
        // Eigen::Triplet<float> dat_bellville(cpt_row,cpt_row,1+Fi);
        // belleville.push_back(dat_bellville); 
        // Second triplet is the receivr one (if we are not dealing with da baselevel ofc):
        // if(j!=0)
        // {
          // int mat_col = node_ID_to_recrow[this_node];
          // dat_bellville = Eigen::Triplet<float>(cpt_row,mat_col,(-1*Fi));belleville.push_back(dat_bellville);
        // }

        // and now the donors, thanks to the stack order
        // for(int nd=1; nd <= tildAi; nd++)
          // {dat_bellville = Eigen::Triplet<float>(cpt_row,cpt_row+1,(-1*Fi));belleville.push_back(dat_bellville);}

        // incrementing the row counter
        // cpt_row++;
      }        
        
      //DEBUG
      // std::cout << "Managed to gather all the triplet, let' now build the bloody matrix" << std::endl;
      // xirtam.setFromTriplets(belleville.begin(), belleville.end());

      // My matrix being build (btw I I die before finishing that algorithm, it compiles until that point!!)
      // Alright let's now try the Gauss-Seidel-tamere iteration scheme
      // a clever way to rearrange the equation is detailed in (21) and uses stack order to solve all the equations

      float max_iter_tol = -9999; // check if the wanted tolerance is achieved or not
      // Setting a map of element to reach the previous node new elev
      std::map<int,float> current_elev_rec;
      // std::cout << "Here?? :: " << current_node_orders["inverted_stack_order"][i].size() << std::endl;
      // -> first element obviously being the baselevel yo
      current_elev_rec[current_node_orders["stack_order"][i][0]] = dat_new_elev[0];
      // Setting few checkers
      bool nedd_to_cumulate = false;
      int n_iterations =0;
      // Entering the Gauss-Seidel iterative scheme. Impressive aye?
      // So far it is pretty dumb ----> see equation (21)
      do
      {
        n_iterations++;
        // Reinitialising my tolerance for this iteration
        max_iter_tol = -9999;
        // For that firste step I need to recalculate the upstream cumulated elevation of each nodes for that basin:
        if(nedd_to_cumulate == true)
        {
          for(size_t j=0;j<current_node_orders["stack_order"][i].size();j++)
          {
            int this_node = current_node_orders["inverted_stack_order"][i][j];
            int receiving_row, receiving_col, receiving_node; current_FI.retrieve_receiver_information(this_node,receiving_node,receiving_row,receiving_col);
            // std::cout << "ACCUMUCHECK: row||col||row||col :-> " << current_node_orders["inverted_rows"][i][j] << "||" << current_node_orders["inverted_cols"][i][j] << "||" << receiving_row << "||" << receiving_col << endl;
            if(this_node!=receiving_node)
            {
              cumulated_array_of_upstream_dist[receiving_row][receiving_col] += (cumulated_array_of_upstream_dist[current_node_orders["inverted_rows"][i][j]][current_node_orders["inverted_cols"][i][j]] + dat_new_elev[dat_new_elev.size() - 1 - j]); // Note that my elevation is stackordered in my vector dat_new_elev so i want it backward (inverted stakc)              
            }
          }
        }
        // Need to reaccumulate at each iterations
        nedd_to_cumulate = true;

        for(size_t j=0; j<current_node_orders["stack_order"][i].size(); j++)
        {

          // Getting my stacks
          int this_node = current_node_orders["stack_order"][i][j];
          int this_row = current_node_orders["rows"][i][j];
          int this_col = current_node_orders["cols"][i][j];
          int receiving_node; current_FI.retrieve_receiver_information(this_node,receiving_node);

          // Simplifying that part of the equation!
          float second_part_of_nnew_zeta_equ = 0;
          // Avoid /0 if no donors. Need to check lim nah should be good
          if(current_FI.retrieve_ndonors_to_node(this_node) > 0)
            second_part_of_nnew_zeta_equ = (Gb/current_FI.retrieve_ndonors_to_node(this_node)) * cumulated_array_of_upstream_dist[this_row][this_col];

          // Solving equation (20)
          float new_zeta = (b_t[j] - second_part_of_nnew_zeta_equ + stack_ordered_Fi[j] * current_elev_rec[receiving_node]) / (1+stack_ordered_Fi[j]);
          // Saving the elevation o my node in the binary tree of receiving nodes
          current_elev_rec[this_node] = new_zeta;
          // Checking the tolerance
          if(std::abs(dat_new_elev[j] - new_zeta) > max_iter_tol )
            max_iter_tol = std::abs(dat_new_elev[j] - new_zeta);

          // Implementing my new elevation for next iteration
          dat_new_elev[j] = new_zeta;

          // DEBUG TO KEEP
          // std::cout << "proc: " << j << " NZ = " << new_zeta << " bt = " << b_t[j] << " Fi = " << stack_ordered_Fi[j]  << " Strating elevation: " << current_topography.get_data_element(this_row,this_col) << " Cumulat: " << cumulated_array_of_upstream_dist[this_row][this_col] << std::endl;
          // If a nan is generated, the code panics and die, so I put a hard lock here!
          if(isnan(new_zeta))
          {
            std::cout << "FATAL ERROR: NAAAAAAAAAAAAN. Something critical went wrong during the Gauss-Seidel solving. Sorry not sorry." << std::endl;
            exit(EXIT_FAILURE);
          }
          // Need to reinitialise the upstream accumulated elevation  as I need to recalculate it at each iterations!
          cumulated_array_of_upstream_dist[this_row][this_col] = 0;
        }


        // Need to reinitialise the upstream accumulated elevation !
        if(n_iterations>250)
        {
          std::cout << "FATAL WARNING :: Stopping the Gauss-Seidel iterations because it reached 250 iterations. That's not normal! If this happens to a small anomaleous basin it is ok but beware!" << std::endl;
          max_iter_tol = 0; // forcing the stop
        }

      }while(max_iter_tol> 1e-3); // iterative scheme runs until tol is achieved. in that case, mm. It can matter is really short time steps or really small ulpift are wanted. Needs to be significantly smaller than dt*U
      // std::cout << "Solved  after " << n_iterations << " iterations " << std::endl;
      // Finally getting the new elevation!
      for(size_t j=0; j<current_node_orders["stack_order"][i].size(); j++)
      {
        int resolving_row = current_node_orders["rows"][i][j];
        int resolving_col = current_node_orders["cols"][i][j];
        if(resolving_row != 0 && resolving_row != current_topography.get_NRows()-1 && resolving_col != 0 && resolving_col != current_topography.get_NCols()-1)
          new_elevation.set_data_element(current_node_orders["rows"][i][j],current_node_orders["cols"][i][j], dat_new_elev[j]);
      }


    }
    // std::cout << "Finished timestep" << std::endl;

    current_topography = new_elevation;
  }
  std::cout << "Formatting the output!" << std::endl;

  std::array<size_t,2> sizla = {current_topography.get_NRows(),current_topography.get_NCols()};
  xt::xtensor<float,2> output(sizla);
  for(size_t i=0;i<current_topography.get_NRows();i++)
  {
    for(size_t j=0;j<current_topography.get_NCols();j++)
    {
      output(i,j) = current_topography.get_data_element(i,j);
    }
  }

  return output;

}

// this functions solves the local newton scheme required at each iterations of the Gauss-Seidel solver
// See Yuan et al., 2019 and espcially equation 23
// Newton-rhapson iterative scheme uses the tangent of a function to find its root: x_n+1 = x_n - epsilon while epsilon > tolerance 
// here I redefine 
// a = K*p^m*A^m*dt
// b = dl_i
// c = elevation of receiver at the same t+dt k+1
// and d = right hand side of equation 23 (everything is known)
// which allows me to rewrite the equation 23 into f(x) = 0 <-> x+ a(x/b -c/b)^n - D = 0
// and its derivative f'(x) = (a*n/b) * (x/b - c/b)^(n-1) + 1
// initial guess will be f(x)/f'(x) with x = elevation at t plus 1 and k
// final aim is to get elevation at t+1 k+1
// Any question about my math, please refer to my hand-written notebook somewhere on my desk, room lower lewis drummond street.
float Carpythians::local_newton_rhapson_solver_for_SSPL(float a, float b, float c, float d, float neq, float htp1k, float toleritude)
{

  // I need to keep track of the iterations
  int n_iterations = 0;
  float zeta_to_guess_oy = htp1k;
  float epsilon = 0;
  // let's iterate
  do
  {
    // This is a comment
    n_iterations++;
    // Calculate new epsilon: f(x)/f'(x)
    epsilon = (zeta_to_guess_oy + a * std::pow(zeta_to_guess_oy/b - c/b, neq) - d) / ( (a*neq* std::pow(zeta_to_guess_oy/b - c/b, neq-1) / b) + 1);
    // Apply the epsilon
    zeta_to_guess_oy -= epsilon;
    // std::cout << "EPSILON = " << epsilon << " a " << a << " b " << b << " c " << c << " d " << d << std::endl;
    // check on my iterations to avoid infinite loop in the case of divergence
    if(n_iterations>250)
    {
      // std::cout << "FATAL WARNING:: Local newton-rhapson scheme reached 250 iterations, Not normal, beware if you see that warning several millions of time" << std::endl;
      epsilon = 0;
    }


  }while(abs(epsilon) > toleritude);

  if(isnan(zeta_to_guess_oy))
  {
    // std::cout << "Zetzet: " << zeta_to_guess_oy << " htp1k " << htp1k << " a " << a << " b " << b << " c " << c << " d " << d << std::endl;
    zeta_to_guess_oy = htp1k;
    FATAL_WARNINGS_NRITERATIONS++;
      // exit(EXIT_FAILURE);
  }


  // I am done yo!
  // This is slightly frustrating that it only takes few lines of code haha
  return zeta_to_guess_oy;
}


// Implementing the general case of STSPL (Yuan et al., 2019)
xt::pytensor<float,2> Carpythians::run_STSPL(float dt, float n_dt, xt::pytensor<float,2>& Kb, xt::pytensor<float,2>& sediment_thickness, float Ks, float Gb, float Gs, xt::pytensor<float,2> uplift,  float meq, float neq, bool lake)
{

  if(lake)
  {
    float fill_epsilon = 0;
    current_topography = current_topography.fill(fill_epsilon);
  } 
  // First step: gather some common info
  // Setting up the new elevation
  TNT::Array2D<float> this_new_topo(current_topography.get_NRows(), current_topography.get_NCols(), 1.5);
  for (size_t i =0; i<current_topography.get_NRows(); i++)
  {
    for (size_t j=0; j<current_topography.get_NCols();j++)
    {
      this_new_topo[i][j] = current_topography.get_data_element(i,j);
    }
  }
  LSDRaster new_elevation(current_topography.get_NRows(), current_topography.get_NCols(), current_topography.get_XMinimum(), 
    current_topography.get_YMinimum(), current_topography.get_DataResolution(), current_topography.get_NoDataValue(), this_new_topo);


  // I need a cumulated terms of equation (24) SUM_j=ups(htj +Ujdt - hjy+dtk)!
  // Basically using the fastscape inverted ordering to accumulate current node elevation to receiver node without including it for oneself
  // this is basically representing the sum part of the bterm!
  float goulg = 0;
  TNT::Array2D<float> cumulated_array_of_upstream_stuff(current_topography.get_NRows(), current_topography.get_NCols(), goulg);

  // AAAAAALLLLLLRIGHT
  // Iterate through all the time steps
  // vector<float> medit;
  for(int dat_t = 1; dat_t <= n_dt; dat_t++)
  {

    std::cout << "processing dt # " << dat_t << std::endl;
    // I need to reprocess FlowInfo to get the new fastscape order
    // First, checking if you want to deal with endoreic depressions

    this->reprocess_FlowInfo();

   

    // Now that this is done I can run the Gauss-Seidel iterative solver per watershed
    #pragma omp parallel for num_threads(max_n_threads) schedule(dynamic,1)
    for(int i=0; i<int(current_node_orders["stack_order"].size()); i++) // recasting size_t to int because windows f*****rd compiler DoEs NoT uNdErStAnD oMp WiTh UnSiGnEd InTeGeR. What a lackadaisical compiler.
    {

      int n_iterations = 0;
      // std::cout << i << "//" << current_node_orders["stack_order"].size() << std::endl;
      if(current_node_orders["stack_order"][i].size()>0)
      {

        // uplifting base-level! temporary thing
        int brow,bcol,bnode; bnode = current_node_orders["stack_order"][i][0] ;current_FI.retrieve_current_row_and_col(bnode,brow,bcol);
        new_elevation.set_data_element(brow, bcol, current_topography.get_data_element(brow,bcol) + dt * uplift(brow,bcol) );
        float max_iter_tol = -9999;
        do
        {
          n_iterations++;
            // std::cout << " Cumulating "<< "||";
          this->cumulate_for_STSPL_eq23_righthand_side(cumulated_array_of_upstream_stuff, new_elevation.get_RasterData(), current_node_orders["inverted_stack_order"][i], uplift,  dt , false, true);

          // this records the maximum delta between each k step of the gauss-siedel scheme in order to know when to stop
          max_iter_tol = -9999;
          for(size_t j=0; j<current_node_orders["stack_order"][i].size(); j++)
          {
            // Getting all my nodes informations
            int this_node = current_node_orders["stack_order"][i][j], this_row,this_col, receiving_node, receiving_row,receiving_col;
            current_FI.retrieve_current_row_and_col(this_node,this_row,this_col); current_FI.retrieve_receiver_information(this_node,receiving_node,receiving_row,receiving_col);
            // Nothing happens if I am a baselevel (so far)
            // std::cout << " TN: "<< this_node << " RM: " << receiving_node << "||";
            if(this_node != receiving_node)
            {
              // getting all the different elevation involved
              int tildAi = current_FI.retrieve_ndonors_to_node(this_node);
              float np1elev,this_elev = current_topography.get_data_element(this_row,this_col), np1_recelev = new_elevation.get_data_element(receiving_row,receiving_col), this_new_elev = new_elevation.get_data_element(this_row,this_col);
              // Getting the a,b,c,d constant for the local newton-rhapson scheme (see the local_newton_rhapson_solver_for_SSPL comments for explanations )
              // Before doing anything, I need to determine my K:
              float this_erod, this_G;
              if(sediment_thickness(this_row,this_col) <= 0)
              {
                // If I am hitting the bedrock, I am having the bedrock K
                this_erod = Kb(this_row, this_col);
                this_G = Gb;
              }
              else
              {
                // else I have the erodibility of the sediment
                this_erod = Ks;
                this_G = Gs;
              }
              float a = this_erod * std::pow(current_FI.get_DrainageArea_square_m(this_node),meq) * dt;
              float b = current_FI.get_Euclidian_distance(this_node,receiving_node);
              // checker for division by 0
              float c = np1_recelev;
              float secpart = 0;
              if(tildAi > 0)
               secpart = (this_G/tildAi) * cumulated_array_of_upstream_stuff[this_row][this_col];
              float d = this_elev + uplift(this_row,this_col) * dt + secpart;
              // if(isnan(d))
                // std::cout << "d: " << d << " tildAi:  " << tildAi << " cumulated_array_of_upstream_stuff[this_row][this_col]: " << cumulated_array_of_upstream_stuff[this_row][this_col] << " secpart " << secpart << " " << std::endl;

              // Soving
              
              np1elev = this->local_newton_rhapson_solver_for_SSPL(a, b,  c,  d, neq, this_new_elev, 1e-3);
              // std::cout << " previous zeta: " << this_new_elev << "|| previous one: " << np1elev << std::endl;
              
              if(isnan(np1elev))
              {
                std::cout << "d: " << d << " tildAi:  " << tildAi << " cumulated_array_of_upstream_stuff[this_row][this_col]: " << cumulated_array_of_upstream_stuff[this_row][this_col] << " secpart " << secpart << " " << std::endl;
                exit(EXIT_FAILURE);
              }

              // I need to adapt my sediment thickness for the next round to make sure I am using the right K
              // Adding the delta (scooping sediment if erosion, aggrading if deposition) 
              sediment_thickness(this_row,this_col) = sediment_thickness(this_row,this_col) + np1elev - this_new_elev;
              if(sediment_thickness(this_row,this_col)<0)
                sediment_thickness(this_row,this_col) = 0;
              

              // std::cout << "j: " << j << " solved! " << "||";
              // std::cout << " the new_elev  "<< np1elev << " old one: " << this_new_elev << " TOL: " <<  max_iter_tol << std::endl;
              if(abs(np1elev - this_new_elev ) > max_iter_tol)
              {
                max_iter_tol = abs(np1elev - this_new_elev);
              }
              
              new_elevation.set_data_element(this_row,this_col,np1elev);
            }
            else
            {
              // TODO -> deal with lakes
              0;
            }
          }
          

          if(max_iter_tol == -9999)
          {
            // std::cout<< "WARNING e754BF::Ignore so far" << std::endl;
            max_iter_tol = 0;
          }

          if(n_iterations>250)
          {
            // std::cout << "FATAL WARNING :: Stopping the Gauss-Seidel iterations because it reached 250 iterations. That's not normal! If this happens to a small anomaleous basin it is ok but beware!" << std::endl;
            max_iter_tol = 0; // forcing the stop
          }
        }while(abs(max_iter_tol)>=1e-3);
        // medit.push_back(float(n_iterations));
      }
      // END OF PARALLEL ZONE

      // std::cout << "Done with " << i << "//" << current_node_orders["stack_order"].size() << std::endl;

      // and we are ready to roll!!
    }
    // std::cout<< "REACH HERE????" << endl;


    current_topography.set_data_array(new_elevation.get_RasterData());
    if(lake)
    {
      float fill_epsilon = 0;
      current_topography = current_topography.fill(fill_epsilon);
    }
    // std::cout<< "REACH HERE2????" << endl;



  }

  // std::cout << "Median n iteration is " << get_median(medit) << std::endl;
  // std::array<size_t,2> sizla = {current_topography.get_NRows(),current_topography.get_NCols()};

  std::array<size_t,2> sizla = {current_topography.get_NRows(),current_topography.get_NCols()};

  xt::xtensor<float,2> output(sizla);
  // std::cout<< "REACH HERE 4????" << endl;
  for(size_t i=0;i<current_topography.get_NRows();i++)
  {
    // std::cout << "i:" << i  << std::endl; // IT CRASHES BEFORE THAT
    for(size_t j=0;j<current_topography.get_NCols();j++)
    {
      // std::cout << "i:" << i << " j:" <<  j << std::endl;// IT CRASHES BEFORE THAT
      output(i,j) = current_topography.get_data_element(i,j);
    }
  }
  // std::cout<< "REACH HERE 5????" << endl;
  return output;
}


// Internal function that reprocess the Flow Informations
void Carpythians::reprocess_FlowInfo()
{
  current_FI = LSDFlowInfo(BoCo, current_topography);
  current_node_orders = current_FI.get_map_of_vectors();
  // checking if everything went well
  if(current_node_orders["stack_order"].size() == 0)
  {
    std::cout << "Not a single node in my stack, something went badly wrong" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Internal function that cumulate the right-hand side of eq.23 (Yuan et al., 2019 JGR -> STSPL)
void Carpythians::cumulate_for_STSPL_eq23_righthand_side(TNT::Array2D<float>& cumularray, TNT::Array2D<float> elevatk, vector<int>& custom_inverted_stack, xt::pytensor<float,2>& uplift, float dt , bool all_nodes, bool reinitialise)
{

  // if you need reinitialisation to 0
  // I need to do it before to keep efficienty while reaccumulating
  if(reinitialise)
  {

    // Are we dealing with all the nodes
    if(all_nodes)
    {
      // if all nodes I assume that this can be parallelised
      #pragma omp parallel for num_threads(max_n_threads)
      for(int i=0; i<current_topography.get_NRows(); i++)
      {
        for(int j=0; j<current_topography.get_NCols(); j++)
        {
          cumularray[i][j] = 0;
        }
      }
    }
    // or just with specific array, which in this case is already in a multi-threaded logic, no need to trigger a threadception
    else
    {
      for(size_t i=0; i<custom_inverted_stack.size();i++)
      {
        int this_node = custom_inverted_stack[i],row,col;current_FI.retrieve_current_row_and_col(this_node,row,col);
        cumularray[row][col] = 0;
      }
    }
  }

  // If all the nodes need to be processed, I can paralellise the code by baselevels
  if(all_nodes)
  { 
    #pragma omp parallel for num_threads(max_n_threads) 
    for(int i=0;i<int(current_node_orders["stack_order"].size());i++)
    {
      for(size_t j=0;j<current_node_orders["stack_order"][i].size();j++)
      {
        int this_node = current_node_orders["inverted_stack_order"][i][j];
        int receiving_row, receiving_col, receiving_node; current_FI.retrieve_receiver_information(this_node,receiving_node,receiving_row,receiving_col);
        // Only cumulating if this is not the baselevel
        if(this_node!=receiving_node)
        {
          int this_col = current_node_orders["inverted_cols"][i][j];
          int this_row = current_node_orders["inverted_rows"][i][j];
          cumularray[receiving_row][receiving_col] += (cumularray[this_row][this_col] + current_topography.get_data_element(this_row,this_col) + dt*uplift(this_row,this_col) - elevatk[this_row][this_col] );
          // std::cout << "ACCUMUCHECK: row||col||val :-> " << current_node_orders["inverted_rows"][i][j] << "||" << current_node_orders["inverted_cols"][i][j] << "||" << custom_inverted_stack[current_node_orders["inverted_rows"][i][j]][current_node_orders["inverted_cols"][i][j]] << endl;
        }
      }
    }
  }
  // Else I am simply using the cumulative from that particular set of node
  else
  {
    for(size_t j=0;j<custom_inverted_stack.size();j++)
    {
      int this_node = custom_inverted_stack[j], this_row,this_col; current_FI.retrieve_current_row_and_col(this_node,this_row,this_col);
      int receiving_row, receiving_col, receiving_node; current_FI.retrieve_receiver_information(this_node,receiving_node,receiving_row,receiving_col);
      // Only cumulating if this is not the baselevel
      if(this_node!=receiving_node)
      {
        cumularray[receiving_row][receiving_col] += (cumularray[this_row][this_col] + current_topography.get_data_element(this_row,this_col) + dt*uplift(this_row,this_col) - elevatk[this_row][this_col] );
        // std::cout << "ACCUMUCHECK: row||col||val :-> " << current_node_orders["inverted_rows"][i][j] << "||" << current_node_orders["inverted_cols"][i][j] << "||" << custom_inverted_stack[current_node_orders["inverted_rows"][i][j]][current_node_orders["inverted_cols"][i][j]] << endl;
      }
    }
  }

  // Done. I donnae return anything as my cumulative array is given in-place.

}



#endif