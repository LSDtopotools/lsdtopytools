#ifndef LSD_xtensor_utils_CPP
#define LSD_xtensor_utils_CPP

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <utility> 
#include <array>
#include <limits>
#include "LSDStatsTools.hpp"

#include "LSD_xtensor_utils.hpp"
#include "LSDMostLikelyPartitionsFinder.hpp"
#include "TNT/tnt.h"


#include <omp.h>  //Used for OpenMP run-time functions
#ifndef _OPENMP
  #define omp_get_thread_num()  0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif


namespace xtlsd
{
	std::map<int, std::map<std::string,float> > _comparison_stats_from_2darrays(xt::pytensor<int,2>& arr1, xt::pytensor<float,2>& arr2, float ignore_value, size_t nRows, size_t nCols)
	{
		// this code generate some grouped stats to compare 2 arrays

		// First I am declaring my vectors containing the data
		// This contains the vector of data for each different values of array 1
		std::map<int,std::vector<float> > key_to_values;
		std::map<int, std::map<std::string,float> > output;
		std::map<int,int> key_to_N, key_to_N_tot;

		int cpt =0;
		// std::vector<size_t> s = arr1.shape();


		// let's feed it
		for(size_t i = 0; i < nRows; i++)
		{
			for(size_t j = 0; j < nCols; j++)
			{
				// Just checking if I need to ignore this value
				float this_value = arr2(i,j);
				if(this_value != ignore_value)
				{
					cpt ++;
					int this_key = arr1(i,j);
					// checking if the value has already been implemented 
					if(key_to_values.count(this_key) == 0)
					{

						// std::cout << "DEBUG::got new value" << std::endl;
						// If not I am creating a new vector
						vector<float> temp;
						temp.push_back(this_value);
						key_to_values[this_key] = temp;
						key_to_N[this_key] = 1;

					}
					else
					{
						// Otherwise I am just adding the value to the existing vector
						key_to_values[this_key].push_back(this_value);
						key_to_N[this_key] = key_to_N[this_key] + 1;
					}
				}
				else
				{
					int this_key = arr1(i,j);
					// checking if the value has already been implemented 
					if(key_to_values.count(this_key) == 0)
					{
						key_to_N_tot[this_key] = 1;
					}
					else
					{
						key_to_N_tot[this_key] = key_to_N_tot[this_key] + 1;
					}
				}
			}
		}
		// std::cout << "DEBUG::got " << cpt << " pixels to analyse" << std::endl;

		// I have all my data, now I need to generates the stats
		std::map<int,int>::iterator it = key_to_N.begin();
		for(;it!=key_to_N.end();it++)
		{
			int this_key = it->first;
			int N_el = it->second;
			vector<float> this_vec = key_to_values[this_key], vec_of_val(9);

			vec_of_val = calculate_descriptive_stats(this_vec);
			std::map<std::string,float> Ma_stats;
			Ma_stats["min"] = vec_of_val[0];
			Ma_stats["first_quartile"] = vec_of_val[1];
			Ma_stats["median"] = vec_of_val[2];
			Ma_stats["third_quartile"] = vec_of_val[3];
			Ma_stats["max"] = vec_of_val[4];
			Ma_stats["mean"] = vec_of_val[5];
			Ma_stats["std_dev"] = vec_of_val[6];
			Ma_stats["std_error"] = vec_of_val[7];
			Ma_stats["MAD"] = vec_of_val[8];
			Ma_stats["N"] = key_to_N[this_key];
			Ma_stats["N_plus_nodata"] = key_to_N_tot[this_key];



			output[this_key] = Ma_stats;
		}

		return output;
	}

	std::map<int, std::vector<float> > _get_groupped_values(xt::pytensor<int,2>& arr1, xt::pytensor<float,2>& arr2, float ignore_value, size_t nRows, size_t nCols)
	{
		// this code extracts groupped of correlated values from arr2 (float) compare to arr1 (integer) 

		// First I am declaring my vectors containing the data
		// This contains the vector of data for each different values of array 1
		std::map<int,std::vector<float> > key_to_values;

		// let's feed it
		for(size_t i = 0; i < nRows; i++)
		{
			for(size_t j = 0; j < nCols; j++)
			{
				// Just checking if I need to ignore this value
				float this_value = arr2(i,j);
				if(this_value != ignore_value)
				{
					int this_key = arr1(i,j);
					// checking if the value has already been implemented 
					if(key_to_values.count(this_key) == 0)
					{

						// std::cout << "DEBUG::got new value" << std::endl;
						// If not I am creating a new vector
						vector<float> temp;
						temp.push_back(this_value);
						key_to_values[this_key]=temp;

					}
					else
					{
						// Otherwise I am just adding the value to the existing vector
						key_to_values[this_key].push_back(this_value);
					}
				}
			}
		}

		return key_to_values;
	}


	// KDE calculation for a vector of float using a gaussian kernel with a bandwith h
	//
	// BG - 04/01/2018
	std::vector<float> _gaussian_KDE(xt::pytensor<float,1>& x_val, xt::pytensor<float,1>& y_val , float h)
	{
	  std::vector<float> vout;
	  // get N
	  int nx = x_val.size();
	  int ny = y_val.size();

	  // Calculate the sum
	  // ### This precision for PI should be acceptable
	  float sum = 0, X = 0, Xi = 0, PI = 3.14159;
	  for(size_t i = 0; i<nx; i++)
	  {
	    // Setting the sample for this run of the loop
	    X = x_val[i]; // this sample
	    sum = 0; // reinitializing the sum for each sample
	    // summing the elements
	    for(size_t j=0; j<ny; j++)
	    {
	      // setting the testing for this run for this sum
	      Xi = y_val[j];
	      float y = 0;
	      y = (X-Xi/h);
	      // incrementing the sum: using a gaussian kernel for each X - Xi
	      sum += 1/(sqrt(2*PI)) * exp(-pow(y,2)/2);
	    }

	    // saving the KDE
	    vout.push_back((1/(ny*h)) * sum);
	  }
	  // Done, not that complicated after all
	  return vout;

	}


	std::map<int, std::vector< std::vector<float> > > _growing_window_stat(xt::pytensor<double,2>& base_array, size_t nRows, size_t nCols, int min_window, int step, int nstep)
	{

		// Output formatting
		std::map<int, std::vector< std::vector<float> > > out;

		// Getting the vector of window size
		std::vector<int> window(nstep);
		for(size_t i=0; i<nstep;i++)
		{
			window[i] = min_window;
			min_window = min_window + step;
		}

		// std::cout<<"DEBUG::Will Get " << std::bitset<8>(nstep).to_string() << " arrays " << std::endl;

		for(size_t i=0; i<nstep;i++)
		{
			std::vector<std::vector<float> > this_rast(nRows, std::vector<float>(nCols));
			int this_window_size = window[i];
			// std::cout << "DEBUG::Processing window size " << std::bitset<8>(this_window_size).to_string() << std::endl;
			// Attempt to multithread 
			// omp_set_num_threads(4);
			#pragma omp parallel num_threads(1)
			{
				#pragma omp for
				for(int row=this_window_size; row<nRows-this_window_size;row++)
				{
					for(int col=this_window_size; col<nCols-this_window_size;col++)
					{
						std::vector<float> these_val_to_reg(std::pow(this_window_size,2));
						size_t incr = 0;
						int tK =  int(row-this_window_size/2), tL = int(col-this_window_size/2);
						
						while(int(tK) % 2 != 0)
						{
							tK++;
						}
						while(int(tL) % 2 != 0)
						{
							tL++;
						}

						for(size_t k=int(tK); k<int(tK+this_window_size);k++)
						{
							for(size_t l=int(tL); l<int(tL+this_window_size);l++)
							{

								these_val_to_reg[incr] = base_array(k,l);
								incr++;
							}
						}

						these_val_to_reg.shrink_to_fit();
						float median = get_median(these_val_to_reg);
						this_rast[row][col] = median;
					}
				}
				#pragma omp barrier


				out[this_window_size] = this_rast;
			}
		}

		return out;


	}

	std::map<std::string, xt::pytensor<float,1> > _get_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<float, 1>& Y, float interval, int nthread)
	{
		// This function generate the required data for long profile analysis:
		// ie -> a median X array, a median Y array and its bounds with a minY array (representing the longest profile) and a 3rd quartile array
		// the arrays have to be sorted by X values

		size_t nelement = X.size();
		// std::cout << "DEBUG::Input size = " << nelement << std::endl;

		// First step in to get the X minimum, assuming you are working in a river network, this should not be too low:
		float x_min = X[0];

		// std::cout << "DEBUG::Min X is = " << x_min << std::endl;

		// I now have the minimum
		// let's get the shit done now: getting all the breaks
		// Basically in the algorithm is run for few elements, this will barely slow down the algorithm. 
		// But it will greatly help the cases where you wanna use multithreading because working on millions of points
		std::vector<pair<size_t,size_t> > breaks; // will host the breaks
		float this_val = x_min, next_val = x_min + interval;
		size_t last_n = 0;
		std::pair<size_t,size_t> tPPN;
		for(size_t n=0;n<nelement;n++)
		{
			this_val = X[n]; // X values to test
			// Here I am checking if the value still is in the same interval
			if(this_val>=next_val)
			{
				// if yes indeed, saving the break and moving to the next iteration
				tPPN = std::make_pair(last_n,n);
				breaks.push_back(tPPN);
				next_val = next_val + interval;
				last_n = n;
			}
		}
		// Adding the last element
		tPPN = std::make_pair(last_n,X.size()-1);
		breaks.push_back(tPPN);
		// std::cout << "DEBUG::break size = " << breaks.size() << std::endl;


		// Allocating memory for the output
		// std::array<size_t, 1> this_size = {breaks.size()}; //Testing ways to allocate new pytensor here;
		std::vector<float> medX(breaks.size());
		std::fill(medX.begin(),medX.end(),0);
		std::vector<float> medY(breaks.size());
		std::fill(medY.begin(),medY.end(),0);
		std::vector<float> min_Y(breaks.size());
		std::fill(min_Y.begin(),min_Y.end(),0);
		std::vector<float> quart3_Y(breaks.size());
		std::fill(quart3_Y.begin(),quart3_Y.end(),0);
		std::vector<float> quart1_Y(breaks.size());
		std::fill(quart1_Y.begin(),quart1_Y.end(),0);
		std::vector<float> N(breaks.size());
		std::fill(N.begin(),N.end(),0);
		// Alright I am now ready to run throughthe vector and get the median and the other stuffs
		#pragma omp parallel num_threads(nthread)
		{
			// the only for loop we want to apply the parallelism is that one
			#pragma omp for schedule(dynamic)
			for(int b=0; b<breaks.size(); b++)
			{
				// now we are in the multi threading thing so let's be careful
				// std::cout << "goulge" << std::endl;
				if(b<breaks.size()-1)
				{
					std::vector<float> tY,tX; // host temporal values
					float tmint = 99999999999;
					for (int it=int(breaks[b].first);it<int(breaks[b].second);it++)
					{
						tX.push_back(X[it]);
						tY.push_back(Y[it]);
						if(Y[it]<tmint)
							tmint = Y[it];
					}
	
					// std::cout << "DEBUG::it = " << b << std::endl;
	
					std::sort (tX.begin(),tX.end());
					std::sort (tY.begin(),tY.end());
					float tmedX = get_median(tX), tmedY = get_median(tY), t3Y= get_percentile(tY, 75), t1Y = get_percentile(tY, 25);
					medX[b] = tmedX;
					medY[b] = tmedY;
					min_Y[b] = tmint;
					quart3_Y[b] = t3Y;
					quart1_Y[b] = t1Y;
					N[b] = int(tY.size());
				}
				// std::cout << "Gabuugwe" << std::endl;

				// if(t3Y<tmedY)
				// 	std::cout << "DEBUG::THS SHOULD NOT HAPPEN, WHY IS THAT HAPPENNING: 3rd quartile " << t3Y << " and median " << tmedY << std::endl;	
			}
		// Implicit barrier
		}
		// yay out of the zone
		// Now I have to format the output
		std::map<std::string, xt::pytensor<float,1> > output;
		std::array<size_t, 1> sizla = {nelement};

		xt::xtensor<float,1>xmedX(sizla);
		xt::xtensor<float,1>xmedY(sizla);
		xt::xtensor<float,1>xmin_Y(sizla);
		xt::xtensor<float,1>xquart3_Y(sizla);
		xt::xtensor<float,1>xquart1_Y(sizla);
		xt::xtensor<float,1>xN(sizla);
		xmedX = xt::adapt(medX);
		xmedY = xt::adapt(medY);
		xmin_Y = xt::adapt(min_Y);
		xquart3_Y = xt::adapt(quart3_Y);
		xquart1_Y = xt::adapt(quart1_Y);
		xN = xt::adapt(N);

		output["X"] = xmedX;
		output["Y"] = xmedY;
		output["min_Y"] = xmin_Y;
		output["third_quartile"] = xquart3_Y;
		output["first_quartile"] = xquart1_Y;
		output["N"] = xN;

		return output;
	}

	std::map<int, std::vector<float> > _proportion_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<int, 1>& Y, float interval, int nthread)
	{
		// This function generate the required data for long profile analysis:
		// ie -> a median X array, a median Y array and its bounds with a minY array (representing the longest profile) and a 3rd quartile array
		// the arrays have to be sorted by X values

		size_t nelement = X.size();
		// std::cout << "DEBUG::Input size = " << nelement << std::endl;

		// First step in to get the X minimum, assuming you are working in a river network, this should not be too low:
		float x_min = X[0];

		// initializing a list of values
		std::map<int,int> list_of_values;


		// I now have the minimum
		// let's get the shit done now: getting all the breaks
		// Basically in the algorithm is run for few elements, this will barely slow down the algorithm. 
		// But it will greatly help the cases where you wanna use multithreading because working on millions of points
		std::vector<pair<size_t,size_t> > breaks; // will host the breaks
		float this_val = x_min, next_val = x_min + interval;
		size_t last_n = 0;
		std::pair<size_t,size_t> tPPN;
		for(size_t n=0;n<nelement;n++)
		{
			this_val = X[n]; // X values to test
			// Here I am checking if the value still is in the same interval
			if(this_val>=next_val)
			{
				// if yes indeed, saving the break and moving to the next iteration
				tPPN = std::make_pair(last_n,n);
				breaks.push_back(tPPN);
				next_val = next_val + interval;
				last_n = n;
			}
			if(list_of_values.count(Y[n])==0)
			{
				list_of_values[Y[n]] = 0; // Setting a perfect map of values = 0
			}
		}
		// Adding the last element
		tPPN = std::make_pair(last_n,X.size()-1);
		breaks.push_back(tPPN);
		// std::cout << "DEBUG::break size = " << breaks.size() << std::endl;


		// Allocating memory for the output
		// std::array<size_t, 1> this_size = {breaks.size()}; //Testing ways to allocate new pytensor here;
		std::vector<float> medX(breaks.size());
		std::fill(medX.begin(),medX.end(),0);
		std::vector<float> proportion_of_Y(breaks.size());
		std::fill(proportion_of_Y.begin(),proportion_of_Y.end(),0);

		// Filling the final map
		std::map<int, std::vector<float> > output;
		for(std::map<int,int>::iterator awah = list_of_values.begin(); awah != list_of_values.end(); awah++)
		{
			// key to the ID
			int this_ID = awah->first;
			// 0 vector
			std::vector<float> meVEC(breaks.size());
			std::fill(meVEC.begin(),meVEC.end(),0);
			// initializing the output with da right size
			output[this_ID] = meVEC;
		}

		// Alright I am now ready to run throughthe vector and get the median and the other stuffs
		#pragma omp parallel num_threads(nthread)
		{
			// the only for loop we want to apply the parallelism is that one
			#pragma omp for schedule(dynamic)
			for(int b=0; b<breaks.size(); b++)
			{
				// now we are in the multi threading thing so let's be careful
				// std::cout << "goulge" << std::endl;
				if(b<breaks.size()-1)
				{
					std::vector<float> tX; // host temporal values
					float tmint = 99999999999;
					std::map<int,int> this_set_of_val;
					this_set_of_val.insert(list_of_values.begin(),list_of_values.end());
					int n_val_tot = 0;
					for (int it=int(breaks[b].first);it<int(breaks[b].second);it++)
					{
						tX.push_back(X[it]);
						this_set_of_val[Y[it]] = this_set_of_val[Y[it]] + 1;
						n_val_tot++ ;
					}
	
					// std::cout << "DEBUG::it = " << b << std::endl;
	
					std::sort (tX.begin(),tX.end());
					float tmedX = get_median(tX);
					medX[b] = tmedX;

					// Now getting the different proportions
					for(std::map<int,int>::iterator awah = this_set_of_val.begin(); awah != this_set_of_val.end(); awah++)
					{
						int this_ID = awah->first;
						float tN = float(this_set_of_val[this_ID]), tNtot = float(n_val_tot);
						output[this_ID][b] = tN/tNtot;
					}

				}
				// std::cout << "Gabuugwe" << std::endl;

				// if(t3Y<tmedY)
				// 	std::cout << "DEBUG::THS SHOULD NOT HAPPEN, WHY IS THAT HAPPENNING: 3rd quartile " << t3Y << " and median " << tmedY << std::endl;	
			}
		// Implicit barrier
		}
		// yay out of the zone
		// we are done basically
		return output;
	}

	// This function generates a grid and calculate the inverse weighted distance interpolation from a range a x,y,z dataset
	// author: B.G. 04/12/2018
	xt::pytensor<float,2> inverse_weighted_distance(xt::pytensor<float,1>& x_coord, xt::pytensor<float,1>& y_coord, xt::pytensor<float,1>& z_coord, float x_min, float x_max, float y_min, float y_max, float res, float exponent)
	{
		// First step is to generate the grid
		size_t n_col = size_t(std::round((x_max - x_min)/res));
		size_t n_row = size_t(std::round((y_max - y_min)/res));
		std::array<size_t,2> shape = {n_row, n_col};
		xt::xtensor<float,2> IWD(shape);

		// Alright now let's code
		for(size_t i=0; i<n_row; i++)
		for(size_t j=0; j<n_col; j++)
		{	
			// First I need to get the distance of the grid point to any other points, to the power of the weighting factor (ahah that's quite unclear yo);
			std::vector<float> distance(x_coord.size());
			float min_distp = std::numeric_limits<float>::max(); // Maximum vaue ever
			size_t minID = -99999; // and its Index
			float sum_of_IvD = 0;

			for(size_t garg=0;garg<x_coord.size();garg++)
			{
				// eucidian distance
				distance[garg] = pow(sqrt(pow(x_coord[garg] - (j*res+x_min),2) +pow(y_coord[garg] - (i*res+y_min),2)), exponent);
				if(distance[garg]<min_distp){min_distp = distance[garg];minID = garg;} // Getting the minimum here
				sum_of_IvD += 1 / (distance[garg]);
			}
			// If I am literally on a point, I am taking its value, hashtag YOLO
			if(min_distp == 0){IWD(i,j) = z_coord[minID];}
			else
			{
				float value = 0;
				for(size_t garg=0;garg<x_coord.size();garg++)
				{
					value += z_coord[garg]/distance[garg]/sum_of_IvD;
				}
				IWD(i,j) = value;
			}
		}
		// And we are basically done here! Hopefully this should represent a nice way of getting quick data interpolation
		return IWD;

	}

} // end of namespace xtlsd


namespace mudd14partitioner
{

	std::map<std::string,xt::pytensor<float,1> > segment_data(xt::pytensor<float,1>& X_data, xt::pytensor<float,1>& Y_data, xt::pytensor<float,1>& sigmas, int min_seg_size)
	{


		std::vector<float> X_data_vec(X_data.size()), Y_data_vec(X_data.size()), sigmas_vec(X_data.size());

		for(size_t i=0; i< X_data.size(); i++)
		{
			X_data_vec[i] = X_data[i];
			Y_data_vec[i] = Y_data[i];
			sigmas_vec[i] = sigmas[i];
		}


		LSDMostLikelyPartitionsFinder Partitioner(min_seg_size,X_data_vec, Y_data_vec);

	    // Partition the data
	    //float sigma = 0.0005;  // this is a placeholder. Later we can use slope uncertainties. NOW USING MEASURED ERROR
	    // We use the standard error of the S values as the sigma in partitioner.
	    //cout << "This basin is: " << this_basin << endl;
	    Partitioner.best_fit_driver_AIC_for_linear_segments(sigmas_vec);

	    // Now we extract all the data from the partitions
	    vector<float> sigma_values;
	    sigma_values.push_back(1);
	    int node = 0;
	    vector<float> b_values;
	    vector<float> m_values;
	    vector<float> r2_values;
	    vector<float> DW_values;
	    vector<float> fitted_y;
	    vector<int> seg_lengths;
	    float this_MLE;
	    int this_n_segments;
	    int this_n_nodes;
	    float this_AIC;
	    float this_AICc;

	    Partitioner.get_data_from_best_fit_lines(node, sigma_values,
	                      b_values, m_values,r2_values, DW_values, fitted_y,seg_lengths,
	                      this_MLE,  this_n_segments,  this_n_nodes,
	                      this_AIC,  this_AICc);
		xt::xtensor<float,1> b_values_out = xt::adapt(b_values); 
		xt::xtensor<float,1> m_values_out = xt::adapt(m_values); 
		xt::xtensor<float,1> r2_values_out = xt::adapt(r2_values); 
		xt::xtensor<float,1> DW_values_out = xt::adapt(DW_values); 
		xt::xtensor<float,1> fitted_y_out = xt::adapt(fitted_y); 
		xt::xtensor<float,1> seg_lengths_out = xt::adapt(seg_lengths); 

		std::map<std::string,xt::pytensor<float,1> > output;
		output["b_values"] = b_values_out;
		output["m_values"] = m_values_out;
		output["r2_values"] = r2_values_out;
		output["DW_values"] = DW_values_out;
		output["fitted_y"] = fitted_y_out;
		output["seg_lengths"] = seg_lengths_out;

		return output;

	}


}
#endif