#ifndef LSD_xtensot_utils_HPP
#define LSD_xtensot_utils_HPP

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <string>

namespace xtlsd
{
	std::map<int, std::map<std::string,float> > _comparison_stats_from_2darrays(xt::pytensor<int,2>& arr1, xt::pytensor<float,2>& arr2, float ignore_value, size_t nRows, size_t nCols);
	std::map<int, std::vector<float> > _get_groupped_values(xt::pytensor<int,2>& arr1, xt::pytensor<float,2>& arr2, float ignore_value, size_t nRows, size_t nCols);
	std::vector<float> _gaussian_KDE(xt::pytensor<float,1>& x_val, xt::pytensor<float,1>& y_val , float h);
	std::map<int, std::vector< std::vector<float> > > _growing_window_stat(xt::pytensor<double,2>& base_array, size_t nRows, size_t nCols, int min_window, int step, int nstep);
	std::map<std::string, xt::pytensor<float,1> > _get_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<float, 1>& Y, float interval, int nthread);
	std::map<int, std::vector<float> > _proportion_median_profile(xt::pytensor<float, 1>& X, xt::pytensor<int, 1>& Y, float interval, int nthread);
	xt::pytensor<float,2> inverse_weighted_distance(xt::pytensor<float,1>& x_coord, xt::pytensor<float,1>& y_coord, xt::pytensor<float,1>& z_coord, float x_min, float x_max, float y_min, float y_max, float res, float exponent);
} //end of namespace xtlsd

namespace mudd14partitioner
{
	std::map<std::string,xt::pytensor<float,1> > segment_data(xt::pytensor<float,1>& X_data, xt::pytensor<float,1>& Y_data, xt::pytensor<float,1>& sigmas, int min_seg_size);	
}

#endif