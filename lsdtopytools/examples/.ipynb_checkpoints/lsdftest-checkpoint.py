import matplotlib;matplotlib.use("Agg")
from matplotlib import pyplot as plt
from lsdtopytools import LSDfsl, numba_tools as nbt
import numpy as np


nrows = 50
ncols = 100

# Generating an initial topo
initial_topo = np.random.random_sample((nrows,ncols))
initial_topo[round(0.27*nrows):round(0.73*nrows),:] += 500
# Initialising the model
memod = LSDfsl(initial_topo, initial_time = 0, model_prefix = "LSDfsl_test1", verbose = 3)
# memod.update_topography_from_array(initial_topo, xl = 20e3, yl = 10e3, I_know_what_I_am_doing = True)

# modify some parameters
uplift = np.zeros((nrows,ncols)) + 0.0001
uplift[round(0.27*nrows):round(0.73*nrows),:] = 0.005 # range in the middle
uplift[0,:] = 0 # Top and bottom line = fixed height
uplift[-1,:] = 0 # Top and bottom line = fixed height
Kf = np.zeros((nrows,ncols)) + 5e-9
Kd = np.zeros((nrows,ncols)) + 3e-5
params = {}
params["xl"] = 20e3
params["yl"] = 10e3
params["dt"] = 2000
params["Kf"] = Kf
params["Ks"] = 1e-4
params["Kd"] = Kd
params["Kds"] = 1e-2
params["m"] = 1.11
params["n"] = 1.11/0.6
params["Gb"] = 0.5
params["Gs"] = 0.7
params["p_mult"] = -1
params["uplift"] = uplift
memod.update_model_parameters(**params)

testres = memod.run_model(n_time_step = 100, save_step = 5, save_to_file = False)

for i in range(len(testres["topography"])):
	fig,ax = plt.subplots()
	# HS = nbt.numba_hillshading(testres["topography"][i],memod.resolution,azimuth = 315,angle_altitude = 45,z_factor = 1., parallel = True, ignore_edge = False)
	# ax.imshow(HS, extent = [0,memod.parameters["xl"],0,memod.parameters["yl"]], cmap = "gray", vmin = 0, zorder = 1)
	cb = ax.imshow(testres["topography"][i], extent = [0,memod.parameters["xl"],0,memod.parameters["yl"]], cmap = "gist_earth", vmin = 0, zorder = 2,alpha = 0.6)
	# cb = ax.imshow(testres["topography"][i]-testres["basement_elevation"][i], extent = [0,memod.parameters["xl"],0,memod.parameters["yl"]], cmap = "viridis", vmin = 0, zorder = 2,alpha = 0.5 )
	cbar = plt.colorbar(cb, fraction=0.046, pad=0.04)
	cbar.ax.set_ylabel("Sediment Thickness (m)")

	plt.savefig("LSDfsl_test1/ST_%s.png"%(testres["time"][i]), dpi = 900)
	plt.close(fig)
