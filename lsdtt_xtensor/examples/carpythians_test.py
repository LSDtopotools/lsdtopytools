import matplotlib;matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from lsdtt_xtensor_python import Carpythians_cpp as CCCP
import time as clock

# Model initiation
nrows = 50
ncols = 100
xmin = 0;ymin = 0
res = 200
ndv = -9999
data = np.random.random( (nrows,ncols) ).astype(np.float32)*5
data[15:35,:] += 500
# data[data<1000] = 0
BC = np.asarray([[0,0],[0,0]]).astype(np.int16)

# meq, neq
meq = 0.6
neq = meq/0.45

# Model runtime param
dt = 2000
n_dt_per_it = 20 # 200000

Kb = np.zeros( (nrows,ncols) ) + 1e-6
Ks = 1e-4
Gb = 0.
Gs = 0.

uplift = np.zeros( (nrows,ncols) )
uplift[0,:] = 0
uplift[-1,:] = 0
uplift[15:35,:] = 0.0005

sedthick = np.zeros( (nrows,ncols) )



USSR = CCCP( nrows,  ncols,  xmin,  ymin,  res,ndv,  data, BC,4)
last = np.copy(data)
for i in range(100):
	st = clock.time()
	test = USSR.run_STSPL(dt, n_dt_per_it, Kb, sedthick, Ks, Gb, Gs, uplift, meq, neq, True)
	print("This run took %s seconds"%(clock.time() - st) )

	cb = plt.imshow(test,extent = [xmin,xmin+ncols*res,ymin,ymin+nrows*res], cmap = "gist_earth", vmin = 0, vmax = np.percentile(test,90))
	plt.colorbar(cb)
	plt.savefig("Carpythians_Test/LAKE_test_%s.png" %i, dpi = 500)
	plt.clf()

uplift[0:15] = 0.0001

for i in range(100,200):
	st = clock.time()
	test = USSR.run_STSPL(dt, n_dt_per_it, Kb, sedthick, Ks, Gb, Gs, uplift, meq, neq, True)
	print("This run took %s seconds"%(clock.time() - st) )

	cb = plt.imshow(test,extent = [xmin,xmin+ncols*res,ymin,ymin+nrows*res], cmap = "gist_earth", vmin = 0, vmax = np.percentile(test,90))
	plt.colorbar(cb)
	plt.savefig("Carpythians_Test/LAKE_test_%s.png" %i, dpi = 500)
	plt.clf()
	
	# last = np.copy(data)