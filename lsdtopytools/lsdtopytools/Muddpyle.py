"""
Object wrapping muddpile, the Parsimonious Integrated Landscape Evolution landscape Evoution Model 
See https://lsdtopotools.github.io/LSDTT_documentation/LSDTT_MuddPILE.html for more details.
Still quite in development and early testing for my Carpathians paper.
Will document when it will be worth it and stable.
B.G.
"""
import pandas as pd
import numpy as np
try:
    from lsdtt_xtensor_python import muddpyle_cpp
    installed = True
except ImportError:
    installed = False
import os, glob, sys
from lsdtopytools import raster_loader as rl, LSDDEM, quickplot as qp



class MuddPyle(object):
    """docstring for MuddPyle"""
    def __init__(self, mode = "initial_surface", verbose = True, save_dir = "./", prefix = "BORIS",
        IS_nrows = 250, IS_ncols = 400, IS_resolution = 30, IS_diamond_square_order = 2, IS_diamond_square_relief = 50, IS_Parabola_relief = 50, IS_roughness_reflief = 5, IS_nitnlh = 0,
        m=1.1, n=1.9,Ke = 1e-7, Diffusion = 0.002, Sc = 1., uplift_field = 0.0005,
        ):
        """
           Constructor for MuddPyle object 
        """

        self.verbose = verbose
        self.save_dir = save_dir
        self.prefix = prefix
        if(mode=="initial_surface"):
            self.cppmod = muddpyle_cpp(self.verbose)
            self.cppmod.set_save_path_name(save_dir, prefix)
            self.nrows = IS_nrows
            self.ncols = IS_ncols
            self.resolution = IS_resolution
            self.cppmod.initialise_model_kimberlite(self.nrows,self.ncols, IS_resolution, IS_diamond_square_order, IS_diamond_square_relief, IS_Parabola_relief, IS_roughness_reflief, IS_nitnlh)
           

        # Checking the inputs
        tke = Ke[0,0] if isinstance(Ke, np.ndarray) else Ke

        #TEMP RECASTING
        uplift_field = np.zeros((self.nrows,self.ncols)) + uplift_field if(~(isinstance(uplift_field,np.ndarray))) else 0
        Ke = np.zeros((self.nrows,self.ncols)) + Ke if(~(isinstance(Ke,np.ndarray))) else 0

        # Default features
        self.cppmod.set_global_default_param(Diffusion, Sc, tke)
        ## Dealing with erodibility type
        if isinstance(Ke, np.ndarray):
            self.erodibility_dtype = "variable" 
        else:
            self.erodibility_dtype = "fixed"

        ## Dealing with erodibility type
        if isinstance(Ke, np.ndarray):
            self.uplift_dtype = "variable" 
        else:
            print("Muddpyle is under construction and only support array of uplift so far.")
            print("See https://lsdtopotools.github.io/LSDTT_documentation/LSDTT_MuddPILE.html for direct use of MuddPILE")
            self.uplift_dtype = "fixed"
            quit()


        self.uplift_field = uplift_field
        self.erodibility_field = Ke

        # Dealing with the cpp object!! important
        self.cppmod.set_uplift_raster(self.uplift_field)
        self.cppmod.set_K_raster(self.erodibility_field)
        ## concavity
        self.cppmod.set_SPL_exponents(m,n)
        ## Leave that alone FROMAGE
        self.cppmod.set_uplift_mode(0)
        ## BOUNDATY CONDITION FIXED FOR THE MOMENT
        self.cppmod.set_boundary_conditions(["b","p","b","p"])
        self.current_frame = 1
        self.current_time = 0



        # Creating the directory if not existing
        directory = os.path.dirname(self.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)



    def run_model(self,dt = 250, time = 250000, save_step = 500, fluvial = True, hillslope = False):
        import math

        self.cppmod.deal_with_time(dt, dt, self.current_time+time,0)    
        self.cppmod.set_print_interval_in_year(save_step)
        # Run the model
        self.cppmod.run_model(fluvial, hillslope,True, self.current_frame, False)
        # updtae the save charac n stuffs
        self.current_frame += math.floor(time/dt)+1
        self.current_time += time

    def update_Ke(self, new_Ke = 1e-5):

        if(~(isinstance(new_Ke,np.ndarray))):
            new_Ke = np.zeros((self.nrows,self.ncols)) + new_Ke

        self.erodibility_field = new_Ke
        self.cppmod.set_K_raster(self.erodibility_field)

    def update_Uplift(self, new_U = 0.0001):
        print("TESTTTTTTTTTT")
        if(~(isinstance(new_U,np.ndarray))):
            new_U = np.zeros((self.nrows,self.ncols)) + new_U

        self.uplift_field = new_U
        self.cppmod.set_uplift_raster(self.uplift_field)

    def plot_anim(self, fps = 10, rename_suffix = "", dpi = 300, HS_alpha = 0.75):

        from matplotlib.animation import FFMpegWriter
        from matplotlib import pyplot as plt

        # Beware of the nested function
        def plotty(direc,fname, ret = False, figt = None, dpi = 300, HS_alpha = 0.85):
            tdem = LSDDEM(path = direc, file_name = fname,already_preprocessed = True)
            mx = np.percentile(tdem.cppdem.get_PP_raster(), 90)
            if(ret):
                return qp.plot_nice_topography(tdem ,figure_width = 5, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, alpha_hillshade = HS_alpha, color_min = 0, color_max = mx, dpi = 500, output = "nothing", format_figure = "png", fontsize_ticks = 6 , fontsize_label = 8,hillshade_cmin = None, hillshade_cmax = None, fig = figt, colorbar = True)
            else:
                qp.plot_nice_topography(tdem ,figure_width = 5, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, alpha_hillshade = HS_alpha, color_min = 0, color_max = mx, dpi = 500, output = "save", format_figure = "png", fontsize_ticks = 6 , fontsize_label = 8,hillshade_cmin = None, hillshade_cmax = None, colorbar = True)


        base_df = pd.read_csv(self.save_dir + self.prefix + "_model_info.csv")
        first_frame = base_df["Frame_num"].iloc[0] if(base_df["Frame_num"].iloc[0]!= -1) else base_df["Frame_num"].iloc[1]
        tdem = LSDDEM(path = self.save_dir, file_name = self.prefix+str(first_frame)+".bil",already_preprocessed = True)
        mx = np.percentile(tdem.cppdem.get_PP_raster(),90)
        fig,ax = qp.plot_nice_topography(tdem ,figure_width = 5, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, alpha_hillshade = HS_alpha, color_min = 0, color_max = mx, dpi = 500, output = "return", format_figure = "png", fontsize_ticks = 6 , fontsize_label = 8,hillshade_cmin = None, hillshade_cmax = None, colorbar = True)
        moviewriter = FFMpegWriter(fps=fps)
        with moviewriter.saving(fig, self.save_dir+self.prefix+rename_suffix+".mp4", dpi=dpi):

            for j in base_df["Frame_num"]:
                if(j!=-1):
                    plotty(self.save_dir,"%s.bil"%(self.prefix+str(j)), True, fig,HS_alpha = HS_alpha)
                    moviewriter.grab_frame()
                    plt.clf()