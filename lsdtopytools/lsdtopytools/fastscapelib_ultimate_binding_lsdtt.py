"""
This file attempt to link lsdtopytools with fastscapelib (at the moment fastscapelibfortran)
the idea si to manage to link our topographic analysis routines with the fastscape model
At the moment we are binding the fastscapelib-fortran, this might be tricky to install on your machine
But binaries shall be available anytime soon!
B.G.
"""
import numpy as np
import pandas as pd
from lsdtopytools import LSDDEM, raster_loader as rl
import matplotlib.pyplot as plt
import fastscapelib_fortran as fs
import scipy,sys,os



class LSDfsl(object):
  """
  The LSDFsl class deals with bindings fastscapelib-fortran model. 
  It helps controlling their python package, can link with basic lsdtopytools analysis and preformat the outputs to make it ready to inget into lsdtopytools
  Authors: B.G.
  """
  def __init__(self,initial_topo, initial_time = 0, model_prefix = "LSDfsl", verbose = 3 ):
    super(LSDfsl, self).__init__()
    # Defines the original time
    self.initial_time = initial_time
    self.current_time = initial_time
    # Defines the Speakability of the model. 0 is no words. 1 is Keep it to the important warnings, 2 activate general words and 3 debugging chats 
    self.verbose = verbose
    # THe model prefix will take care of saving the files
    self.prefix = model_prefix
    if not os.path.exists(self.prefix+"/"):
        os.makedirs(self.prefix+"/")


    # Initialising some checker
    self.is_model_built = False

    # Defaulting the parameters and their explanations
    self.parameters = {}
    self.parameter_explanations = {}
    self.parameters["nx"] = initial_topo.shape[1]; self.parameter_explanations["nx"] = "Number of cells in the x dimension, currently %s"%self.parameters["nx"]
    self.parameters["ny"] = initial_topo.shape[0]; self.parameter_explanations["ny"] = "Number of cells in the x dimension, currently %s"%self.parameters["ny"]
    self.nn = self.parameters["nx"] * self.parameters["ny"]
    self.parameters["xl"] = 10e3; self.parameter_explanations["xl"] = "x length dimension, currently %s metres"%self.parameters["xl"]
    self.parameters["yl"] = 10e3; self.parameter_explanations["yl"] = "y length dimension, currently %s metres"%self.parameters["yl"]
    self.parameters["dt"] = 1e3; self.parameter_explanations["dt"] = "Time step dt, currently %s years"%self.parameters["dt"]
    self.parameters["Kf"] = np.ones(self.nn)*1.e-6; self.parameter_explanations["Kf"] = "Erodibility field for the basement"
    self.parameters["Ks"] = 1e-3; self.parameter_explanations["Ks"] = "Erodibility for soft sediments, currently %s"%self.parameters["Ks"]
    self.parameters["Kd"] = np.ones(self.nn)*1.e-4; self.parameter_explanations["Kd"] = "Hillslope diffusivity field for the basement"
    self.parameters["Kds"] = 1e-2; self.parameter_explanations["Kds"] = "Hillslope diffusivity for soft sediments, currently %s"%self.parameters["Kds"]
    self.parameters["m"] = 1.11; self.parameter_explanations["m"] = "m exponent in the detachement/transport limited model, currently %s"%self.parameters["m"]
    self.parameters["n"] = 2.45; self.parameter_explanations["n"] = "n exponent in the detachement/transport limited model, currently %s"%self.parameters["n"]
    self.parameters["Gb"] = 0.5; self.parameter_explanations["Gb"] = "Sedimentation parameter G on the top of basement, currently %s"%self.parameters["Gb"]
    self.parameters["Gs"] = 0.8; self.parameter_explanations["Gs"] = "Sedimentation parameter G on the top of soft sediment, currently %s"%self.parameters["Gs"]
    self.parameters["p_mult"] = 10.; self.parameter_explanations["p_mult"] = "multi-flow distribution, currently %s"%self.parameters["p_mult"]
    self.parameters["uplift"] = np.zeros(self.nn); self.parameter_explanations["uplift"] = "Uplift field (including BC)"
    
    # self.arrays["H"] = np.random.random_sample((sel.parameters["nn"],)); self.array_explanations["H"] = "Array of elevation (flattened)"

    self.topography = initial_topo
    self.array_to_save = np.zeros((self.nn,))

    self.resolution = self.parameters["xl"]/self.parameters["nx"]

    # Initialising the code ...
    fs.fastscape_init()
    fs.fastscape_set_nx_ny(self.parameters["nx"], self.parameters["ny"])
    fs.fastscape_setup()
    fs.fastscape_init_h (self.topography.flatten())
    fs.fastscape_set_xl_yl(self.parameters["xl"],self.parameters["yl"])
    fs.fastscape_set_dt(self.parameters["dt"])
    fs.fastscape_set_erosional_parameters(self.parameters["Kf"], self.parameters["Ks"], self.parameters["m"], self.parameters["n"], self.parameters["Kd"], self.parameters["Kds"], self.parameters["Gb"], self.parameters["Gs"], self.parameters["p_mult"])
    fs.fastscape_set_u(self.parameters["uplift"])
    
    # Done

    # boundary conditions, so far fixed to periodic on side and fixed on top/bottom
    bc = 1010
    fs.fastscape_set_bc(bc)

    print("Model has been initialised with Default parameters. Please note that this is an experimental binding between LSDTopyTools and Fastscapelib-fortran. Both of these are based on many algorithm, make sure to cite the right ones if you use it!") if self.verbose>=2 else 0


  def update_model_parameters(self, **kwargs):

    # First checking the parameters 
    for key,val in kwargs.items():
      if(key not in kwargs):
        print("WARNING: I cannot recognise the following model parameter: %s"%key) if self.verbose>=1 else 0
        continue

      # Checking if it requires other modification
      if(key in ["nx","ny"]):
        print("Do not change the model nrow/ncols using  like that, use the function update_topography, I am ignoring that param") if self.verbose>=1 else 0
        continue

      if(key in ["Kf", "Kd", "uplift"]):
        if(val.ndim == 2):
          val = val.flatten()

      # updating the parameter
      self.parameters[key] = val

    fs.fastscape_set_xl_yl(self.parameters["xl"],self.parameters["yl"])
    fs.fastscape_set_dt(self.parameters["dt"])
    fs.fastscape_set_erosional_parameters(self.parameters["Kf"], self.parameters["Ks"], self.parameters["m"], self.parameters["n"], self.parameters["Kd"], self.parameters["Kds"], self.parameters["Gb"], self.parameters["Gs"], self.parameters["p_mult"])
    fs.fastscape_set_u(self.parameters["uplift"])
    self.resolution = self.parameters["xl"]/self.parameters["nx"]
    # Done


################################################################################################################
################################################################################################################
# Deprecated so far

  # def update_topography_from_array(self, array, xl = 20e3, yl = 20e3, I_know_what_I_am_doing = False):

  #   if(self.parameters["nx"]!=array.shape[1]):
  #     # Then I am updating the model
  #     old_nx = self.parameters["nx"]
  #     old_ny = self.parameters["ny"]
  #     self.parameters["nx"] = array.shape[1]
  #     self.parameters["ny"] = array.shape[0]

  #     self.nn = self.parameters["nx"] * self.parameters["ny"]
  #     if(I_know_what_I_am_doing == False):
  #       print("WARNING: You are changing the model resolution/dimension (number of cells). I am resampling spatial data (K,uplift,...) with a basic nearest neighbor approach. Hope this is fine, if not make sure to manually cahnge it later.") if self.verbose>=1 else 0
  #       self.parameters["Kf"] = self.parameters["Kf"].reshape(old_ny,old_nx)
  #       self.parameters["Kf"] = scipy.misc.imresize(self.parameters["Kf"],(self.parameters["ny"],self.parameters["nx"]))
  #       self.parameters["Kf"] = self.parameters["Kf"].flatten()
  #       self.parameters["Kd"] = self.parameters["Kd"].reshape(old_ny,old_nx)
  #       self.parameters["Kd"] = scipy.misc.imresize(self.parameters["Kd"],(self.parameters["ny"],self.parameters["nx"]))
  #       self.parameters["Kd"] = self.parameters["Kd"].flatten()
  #       self.parameters["uplift"] = self.parameters["uplift"].reshape(old_ny,old_nx)
  #       self.parameters["uplift"] = scipy.misc.imresize(self.parameters["uplift"],(self.parameters["ny"],self.parameters["nx"]))
  #       self.parameters["uplift"] = self.parameters["uplift"].flatten()
  #       fs.fastscape_set_erosional_parameters(self.parameters["Kf"], self.parameters["Kfs"], self.parameters["m"], self.parameters["n"], self.parameters["Kd"], self.parameters["Kds"], self.parameters["Gb"], self.parameters["Gs"], self.parameters["p_mult"])

  #   else:
  #     fs.fastscape_set_h(array.flatten())

  #   self.parameters["xl"] = xl
  #   self.parameters["yl"] = yl
  #   fs.fastscape_destroy()
  #   fs.fastscape_init()
  #   fs.fastscape_set_nx_ny(self.parameters["nx"], self.parameters["ny"])
  #   fs.fastscape_setup()
  #   fs.fastscape_init_h(array.flatten())

  #   fs.fastscape_set_xl_yl(self.parameters["xl"],self.parameters["yl"])
  #   self.resolution = self.parameters["xl"]/self.parameters["nx"]
  #   fs.fastscape_set_dt(self.parameters["dt"])
  #   fs.fastscape_set_erosional_parameters(self.parameters["Kf"], self.parameters["Ks"], self.parameters["m"], self.parameters["n"], self.parameters["Kd"], self.parameters["Kds"], self.parameters["Gb"], self.parameters["Gs"], self.parameters["p_mult"])
  #   fs.fastscape_set_u(self.parameters["uplift"])

# Deprecated so far
################################################################################################################
################################################################################################################




  def run_model(self, n_time_step = 100, save_step = 5, save_to_file = False, save_last_to_file = True):
    """
    """
    output = {}
    output["topography"] = []
    output["basement_elevation"] = []
    output["DA"] = []
    output["erosion_rate"] = []
    output["lake_depth"] = []

    output["time"] = []
    for i in range(1, n_time_step+1):
      print(i)

      print("Running time = %s ..." % (self.current_time + self.parameters["dt"]))
      self.current_time += self.parameters["dt"]
      fs.fastscape_execute_step()
      print("Done.") if self.verbose >= 3 else 0
      step = fs.fastscape_get_step()
      print("Done.") if self.verbose >= 3 else 0
      # Checking the stuff to export/save
      if(step % save_step == 0):
        print("SAving.") if self.verbose >= 3 else 0
        #### Saving Topography
        self.array_to_save = np.zeros((self.nn,))
        fs.fastscape_copy_h(self.array_to_save)
        print("Got the array.") if self.verbose >= 3 else 0
        self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
        if(save_to_file):
          name = str(self.current_time)
          while(len(name)<8):
            name = "0" + name
          rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
          output["topography"].append(self.prefix+"/%s_%s.tif"%(self.prefix, name))

        else:
          output["topography"].append(np.copy(self.array_to_save))

        ##### Now dealing with the basement elevation
        self.array_to_save = np.zeros((self.nn,))
        fs.fastscape_copy_basement(self.array_to_save)
        self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
        if(save_to_file):
          name = str(self.current_time)
          while(len(name)<8):
            name = "0" + name
          rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_basement_h_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
          output["basement_elevation"].append(self.prefix+"/%s_%s.tif"%(self.prefix, name))
        else:
          output["basement_elevation"].append(np.copy(self.array_to_save))

        ##### Now dealing with the DA
        self.array_to_save = np.zeros((self.nn,))
        fs.fastscape_copy_erosion_rate(self.array_to_save)
        self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
        if(save_to_file):
          name = str(self.current_time)
          while(len(name)<8):
            name = "0" + name
          rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_erosion_rate_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
          output["erosion_rate"].append(self.prefix+"/%s_%s.tif"%(self.prefix, name))
        else:
          output["erosion_rate"].append(np.copy(self.array_to_save))

        ##### Now dealing with the DA
        self.array_to_save = np.zeros((self.nn,))
        fs.fastscape_copy_drainage_area(self.array_to_save)
        self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
        if(save_to_file):
          name = str(self.current_time)
          while(len(name)<8):
            name = "0" + name
          rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_DA_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
          output["DA"].append(self.prefix+"/%s_%s.tif"%(self.prefix, name))
        else:
          output["DA"].append(np.copy(self.array_to_save))

        ##### Now dealing with the DA
        self.array_to_save = np.zeros((self.nn,))
        fs.fastscape_copy_lake_depth(self.array_to_save)
        self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
        if(save_to_file):
          name = str(self.current_time)
          while(len(name)<8):
            name = "0" + name
          rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_lake_depth_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
          output["lake_depth"].append(self.prefix+"/%s_%s.tif"%(self.prefix, name))
        else:
          output["lake_depth"].append(np.copy(self.array_to_save))



       
        ##### Now dealing with the DA
        output["time"].append(self.current_time)

    if(save_last_to_file):
      name = str(self.current_time)
      while(len(name)<8):
        name = "0" + name
      self.array_to_save = np.zeros((self.nn,))
      fs.fastscape_copy_h(self.array_to_save)
      self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
      rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_%s.tif"%(self.prefix, name), fmt = 'GTIFF')
      self.array_to_save = np.zeros((self.nn,))
      fs.fastscape_copy_basement(self.array_to_save)
      self.array_to_save = self.array_to_save.reshape((self.parameters["ny"],self.parameters["nx"]))
      rl.save_raster(self.array_to_save,0,self.parameters["xl"],0,self.parameters["yl"],self.parameters["xl"]/self.parameters["nx"],"EPSG:32601",self.prefix+"/%s_basement_h_%s.tif"%(self.prefix, name), fmt = 'GTIFF')



    return output









