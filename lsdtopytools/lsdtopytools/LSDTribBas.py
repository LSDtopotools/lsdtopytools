"""
Controls the TribBas Model. Highly eperimental and evolves really rapidly So no doc so far sns.
No documentation so far, B.G. is still testing it.
"""
from lsdtt_xtensor_python import run_TribBas_to_steady_state, prebuild_TribBas, run_TribBas # Old stuff
from lsdtt_xtensor_python import LSDTribBas_cpp
import numba
import numpy as np
import pandas as pd
from lsdtopytools import raster_loader as rl, geoconvtools as gc
from lsdtopytools.lsdtopytools_utilities import save_to_database, load_from_database, load_metadata_from_database
from . import LSDDEM


class LSDTribBas(object):
    """
        The LSDTribBas class provides a high level interface to the LSDTribBas Model in LSDTopoTools.
        May disappear at some points but I need that for automation.
        Will develop if needed.
        Boris
    """

    def __init__(self, raster_name = "buzau.bil", sources_method = "threshold", threshold_contributing_pixels = 1000, source_csv = "buzau_Wsources.csv", output_name = "Clegg.h5", outlet_x=0, outlet_y =0, search_node_radius = 100, target_node = 20, MC_iterations = 50, skip = 1, sigma =2, m = 1.4, n = 3.11, A0 = 1, verbose = True, already_preprocessed = True):
        """
        Constructor

        """
        # First: Do you want the model to talk much?
        self.verbose = verbose

        #Loading raster
        print("Loading raster into the system") if self.verbose else 0
        A = rl.load_raster(raster_name)
        print("Raster loaded !") if self.verbose else 0
        if(~already_preprocessed):
            B = LSDDEM(file_name = raster_name)
            B.PreProcessing()
            A["array"] = B.cppdem.get_PP_raster()

        print("Loading my sources") if self.verbose else 0
        # I will need the sources
        ## Either I read it from a csv
        if("sources_method" == "csv"):
            sources = pd.read_csv(source_csv)
        else:
            # Or I recalculate it from threshold
            B.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = threshold_contributing_pixels)
            sources = B.cppdem.get_sources_full()
            for key,val in sources.items():
                print("key: %s shape: %s"%(key, val.shape))
            # sources["node"] = sources["nodeID"]
            sources = pd.DataFrame({"x": sources["X"], "y":sources["Y"],"node":sources["nodeID"].astype(np.int32)})
            # sources = sources.rename(columns={"nodeID": "node", "X": "x", "Y": "y"})
            print(sources)


        # Trimming my sources as they are extracted from a larger raster
        SOX = sources['x'][(sources['x'] > A['x_min']) & (sources['x']<A['x_max'])].values
        SOY = sources['y'][(sources['y'] > A['y_min']) & (sources['y']<A['y_max'])].values
        print("Sources ready to be ingested") if self.verbose else 0

        print(SOX)


        # Prebuilding the model
        self.n = n
        self.m = m
        self.A0 = A0

        # HD5 file
        self.hd5_name = output_name
        self.run_default_ID = 0

        print("Now prebuilding your model: \n-> getting all the flow informations\n-> ingesting the sources to extract a river network\n-> Calculate initial ksn using Mudd et al., 2014 (JGR)\n-> Order and format my nodes\nIt can take a bit of time...") if self.verbose else 0
        self.TBcpp = LSDTribBas_cpp(A["nrows"], A["ncols"], A["x_min"], A["y_min"], A["res"], A["nodata"][0], A["array"],SOX, SOY, m, n, A0, outlet_x, outlet_y, search_node_radius, target_node, MC_iterations, skip, 10, sigma)
        print("Model constructed!") if self.verbose else 0
        self.n_node = self.TBcpp.m_chi().shape[0]

        self.base_param = [A["nrows"], A["ncols"], A["x_min"], A["y_min"], A["res"], A["nodata"][0], A["array"],SOX, SOY, m, n, A0, outlet_x, outlet_y, search_node_radius, target_node, MC_iterations, skip, 10, sigma]

        print("Saving the initial state to hd5...")
        storer = pd.HDFStore(self.hd5_name)
        storer.put("initial",pd.DataFrame({"flow_distance": self.TBcpp.flow_distance(), "elevation": self.TBcpp.elevation()}),format = "t")
        storer.close()

        # self.K_to_SS = {}

    def run_model(self,K_field = 5e-8, uplift_field = 0.001, tolerance_p = 0.95, tolerance_delta = 0.5, timestep = 1000, final_tol = 0.95, save_dir = "./", n_timestep = "auto", save_step = 10, min_n_timestep = 25 
        , max_n_timestep = 10000, run_save_ID = None):
        """
            Just run the model
            K and U fields can be scalar or ndarray
        """

        # getting the uplift and K to the right format
        if(~isinstance(uplift_field,np.ndarray)):
            uplift_field = np.zeros(self.n_node) + uplift_field
        if(~isinstance(K_field,np.ndarray)):
            K_field = np.zeros(self.n_node) + K_field

        if(isinstance(n_timestep,str)):
            if(n_timestep == "auto"):
                results = self.TBcpp.run_model(timestep, save_step,tolerance_p, tolerance_delta, min_n_timestep , uplift_field, K_field, max_n_timestep)
        else:
            results = self.TBcpp.run_model(timestep, save_step,tolerance_p, tolerance_delta, n_timestep, uplift_field, K_field, n_timestep)

        if(run_save_ID is None):
            outID = "run_" + str(self.run_default_ID + 1)
            print("Defaulting the name to %s" %(outID))
        else:
            outID = str(run_save_ID)

        storer = pd.HDFStore(self.hd5_name)
        storer.put(outID,pd.DataFrame(results), format="t")
        storer.close()


    def m_and_ns(self, min_n = 1, min_m = 1, step = 0.1, nsteps = 40, min_K = 1e-9, range_of_K = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3], uplift_field = 0.001, tolerance_p = 0.95, tolerance_delta = 0.5, timestep = 1000, 
        final_tol = 0.95, save_dir = "./", min_n_timestep = 25 , max_n_timestep = 10000, run_save_ID = None, interval_of_test = 1000):
        """
        try stuff on best fit m and n
        """    
        from lsdtt_xtensor_python import get_median_profile

        print("Initialising the model, it will take a while as I am calculating m_chi for all m/n combination")
        # initialising the ranges
        range_m = []
        range_n = []
        m_chi_in_theta = {} # Dictionnary to avoid duplicity of data
        chi_in_theta = {}
        compmet = {}

        this_U = uplift_field
        unique_ID_m_and_ns = 0
        # Output managers
        analysis_names = []# 

        # Getting the ranges of m n and m/n
        for i in range(nsteps):
            range_m.append(min_m + step * i)
            range_n.append(min_n + step * i)
        

        for tn in range_n:
            for tm in range_m:
                if(tm<tn):
                    ratio = tm/tn
                    ratio = str(ratio)[:4]
                    if(~(ratio in m_chi_in_theta)):
                        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",ratio, tm, tn)
                        temp_OB = LSDTribBas_cpp(self.base_param[0], self.base_param[1], self.base_param[2], self.base_param[3], self.base_param[4], self.base_param[5],self.base_param[6],self.base_param[7], self.base_param[8], range_m[-1] , range_n[-1], self.base_param[11], self.base_param[12], self.base_param[13], self.base_param[14], self.base_param[15], self.base_param[16], self.base_param[17], self.base_param[18], self.base_param[19])
                        m_chi_in_theta[ratio] = temp_OB.m_chi()
                        chi_in_theta[ratio] = temp_OB.chi()

        # for key,val in m_chi_in_theta.items():
        #     print(key)

        print("Model initialised!")

        save_to_database(self.hd5_name,"reference_mchi", pd.DataFrame(m_chi_in_theta))


        print("Ok Let's roll")
        this_K = min_K
        for this_K in (range_of_K):
            # getting the uplift and K to the right format
            if(~isinstance(uplift_field,np.ndarray)):
                uplift_field = np.zeros(self.n_node) + uplift_field

            K_field = np.zeros(self.n_node) + this_K

            for m in range_m:
                for n in range_n:
                    if(m<n): # Checking that m/n < 1
                        print("Calculating m: %s and n: %s" %(m,n))
                        self.TBcpp.set_m_and_n(m,n)
                        # Running the model to steady state
                        results = pd.DataFrame(self.TBcpp.run_model(timestep, 100000,tolerance_p, tolerance_delta, min_n_timestep , uplift_field, K_field, max_n_timestep))
                        maxi = 0
                        for col in results.columns :
                            if( float(col) > maxi):
                                maxi = col
                        # Getting the results
                        this_elev_array = results[maxi].values
                        temp_df = pd.DataFrame({"flow_distance": self.TBcpp.flow_distance(), "initial_elevation": self.TBcpp.elevation(), "SS_elevation": this_elev_array})
                        temp_df = temp_df.sort_values("flow_distance")

                        # Getting the median values
                        compa = get_median_profile(temp_df["flow_distance"].values, temp_df["SS_elevation"].values - temp_df["initial_elevation"].values,interval_of_test, 1)
                        # saving the values
                        kalak = str(m) + "_" + str(n)
                        compmet[kalak + "_elev"] = compa["Y"]

                        # dealing with median m_chi
                        this_ratio = str(m/n)[:4]
                        this_mchi = self.TBcpp.first_order_m_chi_from_custarray(chi_in_theta[this_ratio],this_elev_array)
                        compa_2 = get_median_profile(temp_df["flow_distance"].values, m_chi_in_theta[this_ratio] - this_mchi, interval_of_test, 1)
                        compmet[kalak + "_mchi"] = compa_2["Y"]



            compmet["X"] = compa["X"] #  for plotting. Because flow distance is always the same, X is as well
            df = pd.DataFrame(compmet)
            # temp for testing

            # df.to_csv("test_of_mn.csv", index = False)
            analysis_names.append("mn_test_" + str(unique_ID_m_and_ns))
            save_to_database(self.hd5_name,"mn_test_" + str(unique_ID_m_and_ns), df, metadata = {"uplift_field": this_U, "K_field": this_K})
            # storer = pd.HDFStore(self.hd5_name)
            # storer.put("U%s_K%s"%(this_U,this_K),df, format="t")
            # storer.close()
            unique_ID_m_and_ns += 1
            # this_K = 10 ** (np.log10(this_K) + 0.1)
        # Saving lists of m,n,k and U that have been effectively tested
        save_to_database(self.hd5_name,"mn_test_metadata", pd.DataFrame({"test": [0,0,0,1,2]}), metadata = {"analysis_names": analysis_names})



    def TODO(self):
        print("""
            TODO:
            - Adapt the run_model's output to get the stat of fit for each timestep
            - implement m_chi calculation and comparison
            - Simplification/multithreading
            - Dorce K or U to behave by segments
            """)





class Analytic_TribBas(object):
    """
        This class build and provides analytic solutions for solving the lithology vs uplift problem. It uses the equation of k_sn calculated with M_chi.
    """
    
    def __init__(self, m = 1.08, n=3.11, min_K = -10, max_K = -4, min_U = 0.0001, max_U = 0.005, step_K = 0.01 , step_U = 0.00001, K_in_log_space = True):


        super(Analytic_TribBas, self).__init__()
        self.m = m
        self.n = n
        self.K_in_log_space = True        

        ######################## Generate 3D function
        # Need here to get the ranges of possibilities for U and K
        self.range_K = []
        tKsat = min_K
        if(K_in_log_space):
            while(tKsat <= max_K):
                # tKsat += 0.01 * 10**(math.floor(math.log10(tKsat))) not in log space
                self.range_K.append(tKsat)
                tKsat += step_K

        else:
            print("TODO: not in log space (you probably don't want to do that though)")
            quit()


        self.range_uplift = []
        step_U = min_U
        while(step_U <= max_U):
            self.range_uplift.append(step_U)
            step_U += 0.00001


        self.range_K = np.array(self.range_K)
        self.range_uplift = np.array(self.range_uplift)



    def get_U_from_mchi_and_K(self, this_K, this_m_chi, logged = False):
        if(logged):
            this_K = 10**this_K
        return this_K * np.exp(self.n * np.log(this_m_chi))

    def get_K_from_mchi_and_U(self, this_uplift, this_m_chi):
        return this_uplift / np.exp(self.n * np.log(this_m_chi))


    def generate_2D_state_of_equilibrium_from_U_and_m_chi(self, range_of_U, range_of_mchi):
        return get_K_from_mchi_and_U(range_of_U[:,np.newaxis],range_of_mchi[np.newaxis,:])









































































































































































































################## OLD ROUTINES I USED FOR CHRISTMAS. BASIC AND BRUTFORCE

    # def hunt_my_K(self,initial_k = 5e-5, reference_k_step = 1e-5, uplift_field = 0.001, min_iteration_per_run = 20, tolerance_p = 0.95, tolerance_delta = 0.5, timestep = 1000, final_tol = 0.95,  final_tol_delta = 0.5, save_dir = "./", max_iterations_per_test= 500, max_global_iterations = 100 ):
    #     """
    #         Iterative method to force K refining
    #     """
    #     # We force an uplift field to remain constant
    #     U = np.zeros(self.prebuilt["elevation"].shape[0]) + uplift_field
    #     # Initition of erodibility
    #     K = np.zeros(self.prebuilt["elevation"].shape[0]) + initial_k
    #     # Initialization of adaptative erodibility step
    #     refK = np.zeros(self.prebuilt["elevation"].shape[0]) + reference_k_step
    #     # sign of K last time
    #     signk = np.zeros(self.prebuilt["elevation"].shape[0])
    #     last_sign_k = np.zeros(self.prebuilt["elevation"].shape[0])
    #     how_many_time_unchanged = np.zeros(self.prebuilt["elevation"].shape[0])


    #     run = {}

    #     catched = False

    #     cpt = 0
    #     this_prebuilt = self.prebuilt.copy()

    #     bdf = pd.DataFrame(data = self.prebuilt)
    #     bdf.to_csv(save_dir+"base.csv", index = False)



    #     print("Chasing my K!")
    #     while(catched == False and cpt<max_global_iterations): # safety to 100
    #         cpt+=1 # ᶘ °㉨°ᶅ
    #         print("MODEL RUN %s"%(cpt), end="\r")

    #         # Run little K
    #         results = run_TribBas_to_steady_state(self.m,self.n, timestep, 10000, tolerance_p, tolerance_delta, 20, U, K, this_prebuilt, max_iterations_per_test)

    #         # print("FOUND STEADY-STATE, REFINING MY K")

    #         # get the latest result
    #         testk = 0
    #         maxk = 0
    #         for key,val in results.items():
    #             testk = int(key) # string to int
    #             print(key)
    #             if(testk>maxk):
    #                 maxk = testk
    #         maxk = str(int(maxk)) # getting the right format
    #         maxk = maxk.split(".")[0] # Making sure no decimal are there
    #         print("DEBUG::maxk:", maxk)
    #         new_elev = results[maxk]
    #         base_elev = np.copy(self.prebuilt["elevation"])
    #         print("DEBUG::mean_Baseelev:", np.mean(base_elev))

    #         n_tol = new_elev[np.abs(new_elev-base_elev)<final_tol_delta].shape[0]
    #         n_tot = new_elev.shape[0]
    #         if(n_tol/n_tot>final_tol):
    #             catched = True
    #             print("I THINK I FOUND MY K !!!")
    #         else:
    #             print("REFINING MY K ...")
    #             signk[new_elev<base_elev] = -1
    #             signk[new_elev>base_elev] = 1
    #             signk[np.abs(new_elev-base_elev)<final_tol_delta] = 0

    #             if(cpt==1): # first run is 1 (incrementation at the beginning of the loop, see ᶘ °㉨°ᶅ)
    #                 last_sign_k = np.copy(signk)

    #             K, refK, how_many_time_unchanged = self.adjust_my_K_wild(base_elev,new_elev,refK,signk,last_sign_k,final_tol,K,how_many_time_unchanged)

    #             print("I ADAPTED %s/%s K."%(signk[signk!=0].shape[0],signk.shape[0]))

    #         df = pd.DataFrame(data = {"K":K, "elevation": new_elev})

    #         name = str(cpt)
    #         while(len(name)<4):
    #             name = "0" +name

    #         df.to_csv(save_dir + "attempt_%s_K.csv"%(name))
    #         this_prebuilt["elevation"] = new_elev
    #         last_sign_k = np.copy(signk)
    #     return K



    # # Temp function, will move to cpp
    # @numba.jit()
    # def adjust_my_K_wild(self, base_elev,new_elev,refK,signk,last_sign_k,final_tol,K,how_many_time_unchanged):

    #     for i in range(base_elev.shape[0]):
    #         # Need adjustment?
    #         if(abs(new_elev[i]-base_elev[i])>final_tol):
    #             if(signk[i] != last_sign_k[i]):
    #                 refK[i] = refK[i]/2
    #                 how_many_time_unchanged[i] = 0
    #             else:
    #                 how_many_time_unchanged[i] = how_many_time_unchanged[i]+1
    #                 if(how_many_time_unchanged[i]>10):
    #                     refK[i] = refK[i]*1.5
    #                     how_many_time_unchanged[i] = 0



    #             last_K = K[i]
    #             K[i] += signk[i]*refK[i]
    #             if(K[i]<=0):
    #                 K[i] = refK[i]/2 # HACKY WAY HERE
    #                 refK[i] = refK[i]/2 # HACKY WAY HERE


    #     return K, refK,how_many_time_unchanged

    # def hunt_my_U(self,initial_uplift = 0.001, reference_uplift_step = 0.0001, min_iteration_per_run = 20 , K_field = 0.001, tolerance_p = 0.95, tolerance_delta = 0.5, timestep = 1000, final_tol = 0.95,  final_tol_delta = 0.5, save_dir = "./", max_iterations_per_test= 500, max_global_iterations = 100 ):
    #     """
    #         Iterative method to force K refining
    #     """
    #     # We force an uplift field to remain constant
    #     U = np.zeros(self.prebuilt["elevation"].shape[0]) + initial_uplift
    #     # Initition of erodibility
    #     K = np.zeros(self.prebuilt["elevation"].shape[0]) + K_field
    #     # Initialization of adaptative erodibility step
    #     refU = np.zeros(self.prebuilt["elevation"].shape[0]) + reference_uplift_step
    #     # sign of K last time
    #     signU = np.zeros(self.prebuilt["elevation"].shape[0])
    #     last_sign_k = np.zeros(self.prebuilt["elevation"].shape[0])
    #     how_many_time_unchanged = np.zeros(self.prebuilt["elevation"].shape[0])


    #     run = {}

    #     catched = False

    #     cpt = 0
    #     this_prebuilt = self.prebuilt.copy()

    #     bdf = pd.DataFrame(data = self.prebuilt)
    #     bdf.to_csv(save_dir+"base.csv", index = False)



    #     print("Chasing my theoretical Uplift field!")
    #     while(catched == False and cpt<max_global_iterations): # safety to 100
    #         cpt+=1 # ᶘ °㉨°ᶅ
    #         print("MODEL RUN %s"%(cpt))

    #         # Run little K
    #         results = run_TribBas_to_steady_state(self.m,self.n, timestep, 10000, tolerance_p, tolerance_delta, min_iteration_per_run, U, K, this_prebuilt, max_iterations_per_test)

    #         print("FOUND STEADY-STATE, REFINING MY K")

    #         # get the latest result
    #         testU = 0
    #         maxU = 0
    #         for key,val in results.items():
    #             testU = int(key) # string to int
    #             print(key)
    #             if(testU>maxU):
    #                 maxU = testU
    #         maxU = str(int(maxU)) # getting the right format
    #         maxU = maxU.split(".")[0] # Making sure no decimal are there
    #         # print("DEBUG::maxU:", maxU)
    #         new_elev = results[maxU]
    #         base_elev = np.copy(self.prebuilt["elevation"])
    #         # print("DEBUG::mean_Baseelev:", np.mean(base_elev))

    #         n_tol = new_elev[np.abs(new_elev-base_elev)<final_tol_delta].shape[0]
    #         n_tot = new_elev.shape[0]
    #         if(n_tol/n_tot>final_tol):
    #             catched = True
    #             print("I THINK I FOUND MY U :)")
    #         else:
    #             print("REFINING MY U :)")
    #             signU[new_elev<base_elev] = 1
    #             signU[new_elev>base_elev] = -1
    #             signU[np.abs(new_elev-base_elev)<final_tol_delta] = 0

    #             if(cpt==1): # first run is 1 (incrementation at the beginning of the loop, see ᶘ °㉨°ᶅ)
    #                 last_sign_U = np.copy(signU)

    #             U, refU, how_many_time_unchanged = self.adjust_my_U_wild(base_elev,new_elev,refU,signU,last_sign_U,final_tol,U,how_many_time_unchanged)

    #             print("I ADAPTED %s/%s U."%(signU[signU!=0].shape[0],signU.shape[0]))

    #         df = pd.DataFrame(data = {"U":U, "elevation": new_elev})

    #         name = str(cpt)
    #         while(len(name)<4):
    #             name = "0" +name

    #         df.to_csv(save_dir + "attempt_%s_U.csv"%(name))
    #         this_prebuilt["elevation"] = new_elev
    #         last_sign_U = np.copy(signU)

    #     return U



    # # Temp function, will move to cpp
    # @numba.jit()
    # def adjust_my_U_wild(self, base_elev,new_elev,refU,signU,last_sign_U,final_tol,U,how_many_time_unchanged):

    #     for i in range(base_elev.shape[0]):
    #         # Need adjustment?
    #         if(abs(new_elev[i]-base_elev[i])>final_tol):
    #             if(signU[i] != last_sign_U[i]):
    #                 refU[i] = refU[i]/2
    #                 how_many_time_unchanged[i] = 0
    #             else:
    #                 how_many_time_unchanged[i] = how_many_time_unchanged[i]+1
    #                 if(how_many_time_unchanged[i]>10):
    #                     refU[i] = refU[i]*1.5
    #                     how_many_time_unchanged[i] = 0



    #             last_U = U[i]
    #             U[i] += signU[i]*refU[i]
    #             if(U[i]<=0):
    #                 U[i] = refU[i]/2 # HACUY WAY HERE
    #                 refU[i] = refU[i]/2 # HACUY WAY HERE


    #     return U, refU,how_many_time_unchanged












