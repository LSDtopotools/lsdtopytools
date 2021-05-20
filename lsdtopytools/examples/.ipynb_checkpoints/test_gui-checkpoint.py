from lsdtktools import Window
from tkinter import *
import numpy as np
from matplotlib import cm, pyplot as plt
from lsdtopytools import LSDDEM
from skimage.transform import resize as imresize 
# from pillow import image

my_raster_path = "./" # I am telling saving my path to a variable. In this case, I assume the rasters is in the same folder than my script
file_name = "WAWater.bil" # The name of your raster with extension. RasterIO then takes care internally of decoding it. Good guy rasterio!

mydem = LSDDEM(path = my_raster_path, file_name = file_name, already_preprocessed = True)

root = Tk()
my_window = Window(root)


print("lasjdflajsdlf")

myarray = imresize(mydem.cppdem.get_PP_raster(), (300,400))

print(np.uint8(cm.gist_earth(myarray)*255))

my_window.plot_array(myarray)

root.mainloop()

