import pandas as pd
import geopandas as gpd
import rasterio
import os 
import numpy as np
import glob
import matplotlib
from rasterio.plot import show
from scipy.spatial import cKDTree
from shapely.geometry import Point
SavetheTrees = ('C:/Users/sneez/OneDrive/Documents/lab4data/data/protected_areas.tif')
CityofLights = ('C:/Users/sneez/OneDrive/Documents/lab4data/data/urban_areas.tif')
HtwoO = ('C:/Users/sneez/OneDrive/Documents/lab4data/data/water_bodies.tif')
ItWimdy = ('C:/Users/sneez/OneDrive/Documents/lab4data/data/ws80m.tif')
SlipperySlope = ('C:/Users/sneez/OneDrive/Documents/lab4data/data/slope.tif')
Tree = rasterio.open(SavetheTrees)
Light = rasterio.open(CityofLights)
Water = rasterio.open(HtwoO)
Wind = rasterio.open(ItWimdy)
Slope = rasterio.open(SlipperySlope) 
treedata = Tree.read(1)
lightdata = Light.read(1)
waterdata = Water.read(1)
winddata = Wind.read(1)
slopedata = Slope.read(1)
lab4 = ('C:/Users/sneez/data/data/rasters')
def window_raster(data):
    temp_arr = np.zeros_like(data) 
    for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                    win = data[row : row + 11, col : col + 9]
                    temp_arr[row, col] = win.mean()
                    data = temp_arr
window_raster(winddata)
window_raster(treedata)
window_raster(slopedata)
window_raster(waterdata)
window_raster(lightdata)
SlipperyBool = np.where(slopedata < 15, 1, 0)
WetBool = np.where(waterdata < .02, 1, 0)
LightBool = np.where(lightdata ==1, 0, 1)
WindyBool = np.where(winddata > 8.5, 1, 0)
ProtectedBool = np.where(treedata < .05, 1, 0)
add1 = np.add(SlipperyBool, WetBool)
add2 = np.add(add1, LightBool)
add3 = np.add(add2, WindyBool)
whole_enchilada = np.add(add3, ProtectedBool)
unique, counts = np.unique(whole_enchilada, return_counts= 5)
print(np.asarray((unique,counts)))
with rasterio.open(r'C:\Users\sneez\data\data\finalrasters.tif', 'w',
                       driver='GTiff',
                       height=whole_enchilada.shape[0],
                       width=whole_enchilada.shape[1],
                       count=1,
                       dtype='float32',
                       crs=Slope.crs,
                       #transform=new_transform,
                       #nodata=slope.nodata,
    ) as five_raster:
        data = whole_enchilada.astype('int8')
        five_raster.write(data, indexes=1)
stations = (r'C:\Users\sneez\data\data\transmission_stations.txt')
pd.read_csv(stations)
booraster = rasterio.open(r'C:\Users\sneez\data\data\finalrasters.tif')
show(booraster)
extent = five_raster.bounds
cell_size = five_raster.transform[0]
x_ccords = np.arange(extent[0] + cell_size / 2, extent[2], cell_size)
y_ccords = np.arange(extent[1] + cell_size / 2, extent[3], cell_size)


money = np.where(whole_enchilada ==5)
print(money)
priceisright = pd.DataFrame(np.transpose(money))
stations = pd.read_csv(r'C:\Users\sneez\data\data\transmission_stations.txt')


from itertools import product

dat2array = np.array([np.array(x) for x in priceisright.values])
dat2array2 = np.array([np.array(x) for x in stations.values])

added_arr = np.array([x for x in product(dat2array, dat2array2)])

distance_array = np.linalg.norm(np.diff(added_arr, axis=1)[:,0,:],axis=1)
distance_array.shape
min(distance_array)
max(distance_array)