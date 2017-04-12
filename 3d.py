# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab


import numpy as np
from osgeo import gdal, osr
#import OsmApi
import math
import os




# filename = "elevationdata/srtm_39_04/srtm_39_04.tif"

startingpoint = (44.6478300, 10.9253900)
track_lat = np.arange(startingpoint[0], startingpoint[0]+0.01, 0.00001).transpose()
track_lon = np.tile(startingpoint[1], (len(track_lat)))
margin = 1000






def SRTMTile(lat, lon):
    xtile = int(np.round((lon - (-180)) / (360/72) + 1))
    ytile = int(np.round((60 - lat) / (360/72) + 1))
    name = "srtm_{:02d}_{:02d}".format(xtile, ytile)
    return name

earthRadius = 6371000 # Earth radius in meters (yes, it's an approximation) https://en.wikipedia.org/wiki/Earth_radius

# Determine central track coordinates and area width
mylocation = ((np.max(track_lat) + np.min(track_lat))/2, (np.max(track_lon) + np.min(track_lon))/2)
span_deg = np.max([np.max(track_lat)-np.min(track_lat), np.max(track_lon)-np.min(track_lon)])

# Choosing the right tile
tile = SRTMTile(mylocation[0], mylocation[1])
filename = "elevationdata/{}/{}.tif".format(tile, tile)
print filename

if not os.path.isfile(filename):
    print "Elevation profile for this location ({}) has not been downloaded.".format(tile)

# Read SRTM GeoTiff elevation file 
ds = gdal.Open(filename)
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
tile_lon_min = gt[0]
tile_lon_max = gt[0] + width*gt[1] + height*gt[2]
tile_lat_min = gt[3] + width*gt[4] + height*gt[5] 
tile_lat_max = gt[3]
tile_ele = ds.GetRasterBand(1)
span_px = int(np.round(span_deg/gt[1] + margin))

# Check if the location specified is in this tile
if (mylocation[1] < tile_lon_min) or (mylocation[1] > tile_lon_max) or (mylocation[0] < tile_lat_min) or (mylocation[0] > tile_lat_max):
    print "Error: the selected location is not covered by the current map"
mylocation_px = ((mylocation[0]-tile_lat_max)/gt[5], (mylocation[1]-tile_lon_min)/gt[1])

zone_x_min_tmp = np.round(mylocation_px[1] - span_px * 0.5)
zone_x_size = span_px
zone_y_min_tmp = np.round(mylocation_px[0] - span_px * 0.5)
zone_y_size = span_px

# Trim boundaries
zone_x_min = zone_x_min_tmp if zone_x_min_tmp>=0 else 0
zone_y_min = zone_y_min_tmp if zone_y_min_tmp>=0 else 0
zone_x_size = zone_x_size if (zone_x_min_tmp + zone_x_size)<width else width
zone_y_size = zone_y_size if (zone_y_min_tmp + zone_y_size)<height else height

# Actual elevation data
zone_ele = tile_ele.ReadAsArray(zone_x_min, zone_y_min, zone_x_size, zone_y_size).astype(np.float)

def degrees2metersLongX(latitude, longitudeSpan):
  """ latitude (in degrees) is used to convert a longitude angle to a distance in meters """
  return 2.0*math.pi*earthRadius*math.cos(math.radians(latitude))*longitudeSpan/360.0

def degrees2metersLatY(latitudeSpan):
  """ Convert a latitude angle span to a distance in meters """
  return 2.0*math.pi*earthRadius*latitudeSpan/360.0

def degrees2meters(longitude, latitude):
  return (degrees2metersLongX(latitude, longitude), degrees2metersLatY(latitude))

# Create X,Y coordinates for zone_ele array (contains Z in meters)
line_x_deg = np.arange(tile_lon_min+zone_x_min*gt[1], tile_lon_min+(zone_x_min+zone_x_size)*gt[1], gt[1])[0:zone_x_size]
array_x_deg = np.tile(line_x_deg, (len(zone_ele), 1)).transpose()

line_y_deg = np.arange(tile_lat_max+zone_y_min*gt[5], tile_lat_max+(zone_y_min+zone_y_size)*gt[5], gt[5])[0:zone_y_size]
line_y_m = np.array([degrees2metersLatY(j) for j in line_y_deg])
array_y_m = np.tile(line_y_m, (len(zone_ele[0]), 1))

array_x_m = np.empty_like(array_x_deg)
for x, y in np.ndindex(array_x_deg.shape):
  array_x_m[x,y] = degrees2metersLongX(line_y_deg[y], array_x_deg[x,y])

zscale = 1

# Display 3D surface
mlab.mesh(array_x_m, array_y_m, zone_ele.transpose() * zscale)

# Hiking path
track_x_m = list()
track_y_m = list()
track_z_m = list()
for i in range(np.size(track_lat, axis=0)):
  (x,y) = degrees2meters(track_lon[i], track_lat[i])
  track_x_m.append(x)
  track_y_m.append(y)
  zz = zone_ele.transpose()[int(round((track_lon[i] - (tile_lon_min+zone_x_min*gt[1])) / gt[1])), int(round((track_lat[i] - (tile_lat_max+zone_y_min*gt[5])) / gt[5]))]
  track_z_m.append(zz+30.0) # We display the path 30m over the surface for it to be visible

# Display path nodes as spheres
mlab.points3d(track_x_m, track_y_m, track_z_m, color=(1,0,0), mode='sphere', scale_factor=100)
# Displaying the line does not work as nodes are not listed in the correct order
# mlab.plot3d(track_x_m,track_y_m,track_z_m, color=(1,1,0), line_width=15.0) # representation='surface' 'wireframe' 'points'

# Show the 3D map
mlab.show()
