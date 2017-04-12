# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab
import numpy
from osgeo import gdal, osr
#import OsmApi
import math
import os

earthRadius = 6371000 # Earth radius in meters (yes, it's an approximation) https://en.wikipedia.org/wiki/Earth_radius



filename = "C:\Users\salatis\Downloads\sport-ste-edits3\elevationdata\srtm_39_04\srtm_39_04.tif"
mylocation = (44.6478300, 10.9253900)
myspan = 500
track_lat = numpy.arange(mylocation[0], mylocation[0]+0.01, 0.00001).transpose()
track_lon = numpy.tile(mylocation[1], (len(track_lat)))
track = numpy.vstack((track_lat, track_lon)).transpose()



def SRTMTile(lat, lon):
    xtile = int(numpy.round((lon - (-180)) / (360/72) + 1))
    ytile = int(numpy.round((60 - lat) / (360/72) + 1))
    name = "srtm_{:02d}_{:02d}".format(xtile, ytile)
    return name

# Choosing the right tile
tile = SRTMTile(mylocation[0], mylocation[1])
filename = "elevationdata\{}\{}.tif".format(tile, tile)

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

# Check if the location specified is in this tile
if (mylocation[1] < tile_lon_min) or (mylocation[1] > tile_lon_max) or (mylocation[0] < tile_lat_min) or (mylocation[0] > tile_lat_max):
    print "Error: the selected location is not covered by the current map"
mylocation_px = ((mylocation[0]-tile_lat_max)/gt[5], (mylocation[1]-tile_lon_min)/gt[1])

# Zone to display, trimming boundaries
zone_x_min_tmp = numpy.round(mylocation_px[1] - myspan * 0.5)
zone_x_size = myspan
zone_y_min_tmp = numpy.round(mylocation_px[0] - myspan * 0.5)
zone_y_size = myspan
zone_x_min = zone_x_min_tmp if zone_x_min_tmp>=0 else 0
zone_y_min = zone_y_min_tmp if zone_y_min_tmp>=0 else 0
zone_x_size = zone_x_size if (zone_x_min_tmp + zone_x_size)<width else width
zone_y_size = zone_y_size if (zone_y_min_tmp + zone_y_size)<height else height

# Actual elevation data
zone_ele = tile_ele.ReadAsArray(zone_x_min, zone_y_min, zone_x_size, zone_y_size).astype(numpy.float)

def degrees2metersLongX(latitude, longitudeSpan):
  """ latitude (in degrees) is used to convert a longitude angle to a distance in meters """
  return 2.0*math.pi*earthRadius*math.cos(math.radians(latitude))*longitudeSpan/360.0

def degrees2metersLatY(latitudeSpan):
  """ Convert a latitude angle span to a distance in meters """
  return 2.0*math.pi*earthRadius*latitudeSpan/360.0

def degrees2meters(longitude, latitude):
  return (degrees2metersLongX(latitude, longitude), degrees2metersLatY(latitude))

# Create X,Y coordinates for zone_ele array (contains Z in meters)
line_x_deg = numpy.arange(tile_lon_min+zone_x_min*gt[1], tile_lon_min+(zone_x_min+zone_x_size)*gt[1], gt[1])[0:zone_x_size]
array_x_deg = numpy.tile(line_x_deg, (len(zone_ele), 1)).transpose()

line_y_deg = numpy.arange(tile_lat_max+zone_y_min*gt[5], tile_lat_max+(zone_y_min+zone_y_size)*gt[5], gt[5])[0:zone_y_size]
line_y_m = numpy.array([degrees2metersLatY(j) for j in line_y_deg])
array_y_m = numpy.tile(line_y_m, (len(zone_ele[0]), 1))

array_x_m = numpy.empty_like(array_x_deg)
for x, y in numpy.ndindex(array_x_deg.shape):
  array_x_m[x,y] = degrees2metersLongX(line_y_deg[y], array_x_deg[x,y])

zscale = 1

# Display 3D surface
mlab.mesh(array_x_m, array_y_m, zone_ele.transpose() * zscale)

# Hiking path
track_x_m = list()
track_y_m = list()
track_z_m = list()
for i in range(numpy.size(track, axis=0)):
  (x,y) = degrees2meters(track_lon[i], track_lat[i])
  track_x_m.append(x)
  track_y_m.append(y)
  zz = zone_ele.transpose()[int(round((track_lon[i] - (tile_lon_min+zone_x_min*gt[1])) / gt[1])), int(round((track_lat[i] - (tile_lat_max+zone_y_min*gt[5])) / gt[5]))]
  track_z_m.append(zz+30.0) # We display the path 30m over the surface for it to be visible

# Display path nodes as spheres
mlab.points3d(track_x_m, track_y_m, track_z_m, color=(1,0,0), mode='sphere', scale_factor=100)
# Displaying the line does not work as nodes are not listed in the correct order
# mlab.plot3d(track_x_m,track_y_m,track_z_m, color=(1,1,0), line_width=15.0) # representation='surface' 'wireframe' 'points'

# Display a black dot on Grenoble's coordinates
mylocation_m = degrees2meters(mylocation[1], mylocation[0])
mlab.points3d([mylocation_m[0]], [mylocation_m[1]], [200], color=(0,0,0), mode='sphere', scale_factor=500)

# Show the 3D map
mlab.show()