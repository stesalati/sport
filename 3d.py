# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab


import numpy as np
from osgeo import gdal, osr
#import OsmApi
import math
import os
import sys
import gdal_merge as gm

sys.path.append('C:/Users/salatis/AppData/Local/Continuum/Anaconda2/Scripts')





startingpoint = (44.6478300, 10.9253900)
track_lat = np.arange(startingpoint[0], startingpoint[0]+0.01, 0.00001).transpose()
track_lon = np.tile(startingpoint[1], (len(track_lat)))
margin = 400









"""
ACTUAL FUNCTION UNDER TEST
"""

def SRTMTile(lat, lon):
    xtile = int(np.trunc((lon - (-180)) / (360/72) + 1))
    ytile = int(np.trunc((60 - lat) / (360/72) + 1))
    return (xtile, ytile)

def degrees2metersLongX(latitude, longitudeSpan):
  # latitude (in degrees) is used to convert a longitude angle to a distance in meters
  return 2.0*math.pi*earthRadius*math.cos(math.radians(latitude))*longitudeSpan/360.0

def degrees2metersLatY(latitudeSpan):
  # Convert a latitude angle span to a distance in meters
  return 2.0*math.pi*earthRadius*latitudeSpan/360.0

def degrees2meters(longitude, latitude):
  return (degrees2metersLongX(latitude, longitude), degrees2metersLatY(latitude))

earthRadius = 6371000 # Earth radius in meters (yes, it's an approximation) https://en.wikipedia.org/wiki/Earth_radius
px2deg = 0.0008333333333333334

# Determine the coordinates of the area we are interested in
center = ((np.max(track_lat) + np.min(track_lat))/2, (np.max(track_lon) + np.min(track_lon))/2)
span_deg = np.max([np.max(track_lat)-np.min(track_lat), np.max(track_lon)-np.min(track_lon)])
lat_min = np.min(track_lat) - margin * px2deg
lat_max = np.max(track_lat) + margin * px2deg
lon_min = np.min(track_lon) - margin * px2deg
lon_max = np.max(track_lon) + margin * px2deg

# Determine which tiles are necessary
tile_corner_min = SRTMTile(lat_min, lon_min)
tile_corner_max = SRTMTile(lat_max, lon_max)
tiles_lat = range(tile_corner_min[0], tile_corner_max[0]+1)
tiles_lon = range(tile_corner_min[1], tile_corner_max[1]+1)

# For debug
#tiles_lat = range(38, 39+1)
#tiles_lon = range(3, 3+1)

print "\nHorizontal tiles: {}".format(tiles_lat)
print "Vertical tiles: {}\n".format(tiles_lon)

if len(tiles_lat) > 1 or len(tiles_lon) > 1:
    # More than one tile is needed: generate tile names and merge them
    gdal_merge_command_list = ['', '-o', 'elevationdata/tmp.tif']
    for tile_lat in tiles_lat:
        for tile_lon in tiles_lon:
            # Generate tile filename and append it to the list
            tilename = "srtm_{:02d}_{:02d}".format(tile_lat, tile_lon)
            filename = "elevationdata/{}/{}.tif".format(tilename, tilename)
            gdal_merge_command_list.append(filename)
            if not os.path.isfile(filename):
                print "Elevation profile for this location ({}) has not been downloaded.".format(tilename)
    # Merge
    print gdal_merge_command_list
    gm.main(gdal_merge_command_list)
    filename = "elevationdata/{}/{}.tif".format(tilename, tilename)
else:
    tilename = "srtm_{:02d}_{:02d}".format(tiles_lat[0], tiles_lon[0])
    filename = "elevationdata/{}/{}.tif".format(tilename, tilename)
    if not os.path.isfile(filename):
        print "Elevation profile for this location ({}) has not been downloaded.".format(tilename)

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

print "Latitude:  {} <-- {} -- {} --> {}".format(tile_lat_min, lat_min, lat_max, tile_lat_max)
print "Longitude: {} <-- {} -- {} --> {}\n".format(tile_lon_min, lon_min, lon_max, tile_lon_max)

# Selecting only a zone of the whole map, the one we're interested in plotting
zone_x_min = int((lon_min - tile_lon_min)/gt[1])
zone_x_size = int((lon_max - lon_min)/gt[1])
zone_y_min = int((lat_max - tile_lat_max)/gt[5])
zone_y_size = int((lat_max - lat_min)/gt[1])

# Read elevation data
zone_ele = tile_ele.ReadAsArray(zone_x_min, zone_y_min, zone_x_size, zone_y_size).astype(np.float)

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
  track_z_m.append(zz)

# Display path nodes as spheres
# mlab.points3d(track_x_m, track_y_m, track_z_m, color=(1,0,0), mode='sphere', scale_factor=100)
# Display path as line
mlab.plot3d(track_x_m, track_y_m, track_z_m, color=(255.0/255.0,102.0/255.0,0), line_width=10.0, tube_radius=50.0)

# Show the 3D map
mlab.show()
