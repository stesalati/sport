# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab
import numpy as np
from osgeo import gdal, osr
#import OsmApi
import math
import os
import sys
import bombo as bombo

#sys.path.append('C:/Users/salatis/AppData/Local/Continuum/Anaconda2/Scripts')
#sys.path.append('~/anaconda/bin/')
import gdal_merge as gm

TRACE_SIZE_ON_3DMAP = 200.0
ELEVATION_DATA_FOLDER = "elevationdata/"
TILES_DOWNLOAD_LINK = "http://dwtkns.com/srtm/"

track_lat = None
track_lon = None
tile_selection = 'auto'
# tile_selection = 'iceland.tif'
margin=20
elevation_scale=1
plot=True
verbose=True

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

textsize = margin * 10

# If track_lat and track_lon are None, run a demo
if track_lat == track_lon == None:
    # startingpoint = (44.1938472, 10.7012833)    # Cimone
    # startingpoint = (46.5145639, 11.7398472)    # Rif. Demetz
    startingpoint = (-8.4166000, 116.4666000)   # Rinjani
    # startingpoint = (64.0158333, -016.6747222)  # Peak in Iceland
    R = 0.01
    track_lat1 = np.linspace(-R, R, 1000).transpose()
    track_lon1 = np.sqrt(R**2 - track_lat1[0:1000]**2)
    track_lat2 = np.linspace(R, -R, 1000).transpose()
    track_lon2 = -np.sqrt(R**2 - track_lat2[0:1000]**2)
    track_lat = np.hstack((track_lat1[0:-2], track_lat2))
    track_lon = np.hstack((track_lon1[0:-2], track_lon2))
    track_lat = track_lat + startingpoint[0]
    track_lon = track_lon + startingpoint[1]
    
    # Determine the coordinates of the area we are interested in
    lat_min = np.min(track_lat) - margin * px2deg
    lat_max = np.max(track_lat) + margin * px2deg
    lon_min = np.min(track_lon) - margin * px2deg
    lon_max = np.max(track_lon) + margin * px2deg

if tile_selection == 'auto':
    # Tiles will be determined automatically
    # Determine which tiles are necessary
    tile_corner_min = SRTMTile(lat_min, lon_min)
    tile_corner_max = SRTMTile(lat_max, lon_max)
    tiles_x = range(tile_corner_min[0], tile_corner_max[0]+1)
    tiles_y = range(tile_corner_max[1], tile_corner_min[1]+1) # Inverted min and max as tiles are numbered, vertically, from north to south 
    
    if verbose:
        print "Required tiles:"
        print "X: {}".format(tiles_x)
        print "Y: {}".format(tiles_y)
        
    if len(tiles_x) > 1 or len(tiles_y) > 1:
        # More than one tile is needed, check if the mosaic has already been
        # generated in the past or if we need to generate it now
        merged_tile_name = "from_{:02}_{:02}_to_{:02}_{:02}.tif".format(tiles_x[0], tiles_y[0], tiles_x[-1], tiles_y[-1])
        if not os.path.isfile(ELEVATION_DATA_FOLDER + merged_tile_name):
            # Create mosaic: generate tile names and merge them
            gdal_merge_command_list = ['', '-o', ELEVATION_DATA_FOLDER + merged_tile_name]
            for tile_x in tiles_x:
                for tile_y in tiles_y:
                    # Generate tile filename and append it to the list
                    tilename = "srtm_{:02d}_{:02d}".format(tile_x, tile_y)
                    filename = ELEVATION_DATA_FOLDER + "{}/{}.tif".format(tilename, tilename)
                    gdal_merge_command_list.append(filename)
                    if not os.path.isfile(filename):
                        print "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.".format(tilename, TILES_DOWNLOAD_LINK)
                        #return
            if verbose:
                print "A tile mosaic is required: this merge command will be run: {}".format(gdal_merge_command_list)
            gm.main(gdal_merge_command_list)
        filename = ELEVATION_DATA_FOLDER + merged_tile_name
    else:
        # Only one tile is needed
        tilename = "srtm_{:02d}_{:02d}".format(tiles_x[0], tiles_y[0])
        filename = ELEVATION_DATA_FOLDER + "{}/{}.tif".format(tilename, tilename)
        if not os.path.isfile(filename):
            print "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.".format(tilename, TILES_DOWNLOAD_LINK)
            #return
            
else:
    # The tile name is provided (useful for those areas, e.g. Iceland, not covered by the SRTM survey)
    filename = ELEVATION_DATA_FOLDER + tile_selection

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

if verbose:
    print "\nCoordinate boundaries:"
    print "Longitude (X): {} <-- {} -- {} --> {}".format(tile_lon_min, lon_min, lon_max, tile_lon_max)
    print "Latitude (Y):  {} <-- {} -- {} --> {}".format(tile_lat_min, lat_min, lat_max, tile_lat_max)

# Selecting only a zone of the whole map, the one we're interested in plotting
zone_x_min = int((lon_min - tile_lon_min)/gt[1])
zone_x_size = int((lon_max - lon_min)/gt[1])
zone_y_min = int((lat_max - tile_lat_max)/gt[5])
zone_y_size = int((lat_min - lat_max)/gt[5])  # Inverted min and max as tiles are numbered, vertically, from north to south 

if verbose:
    print "\nSelected zone:"
    print "Longitude (X): Start: {}, Size: {}".format(zone_x_min, zone_x_size)
    print "Latitude (Y):  Start: {}, Size: {}".format(zone_y_min, zone_y_size)

# Read elevation data
zone_ele = tile_ele.ReadAsArray(zone_x_min, zone_y_min, zone_x_size, zone_y_size).astype(np.float)

# Set sea level at 0m instead of -32768 (Dead Sea level used as minimum value)
zone_ele[zone_ele < 418] = 0

# Create X,Y coordinates for zone_ele array (contains Z in meters)
line_x_deg = np.arange(tile_lon_min+zone_x_min*gt[1], tile_lon_min+(zone_x_min+zone_x_size)*gt[1], gt[1])[0:zone_x_size]
array_x_deg = np.tile(line_x_deg, (len(zone_ele), 1)).transpose()

line_y_deg = np.arange(tile_lat_max+zone_y_min*gt[5], tile_lat_max+(zone_y_min+zone_y_size)*gt[5], gt[5])[0:zone_y_size]
line_y_m = np.array([degrees2metersLatY(j) for j in line_y_deg])
array_y_m = np.tile(line_y_m, (len(zone_ele[0]), 1))

array_x_m = np.empty_like(array_x_deg)
for x, y in np.ndindex(array_x_deg.shape):
    array_x_m[x,y] = degrees2metersLongX(line_y_deg[y], array_x_deg[x,y])

zone_ele = zone_ele.transpose()

# Display 3D surface
if plot:
    mlab.mesh(array_x_m, array_y_m, zone_ele * elevation_scale)

# Hiking path
track_x_m = list()
track_y_m = list()
track_z_m = list()
for i in range(np.size(track_lat, axis=0)):
  (x,y) = degrees2meters(track_lon[i], track_lat[i])
  track_x_m.append(x)
  track_y_m.append(y)
  zz = zone_ele[int(round((track_lon[i] - (tile_lon_min+zone_x_min*gt[1])) / gt[1])), int(round((track_lat[i] - (tile_lat_max+zone_y_min*gt[5])) / gt[5]))]
  track_z_m.append(zz)

if plot:
    # Display path nodes as spheres
    # mlab.points3d(track_x_m, track_y_m, track_z_m, color=(1,0,0), mode='sphere', scale_factor=100)
    # Display path as line
    mlab.plot3d(track_x_m, track_y_m, track_z_m, color=(255.0/255.0, 102.0/255.0, 0), line_width=10.0, tube_radius=TRACE_SIZE_ON_3DMAP)
    
mlab.text3d((array_x_m[0][0] + array_x_m[-1][0])/2, array_y_m[0][0], np.max(zone_ele), "NORTH", scale=(textsize, textsize, textsize))
mlab.text3d(track_x_m[0], track_y_m[0], track_z_m[0]*1.5, "START", scale=(textsize, textsize, textsize))
    
if plot:
    # Set camera position
    mlab.view(azimuth=-90.0,
              elevation=60.0,
              # distance=1.0,
              distance='auto',
              # focalpoint=(1000.0, 1000.0, 1000.0),
              focalpoint='auto',
              roll=0.0)
    
    # Show the 3D map
    mlab.show()

# Creating the export dictionary
terrain = {'x': array_x_m, 
           'y': array_y_m,
           'z': zone_ele * elevation_scale}
track = {'x': track_x_m,
         'y': track_y_m,
         'z': track_z_m,
         'color': (255.0/255.0, 102.0/255.0, 0),
         'line_width': 10.0,
         'tube_radius': TRACE_SIZE_ON_3DMAP}
