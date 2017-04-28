# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab
import numpy as np
from osgeo import gdal, osr
import math
import os
import sys
import vtk
from tvtk.api import tvtk
import StringIO
from PIL import Image
import PIL
#from mpld3 import plugins, utils
import urllib2, urllib
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.pyplot as mpld3
from pylab import imshow, imread, show
import scipy.misc
import bombo as bombo

#sys.path.append('C:/Users/salatis/AppData/Local/Continuum/Anaconda2/Scripts')
#sys.path.append('~/anaconda/bin/')
import gdal_merge as gm

TRACE_SIZE_ON_3DMAP = 200.0
COORDS_PLOT_SCALE = 10000
ELEVATION_DATA_FOLDER = "elevationdata/"
TILES_DOWNLOAD_LINK = "http://dwtkns.com/srtm/"
USE_PROXY = False
PROXY_DATA = 'salatis:Alzalarosa01@userproxy.tmg.local:8080'

track_lat = None
track_lon = None
tile_selection = 'auto'
# tile_selection = 'iceland.tif'
margin=300
elevation_scale=1
plot = 'coords'
#plot = 'meters'
use_osm_texture = True
verbose=True





"""
ACTUAL FUNCTION UNDER TEST
"""


def MapTilesDeg2Num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def MapTilesNum2Deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def GetMapImageCluster(use_proxy, proxy_data, lat_deg, lon_deg, delta_lat, delta_long, zoom, verbose):
    # Proxy setup with urllib
    # proxies = {'http': 'salatis:Alzalarosa01@userproxy.tmg.local:8080'}
    
    # Proxy setup with urllib2
    if use_proxy:
        proxy = urllib2.ProxyHandler({'http': proxy_data})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
    
    # Local storage folder
    image_storage_path = 'map_tiles/'
    savename = r"{0}_{1}_{2}.png"
    
    # Request url
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    
    # These are the tiles to download
    xmin, ymax = MapTilesDeg2Num(lat_deg, lon_deg, zoom)
    xmax, ymin = MapTilesDeg2Num(lat_deg + delta_lat, lon_deg + delta_long, zoom)
    if verbose:
        print "\nOSM tiles %d - %d (horizontally) and %d - %d (vertically) are needed" % (xmin, xmax, ymin, ymax)
    
    # Margin coordinates of the tiles that were downloaded (adding 1 to the max
    # tiles as apparently the coordinates returned by MapTilesNum2Deg refer to
    # the origin of the tile)
    lat_min, lon_min = MapTilesNum2Deg(xmin, ymin, zoom)
    lat_max, lon_max = MapTilesNum2Deg(xmax + 1, ymax + 1, zoom)
    lat_min_actual = np.min((lat_min, lat_max))
    lat_max_actual = np.max((lat_min, lat_max))
    lon_min_actual = np.min((lon_min, lon_max))
    lon_max_actual = np.max((lon_min, lon_max))
    osm_tiles_edges = {"lat_min": lat_min_actual,
                       "lat_max": lat_max_actual,
                       "lon_min": lon_min_actual,
                       "lon_max": lon_max_actual}
    if verbose:
        print "\nOSM coordinate boundaries (limit <-- min -- max --> limit):"
        print "Longitude (X): {} <-- {} -- {} --> {}".format(osm_tiles_edges['lon_min'], lon_deg, lon_deg + delta_long, osm_tiles_edges['lon_max'])
        print "Latitude (Y):  {} <-- {} -- {} --> {}".format(osm_tiles_edges['lat_min'], lat_deg, lat_deg + delta_lat, osm_tiles_edges['lat_max'])
    
    # Populate the desired map with tiles
    Cluster = Image.new('RGB',((xmax-xmin+1)*256-1, (ymax-ymin+1)*256-1))
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            try:
                # Check if the tile is already present locally
                if os.path.isfile(image_storage_path + savename.format(zoom, xtile, ytile)):
                    tile = Image.open(image_storage_path + savename.format(zoom, xtile, ytile))
                else:
                    # Download from the Internet and save it locally for future
                    # use
                    imgurl = smurl.format(zoom, xtile, ytile)
                    print("OSM tile not found locally, have to download it from: " + imgurl)
                    imgstr = urllib2.urlopen(imgurl).read()
                    tile = Image.open(StringIO.StringIO(imgstr))
                    with open(image_storage_path + savename.format(zoom, xtile, ytile), 'wb') as f:
                        f.write(imgstr)
                        f.close()
                # Append it to the rest of the cluster
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except: 
                print("OSM tile loading failed.")
                tile = None
                
    return Cluster, osm_tiles_edges

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

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

earthRadius = 6371000 # Earth radius in meters (yes, it's an approximation) https://en.wikipedia.org/wiki/Earth_radius
px2deg = 0.0008333333333333334

textsize = margin * 10

# If track_lat and track_lon are None, run a demo
if track_lat == track_lon == None:
    # startingpoint = (44.1938472, 10.7012833)    # Cimone
    # startingpoint = (46.5145639, 11.7398472)    # Rif. Demetz
    startingpoint = (-08.4113472, 116.4166667)   # Rinjani
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
    
    track_lat = [startingpoint[0]]
    track_lon = [startingpoint[1]]
    
    # Determine the coordinates of the area we are interested in
    lat_min = np.min(track_lat) - margin * px2deg
    lat_max = np.max(track_lat) + margin * px2deg
    lon_min = np.min(track_lon) - margin * px2deg * 2.0 # per avere una cosa rettangolare e capire bene le direzioni
    lon_max = np.max(track_lon) + margin * px2deg * 2.0

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
    print "\nElevation coordinate boundaries (limit <-- min -- max --> limit):"
    print "Longitude (X): {} <-- {} -- {} --> {}".format(tile_lon_min, lon_min, lon_max, tile_lon_max)
    print "Latitude (Y):  {} <-- {} -- {} --> {}".format(tile_lat_min, lat_min, lat_max, tile_lat_max)

# Selecting only a zone of the whole map, the one we're interested in plotting
zone_x_min = int((lon_min - tile_lon_min)/gt[1])
zone_x_size = int((lon_max - lon_min)/gt[1])
zone_y_min = int((lat_max - tile_lat_max)/gt[5])
zone_y_size = int((lat_min - lat_max)/gt[5])  # Inverted min and max as tiles are numbered, vertically, from north to south 

if verbose:
    print "\nElevation selected zone:"
    print "Longitude (X): Start: {}, Size: {}".format(zone_x_min, zone_x_size)
    print "Latitude (Y):  Start: {}, Size: {}".format(zone_y_min, zone_y_size)

# Read elevation data
zone_ele = tile_ele.ReadAsArray(zone_x_min, zone_y_min, zone_x_size, zone_y_size).astype(np.float)

# Set sea level at 0m instead of -32768 (Dead Sea level used as minimum value)
zone_ele[zone_ele < -418] = -500

# Create X,Y coordinates for zone_ele array (contains Z in meters)
line_x_deg = np.arange(tile_lon_min+zone_x_min*gt[1], tile_lon_min+(zone_x_min+zone_x_size)*gt[1], gt[1])[0:zone_x_size]
array_x_deg = np.tile(line_x_deg, (len(zone_ele), 1)).transpose()

line_y_deg = np.arange(tile_lat_max+zone_y_min*gt[5], tile_lat_max+(zone_y_min+zone_y_size)*gt[5], gt[5])[0:zone_y_size]
array_y_deg = np.tile(line_y_deg, (len(zone_ele[0]), 1))
line_y_m = np.array([degrees2metersLatY(j) for j in line_y_deg])
array_y_m = np.tile(line_y_m, (len(zone_ele[0]), 1))

array_x_m = np.empty_like(array_x_deg)
for x, y in np.ndindex(array_x_deg.shape):
    array_x_m[x,y] = degrees2metersLongX(line_y_deg[y], array_x_deg[x,y])

zone_ele = zone_ele.transpose()

# Display 3D elevation
if plot == 'meters':
    mesh = mlab.mesh(array_x_m, array_y_m, zone_ele * elevation_scale,
                     color=(1, 1, 1))

if plot == 'coords':
    """
    mesh = mlab.surf(np.fliplr(zone_ele) * elevation_scale,
                     warp_scale=0.020,
                     color=(1, 1, 1))
    """
    mesh = mlab.mesh(array_x_deg * COORDS_PLOT_SCALE, array_y_deg * COORDS_PLOT_SCALE, zone_ele * elevation_scale * 0.1,
                     color=(1, 1, 1))
    
    # Display the OSM texture overlay    
    if use_osm_texture:
        # Create the texture
        a, osm_tiles_edges = GetMapImageCluster(use_proxy=USE_PROXY, proxy_data=PROXY_DATA,
                                                lat_deg=lat_min, lon_deg=lon_min,
                                                delta_lat=(lat_max-lat_min), delta_long=(lon_max-lon_min),
                                                zoom=13,
                                                verbose=True)
        
        """
        # Provo a vedere se i punti corrispondono sulle due mappe
        # Questa verifica va fatta prima di modificare l'immagine, per vedere che i punti corrispondano
        fig, ax = mpld3.subplots()
        img = np.asarray(a)
        ax.imshow(img, extent=[osm_tiles_edges['lon_min'], osm_tiles_edges['lon_max'], osm_tiles_edges['lat_min'], osm_tiles_edges['lat_max']], zorder=0)#, origin="lower")
        points = ax.scatter(track_lon, track_lat, s=4)
        ax.grid(True)
        mpld3.show()
        # La mappa sembra ok, le coordinate corrispondono
        """
        
        # Trim the map accordingly
        print("\nHow much needs to be trimmed")
        print "Longitude (X): {} <-- {} -- {} --> {}".format(osm_tiles_edges['lon_min'], lon_min, lon_max, osm_tiles_edges['lon_max'])
        print "Latitude (Y):  {} <-- {} -- {} --> {}".format(osm_tiles_edges['lat_min'], lat_min, lat_max, osm_tiles_edges['lat_max'])
        
        #a = a.rotate(180)
        #a = a.transpose(Image.TRANSPOSE)
        #a = a.transpose(Image.FLIP_TOP_BOTTOM)
        #img = np.transpose(img, axes=(1, 0, 2))
        
        height = a.size[1]
        width = a.size[0]
        
        # Method 1: with vectors
        h_coord_vector = np.linspace(osm_tiles_edges['lon_min'], osm_tiles_edges['lon_max'], width)
        h_pixel_vector = np.linspace(0, width-1, width)
        h_min_idx, h_min_value = find_nearest(h_coord_vector, lon_min)
        h_max_idx, h_max_value = find_nearest(h_coord_vector, lon_max)
        
        v_coord_vector = np.linspace(osm_tiles_edges['lat_max'], osm_tiles_edges['lat_min'], height)
        v_pixel_vector = np.linspace(0, height-1, height)
        v_min_idx, v_min_value = find_nearest(v_coord_vector, lat_min)
        v_max_idx, v_max_value = find_nearest(v_coord_vector, lat_max)
        
        trim_margins = {'left': int(h_min_idx),
                        'right': int(h_max_idx),
                        'bottom': int(v_min_idx),
                        'top': int(v_max_idx)}
        a_trimmed = a.crop((trim_margins['left'], trim_margins['top'], trim_margins['right'], trim_margins['bottom']))
        
        # Method 2: operations
        """
        h_deg2px_ratio = width / (osm_tiles_edges['lon_max'] - osm_tiles_edges['lon_min'])
        v_deg2px_ratio = height / (osm_tiles_edges['lat_max'] - osm_tiles_edges['lat_min'])
        
        trim_margins = {'left': int(np.round((lon_min - osm_tiles_edges['lon_min']) * h_deg2px_ratio)),
                        'right': int(np.round((osm_tiles_edges['lon_max'] - lon_max) * h_deg2px_ratio)),
                        'bottom': int(np.round((lat_min - osm_tiles_edges['lat_min']) * v_deg2px_ratio)),
                        'top': int(np.round((osm_tiles_edges['lon_max'] - lat_max) * v_deg2px_ratio))}
        a = a.crop((trim_margins['left'], trim_margins['top'], width-trim_margins['right'], height-trim_margins['bottom']))
        """
        
        # Save the texture as a JPG
        img = np.asarray(a_trimmed)
        scipy.misc.imsave('texture.jpg', img)
        
        # Read and apply texture
        """
        textureReader = vtk.vtkJPEGReader(file_name='texture.jpg')
        texture = vtk.vtkTexture()
        texture.SetInputConnection(textureReader.GetOutputPort())
        """
        bmp = tvtk.JPEGReader(file_name='texture.jpg')
        texture = tvtk.Texture(input_connection=bmp.output_port, interpolate=0)
        
        mesh.actor.actor.mapper.scalar_visibility=False
        mesh.actor.enable_texture = True
        mesh.actor.tcoord_generator_mode = 'plane'
        mesh.actor.actor.texture = texture
    
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

if plot == 'meters':
    pass
    # Display path nodes as spheres
    mlab.points3d(track_x_m, track_y_m, track_z_m, color=(255.0/255.0, 102.0/255.0, 0), mode='sphere', scale_factor=500)
    # Display path as line
    # mlab.plot3d(track_x_m, track_y_m, track_z_m, color=(255.0/255.0, 102.0/255.0, 0), line_width=10.0, tube_radius=TRACE_SIZE_ON_3DMAP)
if plot == 'coords':
    mlab.points3d(track_x_c, track_y_c, track_z_m, color=(255.0/255.0, 102.0/255.0, 0), mode='sphere', scale_factor=500)
    
mlab.text3d((array_x_m[0][0] + array_x_m[-1][0])/2, array_y_m[0][0], np.max(zone_ele), "NORTH", scale=(textsize, textsize, textsize))
mlab.text3d(track_x_m[0], track_y_m[0], track_z_m[0]*1.5, "START", scale=(textsize, textsize, textsize))
    
if plot == 'coords' or plot == 'meters':
    # Set camera position
    """
    mlab.view(azimuth=-90.0,
              elevation=60.0,
              # distance=1.0,
              distance='auto',
              # focalpoint=(1000.0, 1000.0, 1000.0),
              focalpoint='auto',
              roll=0.0)
    """
    
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
