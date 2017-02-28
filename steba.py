# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
"""

# Use "%matplotlib qt" to plot in a different window

# DOCUMENTAZIONE PIU' UTILE
# https://github.com/FlorianWilhelm/gps_data_with_python
# https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks
# http://nbviewer.jupyter.org/format/slides/github/FlorianWilhelm/gps_data_with_python/blob/master/talk.ipynb#/8/2

# http://www.trackprofiler.com/gpxpy/index.html

import numpy as np
from scipy import signal, fftpack#, misc
import matplotlib.pyplot as plt, mpld3
from matplotlib import gridspec, patches
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import gpxpy
import datetime
import urllib2#, urllib
# from pylab import imshow, imread, show
from mpld3 import plugins#, utils
import mplleaflet
import os.path
import folium
from folium import plugins as fp
import webbrowser
import vincent
import json
import sys
import math
import StringIO
from PIL import Image
from pykalman import KalmanFilter
import srtm
import pandas as pd
import platform


###############################################################################


def LoadGPX(filename, use_srtm_elevation):
    gpx_file = open(filename, 'r')
    gpx = gpxpy.parse(gpx_file)
    
    # In case there's more than one track/segment
    # for track in gpx.tracks:
    #     for segment in track.segments:        
    #         for point in segment.points:
    
    segment = gpx.tracks[0].segments[0]
    coords = pd.DataFrame([
            {'idx': i,
             'lat': p.latitude, 
             'lon': p.longitude, 
             'ele': p.elevation,
             'time': p.time,
             'time_sec': (p.time - datetime.datetime(2000,1,1,0,0,0)).total_seconds()} for i, p in enumerate(segment.points)])
    coords.set_index('time', drop = True, inplace = True)
    coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
    
    print "\nStats based on the GPX file"
    print segment.get_uphill_downhill()
    print segment.get_elevation_extremes()
    print segment.get_moving_data()
    
    # https://github.com/tkrajina/srtm.py
    if use_srtm_elevation:
        try:
            # Delete elevation data (it's already saved in coords)
            for p in gpx.tracks[0].segments[0].points:
                p.elevation = None
                
            # Get elevation from SRTM
            elevation_data = srtm.get_data()
            elevation_data.add_elevations(gpx, smooth=True)
            coords['srtm'] = [p.elevation for p in gpx.tracks[0].segments[0].points]
            coords[['ele','srtm']].plot(title='Elevation')
            print "\nStats based on SRTM corrected elevation"
            print segment.get_uphill_downhill()
            print segment.get_elevation_extremes()
                  
        except:
            print "SRTM not working for some reason, probably a shitty proxy."
    
    # Round sampling points at 1s. The error introduced should be negligible
    # the processing would be simplified
    coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
    
    # Add speed using embedded function (it won't be used, it's just to completeness)
    segment.points[0].speed, segment.points[-1].speed = 0., 0.
    gpx.add_missing_speeds()
    coords['speed'] = [p.speed for p in gpx.tracks[0].segments[0].points]
    
    return gpx, coords


def FindQuadrant(deg):
    n = np.zeros(len(deg))
    n[np.where((deg >= 0) & (deg < 90) )] = 1
    n[np.where((deg >= 90) & (deg < 180) )] = 4
    n[np.where((deg >= 180) & (deg < 270) )] = 3
    n[np.where((deg >= 270) & (deg < 360) )] = 2
    n[np.where((deg < 0) & (deg >= -90) )] = 2
    n[np.where((deg < -90) & (deg >= -180) )] = 3
    n[np.where((deg < -180) & (deg >= -270) )] = 4
    n[np.where((deg < -270) & (deg >= -360) )] = 1
    return n


def PlotOnMap(lat, lon, data, sides, palette, library, s, h, gradient, speed_h):

    # Create new figure
    fig, ax = plt.subplots()

    # Actual math
    dtrace_lon = np.diff(lon)
    dtrace_lat = np.diff(lat)
    m = dtrace_lat/dtrace_lon
    deg = np.arctan2(dtrace_lat, dtrace_lon) / (2*np.pi) * 360
    m[np.where(m == 0)] = 0.0000001
    m_p = -1/m
    quad = FindQuadrant(deg+90)
    
    # For each data vectors (columns of data)
    distances = list()
    M = np.size(data, axis = 1)
    for col in range(M):
        tmp_x = data[1:,col] / np.sqrt(1+m_p**2)
        tmp_y = tmp_x * m_p
        tmp_side = sides[col]
        
        idx_quad_1 = np.where(quad == 1)
        idx_quad_2 = np.where(quad == 2)
        idx_quad_3 = np.where(quad == 3)
        idx_quad_4 = np.where(quad == 4)
        
        if tmp_side == 0:
            tmp_x[idx_quad_1] = tmp_x[idx_quad_1]
            tmp_y[idx_quad_1] = tmp_y[idx_quad_1]
            tmp_x[idx_quad_2] = tmp_x[idx_quad_2]
            tmp_y[idx_quad_2] = tmp_y[idx_quad_2]
            tmp_x[idx_quad_3] = -tmp_x[idx_quad_3]
            tmp_y[idx_quad_3] = -tmp_y[idx_quad_3]
            tmp_x[idx_quad_4] = -tmp_x[idx_quad_4]
            tmp_y[idx_quad_4] = -tmp_y[idx_quad_4]
        else:
            tmp_x[idx_quad_1] = -tmp_x[idx_quad_1]
            tmp_y[idx_quad_1] = -tmp_y[idx_quad_1]
            tmp_x[idx_quad_2] = -tmp_x[idx_quad_2]
            tmp_y[idx_quad_2] = -tmp_y[idx_quad_2]
            tmp_x[idx_quad_3] = tmp_x[idx_quad_3]
            tmp_y[idx_quad_3] = tmp_y[idx_quad_3]
            tmp_x[idx_quad_4] = tmp_x[idx_quad_4]
            tmp_y[idx_quad_4] = tmp_y[idx_quad_4]
        
        distances.append((tmp_x, tmp_y))
        
    # Coordinates center
    lat_center = np.median(lat)
    lon_center = np.median(lon)
    
    # Depending on the library used for plotting (1 = mplleaflet, 2 = folium)
    # https://www.youtube.com/watch?v=BwqBNpzQwJg
    if library == 1:
        # Plot trace
        ax.plot(lon, lat, 'k', linewidth = 4)
        
        # Plot data
        for col in range(M):
            tmp_lon = lon[1:] + distances[col][0]
            tmp_lat = lat[1:] + distances[col][1]
            tmp_poly_lon = np.hstack((lon[1:], np.flipud(tmp_lon)))
            tmp_poly_lat = np.hstack((lat[1:], np.flipud(tmp_lat)))
            tmp_poly = np.vstack((tmp_poly_lon,tmp_poly_lat)).T
            ax.add_patch(patches.Polygon(tmp_poly, hatch="o", facecolor=palette[col], alpha = 1.0))
            
        # Plot on OSM
        # http://matthiaseisen.com/pp/patterns/p0203/
        mplleaflet.show(fig = ax.figure)
        
    elif library == 2:
        # https://github.com/python-visualization/folium/tree/master/examples
        # Initialize map
        map_osm = folium.Map(location=[lat_center, lon_center], zoom_start=13)#, tiles='Stamen Terrain')
        
        # Data to plot made with Vincent
        # http://vincent.readthedocs.io/en/latest/quickstart.html
        index = np.ndarray.tolist(s)
        
        # Some stats
        #ascent_text = "Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh_filtered, 1), TotalAscentDescent(dh_filtered, -1))
        #ascent_text = "Total ascent. %dm" % (TotalAscentDescent(dh_filtered, 1))
        
        # Altitude
        plot_h = {'index': index}
        plot_h['h'] = np.ndarray.tolist(h[1:])               
        line = vincent.Area(plot_h, iter_idx='index')
        line.axis_titles(x='Distance', y='Altitude')
        #line.legend(title = ascent_text)
        line.to_json('plot_h.json')
        marker_pos1 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))]]
        
        # Gradient
        plot_gradient = {'index': index}
        plot_gradient['gradient'] = np.ndarray.tolist(gradient)               
        line = vincent.Line(plot_gradient, iter_idx='index')
        line.axis_titles(x='Distance', y='Altitude')
        # line.legend(title='Categories')
        line.to_json('plot_gradient.json')
        #marker_pos2 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))] + 0.01 * (np.max(lon) - np.min(lon))]
        
        # Speed_h
        plot_speed_h = {'index': index}
        plot_speed_h['speed_h'] = np.ndarray.tolist(speed_h)               
        line = vincent.Line(plot_speed_h, iter_idx='index')
        line.axis_titles(x='Distance', y='Altitude')
        # line.legend(title='Categories')
        line.to_json('plot_speed_h.json')
        #marker_pos3 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))] + 0.02 * (np.max(lon) - np.min(lon))]
        
        # Plot markers
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Quickstart.ipynb
        # Icons: 'ok-sign', 'cloud', 'info-sign', 'remove-sign', http://getbootstrap.com/components/
        map_osm.add_children(folium.Marker([lat[0], lon[0]],
                                           popup = "Start",
                                           icon=folium.Icon(color='green', icon='circle-arrow-up'),))
        map_osm.add_children(folium.Marker([lat[-1], lon[-1]], 
                                           popup = "End",
                                           icon=folium.Icon(color='red', icon='circle-arrow-down')))
        marker_highest_point = np.where(h == np.max(h))[0][0]
        
        highest_point_popup = folium.Popup(max_width = 1200).add_child(
                                folium.Vega(json.load(open('plot_h.json')), width = 1200, height = 600))
        map_osm.add_children(folium.Marker([lat[marker_highest_point], lon[marker_highest_point]], 
                                           # popup = "Highest point",
                                           popup = highest_point_popup,
                                           icon=folium.Icon(icon='cloud')))
        
        # Plot "button" markers for plots
        #folium.RegularPolygonMarker(
        #    location = marker_pos1,
        #    fill_color = '#00FFFF',
        #    radius = 12,
        #    number_of_sides = 3,
        #    popup = ascent_text
        #).add_to(map_osm)
        #folium.RegularPolygonMarker(
        #    location = marker_location_gradient,
        #    fill_color = '#00FFFF',
        #    radius = 12,
        #    number_of_sides = 3,
        #    popup=folium.Popup(max_width = 1000).add_child(
        #        folium.Vega(json.load(open('plot_gradient.json')), width = 1000, height = 250))
        #).add_to(map_osm)
        #
        #folium.RegularPolygonMarker(
        #    location = marker_location_speed_h,
        #    fill_color = '#FF0000',
        #    radius = 12,
        #    number_of_sides = 3,
        #    popup=folium.Popup(max_width = 1000).add_child(
        #        folium.Vega(json.load(open('plot_speed_h.json')), width = 1000, height = 250))
        #).add_to(map_osm)
        
        # Plot data
        # Create patches the mplleaflet way, one for every data we want to plot
        for col in range(M):
            tmp_lon = lon[1:] + distances[col][0]
            tmp_lat = lat[1:] + distances[col][1]
            tmp_poly_lon = np.hstack((lon[1:], np.flipud(tmp_lon)))
            tmp_poly_lat = np.hstack((lat[1:], np.flipud(tmp_lat)))
            tmp_poly = np.vstack((tmp_poly_lon,tmp_poly_lat)).T
            ax.add_patch(patches.Polygon(tmp_poly, hatch="o", facecolor=palette[col], alpha = 1.0))
        # Convert them to GeoJson (apparently this way it works, but in theory it should be the same thing as writing the polygon in json directly)
        # https://pypi.python.org/pypi/folium
        # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Folium_and_mplleaflet.ipynb
        # http://python-visualization.github.io/folium/module/features.html
        # https://github.com/python-visualization/folium/issues/318
        data_patches = mplleaflet.fig_to_geojson(fig=fig)
        # Now apply the data, including formatting, contained in data_patches
        style_function = lambda feature : dict(
            color = feature['properties']['fillColor'],
            weight = feature['properties']['weight'],
            opacity = feature['properties']['opacity'])
        for feature in data_patches['features']:
            gj = folium.GeoJson(feature, style_function=style_function)
            gj.add_to(map_osm)
            
            # POLYGON WITG JSON
            # a = [[[27, 43], [33, 43], [33, 47], [27, 47]]]
            # a = [np.ndarray.tolist(np.vstack((tmp_poly_lat, tmp_poly_lon)).T)]
            # gj_poly = folium.GeoJson(data={"type": "Polygon", "coordinates": a})
            # gj_poly.add_to(map_osm)
            
            # NOT SUPPORTED BY THE CURRENT VERSION OF FOLIUM, MUST UPGRADE IT
            #folium.features.PolygonMarker(
            #    np.vstack((tmp_poly_lat, tmp_poly_lon)).T,
            #    color='blue',
            #    weight=10,
            #    fill_color='red',
            #    fill_opacity=0.5,
            #    popup='Tokyo, Japan').add_to(map_osm)

        # Plot trace
        map_osm.add_children(folium.PolyLine(np.vstack((lat, lon)).T, color='#000000', weight = 4))

        # Add fullscreen capability
        try:
            fp.Fullscreen(position='topright',
                          title='Expand me',
                          titleCancel='Exit me',
                          forceSeparateButton=True).add_to(map_osm)
        except:
            print "Update Folium"
        
        # Create map
        map_osm.save("osm.html", close_file=False)
        # On Windows
        # webbrowser.open("osm.html", new=2)
        # On MAC
        if platform.system() == "Darwin":
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + "osm.html")
        
    return fig


## Functions to download tiles from OSM, useless as I use folium and mplleaflet
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
        print "Tiles %d - %d (horizontally) and %d - %d (vertically) are needed" % (xmin, xmax, ymin, ymax)
    
    # Margin coordinates of the tiles that were downloaded (adding 1 to the max
    # tiles as apparently the coordinates returned by MapTilesNum2Deg refer to
    # the origin of the tile)
    lat_min, lon_min = MapTilesNum2Deg(xmin, ymin, zoom)
    lat_max, lon_max = MapTilesNum2Deg(xmax + 1, ymax + 1, zoom)
    lat_min_actual = np.min((lat_min, lat_max))
    lat_max_actual = np.max((lat_min, lat_max))
    lon_min_actual = np.min((lon_min, lon_max))
    lon_max_actual = np.max((lon_min, lon_max))
    tiles_edges_coords = (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)
    if verbose:
        print "User requested area >>>>>>>>>>>>> lat: %f - %f, lon: %f - %f" % (lat_deg, lat_deg + delta_lat, lon_deg, lon_deg + delta_long)
        print "Returned area (must be wider) >>> lat: %f - %f, lon: %f - %f" % (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)
    
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
                    print("Tile not found locally, have to download it from: " + imgurl)
                    imgstr = urllib2.urlopen(imgurl).read()
                    tile = Image.open(StringIO.StringIO(imgstr))
                    with open(image_storage_path + savename.format(zoom, xtile, ytile), 'wb') as f:
                        f.write(imgstr)
                        f.close()
                # Append it to the rest of the cluster
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except: 
                print("Tile loading (either from local repository or the Internet failed.")
                tile = None
                
    return Cluster, tiles_edges_coords


## "Homemade" processing functions
def RemoveOutliers(coords):
    # It's necessary to do so before filtering as they would alter the results quite significantly
    
    # Constants
    LIMIT_POS_SPEED_H = 12.0
    LIMIT_NEG_SPEED_H = -12.0
    LIMIT_POS_SPEED_V = 3.0
    LIMIT_NEG_SPEED_V = -3.0
    LIMIT_POS_GRADIENT = 4.0
    LIMIT_NEG_GRADIENT = -4.0
    
    # Renaming variables for ease of use
    lat = coords['lat'].values
    lon = coords['lon'].values
    h = coords['ele'].values
    t = coords['time_sec'].values
    s = np.cumsum(HaversineDistance(lat, lon))
    
    # Empty lists ready to be filled
    ds_list = list()
    speed_h_list = list()
    speed_v_list = list()
    gradient_list = list()
    valid = list()          # list of valid points
    valid.append(0)         # let's assume the first point is valid
    
    for i in range(1,len(h)):
        # It wouldn't be legit to compute the differentials with the last point if
        # that point is not valid. So first thing to do is to find the last valid
        # point...
        k = 1
        while (i-k) not in valid:
            k = k + 1
        # ...found, now let's recompute the differentials for this point.
        dh = h[i] - h[i-k]
        dt = t[i] - t[i-k]
        ds = HaversineDistanceBetweenTwoPoints(lat[i-k], lon[i-k], lat[i], lon[i])
        speed_h = ds/dt
        speed_v = dh/dt
        gradient = dh/ds
        # If the current point's stats are within the boundaries...
        current_point_valid = (speed_v < LIMIT_POS_SPEED_V) & (speed_v > LIMIT_NEG_SPEED_V) & (gradient < LIMIT_POS_GRADIENT) & (gradient > LIMIT_NEG_GRADIENT)
        if current_point_valid:
            # ...then the point is valid, update distance (computed from the last valid point)
            # print "Point %d is VALID" % i
            valid.append(i)
            ds_list.append(ds)
            speed_h_list.append(speed_h)
            speed_v_list.append(speed_v)
            gradient_list.append(gradient)
        else:
            # ...the the point is invalid
            if VERBOSE:
                print "Point %d is INVALID, compared with %d" % (i, i-k)
                print "  -> DIFFERENTIALS speed_h: %2.2f  speed_v: %2.2f  gradient: %2.2f" % (speed_h, speed_v, gradient)
    
    # plt.plot(s, h[1:len(h)], 'k-', s_cleaned, h_cleaned[1:len(h_cleaned)], 'r-')
    
    # Removing points declared invalid
    lat_cleaned = lat[valid]
    lon_cleaned = lon[valid]
    h_cleaned = h[valid]
    t_cleaned = t[valid]
    ds = np.asarray(ds_list)
    speed_h = np.asarray(speed_h_list)
    speed_v = np.asarray(speed_v_list)
    gradient = np.asarray(gradient_list)
    s_cleaned = np.cumsum(ds)
    
    return lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds, speed_h, speed_v, gradient


def ElevationFilter(h, ds, window):
    #b = np.sinc(0.25*np.linspace(-np.floor(window/2), np.floor(window/2), num=window, endpoint=True))# * signal.hamming(window)
    b = signal.hann(window)
    b = b/np.sum(b)
    #PlotFilter(b)
    h_filtered = signal.filtfilt(b, 1, h)
    
    # Recomputing speed and gradient (as they depend on h)
    dh_filtered = np.diff(h_filtered)
    gradient_filtered = dh_filtered/ds
    
    return h_filtered, dh_filtered, gradient_filtered


def PlotFilter(x):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(x)
    plt.title("Time response")
    plt.grid()
    
    ax2 = fig.add_subplot(1,2,2)
    X = fftpack.fft(x, 512) / (len(x)/2.0)
    freq = np.linspace(-0.5, 0.5, len(X))
    response = 20 * np.log10(np.abs(fftpack.fftshift(X / abs(X).max())))
    ax2.plot(freq, response)
    plt.axis([-0.5, 0.5, -70, 0])
    plt.title("Frequency response")
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.grid()
    

def HaversineDistance(lat, lon):
    # http://www.movable-type.co.uk/scripts/latlong.html
    lat_rad = lat/360*2*np.pi
    lon_rad = lon/360*2*np.pi
    dlat_rad = np.diff(lat_rad)
    dlon_rad = np.diff(lon_rad)
    a = np.power(np.sin(dlat_rad/2),2) + np.cos(lat_rad[0:-1]) * np.cos(lat_rad[1:len(lat_rad)]) * np.power(np.sin(dlon_rad/2),2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6371000*c
    return d


def HaversineDistanceBetweenTwoPoints(lat1, lon1, lat2, lon2):
    # http://www.movable-type.co.uk/scripts/latlong.html
    lat1_rad = lat1/360*2*np.pi
    lon1_rad = lon1/360*2*np.pi
    lat2_rad = lat2/360*2*np.pi
    lon2_rad = lon2/360*2*np.pi
    dlat_rad = lat2_rad - lat1_rad
    dlon_rad = lon2_rad - lon1_rad
    a = np.power(np.sin(dlat_rad/2),2) + np.cos(lat1_rad) * np.cos(lat2_rad) * np.power(np.sin(dlon_rad/2),2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6371000*c
    return d

    
#def Compute(lat, lon, h, t):
#    # Differentials
#    dh = np.diff(h)
#    dt = np.diff(t)
#    # Distance between consecutive points
#    ds = HaversineDistance(lat, lon)
#    s = np.cumsum(ds)
#    # Horizontal and vertical speeds between consecutive points
#    speed_h = ds/dt
#    speed_v = dh/dt
#    # Elevation gradient between consecutive points
#    gradient = dh/ds
#    return dh, dt, ds, s, speed_h, speed_v, gradient


def MapInteractivePlot(fig, s, h, dh, lat, lon, zoom_level, margin_percentage, use_proxy, proxy_data, verbose):
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 0.001, 5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    X = np.vstack((s, h[0:-1], lon[0:-1], lat[0:-1]))
    
    # Elevation over distance
    points = ax0.plot(X[0], X[1], color = '0.5')
    points = ax0.scatter(X[0], X[1], s=4)
    ax0.set_ylabel("Elevation (m)")
    ax0.set_xlabel("Distance (m)")
    ax0.grid(True)
    ax0.set_xlim(np.min(s), np.max(s))
    ax0.set_ylim(0, np.max(h) + 100)
    
    # Total ascent/descent
    #at = AnchoredText("Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh_filtered, 1), TotalAscentDescent(dh_filtered, -1)),
    #                  prop=dict(size=12), frameon=True,
    #                  loc=2,
    #                  )
    #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #ax1.add_artist(at)
    
    # Fake subplot
    points = ax1.scatter(X[0], X[2])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    
    # Map with route
    margin = np.max((np.max(lat) - np.min(lat), np.max(lon) - np.min(lon))) * margin_percentage
    lat_origin = np.min(lat) - margin
    lon_origin = np.min(lon) - margin
    lat_span = np.max(lat) - np.min(lat) + 2 * margin
    lon_span = np.max(lon) - np.min(lon) + 2 * margin
    a, tiles_edges_coords = GetMapImageCluster(use_proxy, proxy_data, lat_origin, lon_origin, lat_span, lon_span, zoom_level, verbose)
    fig.patch.set_facecolor('white')
    points = ax2.plot(X[2], X[3], color = '0.5')
    img = np.asarray(a)
    # Extent simply associates values, in this case longitude and latitude, to
    #	the map's corners.
    ax2.imshow(img, extent=[tiles_edges_coords[2], tiles_edges_coords[3], tiles_edges_coords[0], tiles_edges_coords[1]], zorder=0, origin="lower")
    points = ax2.scatter(X[2], X[3], s=4)
    ax2.set_xlim(lon_origin, lon_origin + lon_span)
    ax2.set_ylim(lat_origin, lat_origin + lat_span)    
    ax2.set_xlabel("Lat")
    ax2.set_ylabel("Lon")
    ax2.grid(True)
    plugins.connect(fig, plugins.LinkedBrush(points))
    mpld3.show()
    
    return fig


def SummaryPlot(ax1, ax2, ax3, ax4, s, h, dh, speed_h, speed_v, gradient):
    # Elevation over distance
    # cursor = SnaptoCursor(ax1, s, h[0:-1])
    # plt.connect('motion_notify_event', cursor.mouse_move)
    ax1.plot(s, h[0:-1])
    plt.ylabel("Elevation over distance (m)")
    ax1.grid(True)
    plt.xlim([np.min(s), np.max(s)])
    plt.ylim([np.min(h), np.max(h)])
    
    # Total ascent/descent
    at = AnchoredText("Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh, 1), TotalAscentDescent(dh, -1)),
                  prop=dict(size=12), frameon=True,
                  loc=2,
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)
    
    # Horizontal speed
    ax2.plot(s, speed_h*3.6)
    plt.ylabel("Horizontal speed (km/h)")
    ax2.grid(True)
    plt.ylim([0, 50])
    
    # Vertical speed
    ax3.plot(s, speed_v)
    plt.ylabel("Vertical speed (m/s)")
    ax3.grid(True)
    plt.ylim([-5, 5])
    
    # Gradient
    ax4.plot(s, gradient)
    plt.ylabel("Gradient (m/m)")
    plt.xlabel("Distance (m)")
    ax4.grid(True)
    plt.ylim([-5, 5])
    
    return ax1, ax2, ax3, ax4#, cursor


###############################################################################


print "    _______  _______  ________      __    __  __  ________"
print "   / _____/ / ___  / / ______/      | |  / / / / / ______/"
print "  / / ___  / /__/ / / /_____  _____ | | / / / / / /_____  "
print " / / /  / / ____ / /_____  / /____/ | |/ / / / /_____  /  "
print "/ /__/ / / /      ______/ /         |   / / / ______/ /   "
print "\_____/ /_/      /_______/          |__/ /_/ /_______/    "
print ""


## Arguments
if len(sys.argv) == 2:
    if (sys.argv[1].endswith('.gpx') | sys.argv[1].endswith('.GPX')):
        FILENAME = sys.argv[1]
        print "GPX file to load: %s" % FILENAME
else:
    FILENAME = "original.gpx"
    print "No GPX file provided, loading default: %s" % FILENAME


## Control constants
VERBOSE = False


## Proxy setup
USE_PROXY = False
PROXY_DATA = 'salatis:Alzalarosa01@userproxy.tmg.local:8080'


## Loading .gpx file
gpx, coords = LoadGPX(FILENAME, True)


## "Homemade" processing
lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds, speed_h, speed_v, gradient = RemoveOutliers(coords)
#fig, ax = SummaryPlot(10, s_cleaned, h_cleaned, speed_h, speed_v, gradient)
h_filtered, dh_filtered, gradient_filtered = ElevationFilter(h_cleaned, ds, 7)
#n = 10
#fig = plt.figure(n)
#ax1 = fig.add_subplot(4, 1, 1)
#ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
#ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
#ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
#ax1, ax2, ax3, ax4, cursor = SummaryPlot(ax1, ax2, ax3, ax4,
#                                         s_cleaned,
#                                         h_filtered, dh_filtered,
#                                         speed_h, speed_v_filtered, gradient_filtered,
#                                         lat_cleaned, lon_cleaned)
#
#ax1 = fig.add_subplot(2, 1, 1)
#ax2 = fig.add_subplot(2, 1, 2)
#ax1, ax2, cursor = MapPlot(ax1, ax2,
#                           s_cleaned,
#                           h_filtered, dh_filtered,
#                           speed_h, speed_v_filtered, gradient_filtered,
#                           lat_cleaned, lon_cleaned)
#
#fig = plt.figure(figsize=(20, 8))
#ZOOM_LEVEL = 14
#MARGIN_PERCENTAGE = 0.1
#MapInteractivePlot(fig, s_cleaned, h_filtered, dh_filtered, lat_cleaned, lon_cleaned, ZOOM_LEVEL, MARGIN_PERCENTAGE, USE_PROXY, PROXY_DATA, VERBOSE)


## Kalman smoothing
# https://pykalman.github.io/
# https://github.com/balzer82/Kalman




# coords = coords.resample('1S').asfreq()













#==============================================================================
# ## Plot on OSM in browser
# data = np.ones((len(lat_cleaned),2))
# data[:,0] = h_filtered / np.max(h_filtered) * 0.0004
# data[:,1] = np.hstack((np.asarray([0]), speed_h)) / np.max(np.hstack((np.asarray([0]), speed_h))) * 0.0004
# # PlotOnMap(lat_cleaned, lon_cleaned, data, (0, 1), ('b','r'), 1)
# PlotOnMap(lat_cleaned, lon_cleaned, data, (0, 1), ('blue','red'), 2, s_cleaned, h_filtered, gradient_filtered, speed_h)
#==============================================================================




















RESAMPLE = False
USE_SRTM_ELEVATION = False




# Documentation
# https://pykalman.github.io
# https://github.com/pykalman/pykalman/tree/master/examples/standard
# https://github.com/MathYourLife/Matlab-Tools/commit/246131c02babac27c52fd759ed08c00ae78ba989
# http://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps

# Decide whether to use raw elevation or srtm corrected elevation
if not USE_SRTM_ELEVATION:
    measurements = coords[['lat','lon','ele']].values
else:
    measurements = coords[['lat','lon','srtm']].values

if RESAMPLE:
    # Pre-process coords by resampling and filling the missing values with NaNs
    coords = coords.resample('1S').asfreq()
    # Mask those NaNs
    measurements = np.ma.masked_invalid(measurements)
    
# Setup the Kalman filter & smoother
if not RESAMPLE:
    # The samples are randomly spaced in time, so dt varies with time and a
    # time dependent transition matrix is necessary
    timesteps = np.asarray(coords['time_sec'][1:]) - np.asarray(coords['time_sec'][0:-1])
    transition_matrices = np.zeros(shape = (len(timesteps), 6, 6))
    for i in range(len(timesteps)):
        transition_matrices[i] = np.array([[1, 0, 0, timesteps[i], 0, 0],
                                           [0, 1, 0, 0, timesteps[i], 0],
                                           [0, 0, 1, 0, 0, timesteps[i]],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]])
else:
    # The data have been resampled so there's no need for a time-variant
    # transition matrix
    transition_matrices = np.array([[1, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])

# All the rest isn't influenced by the resampling
observation_matrices = np.array([[1, 0, 0, 0, 0, 0],                           # It's time independent as we only observe the position
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0]])

COORDINATES_COV = 1e-4
ELEVATION_COV = 30
VELOCITY_COV = 1e-6
observation_covariance = np.diag([COORDINATES_COV, COORDINATES_COV, ELEVATION_COV])**2 # Position = 0.0001° = 11.1m, Altitude = 30m

initial_state_mean = np.hstack([measurements[0, :], 3*[0.]])  # Initial position and zero velocity
initial_state_covariance = np.diag([COORDINATES_COV, COORDINATES_COV, ELEVATION_COV,
                                    VELOCITY_COV, VELOCITY_COV, VELOCITY_COV])**2 # Same as the observation covariance

kf = KalmanFilter(transition_matrices = transition_matrices,
                  observation_matrices = observation_matrices,
                  # transition_covariance = transition_covariance,
                  observation_covariance = observation_covariance,
                  # transition_offsets = transition_offsets,
                  # observation_offsets = observation_offsets,
                  initial_state_mean = initial_state_mean,
                  initial_state_covariance = initial_state_covariance,
                  em_vars = ['transition_covariance'])

# Fit the transition covariance matrix
kf = kf.em(measurements, n_iter = 5, em_vars='transition_covariance')

# Smoothing
state_means, state_vars = kf.smooth(measurements)

# Debug plot to see the difference between measured and estimated coordinates
if not RESAMPLE:
    fig_debug, ax_debug = plt.subplots(1,2, sharex = True)
    ax_debug[0].plot(state_means[:,0] - np.asarray(coords['lat']))
    ax_debug[1].plot(state_means[:,1] - np.asarray(coords['lon']))
    ax_debug[0].set_ylim(-1*COORDINATES_COV, +1*COORDINATES_COV)
    ax_debug[1].set_ylim(-1*COORDINATES_COV, +1*COORDINATES_COV)

# Saving to a new coords
k_coords = pd.DataFrame([
              {'lat': state_means[i,0],
               'lon': state_means[i,1],
               'ele': state_means[i,2],
               'time': coords.index[i],
               'time_sec': coords['time_sec'][i]} for i in range(0,len(state_means))])
k_coords.set_index('time', drop = True, inplace = True)

# Plot original/corrected map
fig_map, ax_map = plt.subplots()
ax_map.plot(coords['lon'], coords['lat'], '0.5', linewidth = 2)
ax_map.plot(k_coords['lon'], k_coords['lat'], 'r', linewidth = 2)
mplleaflet.show(fig = ax_map.figure)

# Plot original/corrected altitude profile
fig_alt, ax_alt = plt.subplots()
ax_alt.plot(coords['ele'], '0.5')
ax_alt.plot(k_coords['ele'], 'r')
ax_alt.grid(True)

# Saving to gpx format to take advantage of all the functions provided by gpxpy
k_gpx = gpx
k_segment = k_gpx.tracks[0].segments[0]
for i, p in enumerate(k_segment.points):
    p.speed = None
    p.elevation = k_coords['ele'][i]
    p.longitude = k_coords['lon'][i]
    p.latitude = k_coords['lat'][i]
print k_segment.get_uphill_downhill()
k_gpx.tracks[0].segments[0] = k_segment

# Add speed using embedded function
k_segment.points[0].speed, k_segment.points[-1].speed = 0., 0.
k_gpx.add_missing_speeds()
k_coords['speed'] = [p.speed for p in k_gpx.tracks[0].segments[0].points]

print "\nNEW STATS AFTER KALMAN"
print k_segment.get_uphill_downhill()
print k_segment.get_elevation_extremes()
print k_segment.get_moving_data()
