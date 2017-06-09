#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
@mail: stef.salati@gmail.com
"""

import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib import patches
# from matplotlib.pyplot import ion, show
import re
import gpxpy
import datetime
import mplleaflet
#import os.path
import folium
# from folium import plugins as fp
import webbrowser
import vincent
import json
# import sys
from pykalman import KalmanFilter
import srtm
import pandas as pd
import platform
from rdp import rdp
import scipy.io as sio
import colorsys
from osgeo import gdal#, osr
#import OsmApi
import math
import os
from mayavi import mlab
from PIL import Image
#import vtk
from tvtk.api import tvtk
#from tvtk.common import configure_input
#from traits.api import HasTraits, Instance#, on_trait_change
#from traitsui.api import View, Item
#from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import StringIO
import urllib2#, urllib
#from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#import matplotlib.pyplot as mpld3
#from pylab import imshow, imread, show
import scipy.misc

import gdal_merge as gm

"""
DOCUMENTATION
%matplotlib qt to plot in a different window

https://github.com/FlorianWilhelm/gps_data_with_python
https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks
http://nbviewer.jupyter.org/format/slides/github/FlorianWilhelm/gps_data_with_python/blob/master/talk.ipynb#/8/2
http://www.trackprofiler.com/gpxpy/index.html

https://pykalman.github.io
https://github.com/pykalman/pykalman/tree/master/examples/standard
https://github.com/pykalman/pykalman/blob/master/examples/standard/plot_em.py
https://github.com/pykalman/pykalman/blob/master/examples/standard/plot_missing.py

https://github.com/MathYourLife/Matlab-Tools/commit/246131c02babac27c52fd759ed08c00ae78ba989
http://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps
https://github.com/balzer82/Kalman
"""

#==============================================================================
# Constants
#==============================================================================
PLOT_FONTSIZE = 9 # pt
METHOD_2_MAX_GAP = 2 # seconds
KALMAN_N_ITERATIONS = 5

TRACKS_FOLDER = "tracks/"
MAP_2D_FILENAME = "osm.html"
OSM_DATA_FOLDER = "maps/osm/"
TEXTURE_FILE = OSM_DATA_FOLDER + 'texture.png'
ELEVATION_DATA_FOLDER = "maps/srtm/"
TILES_DOWNLOAD_LINK = "http://dwtkns.com/srtm/"
TRACE_SIZE_ON_3DMAP = 50.0
COORDS_MAPPING_SCALE = 10000
COORDS_MAPPING_ZSCALE = 0.1

DEFAULT_USE_PROXY = False
DEFAULT_PROXY_DATA = 'salatis:Alzalarosa01@userproxy.tmg.local:8080'

#==============================================================================
# Kalman processing functions
#==============================================================================
def ApplyKalmanFilter(coords, gpx, method, use_acceleration, extra_smooth, debug_plot):    
    HTML_FILENAME = "osm_kalman.html"
    dinfos = {}
    
    orig_measurements = coords[['lat','lon','ele']].values
    if method == 0:
        """
        Method 0: just use the data available
        The resulting sampling time is not constant
        """
        # Create the "measurement" array
        measurements = coords[['lat','lon','ele']].values
        dinfos['nsamples'] = "{}".format(len(measurements))
        # This is not necessary here, I just add it so "measurements" is always a masked array,
        # regardless of the method used
        # measurements = np.ma.masked_invalid(measurements)
        
    elif method == 1:
        """
        Method 1: resample at T=1s and fill the missing values with NaNs.
        The resulting sampling time is constant
        """
        coords = coords.resample('1S').asfreq()
        # Create the "measurement" array and mask NaNs
        measurements = coords[['lat','lon','ele']].values
        dinfos['nsamples'] = "{} --> {} (+{:.0f}%)".format(len(orig_measurements), len(measurements), 100 * (float(len(measurements)) - float(len(orig_measurements))) / float(len(orig_measurements)) )
        measurements = np.ma.masked_invalid(measurements)
        
    elif method == 2:
        """
        Method 2: fill the gaps between points close to each other's with NaNs and leave the big holes alone.
        The resulting sampling time is not constant
        """
        for i in range(0,len(coords)-1):
            gap = coords.index[i+1] - coords.index[i]
            if gap <= datetime.timedelta(seconds=METHOD_2_MAX_GAP):
                gap_idx = pd.DatetimeIndex(start=coords.index[i]+datetime.timedelta(seconds=1),
                                           end=coords.index[i+1]-datetime.timedelta(seconds=1),
                                           freq='1S')
                gap_coords = pd.DataFrame(coords, index=gap_idx)
                coords = coords.append(gap_coords)
                # print "Added {} points in between {} and {}".format(len(gap_idx), coords.index[i], coords.index[i+1])
        # Sort all points        
        coords = coords.sort_index()
        # Fill the time_sec column
        for i in range(0,len(coords)):
            coords['time_sec'][i] = (coords.index[i] - datetime.datetime(2000,1,1,0,0,0)).total_seconds()
        coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
        # Create the "measurement" array and mask NaNs
        measurements = coords[['lat','lon','ele']].values
        dinfos['nsamples'] = "{} --> {} (+{:.0f}%)".format(len(orig_measurements), len(measurements), 100 * (float(len(measurements)) - float(len(orig_measurements))) / float(len(orig_measurements)) )
        measurements = np.ma.masked_invalid(measurements)
        
    # Setup the Kalman filter & smoother
        
    # Covariances: Position = 0.0001deg = 11.1m, Altitude = 30m
    cov = {'coordinates': 1.,
           'elevation': 30.,
           'horizontal_velocity': 1e-4,
           'elevation_velocity': 1e-4,
           'horizontal_acceleration': 1e-6 * 1000,
           'elevation_acceleration': 1e-6 * 1000}
        
    if not use_acceleration:
        if method == 1:
            # The data have been resampled so there's no need for a time-variant
            # transition matrix
            c = 1.
            transition_matrices = np.array([[1., 0., 0., c,  0., 0.],
                                            [0., 1., 0., 0., c,  0.],
                                            [0., 0., 1., 0., 0., c ],
                                            [0., 0., 0., 1., 0., 0.],
                                            [0., 0., 0., 0., 1., 0.],
                                            [0., 0., 0., 0., 0., 1.]])
            
        else:
            # The samples are randomly spaced in time, so dt varies with time and a
            # time dependent transition matrix is necessary
            timesteps = np.asarray(coords['time_sec'][1:]) - np.asarray(coords['time_sec'][0:-1])
            transition_matrices = np.zeros(shape = (len(timesteps), 6, 6))
            for i in range(len(timesteps)):
                transition_matrices[i] = np.array([[1., 0., 0., timesteps[i], 0., 0.],
                                                   [0., 1., 0., 0., timesteps[i], 0.],
                                                   [0., 0., 1., 0., 0., timesteps[i]],
                                                   [0., 0., 0., 1., 0., 0.],
                                                   [0., 0., 0., 0., 1., 0.],
                                                   [0., 0., 0., 0., 0., 1.]])
        
        # All the rest isn't influenced by the resampling
        observation_matrices = np.array([[1., 0., 0., 0., 0., 0.],
                                         [0., 1., 0., 0., 0., 0.],
                                         [0., 0., 1., 0., 0., 0.]])
        
        observation_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation']])**2
        
        # Initial position and zero velocity
        initial_state_mean = np.hstack([measurements[0, :], 3*[0.]])
        initial_state_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation'],
                                            cov['horizontal_velocity'], cov['horizontal_velocity'], cov['elevation_velocity']])**2
        
    else:
        if method == 1:
            # The data have been resampled so there's no need for a time-variant
            # transition matrix
            transition_matrices = np.array([[1., 0., 0., 1., 0., 0., 0.5, 0.,  0. ],
                                            [0., 1., 0., 0., 1., 0., 0.,  0.5, 0. ],
                                            [0., 0., 1., 0., 0., 1., 0.,  0.,  0.5],
                                            [0., 0., 0., 1., 0., 0., 1.,  0.,  0. ],
                                            [0., 0., 0., 0., 1., 0., 0.,  1.,  0. ],
                                            [0., 0., 0., 0., 0., 1., 0.,  0.,  1. ],
                                            [0., 0., 0., 0., 0., 0., 1.,  0.,  0. ],
                                            [0., 0., 0., 0., 0., 0., 0.,  1.,  0. ],
                                            [0., 0., 0., 0., 0., 0., 0.,  0.,  1. ]])
        
        else:
            # The samples are randomly spaced in time, so dt varies with time and a
            # time dependent transition matrix is necessary
            timesteps = np.asarray(coords['time_sec'][1:]) - np.asarray(coords['time_sec'][0:-1])
            transition_matrices = np.zeros(shape = (len(timesteps), 9, 9))
            for i in range(len(timesteps)):
                transition_matrices[i] = np.array([[1., 0., 0., timesteps[i], 0., 0., 0.5*(timesteps[i]**2), 0., 0.],
                                                   [0., 1., 0., 0., timesteps[i], 0., 0., 0.5*(timesteps[i]**2), 0.],
                                                   [0., 0., 1., 0., 0., timesteps[i], 0., 0., 0.5*(timesteps[i]**2)],
                                                   [0., 0., 0., 1., 0., 0., timesteps[i], 0., 0.],
                                                   [0., 0., 0., 0., 1., 0., 0., timesteps[i], 0.],
                                                   [0., 0., 0., 0., 0., 1., 0., 0., timesteps[i]],
                                                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        
        # All the rest isn't influenced by the resampling
        observation_matrices = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                         [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        
        observation_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation']])**2
        
        initial_state_mean = np.hstack([measurements[0, :], 6*[0.]])
        initial_state_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation'],
                                            cov['horizontal_velocity'], cov['horizontal_velocity'], cov['elevation_velocity'],
                                            cov['horizontal_acceleration'], cov['horizontal_acceleration'], cov['elevation_acceleration']])**2
    
    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      # transition_covariance=transition_covariance,
                      observation_covariance=observation_covariance,
                      # transition_offsets=transition_offsets,
                      # observation_offsets=observation_offsets,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance,
                      em_vars=['transition_covariance',
                               #'observation_covariance',
                               'transition_offsets',
                               'observation_offsets',
                               #'initial_state_mean',
                               #'initial_state_covariance',
                               ])
        
    # Learn good values for parameters named in `em_vars` using the EM algorithm
#    loglikelihoods = np.zeros(KALMAN_N_ITERATIONS)
#    cacca = np.random.rand(100,3)
#    for i in range(len(loglikelihoods)):
#        kf = kf.em(X=cacca, n_iter=1)
#        loglikelihoods[i] = kf.loglikelihood(cacca)
#        print loglikelihoods[i]
#    print loglikelihoods
    
    # Estimating missing parameters
    kf = kf.em(X=measurements, n_iter=KALMAN_N_ITERATIONS)

    # Smoothing
    state_means, state_vars = kf.smooth(measurements)
    
    """
    # Saving
    sio.savemat("kalman_output.mat", {'state_means':state_means,
                                      'state_vars':state_vars})
    """
    
    # Analize variance and remove points whose variance is too high. It works
    # in principle, but the problem comes when the majority of points that are
    # removed are those that were already masked, that, being artificially
    # added NaNs, the result doesn't change much.
    #variance_coord = np.trace(state_vars[:,:2,:2], axis1=1, axis2=2)
    #variance_ele = state_vars[:,2,2]
        
    """
    COMMENTED AWAITING TO CLARIFY WHAT TO DO HERE
    THRESHOLD_NR_POINTS_TO_RERUN_KALMAN = 10
    idx_var_too_high = np.where( coord_var > (np.mean(coord_var)+2*np.std(coord_var)) )
    infos = infos + "\nANALYZING RESULTING VARIANCE\n"
    infos = infos + "Nr. points with high variance: {}\n".format(len(idx_var_too_high[0]))
    
    if extra_smooth:
        nr_further_points_to_mask = np.count_nonzero(np.logical_not(measurements.mask[idx_var_too_high,0]))
        infos = infos + "Number of real points to be removed: {}\n".format(nr_further_points_to_mask)
        
        if nr_further_points_to_mask > THRESHOLD_NR_POINTS_TO_RERUN_KALMAN:
            # ... then it's worth continuing
            infos = infos + "It's worth smoothing the signal further\n"
            measurements.mask[idx_var_too_high, :] = True
            state_means2, state_vars2 = kf.smooth(measurements) 
            coord_var2 = np.trace(state_vars2[:,:2,:2], axis1=1, axis2=2)
        else:
            # ... then turn off extra_smooth cos it's not worth
            infos = infos + "It's not worth smoothing the signal further\n"
            extra_smooth = False
    """
    
    if debug_plot:
        # Plot original/corrected map
        lat_center = (np.max(state_means[:,0]) + np.min(state_means[:,0])) / 2.
        lon_center = (np.max(state_means[:,1]) + np.min(state_means[:,1])) / 2.
        map_osm = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        map_osm.add_child(folium.PolyLine(orig_measurements[:,:2], 
                                          color='#666666', weight = 4, opacity=1))
        map_osm.add_child(folium.PolyLine(state_means[:,:2], 
                                          color='#FF0000', weight = 4, opacity=1))
        
        # Create and save map
        map_osm.save(HTML_FILENAME, close_file=False)
        if platform.system() == "Darwin":
            # On MAC
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + HTML_FILENAME)
        elif platform.system() == 'Windows':
            # On Windows
            webbrowser.open(HTML_FILENAME, new=2)
            
    return coords, measurements, state_means, state_vars, dinfos

def ComputeDistance(state_means):
    # Horizontal distance
    ddistance = HaversineDistance(np.asarray(state_means[:,0]), np.asarray(state_means[:,1]))
    ddistance = np.hstack(([0.], ddistance))  
    #distance = np.cumsum(ddistance)
    # Vertical distance
    delevation = np.diff(np.asarray(state_means[:,2]))
    delevation = np.hstack(([0.], delevation))    
    # 3d distance
    ddistance3d = np.sqrt(ddistance**2+delevation**2)
    distance3d = np.cumsum(ddistance3d)
    
    #print "Total 2d distance: {}m, 3d distance: {}m".format(np.sum(ddistance), np.sum(ddistance3d))
    return distance3d
    
def SaveDataToCoordsAndGPX(coords, state_means):
    # Saving to a new coords
    new_coords = pd.DataFrame([
                  {'lat': state_means[i,0],
                   'lon': state_means[i,1],
                   'ele': state_means[i,2],
                   'time': coords.index[i],
                   'time_sec': coords['time_sec'][i]} for i in range(0,len(state_means))])
    new_coords.set_index('time', drop = True, inplace = True)
    
    # Saving to gpx format to take advantage of all the functions provided by gpxpy
    new_gpx = gpxpy.gpx.GPX()
    new_gpx.tracks.append(gpxpy.gpx.GPXTrack())
    new_gpx.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())
    for i in range(0, len(new_coords)):
        new_gpx.tracks[0].segments[0].points.append(gpxpy.gpx.GPXTrackPoint(latitude=new_coords['lat'][i],
                                                                            longitude=new_coords['lon'][i],
                                                                            elevation=new_coords['ele'][i],
                                                                            speed=None,
                                                                            time=new_coords.index[i]))
    
    # Alternative method: instead of creating a new gpx object, clone th existing one and fill it with new values.
    # The other is more correct in principle but I couldn't find any documentation to prove that is correct so I prefer
    # to keep also this "dirty" method on record.
    #new_gpx = gpx
    #new_segment = new_gpx.tracks[0].segments[0]
    #for i in range(0, len(k_coords)):
    #    new_segment.points[i].speed = None
    #    new_segment.points[i].elevation = k_coords['ele'][i]
    #    new_segment.points[i].longitude = k_coords['lon'][i]
    #    new_segment.points[i].latitude = k_coords['lat'][i]
    #    new_segment.points[i].time = k_coords.index[i]
    #    new_gpx.tracks[0].segments[0] = new_segment
        
    # Add speed using embedded function
    new_gpx.tracks[0].segments[0].points[0].speed = 0.
    new_gpx.tracks[0].segments[0].points[-1].speed = 0.
    new_gpx.add_missing_speeds()
    
    # Compute stats
    tmp_dinfos1 =  GiveStats(new_gpx.tracks[0].segments[0])
    tmp_dinfos2 =  GiveMyStats(state_means)
    dinfos = MergeDictionaries(tmp_dinfos1, tmp_dinfos2)
    
    return new_coords, new_gpx, dinfos

def PlotElevation(ax, measurements, state_means, clean_before=True, color="#FFAAAA"):    
    # Compute distance
    distance = ComputeDistance(state_means)
    # Clean
    if clean_before:
        ax.cla()
    # Plot
    ax.plot(distance, measurements[:,2], color=color, alpha=0.3, linestyle="None", marker=".")
    ax.plot(distance, state_means[:,2], color=color, linestyle="-", marker="None")
    # Style
    ax.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Elevation (m)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)
    # Legend
    # l = ax.legend(['Measured', 'Estimated'])
    # ltext  = l.get_texts()
    # plt.setp(ltext, fontsize='small')
    return ax, (distance, measurements[:,2])

def PlotElevationVariance(ax, state_means, state_vars):
    # Compute distance
    distance = ComputeDistance(state_means)
    # Compute variance
    variance_ele = state_vars[:,2,2]
    # Clean
    ax.cla()
    # Plot
    ax.plot(distance, variance_ele, color="#FF0000", linestyle="-", marker=".")
    # Style
    ax.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Variance (m)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)    
    return ax, (distance, variance_ele)

def PlotCoordinates(ax, state_means):
    # Compute distance
    distance = ComputeDistance(state_means)
    # Clean and plot
    ax.cla()
    ax.plot(state_means[:,1], state_means[:,0], color="#FF0000", linestyle="-", marker=".")
    # Style
    ax.set_xlabel("Longitude (deg)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Latitude (deg)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)
    return ax, (distance, state_means[:,0], state_means[:,1])

def PlotCoordinatesVariance(ax, state_means, state_vars):
    # Compute distance
    distance = ComputeDistance(state_means)
    # Compute variance
    variance_coord = np.trace(state_vars[:,:2,:2], axis1=1, axis2=2)
    # Clean and plot
    ax.cla()
    ax.plot(distance, variance_coord, color="#FF0000", linestyle="-", marker=".")
    # Style
    ax.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Variance (deg)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)    
    return ax, (distance, variance_coord)

def PlotSpeed(ax, gpx_segment, clean_before=True, color="#FFAAAA"):
    # Compute speed and extract speed from gpx segment
    # (the speed is better this way, as it's computed in 3D and not only 2D, I think)
    coords = pd.DataFrame([
            {'idx': i,
             'lat': p.latitude,
             'lon': p.longitude,
             'ele': p.elevation,
             'speed': p.speed,
             'time': p.time,
             'time_sec': (p.time - datetime.datetime(2000,1,1,0,0,0)).total_seconds()} for i, p in enumerate(gpx_segment.points)])
    coords.set_index('time', drop = True, inplace = True)
    coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
    
    # Compute distance
    distance = np.cumsum(HaversineDistance(np.asarray(coords['lat']), np.asarray(coords['lon'])))
    distance = np.hstack(([0.], distance))
    
    # Clean
    if clean_before:
        ax.cla()
    # Plot
    #ax.plot(distance, measurements[:,2], color="0.5", linestyle="None", marker=".")
    ax.plot(distance, coords['speed']*3.6, color=color, linestyle="-", marker="None")
    # Style
    ax.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Speed (km/h)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)
    # Legend
    # l = ax.legend(['Measured', 'Estimated'])
    # l = ax.legend(['Estimated'])
    # ltext  = l.get_texts()
    # plt.setp(ltext, fontsize='small')
    return ax, (distance, coords['speed']*3.6)

def PlotSpeedVariance(ax, state_means, state_vars):
    # Compute distance
    distance = ComputeDistance(state_means)
    # Compute variance
    variance_speed = np.trace(state_vars[:,3:5,3:5], axis1=1, axis2=2)
    # Clean and plot
    ax.cla()
    ax.plot(distance, variance_speed, color="#FF0000", linestyle="-", marker=".")
    # Style
    ax.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel("Variance (deg/s)", fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
    ax.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    ax.grid(True)    
    return ax, (distance, variance_speed)


#==============================================================================
# Homemade processing functions
#==============================================================================
def RemoveOutliers(coords, VERBOSE):
    # Constants
    # LIMIT_POS_SPEED_H = 12.0
    # LIMIT_NEG_SPEED_H = -12.0
    LIMIT_POS_SPEED_V = 3.0
    LIMIT_NEG_SPEED_V = -3.0
    LIMIT_POS_GRADIENT = 4.0
    LIMIT_NEG_GRADIENT = -4.0
    
    # Renaming variables for ease of use
    lat = coords['lat'].values
    lon = coords['lon'].values
    h = coords['ele'].values
    t = coords['time_sec'].values
    
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
    ds_cleaned = np.asarray(ds_list)
    speed_h = np.asarray(speed_h_list)
    speed_v = np.asarray(speed_v_list)
    gradient = np.asarray(gradient_list)
    s_cleaned = np.cumsum(ds_cleaned)
    
    return lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds_cleaned, speed_h, speed_v, gradient

def FilterElevation(dt, h, ds, window):
    #b = np.sinc(0.25*np.linspace(-np.floor(window/2), np.floor(window/2), num=window, endpoint=True))# * signal.hamming(window)
    b = signal.hann(window)
    b = b/np.sum(b)
    #PlotFilter(b)
    h_filtered = signal.filtfilt(b, 1, h)
    
    # Recomputing speed and gradient (as they depend on h)
    dh_filtered = np.diff(h_filtered)
    speed_v_filtered = dh_filtered/dt
    gradient_filtered = dh_filtered/ds
    
    return h_filtered, dh_filtered, speed_v_filtered, gradient_filtered

def PlotFilterResponse(x):
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
    
def PlotSummary(ax, s, h, dh, speed_h, speed_v, gradient):
    # Elevation over distance
    ax[0].plot(s, h[0:-1])
    ax[0].set_ylabel("Elevation over distance (m)")
    ax[0].grid(True)
    ax[0].set_xlim(np.min(s), np.max(s))
    ax[0].set_ylim(np.min(h), np.max(h))
    ax[0].set_title("SUMMARY")
    
    # Horizontal speed
    ax[1].plot(s, speed_h*3.6)
    ax[1].set_ylabel("Horizontal speed (km/h)")
    ax[1].grid(True)
    ax[1].set_ylim(0, 50)
    
    # Vertical speed
    ax[2].plot(s, speed_v)
    ax[2].set_ylabel("Vertical speed (m/s)")
    ax[2].grid(True)
    ax[2].set_ylim(np.min(speed_v), np.max(speed_v))
    
    # Gradient
    ax[3].plot(s, gradient)
    ax[3].set_ylabel("Gradient (m/m)")
    ax[3].set_xlabel("Distance (m)")
    ax[3].grid(True)
    ax[3].set_ylim(-5, 5)
    
    return ax


#==============================================================================
# Generic functions
#==============================================================================
def LoadGPX(filename):
    gpx_file = open(filename, 'r')
    gpx = gpxpy.parse(gpx_file)

    Nsegments = 0
    id_longest_track = 0
    id_longest_segment = 0
    length_longest_segment = 0
    text = ""
    for itra, track in enumerate(gpx.tracks):
        text = text + "Track {}\n".format(itra)
        # Check if this is the track with more segments
        Nsegments = len(track.segments) if len(track.segments)>Nsegments else Nsegments
        for iseg, segment in enumerate(track.segments):
            # Keep track of the longest segment
            if len(segment.points) > length_longest_segment:
                length_longest_segment = len(segment.points)
                id_longest_track = itra
                id_longest_segment = iseg
            info = segment.get_moving_data()
            text = text + "  Segment {} >>> time: {:.2f}min, distance: {:.0f}m\n".format(iseg, info[0]/60., info[2])
    
    return gpx, (id_longest_track, id_longest_segment), len(gpx.tracks), Nsegments, text

def SelectOneTrackAndSegmentFromGPX(igpx, chosentrack, chosensegment):
    # Create a brand new gpx structure containing only the specified segment
    ogpx = gpxpy.gpx.GPX()
    ogpx.tracks.append(gpxpy.gpx.GPXTrack())
    ogpx.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())
    ogpx.tracks[0].segments[0] = igpx.tracks[chosentrack].segments[chosensegment]
    return ogpx

def MergeAllTracksAndSegmentsFromGPX(igpx):
    # Create a brand new gpx structure containing all the tracks/segments
    ogpx = gpxpy.gpx.GPX()
    ogpx.tracks.append(gpxpy.gpx.GPXTrack())
    ogpx.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())
    
    # Scroll all tracks and segments
    for itra, track in enumerate(igpx.tracks):
        for iseg, segment in enumerate(track.segments):
            # print "T: {}, S: {}, P: {}".format(itra, iseg, len(segment.points))
            for ipoi, point in enumerate(segment.points):
                # Fill the vector
                ogpx.tracks[0].segments[0].points.append(gpxpy.gpx.GPXTrackPoint(latitude=point.latitude,
                                                                                 longitude=point.longitude,
                                                                                 elevation=point.elevation,
                                                                                 speed=None,
                                                                                 time=point.time))
    return ogpx
            
def ParseGPX(gpx, track_nr, segment_nr, use_srtm_elevation):
    segment = gpx.tracks[track_nr].segments[segment_nr]
    
    # Creating a Pandas dataframe with the GPX data.
    # If time is missing (strange, but can happen), just fake it with 1 second per sample starting from 01/01/2000
    # If elevation is missing, set it at 0m
    coords = pd.DataFrame([
            {'idx': i,
             'lat': p.latitude,
             'lon': p.longitude,
             'ele': p.elevation if p.elevation is not None else 0,
             'time': p.time if p.time is not None else (datetime.datetime(2000,1,1,0,0,0) + datetime.timedelta(0,i)),
             'time_sec': (p.time - datetime.datetime(2000,1,1,0,0,0)).total_seconds() if p.time is not None else (datetime.datetime(2000,1,1,0,0,0) + datetime.timedelta(0,i)) } for i, p in enumerate(segment.points)])
    
    coords.set_index('time', drop = True, inplace = True)
    coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
    
    dinfos = GiveStats(segment)
    warnings = ""
    
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
        except:
            warnings = "SRTM correction failed for some reason, probably a shitty proxy.\n"
    
    # Round sampling points at 1s. The error introduced should be negligible
    # the processing would be simplified
    coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
    
    # Add speed using embedded function (it won't be used, it's just to completeness)
    segment.points[0].speed, segment.points[-1].speed = 0., 0.
    gpx.add_missing_speeds()
    coords['speed'] = [p.speed for p in gpx.tracks[track_nr].segments[segment_nr].points]
    
    return gpx, coords, dinfos, warnings

def HaversineDistance(lat_deg, lon_deg):
    # http://www.movable-type.co.uk/scripts/latlong.html
    lat_rad = lat_deg / 360 * 2 * np.pi
    lon_rad = lon_deg / 360 * 2 * np.pi
    dlat_rad = np.diff(lat_rad)
    dlon_rad = np.diff(lon_rad)
    a = np.power(np.sin(dlat_rad/2),2) + np.cos(lat_rad[0:-1]) * np.cos(lat_rad[1:len(lat_rad)]) * np.power(np.sin(dlon_rad/2),2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6371000 * c
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
    
def GiveStats(segment):
    dinfos = {}
    info = segment.get_moving_data()
    m, s = divmod(info[0], 60)
    h, m = divmod(m, 60)
    dinfos['total_distance'] = "{:.3f}km".format((info[2]+info[3])/1000.0)
    dinfos['moving_time'] = "{:2.0f}:{:2.0f}:{:2.0f}".format(h, m, s)
    dinfos['moving_distance'] = "{:.3f}km".format(info[2]/1000.0)
    m, s = divmod(info[1], 60)
    h, m = divmod(m, 60)
    dinfos['idle_time'] = "{:2.0f}:{:2.0f}:{:2.0f}".format(h, m, s)
    dinfos['idle_distance'] = "{:.3f}km".format(info[3]/1000.0)
    
    if segment.has_elevations():
        info = segment.get_elevation_extremes()
        dinfos['elevation'] = "{:.0f}m <-> {:.0f}m".format(info[0], info[1])
    else:
        dinfos['elevation'] = "NA"
    
    if segment.has_elevations():
        info = segment.get_uphill_downhill()
        dinfos['climb'] = "+{:.0f}m, -{:.0f}m".format(info[0], info[1])
    else:
        dinfos['climb'] = "NA"
    
    return dinfos

def GiveMyStats(state_means):
    dinfos = {}
    dinfos['total_distance_my'] = "{:.3f}km".format(ComputeDistance(state_means)[-1]/1000.0)
    delevation = np.diff(np.asarray(state_means[:,2]))
    dinfos['climb_my'] = "+{:.0f}m, {:.0f}m\n".format(np.sum(delevation[np.where(delevation > 0)]), 
                                                      np.sum(delevation[np.where(delevation < 0)]))
    return dinfos

def MergeDictionaries(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

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

def GeneratePalette(N=5):
    SATURATION = 1.0
    VALUE = 1.0
    HSV_tuples = [(x*1.0/N, SATURATION, VALUE) for x in xrange(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        hex_out.append("#" + "".join(map(lambda x: chr(x).encode('hex'),rgb)))
    return hex_out

def PlotOnMap(coords_array_list, coords_array2_list, coords_palette, tangentdata, balloondata_list, rdp_reduction=False, showmap=False):
    """
    Documentation
    https://www.youtube.com/watch?v=BwqBNpzQwJg
    http://matthiaseisen.com/pp/patterns/p0203/
    https://github.com/python-visualization/folium/tree/master/examples
    http://vincent.readthedocs.io/en/latest/quickstart.html
    http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb
    http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Quickstart.ipynb
    Icons: 'ok-sign', 'cloud', 'info-sign', 'remove-sign', http://getbootstrap.com/components/
    
    http://www.digital-geography.com/tag/leaflet/#.WOyjfFWGOUk
    
    3D
    http://nbviewer.jupyter.org/github/ocefpaf/folium_notebooks/blob/a68a89ae28587e6a9fbb21e7a1bd6042183a11bf/test_3D.ipynb
    http://geoexamples.blogspot.de/2014/02/3d-terrain-visualization-with-python.html
    http://www.shadedrelief.com/
    """
    
    # RDP (Ramer Douglas Peucker) reduction
    RDP_EPSILON = 1e-4
    
    # Center coordinates
    center_lat = list()
    center_lon = list()
    
    # Initialize map
    map_osm = folium.Map()
    folium.TileLayer('openstreetmap').add_to(map_osm)
    #folium.TileLayer('stamenterrain').add_to(map_osm)
    
    # See what's in the list
    for icoords_array, coords_array in enumerate(coords_array_list):
        
        # Unpacking coordinates
        lat = coords_array[:,0]
        lon = coords_array[:,1]
        if coords_array2_list is not None:
            coords_array2 = coords_array2_list[icoords_array]
        else:
            coords_array2 = None
            
        # The center of this track
        center_lat.append((np.max(lat) + np.min(lat)) / 2)
        center_lon.append((np.max(lon) + np.min(lon)) / 2)
        
        # Prepare extra data to be plotted along the trace
        if tangentdata is not None:
            # Unpacking tangentdata
            data = tangentdata['data']
            sides = tangentdata['sides']
            palette = tangentdata['palette']
            # Determine the perpendicular axis for each pair of consecutive points
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
        
        # Balloon plots (made with Vincent)
        if balloondata_list is not None:
            balloondata = balloondata_list[icoords_array]
            index = np.ndarray.tolist(balloondata['distance'])
                
            # Altitude, also used to plot the highest elevation marker
            if balloondata['elevation'] is not None:
                plot_h = {'index': index}
                plot_h['h'] = np.ndarray.tolist(balloondata['elevation'][1:])               
                line = vincent.Area(plot_h, iter_idx='index')
                line.axis_titles(x='Distance', y='Altitude')
                line.to_json('plot_h.json')
                # marker_pos1 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))]]
                
                marker_highest_point = np.where(balloondata['elevation'] == np.max(balloondata['elevation']))[0][0]
                if marker_highest_point == 0:
                    marker_highest_point = 10
                if marker_highest_point == len(balloondata['elevation']):
                    marker_highest_point = len(balloondata['elevation']) - 10
                
                highest_point_popup = folium.Popup(max_width = 1200).add_child(
                                        folium.Vega(json.load(open('plot_h.json')), width = 1000, height = 550))
                map_osm.add_child(folium.Marker([lat[marker_highest_point], lon[marker_highest_point]], 
                                                   # popup = "Highest point",
                                                   popup = highest_point_popup,
                                                   icon=folium.Icon(icon='cloud')))
            
            # Speed_h
            if balloondata['speed'] is not None:
                plot_speed_h = {'index': index}
                plot_speed_h['speed_h'] = np.ndarray.tolist(balloondata['speed'])
                line = vincent.Line(plot_speed_h, iter_idx='index')
                line.axis_titles(x='Distance', y='Altitude')
                line.to_json('plot_speed_h.json')
                # marker_pos3 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))] + 0.02 * (np.max(lon) - np.min(lon))]
                
                #folium.RegularPolygonMarker(
                #    location = marker_location_speed_h,
                #    fill_color = '#FF0000',
                #    radius = 12,
                #    number_of_sides = 3,
                #    popup=folium.Popup(max_width = 1000).add_child(
                #        folium.Vega(json.load(open('plot_speed_h.json')), width = 1000, height = 250))
                #).add_to(map_osm)
            
        # Extra data along the tracks
        if tangentdata is not None:
            fig, ax = plt.subplots()
            # Method 1: Create patches the mplleaflet way, one for every data we want to plot
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
                
            # Method 2: Polygon with JSON (not working, dunno why)
            # a = [[[27, 43], [33, 43], [33, 47], [27, 47]]]
            # a = [np.ndarray.tolist(np.vstack((tmp_poly_lat, tmp_poly_lon)).T)]
            # gj_poly = folium.GeoJson(data={"type": "Polygon", "coordinates": a})
            # gj_poly.add_to(map_osm)
            
            # Method 3: Folium polygon maker (the simplest, but not supported by the current version of Folium, only by the dev version)
            #folium.features.PolygonMarker(
            #    np.vstack((tmp_poly_lat, tmp_poly_lon)).T,
            #    color='blue',
            #    weight=10,
            #    fill_color='red',
            #    fill_opacity=0.5,
            #    popup='Tokyo, Japan').add_to(map_osm)
    
        # Plot tracks
        if rdp_reduction:
            if tangentdata is not None:
                print "\nWARNING: RDP reduction activated with tangentdata, trace/polygons misallignments are possible"
            coords_array = rdp(coords_array, RDP_EPSILON)
        # Plot first the 2nd trace, if available, so it stays on the background
        if coords_array2 is not None:
            map_osm.add_child(folium.PolyLine(coords_array2, color='#444444', weight = 4, opacity=1.0))
        map_osm.add_child(folium.PolyLine(coords_array, color=coords_palette[icoords_array], weight = 4, opacity=1.0))
        
        # Plot start/finish markers
        map_osm.add_child(folium.Marker([lat[0], lon[0]],
                                        popup = "Start",
                                        icon=folium.Icon(color='green', icon='circle-arrow-up')))
        map_osm.add_child(folium.Marker([lat[-1], lon[-1]], 
                                        popup = "Finish",
                                        icon=folium.Icon(color='red', icon='circle-arrow-down')))

    # Center map
    map_osm.location = [np.mean(np.asarray(center_lat)), np.mean(np.asarray(center_lon))]
    map_osm.zoom_start = 12
            
    # Create and save map
    folium.LayerControl().add_to(map_osm)
    map_osm.save(MAP_2D_FILENAME, close_file=False)
    
    # Open map in external browser
    if showmap:
        if platform.system() == "Darwin":
            # On MAC
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + MAP_2D_FILENAME)
        elif platform.system() == 'Windows':
            # On Windows
            webbrowser.open(MAP_2D_FILENAME, new=2)
        
    return

"""
Coordinate conversion

http://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/
https://glenbambrick.com/2015/06/24/dd-to-dms/
http://www.earthpoint.us/Convert.aspx
http://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/coordinates.html
"""
def dms2dd(degrees, minutes, seconds, direction):
    dd = degrees + minutes/60 + seconds/(60*60);
    if direction == 'S' or direction == 'W':
        dd *= -1
    return dd;

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms):
    parts = re.split('[^\d\w]+', dms)
    if len(parts) == 8:
        # Seconds are without decimals
        lat = dms2dd(float(parts[0]), float(parts[1]), float(parts[2]), parts[3])
        lng = dms2dd(float(parts[4]), float(parts[5]), float(parts[6]), parts[7])
    if len(parts) == 10:
        # Seconds are with decimals, distinguish if they're 0 or actually meaningful
        if float(parts[3]) > 0:
            lat = dms2dd(float(parts[0]), float(parts[1]), float(parts[2]) + float(parts[3])/(10**(np.trunc(np.log10(float(parts[3]))+1))), parts[4])
        else:
            lat = dms2dd(float(parts[0]), float(parts[1]), float(parts[2]), parts[4])
            
        if float(parts[8]) > 0:
            lng = dms2dd(float(parts[5]), float(parts[6]), float(parts[7]) + float(parts[8])/(10**(np.trunc(np.log10(float(parts[8]))+1))), parts[9])
        else:
            lng = dms2dd(float(parts[5]), float(parts[6]), float(parts[7]), parts[9])
    return (lat, lng)

"""
3D Mapping

Example
http://www.pyvotons.org/?page_id=183
https://conference.scipy.org/SciPy2008/static/wiki/mayavi_tutorial_scipy08.pdf

Mayavi
http://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html
http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
http://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html
http://docs.enthought.com/mayavi/mayavi/auto/mlab_other_functions.html

Tile download
http://dwtkns.com/srtm/
http://www.pyvotons.org/?page_id=178
http://srtm.csi.cgiar.org/SELECTION/inputCoord.asp

Iceland
http://www.eea.europa.eu/data-and-maps/data/eu-dem
https://www.planetside.co.uk/forums/index.php?topic=19089.0
http://wiki.openstreetmap.org/wiki/Contours_for_Iceland
http://viewfinderpanoramas.org/dem3.html
https://gdex.cr.usgs.gov/gdex/

Altri metodi per scaricare l'altitudine (scartati per vari motivi)
http://vterrain.org/Elevation/global.html
https://pypi.python.org/pypi/elevation
http://elevation.bopen.eu/en/stable/
https://pypi.python.org/pypi/py-altimetry/0.3.1
https://algorithmia.com/algorithms/Gaploid/Elevation -> a pagamento
http://stackoverflow.com/questions/11504444/raster-how-to-get-elevation-at-lat-long-using-python
http://gis.stackexchange.com/questions/59316/python-script-for-getting-elevation-difference-between-two-points
"""
def GetOSMImageCluster(lat_deg, lon_deg, delta_lat, delta_long, zoom=13, use_proxy=DEFAULT_USE_PROXY, proxy_data=DEFAULT_PROXY_DATA, verbose=False):
    
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
    
    warnings = ""
    
    # Proxy setup with urllib2
    if use_proxy:
        proxy = urllib2.ProxyHandler({'http': proxy_data})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
    
    # Tile file name
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
                if os.path.isfile(OSM_DATA_FOLDER + savename.format(zoom, xtile, ytile)):
                    tile = Image.open(OSM_DATA_FOLDER + savename.format(zoom, xtile, ytile))
                else:
                    # Download from the Internet and save it locally for future
                    # use
                    imgurl = smurl.format(zoom, xtile, ytile)
                    warnings = warnings + "OSM tile not found locally, downloading it from {} \n".format(imgurl)
                    print "OSM tile not found locally, downloading it from {} ".format(imgurl)
                    imgstr = urllib2.urlopen(imgurl).read()
                    tile = Image.open(StringIO.StringIO(imgstr))
                    with open(OSM_DATA_FOLDER + savename.format(zoom, xtile, ytile), 'wb') as f:
                        f.write(imgstr)
                        f.close()
                # Append it to the rest of the cluster
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except:
                warnings = warnings + "OSM tile loading failed!\n"
                tile = None
                
    return Cluster, osm_tiles_edges, warnings


def GetGeoTIFFImageCluster(lat_min, lat_max, lon_min, lon_max, tile_selection='auto', margin=100, verbose=False):
    
    def SRTMTile(lat, lon):
        xtile = int(np.trunc((lon - (-180)) / (360/72) + 1))
        ytile = int(np.trunc((60 - lat) / (360/72) + 1))
        return (xtile, ytile)
    
    warnings = ""
    
    if tile_selection == 'auto':
        # Tiles will be determined automatically
        
        # Determine which tiles are necessary
        tile_corner_min = SRTMTile(lat_min, lon_min)
        tile_corner_max = SRTMTile(lat_max, lon_max)
        tiles_x = range(tile_corner_min[0], tile_corner_max[0]+1)
        tiles_y = range(tile_corner_max[1], tile_corner_min[1]+1) # Inverted min and max as tiles are numbered, vertically, from north to south 
        
        if verbose:
            print "Required elevation tiles:"
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
                        filename = ELEVATION_DATA_FOLDER + "{}.tif".format(tilename)
                        gdal_merge_command_list.append(filename)
                        if not os.path.isfile(filename):
                            warnings = warnings + "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.\n".format(tilename, TILES_DOWNLOAD_LINK)
                            print "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.\n".format(tilename, TILES_DOWNLOAD_LINK)
                            return None, None, None, None, None, None, None, warnings
                if verbose:
                    print "A tile mosaic is required: this merge command will be run: {}".format(gdal_merge_command_list)
                gm.main(gdal_merge_command_list)
            filename = ELEVATION_DATA_FOLDER + merged_tile_name
        else:
            # Only one tile is needed
            tilename = "srtm_{:02d}_{:02d}".format(tiles_x[0], tiles_y[0])
            filename = ELEVATION_DATA_FOLDER + "{}.tif".format(tilename)
            if not os.path.isfile(filename):
                warnings = warnings + "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.\n".format(tilename, TILES_DOWNLOAD_LINK)
                print "Error: Elevation profile for this location ({}) not found. It can be donwloaded here: {}.\n".format(tilename, TILES_DOWNLOAD_LINK)
                return None, None, None, None, None, None, None, warnings
                
    else:
        # The tile name is provided (useful for those areas, e.g. Iceland, not covered by the SRTM survey)
        filename = ELEVATION_DATA_FOLDER + tile_selection
    
    # Read GeoTiff elevation file 
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
    # Vertically inverted min and max as tiles are numbered, vertically, from north to south       
    zone = {'x_min': int(np.round( (lon_min - tile_lon_min) / gt[1] )),
            'x_size': int(np.round( (lon_max - lon_min) / gt[1] )),
            'y_min': int(np.round( (lat_max - tile_lat_max) / gt[5] )),
            'y_size': int(np.round( (lat_min - lat_max) / gt[5] ))}
    
    if verbose:
        print "\nSelected elevation zone:"
        print "Longitude (X): Start: {}, Size: {}".format(zone['x_min'], zone['x_size'])
        print "Latitude (Y):  Start: {}, Size: {}".format(zone['y_min'], zone['y_size'])
    
    # Read elevation data
    array_ele = tile_ele.ReadAsArray(zone['x_min'], zone['y_min'], zone['x_size'], zone['y_size']).astype(np.float)
    
    # Set sea level at 0m instead of -32768 (Dead Sea level used as minimum value)
    array_ele[array_ele < -418] = 0
    
    # Create X,Y coordinates for zone_ele array
    line_x_deg = np.arange(tile_lon_min + zone['x_min'] * gt[1], tile_lon_min + (zone['x_min'] + zone['x_size']) * gt[1], gt[1])[0:zone['x_size']]
    array_x_deg = np.tile(line_x_deg, (len(array_ele), 1)).transpose()
    line_y_deg = np.arange(tile_lat_max + zone['y_min'] * gt[5], tile_lat_max + (zone['y_min'] + zone['y_size']) * gt[5], gt[5])[0:zone['y_size']]
    array_y_deg = np.tile(line_y_deg, (len(array_ele[0]), 1))
    """
    # In theory, that's the same as this, but the rounding issues are less with the above implementation
    line_x_deg = np.arange(lon_min, lon_max, gt[1])[0:zone['x_size']]
    array_x_deg = np.tile(line_x_deg, (len(array_ele), 1)).transpose()
    line_y_deg = np.arange(lat_max, lat_min, gt[5])[0:zone['y_size']]
    array_y_deg = np.tile(line_y_deg, (len(array_ele[0]), 1))
    """
    
    return zone, line_x_deg, array_x_deg, line_y_deg, array_y_deg, array_ele, gt, warnings
    

def Generate3DMap(track_lat, track_lon,
                  tile_selection='auto',
                  margin=100,
                  elevation_scale=1.0,
                  mapping='coords',
                  use_osm_texture=True, texture_type='osm', texture_zoom=13, texture_invert=False,
                  use_proxy=DEFAULT_USE_PROXY, proxy_data=DEFAULT_PROXY_DATA,
                  verbose=False):
    
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
    warnings = ""
    
    # If track_lat and track_lon are None, run a demo
    if len(track_lat) == 0 or len(track_lon) == 0:
        # startingpoint = (44.1938472, 10.7012833)    # Cimone
        # startingpoint = (46.5145639, 11.7398472)    # Rif. Demetz
        startingpoint = (-08.4113472, 116.4166667)    # Rinjani
        # startingpoint = (64.0158333, -016.6747222)  # Peak in Iceland
        
        # Circle
        R = 0.01
        track_lat1 = np.linspace(-R, R, 1000).transpose()
        track_lon1 = np.sqrt(R**2 - track_lat1[0:1000]**2)
        track_lat2 = np.linspace(R, -R, 1000).transpose()
        track_lon2 = -np.sqrt(R**2 - track_lat2[0:1000]**2)
        track_lat = np.hstack((track_lat1[0:-2], track_lat2))
        track_lon = np.hstack((track_lon1[0:-2], track_lon2))
        track_lat = track_lat + startingpoint[0]
        track_lon = track_lon + startingpoint[1]
        
        """
        # Dot
        track_lat = [startingpoint[0]]
        track_lon = [startingpoint[1]]
        """
        
    lat_min = np.min(track_lat) - margin * px2deg
    lat_max = np.max(track_lat) + margin * px2deg
    lon_min = np.min(track_lon) - margin * px2deg
    lon_max = np.max(track_lon) + margin * px2deg
    
    # Get GeoTIFF elevation data
    zone, line_x_deg, array_x_deg, line_y_deg, array_y_deg, array_z, gt, geotiff_warnings = GetGeoTIFFImageCluster(lat_min=lat_min,
                                                                                                                   lat_max=lat_max,
                                                                                                                   lon_min=lon_min,
                                                                                                                   lon_max=lon_max,
                                                                                                                   tile_selection=tile_selection,
                                                                                                                   margin=margin,
                                                                                                                   verbose=verbose)
    warnings = warnings + geotiff_warnings
    
    # Check if GeoTIFF data were generated correctly, otherwise just return 
    if zone is None:
        return None, None, warnings
    
    # Create figure
    #fig = mlab.figure(figure='3dmap', size=(500, 500))
    
    # Display 3D elevation, depending on the plot type specified
    if mapping == 'meters':
        # Convert the coordinates in meters
        array_x_m = np.empty_like(array_x_deg)
        for x, y in np.ndindex(array_x_deg.shape):
            array_x_m[x,y] = degrees2metersLongX(line_y_deg[y], array_x_deg[x,y])
            
        line_y_m = np.array([degrees2metersLatY(j) for j in line_y_deg])
        array_y_m = np.tile(line_y_m, (len(array_z[0]), 1))
        
        array_z = array_z.transpose()
    
    if mapping == 'coords':
        array_z = array_z.transpose()
        
        # OSM texture
        if use_osm_texture:
            
            # Create the texture
            a, osm_tiles_edges, osm_warnings = GetOSMImageCluster(lat_deg=lat_min, lon_deg=lon_min,
                                                                  delta_lat=(lat_max-lat_min), delta_long=(lon_max-lon_min),
                                                                  zoom=texture_zoom,
                                                                  use_proxy=use_proxy, proxy_data=proxy_data,
                                                                  verbose=verbose)
            warnings = warnings + osm_warnings
            
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
            if verbose:
                print("\nHow much needs to be trimmed")
                print "Longitude (X): {} <-- {} -- {} --> {}".format(osm_tiles_edges['lon_min'], lon_min, lon_max, osm_tiles_edges['lon_max'])
                print "Latitude (Y):  {} <-- {} -- {} --> {}".format(osm_tiles_edges['lat_min'], lat_min, lat_max, osm_tiles_edges['lat_max'])
            
            height = a.size[1]
            width = a.size[0]
            
            # Method 1: with vectors
            """
            h_coord_vector = np.linspace(osm_tiles_edges['lon_min'], osm_tiles_edges['lon_max'], width)
            h_min_idx, h_min_value = find_nearest(h_coord_vector, lon_min)
            h_max_idx, h_max_value = find_nearest(h_coord_vector, lon_max)
            
            v_coord_vector = np.linspace(osm_tiles_edges['lat_max'], osm_tiles_edges['lat_min'], height)
            v_min_idx, v_min_value = find_nearest(v_coord_vector, lat_min)
            v_max_idx, v_max_value = find_nearest(v_coord_vector, lat_max)
            
            trim_margins = {'left': int(h_min_idx),
                            'right': int(h_max_idx),
                            'bottom': int(v_min_idx),
                            'top': int(v_max_idx)}
            """
            # Method 2: operations
            h_deg2px_ratio = width / (osm_tiles_edges['lon_max'] - osm_tiles_edges['lon_min'])
            v_deg2px_ratio = height / (osm_tiles_edges['lat_max'] - osm_tiles_edges['lat_min'])
            
            trim_margins = {'left': int(np.round((lon_min - osm_tiles_edges['lon_min']) * h_deg2px_ratio)),
                            'right': int(np.round( width - (osm_tiles_edges['lon_max'] - lon_max) * h_deg2px_ratio )),
                            'bottom': int(np.round( height - (lat_min - osm_tiles_edges['lat_min']) * v_deg2px_ratio )),
                            'top': int(np.round((osm_tiles_edges['lat_max'] - lat_max) * v_deg2px_ratio))}
            # print "Left: {}\nRight: {}\nBottom: {}\nTop: {}".format(trim_margins['left'], trim_margins['right'], trim_margins['bottom'], trim_margins['top'])
            
            # Shifting the texture, conceptually not right
            """
            h_shift = 7
            v_shift = 12
            trim_margins['left'] = trim_margins['left'] - h_shift
            trim_margins['right'] = trim_margins['right'] - h_shift
            trim_margins['top'] = trim_margins['top'] - v_shift
            trim_margins['bottom'] = trim_margins['bottom'] - v_shift
            """
            # print "Horizontal shift: {}deg".format(h_shift / h_deg2px_ratio)
            # print "Vertical shift: {}deg".format(v_shift / v_deg2px_ratio)
                        
            a_trimmed = a.crop((trim_margins['left'], trim_margins['top'], trim_margins['right'], trim_margins['bottom']))
            
            # BUG TODO
            # The image is processed and saved correctly but sometimes it's displayed
            # transposed and rotated. This is pretty strange as it depends on the size of
            # the image, with a margin on 300 it's fine, below is loaded weirdly.
            # A possible solution is to treat the image before saving it. However, a better
            # solution needs to be found as in any case this is not an image processing
            # problem but a loading and texturing problem.
            #if margin < 235:
            if texture_invert:
                a_trimmed = a_trimmed.transpose(Image.TRANSPOSE)
                a_trimmed = a_trimmed.rotate(180)
            
            # Save the texture as a PNG
            img = np.asarray(a_trimmed)
            scipy.misc.imsave(TEXTURE_FILE, img)
        
    # Track
    track_x_m = list()
    track_y_m = list()
    track_z_m = list()
    track_x_deg = list()
    track_y_deg = list()
    track_z_deg = list()
    for i in range(np.size(track_lat, axis=0)):
        (x,y) = degrees2meters(track_lon[i], track_lat[i])
        track_x_m.append(x)
        track_y_m.append(y)
        track_x_deg.append(track_lon[i] * COORDS_MAPPING_SCALE)
        track_y_deg.append(track_lat[i] * COORDS_MAPPING_SCALE)
        zz = array_z[int(np.round((track_lon[i] - (gt[0]+zone['x_min']*gt[1])) / gt[1])), int(np.round((track_lat[i] - (gt[3]+zone['y_min']*gt[5])) / gt[5]))]
        track_z_m.append(zz * elevation_scale)
        track_z_deg.append(zz * COORDS_MAPPING_ZSCALE * elevation_scale)
       
    # Creating the export dictionaries
    if mapping == 'meters':
        terrain = {'x': array_x_m, 
                   'y': array_y_m,
                   'z': array_z * elevation_scale}
        track = {'x': track_x_m,
                 'y': track_y_m,
                 'z': track_z_m,
                 'color': (255.0/255.0, 102.0/255.0, 0),
                 'line_width': 10.0,
                 'line_radius': TRACE_SIZE_ON_3DMAP,
                 'textsize': textsize}
        
    if mapping == 'coords':
        terrain = {'x': array_x_deg * COORDS_MAPPING_SCALE, 
                   'y': array_y_deg * COORDS_MAPPING_SCALE,
                   'z': array_z * elevation_scale * COORDS_MAPPING_ZSCALE}
        track = {'x': track_x_deg,
                 'y': track_y_deg,
                 'z': track_z_deg,
                 'color': (255.0/255.0, 102.0/255.0, 0),
                 'line_width': 10.0,
                 'line_radius': TRACE_SIZE_ON_3DMAP / 20,
                 'textsize': textsize / 20}
        
    return terrain, track, warnings


def Plot3DMap(terrain, track, use_osm_texture, animated=False):
    fig = mlab.figure(figure='3D Map', size=(500, 500))
    
    # Plot the elevation mesh
    elevation_mesh = mlab.mesh(terrain['x'],
                               terrain['y'],
                               terrain['z'],
                               figure=fig)
    
    # Read and apply texture
    if use_osm_texture:
        bmp = tvtk.PNGReader(file_name=TEXTURE_FILE)
        texture = tvtk.Texture(input_connection=bmp.output_port, interpolate=1)
        elevation_mesh.actor.actor.mapper.scalar_visibility = False
        elevation_mesh.actor.enable_texture = True
        elevation_mesh.actor.tcoord_generator_mode = 'plane'
        elevation_mesh.actor.actor.texture = texture
    
    # Display path nodes
    if len(track['x']) == 1:
        track_line = mlab.points3d(track['x'], track['y'], track['z'],
                                   figure=fig,
                                   color=track['color'], mode='sphere', scale_factor=track['line_radius']*10)
    else:
        track_line = mlab.plot3d(track['x'], track['y'], track['z'],
                                 figure=fig,
                                 color=track['color'], line_width=10.0, tube_radius=track['line_radius'])
    
    # Display north text
    north_label = mlab.text3d((terrain['x'][0][0] + terrain['x'][-1][0]) / 2,
                              terrain['y'][0][0],
                              np.max(terrain['z']),
                              "NORTH",
                              figure=fig,
                              scale=(track['textsize'], track['textsize'], track['textsize']))
    
    # Displaying start test
    if len(track['x']) > 1:
        start_label = mlab.text3d(track['x'][0],
                                  track['y'][0],
                                  track['z'][0] * 1.5,
                                  "START",
                                  figure=fig,
                                  scale=(track['textsize'], track['textsize'], track['textsize']))
    
    # Set camera position
    mlab.view(azimuth=-90.0,
              elevation=60.0,
              # distance=1.0,
              distance='auto',
              # focalpoint=(1000.0, 1000.0, 1000.0),
              focalpoint='auto',
              roll=0.0,
              figure=fig)
    
    # Show the 3D map
    mlab.show()
    
    """
    if animated:
        
        @mlab.animate(delay=1000, ui=True)
        def anim():
            while 1:
                fig.scene.camera.azimuth(10)
                fig.scene.render()
                yield
                
        a = anim() # Starts the animation without a UI.
    """
    
    # Returning generated elements for future use
    map_elements = {'figure': fig,
                    'elevation_mesh': elevation_mesh,
                    'track_line': track_line,
                    'north_label': north_label,
                    'start_label': start_label}
    
    return map_elements


"""
Homemade processing
if False:
    lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds_cleaned, speed_h, speed_v, gradient = RemoveOutliers(coords, VERBOSE=False)
    h_filtered, dh_filtered, speed_v_filtered, gradient_filtered = FilterElevation(np.diff(t_cleaned), h_cleaned, ds_cleaned, 7)
    
    fig, ax = plt.subplots(4, 1, sharex=True, squeeze=True)
    ax = PlotSummary(ax, s_cleaned, h_filtered, dh_filtered, speed_h, speed_v_filtered, gradient_filtered)
"""
