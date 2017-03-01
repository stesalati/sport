# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
"""
# %matplotlib qt to plot in a different window

# DOCUMENTATION
# https://github.com/FlorianWilhelm/gps_data_with_python
# https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks
# http://nbviewer.jupyter.org/format/slides/github/FlorianWilhelm/gps_data_with_python/blob/master/talk.ipynb#/8/2
# http://www.trackprofiler.com/gpxpy/index.html

import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib import patches
import gpxpy
import datetime
import mplleaflet
import os.path
import folium
from folium import plugins as fp
import webbrowser
import vincent
import json
import sys
from pykalman import KalmanFilter
import srtm
import pandas as pd
import platform

#==============================================================================
# Kalman processing functions
#==============================================================================
def ApplyKalmanFilter(coords, gpx, RESAMPLE, USE_ACCELERATION, PLOT):
    # Documentation
    # https://pykalman.github.io
    # https://github.com/pykalman/pykalman/tree/master/examples/standard
    # https://github.com/MathYourLife/Matlab-Tools/commit/246131c02babac27c52fd759ed08c00ae78ba989
    # http://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps
    # https://github.com/balzer82/Kalman
    
    # Resample is used to artificially increase the number of points and see how
    # the Kalman filter behaves in between breakpoints
    RESAMPLE = False
    USE_ACCELERATION = False
    
    if not RESAMPLE:
        measurements = coords[['lat','lon','ele']].values
    else:
        # Pre-process coords by resampling and filling the missing values with NaNs
        coords = coords.resample('1S').asfreq()
        # Mask those NaNs
        measurements = coords[['lat','lon','ele']].values
        measurements = np.ma.masked_invalid(measurements)
        
    # Setup the Kalman filter & smoother
        
    # Covariances: Position = 0.0001° = 11.1m, Altitude = 30m
    cov = {'coordinates': 1.,
           'elevation': 30.,
           'horizontal_velocity': 1e-4,
           'elevation_velocity': 1e-4,
           'horizontal_acceleration': 1e-6 * 1000,
           'elevation_acceleration': 1e-6 * 1000}
        
    if not USE_ACCELERATION:
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
        observation_matrices = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0]])
        
        observation_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation']])**2
        
        # Initial position and zero velocity
        initial_state_mean = np.hstack([measurements[0, :], 3*[0.]])
        initial_state_covariance = np.diag([cov['coordinates'], cov['coordinates'], cov['elevation'],
                                            cov['horizontal_velocity'], cov['horizontal_velocity'], cov['elevation_velocity']])**2
        
    else:
        # The data have been resampled so there's no need for a time-variant
        # transition matrix
        transition_matrices = np.array([[1, 0, 0, 1, 0, 0, 0.5, 0, 0],
                                        [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
                                        [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
                                        [0, 0, 0, 1, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # All the rest isn't influenced by the resampling
        observation_matrices = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        
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
                      em_vars=['transition_covariance'])
    
    # Fit the transition covariance matrix
    kf = kf.em(measurements, n_iter = 5, em_vars='transition_covariance')
    
    # Smoothing
    state_means, state_vars = kf.smooth(measurements)
    
    if PLOT:
        # Plot original/corrected map
        fig_map, ax_map = plt.subplots()
        ax_map.plot(state_means[:,1], state_means[:,0], color="0.5", linestyle="-", marker="None")
        ax_map.plot(coords['lon'], coords['lat'], color="r", linestyle="None", marker=".")
        # ax_map.legend(['Measured', 'Estimated'])
        ax_map.grid(True)
        # Plot on map only if the number of points is small, otherwise it will not
        # work and it's just betetr to use a normal plot
        if len(state_means) < 2000:
            mplleaflet.show(fig = ax_map.figure)
        
        # Plot original/corrected altitude profile
        fig_alt, ax_alt = plt.subplots()
        #ax_alt.plot(coords['ele'], color='0.5', marker=".")
        ax_alt.plot(measurements[:,2], color="0.5", linestyle="None", marker=".")
        ax_alt.plot(state_means[:,2], color="r", linestyle="-")
        ax_alt.legend(['Measured', 'Estimated'])
        ax_alt.grid(True)
        
        # Stats
        print "Distance: %0.fm" % MyTotalDistance(state_means[:,0], state_means[:,1])
        print "Uphill: %.0fm, Dowhhill: %.0fm" % MyUphillDownhill(state_means[:,2])
        
        # Saving back to the coords dataframe and gpx
        k_coords, k_gpx = SaveDataToCoordsAndGPX(coords, state_means)
    
    return k_coords

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
#    new_gpx = gpx
#    new_segment = new_gpx.tracks[0].segments[0]
#    for i, p in enumerate(new_segment.points):
#        p.speed = None
#        p.elevation = new_coords['ele'][i]
#        p.longitude = new_coords['lon'][i]
#        p.latitude = new_coords['lat'][i]
#    print new_segment.get_uphill_downhill()
    #new_gpx.tracks[0].segments[0] = new_segment
    
    # Add speed using embedded function
#    new_segment.points[0].speed, new_segment.points[-1].speed = 0., 0.
#    new_gpx.add_missing_speeds()
    #new_coords['speed'] = [p.speed for p in new_gpx.tracks[0].segments[0].points]
    
#    print "\nNEW STATS AFTER KALMAN"
#    print new_segment.get_uphill_downhill()
#    print new_segment.get_elevation_extremes()
#    print new_segment.get_moving_data()
    new_gpx = 1
    return new_coords, new_gpx


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

def MyUphillDownhill(ele):
    dele = np.diff(ele)
    return np.sum(dele[np.where(dele > 0)]), np.sum(dele[np.where(dele < 0)])
    
def MyTotalDistance(lat, lon):
    ds = HaversineDistance(lat, lon)
    s = np.sum(ds)
    return s
    
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
        
        # Altitude
        plot_h = {'index': index}
        plot_h['h'] = np.ndarray.tolist(h[1:])               
        line = vincent.Area(plot_h, iter_idx='index')
        line.axis_titles(x='Distance', y='Altitude')
        #line.legend(title = ascent_text)
        line.to_json('plot_h.json')
        # marker_pos1 = [lat[np.where(lat == np.min(lat))], lon[np.where(lon == np.min(lon))]]
        
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
            print "Fullscreen capability not available, try updating Folium"
        
        # Create map
        map_osm.save("osm.html", close_file=False)
        
        if platform.system() == "Darwin":
            # On MAC
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + "osm.html")
        elif platform.system() == 'Windows':
            # On Windows
            webbrowser.open("osm.html", new=2)
        
    return fig


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
    
    print "    _______  _______  ________      __    __  __  ________"
    print "   / _____/ / ___  / / ______/      | |  / / / / / ______/"
    print "  / / ___  / /__/ / / /_____  _____ | | / / / / / /_____  "
    print " / / /  / / ____ / /_____  / /____/ | |/ / / / /_____  /  "
    print "/ /__/ / / /      ______/ /         |   / / / ______/ /   "
    print "\_____/ /_/      /_______/          |__/ /_/ /_______/    "
    print ""
    
    # Arguments
    if len(sys.argv) == 2:
        if (sys.argv[1].endswith('.gpx') | sys.argv[1].endswith('.GPX')):
            FILENAME = sys.argv[1]
            print "GPX file to load: %s" % FILENAME
    else:
        FILENAME = "original.gpx"
        print "No GPX file provided, loading default: %s" % FILENAME
    
    # Control constants
    VERBOSE = False
    
    # Loading .gpx file
    gpx, coords = LoadGPX(FILENAME, False)
    
    #==============================================================================
    # Homemade processing
    #==============================================================================
    if False:
        lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds_cleaned, speed_h, speed_v, gradient = RemoveOutliers(coords, VERBOSE)
        h_filtered, dh_filtered, speed_v_filtered, gradient_filtered = FilterElevation(np.diff(t_cleaned), h_cleaned, ds_cleaned, 7)
        
        fig, ax = plt.subplots(4, 1, sharex=True, squeeze=True)
        ax = PlotSummary(ax, s_cleaned, h_filtered, dh_filtered, speed_h, speed_v_filtered, gradient_filtered)
        
        data = np.ones((len(lat_cleaned),2))
        data[:,0] = h_filtered / np.max(h_filtered) * 0.0004
        data[:,1] = np.hstack((np.asarray([0]), speed_h)) / np.max(np.hstack((np.asarray([0]), speed_h))) * 0.0004
        PlotOnMap(lat_cleaned, lon_cleaned, data, (0, 1), ('blue','red'), 2, s_cleaned, h_filtered, gradient_filtered, speed_h)
    
    #==============================================================================
    # Kalman processing
    #==============================================================================
    if True:
        k_coords = ApplyKalmanFilter(coords, gpx, RESAMPLE=False, USE_ACCELERATION=False, PLOT=True)


if __name__ == "__main__":
    main()
