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
from matplotlib.pyplot import ion, show
import gpxpy
import datetime
import mplleaflet
import os.path
import folium
# from folium import plugins as fp
import webbrowser
import vincent
import json
import sys
from pykalman import KalmanFilter
import srtm
import pandas as pd
import platform
from rdp import rdp

#==============================================================================
# Kalman processing functions
#==============================================================================
def ApplyKalmanFilter(coords, gpx, method, use_acceleration, variance_smooth, plot, usehtml):
    # Documentation
    # https://pykalman.github.io
    # https://github.com/pykalman/pykalman/tree/master/examples/standard
    # https://github.com/MathYourLife/Matlab-Tools/commit/246131c02babac27c52fd759ed08c00ae78ba989
    # http://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps
    # https://github.com/balzer82/Kalman
    
    # Methods:
    # The only measninful ones are 0 and 1, better 0 with enough points. All
    # others are just tries
    
    HTML_FILENAME = "osm_kalman.html"
    infos = ""
    
    # Method is used to artificially increase the number of points and see how
    # the Kalman filter behaves in between breakpoints
    orig_measurements = coords[['lat','lon','ele']].values
    if not method:
        # Method 0: just use the data available
        # Create the "measurement" array
        measurements = coords[['lat','lon','ele']].values
        infos = infos + "Number of samples: {}".format(len(measurements))
        infos = infos + ("<br>" if usehtml else "\n")
        
    elif method == 1:
        # Method 1: resample at T=1s and fill the missing values with NaNs
        coords = coords.resample('1S').asfreq()
        # Create the "measurement" array and mask NaNs
        measurements = coords[['lat','lon','ele']].values
        infos = infos + "Number of samples: {} --> {} (+{:.0f}%)".format(len(orig_measurements), len(measurements), 100 * (float(len(measurements)) - float(len(orig_measurements))) / float(len(orig_measurements)) )
        infos = infos + ("<br>" if usehtml else "\n")
        measurements = np.ma.masked_invalid(measurements)
        
    elif method == 2:
        # Method 2: resample but fill NaNs with interpolation, there's no need
        # to mask values this way
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
        coords = coords.resample('1S').asfreq()
        coords = coords.interpolate(method='linear')
        # Create the "measurement" array
        measurements = coords[['lat','lon','ele']].values
        infos = infos + "Number of samples: {} --> {} (+{:.0f}%)".format(len(orig_measurements), len(measurements), 100 * (float(len(measurements)) - float(len(orig_measurements))) / float(len(orig_measurements)) )
        infos = infos + ("<br>" if usehtml else "\n")
        
    elif method == 3:
        # Method 3: fill the gaps between points too far away with NaNs. It's
        # different from resampling as T is not constant
        for i in range(0,len(coords)-1):
            gap = coords.index[i+1] - coords.index[i]
            GAP_CADENCE = 10 # seconds
            if gap >= datetime.timedelta(seconds=GAP_CADENCE * 2):
                gap_idx = pd.DatetimeIndex(start=coords.index[i]+datetime.timedelta(seconds=GAP_CADENCE),
                                           end=coords.index[i+1]-datetime.timedelta(seconds=2),
                                           freq='10S')
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
        infos = infos + "Number of samples: {} --> {} (+{:.0f}%)".format(len(orig_measurements), len(measurements), 100 * (float(len(measurements)) - float(len(orig_measurements))) / float(len(orig_measurements)) )
        infos = infos + ("<br>" if usehtml else "\n")
        measurements = np.ma.masked_invalid(measurements)
        
    # Setup the Kalman filter & smoother
        
    # Covariances: Position = 0.0001° = 11.1m, Altitude = 30m
    cov = {'coordinates': 1.,
           'elevation': 30.,
           'horizontal_velocity': 1e-4,
           'elevation_velocity': 1e-4,
           'horizontal_acceleration': 1e-6 * 1000,
           'elevation_acceleration': 1e-6 * 1000}
        
    if not use_acceleration:
        if method == 0 or method == 3:
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
        else:
            # The data have been resampled so there's no need for a time-variant
            # transition matrix
            c = 1.
            transition_matrices = np.array([[1., 0., 0., c,  0., 0.],
                                            [0., 1., 0., 0., c,  0.],
                                            [0., 0., 1., 0., 0., c ],
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
        if method == 0:
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
        
        elif method == 1:
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
            print "ERROR: Cannot use acceleration with methods other than 0 or 1"
        
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
                      em_vars=['transition_covariance'])
    
    # Fit the transition covariance matrix
    kf = kf.em(measurements, n_iter = 5, em_vars='transition_covariance')
    
    # Smoothing
    state_means, state_vars = kf.smooth(measurements)
    
    # Analize variance and remove points whose variance is too high. It works
    # in principle, but the problem comes when the majority of points that are
    # removed are those that were already masked, that, being artificially
    # added NaNs, the result doesn't change much.
    THRESHOLD_NR_POINTS_TO_RERUN_KALMAN = 10
    if variance_smooth:
        coord_var = np.trace(state_vars[:,:2,:2], axis1=1, axis2=2)
        coord_var_r = np.median(np.sort(coord_var)[:-20:-1]) / np.mean(coord_var)
        print "\nANALYZING RESULTING VARIANCE"
        print "Ratio between mean_variance_top20/median_variance: {}".format(coord_var_r)
        idx_var_too_high = np.where(coord_var > (10 * np.median(coord_var)))
        print "Number of points whose variance is considered too high: {}".format(len(idx_var_too_high[0]))
        nr_further_points_to_mask = np.count_nonzero(np.logical_not(measurements.mask[idx_var_too_high,0]))
        print "Number of points that were already masked (being NaNs): {}".format(nr_further_points_to_mask)
        if nr_further_points_to_mask > THRESHOLD_NR_POINTS_TO_RERUN_KALMAN:
            # ... then it's worth continuing
            measurements.mask[idx_var_too_high, :] = True
            state_means2, state_vars2 = kf.smooth(measurements)
            coord_var2 = np.trace(state_vars2[:,:2,:2], axis1=1, axis2=2)
        else:
            # ... then turn off variance_smooth cos it's not worth
            variance_smooth = False
    
    if plot:
        # Plot original/corrected map
        lat_center = (np.max(state_means[:,0]) + np.min(state_means[:,0])) / 2.
        lon_center = (np.max(state_means[:,1]) + np.min(state_means[:,1])) / 2.
        map_osm = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        map_osm.add_children(folium.PolyLine(orig_measurements[:,:2], 
                                             color='#666666', weight = 4, opacity=1))
        map_osm.add_children(folium.PolyLine(state_means[:,:2], 
                                             color='#FF0000', weight = 4, opacity=1))
        if variance_smooth:
            map_osm.add_children(folium.PolyLine(state_means2[:,:2], 
                                                 color='#00FF00', weight = 4, opacity=1))
        
        # Create and save map
        map_osm.save(HTML_FILENAME, close_file=False)
        if platform.system() == "Darwin":
            # On MAC
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + HTML_FILENAME)
        elif platform.system() == 'Windows':
            # On Windows
            webbrowser.open(HTML_FILENAME, new=2)
                
        # Plot original/corrected altitude profile
        fig_alt, ax_alt = plt.subplots()
        #ax_alt.plot(coords['ele'], color='0.5', marker=".")
        ax_alt.plot(measurements[:,2], color="0.5", linestyle="None", marker=".")
        ax_alt.plot(state_means[:,2], color="r", linestyle="-", marker="None")
        ax_alt.legend(['Measured', 'Estimated'])
        ax_alt.grid(True)
        fig_alt.show()
        
        # Plot coordinates variance
        if variance_smooth:
            fig_coordvar, ax_coordvar = plt.subplots(2,1)
            ax_coordvar[0].plot(coord_var, color="r", linestyle="-", marker="None")
            if variance_smooth:
                ax_coordvar[0].plot(coord_var2, color="g", linestyle="-", marker="None")
            ax_coordvar[0].grid(True)
            ax_coordvar[1].hist(np.log10(coord_var), bins=100)
            ax_coordvar[1].grid(True)
            fig_coordvar.show()
        
        # Stats
        # print "Distance: %0.fm" % MyTotalDistance(state_means[:,0], state_means[:,1])
        # print "Uphill: %.0fm, Dowhhill: %.0fm" % MyUphillDownhill(state_means[:,2])
            
    return coords, measurements, state_means, state_vars, infos

def PlotKalmanFilterResults():
    return

def SaveDataToCoordsAndGPX(coords, state_means, usehtml):
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
    
    infos = ("<br>" if usehtml else "\n")
    infos = infos + ("<br>" if usehtml else "\n")
    infos = "STATS AFTER FILTERING"
    infos = infos + ("<br>" if usehtml else "\n")
    # infos = infos + GiveStats(new_gpx, track_nr=0, segment_nr=0, usehtml=False)
    infos = infos + GiveStats(new_gpx.tracks[0].segments[0], usehtml=False)
    
    #new_coords['speed'] = [p.speed for p in new_gpx.tracks[0].segments[0].points]
    
    return new_coords, new_gpx, infos


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
def LoadGPX(filename, usehtml):
    gpx_file = open(filename, 'r')
    gpx = gpxpy.parse(gpx_file)
    
    if usehtml:
        s = "GPX file structure:<br>"
        for itra, track in enumerate(gpx.tracks):
            s = s + "Track {}<br>".format(itra)
            for iseg, segment in enumerate(track.segments):
                info = segment.get_moving_data()
                s = s + "<tab>Segment {} >>> time: {:.2f}min, distance: {:.0f}m<br>".format(iseg, info[0]/60., info[2])
    else:
        s = "GPX file structure:\n"
        for itra, track in enumerate(gpx.tracks):
            s = s + "Track {}\n".format(itra)
            for iseg, segment in enumerate(track.segments):
                info = segment.get_moving_data()
                s = s + "  Segment {} >>> time: {:.2f}min, distance: {:.0f}m\n".format(iseg, info[0]/60., info[2])
    
    return gpx, s
            
def ParseGPX(gpx, track_nr, segment_nr, use_srtm_elevation, usehtml):
    infos = "Loading track[{}] >>> segment [{}]".format(track_nr, segment_nr)
    infos = infos + ("<br>" if usehtml else "\n")
    segment = gpx.tracks[track_nr].segments[segment_nr]
    coords = pd.DataFrame([
            {'idx': i,
             'lat': p.latitude,
             'lon': p.longitude,
             'ele': p.elevation,
             'time': p.time,
             'time_sec': (p.time - datetime.datetime(2000,1,1,0,0,0)).total_seconds()} for i, p in enumerate(segment.points)])
    coords.set_index('time', drop = True, inplace = True)
    coords['time_sec'] = coords['time_sec'] - coords['time_sec'][0]
    
    infos = infos + ("<br>" if usehtml else "\n")
    infos = infos + "STATS BASED ON THE GPX FILE"
    infos = infos + ("<br>" if usehtml else "\n")
    # infos = infos + GiveStats(gpx, track_nr, segment_nr, usehtml)
    infos = infos + GiveStats(segment, usehtml)
    
    # https://github.com/tkrajina/srtm.py
    infos = infos + ("<br>" if usehtml else "\n")
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
            
            infos = infos + "SRTM elevation correction done."
            infos = infos + ("<br>" if usehtml else "\n")
            infos = infos + ("<br>" if usehtml else "\n")
            infos = infos + "STATS BASED ON THE GPX FILE AFTER SRTM CORRECTION"
            infos = infos + GiveStats(gpx.tracks[0].segments[0], usehtml=usehtml)
            infos = infos + ("<br>" if usehtml else "\n")                  
        except:
            infos = infos + "SRTM correction failed for some reason, probably a shitty proxy."
            infos = infos + ("<br>" if usehtml else "\n")
    
    # Round sampling points at 1s. The error introduced should be negligible
    # the processing would be simplified
    coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
    
    # Add speed using embedded function (it won't be used, it's just to completeness)
    segment.points[0].speed, segment.points[-1].speed = 0., 0.
    gpx.add_missing_speeds()
    coords['speed'] = [p.speed for p in gpx.tracks[track_nr].segments[segment_nr].points]
    
    return gpx, coords, infos

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
    
def GiveStats(segment, usehtml):
    info = segment.get_moving_data()
    m, s = divmod(info[0], 60)
    h, m = divmod(m, 60)
    # string = "Moving      >>> Time: {:.0f}:{:.0f}:{:.0f}, Distance: {:.3f}km, Max speed: {:.1f}km/h".format(h, m, s, info[2]/1000., info[4]*3.6)
    infos = "Moving      >>> Time: {:.0f}:{:.0f}:{:.0f}, Distance: {:.3f}km".format(h, m, s, info[2]/1000.)
    infos = infos + ("<br>" if usehtml else "\n")
    m, s = divmod(info[2], 60)
    h, m = divmod(m, 60)
    infos = infos + "Stopped     >>> Time: {:.0f}:{:.0f}:{:.0f}, Distance: {:.3f}km".format(h, m, s, info[3]/1000.)
    infos = infos + ("<br>" if usehtml else "\n")
    
    info = segment.get_elevation_extremes()
    infos = infos + "Elevation   >>> {:.0f}m <---> {:.0f}m".format(info[0], info[1])
    infos = infos + ("<br>" if usehtml else "\n")
    
    info = segment.get_uphill_downhill()
    infos = infos + "Total climb >>> +{:.0f}m, -{:.0f}m".format(info[0], info[1])
    infos = infos + ("<br>" if usehtml else "\n")
    
    return infos
    
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

def PlotOnMap(coords_array, coords_array2, onmapdata, balloondata, rdp_reduction):
    # Documentation
    # https://www.youtube.com/watch?v=BwqBNpzQwJg
    # http://matthiaseisen.com/pp/patterns/p0203/
    # https://github.com/python-visualization/folium/tree/master/examples
    # http://vincent.readthedocs.io/en/latest/quickstart.html
    # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb
    # http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Quickstart.ipynb
    # Icons: 'ok-sign', 'cloud', 'info-sign', 'remove-sign', http://getbootstrap.com/components/
    
    # Mapping parameters
    HTML_FILENAME = "osm.html"
    MAPPING_LIBRARY = "folium"
    # MAPPING_LIBRARY = "mplleaflet"
    
    # RDP (Ramer–Douglas–Peucker) reduction
    RDP_EPSILON = 1e-4
    
    # Unpacking coordinates
    lat = coords_array[:,0]
    lon = coords_array[:,1]
    if coords_array2 is not None:
        lat2 = coords_array2[:,0]
        lon2 = coords_array2[:,1]
    
    # Process data to plot them along the trace
    if onmapdata is not None:
        # Unpacking onmapdata
        data = onmapdata['data']
        sides = onmapdata['sides']
        palette = onmapdata['palette']
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
        
    # Coordinates center
    lat_center = (np.max(lat) + np.min(lat)) / 2.
    lon_center = (np.max(lon) + np.min(lon)) / 2.
    
    # Depending on the library used for plotting
    if MAPPING_LIBRARY == "mplleaflet":
        # Creating figure
        fig, ax = plt.subplots()
        
        # Plot trace
        ax.plot(lon, lat, 'k', linewidth = 4)
        if coords_array2 is not None:
            ax.plot(lon2, lat2, 'r', linewidth = 4)
        
        # Plot data
        if onmapdata is not None:
            for col in range(M):
                tmp_lon = lon[1:] + distances[col][0]
                tmp_lat = lat[1:] + distances[col][1]
                tmp_poly_lon = np.hstack((lon[1:], np.flipud(tmp_lon)))
                tmp_poly_lat = np.hstack((lat[1:], np.flipud(tmp_lat)))
                tmp_poly = np.vstack((tmp_poly_lon,tmp_poly_lat)).T
                ax.add_patch(patches.Polygon(tmp_poly, hatch="o", facecolor=palette[col], alpha = 1.0))
            
        # Plot on OSM
        mplleaflet.show(fig = ax.figure, path=HTML_FILENAME)
        
    elif MAPPING_LIBRARY == "folium":
        # Initialize map
        map_osm = folium.Map(location=[lat_center, lon_center], zoom_start=13)#, tiles='Stamen Terrain')
        
        # Balloon plots (made with Vincent)
        if balloondata is not None:
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
                map_osm.add_children(folium.Marker([lat[marker_highest_point], lon[marker_highest_point]], 
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
        
        # Plot start/finish markers
        map_osm.add_children(folium.Marker([lat[0], lon[0]],
                                               popup = "Start",
                                               icon=folium.Icon(color='green', icon='circle-arrow-up')))
        map_osm.add_children(folium.Marker([lat[-1], lon[-1]], 
                                           popup = "Finish",
                                           icon=folium.Icon(color='red', icon='circle-arrow-down')))
        
        # Plot data
        if onmapdata is not None:
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

        # Plot trace
        if rdp_reduction:
            if onmapdata is not None:
                print "\nWARNING: RDP reduction activated with onmapdata, trace/polygons misallignments are possible"
            coords_array = rdp(coords_array, RDP_EPSILON)
            
        map_osm.add_children(folium.PolyLine(coords_array, color='#000000', weight = 4, opacity=1))
        if coords_array2 is not None:
            map_osm.add_children(folium.PolyLine(coords_array2, color='#FF0000', weight = 4, opacity=1))
        
        # Create and save map
        map_osm.save(HTML_FILENAME, close_file=False)
        if platform.system() == "Darwin":
            # On MAC
            cwd = os.getcwd()
            webbrowser.open("file://" + cwd + "/" + HTML_FILENAME)
        elif platform.system() == 'Windows':
            # On Windows
            webbrowser.open(HTML_FILENAME, new=2)
        
    return


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    print("############################ GPX VIEWER ############################\n")
    
    # Arguments
    track_nr = 0
    segment_nr = 0
    FILENAME = "ahrtal.gpx"
    # FILENAME = "casa-lavoro1.gpx"
    # FILENAME = "casa-lavoro2.gpx"
    
    if len(sys.argv) >= 2:
        if (sys.argv[1].endswith('.gpx') | sys.argv[1].endswith('.GPX')):
            FILENAME = sys.argv[1]
            if len(sys.argv) == 4:
                track_nr = int(sys.argv[2])
                segment_nr = int(sys.argv[3])
    else:
        print "No GPX file provided, the default file will be loaded."
    
    # Loading .gpx file
    print "Loading {} >>> track {} >>> segment {}". format(FILENAME, track_nr, segment_nr)
    gpx, infos = LoadGPX(FILENAME, usehtml=False)
    print infos
    gpx, coords, infos = ParseGPX(gpx, track_nr, segment_nr, use_srtm_elevation=False, usehtml=False)
    print infos
    
    #==============================================================================
    # Homemade processing
    #==============================================================================
    if False:
        lat_cleaned, lon_cleaned, h_cleaned, t_cleaned, s_cleaned, ds_cleaned, speed_h, speed_v, gradient = RemoveOutliers(coords, VERBOSE=False)
        h_filtered, dh_filtered, speed_v_filtered, gradient_filtered = FilterElevation(np.diff(t_cleaned), h_cleaned, ds_cleaned, 7)
        
        fig, ax = plt.subplots(4, 1, sharex=True, squeeze=True)
        ax = PlotSummary(ax, s_cleaned, h_filtered, dh_filtered, speed_h, speed_v_filtered, gradient_filtered)
        
        data = np.ones((len(lat_cleaned),2))
        data[:,0] = h_filtered / np.max(h_filtered) * 0.0004
        data[:,1] = np.hstack((np.asarray([0]), speed_h)) / np.max(np.hstack((np.asarray([0]), speed_h))) * 0.0004
        onmapdata = {'data': data,
                     'sides': (0, 1),
                     'palette': ('blue','red')}
        balloondata = {'distance': s_cleaned,
                       'elevation': h_filtered,
                       'speed': speed_h}
        PlotOnMap(np.vstack((lat_cleaned, lon_cleaned)).T, None, onmapdata=onmapdata, balloondata=balloondata, rdp_reduction=False)
    
    #==============================================================================
    # Kalman processing
    #==============================================================================
    if True:
        coords, measurements, state_means, state_vars, infos = ApplyKalmanFilter(coords, gpx,
                                                                                 method=0, 
                                                                                 use_acceleration=False,
                                                                                 variance_smooth=False,
                                                                                 plot=True,
                                                                                 usehtml=False)
        print infos
        
        new_coords, new_gpx, infos = SaveDataToCoordsAndGPX(coords, state_means, usehtml=False)
        print infos
        
    #    balloondata = {'distance': np.cumsum(HaversineDistance(np.asarray(k_coords['lat']), np.asarray(k_coords['lon']))),
    #                   'elevation': np.asarray(k_coords['ele']),
    #                   'speed': None}
    #    PlotOnMap(np.vstack((k_coords['lat'], k_coords['lon'])).T,
    #              np.vstack((coords['lat'], coords['lon'])).T,
    #              onmapdata=None, balloondata=balloondata, rdp_reduction=False)


if __name__ == "__main__":
    main()
