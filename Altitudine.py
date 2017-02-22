# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
"""

# Use "%matplotlib qt" to plot in a different window

import numpy as np
from scipy import signal, fftpack
from scipy.misc import imread
import matplotlib.pyplot as plt, mpld3
from matplotlib import gridspec
import gpxpy
import datetime
# from osmapi import OsmApi
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
# from urllib2 import urlopen
import urllib2
from pylab import imshow, imread, show
import mpld3
from mpld3 import plugins, utils
import mplleaflet

import math
import StringIO
from PIL import Image

###############################################################################

class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        draw()

class SnaptoCursor:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes: return

        x, y = event.xdata, event.ydata

        indx = searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        print ('x=%1.2f, y=%1.2f'%(x,y))
        draw()

def LoadGPX(filename):
    gpx_file = open(filename, 'r')
    gpx = gpxpy.parse(gpx_file)
    lats = []
    lons = []
    elevations = []
    times = []
    for track in gpx.tracks:
        for segment in track.segments:        
            for point in segment.points:
                # print 'Point at {0}, ({1},{2}), elev: {3}'.format(point.time, point.latitude, point.longitude, point.elevation)
                lats.append(point.latitude)
                lons.append(point.longitude)
                elevations.append(point.elevation)
                times.append((point.time - datetime.datetime(2000,1,1,0,0,0)).total_seconds())
    # Convert to arrays. Units are meters for distance and seconds for time
    lat = np.asarray(lats)
    lon = np.asarray(lons)
    h = np.asarray(elevations)
    t = np.asarray(times)
    t = t-t[0]
    return lat, lon, h, t

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
    
def PlotSamplingTimeAndDistance(n, dt, ds):
    fig1 = plt.figure(n)
    # Sampling time
    ax11 = fig1.add_subplot(6,1,1)
    ax11.plot(dt)
    plt.ylabel("Sampling time (s)")
    plt.xlabel("Sample")
    plt.grid()
    
    # Distance
    ax12 = fig1.add_subplot(6,1,2)
    ax12.plot(ds)
    plt.ylabel("Distance (m)")
    plt.xlabel("Sample")
    plt.grid()
    
def MapInteractivePlot(fig, s, h, dh, lat, lon):
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.001, 5]) 
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
    #ax0.set_ylim(np.min(h), np.max(h))
    
    # Total ascent/descent
#    at = AnchoredText("Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh_filtered, 1), TotalAscentDescent(dh_filtered, -1)),
#                      prop=dict(size=12), frameon=True,
#                      loc=2,
#                      )
#    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#    ax1.add_artist(at)
    
    # Fake subplot
    points = ax1.scatter(X[0], X[2])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    
    # Map with route
    lat_origin = np.min(lat)
    lon_origin = np.min(lon)
    lat_span = np.max(lat) - np.min(lat)
    lon_span = np.max(lon) - np.min(lon)
    zoom_level = 14
    
    a, margin_coords = getImageCluster(lat_origin, lon_origin, lat_span,  lon_span, zoom_level)
    #fig.patch.set_facecolor('white')
    points = ax2.plot(X[2], X[3], color = '0.5')
    img = np.asarray(a)
    # Extent simply associates values, in this case longitude and latitude, to the
    # map's corners.
    ax2.imshow(img, extent=[margin_coords[2], margin_coords[3], margin_coords[0], margin_coords[1]], zorder=0, origin="lower")
    points = ax2.scatter(X[2], X[3], s=4)
    ax2.set_xlim(np.min(lon), np.max(lon))
    ax2.set_ylim(np.min(lat), np.max(lat))    
    ax2.set_xlabel("Lat")
    ax2.set_ylabel("Lon")
    ax2.grid(True)
    plugins.connect(fig, plugins.LinkedBrush(points))
    mpld3.show()
    
    return fig
    
def MapPlot(ax1, ax2, s, h, dh, speed_h, speed_v, gradient, lat, lon):
    # Elevation over distance
    cursor1 = SnaptoCursor(ax1, s, h[0:-1])
    plt.connect('motion_notify_event', cursor1.mouse_move)
    ax1.plot(s, h[0:-1])
    ax1.set_ylabel("Elevation (m)")
    ax1.set_xlabel("Distance (m)")
    ax1.grid(True)
    ax1.set_xlim(np.min(s), np.max(s))
    ax1.set_ylim(np.min(h), np.max(h))
    
    # Total ascent/descent
    at = AnchoredText("Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh, 1), TotalAscentDescent(dh, -1)),
                  prop=dict(size=12), frameon=True,
                  loc=2,
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)
    
    # Coordinates
    ax2.plot(lat, lon)
    ax2.set_xlabel("Lat")
    ax2.set_ylabel("Lon")
    ax2.grid(True)
    
    return ax1, ax2, cursor1
    
def SummaryPlot(ax1, ax2, ax3, ax4, s, h, dh, speed_h, speed_v, gradient):
    # Elevation over distance
    cursor = SnaptoCursor(ax1, s, h[0:-1])
    plt.connect('motion_notify_event', cursor.mouse_move)
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
    
    return ax1, ax2, ax3, ax4, cursor
    
def TotalAscentDescent(h, dir):
    if dir >= 0:
        h_selected = h[np.where(h > 0)]
    else:
        h_selected = h[np.where(h < 0)]
    return sum(h_selected)
    
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
    
def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom):
    print "\nDownloading tiles from OpenStreetMaps"
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    
    # These are the tiles to download
    xmin, ymax = deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin = deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)
    print "Tiles %d - %d (horizontally) and %d - %d (vertically) are needed" % (xmin, xmax, ymin, ymax)
    
    # Margin coordinates of the tiles that were downloaded (adding 1 to the max
    # tiles as apparently the coordinates returned by num2deg refer to the
    # origin of the tile)
    lat_min, lon_min = num2deg(xmin, ymin, zoom)
    lat_max, lon_max = num2deg(xmax + 1, ymax + 1, zoom)
    lat_min_actual = np.min((lat_min, lat_max))
    lat_max_actual = np.max((lat_min, lat_max))
    lon_min_actual = np.min((lon_min, lon_max))
    lon_max_actual = np.max((lon_min, lon_max))
    margin_coords = (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)
    print "User requested area >>>>>>>>>>>>> lat: %f - %f, lon: %f - %f" % (lat_deg, lat_deg + delta_lat, lon_deg, lon_deg + delta_long)
    print "Returned area (must be wider) >>> lat: %f - %f, lon: %f - %f" % (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)

    Cluster = Image.new('RGB',((xmax-xmin+1)*256-1,(ymax-ymin+1)*256-1) ) 
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            try:
                imgurl=smurl.format(zoom, xtile, ytile)
                print("Opening: " + imgurl)
                
                proxy = urllib2.ProxyHandler({'http': 'salatis:Alzalarosa01@userproxy.tmg.local:8080'})
                opener = urllib2.build_opener(proxy)
                urllib2.install_opener(opener)
                
                imgstr = urllib2.urlopen(imgurl).read()
                tile = Image.open(StringIO.StringIO(imgstr))
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except: 
                print("Tile download failed for some reason.")
                tile = None

    return Cluster, margin_coords

###############################################################################
# http://www.trackprofiler.com/gpxpy/index.html

print "VISUALIZZATORE ALTITUDINE"
LIMIT_POS_SPEED_H = 12.0
LIMIT_NEG_SPEED_H = -12.0
LIMIT_POS_SPEED_V = 3.0
LIMIT_NEG_SPEED_V = -3.0
LIMIT_POS_GRADIENT = 4.0
LIMIT_NEG_GRADIENT = -4.0

filename = "original.gpx"
lat, lon, h, t = LoadGPX(filename)
s = np.cumsum(HaversineDistance(lat, lon))

# Getting rid of outliers. If they're left to filtering they alter the results quite significantly.
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

# Summary plot
# SummaryPlot(10, s_cleaned, h_cleaned, speed_h, speed_v, gradient)

# Filtering
window = 7
#b = np.sinc(0.25*np.linspace(-np.floor(window/2), np.floor(window/2), num=window, endpoint=True))# * signal.hamming(window)
b = signal.hann(window)
b = b/np.sum(b)
#PlotFilter(b)
h_filtered = signal.filtfilt(b, 1, h_cleaned)

# Recomputing speed and gradient (as they depend on h)
dh_filtered = np.diff(h_filtered)
speed_v_filtered = dh_filtered/dt
gradient_filtered = dh_filtered/ds

# Summary plot
#fig, ax = SummaryPlot(10, s_cleaned, h_filtered, dh_filtered, speed_h, speed_v_filtered, gradient_filtered)
#fig, ax = plt.subplots()
#ax, cursor = FinalPlot(ax, s_cleaned, h_filtered)


n = 10
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

#ax1 = fig.add_subplot(2, 1, 1)
#ax2 = fig.add_subplot(2, 1, 2)
#ax1, ax2, cursor = MapPlot(ax1, ax2,
#                           s_cleaned,
#                           h_filtered, dh_filtered,
#                           speed_h, speed_v_filtered, gradient_filtered,
#                           lat_cleaned, lon_cleaned)
#

fig = plt.figure(figsize=(12, 12))
MapInteractivePlot(fig, s_cleaned, h_filtered, dh_filtered, lat_cleaned, lon_cleaned)





#
#fig.subplots_adjust(hspace=0.1, wspace=0.1)
#gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.001, 5]) 
#ax0 = plt.subplot(gs[0])
#ax1 = plt.subplot(gs[1])
#ax2 = plt.subplot(gs[2])
#
#lat_origin = np.min(lat_cleaned)
#lon_origin = np.min(lon_cleaned)
#lat_span = np.max(lat_cleaned) - np.min(lat_cleaned)
#lon_span = np.max(lon_cleaned) - np.min(lon_cleaned)
#zoom_level = 14
#
#a, margin_coords = getImageCluster(lat_origin, lon_origin, lat_span,  lon_span, zoom_level)
#points = ax2.plot(lon_cleaned, lat_cleaned)
#img = np.asarray(a)
## Extent simply associates values, in this case longitude and latitude, to the
## map's corners.
#ax2.imshow(img, zorder=0, extent=[margin_coords[2], margin_coords[3], margin_coords[0], margin_coords[1]], origin="upper")
#
#
#points = ax2.scatter(lon_cleaned, lat_cleaned, s=4)
#ax2.set_xlim(np.min(lon), np.max(lon))
#ax2.set_ylim(np.min(lat), np.max(lat))    
#ax2.set_xlabel("Lat")
#ax2.set_ylabel("Lon")
#ax2.grid(True)


















#plt.hold(True)
#plt.plot(lon_cleaned, lat_cleaned, 'b') # Draw blue line
#plt.plot(lon_cleaned, lat_cleaned, 'rs') # Draw red squares
#
#mplleaflet.show()

#http://api.openstreetmap.org/api/0.6/map?bbox=11.54,48.14,11.543,48.145&scale=700000&format=png

#https://api.mapbox.com/styles/v1/mapbox/streets-v8/static/-122.4241,37.78,14.25,0,60/600x600?access_token=pk.eyJ1Ijoic3Rlc2FsYXRpIiwiYSI6ImNpejlxY3YyOTAyYmkyd21wc3BuaDUzbmsifQ.CsjWxwu7arzlJxPSpMR2fg

#print "Downloading map... "
#tries = 0
#url = None
#while tries < 60:
#    tries += 1
#    print 'Try {}...'.format(tries)
#    try:
#        url = urlopen('http://parent.tile.openstreetmap.org/cgi-bin/export?'
#                      'bbox={lat1:.2f},{lon1:.2f},{lat2:.2f},{lon2:.2f}&'
#                      'scale={scale:d}&format=png'.format(lat1=lat[0],
#                      lat2=lat[1],
#                      lon1=lon[0],
#                      lon2=lon[1],
#                      scale=scale))
#    except HTTPError:
#        sleep(5)
#        continue
#    else:
#        print 'Map successfully downloaded.'
#        break
#
#if url is None:
#    print 'Failed to download a map.'
#else:
#    m = imread(url)
#    imshow(m, extent=lat+lon, aspect='equal')
#    show()
#              
## http://parent.tile.openstreetmap.org/cgi-bin/export?bbox={lat1:18.6},{lon1:50.35},{lat2:20.2},{lon2:20.2}&scale={scale:700000}&format=png
#              
#
## plot the map
#imshow(imread(url), extent=lat+lon, aspect='equal')
#
## plot other data here
#
#show()

#MyApi = OsmApi(
#            api = "http://www.openstreetmap.org",
#            username = u"stef.salati@gmail.com",
#            password = u"caccola00")
#            #proxy_host = 'userproxy.tmg.local',
#            #proxy_port = 8080
#        #)
#print MyApi.NodeGet(123)
#{u'changeset': 532907, u'uid': 14298,
# u'timestamp': u'2007-09-29T09:19:17Z',
# u'lon': 10.790009299999999, u'visible': True,
# u'version': 1, u'user': u'Mede',
# u'lat': 59.9503044, u'tag': {}, u'id': 123}
#
#print node

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

#MyApi = OsmApi(username = u"EtienneChove", password = u"*******")
#MyApi.ChangesetCreate({u"comment": u"My first test"})
#print MyApi.NodeCreate({u"lon":1, u"lat":1, u"tag": {}})
#{u'changeset': 532907, u'lon': 1, u'version': 1, u'lat': 1, u'tag': {}, u'id': 164684}
#MyApi.ChangesetClose()
