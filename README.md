# TrackAnalyser

## Overview
TrackAnalyser is a small software, written in Python, to open, process and plot .gpx files. It can be launched with:
> python TrackAnalyser.py

## Files and folders
* __TrackAnalyser.py__ is the file to be launched, the GUI. Originally written in PyQt5, has been made backcompatible with PyQt4 using qtpy to allow the use of the mayavi module, that is only compatible with PyQt4.
* __bombo.py__ is where the actual computational functions are.
* __gdal_merge.py__ is a script to merge .tif map elevation data. Originally part of the gdal module, it has been copied here so to provide an easy installation-independent access.
* __osm.html__ is the generated openstreetmap map.
* __tracks/__ is the default folder where user's .gpx files can be stored.
* __maps/osm__ is where 2D OpenStreetMaps tiles are stored. These tiles are downloaded aumatically when needed from http://a.tile.openstreetmap.org/ZOOM/XTILE/YTILE.png. Not to overload the server with repeated needless requests, it's advised not to clear this local repository so already downloaded tiles can be used.
* __maps/srtm__ is where elevation maps are stored. This folder must be populated manually (i.e. tiles need to be copy/pasted manually in this folder for the 3D elevation profile to be plotted): .tif SRTM files must be located for the software to find and use them. The usual path (mandatory for tiles that are loaded and merged automatically) is __srtm_XX_YY.tif__, files can be downloaded here: http://dwtkns.com/srtm/ or http://srtm.csi.cgiar.org/SELECTION/inputCoord.asp. Alternatively, tiles can also be used manually (e.g. "iceland.tif"), in this case the file name must be specified in TrackAnalyser.
* __icons/__ is self-explanatory.
* All other files are generated and auxiliary.

## List of the required Python modules
numpy, scipy, matplotlib, re, gpxpy, datetime, mplleaflet, os, folium, webbrowser, vincent, json, sys, pykalman, srtm, pandas, platform, rdp, colorsys, osgeo.gdal, math, mayavi.mlab, qtpy, pyqt4, traits.api, traitsui.api, ctypes, sip
