# TrackAnalyser
Small SW to read .gpx files, clean and plot them in an interactive plot.

Files:<br>
TrackAnalyser.py is the GUI, written in PyQt5 but made compatible with PyQt4 (that is required by mayavi, used for the 3D map) using qtpy.<br>
bombo.py is where the actual computational functions are. All of the "intelligence" is in here.<br>
