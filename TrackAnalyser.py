#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
@mail: stef.salati@gmail.com
"""

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QApplication, qApp, QAction,
                             QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QTextEdit,
                             QCheckBox, QComboBox, QSizePolicy, QTabWidget,
                             QListWidget, QListWidgetItem, QInputDialog, QAbstractItemView)
                             # QToolTip, QLineEdit, QPushButton
from PyQt5 import QtGui, QtCore
# from PyQt5.QtCore import QSettings
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.widgets import MultiCursor#, Cursor
import platform
import ctypes

import bombo as bombo

"""
TODO


"""


"""
DOCUMENTATION

PyQt4
http://www.python.it/wiki/show/qttutorial/
http://zetcode.com/gui/pyqt4/menusandtoolbars/
http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html

PyQt5
http://zetcode.com/gui/pyqt5/layout/
http://zetcode.com/gui/pyqt5/dialogs/
https://pythonspot.com/en/pyqt5-matplotlib/
https://pythonspot.com/en/pyqt5/

Plots
http://stackoverflow.com/questions/36350771/matplotlib-crosshair-cursor-in-pyqt-dialog-does-not-show-up
http://stackoverflow.com/questions/35414003/python-how-can-i-display-cursor-on-all-axes-vertically-but-only-on-horizontall
http://matplotlib.org/users/annotations.html
"""

FONTSIZE = 8
PLOT_FONTSIZE = 9

class MultiCursorLinkedToTrace(object):
    def __init__(self, ax1, x1, y1, ax2, x2, y2):
        # Axis 1
        self.ax1 = ax1
        self.lx1 = ax1.axhline(linewidth=1, color='k', alpha=0.5, label="caccola")  # the horiz line
        self.ly1 = ax1.axvline(linewidth=1, color='k', alpha=0.5)  # the vert line
        self.x1 = x1
        self.y1 = y1
        # Axis 2
        self.ax2 = ax2
        self.lx2 = ax2.axhline(linewidth=1, color='k', alpha=0.5)  # the horiz line
        self.ly2 = ax2.axvline(linewidth=1, color='k', alpha=0.5)  # the vert line
        self.x2 = x2
        self.y2 = y2
        # Annotation boxes
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
        self.txt1 = ax1.text(0, 0, "", alpha=0.5, bbox=bbox_props)
        self.txt2 = ax2.text(0, 0, "", alpha=0.5, bbox=bbox_props)
        
    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        # It needs to be inside a try statement in order not to crash when the cursor
        # is moved out of the x range
        try:
            indx = np.searchsorted(self.x1, [x])[0]
            x1 = self.x1[indx]
            y1 = self.y1[indx]
            x2 = self.x2[indx]
            y2 = self.y2[indx]
            # update the line positions
            self.lx1.set_ydata(y1)
            self.ly1.set_xdata(x1)
            self.lx2.set_ydata(y2)
            self.ly2.set_xdata(x2)
            # Update annotations
            self.txt1.set_text("{:.1f}m".format(y1))
            self.txt1.set_position((x1, y1))
            self.txt2.set_text("{:.1f}km/h".format(y2))
            self.txt2.set_position((x2, y2))
        except:
            return
        plt.draw()


class EmbeddedPlot_ElevationSpeed(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.top_axis = self.fig.add_subplot(211)
        self.bottom_axis = self.fig.add_subplot(212, sharex=self.top_axis)
        self.fig.set_facecolor("w")
        self.fig.set_tight_layout(True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def clear_figure(self):
        self.top_axis.cla()
        self.bottom_axis.cla()
        
    def update_figure(self, measurements, state_means, segment):
        # Draw plots
        self.top_axis, tmp_ele = bombo.PlotElevation(self.top_axis, measurements, state_means)
        self.bottom_axis, tmp_speed = bombo.PlotSpeed(self.bottom_axis, segment)
        # Add cursor
        def onclick(event):
            cursor_anchored.mouse_move(event)
            self.draw()
        if platform.system() == "Darwin":
            # Cursor on both plots but not linked to the trace
            self.multi = MultiCursor(self.fig.canvas, (self.top_axis, self.bottom_axis), color='r', lw=1, vertOn=True, horizOn=True)
        elif platform.system() == 'Windows':
            cursor_anchored = MultiCursorLinkedToTrace(self.top_axis, tmp_ele[0], tmp_ele[1],
                                                       self.bottom_axis, tmp_speed[0], tmp_speed[1])
            self.mpl_connect('motion_notify_event', onclick)
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()
        
    def update_figure_multiple_tracks(self, measurements_list, state_means_list, gpx_list):
        # Draw plots
        for i, measurements in enumerate(measurements_list):
            state_means = state_means_list[i]
            gpx = gpx_list[i]
            self.top_axis, tmp_ele = bombo.PlotElevation(self.top_axis, measurements, state_means)
            self.bottom_axis, tmp_speed = bombo.PlotSpeed(self.bottom_axis, gpx.tracks[0].segments[0])
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()
        
class EmbeddedPlot_CoordinatesVariance(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.top_axis = self.fig.add_subplot(211)
        self.bottom_axis = self.fig.add_subplot(212)#, sharex=self.top_axis)
        self.fig.set_facecolor("w")
        self.fig.set_tight_layout(True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def clear_figure(self):
        self.top_axis.cla()
        self.bottom_axis.cla()
        
    def update_figure(self, measurements, state_means, state_vars):
        # Draw plots
        self.top_axis, tmp_coords = bombo.PlotCoordinates(self.top_axis, state_means)
        self.bottom_axis, tmp_coordsvar = bombo.PlotCoordinatesVariance(self.bottom_axis, state_means, state_vars)
        # Add cursor
        """
        def onclick(event):
            cursor_anchored.mouse_move(event)
            self.draw()
        if platform.system() == "Darwin":
            # Cursor on both plots but not linked to the trace
            self.multi = MultiCursor(self.fig.canvas, (self.top_axis, self.bottom_axis), color='r', lw=1, vertOn=True, horizOn=True)
        elif platform.system() == 'Windows':
            cursor_anchored = MultiCursorLinkedToTrace(self.top_axis, tmp_ele[0], tmp_ele[1],
                                                       self.bottom_axis, tmp_speed[0], tmp_speed[1])
            self.mpl_connect('motion_notify_event', onclick)
        """
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()

class EmbeddedPlot_ElevationVariance(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.top_axis = self.fig.add_subplot(211)
        self.bottom_axis = self.fig.add_subplot(212, sharex=self.top_axis)
        self.fig.set_facecolor("w")
        self.fig.set_tight_layout(True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def clear_figure(self):
        self.top_axis.cla()
        self.bottom_axis.cla()
        
    def update_figure(self, measurements, state_means, state_vars):
        # Draw plots
        self.top_axis, tmp_ele = bombo.PlotElevation(self.top_axis, measurements, state_means)
        self.bottom_axis, tmp_elevar = bombo.PlotElevationVariance(self.bottom_axis, state_means, state_vars)
        # Add cursor
        def onclick(event):
            cursor_anchored.mouse_move(event)
            self.draw()
        if platform.system() == "Darwin":
            # Cursor on both plots but not linked to the trace
            self.multi = MultiCursor(self.fig.canvas, (self.top_axis, self.bottom_axis), color='r', lw=1, vertOn=True, horizOn=True)
        elif platform.system() == 'Windows':
            cursor_anchored = MultiCursorLinkedToTrace(self.top_axis, tmp_ele[0], tmp_ele[1],
                                                       self.bottom_axis, tmp_elevar[0], tmp_elevar[1])
            self.mpl_connect('motion_notify_event', onclick)
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()
        
class EmbeddedPlot_SpeedVariance(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.top_axis = self.fig.add_subplot(211)
        self.bottom_axis = self.fig.add_subplot(212, sharex=self.top_axis)
        self.fig.set_facecolor("w")
        self.fig.set_tight_layout(True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def clear_figure(self):
        self.top_axis.cla()
        self.bottom_axis.cla()
        
    def update_figure(self, measurements, state_means, state_vars, segment):
        # Draw plots
        self.top_axis, tmp_speed = bombo.PlotSpeed(self.top_axis, segment)
        self.bottom_axis, tmp_speedvar = bombo.PlotSpeedVariance(self.bottom_axis, state_means, state_vars)
        # Add cursor
        def onclick(event):
            cursor_anchored.mouse_move(event)
            self.draw()
        if platform.system() == "Darwin":
            # Cursor on both plots but not linked to the trace
            self.multi = MultiCursor(self.fig.canvas, (self.top_axis, self.bottom_axis), color='r', lw=1, vertOn=True, horizOn=True)
        elif platform.system() == 'Windows':
            cursor_anchored = MultiCursorLinkedToTrace(self.top_axis, tmp_speed[0], tmp_speed[1],
                                                       self.bottom_axis, tmp_speedvar[0], tmp_speedvar[1])
            self.mpl_connect('motion_notify_event', onclick)
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()


class MainWindow(QMainWindow):
    
    def selectFileToOpen(self):
        
        def getPreProcessingChoice(self, filename, filestructure):
            items = ("Choose the longest", "Merge all")
            item, okPressed = QInputDialog.getItem(self,
                                                   "Multiple tracks/segments",
                                                   "File '" + filename + "' contains more than one track/segment\n\n" + infos + "\nWhat to do?",
                                                   items, 0, False)
            if okPressed and item:
                return items.index(item)
            else:
                return 0
        
        # Try to recover the last used directory
        old_directory = self.settings.value("lastdirectory", str)
        # print "Last used directory: {}".format(old_directory)
        
        # Check if the setting exists
        if old_directory is not None:
            # Check if it's not empty
            if old_directory:
                old_directory = old_directory
            else:
                old_directory = "tracks"
        else:
            old_directory = "tracks"
        
        # Open the dialog box
        fullfilename_list = QFileDialog.getOpenFileNames(self,
                                                         'Open .gpx',
                                                         "tracks",
                                                         "GPX files (*.gpx)")
        
        # Process every selected file
        for i, fullfilename in enumerate(fullfilename_list[0]):
            # Process filename
            directory, filename = os.path.split(str(fullfilename))
            filename, fileextension = os.path.splitext(filename)
            
            # Save the new directory in the application settings (it only
            # needs to be done once)
            if i == 0:
                self.settings.setValue("lastdirectory", QtCore.QVariant(str(directory)))
            
            # Open file and inspect what's inside
            gpxraw, longest_traseg, Ntracks, Nsegments, infos = bombo.LoadGPX(fullfilename)
            
            # If there's more than one track or segment, ask how to proceed
            if (Ntracks > 1) or (Nsegments > 1):
                preprocessingchoice = getPreProcessingChoice(self, filename, infos)
                if preprocessingchoice == 0:
                    preprocessedgpx = bombo.SelectOneTrackAndSegmentFromGPX(gpxraw, longest_traseg[0], longest_traseg[1])
                    listname = filename + " (longest)"
                elif preprocessingchoice == 1:
                    preprocessedgpx = bombo.MergeAllTracksAndSegmentsFromGPX(gpxraw)
                    listname = filename + " (merged)"
            else:
                preprocessedgpx = gpxraw
                listname = filename
            
            # Append the list of open GPX files using the next available color (that's the size of the list -1)
            self.gpxlist.append(preprocessedgpx)
            self.gpxnamelist.append(listname)
            newitem = QListWidgetItem(listname)
            newitem.setBackground(QtGui.QColor(self.palette[len(self.gpxlist)-1]))
            self.tracklist.addItem(newitem)
            
        return
        
    def Go(self):
        if len(self.gpxselectedlist) > 0:
            # Temporarily change cursor
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Clear up global variables
            self.proc_coords = []
            self.proc_measurements = []
            self.proc_state_means = []
            self.proc_state_vars = []
            self.proc_new_coords = []
            self.proc_new_gpx = []
            self.proc_infos = []
            self.proc_coords_to_plot = []
            self.proc_coords_to_plot2 = []
            self.proc_balloondata = []
            
            # For every GPX file that is selected
            self.textOutput.setText("")
            for i, currentgpx in enumerate(self.gpxselectedlist):
                self.textOutput.append("**** {} ****\n".format(self.gpxselectednamelist[i]))
                
                # Parse the GPX file
                gpx, coords, infos = bombo.ParseGPX(currentgpx,
                                                    track_nr=0,
                                                    segment_nr=0,
                                                    use_srtm_elevation=bool(self.checkUseSRTM.isChecked()))
                self.textOutput.append(infos)
                
                # Kalman processing
                coords, measurements, state_means, state_vars, infos = bombo.ApplyKalmanFilter(coords,
                                                                                               gpx,
                                                                                               method=self.comboBoxProcessingMethod.currentIndex(), 
                                                                                               use_acceleration=self.checkUseAcceleration.isChecked(),
                                                                                               extra_smooth=self.checkExtraSmooth.isChecked(),
                                                                                               debug_plot=False)
                self.textOutput.append(infos)
                
                # Save data in GPX structure to compute speed and elevations
                new_coords, new_gpx, infos = bombo.SaveDataToCoordsAndGPX(coords, state_means)
                self.textOutput.append(infos)
        
                # Create balloondata for the html plot
                balloondata = {'distance': np.cumsum(bombo.HaversineDistance(np.asarray(new_coords['lat']), np.asarray(new_coords['lon']))),
                               'elevation': np.asarray(new_coords['ele']),
                               'speed': None}
                
                # Create extra data for the html plot (fully implemented in bombo, not here)
                """
                data = np.ones((len(lat_cleaned),2))
                data[:,0] = h_filtered / np.max(h_filtered) * 0.0004
                data[:,1] = np.hstack((np.asarray([0]), speed_h)) / np.max(np.hstack((np.asarray([0]), speed_h))) * 0.0004
                tangentdata = {'data': data,
                               'sides': (0, 1),
                               'palette': ('blue','red')}
                """
                
                # Saverelevant output in global variables
                self.proc_coords.append(coords)
                self.proc_measurements.append(measurements)
                self.proc_state_means.append(state_means)
                self.proc_state_vars.append(state_vars)
                self.proc_new_coords.append(new_coords)
                self.proc_new_gpx.append(new_gpx)
                self.proc_infos.append(infos)
                self.proc_coords_to_plot.append(np.vstack((new_coords['lat'], new_coords['lon'])).T)
                self.proc_coords_to_plot2.append(np.vstack((coords['lat'], coords['lon'])).T)
                self.proc_balloondata.append(balloondata)
                
            
            # Restore original cursor
            QApplication.restoreOverrideCursor()
            
            # Generate embedded plots
            if len(self.gpxselectedlist) == 1:
                self.plotEmbedded1.update_figure(measurements, state_means, new_gpx.tracks[0].segments[0])
                self.plotEmbedded2.update_figure(measurements, state_means, state_vars)
                self.plotEmbedded3.update_figure(measurements, state_means, state_vars)
                self.plotEmbedded4.update_figure(measurements, state_means, state_vars, new_gpx.tracks[0].segments[0])
            else:
                # Commentato per adesso
                # self.plotEmbedded1.update_figure_multiple_tracks(self.proc_measurements, self.proc_state_means, self.proc_new_gpx)
                self.plotEmbedded1.clear_figure()
                self.plotEmbedded2.clear_figure()
                self.plotEmbedded3.clear_figure()
                self.plotEmbedded4.clear_figure()
            
            # Generate html plot
            # If only one track is selected, proceed with the complete output, otherwise just plot the traces
            if len(self.gpxselectedlist) is 1:
                bombo.PlotOnMap(coords_array_list=self.proc_coords_to_plot,
                                coords_array2_list=self.proc_coords_to_plot2,
                                coords_palette = self.selectedpalette,
                                tangentdata=None,
                                balloondata_list=self.proc_balloondata,
                                rdp_reduction=self.checkUseRDP.isChecked())
            else:
                bombo.PlotOnMap(coords_array_list=self.proc_coords_to_plot,
                                coords_array2_list=None,
                                coords_palette = self.selectedpalette,
                                tangentdata=None,
                                balloondata_list=self.proc_balloondata,
                                rdp_reduction=self.checkUseRDP.isChecked())
                
            # Generate 3D plot
            bombo.PlotOnMap3D(new_coords['lat'], new_coords['lon'], 400)
                
        else:
            self.textOutput.setText("You need to open a .gpx file before!")
        return

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.initVariables()
        self.initUI()
        
    def initVariables(self):
        self.gpxlist = list()
        self.gpxnamelist = list()
        self.gpxselectedlist = list()
        self.gpxselectednamelist = list()
        self.palette = bombo.GeneratePalette(N=10) * 5 # replicated 5 times
        #self.palette = ["#0000FF", "#00FF00", "#00FFFF", "#FF0000", "#FF00FF", "#FFFF00", "#FFFFFF"] # test palette
        self.selectedpalette = list()
        
        self.proc_coords = list()
        self.proc_measurements = list()
        self.proc_state_means = list()
        self.proc_state_vars = list()
        self.proc_new_coords = list()
        self.proc_new_gpx = list()
        self.proc_infos = list()
        self.proc_coords_to_plot = list()
        self.proc_coords_to_plot2 = list()
        self.proc_balloondata = list()
        
    def initUI(self):
        def selection_changed():
            # Retrieve selected items
            # selecteditems = self.tracklist.selectedItems()
            selectedindexes = self.tracklist.selectedIndexes()
            
            # Adding the selected items to the processing list
            self.gpxselectedlist[:] = []
            self.gpxselectednamelist[:] = []
            self.selectedpalette[:] = []
            for i in selectedindexes:
                # print str(i.text())
                self.gpxselectedlist.append(self.gpxlist[i.row()])
                self.gpxselectednamelist.append(self.gpxnamelist[i.row()])
                self.selectedpalette.append(self.palette[i.row()])

        # Application Settings
        QtCore.QCoreApplication.setOrganizationName("Steba")
        QtCore.QCoreApplication.setOrganizationDomain("https://github.com/stesalati/sport/")
        QtCore.QCoreApplication.setApplicationName("Steba")
        self.settings = QtCore.QSettings(self)
        
        # Actions
        openfile = QAction(QtGui.QIcon("icons/openfile.png"), "Open .gpx", self)
        openfile.setShortcut("Ctrl+O")
        openfile.setStatusTip("Open file")
        openfile.triggered.connect(self.selectFileToOpen)
        
        go = QAction(QtGui.QIcon("icons/go.png"), "Go!", self)
        go.setShortcut("Ctrl+R")
        go.setStatusTip("Run analysis")
        go.triggered.connect(self.Go)
        
        sep = QAction(self)
        sep.setSeparator(True)
        
        quitapp = QAction(QtGui.QIcon("icons/quit.png"), "Quit", self)
        quitapp.setShortcut("Ctrl+Q")
        quitapp.setStatusTip("Quit application")
        quitapp.triggered.connect(qApp.quit)
        
        # Status bar
        self.statusBar().show()
        
        # Toolbar
        toolbar = self.addToolBar('My tools')
        toolbar.addAction(openfile)
        toolbar.addAction(go)
        toolbar.addAction(quitapp)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        
        # Menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&What do you wanna do?')
        fileMenu.addAction(openfile)
        
        # Main widget (everything that's not toolbar, statusbar or menubar must be in this widget)
        self.scatola = QWidget()
        
        # Main horizontal impagination
        hBox = QHBoxLayout()
        hBox.setSpacing(5)
        
        # Vertical left column
        vBox_left = QVBoxLayout()
        vBox_left.setSpacing(5)
        
        # 1st vertical box, a list
        self.tracklist = QListWidget()
        vBox_left.addWidget(self.tracklist)
        self.tracklist.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tracklist.itemSelectionChanged.connect(selection_changed)
        
        # 2nd vertical box, containing several horizontal boxes, one for each setting
        vBox2 = QVBoxLayout()
        vBox2.setSpacing(5)
        
        # Just the group label
        labelSettings = QLabel('Settings')
        vBox2.addWidget(labelSettings)
        
        # Use/don't use corrected altitude
        self.checkUseSRTM = QCheckBox("Use SRTM corrected elevation (needs Internet)")
        self.checkUseSRTM.setChecked(False)
        vBox2.addWidget(self.checkUseSRTM)
        
        # Choose processing method
        hBoxProcessingMethod = QHBoxLayout()
        labelProcessingMethod = QLabel('Processing method')
        hBoxProcessingMethod.addWidget(labelProcessingMethod)
        self.comboBoxProcessingMethod = QComboBox()
        self.comboBoxProcessingMethod.addItem("Just use available data")
        self.comboBoxProcessingMethod.addItem("Fill all gaps at T=1s (resample)")
        self.comboBoxProcessingMethod.addItem("Fill only smaller gaps at T=1s")
        hBoxProcessingMethod.addWidget(self.comboBoxProcessingMethod)
        vBox2.addLayout(hBoxProcessingMethod)
        
        # Use/don't use acceleration
        self.checkUseAcceleration = QCheckBox("Use acceleration in Kalman filter")
        self.checkUseAcceleration.setChecked(False)
        vBox2.addWidget(self.checkUseAcceleration)
        
        # Use/don't use variance smooth
        self.checkExtraSmooth = QCheckBox("Extra smooth")
        self.checkExtraSmooth.setChecked(False)
        vBox2.addWidget(self.checkExtraSmooth)
        
        # Use/don't reduction algorithm for plotting on the map
        self.checkUseRDP = QCheckBox("Use RDP to reduce number of points displayed")
        self.checkUseRDP.setChecked(False)
        vBox2.addWidget(self.checkUseRDP)
        
        vBox_left.addLayout(vBox2)
                
        # 3rd text, containing the processing output
        self.textOutput = QTextEdit()
        self.textOutput.setReadOnly(True)
        self.textOutput.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textOutput.clear()
        vBox_left.addWidget(self.textOutput)
        
        # I put "vBox_left" inside a widget and then the widget inside "hBox"
        # instead of just doing "hBox.addLayout(vBox_left) so I can set its
        # maximum width.
        vBox_left_widget = QWidget()
        vBox_left_widget.setLayout(vBox_left)
        vBox_left_widget.setMinimumWidth(300)
        vBox_left_widget.setMaximumWidth(400)
        hBox.addWidget(vBox_left_widget)
        
        # Vertical right column
        tab = QTabWidget()
        
        # Tab 1: Summary: elevation and speed
        tab1 = QWidget()
        # The tab layout
        vBox_right = QVBoxLayout()
        vBox_right.setSpacing(5)
        # Plot area
        self.plotEmbedded1 = EmbeddedPlot_ElevationSpeed(width=5, height=4, dpi=100)
        self.plotEmbedded1.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar1 = NavigationToolbar(self.plotEmbedded1, self.scatola)
        # Add widgets to the layout
        vBox_right.addWidget(self.plotEmbedded1)
        vBox_right.addWidget(self.mpl_toolbar1)
        # Associate the layout to the tab
        tab1.setLayout(vBox_right)
        
        # Tab 2: Coordinates and variance
        tab2 = QWidget()
        # The tab layout
        vBox_right = QVBoxLayout()
        vBox_right.setSpacing(5)
        # Plot area
        self.plotEmbedded2 = EmbeddedPlot_CoordinatesVariance(width=5, height=4, dpi=100)
        self.plotEmbedded2.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar2 = NavigationToolbar(self.plotEmbedded2, self.scatola)
        # Add widgets to the layout
        vBox_right.addWidget(self.plotEmbedded2)
        vBox_right.addWidget(self.mpl_toolbar2)
        # Associate the layout to the tab
        tab2.setLayout(vBox_right)
        
        # Tab 3: Elevation and variance
        tab3 = QWidget()
        # The tab layout
        vBox_right = QVBoxLayout()
        vBox_right.setSpacing(5)
        # Plot area
        self.plotEmbedded3 = EmbeddedPlot_ElevationVariance(width=5, height=4, dpi=100)
        self.plotEmbedded3.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar3 = NavigationToolbar(self.plotEmbedded3, self.scatola)
        # Add widgets to the layout
        vBox_right.addWidget(self.plotEmbedded3)
        vBox_right.addWidget(self.mpl_toolbar3)
        # Associate the layout to the tab
        tab3.setLayout(vBox_right)
        
        # Tab 4: Speed and variance
        tab4 = QWidget()
        # The tab layout
        vBox_right = QVBoxLayout()
        vBox_right.setSpacing(5)
        # Plot area
        self.plotEmbedded4 = EmbeddedPlot_SpeedVariance(width=5, height=4, dpi=100)
        self.plotEmbedded4.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar4 = NavigationToolbar(self.plotEmbedded4, self.scatola)
        # Add widgets to the layout
        vBox_right.addWidget(self.plotEmbedded4)
        vBox_right.addWidget(self.mpl_toolbar4)
        # Associate the layout to the tab
        tab4.setLayout(vBox_right)
        
        # Associate tabs
        tab.addTab(tab1, "Summary")
        tab.addTab(tab2, "Coordinates and variance")
        tab.addTab(tab3, "Elevation and variance")
        tab.addTab(tab4, "Speed and variance")
        
        hBox.addWidget(tab)
        
        # Setting hBox as main box
        self.scatola.setLayout(hBox)
        self.setCentralWidget(self.scatola)
        
        # Application settings
        self.setWindowTitle('STEBA GUI')
        self.setWindowIcon((QtGui.QIcon('icons/app.png')))
        self.setGeometry(100, 100, 1200, 700)
        self.show()


# Creating the application
app = QApplication(sys.argv)
main = MainWindow()

# Showing the right icon in the taskbar
if platform.system() == "Darwin":
    # On MAC
    pass
elif platform.system() == 'Windows':
    # On Windows
    myappid = 'Steba.Steba.Steba.v0.1' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

main.show()
sys.exit(app.exec_())
# Alternative:
# app.exec_()
