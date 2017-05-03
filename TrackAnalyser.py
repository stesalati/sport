#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Stefano Salati
@mail: stef.salati@gmail.com
"""
"""
TODO
- Capire perché le aree quadrate vengono rappresentate come dei rettangoli in mayavi, e storte quando le si guarda dall'alto. Prospettiva strana...
"""

import os
os.environ['QT_API'] = 'pyqt'
os.environ['ETS_TOOLKIT'] = 'qt4'

from qtpy.QtWidgets import (QMainWindow, QWidget, QApplication, qApp, QAction,
                            QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QTextEdit,
                            QCheckBox, QComboBox, QSizePolicy, QTabWidget,
                            QListWidget, QListWidgetItem, QInputDialog, QAbstractItemView,
                            QTreeView, QSpinBox, QDoubleSpinBox, QPushButton, QDialog,
                            QLineEdit, QFrame, QGridLayout)
from qtpy import QtGui, QtCore#, QtWebEngineWidgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
if os.environ['QT_API'] == 'pyqt':
    # To be used qith PyQt4
    from matplotlib.backends.backend_qt4agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar)
else:
    # To be used qith PyQt5
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar)
from matplotlib.widgets import MultiCursor
import platform
import ctypes
import sys
import sip

# Comments for mayavi integration in pyqt (I ignored them and it works nevertheless)
# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
#from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)
from traits.api import HasTraits, Instance#, on_trait_change
from traitsui.api import View, Item
from tvtk.api import tvtk
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import bombo as bombo

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

QtPy
I need to use this cos I've already upgraded my code from 4 to 5 and now I see I need to use 4 as mayavi only works with 4.
https://pypi.python.org/pypi/QtPy

Plots
http://stackoverflow.com/questions/36350771/matplotlib-crosshair-cursor-in-pyqt-dialog-does-not-show-up
http://stackoverflow.com/questions/35414003/python-how-can-i-display-cursor-on-all-axes-vertically-but-only-on-horizontall
http://matplotlib.org/users/annotations.html
"""

FONTSIZE = 8

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
        
        
class EmbeddedPlot_Details(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        self.axis_coords = self.fig.add_subplot(231)
        self.axis_elevation = self.fig.add_subplot(232)
        self.axis_speed = self.fig.add_subplot(233, sharex=self.axis_elevation)
        self.axis_coords_variance = self.fig.add_subplot(234, sharex=self.axis_elevation)
        self.axis_elevation_variance = self.fig.add_subplot(235, sharex=self.axis_elevation)
        self.axis_speed_variance = self.fig.add_subplot(236, sharex=self.axis_elevation)
        
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
        self.axis_coords.cla()
        self.axis_elevation.cla()
        self.axis_speed.cla()
        self.axis_coords_variance.cla()
        self.axis_elevation_variance.cla()
        self.axis_speed_variance.cla()
        
    def update_figure(self, measurements, state_means, state_vars, segment):
        # Draw plots
        self.axis_coords, tmp_coords = bombo.PlotCoordinates(self.axis_coords, state_means)
        self.axis_elevation, tmp_ele = bombo.PlotElevation(self.axis_elevation, measurements, state_means)
        self.axis_speed, tmp_speed = bombo.PlotSpeed(self.axis_speed, segment)
        self.axis_coords_variance, tmp_coordsvar = bombo.PlotCoordinatesVariance(self.axis_coords_variance, state_means, state_vars)
        self.axis_elevation_variance, tmp_elevar = bombo.PlotElevationVariance(self.axis_elevation_variance, state_means, state_vars)
        self.axis_speed_variance, tmp_speedvar = bombo.PlotSpeedVariance(self.axis_speed_variance, state_means, state_vars)
        
        # Add cursor
        def onclick(event):
            cursor_anchored2.mouse_move(event)
            cursor_anchored3.mouse_move(event)
            self.draw()
        if platform.system() == "Darwin":
            # Cursor on both plots but not linked to the trace
            self.multi = MultiCursor(self.fig.canvas,
                                     (self.axis_elevation, self.axis_speed, self.axis_coords_variance, self.axis_elevation_variance, self.axis_speed_variance),
                                     color='r', lw=1, vertOn=True, horizOn=True)
        elif platform.system() == 'Windows':
            cursor_anchored2 = MultiCursorLinkedToTrace(self.axis_elevation, tmp_ele[0], tmp_ele[1],
                                                        self.axis_elevation_variance, tmp_elevar[0], tmp_elevar[1])
            cursor_anchored3 = MultiCursorLinkedToTrace(self.axis_speed, tmp_speed[0], tmp_speed[1],
                                                        self.axis_speed_variance, tmp_speedvar[0], tmp_speedvar[1])
            self.mpl_connect('motion_notify_event', onclick)
            
        # Draw
        self.fig.set_tight_layout(True)
        self.draw()


class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
    
    # I created this function to call the visualization.update_plot function, that is part of the Visualization class
    def update_plot(self, terrain, track):
        self.visualization.update_plot(terrain, track)
        
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    #@on_trait_change('scene.activated')
    def update_plot(self, terrain, track):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
        # self.scene.mlab.test_points3d()
        
        # Here's were I embedded my code
        self.scene.mlab.clf()
        elevation_mesh = self.scene.mlab.mesh(terrain['x'], terrain['y'], terrain['z'])
        
        bmp = tvtk.PNGReader(file_name=bombo.TEXTURE_FILE)
        texture = tvtk.Texture(input_connection=bmp.output_port, interpolate=1)
        elevation_mesh.actor.actor.mapper.scalar_visibility=False
        elevation_mesh.actor.enable_texture = True
        elevation_mesh.actor.tcoord_generator_mode = 'plane'
        elevation_mesh.actor.actor.texture = texture
           
        self.scene.mlab.text3d((terrain['x'][0][0] + terrain['x'][-1][0]) / 2,
                                terrain['y'][0][0],
                                np.max(terrain['z']),
                                "NORTH", scale=(track['textsize'], track['textsize'], track['textsize']))
        
        self.scene.mlab.plot3d(track['x'], track['y'], track['z'],
                               color=track['color'],
                               line_width=track['line_width'],
                               tube_radius=track['line_radius'])

    # the layout of the dialog screated
    view = View(Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=250,
                     width=300,
                     show_label=False),
                resizable=True # We need this to resize with the parent widget
                )


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
        # print "Last used directory: {}\n".format(old_directory)
        
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
        if os.environ['QT_API'] == 'pyqt':
            pass
        elif os.environ['QT_API'] == 'pyqt5':
            fullfilename_list = fullfilename_list[0]
        
        # Process every selected file
        for i, fullfilename in enumerate(fullfilename_list):
            # Process filename
            directory, filename = os.path.split(str(fullfilename))
            filename, fileextension = os.path.splitext(filename)
            
            # Save the new directory in the application settings (it only
            # needs to be done once)
            if i == 0:
                # print "New directory to be saved: {}\n".format(directory)
                if os.environ['QT_API'] == 'pyqt':
                    self.settings.setValue("lastdirectory", str(directory))
                elif os.environ['QT_API'] == 'pyqt5':
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
            self.proc_coords_to_plot = []
            self.proc_coords_to_plot2 = []
            self.proc_balloondata = []
            
            # For every GPX file that is selected
            self.textWarningConsole.clear()
            for i, currentgpx in enumerate(self.gpxselectedlist):
                # Parse the GPX file
                gpx, coords, dinfos_before, warnings = bombo.ParseGPX(currentgpx,
                                                                      track_nr=0,
                                                                      segment_nr=0,
                                                                      use_srtm_elevation=bool(self.checkUseSRTM.isChecked()))
                self.textWarningConsole.append(warnings)
                
                # Kalman processing
                coords, measurements, state_means, state_vars, dinfos_during = bombo.ApplyKalmanFilter(coords,
                                                                                                       gpx,
                                                                                                       method=self.comboBoxProcessingMethod.currentIndex(), 
                                                                                                       use_acceleration=self.checkUseAcceleration.isChecked(),
                                                                                                       extra_smooth=self.checkExtraSmooth.isChecked(),
                                                                                                       debug_plot=False)
                
                # Save data in GPX structure to compute speed and elevations
                new_coords, new_gpx, dinfos_after = bombo.SaveDataToCoordsAndGPX(coords, state_means)
                
                # Update GUI with the computed stats               
                parent = QtGui.QStandardItem(self.gpxselectednamelist[i])
                
                parent_beforeprocessing = QtGui.QStandardItem("Raw GPX stats")
                parent_beforeprocessing.appendRow([QtGui.QStandardItem("Total distance"), QtGui.QStandardItem(dinfos_before['total_distance'])])
                parent_beforeprocessing_moving = QtGui.QStandardItem("Moving")
                parent_beforeprocessing_moving.appendRow([QtGui.QStandardItem("Time"), QtGui.QStandardItem(dinfos_before['moving_time'])])
                parent_beforeprocessing_moving.appendRow([QtGui.QStandardItem("Distance"), QtGui.QStandardItem(dinfos_before['moving_distance'])])
                parent_beforeprocessing.appendRow(parent_beforeprocessing_moving)
                parent_beforeprocessing_idle = QtGui.QStandardItem("Idle")
                parent_beforeprocessing_idle.appendRow([QtGui.QStandardItem("Time"), QtGui.QStandardItem(dinfos_before['idle_time'])])
                parent_beforeprocessing_idle.appendRow([QtGui.QStandardItem("Distance"), QtGui.QStandardItem(dinfos_before['idle_distance'])])
                parent_beforeprocessing.appendRow(parent_beforeprocessing_idle)
                parent_beforeprocessing.appendRow([QtGui.QStandardItem("Elevation"), QtGui.QStandardItem(dinfos_before['elevation'])])
                parent_beforeprocessing.appendRow([QtGui.QStandardItem("Climb"), QtGui.QStandardItem(dinfos_before['climb'])])
                parent.appendRow(parent_beforeprocessing)
                
                parent.appendRow([QtGui.QStandardItem("Samples"), QtGui.QStandardItem(dinfos_during['nsamples'])])
                parent.appendRow([QtGui.QStandardItem("Total distance"), QtGui.QStandardItem(dinfos_after['total_distance'])])
                parent_moving = QtGui.QStandardItem("Moving")
                parent_moving.appendRow([QtGui.QStandardItem("Time"), QtGui.QStandardItem(dinfos_after['moving_time'])])
                parent_moving.appendRow([QtGui.QStandardItem("Distance"), QtGui.QStandardItem(dinfos_after['moving_distance'])])
                parent.appendRow(parent_moving)
                parent_idle = QtGui.QStandardItem("Idle")
                parent_idle.appendRow([QtGui.QStandardItem("Time"), QtGui.QStandardItem(dinfos_after['idle_time'])])
                parent_idle.appendRow([QtGui.QStandardItem("Distance"), QtGui.QStandardItem(dinfos_after['idle_distance'])])
                parent.appendRow(parent_idle)
                parent.appendRow([QtGui.QStandardItem("Elevation"), QtGui.QStandardItem(dinfos_after['elevation'])])
                parent.appendRow([QtGui.QStandardItem("Climb"), QtGui.QStandardItem(dinfos_after['climb'])])
                self.treemodel.appendRow(parent)
        
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
                
                # Save relevant output in global variables
                self.proc_coords.append(coords)
                self.proc_measurements.append(measurements)
                self.proc_state_means.append(state_means)
                self.proc_state_vars.append(state_vars)
                self.proc_new_coords.append(new_coords)
                self.proc_new_gpx.append(new_gpx)
                self.proc_coords_to_plot.append(np.vstack((new_coords['lat'], new_coords['lon'])).T)
                self.proc_coords_to_plot2.append(np.vstack((coords['lat'], coords['lon'])).T)
                self.proc_balloondata.append(balloondata)
                
            
            # Restore original cursor
            QApplication.restoreOverrideCursor()
            
            # Generate embedded plots
            if len(self.gpxselectedlist) == 1:
                self.plotEmbeddedElevationAndSpeed.update_figure(measurements, state_means, new_gpx.tracks[0].segments[0])
                self.plotEmbeddedDetails.update_figure(measurements, state_means, state_vars, new_gpx.tracks[0].segments[0])
            else:
                # Commentato per adesso
                # self.plotEmbeddedElevationAndSpeed.update_figure_multiple_tracks(self.proc_measurements, self.proc_state_means, self.proc_new_gpx)
                self.plotEmbeddedElevationAndSpeed.clear_figure()
                self.plotEmbeddedDetails.clear_figure()
            
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
            if len(self.gpxselectedlist) == 1:
                if self.check3DMapSelection.isChecked():
                    tile_selection = 'auto'
                else:
                    tile_selection = self.text3DMapName.text()
                terrain, track, warnings = bombo.Generate3DMap(new_coords['lat'], new_coords['lon'],
                                                               tile_selection=tile_selection,
                                                               margin=self.spinbox3DMargin.value(),
                                                               elevation_scale=self.spinbox3DElevationScale.value(),
                                                               mapping='coords',
                                                               use_osm_texture=True,
                                                               texture_type='osm',
                                                               texture_zoom=self.spinbox3DOSMZoom.value(),
                                                               texture_invert=self.check3DOSMInvert.isChecked(),
                                                               verbose=False)
                
                self.textWarningConsole.append(warnings)
                
                if terrain is not None:    
                    self.map3d.update_plot(terrain, track)
        else:
            self.textWarningConsole.setText("You need to open a .gpx file before!")
        return
    
    def PlotSpecificAreaDialog(self):
        
        def PlotSpecificArea():
            # Save coordinates for the next time
            if os.environ['QT_API'] == 'pyqt':
                self.settings.setValue("last_point_coord_lat", self.spinboxLatDec.value())
                self.settings.setValue("last_point_coord_lon", self.spinboxLonDec.value())
            elif os.environ['QT_API'] == 'pyqt5':
                self.settings.setValue("last_point_coord_lat", QtCore.QVariant(self.spinboxLatDec.value()))
                self.settings.setValue("last_point_coord_lon", QtCore.QVariant(self.spinboxLonDec.value()))
            
            # Select the 3D tab
            self.tab.setCurrentIndex(1)
            
            # Plot
            if self.check3DMapSelection.isChecked():
                tile_selection = 'auto'
            else:
                tile_selection = self.text3DMapName.text()
            
            terrain, track, warnings = bombo.Generate3DMap([self.spinboxLatDec.value()], [self.spinboxLonDec.value()],
                                                           tile_selection=tile_selection,
                                                           margin=self.spinbox3DMargin.value(),
                                                           elevation_scale=self.spinbox3DElevationScale.value(),
                                                           mapping='coords',
                                                           use_osm_texture=True,
                                                           texture_type='osm',
                                                           texture_zoom=self.spinbox3DOSMZoom.value(),
                                                           texture_invert=self.check3DOSMInvert.isChecked(),
                                                           verbose=False)
            
            self.textWarningConsole.append(warnings)
            
            if terrain is not None:    
                self.map3d.update_plot(terrain, track)
            d.done(0)
            
        def Convert():
            try:
                dd = bombo.parse_dms(self.textLatLonGMS.text())
                self.spinboxLatDec.setValue(dd[0])
                self.spinboxLonDec.setValue(dd[1])
            except:
                pass
            
        d = QDialog()
        grid = QGridLayout()

        hBox_coordsGMS = QHBoxLayout()
        hBox_coordsGMS.setSpacing(5)
        label = QLabel('Coordinates (gms)')
        grid.addWidget(label, 0, 0)
        self.textLatLonGMS = QLineEdit()
        self.textLatLonGMS.setText("")
        grid.addWidget(self.textLatLonGMS, 0, 1, 1, 2)
        
        button1 = QPushButton("Convert to decimal")
        button1.clicked.connect(Convert)
        grid.addWidget(button1, 0, 3)
        
        label = QLabel('Coordinates (decimal)')
        grid.addWidget(label, 1, 0)
        self.spinboxLatDec = QDoubleSpinBox()
        self.spinboxLatDec.setRange(-90,+90)
        self.spinboxLatDec.setSingleStep(0.0000001)
        self.spinboxLatDec.setDecimals(7)
        grid.addWidget(self.spinboxLatDec, 1, 1)
        self.spinboxLonDec = QDoubleSpinBox()
        self.spinboxLonDec.setRange(-180,+180)
        self.spinboxLonDec.setSingleStep(0.0000001)
        self.spinboxLonDec.setDecimals(7)
        grid.addWidget(self.spinboxLonDec, 1, 2)
        
        # Try to recover the last used points
        try:
            old_lat = self.settings.value("last_point_coord_lat", type=float)
            old_lon = self.settings.value("last_point_coord_lon", type=float)
            self.spinboxLatDec.setValue(old_lat)
            self.spinboxLonDec.setValue(old_lon)
        except:
            # Coordinates of Mt. Rinjani in Indonesia
            self.spinboxLatDec.setValue(-8.4166000)
            self.spinboxLonDec.setValue(116.4666000)
        
        button2 = QPushButton("Show 3D map")
        button2.clicked.connect(PlotSpecificArea)
        grid.addWidget(button2, 1, 3)

        d.setWindowTitle("Show point on 3D map")
        d.setLayout(grid)
        d.setWindowModality(QtCore.Qt.ApplicationModal)
        d.exec_()
        
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
        QtCore.QCoreApplication.setOrganizationName("Ste")
        QtCore.QCoreApplication.setOrganizationDomain("https://github.com/stesalati/sport/")
        QtCore.QCoreApplication.setApplicationName("TrackAnalyser")
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
        
        showpoint = QAction(QtGui.QIcon("icons/point.png"), "Show point", self)
        showpoint.setShortcut("Ctrl+P")
        showpoint.setStatusTip("Show point")
        showpoint.triggered.connect(self.PlotSpecificAreaDialog)
        
        sep = QAction(self)
        sep.setSeparator(True)
        
        quitapp = QAction(QtGui.QIcon("icons/quit.png"), "Quit", self)
        quitapp.setShortcut("Ctrl+Q")
        quitapp.setStatusTip("Quit application")
        quitapp.triggered.connect(qApp.quit)
        
        # Toolbar
        toolbar = self.addToolBar('My tools')
        toolbar.addAction(openfile)
        toolbar.addAction(go)
        toolbar.addAction(showpoint)
        toolbar.addAction(quitapp)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        toolbar.setIconSize(QtCore.QSize(24,24))
                
        # Status bar
        self.statusBar().show()
        
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
        self.tracklist.setMaximumHeight(120)
        
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
        
        # Choose processing method + use/don't use acceleration
        hBoxProcessingMethod = QHBoxLayout()
        labelProcessingMethod = QLabel('Processing method')
        hBoxProcessingMethod.addWidget(labelProcessingMethod)
        self.comboBoxProcessingMethod = QComboBox()
        self.comboBoxProcessingMethod.addItem("Just use available data")
        self.comboBoxProcessingMethod.addItem("Fill all gaps at T=1s (resample)")
        self.comboBoxProcessingMethod.addItem("Fill only smaller gaps at T=1s")
        hBoxProcessingMethod.addWidget(self.comboBoxProcessingMethod)
        self.checkUseAcceleration = QCheckBox("Use acceleration")
        self.checkUseAcceleration.setChecked(False)
        hBoxProcessingMethod.addWidget(self.checkUseAcceleration)
        vBox2.addLayout(hBoxProcessingMethod)
        
        # Use/don't use variance smooth
        self.checkExtraSmooth = QCheckBox("Extra smooth")
        self.checkExtraSmooth.setChecked(False)
        vBox2.addWidget(self.checkExtraSmooth)
        
        # Use/don't reduction algorithm for plotting on the map
        self.checkUseRDP = QCheckBox("Use RDP to reduce number of points displayed on 2D map")
        self.checkUseRDP.setChecked(False)
        vBox2.addWidget(self.checkUseRDP)
        
        line = QFrame()
        #line.setGeometry(QtCore.QRect(320, 150, 118, 3))
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        vBox2.addWidget(line)
        
        # Settings for the 3D map
        label3DViewSettings = QLabel('3D view settings')
        vBox2.addWidget(label3DViewSettings)
        
        hBox3DMapSelection = QHBoxLayout()
        self.check3DMapSelection = QCheckBox("Select tiles automatically, otherwise")
        self.check3DMapSelection.setChecked(True)
        hBox3DMapSelection.addWidget(self.check3DMapSelection)
        self.text3DMapName = QLineEdit()
        self.text3DMapName.setText("Iceland.tif")
        hBox3DMapSelection.addWidget(self.text3DMapName)
        vBox2.addLayout(hBox3DMapSelection)
        
        hBox3D = QHBoxLayout()
        label3DMargin = QLabel('Margin')
        hBox3D.addWidget(label3DMargin)
        self.spinbox3DMargin = QSpinBox()
        self.spinbox3DMargin.setRange(50,1000)
        self.spinbox3DMargin.setValue(100)
        self.spinbox3DMargin.setSingleStep(10)
        hBox3D.addWidget(self.spinbox3DMargin)
        
        labelSpace = QLabel('  ')
        hBox3D.addWidget(labelSpace)
        
        label3DElevationScale = QLabel('Elev. scale')
        hBox3D.addWidget(label3DElevationScale)
        self.spinbox3DElevationScale = QDoubleSpinBox()
        self.spinbox3DElevationScale.setRange(1,50)
        self.spinbox3DElevationScale.setSingleStep(0.1)
        hBox3D.addWidget(self.spinbox3DElevationScale)
        
        hBox3D.addWidget(labelSpace)
        
        label3DOSMZoom = QLabel('Zoom')
        hBox3D.addWidget(label3DOSMZoom)
        self.spinbox3DOSMZoom = QSpinBox()
        self.spinbox3DOSMZoom.setRange(8,15)
        self.spinbox3DOSMZoom.setValue(13)
        self.spinbox3DOSMZoom.setSingleStep(1)
        hBox3D.addWidget(self.spinbox3DOSMZoom)
        
        hBox3D.addWidget(labelSpace)
        
        self.check3DOSMInvert = QCheckBox("Invert")
        self.check3DOSMInvert.setChecked(False)
        hBox3D.addWidget(self.check3DOSMInvert)
        vBox2.addLayout(hBox3D)
        
        vBox_left.addLayout(vBox2)
        
        # 3rd stats tree
        self.tree = QTreeView()
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.treemodel = QtGui.QStandardItemModel()
        self.treemodel.setHorizontalHeaderLabels(['Name', 'Value'])
        self.tree.setModel(self.treemodel)
        self.tree.setUniformRowHeights(True)
        self.tree.setColumnWidth(0, 200)
        
        #index = self.treemodel.indexFromItem(parent1)
        #self.tree.expand(index)
        #selmod = self.tree.selectionModel()
        #index2 = self.treemodel.indexFromItem(child2)
        #selmod.select(index2, QtCore.QItemSelectionModel.Select|QtCore.QItemSelectionModel.Rows)
        vBox_left.addWidget(self.tree)
        
        # 4th text, containing text messages/errors
        self.textWarningConsole = QTextEdit()
        self.textWarningConsole.setReadOnly(True)
        self.textWarningConsole.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textWarningConsole.clear()
        self.textWarningConsole.setMaximumHeight(50)
        vBox_left.addWidget(self.textWarningConsole)
        
        # I put "vBox_left" inside a widget and then the widget inside "hBox"
        # instead of just doing "hBox.addLayout(vBox_left) so I can set its
        # maximum width.
        vBox_left_widget = QWidget()
        vBox_left_widget.setLayout(vBox_left)
        vBox_left_widget.setMinimumWidth(400)
        vBox_left_widget.setMaximumWidth(500)
        hBox.addWidget(vBox_left_widget)
        
        # Vertical right column
        self.tab = QTabWidget()
        
        # Tab 1: Summary: elevation and speed
        tab1 = QWidget()
        # The tab layout
        vBox_tab = QVBoxLayout()
        vBox_tab.setSpacing(5)
        # Plot area
        self.plotEmbeddedElevationAndSpeed = EmbeddedPlot_ElevationSpeed(width=5, height=4, dpi=100)
        self.plotEmbeddedElevationAndSpeed.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar1 = NavigationToolbar(self.plotEmbeddedElevationAndSpeed, self.scatola)
        # Add widgets to the layout
        vBox_tab.addWidget(self.plotEmbeddedElevationAndSpeed)
        vBox_tab.addWidget(self.mpl_toolbar1)
        # Associate the layout to the tab
        tab1.setLayout(vBox_tab)
        
        # Tab 2: 3D plot
        tab2 = QWidget()
        # The tab layout
        vBox_tab = QVBoxLayout()
        vBox_tab.setSpacing(5)
        # Area
        self.map3d = MayaviQWidget()
        # Add widgets to the layout
        vBox_tab.addWidget(self.map3d)
        # Associate the layout to the tab
        tab2.setLayout(vBox_tab)
        
        """
        # Tab 4: html 2D map
        tab4 = QWidget()
        # The tab layout
        vBox_tab = QVBoxLayout()
        vBox_tab.setSpacing(5)
        # Area
        self.browser = QtWebEngineWidgets.QWebEngineView()
        self.browser.load(QtCore.QUrl("osm.html"))
        self.browser.show()
        # Add widgets to the layout
        vBox_tab.addWidget(self.browser)
        # Associate the layout to the tab
        tab4.setLayout(vBox_tab)
        """
        
        # Tab 3: Details
        tab3 = QWidget()
        # The tab layout
        vBox_tab = QVBoxLayout()
        vBox_tab.setSpacing(5)
        # Plot area
        self.plotEmbeddedDetails = EmbeddedPlot_Details(width=5, height=4, dpi=100)
        self.plotEmbeddedDetails.setMinimumWidth(800)
        # Add toolbar to the plot
        self.mpl_toolbar2 = NavigationToolbar(self.plotEmbeddedDetails, self.scatola)
        # Add widgets to the layout
        vBox_tab.addWidget(self.plotEmbeddedDetails)
        vBox_tab.addWidget(self.mpl_toolbar2)
        # Associate the layout to the tab
        tab3.setLayout(vBox_tab)
                
        # Associate tabs
        self.tab.addTab(tab1, "Summary")
        self.tab.addTab(tab2, "3D")
        # self.tab.addTab(tab6, "Map")
        self.tab.addTab(tab3, "Details")
        
        hBox.addWidget(self.tab)
        
        # Setting hBox as main box
        self.scatola.setLayout(hBox)
        self.setCentralWidget(self.scatola)
        
        # Application settings
        self.setWindowTitle('TrackAnalyser')
        self.setWindowIcon((QtGui.QIcon('icons/app.png')))
        self.setGeometry(100, 100, 1200, 700)
        self.show()


def main():
    # Creating the application
    app = 0
    app = QApplication(sys.argv)
    main = MainWindow()
    app.setActiveWindow(main)
    
    # Showing the right icon in the taskbar
    if platform.system() == "Darwin":
        # On MAC
        pass
    elif platform.system() == 'Windows':
        # On Windows
        myappid = 'Ste.Sport.TrackAnalyser.v0.1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    main.show()
    
    # I added this line to rpevent the app from crashing on exit. The app was
    # closing fine when I was using pyqt4 and pyqt5 but started crashing when
    # I started using qtpy. This makes me think that my code is fine but
    # the way qtpy is implemented causes random behaviours on exit. After
    # trying different suggestions, this is the only one that works:
    sip.setdestroyonexit(False)
    
    sys.exit(app.exec_())
    #app.closeAllWindows()

if __name__ == "__main__":
    main()
