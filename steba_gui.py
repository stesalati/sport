#!/usr/bin/python
import sys
from PyQt4 import QtGui, QtCore
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import platform
import ctypes
from matplotlib.widgets import Cursor

# from matplotlib.backends import qt4_compat
# use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
#if use_pyside:
#    from PySide.QtCore import *
#    from PySide.QtGui import *
#else:
#    from PyQt4 import QtGui, QtCore

import steba as ste


"""
Documentation

PyQT
http://www.python.it/wiki/show/qttutorial/
http://zetcode.com/gui/pyqt4/menusandtoolbars/

Plots
http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
"""

FONTSIZE = 8
PLOT_FONTSIZE = 9

# Documentation
# http://stackoverflow.com/questions/36350771/matplotlib-crosshair-cursor-in-pyqt-dialog-does-not-show-up
# http://stackoverflow.com/questions/35414003/python-how-can-i-display-cursor-on-all-axes-vertically-but-only-on-horizontall
class SnaptoCursor(object):
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
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        
        # ostring = "{}m @{}m".format(y,x)
        # self.textElevation.setText(ostring)

        #self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        #print('x=%1.2f, y=%1.2f' % (x, y))
        
        # plt.draw()


class ElevationPlot(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    # Plot original/corrected altitude profile
    #def __init__(self, *args, **kwargs):
        #MyMplCanvas.__init__(self, *args, **kwargs)
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(211)
        self.axes_bottom = self.fig.add_subplot(212, sharex=self.axes)
        self.fig.set_facecolor("w")
        
        self.axes.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
        self.axes.set_ylabel("Elevation (m)", fontsize=PLOT_FONTSIZE)
        self.axes.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
        self.axes.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
        self.axes_bottom.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
        self.axes_bottom.set_ylabel("Speed (m/s)", fontsize=PLOT_FONTSIZE)
        self.axes_bottom.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
        self.axes_bottom.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
        self.fig.tight_layout()
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def update_figure(self, measurements, state_means, segment):
        self.axes, tmp_ele = ste.PlotElevation(self.axes, measurements, state_means)
        self.axes_bottom, tmp_speed = ste.PlotSpeed(self.axes_bottom, segment)
                
        # Experiment 1: add a free interactive cursor (WORKING)
        # cursor = Cursor(self.axes, useblit=False, color='red', linewidth=1)
        # def onclick(event):
        #     cursor.onmove(event)
        # self.mpl_connect('button_press_event', onclick)
        
        # Experiment 2, add an anchored interactive cursor (WORKING BUT SLOW)
        # cursor_anchored = SnaptoCursor(self.axes, tmp_ele[0], tmp_ele[1])
        # def onclick(event):
        #     cursor_anchored.mouse_move(event)
        #     self.draw()
        # self.mpl_connect('motion_notify_event', onclick)
        
        self.fig.tight_layout()
        self.draw()


class MainWindow(QtGui.QMainWindow):
    
    def selectFileToOpen(self):
        # Clear the file-structure text field
        self.textGPXFileStructure.clear()
        
        # Try to recover the last used directory
        old_directory = self.settings.value("lastdirectory").toString()
        if not old_directory:
            old_directory = "tracks"
        
        # Open the dialog box
        filename = QtGui.QFileDialog.getOpenFileName(caption='Open .gpx',
                                                     directory=old_directory,
                                                     filter="GPX files (*.gpx)")
        directory = os.path.split(str(filename))
        # Save the new directory in the application settings
        self.settings.setValue("lastdirectory", QtCore.QVariant(str(directory[0])))
        
        self.rawgpx, longest_traseg, Ntracks, Nsegments, infos = ste.LoadGPX(filename, usehtml=False)        
        self.spinTrack.setRange(0, Ntracks-1)
        self.spinTrack.setValue(longest_traseg[0])
        self.spinSegment.setRange(0, Nsegments-1)
        self.spinSegment.setValue(longest_traseg[1])
        self.textGPXFileStructure.setText(infos)
        return
        
    def Go(self):
        # Read settings from GUI
        use_variance_smooth = self.checkUseVarianceSmooth.isChecked()
        
        # Temporarily change cursor
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        # Parse the GPX file
        gpx, coords, infos = ste.ParseGPX(self.rawgpx,
                                          track_nr=int(self.spinTrack.value()),
                                          segment_nr=int(self.spinSegment.value()),
                                          use_srtm_elevation=bool(self.checkUseSRTM.isChecked()),
                                          usehtml=False)
        self.textOutput.setText(infos)
        
        # Kalman processing
        coords, measurements, state_means, state_vars, infos = ste.ApplyKalmanFilter(coords,
                                                                                     gpx,
                                                                                     method=self.comboBoxProcessingMethod.currentIndex(), 
                                                                                     use_acceleration=self.checkUseAcceleration.isChecked(),
                                                                                     variance_smooth=use_variance_smooth,
                                                                                     plot=False,
                                                                                     usehtml=False)
        self.textOutput.append(infos)
        
        new_coords, new_gpx, infos = ste.SaveDataToCoordsAndGPX(coords, state_means, usehtml=False)
        self.textOutput.append(infos)

        # Update embedded plots
        self.plotElevation.update_figure(measurements, state_means, new_gpx.tracks[0].segments[0])
        
        # Restore original cursor
        QtGui.QApplication.restoreOverrideCursor()
        
        # Generate html plot
        balloondata = {'distance': np.cumsum(ste.HaversineDistance(np.asarray(new_coords['lat']), np.asarray(new_coords['lon']))),
                       'elevation': np.asarray(new_coords['ele']),
                       'speed': None}
        ste.PlotOnMap(np.vstack((new_coords['lat'], new_coords['lon'])).T,
                      np.vstack((coords['lat'], coords['lon'])).T,
                      onmapdata=None,
                      balloondata=balloondata,
                      rdp_reduction=self.checkUseRDP.isChecked())
        
        return

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('STEBA GUI')
        self.setWindowIcon((QtGui.QIcon('icons/app.png')))
        #self.setStyle()
        self.resize(1200, 700)
        #self.move(100, 100)
        
        # Application Settings
        QtCore.QCoreApplication.setOrganizationName("Steba")
        QtCore.QCoreApplication.setOrganizationDomain("https://github.com/stesalati/sport/")
        QtCore.QCoreApplication.setApplicationName("Steba")
        self.settings = QtCore.QSettings(self)
        
        # Toolbar
        openfile = QtGui.QAction(QtGui.QIcon("icons/openfile.png"), "Open .gpx", self)
        openfile.setShortcut("Ctrl+O")
        openfile.setStatusTip("Open file")
        openfile.triggered.connect(self.selectFileToOpen)
        
        go = QtGui.QAction(QtGui.QIcon("icons/go.png"), "Go!", self)
        go.setShortcut("Ctrl+R")
        go.setStatusTip("Run analysis")
        go.triggered.connect(self.Go)
        
        sep = QtGui.QAction(self)
        sep.setSeparator(True)
        
        quitapp = QtGui.QAction(QtGui.QIcon("icons/quit.png"), "Quit", self)
        quitapp.setShortcut("Ctrl+Q")
        quitapp.setStatusTip("Quit application")
        self.connect(quitapp, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        
        self.statusBar().show()
        toolbar = self.addToolBar('My tool')
        toolbar.addAction(openfile)
        toolbar.addAction(go)
        toolbar.addAction(quitapp)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
  
        # Main area
        self.main_frame = QtGui.QWidget(self)
        
        # Main horizontal impagination
        hBox = QtGui.QHBoxLayout()
        hBox.setSpacing(20)
        
        # Vertical left column
        vBox_left = QtGui.QVBoxLayout()
        vBox_left.setSpacing(20)
        
        # 1st horizontal box, containing label and text
        hBox1 = QtGui.QHBoxLayout()
        hBox1.setSpacing(5)
        
        self.textGPXFileStructure = QtGui.QTextEdit(self.main_frame)
        self.textGPXFileStructure.setReadOnly(True)
        self.textGPXFileStructure.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textGPXFileStructure.clear()
        
        #sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        #sizePolicy.setHorizontalStretch(1)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.textGPXFileStructure.sizePolicy().hasHeightForWidth())
        
        self.textGPXFileStructure.setMaximumHeight(150)   
        #self.textGPXFileStructure.setSizePolicy(sizePolicy)
        hBox1.addWidget(self.textGPXFileStructure)
        
        vBox_left.addLayout(hBox1)
        
        # 2nd vertical box, containing several horizontal boxes, one for each setting
        vBox2 = QtGui.QVBoxLayout()
        vBox2.setSpacing(5)
        
        # Just the group label
        labelSettings = QtGui.QLabel('Settings', self.main_frame)
        vBox2.addWidget(labelSettings)
        
        # Track/segment selection
        hBox21 = QtGui.QHBoxLayout()
        labelTrack = QtGui.QLabel('Track/Segment', self.main_frame)
        hBox21.addWidget(labelTrack)
        
        self.spinTrack = QtGui.QSpinBox(self.main_frame)
        self.spinTrack.setRange(0, 100)
        self.spinTrack.setValue(0)
        self.spinTrack.setSingleStep(1)
        hBox21.addWidget(self.spinTrack)
        self.spinSegment = QtGui.QSpinBox(self.main_frame)
        self.spinSegment.setRange(0, 100)
        self.spinSegment.setValue(0)
        self.spinSegment.setSingleStep(1)
        hBox21.addWidget(self.spinSegment)
        vBox2.addLayout(hBox21)
        
        # Use/don't use corrected altitude
        self.checkUseSRTM = QtGui.QCheckBox("Use SRTM corrected elevation", self.main_frame)
        self.checkUseSRTM.setChecked(False)
        vBox2.addWidget(self.checkUseSRTM)
        
        # Choose processing method
        hBoxProcessingMethod = QtGui.QHBoxLayout()
        labelProcessingMethod = QtGui.QLabel('Processing method', self.main_frame)
        hBoxProcessingMethod.addWidget(labelProcessingMethod)
        self.comboBoxProcessingMethod = QtGui.QComboBox()
        self.comboBoxProcessingMethod.addItem("Just use available data")
        self.comboBoxProcessingMethod.addItem("Resample at 1Hz")
        hBoxProcessingMethod.addWidget(self.comboBoxProcessingMethod)
        vBox2.addLayout(hBoxProcessingMethod)
        
        # Use/don't use acceleration
        self.checkUseAcceleration = QtGui.QCheckBox("Use acceleration in Kalman filter", self.main_frame)
        self.checkUseAcceleration.setChecked(False)
        vBox2.addWidget(self.checkUseAcceleration)
        
        # Use/don't use variance smooth
        self.checkUseVarianceSmooth = QtGui.QCheckBox("Use variance smooth", self.main_frame)
        self.checkUseVarianceSmooth.setChecked(False)
        vBox2.addWidget(self.checkUseVarianceSmooth)
        
        # Use/don't reduction algorithm for plotting on the map
        self.checkUseRDP = QtGui.QCheckBox("Allow using RDP to reduce number of points displayed", self.main_frame)
        self.checkUseRDP.setChecked(False)
        vBox2.addWidget(self.checkUseRDP)
        
        vBox_left.addLayout(vBox2)
                
        # 3rd vertical box, containing the textual output
        self.textOutput = QtGui.QTextEdit(self.main_frame)
        self.textOutput.setReadOnly(True)
        self.textOutput.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textOutput.clear()
        vBox_left.addWidget(self.textOutput)
        
        hBox.addLayout(vBox_left)
        
        # Vertical right column
        vBox_right = QtGui.QVBoxLayout()
        vBox_right.setSpacing(20)
        
        # Plot area
        self.plotElevation = ElevationPlot(self.main_frame, width=5, height=4, dpi=100)
        self.plotElevation.setMinimumWidth(800)
        
        # Add elevation/speed output fields
        #hBoxElevationDistance = QtGui.QHBoxLayout()
        #labelElevation = QtGui.QLabel('Elevation at point:', self.main_frame)
        #hBoxElevationDistance.addWidget(labelElevation)
        #self.textElevation = QtGui.QLineEdit(self.main_frame)
        #self.textElevation.setReadOnly(True)
        #self.textElevation.setFont(QtGui.QFont("Courier New", FONTSIZE))
        #self.textElevation.clear()
        #hBoxElevationDistance.addWidget(self.textElevation)
        
        # Add toolbar to the plot
        self.mpl_toolbar = NavigationToolbar(self.plotElevation, self.main_frame)
        
        vBox_right.addWidget(self.plotElevation)
        #vBox_right.addLayout(hBoxElevationDistance)
        vBox_right.addWidget(self.mpl_toolbar)
        
        hBox.addLayout(vBox_right)

        # Setting vBox as main box
        self.main_frame.setLayout(hBox)
        self.setCentralWidget(self.main_frame)

app = QtGui.QApplication(sys.argv)
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

