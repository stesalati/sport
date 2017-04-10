ciao
#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QApplication, qApp, QAction,
                             QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QTextEdit,
                             QCheckBox, QComboBox, QSpinBox, QSizePolicy, QTabWidget,
                             QListWidget, QListWidgetItem)
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
from matplotlib.widgets import Cursor, MultiCursor
import platform
import ctypes

import steba as ste

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
        
    def update_figure(self, measurements, state_means, segment):
        # Draw plots
        self.top_axis, tmp_ele = ste.PlotElevation(self.top_axis, measurements, state_means)
        self.bottom_axis, tmp_speed = ste.PlotSpeed(self.bottom_axis, segment)
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
        
    def update_figure(self, measurements, state_means, state_vars):
        # Draw plots
        self.top_axis, tmp_coords = ste.PlotCoordinates(self.top_axis, state_means)
        self.bottom_axis, tmp_coordsvar = ste.PlotCoordinatesVariance(self.bottom_axis, state_means, state_vars)
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
        
    def update_figure(self, measurements, state_means, state_vars):
        # Draw plots
        self.top_axis, tmp_ele = ste.PlotElevation(self.top_axis, measurements, state_means)
        self.bottom_axis, tmp_elevar = ste.PlotElevationVariance(self.bottom_axis, state_means, state_vars)
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
        
    def update_figure(self, measurements, state_means, state_vars, segment):
        # Draw plots
        self.top_axis, tmp_speed = ste.PlotSpeed(self.top_axis, segment)
        self.bottom_axis, tmp_speedvar = ste.PlotSpeedVariance(self.bottom_axis, state_means, state_vars)
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
        # Clear the file-structure text field
        self.textGPXFileStructure.clear()
        
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
        filename = QFileDialog.getOpenFileName(self,
                                               'Open .gpx',
                                               "tracks",
                                               "GPX files (*.gpx)")
        if filename[0]:
            directory = os.path.split(str(filename[0]))
            # print "File to open: {}".format(filename[0])
            # Save the new directory in the application settings
            self.settings.setValue("lastdirectory", QtCore.QVariant(str(directory[0])))
            
            self.rawgpx, longest_traseg, Ntracks, Nsegments, infos = ste.LoadGPX(filename[0])
            self.spinTrack.setRange(0, Ntracks-1)
            self.spinTrack.setValue(longest_traseg[0])
            self.spinSegment.setRange(0, Nsegments-1)
            self.spinSegment.setValue(longest_traseg[1])
            self.textGPXFileStructure.setText(infos)
        else:
            self.textGPXFileStructure.setText("No file was selected!")
        return
        
    def Go(self):
        if self.rawgpx is not None:            
            # Temporarily change cursor
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Parse the GPX file
            gpx, coords, infos = ste.ParseGPX(self.rawgpx,
                                              track_nr=int(self.spinTrack.value()),
                                              segment_nr=int(self.spinSegment.value()),
                                              use_srtm_elevation=bool(self.checkUseSRTM.isChecked()))
            self.textOutput.setText(infos)
            
            # Kalman processing
            coords, measurements, state_means, state_vars, infos = ste.ApplyKalmanFilter(coords,
                                                                                         gpx,
                                                                                         method=self.comboBoxProcessingMethod.currentIndex(), 
                                                                                         use_acceleration=self.checkUseAcceleration.isChecked(),
                                                                                         extra_smooth=self.checkExtraSmooth.isChecked(),
                                                                                         debug_plot=False)
            self.textOutput.append(infos)
            
            new_coords, new_gpx, infos = ste.SaveDataToCoordsAndGPX(coords, state_means)
            self.textOutput.append(infos)
    
            # Update embedded plots
            self.plotEmbedded1.update_figure(measurements, state_means, new_gpx.tracks[0].segments[0])
            self.plotEmbedded2.update_figure(measurements, state_means, state_vars)
            self.plotEmbedded3.update_figure(measurements, state_means, state_vars)
            self.plotEmbedded4.update_figure(measurements, state_means, state_vars, new_gpx.tracks[0].segments[0])
            
            # Restore original cursor
            QApplication.restoreOverrideCursor()
            
            # Generate html plot
            balloondata = {'distance': np.cumsum(ste.HaversineDistance(np.asarray(new_coords['lat']), np.asarray(new_coords['lon']))),
                           'elevation': np.asarray(new_coords['ele']),
                           'speed': None}
            ste.PlotOnMap(np.vstack((new_coords['lat'], new_coords['lon'])).T,
                          np.vstack((coords['lat'], coords['lon'])).T,
                          onmapdata=None,
                          balloondata=balloondata,
                          rdp_reduction=self.checkUseRDP.isChecked())
        else:
            self.textGPXFileStructure.setText("You need to open a .gpx file before!")
        return

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.initVariables()
        self.initUI()
        
    def initVariables(self):
        self.rawgpx = None
        
    def initUI(self):
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
        
        # 1st widget, text
        self.textGPXFileStructure = QTextEdit()
        self.textGPXFileStructure.setReadOnly(True)
        self.textGPXFileStructure.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textGPXFileStructure.clear()
        
        self.textGPXFileStructure.setMaximumHeight(150)
        vBox_left.addWidget(self.textGPXFileStructure)
        
        # 2nd vertical box, a list
        self.tracklist = QListWidget()
        item1 = QListWidgetItem('Example trace 1')
        item2 = QListWidgetItem('Example trace 2')
        self.tracklist.addItem(item1)
        self.tracklist.addItem(item2)
        vBox_left.addWidget(self.tracklist)
        
        # 2nd vertical box, containing several horizontal boxes, one for each setting
        vBox2 = QVBoxLayout()
        vBox2.setSpacing(5)
        
        # Just the group label
        labelSettings = QLabel('Settings')
        vBox2.addWidget(labelSettings)
        
        # Track/segment selection
        hBox21 = QHBoxLayout()
        labelTrack = QLabel('Track/Segment')
        hBox21.addWidget(labelTrack)
        
        self.spinTrack = QSpinBox()
        self.spinTrack.setRange(0, 100)
        self.spinTrack.setValue(0)
        self.spinTrack.setSingleStep(1)
        hBox21.addWidget(self.spinTrack)
        self.spinSegment = QSpinBox()
        self.spinSegment.setRange(0, 100)
        self.spinSegment.setValue(0)
        self.spinSegment.setSingleStep(1)
        hBox21.addWidget(self.spinSegment)
        vBox2.addLayout(hBox21)
        
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
