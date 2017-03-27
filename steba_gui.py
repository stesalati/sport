#!/usr/bin/python
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QToolTip, 
                             QPushButton, QApplication, qApp,
                             QAction, QLabel, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QLineEdit, 
                             QTextEdit, QCheckBox, QComboBox,
                             QSpinBox, QSizePolicy)
from PyQt5 import QtGui, QtCore
# from PyQt5.QtCore import QSettings
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import platform
import ctypes
#from matplotlib.widgets import Cursor, MultiCursor

import steba as ste


"""
Documentation

PyQt4
http://www.python.it/wiki/show/qttutorial/
http://zetcode.com/gui/pyqt4/menusandtoolbars/
http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html

PyQt5
http://zetcode.com/gui/pyqt5/layout/
http://zetcode.com/gui/pyqt5/dialogs/
https://pythonspot.com/en/pyqt5-matplotlib/

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
            self.txt1.set_text("{}m".format(y1))
            self.txt1.set_position((x1, y1))
            self.txt2.set_text("{}m/s".format(y2))
            self.txt2.set_position((x2, y2))
        except:
            return
        plt.draw()


class EmbeddedPlot(FigureCanvas):
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
        self.axes, tmp_ele = ste.PlotElevation(self.axes, measurements, state_means)
        self.axes_bottom, tmp_speed = ste.PlotSpeed(self.axes_bottom, segment)
        
        # Add cursor
        # cursor_anchored = SingleCursorLinkedToTrace(self.axes, tmp_ele[0], tmp_ele[1])
        cursor_anchored = MultiCursorLinkedToTrace(self.axes, tmp_ele[0], tmp_ele[1],
                                                   self.axes_bottom, tmp_speed[0], tmp_speed[1])
        def onclick(event):
            cursor_anchored.mouse_move(event)
            self.draw()
        self.mpl_connect('motion_notify_event', onclick)
        
        # Alternative: cursor on both plots but not linked to the trace
        #self.multi = MultiCursor(self.fig.canvas, (self.axes, self.axes_bottom), color='r', lw=1, vertOn=True, horizOn=True)
        
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
                                               caption='Open .gpx',
                                               directory=old_directory,
                                               filter="GPX files (*.gpx)")
        if filename[0]:
            directory = os.path.split(str(filename[0]))
            # print "File to open: {}".format(filename[0])
            # Save the new directory in the application settings
            self.settings.setValue("lastdirectory", QtCore.QVariant(str(directory[0])))
            
            self.rawgpx, longest_traseg, Ntracks, Nsegments, infos = ste.LoadGPX(filename[0], usehtml=False)
            self.spinTrack.setRange(0, Ntracks-1)
            self.spinTrack.setValue(longest_traseg[0])
            self.spinSegment.setRange(0, Nsegments-1)
            self.spinSegment.setValue(longest_traseg[1])
            self.textGPXFileStructure.setText(infos)
        else:
            self.textGPXFileStructure.setText("No file was selected!")
        return
        
    def Go(self):
        # Read settings from GUI
        use_variance_smooth = self.checkUseVarianceSmooth.isChecked()
        
        # Temporarily change cursor
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
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
        self.plotEmbedded.update_figure(measurements, state_means, new_gpx.tracks[0].segments[0])
        
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
        
        return

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.initUI()
        
    def initUI(self):        
        # Application Settings
        QtCore.QCoreApplication.setOrganizationName("Steba")
        QtCore.QCoreApplication.setOrganizationDomain("https://github.com/stesalati/sport/")
        QtCore.QCoreApplication.setApplicationName("Steba")
        self.settings = QtCore.QSettings(self)
        
        # Toolbar
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
        
        self.statusBar().show()
        toolbar = self.addToolBar('My tools')
        toolbar.addAction(openfile)
        toolbar.addAction(go)
        toolbar.addAction(quitapp)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        
        # Main widget (everything that's not toolbar, statusbar or menubar must be in this widget)
        self.scatola = QWidget()
        
        # Main horizontal impagination
        hBox = QHBoxLayout()
        hBox.setSpacing(20)
        
        # Vertical left column
        vBox_left = QVBoxLayout()
        vBox_left.setSpacing(20)
        
        # 1st widget, text
        self.textGPXFileStructure = QTextEdit()
        self.textGPXFileStructure.setReadOnly(True)
        self.textGPXFileStructure.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textGPXFileStructure.clear()
        
        self.textGPXFileStructure.setMaximumHeight(150)
        vBox_left.addWidget(self.textGPXFileStructure)
        
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
        self.checkUseSRTM = QCheckBox("Use SRTM corrected elevation")
        self.checkUseSRTM.setChecked(False)
        vBox2.addWidget(self.checkUseSRTM)
        
        # Choose processing method
        hBoxProcessingMethod = QHBoxLayout()
        labelProcessingMethod = QLabel('Processing method')
        hBoxProcessingMethod.addWidget(labelProcessingMethod)
        self.comboBoxProcessingMethod = QComboBox()
        self.comboBoxProcessingMethod.addItem("Just use available data")
        self.comboBoxProcessingMethod.addItem("Resample at 1Hz")
        hBoxProcessingMethod.addWidget(self.comboBoxProcessingMethod)
        vBox2.addLayout(hBoxProcessingMethod)
        
        # Use/don't use acceleration
        self.checkUseAcceleration = QCheckBox("Use acceleration in Kalman filter")
        self.checkUseAcceleration.setChecked(False)
        vBox2.addWidget(self.checkUseAcceleration)
        
        # Use/don't use variance smooth
        self.checkUseVarianceSmooth = QCheckBox("Use variance smooth")
        self.checkUseVarianceSmooth.setChecked(False)
        vBox2.addWidget(self.checkUseVarianceSmooth)
        
        # Use/don't reduction algorithm for plotting on the map
        self.checkUseRDP = QCheckBox("Allow using RDP to reduce number of points displayed")
        self.checkUseRDP.setChecked(False)
        vBox2.addWidget(self.checkUseRDP)
        
        vBox_left.addLayout(vBox2)
                
        # 3rd text, containing the processing output
        self.textOutput = QTextEdit()
        self.textOutput.setReadOnly(True)
        self.textOutput.setFont(QtGui.QFont("Courier New", FONTSIZE))
        self.textOutput.clear()
        vBox_left.addWidget(self.textOutput)
        
        hBox.addLayout(vBox_left)
        
        # Vertical right column
        vBox_right = QVBoxLayout()
        vBox_right.setSpacing(20)
        
        # Plot area
        self.plotEmbedded = EmbeddedPlot(width=5, height=4, dpi=100)
        self.plotEmbedded.setMinimumWidth(800)
        
        # Add elevation/speed output fields
        #hBoxElevationDistance = QHBoxLayout()
        #labelElevation = QLabel('Elevation at point:', self.main_frame)
        #hBoxElevationDistance.addWidget(labelElevation)
        #self.textElevation = QLineEdit(self.main_frame)
        #self.textElevation.setReadOnly(True)
        #self.textElevation.setFont(QFont("Courier New", FONTSIZE))
        #self.textElevation.clear()
        #hBoxElevationDistance.addWidget(self.textElevation)
        
        # Add toolbar to the plot
        self.mpl_toolbar = NavigationToolbar(self.plotEmbedded, self.scatola)
        
        vBox_right.addWidget(self.plotEmbedded)
        #vBox_right.addLayout(hBoxElevationDistance)
        vBox_right.addWidget(self.mpl_toolbar)
        
        hBox.addLayout(vBox_right)
        
        # Setting hBox as main box
        self.scatola.setLayout(hBox)
        self.setCentralWidget(self.scatola)
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

