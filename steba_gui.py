import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

# from matplotlib.backends import qt4_compat
# use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
#if use_pyside:
#    from PySide.QtCore import *
#    from PySide.QtGui import *
#else:
#    from PyQt4 import QtGui, QtCore
from PyQt4 import QtGui, QtCore

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

class ElevationPlot(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    # Plot original/corrected altitude profile
    #def __init__(self, *args, **kwargs):
        #MyMplCanvas.__init__(self, *args, **kwargs)
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(211)
        self.axes_bottom = fig.add_subplot(212)
        fig.tight_layout()
        fig.set_facecolor("w")

        self.compute_initial_figure()
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setFocusPolicy(self,
                                    QtCore.Qt.StrongFocus)
        FigureCanvas.setFocus(self)
        
    def compute_initial_figure(self):
        self.axes.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
        self.axes.set_ylabel("Elevation (m)", fontsize=PLOT_FONTSIZE)
        self.axes.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
        self.axes.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
        self.axes_bottom.set_xlabel("Distance (m)", fontsize=PLOT_FONTSIZE)
        self.axes_bottom.set_ylabel("Speed (m/s)", fontsize=PLOT_FONTSIZE)
        self.axes_bottom.tick_params(axis='x', labelsize=PLOT_FONTSIZE)
        self.axes_bottom.tick_params(axis='y', labelsize=PLOT_FONTSIZE)
    
    def update_figure(self, measurements, state_means, segment):
        self.axes = ste.PlotElevation(self.axes, measurements, state_means)
        self.axes_bottom = ste.PlotSpeed(self.axes_bottom, segment)
        self.draw()


class MainWindow(QtGui.QMainWindow):
    
    def selectFileToOpen(self):
        self.textGPXFileStructure.clear()
        filename = QtGui.QFileDialog.getOpenFileName()
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
        
        # Add toolbar to the plot
        self.mpl_toolbar = NavigationToolbar(self.plotElevation, self.main_frame)
        
        
        vBox_right.addWidget(self.plotElevation)
        vBox_right.addWidget(self.mpl_toolbar)
        
        hBox.addLayout(vBox_right)

        # Setting vBox as main box
        self.main_frame.setLayout(hBox)
        self.setCentralWidget(self.main_frame)

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
# Alternative:
# app.exec_()

