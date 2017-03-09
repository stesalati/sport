#!/usr/bin/python

import sys
from PyQt4 import QtGui, QtCore
#from matplotlib.backends import qt_compat
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import steba as ste


"""
Documentation

PyQT
http://www.python.it/wiki/show/qttutorial/
http://zetcode.com/gui/pyqt4/menusandtoolbars/

Plots
http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
"""


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass
        
class ElevationPlot(MyMplCanvas):
    # Plot original/corrected altitude profile

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self, measurements, state_means):
        self.axes.cla()
        self.axes.plot(measurements[:,2], color="0.5", linestyle="None", marker=".")
        self.axes.plot(state_means[:,2], color="r", linestyle="-", marker="None")
        self.axes.legend(['Measured', 'Estimated'])
        self.axes.grid(True)
        self.draw()

class MainWindow(QtGui.QMainWindow):
    
    def selectFileToOpen(self):
        filename = QtGui.QFileDialog.getOpenFileName()
        self.gpx, infos = ste.LoadGPX(filename, usehtml=True)
        self.textGPXFileStructure.setHtml(infos)
        return
        
    def Go(self):
        # Read settings from GUI
        track_nr = int(self.spinTrack.value())
        segment_nr = int(self.spinSegment.value())
        usesrtm = bool(self.checkUseSRTM.isChecked())
        method = self.comboBoxProcessingMethod.currentIndex()
        usea_cceleration = self.checkUseAcceleration.isChecked()
        use_variance_smooth = self.checkUseVarianceSmooth.isChecked()
        
        # Parse the GPX file
        self.gpx, self.coords, infos = ste.ParseGPX(self.gpx, track_nr, segment_nr, use_srtm_elevation=usesrtm, usehtml=True)
        self.textOutput.setHtml(infos)
        
        # Kalman processing
        self.coords, self.measurements, self.state_means, self.state_vars, infos = ste.ApplyKalmanFilter(self.coords,
                                                                                        self.gpx,
                                                                                        method=method, 
                                                                                        use_acceleration=usea_cceleration,
                                                                                        variance_smooth=use_variance_smooth,
                                                                                        plot=True,
                                                                                        usehtml=True)
        self.textOutput.append(infos)
        
        self.plotElevation.update_figure(self.measurements, self.state_means)
        
        self.new_coords, self.new_gpx, infos = ste.SaveDataToCoordsAndGPX(self.coords, self.state_means, usehtml=True)
        self.textOutput.append(infos)
        
        return

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('STEBA GUI')
        self.setWindowIcon((QtGui.QIcon('icons/icon.png')))
        # Toolbar
        openfile = QtGui.QAction(QtGui.QIcon("icons/openfile.png"), "Open .gpx", self)
        openfile.setShortcut("Ctrl+O")
        openfile.setStatusTip("Open file")
        openfile.triggered.connect(self.selectFileToOpen)
        
        go = QtGui.QAction(QtGui.QIcon("icons/go2.png"), "Go!", self)
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
        cWidget = QtGui.QWidget(self)
        
        # Main horizontal impagination
        hBox = QtGui.QHBoxLayout()
        hBox.setSpacing(20)
        
        # Vertical left column
        vBox_left = QtGui.QVBoxLayout()
        vBox_left.setSpacing(20)
        
        # 1st horizontal box, containing label and text
        hBox1 = QtGui.QHBoxLayout()
        hBox1.setSpacing(5)
        
        self.textGPXFileStructure = QtGui.QTextEdit(cWidget)
        self.textGPXFileStructure.setReadOnly(True)
        self.textGPXFileStructure.clear()
        hBox1.addWidget(self.textGPXFileStructure)
        
        vBox_left.addLayout(hBox1)
        
        # 2nd vertical box, containing several horizontal boxes, one for each setting
        vBox2 = QtGui.QVBoxLayout()
        vBox2.setSpacing(5)
        
        # Just the group label
        labelSettings = QtGui.QLabel('Settings', cWidget)
        vBox2.addWidget(labelSettings)
        
        # Track/segment selection
        hBox21 = QtGui.QHBoxLayout()
        labelTrack = QtGui.QLabel('Track/Segment', cWidget)
        hBox21.addWidget(labelTrack)
        
        self.spinTrack = QtGui.QSpinBox(cWidget)
        self.spinTrack.setRange(0, 10)
        self.spinTrack.setValue(0)
        self.spinTrack.setSingleStep(1)
        hBox21.addWidget(self.spinTrack)
        self.spinSegment = QtGui.QSpinBox(cWidget)
        self.spinSegment.setRange(0, 10)
        self.spinSegment.setValue(0)
        self.spinSegment.setSingleStep(1)
        hBox21.addWidget(self.spinSegment)
        vBox2.addLayout(hBox21)
        
        # Use/don't use corrected altitude
        self.checkUseSRTM = QtGui.QCheckBox("Use SRTM corrected elevation", cWidget)
        self.checkUseSRTM.setChecked(False)
        vBox2.addWidget(self.checkUseSRTM)
        
        # Choose processing method
        hBoxProcessingMethod = QtGui.QHBoxLayout()
        labelProcessingMethod = QtGui.QLabel('Processing method', cWidget)
        hBoxProcessingMethod.addWidget(labelProcessingMethod)
        self.comboBoxProcessingMethod = QtGui.QComboBox()
        self.comboBoxProcessingMethod.addItem("Just use available data")
        self.comboBoxProcessingMethod.addItem("Resample at 1Hz")
        hBoxProcessingMethod.addWidget(self.comboBoxProcessingMethod)
        vBox2.addLayout(hBoxProcessingMethod)
        
        # Use/don't use acceleration
        self.checkUseAcceleration = QtGui.QCheckBox("Use acceleration in Kalman filter", cWidget)
        self.checkUseAcceleration.setChecked(False)
        vBox2.addWidget(self.checkUseAcceleration)
        
        # Use/don't use variance smooth
        self.checkUseVarianceSmooth = QtGui.QCheckBox("Use variance smooth", cWidget)
        self.checkUseVarianceSmooth.setChecked(False)
        vBox2.addWidget(self.checkUseVarianceSmooth)
        
        # Use/don't reduction algorithm for plotting on the map
        self.checkUseRDP = QtGui.QCheckBox("Allow using RDP to reduce number of points displayed", cWidget)
        self.checkUseRDP.setChecked(False)
        vBox2.addWidget(self.checkUseRDP)
        
        vBox_left.addLayout(vBox2)
                
        # 3rd vertical box, containing the textual output
        self.textOutput = QtGui.QTextEdit(cWidget)
        self.textOutput.setReadOnly(True)
        vBox_left.addWidget(self.textOutput)
        
        hBox.addLayout(vBox_left)
        
        # Vertical right column
        vBox_right = QtGui.QVBoxLayout()
        vBox_right.setSpacing(20)
        
        # Plot area        
        self.plotElevation = ElevationPlot(cWidget, width=5, height=4, dpi=100)
        vBox_right.addWidget(self.plotElevation)
        
        hBox.addLayout(vBox_right)

        # Setting vBox as main box
        cWidget.setLayout(hBox)
        self.setCentralWidget(cWidget)

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
