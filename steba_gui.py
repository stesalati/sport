#!/usr/bin/python

import sys
from PyQt4 import QtGui, QtCore
#from matplotlib.backends import qt_compat
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import steba as ste




# http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html


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
    
class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MainWindow(QtGui.QMainWindow):
    
    def selectFileToOpen(self):
        filename = QtGui.QFileDialog.getOpenFileName()
        gpx, s = ste.LoadGPX(filename, usehtml=True)
        self.textGPXFileStructure.setHtml(s)
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
        #go.triggered.connect(self.selectFileToOpen)
        
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
        label_settings = QtGui.QLabel('Settings', cWidget)
        vBox2.addWidget(label_settings)
        
        # Track/segment selection
        hBox21 = QtGui.QHBoxLayout()
        label21 = QtGui.QLabel('Track/Segment', cWidget)
        hBox21.addWidget(label21)
        track = QtGui.QLineEdit(cWidget)
        hBox21.addWidget(track)
        segment = QtGui.QLineEdit(cWidget)
        hBox21.addWidget(segment)
        vBox2.addLayout(hBox21)
        
        # Use/don't use corrected altitude
        use_corrected_altitude = QtGui.QCheckBox("Use SRTM corrected elevation", cWidget)
        use_corrected_altitude.setChecked(False)
        vBox2.addWidget(use_corrected_altitude)
        
        # Choose processing method
        hBox_method = QtGui.QHBoxLayout()
        label_method = QtGui.QLabel('Processing method', cWidget)
        hBox_method.addWidget(label_method)
        comboBox_method = QtGui.QComboBox()
        comboBox_method.addItem("Just use available data")
        comboBox_method.addItem("Resample at 1Hz")
        hBox_method.addWidget(comboBox_method)
        vBox2.addLayout(hBox_method)
        
        # Use/don't use acceleration
        use_acceleration_in_kalman = QtGui.QCheckBox("Use acceleration in Kalman filter", cWidget)
        use_acceleration_in_kalman.setChecked(False)
        vBox2.addWidget(use_acceleration_in_kalman)
        
        # Use/don't use variance smooth
        use_variance_smooth = QtGui.QCheckBox("Use variance smooth", cWidget)
        use_variance_smooth.setChecked(False)
        vBox2.addWidget(use_variance_smooth)
        
        vBox_left.addLayout(vBox2)
        
        # 3rd vertical box, containing buttons
        hBox3 = QtGui.QHBoxLayout()
        hBox3.setSpacing(5)
        
        button1 = QtGui.QPushButton('Go!', cWidget)
        hBox3.addWidget(button1)
        
        vBox_left.addLayout(hBox3)
        
        # 4th vertical box, containing the textual output
        textOutput = QtGui.QTextEdit(cWidget)
        textOutput.setReadOnly(True)
        vBox_left.addWidget(textOutput)
        
        hBox.addLayout(vBox_left)
        
        # Vertical right column
        vBox_right = QtGui.QVBoxLayout()
        vBox_right.setSpacing(20)
        
        # Plot area        
        plot = MyStaticMplCanvas(cWidget, width=5, height=4, dpi=100)
        vBox_right.addWidget(plot)
        
        hBox.addLayout(vBox_right)

        # Setting vBox as main box
        cWidget.setLayout(hBox)
        self.setCentralWidget(cWidget)

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
