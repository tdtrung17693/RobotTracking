# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plot_dialog.ui'
#
# Created: Mon May 15 16:43:11 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui
from pyqtgraph import GraphicsLayoutWidget

class Ui_PlotDialog(object):
    def setupUi(self, PlotDialog):
        PlotDialog.setObjectName("PlotDialog")
        PlotDialog.resize(484, 448)
        self.verticalLayout = QtGui.QVBoxLayout(PlotDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = GraphicsLayoutWidget(PlotDialog)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)

        self.retranslateUi(PlotDialog)
        QtCore.QMetaObject.connectSlotsByName(PlotDialog)

    def retranslateUi(self, PlotDialog):
        PlotDialog.setWindowTitle(QtGui.QApplication.translate("PlotDialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))

