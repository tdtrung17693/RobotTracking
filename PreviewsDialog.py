# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'preview_dialog.ui'
#
# Created: Tue Apr 18 08:48:44 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Preview(object):
    def setupUi(self, Preview):
        Preview.setObjectName("Preview")
        Preview.resize(658, 467)
        self.gridLayout = QtGui.QGridLayout(Preview)
        self.gridLayout.setObjectName("gridLayout")
        self.preview_img = QtGui.QLabel(Preview)
        self.preview_img.setMinimumSize(QtCore.QSize(640, 420))
        self.preview_img.setMaximumSize(QtCore.QSize(640, 420))
        self.preview_img.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_img.setObjectName("preview_img")
        self.gridLayout.addWidget(self.preview_img, 0, 0, 1, 1)
        self.close_btn = QtGui.QPushButton(Preview)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close_btn.sizePolicy().hasHeightForWidth())
        self.close_btn.setSizePolicy(sizePolicy)
        self.close_btn.setCheckable(False)
        self.close_btn.setDefault(False)
        self.close_btn.setFlat(False)
        self.close_btn.setObjectName("close_btn")
        self.gridLayout.addWidget(self.close_btn, 1, 0, 1, 1)

        self.retranslateUi(Preview)
        QtCore.QMetaObject.connectSlotsByName(Preview)

    def retranslateUi(self, Preview):
        Preview.setWindowTitle(QtGui.QApplication.translate("Preview", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.preview_img.setText(QtGui.QApplication.translate("Preview", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.close_btn.setText(QtGui.QApplication.translate("Preview", "Close", None, QtGui.QApplication.UnicodeUTF8))

