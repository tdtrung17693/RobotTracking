# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'serial_dialog.ui'
#
# Created: Sat Apr 08 14:59:14 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(484, 448)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.transmit_list = QtGui.QListWidget(Dialog)
        self.transmit_list.setObjectName("transmit_list")
        self.verticalLayout.addWidget(self.transmit_list)
        self.received_list = QtGui.QListWidget(Dialog)
        self.received_list.setObjectName("received_list")
        self.verticalLayout.addWidget(self.received_list)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.autoscroll_chkbox = QtGui.QCheckBox(Dialog)
        self.autoscroll_chkbox.setObjectName("autoscroll_chkbox")
        self.horizontalLayout.addWidget(self.autoscroll_chkbox)
        self.clear_btn = QtGui.QPushButton(Dialog)
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout.addWidget(self.clear_btn)
        self.close_btn = QtGui.QPushButton(Dialog)
        self.close_btn.setObjectName("close_btn")
        self.horizontalLayout.addWidget(self.close_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.autoscroll_chkbox.setText(QtGui.QApplication.translate("Dialog", "Autoscroll", None, QtGui.QApplication.UnicodeUTF8))
        self.clear_btn.setText(QtGui.QApplication.translate("Dialog", "Clear", None, QtGui.QApplication.UnicodeUTF8))
        self.close_btn.setText(QtGui.QApplication.translate("Dialog", "Close", None, QtGui.QApplication.UnicodeUTF8))

