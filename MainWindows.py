# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created: Mon May 15 16:40:01 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

from T_QLabel import T_QLabel


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1757, 1026)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralWidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.formLayout = QtGui.QFormLayout(self.centralWidget)
        self.formLayout.setObjectName("formLayout")
        self.orig_img = T_QLabel(self.centralWidget)
        self.orig_img.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orig_img.sizePolicy().hasHeightForWidth())
        self.orig_img.setSizePolicy(sizePolicy)
        self.orig_img.setMinimumSize(QtCore.QSize(640, 480))
        self.orig_img.setObjectName("orig_img")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.orig_img)
        self.contour_img = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.contour_img.sizePolicy().hasHeightForWidth())
        self.contour_img.setSizePolicy(sizePolicy)
        self.contour_img.setMinimumSize(QtCore.QSize(640, 480))
        self.contour_img.setObjectName("contour_img")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.contour_img)
        self.thresholded_img = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thresholded_img.sizePolicy().hasHeightForWidth())
        self.thresholded_img.setSizePolicy(sizePolicy)
        self.thresholded_img.setMinimumSize(QtCore.QSize(640, 480))
        self.thresholded_img.setObjectName("thresholded_img")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.thresholded_img)
        self.plot_img = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_img.sizePolicy().hasHeightForWidth())
        self.plot_img.setSizePolicy(sizePolicy)
        self.plot_img.setMinimumSize(QtCore.QSize(640, 480))
        self.plot_img.setObjectName("plot_img")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.plot_img)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1757, 20))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtGui.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.dockWidget_2 = QtGui.QDockWidget(MainWindow)
        self.dockWidget_2.setFloating(True)
        self.dockWidget_2.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.dockWidget_2.setObjectName("dockWidget_2")
        self.dockWidgetContents_2 = QtGui.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.show_plots_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.show_plots_btn.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.show_plots_btn, 1, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 13, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 22, 0, 1, 1)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.get_direction_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.get_direction_btn.setObjectName("get_direction_btn")
        self.horizontalLayout_6.addWidget(self.get_direction_btn)
        self.clear_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout_6.addWidget(self.clear_btn)
        self.gridLayout_2.addLayout(self.horizontalLayout_6, 6, 0, 1, 1)
        self.quit_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.quit_btn.setObjectName("quit_btn")
        self.gridLayout_2.addWidget(self.quit_btn, 23, 0, 1, 1)
        self.color_settings_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.color_settings_btn.setObjectName("color_settings_btn")
        self.gridLayout_2.addWidget(self.color_settings_btn, 0, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtGui.QLabel(self.dockWidgetContents_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.serial_port_list = QtGui.QComboBox(self.dockWidgetContents_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.serial_port_list.sizePolicy().hasHeightForWidth())
        self.serial_port_list.setSizePolicy(sizePolicy)
        self.serial_port_list.setObjectName("serial_port_list")
        self.horizontalLayout.addWidget(self.serial_port_list)
        self.rescan_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.rescan_btn.setObjectName("rescan_btn")
        self.horizontalLayout.addWidget(self.rescan_btn)
        self.connect_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.connect_btn.sizePolicy().hasHeightForWidth())
        self.connect_btn.setSizePolicy(sizePolicy)
        self.connect_btn.setObjectName("connect_btn")
        self.horizontalLayout.addWidget(self.connect_btn)
        self.gridLayout_2.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.turnLeft_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.turnLeft_btn.sizePolicy().hasHeightForWidth())
        self.turnLeft_btn.setSizePolicy(sizePolicy)
        self.turnLeft_btn.setObjectName("turnLeft_btn")
        self.horizontalLayout_2.addWidget(self.turnLeft_btn)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.forward_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.forward_btn.setObjectName("forward_btn")
        self.verticalLayout.addWidget(self.forward_btn)
        self.stop_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.stop_btn.setObjectName("stop_btn")
        self.verticalLayout.addWidget(self.stop_btn)
        self.backward_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.backward_btn.setObjectName("backward_btn")
        self.verticalLayout.addWidget(self.backward_btn)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.turnRight_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.turnRight_btn.sizePolicy().hasHeightForWidth())
        self.turnRight_btn.setSizePolicy(sizePolicy)
        self.turnRight_btn.setObjectName("turnRight_btn")
        self.horizontalLayout_2.addWidget(self.turnRight_btn)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)
        self.track_btn = QtGui.QPushButton(self.dockWidgetContents_2)
        self.track_btn.setObjectName("track_btn")
        self.gridLayout_2.addWidget(self.track_btn, 16, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 50, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem1, 3, 0, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(20, 50, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem2, 5, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_2)
        self.dockWidget_2.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget_2)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.show_plots_btn.setText(QtGui.QApplication.translate("MainWindow", "Show Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.get_direction_btn.setText(QtGui.QApplication.translate("MainWindow", "Get Direction", None, QtGui.QApplication.UnicodeUTF8))
        self.clear_btn.setText(QtGui.QApplication.translate("MainWindow", "Clear Map", None, QtGui.QApplication.UnicodeUTF8))
        self.quit_btn.setText(QtGui.QApplication.translate("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.color_settings_btn.setText(QtGui.QApplication.translate("MainWindow", "Color and Controllers Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "Bluetooth Serial:", None, QtGui.QApplication.UnicodeUTF8))
        self.rescan_btn.setText(QtGui.QApplication.translate("MainWindow", "Rescan", None, QtGui.QApplication.UnicodeUTF8))
        self.connect_btn.setText(QtGui.QApplication.translate("MainWindow", "Connect", None, QtGui.QApplication.UnicodeUTF8))
        self.turnLeft_btn.setText(QtGui.QApplication.translate("MainWindow", "Turn left", None, QtGui.QApplication.UnicodeUTF8))
        self.forward_btn.setText(QtGui.QApplication.translate("MainWindow", "Forward", None, QtGui.QApplication.UnicodeUTF8))
        self.stop_btn.setText(QtGui.QApplication.translate("MainWindow", "Stop", None, QtGui.QApplication.UnicodeUTF8))
        self.backward_btn.setText(QtGui.QApplication.translate("MainWindow", "Backward", None, QtGui.QApplication.UnicodeUTF8))
        self.turnRight_btn.setText(QtGui.QApplication.translate("MainWindow", "Turn right", None, QtGui.QApplication.UnicodeUTF8))
        self.track_btn.setText(QtGui.QApplication.translate("MainWindow", "Start Tracking", None, QtGui.QApplication.UnicodeUTF8))
