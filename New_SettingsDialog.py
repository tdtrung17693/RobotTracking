# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_dialog_2.ui'
#
# Created: Tue Apr 18 08:49:18 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Settings(object):
    def setupUi(self, Settings):
        Settings.setObjectName("Settings")
        Settings.resize(715, 648)
        self.verticalLayout = QtGui.QVBoxLayout(Settings)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtGui.QGroupBox(Settings)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_10 = QtGui.QLabel(self.groupBox)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_11 = QtGui.QLabel(self.groupBox)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 1, 1, 1)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalSlider_2 = QtGui.QSlider(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_2.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_2.setSizePolicy(sizePolicy)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalLayout_5.addWidget(self.horizontalSlider_2)
        self.lineEdit_3 = QtGui.QLineEdit(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
        self.lineEdit_3.setSizePolicy(sizePolicy)
        self.lineEdit_3.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_5.addWidget(self.lineEdit_3)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 1, 0, 1, 1)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.initial_pwm_slider_2 = QtGui.QSlider(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.initial_pwm_slider_2.sizePolicy().hasHeightForWidth())
        self.initial_pwm_slider_2.setSizePolicy(sizePolicy)
        self.initial_pwm_slider_2.setOrientation(QtCore.Qt.Horizontal)
        self.initial_pwm_slider_2.setObjectName("initial_pwm_slider_2")
        self.horizontalLayout_9.addWidget(self.initial_pwm_slider_2)
        self.lineEdit_4 = QtGui.QLineEdit(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_4.sizePolicy().hasHeightForWidth())
        self.lineEdit_4.setSizePolicy(sizePolicy)
        self.lineEdit_4.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_9.addWidget(self.lineEdit_4)
        self.gridLayout_2.addLayout(self.horizontalLayout_9, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_8 = QtGui.QGroupBox(Settings)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout = QtGui.QGridLayout(self.groupBox_8)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_9 = QtGui.QGroupBox(self.groupBox_8)
        self.groupBox_9.setObjectName("groupBox_9")
        self.horizontalLayout_14 = QtGui.QHBoxLayout(self.groupBox_9)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.verticalLayout_20 = QtGui.QVBoxLayout()
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.label_19 = QtGui.QLabel(self.groupBox_9)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_20.addWidget(self.label_19)
        self.spinBox_upper_H = QtGui.QSpinBox(self.groupBox_9)
        self.spinBox_upper_H.setObjectName("spinBox_upper_H")
        self.verticalLayout_20.addWidget(self.spinBox_upper_H)
        self.upper_H = QtGui.QSlider(self.groupBox_9)
        self.upper_H.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.upper_H.sizePolicy().hasHeightForWidth())
        self.upper_H.setSizePolicy(sizePolicy)
        self.upper_H.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.upper_H.setOrientation(QtCore.Qt.Vertical)
        self.upper_H.setObjectName("upper_H")
        self.verticalLayout_20.addWidget(self.upper_H)
        self.horizontalLayout_14.addLayout(self.verticalLayout_20)
        self.verticalLayout_21 = QtGui.QVBoxLayout()
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.label_20 = QtGui.QLabel(self.groupBox_9)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_21.addWidget(self.label_20)
        self.spinBox_upper_S = QtGui.QSpinBox(self.groupBox_9)
        self.spinBox_upper_S.setObjectName("spinBox_upper_S")
        self.verticalLayout_21.addWidget(self.spinBox_upper_S)
        self.upper_S = QtGui.QSlider(self.groupBox_9)
        self.upper_S.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.upper_S.sizePolicy().hasHeightForWidth())
        self.upper_S.setSizePolicy(sizePolicy)
        self.upper_S.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.upper_S.setOrientation(QtCore.Qt.Vertical)
        self.upper_S.setObjectName("upper_S")
        self.verticalLayout_21.addWidget(self.upper_S)
        self.horizontalLayout_14.addLayout(self.verticalLayout_21)
        self.verticalLayout_22 = QtGui.QVBoxLayout()
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.label_21 = QtGui.QLabel(self.groupBox_9)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.verticalLayout_22.addWidget(self.label_21)
        self.spinBox_upper_V = QtGui.QSpinBox(self.groupBox_9)
        self.spinBox_upper_V.setObjectName("spinBox_upper_V")
        self.verticalLayout_22.addWidget(self.spinBox_upper_V)
        self.upper_V = QtGui.QSlider(self.groupBox_9)
        self.upper_V.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.upper_V.sizePolicy().hasHeightForWidth())
        self.upper_V.setSizePolicy(sizePolicy)
        self.upper_V.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.upper_V.setOrientation(QtCore.Qt.Vertical)
        self.upper_V.setObjectName("upper_V")
        self.verticalLayout_22.addWidget(self.upper_V)
        self.horizontalLayout_14.addLayout(self.verticalLayout_22)
        self.gridLayout.addWidget(self.groupBox_9, 2, 0, 1, 1)
        self.groupBox_10 = QtGui.QGroupBox(self.groupBox_8)
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_15 = QtGui.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.verticalLayout_11 = QtGui.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_12 = QtGui.QLabel(self.groupBox_10)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_11.addWidget(self.label_12)
        self.spinBox_lower_H = QtGui.QSpinBox(self.groupBox_10)
        self.spinBox_lower_H.setObjectName("spinBox_lower_H")
        self.verticalLayout_11.addWidget(self.spinBox_lower_H)
        self.lower_H = QtGui.QSlider(self.groupBox_10)
        self.lower_H.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lower_H.sizePolicy().hasHeightForWidth())
        self.lower_H.setSizePolicy(sizePolicy)
        self.lower_H.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lower_H.setOrientation(QtCore.Qt.Vertical)
        self.lower_H.setObjectName("lower_H")
        self.verticalLayout_11.addWidget(self.lower_H)
        self.horizontalLayout_15.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtGui.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_22 = QtGui.QLabel(self.groupBox_10)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_12.addWidget(self.label_22)
        self.spinBox_lower_S = QtGui.QSpinBox(self.groupBox_10)
        self.spinBox_lower_S.setObjectName("spinBox_lower_S")
        self.verticalLayout_12.addWidget(self.spinBox_lower_S)
        self.lower_S = QtGui.QSlider(self.groupBox_10)
        self.lower_S.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lower_S.sizePolicy().hasHeightForWidth())
        self.lower_S.setSizePolicy(sizePolicy)
        self.lower_S.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lower_S.setOrientation(QtCore.Qt.Vertical)
        self.lower_S.setObjectName("lower_S")
        self.verticalLayout_12.addWidget(self.lower_S)
        self.verticalLayout_12.setAlignment(self.lower_S, QtCore.Qt.AlignHCenter)
        self.horizontalLayout_15.addLayout(self.verticalLayout_12)
        self.verticalLayout_13 = QtGui.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_23 = QtGui.QLabel(self.groupBox_10)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_13.addWidget(self.label_23)
        self.spinBox_lower_V = QtGui.QSpinBox(self.groupBox_10)
        self.spinBox_lower_V.setObjectName("spinBox_lower_V")
        self.verticalLayout_13.addWidget(self.spinBox_lower_V)
        self.lower_V = QtGui.QSlider(self.groupBox_10)
        self.lower_V.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lower_V.sizePolicy().hasHeightForWidth())
        self.lower_V.setSizePolicy(sizePolicy)
        self.lower_V.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lower_V.setOrientation(QtCore.Qt.Vertical)
        self.lower_V.setObjectName("lower_V")
        self.verticalLayout_13.addWidget(self.lower_V)
        self.horizontalLayout_15.addLayout(self.verticalLayout_13)
        self.gridLayout.addWidget(self.groupBox_10, 2, 1, 1, 1)
        self.param_list = QtGui.QComboBox(self.groupBox_8)
        self.param_list.setObjectName("param_list")
        self.param_list.addItem("")
        self.param_list.addItem("")
        self.param_list.addItem("")
        self.gridLayout.addWidget(self.param_list, 0, 0, 1, 2)
        self.preview_btn = QtGui.QPushButton(self.groupBox_8)
        self.preview_btn.setObjectName("preview_btn")
        self.gridLayout.addWidget(self.preview_btn, 1, 0, 1, 2)
        self.verticalLayout.addWidget(self.groupBox_8)
        self.apply_btn = QtGui.QPushButton(Settings)
        self.apply_btn.setObjectName("apply_btn")
        self.verticalLayout.addWidget(self.apply_btn)
        self.save_n_close_btn = QtGui.QPushButton(Settings)
        self.save_n_close_btn.setObjectName("save_n_close_btn")
        self.verticalLayout.addWidget(self.save_n_close_btn)
        self.verticalLayout_11.setAlignment(self.lower_H, QtCore.Qt.AlignHCenter)
        self.verticalLayout_12.setAlignment(self.lower_S, QtCore.Qt.AlignHCenter)
        self.verticalLayout_13.setAlignment(self.lower_V, QtCore.Qt.AlignHCenter)
        self.verticalLayout_20.setAlignment(self.upper_H, QtCore.Qt.AlignHCenter)
        self.verticalLayout_21.setAlignment(self.upper_S, QtCore.Qt.AlignHCenter)
        self.verticalLayout_22.setAlignment(self.upper_V, QtCore.Qt.AlignHCenter)

        self.retranslateUi(Settings)
        QtCore.QMetaObject.connectSlotsByName(Settings)

    def retranslateUi(self, Settings):
        Settings.setWindowTitle(QtGui.QApplication.translate("Settings", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Settings", "Robot Controller Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Settings", "Goal Offset", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Settings", "Initial PWM [ 0 -> 255 ]", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_8.setTitle(QtGui.QApplication.translate("Settings", "Color Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_9.setTitle(QtGui.QApplication.translate("Settings", "Upper", None, QtGui.QApplication.UnicodeUTF8))
        self.label_19.setText(QtGui.QApplication.translate("Settings", "H", None, QtGui.QApplication.UnicodeUTF8))
        self.label_20.setText(QtGui.QApplication.translate("Settings", "S", None, QtGui.QApplication.UnicodeUTF8))
        self.label_21.setText(QtGui.QApplication.translate("Settings", "V", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_10.setTitle(QtGui.QApplication.translate("Settings", "Lower", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Settings", "H", None, QtGui.QApplication.UnicodeUTF8))
        self.label_22.setText(QtGui.QApplication.translate("Settings", "S", None, QtGui.QApplication.UnicodeUTF8))
        self.label_23.setText(QtGui.QApplication.translate("Settings", "V", None, QtGui.QApplication.UnicodeUTF8))
        self.param_list.setItemText(0, QtGui.QApplication.translate("Settings", "Head", None, QtGui.QApplication.UnicodeUTF8))
        self.param_list.setItemText(1, QtGui.QApplication.translate("Settings", "Tail", None, QtGui.QApplication.UnicodeUTF8))
        self.param_list.setItemText(2, QtGui.QApplication.translate("Settings", "Obstacles", None, QtGui.QApplication.UnicodeUTF8))
        self.preview_btn.setText(QtGui.QApplication.translate("Settings", "Preview", None, QtGui.QApplication.UnicodeUTF8))
        self.apply_btn.setText(QtGui.QApplication.translate("Settings", "Apply", None, QtGui.QApplication.UnicodeUTF8))
        self.save_n_close_btn.setText(QtGui.QApplication.translate("Settings", "Save and Close", None, QtGui.QApplication.UnicodeUTF8))

