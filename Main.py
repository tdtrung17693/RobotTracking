from PySide.QtCore import *
from PySide.QtGui import *

from AStar.Algorithm import AStar
from AStar.Drawer import Drawer
from AStar.MapCreator import MapCreator
from AStar.MapNode import MapNode

from MainWindows import Ui_MainWindow
from SettingsDialog import Ui_Settings
from SerialDialog import Ui_Dialog as Ui_SerialMonitor
from PreviewsDialog import Ui_Preview
from PlotsDialog import Ui_PlotDialog

from SerialScanner import SerialScanner
from Controller import BotController

import numpy as np
import cv2
import sys
import serial

import pyqtgraph as pg

# Local module
import Config


class ControlMainWindow(QMainWindow):
    # noinspection PyUnresolvedReferences
    def __init__(self, parent=None):
        super(ControlMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()

        self.settings_dialog = QDialog(self)
        self.dialog_ui = Ui_Settings()

        self.serial_dialog = QDialog(self)
        self.serial_dialog_ui = Ui_SerialMonitor()

        self.preview_dialog = QDialog(self.settings_dialog)
        self.preview_dialog_ui = Ui_Preview()

        self.plots_dialog = QDialog(self)
        self.plots_dialog_ui = Ui_PlotDialog()

        # --- SETUP NECESSARY VARIABLES ---
        # Camera stuffs
        self.capture = None
        self.fps = None
        self.img_size = QSize(*Config.FRAME_GEOMETRY)

        # Lower and upper HSV values for thresholding
        self.tail_hsv = [[0, 0, 0], [0, 0, 0]]
        self.head_hsv = [[0, 0, 0], [0, 0, 0]]
        self.whole_body_hsv = [[0, 0, 0], [0, 0, 0]]

        # Serial connection stuffs
        self.serial_ports = []
        self.current_serial_port = ""
        self.current_debug_serial_port = ""
        self.current_serial_connection = None
        self.current_debug_serial_connection = None
        self.have_ports_available = False

        # Current position of robot
        self.current_pos = None
        self.current_distance = None
        self.current_angle = None
        self.current_real_pos = None
        self.current_head = None
        self.current_tail = None
        self.agent_size = 0

        # Clicked position on the image
        self.clicked_pos = None
        self.region = None
        self.goal = None
        self.final_point = None
        self.final_plan = None

        # Grid unit for map creating
        self.grid_unit = 20

        # Tracking state
        self.tracking_state = False

        # Controller parameter
        self.initial_pwm = 50
        self.goal_offset = 0
        self.kP_P = 0
        self.kI_P = 0
        self.kD_P = 0
        self.kP_V = 0
        self.kI_V = 0
        self.kD_V = 0
        self.controller = BotController(self.kP_P, self.kI_P, self.kD_P)

        # Graph stuff
        self.x = []
        self.real_x = []
        self.y = []
        self.real_y = []
        self.angle = []
        self.time = []

        # Plot img
        self.plot_img = np.full((self.img_size.height(), self.img_size.width(), 3), 255, np.uint8)

        # Param type
        self.param_type = 0

        # --- INITIALIZE NECESSARY THREAD ---
        self.serial_scanner = SerialScanner(self.serial_ports)
        self.serial_scanner.finished.connect(self.serial_ports_scanned)

        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self.tracking_robot)

        # --- SETUP MAIN UI ---
        self.setup_ui()

        # --- LOAD SAVED SETTINGS
        self.settings = QSettings("config.ini", QSettings.IniFormat)
        self.load_settings()

        # --- SETUP CAMERA
        self.setup_camera()

        self.update_graph = QTimer()
        self.update_graph.timeout.connect(self.do_update_graph)
        self.serial_scanner.start()

    def do_update_graph(self):
        real_tail = self.camera_to_real(self.current_tail)
        self.x.append(real_tail[0])
        self.y.append(real_tail[1])
        v = []
        omega = []
        omega_l = []
        omega_r = []

        for i in xrange(0, len(self.x)):
            if self.time[i] == 0:
                v.append(0)
                omega.append(0)
            else:
                v.append(np.sqrt((self.x[i] - self.x[i-1]) ** 2 + (self.y[i] - self.y[i-1]) ** 2) / 400)
                omega.append((self.angle[i] - self.angle[i-1])/400)

        time = map(lambda v: v*400, self.time)
        self.ui_plot_pos.plot(time, self.x, pen="b")
        self.ui_plot_pos.plot(time, self.y, pen="r")
        self.ui_plot_vel.plot(time, v, pen="b")
        self.ui_plot_ang_vel.plot(time, omega, pen="b")

    def load_settings(self):
        self.settings.beginGroup("Color")

        self.tail_hsv = self.settings.value("tail_range", [[0, 0, 0], [0, 0, 0]])
        self.head_hsv = self.settings.value("head_range", [[0, 0, 0], [0, 0, 0]])

        # k*_P for angle and k*_V for distance
        self.kP_P = float(self.settings.value("Kp_P", 0))
        self.kI_P = float(self.settings.value("Ki_P", 0))
        self.kD_P = float(self.settings.value("Kd_P", 0))

        self.kP_V = float(self.settings.value("Kp_V", 0))
        self.kI_V = float(self.settings.value("Ki_V", 0))
        self.kD_V = float(self.settings.value("Kd_V", 0))

        self.controller.k_P = self.kP_P
        self.controller.k_I = self.kI_P
        self.controller.k_D = self.kD_P

        self.goal_offset = int(self.settings.value("goal_offset", 0))
        self.initial_pwm = int(self.settings.value("initial_pwm", 50))

        self.populate_ui_value()

    def populate_ui_value(self):
        # Populate slider
        self.populate_color_slider()
        self.populate_color_spinbox()

        self.populate_controller_ui()

    def populate_controller_ui(self):
        self.dialog_ui.kP_P.setValue(self.kP_P)
        self.dialog_ui.kI_P.setValue(self.kI_P)
        self.dialog_ui.kD_P.setValue(self.kD_P)

        self.dialog_ui.kP_V.setValue(self.kP_V)
        self.dialog_ui.kI_V.setValue(self.kI_V)
        self.dialog_ui.kD_V.setValue(self.kD_V)

        self.dialog_ui.lineEdit_3.setText("{:d}".format(self.goal_offset))
        self.dialog_ui.lineEdit_4.setText("{:d}".format(self.initial_pwm))

        self.dialog_ui.horizontalSlider_2.setValue(self.goal_offset)
        self.dialog_ui.initial_pwm_slider_2.setValue(self.initial_pwm)

    def populate_color_slider(self):
        if self.param_type == 0:
            subject = self.head_hsv
        elif self.param_type == 1:
            subject = self.tail_hsv
        elif self.param_type == 2:
            pass

        self.dialog_ui.lower_H.setValue(subject[0][0])
        self.dialog_ui.lower_S.setValue(subject[0][1])
        self.dialog_ui.lower_V.setValue(subject[0][2])
        self.dialog_ui.upper_H.setValue(subject[1][0])
        self.dialog_ui.upper_S.setValue(subject[1][1])
        self.dialog_ui.upper_V.setValue(subject[1][2])

    def populate_color_spinbox(self):
        subject = [[]]

        if self.param_type == 0:
            subject = self.head_hsv
        elif self.param_type == 1:
            subject = self.tail_hsv
        elif self.param_type == 2:
            pass

        self.dialog_ui.spinBox_lower_H.setValue(subject[0][0])
        self.dialog_ui.spinBox_lower_S.setValue(subject[0][1])
        self.dialog_ui.spinBox_lower_V.setValue(subject[0][2])
        self.dialog_ui.spinBox_upper_H.setValue(subject[1][0])
        self.dialog_ui.spinBox_upper_S.setValue(subject[1][1])
        self.dialog_ui.spinBox_upper_V.setValue(subject[1][2])

    # noinspection PyAttributeOutsideInit
    def setup_ui(self):
        # --- INITIALIZE MAIN UI ---
        self.ui.setupUi(self)
        self.dialog_ui.setupUi(self.settings_dialog)
        self.serial_dialog_ui.setupUi(self.serial_dialog)
        self.preview_dialog_ui.setupUi(self.preview_dialog)
        self.plots_dialog_ui.setupUi(self.plots_dialog)

        # --- CONFIGURE UI COMPONENTS ---
        self.ui.color_settings_btn.clicked.connect(self.open_setting_dialog)
        self.ui.show_plots_btn.clicked.connect(self.open_plots_dialog)
        self.dialog_ui.kP_P.setDecimals(3)
        self.dialog_ui.kI_P.setDecimals(3)
        self.dialog_ui.kD_P.setDecimals(3)
        self.dialog_ui.kP_V.setDecimals(3)
        self.dialog_ui.kI_V.setDecimals(3)
        self.dialog_ui.kD_V.setDecimals(3)

        self.dialog_ui.initial_pwm_slider_2.setRange(0, 255)
        self.dialog_ui.initial_pwm_slider_2.valueChanged[int].connect(
            lambda value: self.dialog_ui.lineEdit_4.setText("{:d}".format(value)))

        self.dialog_ui.lineEdit_4.textChanged.connect(
            lambda value: self.dialog_ui.initial_pwm_slider_2.setValue(int(value))
        )

        self.dialog_ui.horizontalSlider_2.setRange(0, 255)
        self.dialog_ui.horizontalSlider_2.valueChanged[int].connect(
            lambda value: self.dialog_ui.lineEdit_3.setText("{:d}".format(value)))

        self.dialog_ui.lineEdit_3.textChanged.connect(
            lambda value: self.dialog_ui.horizontalSlider_2.setValue(int(value))
        )

        self.dialog_ui.preview_btn.clicked.connect(self.preview_dialog.show)
        self.dialog_ui.apply_btn.clicked.connect(self.save_settings)

        self.ui.orig_img.pictureClicked.connect(self.img_clicked)
        self.ui.orig_img.selected.connect(self.selected_region)

        # Tail Lower H-S-V sliders
        self.dialog_ui.lower_H.setRange(0, 255)
        self.dialog_ui.lower_S.setRange(0, 255)
        self.dialog_ui.lower_V.setRange(0, 255)
        self.dialog_ui.lower_H.valueChanged[int].connect(lambda value: self.update_lower("h", value))
        self.dialog_ui.lower_S.valueChanged[int].connect(lambda value: self.update_lower("s", value))
        self.dialog_ui.lower_V.valueChanged[int].connect(lambda value: self.update_lower("v", value))

        self.dialog_ui.spinBox_lower_H.setRange(0, 255)
        self.dialog_ui.spinBox_lower_S.setRange(0, 255)
        self.dialog_ui.spinBox_lower_V.setRange(0, 255)
        self.dialog_ui.spinBox_lower_H.valueChanged[int].connect(
            lambda value: self.update_lower("h", int(value), False))
        self.dialog_ui.spinBox_lower_S.valueChanged[int].connect(
            lambda value: self.update_lower("s", int(value), False))
        self.dialog_ui.spinBox_lower_V.valueChanged[int].connect(
            lambda value: self.update_lower("v", int(value), False))

        # Tail Upper H-S-V sliders
        self.dialog_ui.upper_H.setRange(0, 255)
        self.dialog_ui.upper_S.setRange(0, 255)
        self.dialog_ui.upper_V.setRange(0, 255)
        self.dialog_ui.upper_H.valueChanged[int].connect(lambda value: self.update_upper("h", value))
        self.dialog_ui.upper_S.valueChanged[int].connect(lambda value: self.update_upper("s", value))
        self.dialog_ui.upper_V.valueChanged[int].connect(lambda value: self.update_upper("v", value))

        self.dialog_ui.spinBox_upper_H.setRange(0, 255)
        self.dialog_ui.spinBox_upper_S.setRange(0, 255)
        self.dialog_ui.spinBox_upper_V.setRange(0, 255)
        self.dialog_ui.spinBox_upper_H.valueChanged[int].connect(
            lambda value: self.update_upper("h", int(value), False))
        self.dialog_ui.spinBox_upper_S.valueChanged[int].connect(
            lambda value: self.update_upper("s", int(value), False))
        self.dialog_ui.spinBox_upper_V.valueChanged[int].connect(
            lambda value: self.update_upper("v", int(value), False))

        # Select param type
        self.dialog_ui.param_list.currentIndexChanged.connect(self.choose_param_type)

        # Plots
        self.plots_dialog_ui.graphicsView.setBackground('w')
        self.ui_plot_pos = self.plots_dialog_ui.graphicsView.addPlot(title='Position')
        self.ui_plot_vel = self.plots_dialog_ui.graphicsView.addPlot(title='Velocity')
        self.plots_dialog_ui.graphicsView.nextRow()
        self.ui_plot_ang_vel = self.plots_dialog_ui.graphicsView.addPlot(title='Angular Velocity') # type: pg.PlotItem

        # Serial port list, connect button
        self.ui.serial_port_list.setDisabled(True)
        self.ui.serial_port_list.addItem("Scanning for available ports...")
        self.ui.serial_port_list.currentIndexChanged.connect(self.choose_serial_port)

        self.ui.connect_btn.clicked.connect(self.connect_bluetooth_serial)
        self.ui.connect_btn.setDisabled(True)

        # Navigation buttons
        self.ui.turnLeft_btn.clicked.connect(lambda: self.navigate('left'))
        self.ui.turnRight_btn.clicked.connect(lambda: self.navigate('right'))
        self.ui.forward_btn.clicked.connect(lambda: self.navigate('forward'))
        self.ui.backward_btn.clicked.connect(lambda: self.navigate('backward'))
        self.ui.stop_btn.clicked.connect(lambda: self.navigate('stop'))

        # Other stuffs: button send goal, start tracking, quit, .etc
        self.ui.get_direction_btn.clicked.connect(self.get_direction)
        self.ui.track_btn.clicked.connect(self.set_tracking_state)
        self.ui.track_btn.setDisabled(True)

        self.ui.clear_btn.clicked.connect(self.clear_plot)

        self.ui.quit_btn.clicked.connect(self.close)

    def setup_camera(self):
        self.capture = cv2.VideoCapture(Config.CAMERA_NUMBER)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())
        self.capture.set(cv2.CAP_PROP_FOCUS, 0)
        self.capture.set(cv2.CAP_PROP_FPS, 50)

        self.fps = QTimer()
        self.fps.timeout.connect(self.display_video_stream)
        self.fps.start(28)

    # noinspection PyUnboundLocalVariable,PyUnusedLocal
    def display_video_stream(self):
        # frame = cv2.imread("WIN_20170421_13_02_15_Pro.jpg")

        _, frame = self.capture.read()

        frame = cv2.resize(frame, self.img_size.toTuple())

        result_img = frame.copy()

        # Get robot head
        head_contour, head_masked_img = self.get_robot("head", frame)

        if len(head_contour) > 0:
            h_x, h_y, h_w, h_h = self.process_contour(head_contour, "head", result_img)

        # Get robot tail
        tail_contour, tail_masked_img = self.get_robot("tail", frame)

        if len(tail_contour) > 0:
            t_x, t_y, t_w, t_h = self.process_contour(tail_contour, "tail", result_img)

        if len(head_contour) > 0 and len(tail_contour) > 0:
            if t_w * t_h > 1000 and h_w * h_h > 1000:
                cv2.line(result_img, (t_x + t_w / 2, t_y + t_h / 2), (h_x + h_w / 2, h_y + h_h / 2), (255, 0, 0), 3)
                self.current_pos = ((t_x + h_x) / 2, (t_y + h_y) / 2)

        # Show clicked position
        # Draw background for text
        cv2.rectangle(frame, (0, 0), (640, 20), (0, 0, 0), thickness=cv2.FILLED)
        if self.goal:
            coord = tuple(self.goal) + tuple(self.camera_to_real(self.goal))

            cv2.putText(frame, "x: {:d} - y: {:d} | X: {:.3f} - Y: {:.3f}".format(
                *coord), (20, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            cv2.circle(frame, self.goal, 7, (0, 0, 255), cv2.FILLED)

            # Draw line for reference
            self.current_angle = self.calculate_current_angle()
            cv2.putText(result_img, "Current Angle: {:.2f} rad - {:.2f} deg".format(self.current_angle,
                                                                                    self.current_angle * 180 / np.pi),
                        (20, 53), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255))
        else:
            cv2.putText(frame, "Goal position not set", (20, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        if self.final_point is not None:
            self.draw_path_on_frame(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        # Convert thresholded image to RGB to display on QLabel
        masked_img = cv2.cvtColor(cv2.addWeighted(tail_masked_img, 1.0, head_masked_img, 1.0, 0.0), cv2.COLOR_BGR2RGB)
        plot_img = cv2.cvtColor(self.plot_img, cv2.COLOR_BGR2RGB)

        original_image, contour_image, thresholded_img, plot = self.convert_to_qimage(
            frame, result, masked_img, plot_img
        )

        preview_img = np.zeros(frame.shape, np.uint8)

        if self.param_type == 0:
            preview_img = cv2.cvtColor(head_masked_img, cv2.COLOR_BGR2RGB)
        else:
            preview_img = cv2.cvtColor(tail_masked_img, cv2.COLOR_BGR2RGB)

        preview_image = QImage(preview_img, preview_img.shape[1], preview_img.shape[0],
                               preview_img.strides[0], QImage.Format_RGB888)

        self.ui.orig_img.setPixmap(QPixmap.fromImage(original_image))
        self.ui.contour_img.setPixmap(QPixmap.fromImage(contour_image))
        self.ui.thresholded_img.setPixmap(QPixmap.fromImage(thresholded_img))
        self.ui.plot_img.setPixmap(QPixmap.fromImage(plot))
        self.preview_dialog_ui.preview_img.setPixmap(QPixmap.fromImage(preview_image))

    def get_robot(self, part, frame):
        if part == "tail":
            part_range = self.tail_hsv
        elif part == "head":
            part_range = self.head_hsv
        else:
            part_range = self.whole_body_hsv

        # Convert to HSV for easier to threshold
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)

        # Color range thresholding
        mask = cv2.inRange(hsv_img, tuple(part_range[0]), tuple(part_range[1]))
        mask = cv2.dilate(mask, None, iterations=2)

        # Thresholding image
        masked_img = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert to Grayscale (needed for binarizing)
        thresholded_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(thresholded_img,
                                          cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        # Select biggest contour; drop the rest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        return contours, masked_img

    def get_distance(self, result_img):
        interfere_point = [0, 0]
        xt, yt = self.current_tail
        xg, yg = self.clicked_pos
        xs, ys = self.start

        # Thank you so much, MATLAB :*
        interfere_point[0] = ((xt * xg ** 2 - 2 * xt * xg * xs - xg * yg * ys + yt * xg * yg + xg * ys ** 2 -
                               yt * xg * ys + xt * xs ** 2 + xs * yg ** 2 - xs * yg * ys - yt * xs * yg + yt * xs * ys) /
                              (xg ** 2 - 2 * xg * xs + xs ** 2 + yg ** 2 - 2 * yg * ys + ys ** 2))

        interfere_point[1] = ((xg ** 2 * ys - xg * xs * yg - xg * xs * ys + xt * xg * yg -
                               xt * xg * ys + xs ** 2 * yg - xt * xs * yg + xt * xs * ys + yt * yg ** 2 - 2 * yt * yg * ys + yt * ys ** 2) /
                              (xg ** 2 - 2 * xg * xs + xs ** 2 + yg ** 2 - 2 * yg * ys + ys ** 2))

        cv2.line(result_img, self.current_tail, tuple(interfere_point), (100, 95, 20), 3)

        interfere_point = self.camera_to_real(interfere_point)
        current_tail = self.camera_to_real(self.current_tail)

        distance_vector = np.array([interfere_point[0] - current_tail[0], interfere_point[1] - current_tail[1]],
                                   np.double)

        if interfere_point[1] < current_tail[1]:
            sign = 1
        elif interfere_point[1] > current_tail[1]:
            sign = -1
        elif interfere_point[1] == current_tail[1]:
            sign = 0

        return np.linalg.norm(distance_vector) * sign

    contour_count = 0

    def process_contour(self, contour, robot_part, img):
        cnt = contour[0]

        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if robot_part is "tail":
            self.current_tail = (x + np.int(np.floor(w / 2)), np.int(np.floor(y + h / 2)))
        elif robot_part is "head":
            self.current_head = (x + np.int(np.floor(w / 2)), np.int(np.floor(y + h / 2)))
        else:
            raise Exception("Invalid part name")

        robot_part = robot_part.capitalize()

        if w * h > 1000:
            cv2.putText(img, "{:s}: x: {:d} - y: {:d}".format(
                robot_part, x, y), (20, self.contour_count * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255))
            cv2.drawContours(img, [box], -1, (100, 255, 0), 2)  # Not needed; Just to display the difference
        else:
            cv2.putText(img, "{:s}: x: -- - y: --".format(robot_part), (20, self.contour_count * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255))

        if self.contour_count == 1:
            self.contour_count = 0
        else:
            self.contour_count += 1
        return x, y, w, h

    def draw_path_on_frame(self, frame):
        goalNode = self.final_point

        while goalNode.parent is not None:
            prevGoalNode = goalNode
            goalNode = goalNode.parent
            cv2.line(frame,
                     (int((prevGoalNode.x + 0.5) * self.grid_unit), int((prevGoalNode.y + 0.5) * self.grid_unit)),
                     (int((goalNode.x + 0.5) * self.grid_unit), int((goalNode.y + 0.5) * self.grid_unit)),
                     (200, 100, 120), 3)

    @staticmethod
    def convert_to_qimage(orig_img, contour_img, thresholded_img, plot_img):
        return (QImage(orig_img, orig_img.shape[1], orig_img.shape[0],
                       orig_img.strides[0], QImage.Format_RGB888)
                , QImage(contour_img, contour_img.shape[1], contour_img.shape[0],
                         contour_img.strides[0], QImage.Format_RGB888)
                , QImage(thresholded_img, thresholded_img.shape[1], thresholded_img.shape[0],
                         thresholded_img.strides[0], QImage.Format_RGB888)
                , QImage(plot_img, plot_img.shape[1], plot_img.shape[0],
                         plot_img.strides[0], QImage.Format_RGB888))

    def open_setting_dialog(self):
        self.settings_dialog.hide()
        self.settings_dialog.show()

    def open_plots_dialog(self):
        self.plots_dialog.hide()
        self.plots_dialog.show()

    def save_settings(self):
        # Save color settings
        self.settings.setValue("tail_range", self.tail_hsv)
        self.settings.setValue("head_range", self.head_hsv)

        # Save controller settings
        self.controller.k_P = self.dialog_ui.kP_P.value()
        self.controller.k_I = self.dialog_ui.kI_P.value()
        self.controller.k_D = self.dialog_ui.kD_P.value()
        # self.ui.kP_V.value(), self.ui.kI_V.value(), self.ui.kD_V.value()
        self.kP_P = self.dialog_ui.kP_P.value()
        self.kI_P = self.dialog_ui.kI_P.value()
        self.kD_P = self.dialog_ui.kD_P.value()
        self.kP_V = self.dialog_ui.kP_V.value()
        self.kI_V = self.dialog_ui.kI_V.value()
        self.kD_V = self.dialog_ui.kD_V.value()

        self.settings.setValue("Kp_P", "{:.2f}".format(self.dialog_ui.kP_P.value()))
        self.settings.setValue("Ki_P", "{:.2f}".format(self.dialog_ui.kI_P.value()))
        self.settings.setValue("Kd_P", "{:.2f}".format(self.dialog_ui.kD_P.value()))

        self.settings.setValue("Kp_V", "{:.2f}".format(self.dialog_ui.kP_V.value()))
        self.settings.setValue("Ki_V", "{:.2f}".format(self.dialog_ui.kI_V.value()))
        self.settings.setValue("Kd_V", "{:.2f}".format(self.dialog_ui.kD_V.value()))

    def serial_ports_scanned(self):
        self.ui.serial_port_list.clear()

        if len(self.serial_ports) > 0:
            self.ui.connect_btn.setDisabled(False)
            self.have_ports_available = True

            for port in self.serial_ports:
                self.ui.serial_port_list.addItem(port)

            self.ui.serial_port_list.setCurrentIndex(0)
        else:
            self.have_ports_available = False
            self.ui.serial_port_list.addItem("There are no ports available")

        self.ui.serial_port_list.setDisabled(False)

    def get_direction(self):
        _, frame = self.capture.read()
        # frame = cv2.imread("WIN_20170421_13_02_15_Pro.jpg")
        frame = cv2.resize(frame, self.img_size.toTuple())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        creator = MapCreator(
            self.img_size.width(), self.img_size.height(), self.grid_unit
        )

        generated_map, (x_size, y_size) = creator.create_map(
            frame, [[(89, 159, 154), (112, 255, 255)], [(165, 159, 154), (255, 255, 255)]])

        self.plot_img = np.full((self.img_size.height(), self.img_size.width(), 3), 255, np.uint8)

        # Draw grid
        for i in xrange(self.grid_unit, self.img_size.width(), self.grid_unit):
            cv2.line(self.plot_img, (i - 1, 0), (i - 1, self.img_size.height()), (0, 0, 0), 1)

        for i in xrange(0, self.img_size.height(), self.grid_unit):
            cv2.line(self.plot_img, (0, i - 1), (self.img_size.width(), i - 1), (0, 0, 0), 1)

        reshaped_nodes = generated_map.nodes.reshape((x_size * y_size))
        for i in xrange(0, y_size):
            for j in xrange(0, x_size):
                if not reshaped_nodes[i * x_size + j]:
                    cv2.rectangle(self.plot_img, (j * self.grid_unit - 1, i * self.grid_unit - 1),
                                  ((j + 1) * self.grid_unit - 1, (i + 1) * self.grid_unit - 1), (0, 0, 0),
                                  cv2.FILLED)

        start_pos, _ = self.img_to_grid(self.current_tail, x_size, y_size)
        generated_map.start = MapNode(*start_pos)

        if self.agent_size <= 0:
            # noinspection PyCallByClass
            QMessageBox.critical(self, self.tr("Error!"),
                                 self.tr("You must select region of robot"),
                                 QMessageBox.Cancel, QMessageBox.Cancel)

        # Get robot coordinate
        robot_pos, robot_size = self.img_to_grid(self.start, x_size, y_size, self.agent_size, self.agent_size)
        cv2.rectangle(self.plot_img, (robot_pos[0] * self.grid_unit, robot_pos[1] * self.grid_unit),
                      ((robot_pos[0] + robot_size[1]) * self.grid_unit - 2,
                       (robot_pos[1] + robot_size[1]) * self.grid_unit - 2), (0, 255, 0),
                      cv2.FILLED)

        agent_size_on_grid = np.max(robot_size)

        # Specify goal
        if self.goal is not None:
            goal_pos, _ = self.img_to_grid(self.goal, x_size, y_size)
            generated_map.goal = goal_pos

        path_finder = AStar(generated_map, agent_size_on_grid, 1)

        drawer = Drawer(self.img_size.width(), self.img_size.height(),
                        self.grid_unit, self.plot_img)

        c, o, s, g = path_finder.run()

        if c is not None and o is not None and s is not None and g is not None:
            self.final_point = g
            self.final_plan = []
            self.final_plan.append(((g.x + 0.5) * self.grid_unit, (g.y + 0.5) * self.grid_unit))
            drawer.draw(c, o, s, g)

            g = g.parent

            while g is not None:
                self.final_plan.append(((g.x + 0.5) * self.grid_unit, (g.y + 0.5) * self.grid_unit))
                g = g.parent

        grid = cv2.cvtColor(self.plot_img, cv2.COLOR_BGR2RGB)
        grid_qimg = QImage(grid, grid.shape[1], grid.shape[0], grid.strides[0], QImage.Format_RGB888)
        self.ui.plot_img.setPixmap(QPixmap.fromImage(grid_qimg))

    def img_to_grid(self, img_pos, x_size, y_size, w=0, h=0):
        x_percent = np.float(img_pos[0]) / self.img_size.width()
        y_percent = np.float(img_pos[1]) / self.img_size.height()
        x_pos = np.int(np.floor(x_percent * x_size))
        y_pos = np.int(np.floor(y_percent * y_size))
        x_count = np.int(np.ceil(w / self.grid_unit))
        y_count = np.int(np.ceil(h / self.grid_unit))

        return (x_pos, y_pos), (x_count, y_count)

    def calculate_current_angle(self):
        oriented_dir = (self.current_head[0] - self.current_tail[0], self.current_head[1] - self.current_tail[1])
        goal_dir = (1, 0)
        dot = oriented_dir[0] * goal_dir[0] + oriented_dir[1] * goal_dir[1]
        det = oriented_dir[0] * goal_dir[1] - oriented_dir[1] * goal_dir[0]
        return np.arctan2(det, dot)

    def settings_feedback(self, data):
        self.ui.bluetooth_received_list.addItem(data)
        self.ui.bluetooth_received_list.scrollToBottom()

    def img_clicked(self, pos):
        self.goal = tuple(map(lambda v: np.int(v), pos.toTuple()))
        print self.goal

    def selected_region(self, region):
        self.region = region
        self.agent_size = np.maximum(region.width(), region.height())
        self.start = (region.x(), region.y())

    @staticmethod
    def camera_to_real(pos):
        pos += (1,)

        pos = np.mat(pos)
        pos = np.transpose(pos)

        A = np.dot(np.linalg.inv(Config.INTRINSIC_MAT), Config.TRANSLATION_MAT.item((2, 0)))
        B = np.dot(A, pos) - Config.TRANSLATION_MAT
        real_pos = np.dot(np.linalg.inv(Config.ROTATION_MAT), B)
        real_pos = real_pos.reshape((1, 3))[0, :2].tolist()[0]
        return map(lambda i: np.int(i), real_pos)

    def update_lower(self, valueType, value, slider=True):
        subject = [[]]

        if self.param_type == 1:
            subject = self.tail_hsv
        elif self.param_type == 0:
            subject = self.head_hsv
        elif self.param_type == 2:
            pass

        if valueType == "h":
            subject[0][0] = value
        elif valueType == "s":
            subject[0][1] = value
        elif valueType == "v":
            subject[0][2] = value

        if slider:
            self.populate_color_spinbox()
        else:
            self.populate_color_slider()

    def update_upper(self, valueType, value, slider=True):
        subject = [[]]

        if self.param_type == 1:
            subject = self.tail_hsv
        elif self.param_type == 0:
            subject = self.head_hsv
        elif self.param_type == 2:
            pass

        if valueType == "h":
            subject[1][0] = value
        elif valueType == "s":
            subject[1][1] = value
        elif valueType == "v":
            subject[1][2] = value

        if slider:
            self.populate_color_spinbox()
        else:
            self.populate_color_slider()

    def rescan_serial_port(self):
        self.ui.rescan_btn.setDisabled(True)
        self.ui.connect_btn.setDisabled(True)

        self.ui.serial_port_list.setDisabled(True)
        self.ui.serial_port_list.clear()

        self.serial_ports = []
        self.ui.serial_port_list.addItem("Scanning for available ports...")

        self.serial_scanner.ports = self.serial_ports
        self.serial_scanner.start()

    def connect_bluetooth_serial(self):
        self.ui.connect_btn.setDisabled(True)

        if not self.current_serial_connection:
            self.current_serial_connection = serial.Serial(self.current_serial_port, Config.SERIAL[1], timeout=0)

            self.ui.connect_btn.setText("Disconnect")
            self.ui.track_btn.setDisabled(False)
        else:
            self.current_serial_connection.close()
            self.current_serial_connection = None

            self.ui.connect_btn.setText("Connect")
            self.ui.track_btn.setDisabled(True)

        self.ui.connect_btn.setDisabled(False)

    def connect_debug_serial(self):
        self.ui.connect_2_btn.setDisabled(True)

        if not self.current_debug_serial_connection:
            self.current_debug_serial_connection = serial.Serial(self.current_debug_serial_port, Config.SERIAL[1],
                                                                 timeout=0)

            self.debug_serial_connection_listener.set_connection(self.current_debug_serial_connection)
            self.debug_serial_connection_listener.stopped = False
            self.debug_serial_connection_listener.start()

            self.ui.connect_2_btn.setText("Disconnect")
        else:
            self.debug_serial_connection_listener.stopped = True
            self.debug_serial_connection_listener.set_connection(None)
            self.debug_serial_connection_listener.quit()

            self.current_debug_serial_connection.close()
            self.current_debug_serial_connection = None

            self.ui.connect_2_btn.setText("Connect")

        self.ui.connect_2_btn.setDisabled(False)

    def navigate(self, direction):
        if direction == 'left':
            self.current_serial_connection.write('cl')
        elif direction == 'right':
            self.current_serial_connection.write('cr')
        elif direction == 'forward':
            self.current_serial_connection.write('cf')
        elif direction == 'backward':
            self.current_serial_connection.write('cb')
        else:
            self.current_serial_connection.write('cs')

    def go_to_goal(self, goal):
        oriented_dir = (self.current_head[0] - self.current_tail[0], self.current_head[1] - self.current_tail[1])
        goal_dir = (-self.current_tail[0] + goal[0], -self.current_tail[1] + goal[1])
        dot = oriented_dir[0] * goal_dir[0] + oriented_dir[1] * goal_dir[1]
        det = oriented_dir[0] * goal_dir[1] - oriented_dir[1] * goal_dir[0]
        return np.arctan2(det, dot)

    def set_tracking_state(self):
        if not self.current_serial_connection:
            QMessageBox.critical(self, self.tr("Error!"), self.tr("Please connect to a serial port before tracking."),
                                 QMessageBox.Cancel, QMessageBox.Cancel)
            return

        self.tracking_state = not self.tracking_state

        if self.tracking_state:
            # self.serial_connection_listener.tracking = True
            # self.current_serial_connection.write('ct')
            # self.serial_connection_listener.start()
            self.ui_plot_pos.clear()
            self.ui_plot_vel.clear()
            self.ui_plot_ang_vel.clear()
            self.final_plan.pop()
            self.current_point = self.final_plan.pop()
            self.tracking_timer.start(300)

            self.ui.track_btn.setText("Stop tracking")
            self.ui.connect_btn.setDisabled(True)
        else:
            # self.serial_connection_listener.tracking = False
            # self.serial_connection_listener.quit()
            self.current_serial_connection.write('cs')

            self.x = []
            self.y = []
            self.time = []

            self.tracking_timer.stop()
            self.controller.reset()
            self.ui.connect_btn.setDisabled(False)
            self.ui.track_btn.setText("Start tracking")
            self.ui.status_line.setText("Start tracking to show coordination...")

    def tracking_robot(self):
        real_tail = self.camera_to_real(self.current_tail)
        real_goal = self.camera_to_real(self.current_point)
        gate = np.linalg.norm((-real_tail[0] + real_goal[0], -real_tail[1] + real_goal[1])) > 20

        if not gate:
            if len(self.final_plan) > 0:
                self.current_point = self.final_plan.pop()
                self.controller.reset()
            else:
                self.set_tracking_state()

            return

        value = self.go_to_goal(self.current_point)
        self.angle.append(value)
        signal = self.controller.pid_control(value)

        if abs(value * 180 / np.pi) < 15:
            base_pwm = 100

            if value > 0:
                l_w = base_pwm + signal
                r_w = base_pwm - signal
            elif value < 0:
                l_w = base_pwm - signal
                r_w = base_pwm + signal
            else:
                l_w = signal
                r_w = signal
            l_w, r_w = int(l_w), int(r_w)

            self.current_serial_connection.write("{:d}:{:d}e".format(l_w, r_w))
        else:
            base_pwm = 60

            l_w = base_pwm + abs(signal)
            r_w = base_pwm + abs(signal)
            l_w, r_w = int(l_w), int(r_w)

            if value > 0:
                self.current_serial_connection.write("{:d}:{:d}e".format(l_w, 0 - r_w))
            else:
                self.current_serial_connection.write("{:d}:{:d}e".format(0 - l_w, r_w))

        # l_w, r_w = self.controller.unicycle_2_differential(signal)

        if len(self.time) == 0:
            self.time.append(0)
        else:
            self.time.append(self.time[-1] + 1)

        self.do_update_graph()

        print l_w, r_w

    def clear_plot(self):
        self.plot_img = np.full((self.img_size.height(), self.img_size.width(), 3), 255, np.uint8)

    def choose_param_type(self, value):
        self.param_type = value

        self.populate_color_slider()

    def choose_serial_port(self, index):
        if self.have_ports_available:
            self.current_serial_port = self.serial_ports[index]

    def choose_debug_serial_port(self, index):
        if self.have_ports_available:
            self.current_debug_serial_port = self.serial_ports[index]

    def closeEvent(self, *args, **kwargs):
        if self.current_serial_connection and self.current_serial_connection.isOpen():
            self.current_serial_connection.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mySW = ControlMainWindow()
    mySW.setWindowTitle("Mobile Robot Tracking - by Trung")
    # app.setStyleSheet(qdarkstyle.load_stylesheet())
    mySW.show()
    sys.exit(app.exec_())
