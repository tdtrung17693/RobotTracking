from PySide.QtCore import *
from time import sleep
import datetime
import serial
import Config


class SerialListener(QThread):
    received = Signal(str)
    feedback = Signal(str)
    arrived = Signal(str)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.stopped = False
        self.tracking = False
        self.connection = None

    def run(self):
        data_chain = ""
        while self.tracking:
            data = self.connection.read()

            if len(data) > 0 and data == 's':
                self.received.emit("Received")
            sleep(1)

    def set_connection(self, connection):
        self.connection = connection