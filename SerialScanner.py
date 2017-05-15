from PySide.QtCore import *
import sys
import serial
import glob


class SerialScanner(QThread):
    def __init__(self, serial_ports, parent=None):
        QThread.__init__(self, parent)
        self.ports = serial_ports

    def run(self):
        try:
            if sys.platform.startswith('win'):
                ports = ['COM%s' % (i + 1) for i in range(256)]
            elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
                # this excludes your current terminal "/dev/tty"
                ports = glob.glob('/dev/tty[A-Za-z]*')
            elif sys.platform.startswith('darwin'):
                ports = glob.glob('/dev/tty.*')
            else:
                raise EnvironmentError('Unsupported platform')

            result = []
            for port in ports:
                try:
                    s = serial.Serial(port)
                    s.close()
                    result.append(port)
                except (OSError, serial.SerialException):
                    pass

            for port in result:
                self.ports.append(port)

        except IOError as e:
            return
