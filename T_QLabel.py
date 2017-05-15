from PySide.QtCore import *
from PySide.QtGui import *


class T_QLabel(QLabel):
    pictureClicked = Signal(object)  # can be other types (list, dict, object...)
    selected = Signal(object)

    def __init__(self, parent=None):
        super(T_QLabel, self).__init__(parent)
        self.rubberBand = None

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MiddleButton:
            self.pictureClicked.emit(event.pos())
            return
        elif event.buttons() == Qt.RightButton:
            if self.rubberBand is not None:
                self.rubberBand = None

            return

        self.origin = event.pos()

        if self.rubberBand is None:
            self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)

        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()

    def mouseMoveEvent(self, event):
        self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self.rubberBand is not None:
            self.rubberBand.hide()
            self.selected.emit(self.rubberBand.geometry())
        else:
            self.selected.emit(None)
