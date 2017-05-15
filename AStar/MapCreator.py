import operator
from Map import *
from Algorithm import *
from Drawer import *


# noinspection PyAttributeOutsideInit
class MapCreator:
    def __init__(self, width, height, grid_unit):
        self.width = width
        self.height = height
        self.grid_unit = grid_unit

    width = property(operator.attrgetter('__width'))
    height = property(operator.attrgetter('__height'))
    grid_unit = property(operator.attrgetter('__grid_unit'))

    @width.setter
    def width(self, w):
        if not w or w < 0:
            raise Exception("Invalid value of width")

        self.__width = w

    @height.setter
    def height(self, h):
        if not h or h < 0:
            raise Exception("Invalid value of height")

        self.__height = h

    @grid_unit.setter
    def grid_unit(self, u):
        if not u or u < 0:
            raise Exception("Invalid value of width")

        if not (self.width % u == 0 and self.height % u == 0):
            raise Exception("Grid unit must be divisibll by width and height")

        self.__grid_unit = u

    def create_map(self, img=None, obstacle_ranges=None):
        if img is None:
            raise Exception("Invalid image")

        if obstacle_ranges is None:
            raise Exception("Range must be specified")

        x_size = np.int(np.round(self.width / self.grid_unit))
        y_size = np.int(np.round(self.height / self.grid_unit))
        generated_map = Map(x_size, y_size)

        imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        imgHSV = cv2.GaussianBlur(imgHSV, (5, 5), 0)

        thresholded_all = np.zeros(img.shape[:2], np.uint8)

        for obstacle_range in obstacle_ranges:
            thresholded = cv2.inRange(imgHSV, obstacle_range[0], obstacle_range[1])
            thresholded = cv2.dilate(thresholded, None, iterations=2)
            # Dilate to make small obstacle bigger, so detectable
            thresholded = cv2.dilate(thresholded, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
            thresholded_all = cv2.bitwise_or(thresholded_all, thresholded)

        _, contours, _ = cv2.findContours(thresholded_all, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                percent_x = np.float(x) / self.width
                percent_y = np.float(y) / self.height

                x_pos = np.int(np.round(percent_x * x_size))
                y_pos = np.int(np.round(percent_y * y_size))
                x_count = np.int(np.ceil(w / self.grid_unit))
                y_count = np.int(np.ceil(h / self.grid_unit))

                generated_map.nodes[y_pos:y_pos + y_count + 1, x_pos:x_pos + x_count + 1] = False

        return generated_map, (x_size, y_size)
