import numpy as np
import cv2

class Drawer:
    def __init__(self, width, height, scale, canvas=None):
        self.BACKGROUND_COLOUR = (245, 245, 245)
        self.OBSTACLE_COLOUR = (33, 33, 33)
        self.PATH_COLOUR = (222, 10, 20)
        self.CLOSED_LIST_COLOUR = (1, 87, 155)
        self.OPEN_LIST_COLOUR = (41, 182, 246)
        self.scale = scale
        self.canvas = np.zeros((height, width, 3), np.uint8) if canvas is None else canvas


    def drawObstacles(self, map):
        for y in xrange(0, map.height):
            for x in xrange(0, map.width):
                if not map.nodes[x][y]:
                    self.drawObstacle(x, y)

    def drawObstacle(self, x, y):
        self.drawNode(x, y, self.OBSTACLE_COLOUR)

    def clearObstacle(self, x, y):
        self.drawNode(x, y, self.BACKGROUND_COLOUR)

    def drawNode(self, x, y, color):
        cv2.rectangle(self.canvas, (x * self.scale, y * self.scale), ((x + 1) * self.scale, (y + 1) * self.scale),
                      color, cv2.FILLED)

    def drawVisited(self, x, y):
        self.drawNode(x, y, self.CLOSED_LIST_COLOUR)

    def showCanvas(self, timeout):
        cv2.imshow("MAP", self.canvas)
        cv2.waitKey(timeout)

    def drawOpenListNode(self, x, y):
        self.drawNode(x, y, self.OPEN_LIST_COLOUR)

    def drawStartGoal(self, x, y):
        self.drawNode(x, y, self.PATH_COLOUR)

    def clearCanvas(self):
        self.canvas = np.zeros((480, 640, 3), np.uint8)
        self.canvas[:] = (245, 245, 245)

    def draw(self, closedList, openList, startNode, goalNode):
        self.drawStartGoal(goalNode.x, goalNode.y)
        self.drawStartGoal(startNode.x, startNode.y)

        while goalNode.parent is not None:
            prevGoalNode = goalNode
            goalNode = goalNode.parent
            cv2.line(self.canvas,
                     (int((prevGoalNode.x+0.5) * self.scale), int((prevGoalNode.y+0.5) * self.scale)),
                     (int((goalNode.x+0.5) * self.scale), int((goalNode.y+0.5) * self.scale)),
                     self.PATH_COLOUR, 3)

    def drawPath(self, goalNode):
        while goalNode.parent is not None:
            goalNode = goalNode.parent
            cv2.line(self.canvas,
                     ((goalNode.x + 0.5) * self.scale, (goalNode.y + 0.5) * self.scale),
                     ((goalNode.x + 0.5) * self.scale, (goalNode.y + 0.5) * self.scale),
                     self.PATH_COLOUR, 2)

    def getCanvasWidth(self):
        return self.canvas.shape[1]

    def getCanvasHeight(self):
        return self.canvas.shape[0]