from MapNode import *
import operator
import numpy as np


# noinspection PyAttributeOutsideInit
class Map(object):
    start = property(operator.attrgetter("_Map__start"))
    goal = property(operator.attrgetter("_Map__goal"))

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.length = self.width
        self.__start = MapNode(0, 0)
        self.__goal = MapNode(width - 1, height - 1)
        self.configuration = None
        self.nodes = np.full((height, width), True, np.bool)
        self.clear()

    @start.setter
    def start(self, start_pos):
        if isinstance(start_pos, MapNode):
            self.__start = start_pos
        elif type(start_pos) is tuple:
            self.__start = MapNode(*start_pos)

    @goal.setter
    def goal(self, goal_pos):
        if isinstance(goal_pos, MapNode):
            self.__goal = goal_pos
        elif type(goal_pos) is tuple:
            self.__goal = MapNode(*goal_pos)

    def clear(self):
        self.nodes[:, :] = True

        self.setGoal(self.width - 1, self.height - 1)

    def setGoal(self, x, y):
        if self.isOnMap(x, y):
            self.goal = MapNode(x, y)

    def generate(self, width, height, obstacleDensity, obstacleSize):
        if obstacleSize <= 0 or obstacleDensity > 90:
            return

        if not obstacleDensity is None:
            self.configuration = dict(obstacle_density=obstacleDensity, obstacle_size=obstacleSize)

        self.width = width
        self.height = height
        self.clear()

        nodesInMap = self.width * self.height
        desiredObstacleCount = np.floor(nodesInMap * self.configuration['obstacle_density'] / 100)
        obstacles = 0

        while obstacles < desiredObstacleCount:
            x = np.int(np.floor(np.random.rand() * self.width))
            y = np.int(np.floor(np.random.rand() * self.height))
            obstacles += self.placeObstacles(x, y, self.configuration['obstacle_size'])

    def isOnMap(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def placeObstacles(self, x, y, size):
        obstacleCount = 0
        lower = np.int(np.floor((size - 1) / 2))
        upper = np.int(np.ceil((size - 1) / 2))

        for loopX in xrange(x - lower, x + upper + 1):
            for loopY in xrange(y - lower, y + upper):
                if not self.isOnMap(loopX, loopY) or not self.nodes[loopX][loopY]:
                    continue

                if ((loopX == self.start.x and loopY == self.start.y) or
                        (loopX == self.goal.x and loopY == self.goal.y)):
                    continue

                self.nodes[loopX][loopY] = False
                obstacleCount += 1

        return obstacleCount
