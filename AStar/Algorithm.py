from MapNode import *
import numpy as np


# noinspection PyMethodMayBeStatic
class AStar:
    COST_STRAIGHT = 1
    COST_DIAGONAL = 1.414

    def __init__(self, generated_map, agent_size, safe_clearance=1):
        self.map = generated_map
        self.agent_size = agent_size
        self.half_agent_size = np.int(np.floor(agent_size / 2)) if agent_size > 1 else 1
        self.safe_clearance = safe_clearance

    def getNeighbourNodes(self, node):
        neighbours = []

        for x in xrange(node.x - 1, node.x + 2):
            for y in xrange(node.y - 1, node.y + 2):
                if x == node.x and y == node.y:
                    continue

                if not self.map.isOnMap(x, y) or not self.map.nodes[y][x]:
                    continue
                print x, y
                if self.calculate_clearance(
                        MapNode(x - self.half_agent_size,
                                y - self.half_agent_size)) < self.agent_size + self.safe_clearance:
                    continue

                cost = AStar.COST_STRAIGHT if (x == node.x or y == node.y) else AStar.COST_DIAGONAL
                neighbours.append(MapNode(x, y, node, cost))

        return neighbours

    # Using Manhattan Distance
    def heuristic(self, node, goal):
        return np.abs(node.x - goal.x) + np.abs(node.y - goal.y)

    def run(self):
        closedList = []
        openList = []
        start = self.map.start
        goal = self.map.goal

        start.f = start.g + self.heuristic(start, goal)
        openList.append(start)

        while len(openList) > 0:
            lowestF = 0
            openList = sorted(openList, key=lambda node: node.f)
            # for i in xrange(1, len(openList)):
            #     if openList[i].f < openList[lowestF].f:
            #         lowestF = i

            current = openList[0]

            if self.at_goal(current):
                return closedList, openList, start, current

            # Remove node
            openList.pop(lowestF)
            closedList.append(current)

            # self.drawer.drawVisited(current.x, current.y)

            neighbours = self.getNeighbourNodes(current)

            self.addNodesToOpenList(neighbours, current, goal, openList, closedList)

        return None, None, None, None

    def at_goal(self, node):
        goal = self.map.goal
        return (False not in self.map.nodes[node.y: node.y + self.safe_clearance,
                             node.x - self.safe_clearance: node.x + self.safe_clearance] and
                node.y - self.safe_clearance <= goal.y <= node.y + self.safe_clearance and
                node.x - self.safe_clearance <= goal.x <= node.x + self.safe_clearance)

    def calculate_clearance(self, node):
        c = 1
        clearance = c
        solid = False

        while not solid:
            if (False in self.map.nodes[node.y:node.y + c + 1, node.x:node.x + c + 1] or (
                            node.y + c >= self.map.nodes.shape[0]) or
                    (node.x + c >= self.map.nodes.shape[1])):
                clearance = c
                solid = True

            c += 1
        print c
        return clearance

    def addNodesToOpenList(self, nodes, current, goal, openList, closedList):
        for i in xrange(0, len(nodes)):
            if nodes[i] not in closedList:
                if nodes[i] not in openList:
                    nodes[i].f = nodes[i].g + self.heuristic(nodes[i], goal)
                    openList.append(nodes[i])
                else:
                    idx = openList.index(nodes[i])
                    if nodes[i].g < openList[idx].g:
                        nodes[i].f = nodes[i].g + self.heuristic(nodes[i], goal)
                        nodes[i].parent = current
                        openList[idx] = nodes[i]
