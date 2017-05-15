class MapNode:
    def __init__(self, x, y, parent=None, cost=None):
        self.x = x
        self.y = y
        self.g = 0
        self.f = 0
        self.parent = parent
        if parent is not None:
            self.g = parent.g + cost

    def equals(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return "{:d},{:d}".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
