from .vector import Vector

class Column(Vector):
    def __init__(self, name, labels, values):
        super(Column, self).__init__(labels, values)
        self.name = name
