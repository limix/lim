from .vector import Vector

class Column(Vector):
    def __init__(self, name, labels, values):
        super(Column, self).__init__(labels, values)
        if name is None:
            raise ValueError("A column must have a name.")
        self.name = name

    def __repr__(self):
        return "Column(" + bytes(self._data) + ")"

    def __str__(self):
        return "Column(" + bytes(self._data) + ")"
