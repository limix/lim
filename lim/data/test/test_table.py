from ..table import Table
from ..column import Column

def test_adding_columns():
    t = Table()

    labels =['sample01', 'sample02', 'sample03']
    values = [34.3, 2.3, 103.4, -030.]
    c = Column('sample_id', labels, values)

    t.add(c)
