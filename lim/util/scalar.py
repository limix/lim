from numpy import asscalar

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def asfloat(value):
    if isfloat(value):
        return value
    return asscalar(value, float)

def isint(value):
    try:
        return float(value) == int(value)
    except ValueError:
        return False
    except TypeError:
        return False

def isnumber(value):
    return isfloat(value) or isint(value)
