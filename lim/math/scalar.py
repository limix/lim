def isint(value):
    try:
        return float(value) == int(value)
    except ValueError:
        return False
