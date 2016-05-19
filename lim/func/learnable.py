from .variables import Variables

class Learnable(object):
    def __init__(self, variables):
        assert isinstance(variables, Variables)
        self._vars = variables

    def gradient(self, *args, **kwargs):
        names = sorted(self._vars.names())
        grad = []
        for name in names:
            g = getattr(self, 'derivative_' + name)(*args, **kwargs)
            grad.append(g)
        return grad

    def fix(self, var_name):
        self._vars[var_name].fix()

    def unfix(self, var_name):
        self._vars[var_name].unfix()

    def isfixed(self, var_name):
        return self._vars[var_name].isfixed()

    def variables(self):
        return self._vars
