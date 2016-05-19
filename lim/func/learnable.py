from .variables import Variables
from .variables import merge_variables

class Learnable(object):
    def __init__(self, variables):
        assert isinstance(variables, Variables)
        self.__vars = variables

    def gradient(self, *args, **kwargs):
        names = sorted(self.__vars.names())
        grad = []
        for name in names:
            g = getattr(self, 'derivative_' + name)(*args, **kwargs)
            grad.append(g)
        return grad

    def fix(self, var_name):
        self.__vars[var_name].fix()

    def unfix(self, var_name):
        self.__vars[var_name].unfix()

    def isfixed(self, var_name):
        return self.__vars[var_name].isfixed()

    def variables(self):
        return self.__vars

class LearnableReduce(object):
    def __init__(self, prefix, learnables):
        vars_list = [l.variables() for l in learnables]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['%s[%d]' % (prefix, i)] = vs
        self.__vars = merge_variables(vd)
        self.__learnables = learnables

    def gradient(self, *args, **kwargs):
        grad = []
        for l in self.__learnables:
            names = l.variables().names()
            for name in names:
                g = getattr(l, 'derivative_' + name)(*args, **kwargs)
                grad.append(g)
        return grad

    def fix(self, var_name):
        raise NotImplementedError

    def unfix(self, var_name):
        raise NotImplementedError

    def isfixed(self, var_name):
        raise NotImplementedError

    def variables(self):
        return self.__vars
