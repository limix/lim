class Block(object):

    def __init__(self, title, length=79):
        self._title = title
        l = length - len(title) - 2
        l0 = l // 2
        l1 = l - l0
        self._left = '-' * (l0 - 1)
        self._right = '-' * (l1 - 1)

    def __enter__(self):
        middle = ' %s ' % self._title
        print(unichr(0x250c).encode('utf-8') + self._left +
              middle + self._right + unichr(0x2510).encode('utf-8'))
        return self.printer

    def __exit__(self, *_):
        middle = '-' * (len(self._title) + 2)
        print(unichr(0x2514).encode('utf-8') + self._left +
              middle + self._right + unichr(0x2518).encode('utf-8'))

    def printer(self, msg):
        msg = str(msg)
        lines = msg.split('\n')
        lines = ['|' + l + ' ' * (77 - len(l.decode("utf-8"))) + '|' for l in lines]
        print('\n'.join(lines))
