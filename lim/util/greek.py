from __future__ import unicode_literals


def get_greek(name):
    d = dict(alpha=0x3b1,
             beta=0x3b2,
             gamma=0x3b3,
             delta=0x3b4,
             epsilon=0x3b5,
             zeta=0x3b6,
             eta=0x3b7,
             theta=0x3b8,
             iota=0x3b9,
             kappa=0x3ba,
             mu=0x3bc,
             nu=0x3bd,
             xi=0x3be,
             omicron=0x3bf,
             pi=0x3c0,
             rho=0x3c1,
             sigmal=0x3c2,
             sigma=0x3c3,
             tau=0x3c4,
             upsilon=0x3c5,
             phi=0x3c6,
             chi=0x3c7,
             psi=0x3c8,
             omega=0x3c9,
             Sigma=0x2211)
    d['lambda'] = 0x3bb
    return unichr(d[name])
