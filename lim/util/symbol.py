from unicodedata import lookup as _lup

alpha = _lup("GREEK SMALL LETTER ALPHA")
beta = _lup("GREEK SMALL LETTER BETA")
gamma = _lup("GREEK SMALL LETTER GAMMA")
delta = _lup("GREEK SMALL LETTER DELTA")
epsilon = _lup("GREEK SMALL LETTER EPSILON")
zeta = _lup("GREEK SMALL LETTER ZETA")
eta = _lup("GREEK SMALL LETTER ETA")
theta = _lup("GREEK SMALL LETTER THETA")
iota = _lup("GREEK SMALL LETTER IOTA")
kappa = _lup("GREEK SMALL LETTER KAPPA")
mu = _lup("GREEK SMALL LETTER MU")
nu = _lup("GREEK SMALL LETTER NU")
xi = _lup("GREEK SMALL LETTER XI")
omicron = _lup("GREEK SMALL LETTER OMICRON")
pi = _lup("GREEK SMALL LETTER PI")
rho = _lup("GREEK SMALL LETTER RHO")
fsigma = _lup("GREEK SMALL LETTER FINAL SIGMA")
sigma = _lup("GREEK SMALL LETTER SIGMA")
tau = _lup("GREEK SMALL LETTER TAU")
upsilon = _lup("GREEK SMALL LETTER UPSILON")
phi = _lup("GREEK SMALL LETTER PHI")
chi = _lup("GREEK SMALL LETTER CHI")
psi = _lup("GREEK SMALL LETTER PSI")
omega = _lup("GREEK SMALL LETTER OMEGA")
nsum = _lup("N-ARY SUMMATION")
bone = _lup("mathematical bold digit one")

if __name__ == '__main__':
    for v in sorted(dir()):
        if v[0] != '_':
            print("%s: %s" % (v, eval(v)))
