from scipy.stats import norm

def bern2lat_correction(h2, prevalence, ascertainment):
    t = norm.ppf(1-prevalence)
    z = norm.pdf(t)
    k = prevalence * (1 - prevalence)
    p = ascertainment * (1 - ascertainment)
    return h2 * k**2 / (z**2 * p)
