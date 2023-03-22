"""
This module takes linear and non-linear P(k) and extrapolates
them linearly in log-space out to a specified high k_max

"""
from builtins import range
from cosmosis.datablock import option_section, names
import numpy as np
from numpy import log, exp


def linear_extend(x, y, xmin, xmax, nmin, nmax, nfit):
    if xmin < x.min():
        xf = x[:nfit]
        yf = y[:nfit]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(xmin, x.min(), nmin, endpoint=False)
        ynew = np.polyval(p, xnew)
        x = np.concatenate((xnew, x))
        y = np.concatenate((ynew, y))
    if xmax > x.max():
        xf = x[-nfit:]
        yf = y[-nfit:]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(x.max(), xmax, nmax, endpoint=True)
        # skip the first point as it is just the xmax
        xnew = xnew[1:]
        ynew = np.polyval(p, xnew)
        x = np.concatenate((x, xnew))
        y = np.concatenate((y, ynew))
    return x, y
    
def extrapolate_section(block, section, kmin, kmax, nmin, nmax, npoint, key):
    # load current values
    k = block[section, "k_h"]
    z = block[section, "z"]
    nk = len(k)
    nz = len(z)
    # load other current values
    k, z, P = block.get_grid(section, "k_h", "z", key)

    # SJ: for weyl x matter, reverse the sign and extrapolate, and put back the sign
    # extrapolate
    P_out = []
    for i in range(nz):
        Pi = P[:, i]
        if section in ['weyl_curvature_matter_power_lin']:
            Pi = np.abs(Pi)
        logk, logp = linear_extend(log(k), log(Pi), log(
            kmin), log(kmax), nmin, nmax, npoint)
        if section in ['weyl_curvature_matter_power_lin']:
            P_out.append(-exp(logp))
        else:
            P_out.append(exp(logp))

    k = exp(logk)
    P_out = np.dstack(P_out).squeeze()
    block.replace_grid(section, "z", z, "k_h", k, key, P_out.T)


def setup(options):
    kmax = options.get_double(option_section, "kmax")
    kmin = options.get_double(option_section, "kmin", default=1e10)
    nmin = options.get_int(option_section, "nmin", default=50)
    npoint = options.get_int(option_section, "npoint", default=3)
    nmax = options.get_int(option_section, "nmax", default=200)
    extrapk = options.get_string(option_section, "power_spectra_names", default='')
    extrapk = extrapk.split(' ')
    return {"kmax": kmax, "kmin": kmin, "nmin": nmin, "nmax": nmax, "npoint": npoint, "extrapk": extrapk}


def execute(block, config):
    kmin = config['kmin']
    kmax = config['kmax']
    nmin = config['nmin']
    nmax = config['nmax']
    npoint = config['npoint']
    extrapk = config['extrapk']
    if '' in extrapk:
        extrapk.remove('')

    # extrapolate non-linear power
    for section in [names.matter_power_nl, names.matter_power_lin]+extrapk:
        if '.' in section:
            section, key = section.split('.')
        else:
            key = 'p_k'
        if block.has_section(section):
            extrapolate_section(block, section, kmin, kmax, nmin, nmax, npoint, key)
    return 0
