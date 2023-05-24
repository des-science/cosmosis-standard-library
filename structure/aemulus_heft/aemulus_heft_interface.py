import sys
import os
from cosmosis.datablock import names, option_section
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.interpolate import interp1d
import numpy as np

dirname = os.path.split(__file__)[0]

# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'



def setup(options):
    input_section = options.get_string(option_section, "input_section", default=names.matter_power_lin)
    output_section = options.get_string(option_section, "output_section", default=names.matter_power_nl)

    # Set up the path to let us import the emulator
    pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
#    install_dir = dirname + f"/ee_install/lib/python{pyversion}/site-packages/"
#    with open(f"{install_dir}/easy-install.pth") as f:
#        pth = f.read().strip()
#        install_dir = install_dir + pth
#    sys.path.insert(0, install_dir)

    # check everything imports
    from aemulus_heft.heft_emu import HEFTEmulator

    emulator = HEFTEmulator()
    #emulator = None
    # euclidemu2.PyEuclidEmulator()

    return [input_section, output_section, emulator]

def _cleft_pk(k, p_lin):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            Note that we require p_lin at the redshift that you wish
            to make predictions for, because in cosmologies with neutrinos
            a constant linear growth rescaling no longer works.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''

    
    cleftobj = RKECLEFT(k, p_lin)
    cleftobj.make_ptable(D=1, kmin=k[0], kmax=k[-1], nk=1000)
    cleftpk = cleftobj.pktable.T

    cleftpk[2, :] /= 2 #(1 d)
    cleftpk[6, :] /= 0.25 # (d2 d2)
    cleftpk[7, :] /= 2 #(1 s)
    cleftpk[8, :] /= 2 #(d s)

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

    return cleftspline, cleftobj

def lpt_spectra(k, pk_m_lin, pk_cb_lin):

    pk_cb_m_lin = np.sqrt(pk_m_lin * pk_cb_lin)
    cleft_m_spline, _ = _cleft_pk(k, pk_m_lin)
    cleft_cb_spline, _ = _cleft_pk(k, pk_cb_lin)
    cleft_cb_m_spline, _ = _cleft_pk(k, pk_cb_m_lin)
    
    pk_m_cleft = cleft_m_spline(k)[1:]
    pk_cb_cleft = cleft_cb_spline(k)[1:]
    pk_cb_m_cleft = cleft_cb_m_spline(k)[1:]
    s_m_map = {1:0, 3: 1, 6: 3, 10: 6}
    s_cb_map = {2: 0, 4: 1, 5: 2, 7: 3, 8: 4, 9: 5, 11: 6, 12: 7, 13: 8, 14: 9}
        
    pk_cleft = np.zeros((15,len(k)))
    for s in np.arange(15):
        if s==0:
            pk_cleft[s, :] = pk_m_cleft[0]
        elif s in [1, 3, 6, 10]:
            pk_cleft[s, :] = pk_cb_m_cleft[s_m_map[s]]
        else:
            pk_cleft[s, :] = pk_cb_cleft[s_cb_map[s]]

    return pk_cleft

def execute(block, config):
#    from aemulus_heft.heft_emu import HEFTEmulator

    # Recover config information
    input_section, output_section, emulator = config
    # (ombh2, omch2, w0, ns, 10^9 As, H0, mnu)
    
    # Get cosmo params from block
    pars = names.cosmological_parameters
    params = [ block[pars, "Omega_b"] * block[pars, "h0"]**2,
               block[pars, "Omega_c"] * block[pars, "h0"]**2,
               block[pars, "w"],
               block[pars, "n_s"],
               10**9*block[pars, "A_s"],
               block[pars, "h0"] * 100,
               block[pars, "mnu"]]
    try:
        bk0 = block.get('heft_matter_bk', 'bkmaL_0')
        bk_alpha = block.get('heft_matter_bk', 'bkmaL_alpha')
    except:
        bk0 = 0
        bk_alpha = 0
        
    r_cut = 2
    z0 = 0.3

    # Get z and k from the NL power section
    z, k, P_m_lin = block.get_grid(names.matter_power_lin, "z", "k_h", "P_k")
    z, k, P_cb_lin = block.get_grid(names.cdm_baryon_power_lin, "z", "k_h", "P_k")
    sigma8z = block[names.growth_parameters, "sigma_8"]
    z_growth = block[names.growth_parameters, "z"]
    assert(np.allclose(z,z_growth))
    
    # Not sure why but b comes back as a dictionary of arrays
    # instead of a 2D array
    params_heft = params.copy()
    params_heft.append(0)
    
    spec_heft_all = np.zeros((len(z), len(k), 15))
    spec_lpt_all = np.zeros((len(z), len(k), 15))

    for i in range(len(z)):
        spec_lpt = lpt_spectra(k, P_m_lin[i,...], P_cb_lin[i,...])
        params_heft[-1] = sigma8z[i]
        spec_heft = emulator.predict(k, np.array(params_heft), spec_lpt)
        spec_heft_all[i,:,:] = spec_heft.T
        spec_lpt_all[i,:,:] = spec_lpt.T
        
    #counterterm = k[np.newaxis,:]**2 * np.exp((k[np.newaxis,:]/k_cut)**2) * spec_heft_all[:,:,0]
    counterterm = k[np.newaxis,:]**2 / (1 + (r_cut * k[np.newaxis,:]**2)) * spec_heft_all[:,:,0]
    spec_heft_all[:,:,0] = spec_heft_all[:,:,0] - bk0 * ((1+z0)/(1 + z[:,np.newaxis]))**bk_alpha * counterterm
        
    if output_section == input_section:
        # save the original grid to a new section, with _dm added
        # and save the new grid back to the input
        block.put_grid(input_section + "_dm", "z", z, "k_h", k, "P_k", P)
        #0th index is nl total matter power spectrum
        block.replace_grid(input_section, "z", z, "k_h", k, "P_k", spec_heft_all[:,:,0])
    else:
        # just save the output
        block.put_grid(output_section, "z", z, "k_h", k, "P_k", spec_heft_all[:,:,0])
    
    for i in range(spec_heft_all.shape[-1]):
        block.put_grid(f'aemulus_heft_P{i}_nl', "z", z, "k_h", k, "Pij_k", spec_heft_all[...,i])
        block.put_grid(f'aemulus_heft_P{i}_1loop', "z", z, "k_h", k, "Pij_k", spec_lpt_all[...,i])

    return 0

if __name__ == '__main__':
    setup(None)
