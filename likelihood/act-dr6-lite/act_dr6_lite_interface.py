try:
    import act_dr6_cmbonly
except ImportError:
    raise RuntimeError('The act_dr6_cmbonly python module is required for the act_dr6_lite likelihood. It can be obtained from https://github.com/ACTCollaboration/DR6-ACT-lite')
from cosmosis.datablock import names
cosmo = names.cosmological_parameters
import numpy as np
import os


cal_params = [
    "A_act",
    "P_act"
]

dirname = os.path.split(__file__)[0]


def setup(options):
    act = act_dr6_cmbonly.ACTDR6CMBonly(packages_path=dirname)

    ell_min_tt = options.get_int(option_section, 'ell_min_tt', default=600)
    ell_min_te = options.get_int(option_section, 'ell_min_te', default=600)
    ell_min_ee = options.get_int(option_section, 'ell_min_ee', default=600)
    ell_max_tt = options.get_int(option_section, 'ell_max_tt', default=6500)
    ell_max_te = options.get_int(option_section, 'ell_max_te', default=6500)
    ell_max_ee = options.get_int(option_section, 'ell_max_ee', default=6500)
    act.ell_cuts['TT'] = [ell_min_tt, ell_max_tt]
    act.ell_cuts['TE'] = [ell_min_te, ell_max_te]
    act.ell_cuts['EE'] = [ell_min_ee, ell_max_ee]
    print ('ell cuts:', act.ell_cuts)

    # location of synthetic data (cosmoSIS theory output)
    sim_data_directory = options.get_string(option_section, 'use_data_from_test', default='')

    # replace real data with synthetic data
    if sim_data_directory != '':
        print ('ACT likelihood uses synthetic data from:', sim_data_directory)
        sim_ell = np.genfromtxt( sim_data_directory + 'cmb_cl/ell.txt')
        f1 = 1.0 #sim_ell * (sim_ell + 1) / (2 * np.pi)
        sim_cl_tt = np.genfromtxt( sim_data_directory + 'cmb_cl/tt.txt') / f1
        sim_cl_te = np.genfromtxt( sim_data_directory + 'cmb_cl/te.txt') / f1
        sim_cl_ee = np.genfromtxt( sim_data_directory + 'cmb_cl/ee.txt') / f1
        sim_cl_dict = {'tt':sim_cl_tt, 'te':sim_cl_te, 'ee':sim_cl_ee}
        nell = sim_ell.shape

        ps_vec = np.zeros_like(act.data_vec)
        for m in act.spec_meta:
            idx = m["idx"]
            win = m["window"].weight.T
            ls = m["window"].values
            pol = m["pol"]
            dat = sim_cl_dict[pol][ls]
            ps_vec[idx] = win @ dat
        act.data_vec = ps_vec

    return act


def execute(block, config):
    act = config

    cl_dict = {
        "tt": np.append(np.array([0,0]), block[names.cmb_cl, 'tt']),
        "te": np.append(np.array([0,0]), block[names.cmb_cl, 'te']),
        "ee": np.append(np.array([0,0]), block[names.cmb_cl, 'ee']),
    }

    if block.has_value("planck", "a_planck"):
        # When ACT is combined with Planck, they share calibration parameters 
        # https://github.com/ACTCollaboration/DR6-ACT-lite/issues/12 
        # when a_planck is found in value.ini, A_act is fixed to a_planck
        block["act_params", "A_act"] = block["planck", "a_planck"]

    nuisance = {}

    for p in cal_params:
        nuisance[p] = block["act_params", p]

    loglike = act.loglike(cl_dict, **nuisance)

    # Then call the act code
    block[names.likelihoods, 'act_dr6_lite_like'] = loglike

    return 0
