from cosmosis.datablock import names, option_section as opt
from cosmosis.datablock.cosmosis_py import errors
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import warnings
import traceback

#import ipdb
#ipdb.set_trace()

# Finally we can now import mgcamb
#import sys
#sys.path.append('/global/u2/s/sjlee88/cosmosis-global-env/lib/python3.9/site-packages/camb-1.3.6-py3.9.egg/')
import mgcamb 
from mgcamb.dark_energy import DarkEnergyModel
from mgcamb.baseconfig import F2003Class

import pdb

cosmo = names.cosmological_parameters
modified_gravity = "modified_gravity"

MODE_BG = "background"
MODE_THERM = "thermal"
MODE_CMB = "cmb"
MODE_POWER = "power"
MODE_ALL = "all"
MODES = [MODE_BG, MODE_THERM, MODE_CMB, MODE_POWER, MODE_ALL]

# See this table for description:
#https://camb.readthedocs.io/en/latest/transfer_variables.html#transfer-variables
matter_power_section_names = {
    'delta_cdm': 'dark_matter_power',
    'delta_baryon': 'baryon_power',
    'delta_photon': 'photon_power',
    'delta_neutrino': 'massless_neutrino_power',
    'delta_nu': 'massive_neutrino_power',
    'delta_tot': 'matter_power',
    'delta_nonu': 'cdm_baryon_power',
    'delta_tot_de': 'matter_de_power',
    'weyl': 'weyl_curvature_power',
    'v_newtonian_cdm': 'cdm_velocity_power',
    'v_newtonian_baryon': 'baryon_velocity_power',
    'v_baryon_cdm': 'baryon_cdm_relative_velocity_power',
    # SJ edit
    'weyl_matter':'weyl_curvature_matter_power'
}


def get_optional_params(block, section, names):
    params = {}

    for name in names:
        # For some parameters we just use the Camb name as the parameter
        # name, but for other we specify a simpler one
        if isinstance(name, (list, tuple)):
            cosmosis_name, output_name = name
        else:
            cosmosis_name = name
            output_name = name

        # We don't try to set our own default for these parameters,
        # we just let camb decide them on its own
        if block.has_value(section, cosmosis_name):
            params[output_name] = block[section, cosmosis_name]
    return params

def get_choice(options, name, valid, default=None, prefix=''):
    choice = options.get_string(opt, name, default=default)
    if choice not in valid:
        raise ValueError("Parameter setting '{}' in camb must be one of: {}.  You tried: {}".format(name, valid, choice))
    return prefix + choice

def setup(options):
    mode = options.get_string(opt, 'mode', default="all")
    if not mode in MODES:
        raise ValueError("Unknown mode {}.  Must be one of: {}".format(mode, MODES))

    # These are parameters for CAMB
    config = {}

    # These are parameters that we do not pass directly to CAMBparams,
    # but use ourselves in some other way
    more_config = {}

    more_config["mode"] = mode
    more_config["max_printed_errors"] = options.get_int(opt, 'max_printed_errors', default=20)
    more_config["n_printed_errors"] = 0
    config['WantCls'] = mode in [MODE_CMB, MODE_ALL]
    config['WantTransfer'] = mode in [MODE_POWER, MODE_ALL]
    config['WantScalars'] = True
    config['WantTensors'] = options.get_bool(opt, 'do_tensors', default=False)
    config['WantVectors'] = options.get_bool(opt, 'do_vectors', default=False)
    config['WantDerivedParameters'] = True
    config['Want_cl_2D_array'] = False
    config['Want_CMB'] = config['WantCls']
    config['DoLensing'] = options.get_bool(opt, 'do_lensing', default=False)
    config['NonLinear'] = get_choice(options, 'nonlinear', ['none', 'pk', 'lens', 'both'], 
                                     default='none' if mode in [MODE_BG, MODE_THERM] else 'both', 
                                     prefix='NonLinear_')

    config['scalar_initial_condition'] = 'initial_' + options.get_string(opt, 'initial', default='adiabatic')
    
    config['want_zdrag'] = mode != MODE_BG
    config['want_zstar'] = config['want_zdrag']

    # SJ edit : for 6x2pt 
    more_config['want_chistar'] = options.get_bool(opt, 'want_chistar', default=False)
    more_config['n_logz'] = options.get_int(opt, 'n_logz', default=0)
    more_config['zmax_logz'] = options.get_double(opt, 'zmax_logz', default = 1100.)

    more_config["lmax_params"] = get_optional_params(options, opt, ["max_eta_k", "lens_potential_accuracy",
                                                                    "lens_margin", "k_eta_fac", "lens_k_eta_reference",
                                                                    #"min_l", "max_l_tensor", "Log_lvalues", , "max_eta_k_tensor"
                                                                     ])
    # lmax is required
    more_config["lmax_params"]["lmax"] = options.get_int(opt, "lmax", default=2600)                                                  
    
    more_config["initial_power_params"] = get_optional_params(options, opt, ["pivot_scalar", "pivot_tensor"])

    more_config["cosmology_params"] = get_optional_params(options, opt, ["neutrino_hierarchy" ,"theta_H0_range"])

    if 'theta_H0_range' in more_config['cosmology_params']:
        more_config['cosmology_params'] = [float(x) for x in more_config['cosmology_params']['theta_H0_range'].split()]

    more_config['do_reionization'] = options.get_bool(opt, 'do_reionization', default=True)
    more_config['use_optical_depth'] = options.get_bool(opt, 'use_optical_depth', default=True)
    more_config["reionization_params"] = get_optional_params(options, opt, ["include_helium_fullreion", "tau_solve_accuracy_boost", 
                                                                            ("tau_timestep_boost","timestep_boost"), ("tau_max_redshift", "max_redshift")])
    
    more_config['use_tabulated_w'] = options.get_bool(opt, 'use_tabulated_w', default=False)
    more_config['use_ppf_w'] = options.get_bool(opt, 'use_ppf_w', default=False)
    more_config['dark_energy_model'] = options.get_string(opt, 'dark_energy_model', default='fluid')
    if (more_config['dark_energy_model'] == 'EarlyQuintessence'):
        more_config['use_zc'] = options.get_bool(opt, 'use_zc', default=False)
    
    more_config['do_bao'] = options.get_bool(opt, 'do_bao', default=True)
    
    more_config["nonlinear_params"] = get_optional_params(options, opt, ["halofit_version", "Min_kh_nonlinear"])

    halofit_version = more_config['nonlinear_params'].get('halofit_version')
    known_halofit_versions = list(mgcamb.nonlinear.halofit_version_names.keys()) + [None]
    if halofit_version not in known_halofit_versions:
        raise ValueError("halofit_version must be one of : {}.  You put: {}".format(known_halofit_versions, halofit_version))

    more_config["accuracy_params"] = get_optional_params(options, opt, 
                                                        ['AccuracyBoost', 'lSampleBoost', 'lAccuracyBoost', 'DoLateRadTruncation'])
                                                        #  'TimeStepBoost', 'BackgroundTimeStepBoost', 'IntTolBoost', 
                                                        #  'SourcekAccuracyBoost', 'IntkAccuracyBoost', 'TransferkBoost',
                                                        #  'NonFlatIntAccuracyBoost', 'BessIntBoost', 'LensingBoost',
                                                        #  'NonlinSourceBoost', 'BesselBoost', 'LimberBoost', 'SourceLimberBoost',
                                                        #  'KmaxBoost', 'neutrino_q_boost', 'AccuratePolarization', 'AccurateBB',  
                                                        #  'AccurateReionization'])

    more_config['zmin'] = options.get_double(opt, 'zmin', default=0.0)
    more_config['zmax'] = options.get_double(opt, 'zmax', default=3.01)
    more_config['nz'] = options.get_int(opt, 'nz', default=150)
    more_config.update(get_optional_params(options, opt, ["zmid", "nz_mid"]))

    more_config['zmin_background'] = options.get_double(opt, 'zmin_background', default=more_config['zmin'])
    more_config['zmax_background'] = options.get_double(opt, 'zmax_background', default=more_config['zmax'])
    more_config['nz_background'] = options.get_int(opt, 'nz_background', default=more_config['nz'])

    more_config["transfer_params"] = get_optional_params(options, opt, ["k_per_logint", "accurate_massive_neutrino_transfers"])
    # Adjust CAMB defaults
    more_config["transfer_params"]["kmax"] = options.get_double(opt, "kmax", default=10.0)
    # more_config["transfer_params"]["high_precision"] = options.get_bool(opt, "high_precision", default=True)

    more_config['kmin'] = options.get_double(opt, "kmin", default=1e-5)
    more_config['kmax'] = options.get_double(opt, "kmax", more_config["transfer_params"]["kmax"])
    more_config['kmax_extrapolate'] = options.get_double(opt, "kmax_extrapolate", default=more_config['kmax'])
    more_config['nk'] = options.get_int(opt, "nk", default=200)

    more_config['power_spectra'] = options.get_string(opt, "power_spectra", default="delta_tot").split()
    bad_power = []
    for p in more_config['power_spectra']:
        if p not in matter_power_section_names:
            bad_power.append(p)
    if bad_power:
        bad_power = ", ".join(bad_power)
        good_power = ", ".join(matter_power_section_names.keys())
        raise ValueError("""These matter power types are not known: {}.
Please use any these (separated by spaces): {}""".format(bad_power, good_power))

    # MGCAMB modified gravity flags 
    more_config['modified_gravity_params'] = {}
    mgconfig = more_config['modified_gravity_params']
    mgconfig['MG_flag'] = options.get_int(opt, "mg_flag", default=0)
    mgconfig['DE_model'] = options.get_int(opt, "DE_model", default=0)
    mgconfig['pure_MG_flag'] = options.get_int(opt, "pure_mg_flag", default=0)
    mgconfig['mugamma_par'] = options.get_int(opt, "mugamma_par", default=0)
    mgconfig['musigma_par'] = options.get_int(opt, "musigma_par", default=0)
    mgconfig['QR_par'] = options.get_int(opt, "QR_par", default=0)
    mgconfig['alt_MG_flag'] = options.get_int(opt, "alt_MG_flag", default=0)

    #ipdb.set_trace()

    mgcamb.set_feedback_level(level=options.get_int(opt, "feedback", default=0))
    return [config, more_config]

# The extract functions convert from the block to camb parameters
# during the execute function

def extract_recombination_params(block, config, more_config):
    default_recomb = mgcamb.recombination.Recfast()
 
    min_a_evolve_Tm = block.get_double('recfast', 'min_a_evolve_Tm', default=default_recomb.min_a_evolve_Tm)
    RECFAST_fudge = block.get_double('recfast', 'RECFAST_fudge', default=default_recomb.RECFAST_fudge)
    RECFAST_fudge_He = block.get_double('recfast', 'RECFAST_fudge_He', default=default_recomb.RECFAST_fudge_He)
    RECFAST_Heswitch = block.get_int('recfast', 'RECFAST_Heswitch', default=default_recomb.RECFAST_Heswitch)
    RECFAST_Hswitch = block.get_bool('recfast', 'RECFAST_Hswitch', default=default_recomb.RECFAST_Hswitch)
    AGauss1 = block.get_double('recfast', 'AGauss1', default=default_recomb.AGauss1)
    AGauss2 = block.get_double('recfast', 'AGauss2', default=default_recomb.AGauss2)
    zGauss1 = block.get_double('recfast', 'zGauss1', default=default_recomb.zGauss1)
    zGauss2 = block.get_double('recfast', 'zGauss2', default=default_recomb.zGauss2)
    wGauss1 = block.get_double('recfast', 'wGauss1', default=default_recomb.wGauss1)
    wGauss2 = block.get_double('recfast', 'wGauss2', default=default_recomb.wGauss2)
    
    recomb = mgcamb.recombination.Recfast(
        min_a_evolve_Tm = min_a_evolve_Tm, 
        RECFAST_fudge = RECFAST_fudge, 
        RECFAST_fudge_He = RECFAST_fudge_He, 
        RECFAST_Heswitch = RECFAST_Heswitch, 
        RECFAST_Hswitch = RECFAST_Hswitch, 
        AGauss1 = AGauss1, 
        AGauss2 = AGauss2, 
        zGauss1 = zGauss1, 
        zGauss2 = zGauss2, 
        wGauss1 = wGauss1, 
        wGauss2 = wGauss2, 
    )

    #Not yet supporting CosmoRec, but not too hard if needed.

    return recomb

def extract_reionization_params(block, config, more_config):
    reion = mgcamb.reionization.TanhReionization()
    if more_config["do_reionization"]:
        if more_config['use_optical_depth']:
            tau = block[cosmo, 'tau']
            reion = mgcamb.reionization.TanhReionization(use_optical_depth=True, optical_depth=tau)
        else:
            sec = 'reionization'
            redshift = block[sec, 'redshift']
            delta_redshift = block[sec, 'delta_redshift']
            reion_params = get_optional_params(block, sec, ["fraction", "helium_redshift", "helium_delta_redshift", "helium_redshiftstart"])
            reion = mgcamb.reionization.TanhReionization(
                use_optical_depth=False,
                redshift = redshift,
                delta_redshift = delta_redshift,
                include_helium_fullreion = include_helium_fullreion,
                **reion_params,
                **more_config["reionization_params"],
            )
    else:
        reion = mgcamb.reionization.TanhReionization()
        reion.Reionization = False
    return reion

def extract_dark_energy_params(block, config, more_config):

    model = more_config['dark_energy_model']
    if more_config['use_ppf_w']:
        dark_energy = F2003Class.make_class_named('DarkEnergyPPF', DarkEnergyModel)
    else:
        dark_energy = F2003Class.make_class_named(model, DarkEnergyModel)

    if model == 'fluid' or model == 'DarkEnergyPPF' or model == 'ppf':
        if more_config['use_tabulated_w']:
            a = block[names.de_equation_of_state, 'a']
            w = block[names.de_equation_of_state, 'w']
            dark_energy.set_w_a_table(a, w)
        else:
            w0 = block.get_double(cosmo, 'w', default=-1.0)
            wa = block.get_double(cosmo, 'wa', default=0.0)
            cs2 = block.get_double(cosmo, 'cs2_de', default=1.0)
            dark_energy.set_params(w=w0, wa=wa, cs2=cs2)
    elif model == 'AxionEffectiveFluid':
            w_n = block.get_double(cosmo, 'w_n', default=1.0)
            zc  = 10**block.get_double(cosmo, 'log10zc_ede', default=3)
            fde_zc = block.get_double(cosmo, 'f_ede_zc', default=0.001)
            theta_i = 0.5*np.pi*block.get_double(cosmo, 'theta_i_o_pi', default=1.0)
            dark_energy.set_params(w_n=w_n, zc=zc, fde_zc=fde_zc, theta_i=theta_i)
    elif model == 'EarlyQuintessence':
            use_zc = more_config['use_zc']
            n = block.get_double(cosmo, 'n', 3.0)
            f = block.get_double(cosmo, 'f', 0.05)
            m = block.get_double(cosmo, 'm', 5e-54)
            theta_i = block.get_double(cosmo, 'theta_i', 3.0)
            zc = 10.**block.get_double(cosmo, 'log10zc', 3.5)
            fde_zc = block.get_double(cosmo, 'fde_zc', 0.1)
            if (use_zc):
                dark_energy.set_params(n=n, theta_i=theta_i, zc=zc, fde_zc=fde_zc, use_zc = use_zc)
            else:
                dark_energy.set_params(n=n, f=f, m=m, theta_i=theta_i, use_zc=use_zc)
            
    return dark_energy

def extract_initial_power_params(block, config, more_config):
    optional_param_names = ["nrun", "nrunrun", "nt", "ntrun", "r"]
    optional_params = get_optional_params(block, cosmo, optional_param_names)

    init_power = mgcamb.InitialPowerLaw()
    init_power.set_params(
        ns = block[cosmo, 'n_s'],
        As = block[cosmo, 'A_s'],
        **optional_params,
        **more_config["initial_power_params"]
    )
    return init_power

def extract_nonlinear_params(block, config, more_config):
    version = more_config["nonlinear_params"].get('halofit_version', '')

    if version == "mead2015" or version == "mead2016" or version == "mead":
        A = block[names.halo_model_parameters, 'A']
        eta0 = block[names.halo_model_parameters, "eta"]
        hmcode_params = {"HMCode_A_baryon": A, "HMCode_eta_baryon":eta0}
    elif version == "mead2020_feedback":
        T_AGN = block[names.halo_model_parameters, 'logT_AGN']
        hmcode_params = {"HMCode_logT_AGN": T_AGN}
    else:
        hmcode_params = {}

    return mgcamb.nonlinear.Halofit(
        **more_config["nonlinear_params"],
        **hmcode_params
    )

# SJ edit: new function to read mg params 
# Only validated for sig-mu and Linder_gamma.
def extract_modified_gravity_params(block, config, more_config):

    mgconfig = more_config['modified_gravity_params']  

    if mgconfig['MG_flag'] == 0: 
        return 0 
    else: 
        GRtrans = block.get_double(modified_gravity, 'gr_trans')
        # Dark Energy Model (wCDM, w0waCDM, user defined)
        if mgconfig['MG_flag'] == 1: 
            if mgconfig['DE_model'] == 0: pass
            elif mgconfig['DE_model'] == 1: # read wo
                mgconfig['w0DE'] = block.get_double(modified_gravity, 'w0DE')
            elif mgconfig['DE_model'] == 2: # read wowa
                mgconfig['w0DE'] = block.get_double(modified_gravity, 'w0DE')
                mgconfig['waDE'] = block.get_double(modified_gravity, 'waDE')
            elif mgconfig['DE_model'] == 3: # reconstruction of w
                # Omega_x bins
                mgconfig['Funcofw_1']  = block.get_double(modified_gravity, 'OmegaX_idx_1' , default=0.7)
                mgconfig['Funcofw_2']  = block.get_double(modified_gravity, 'OmegaX_idx_2' , default=0.7)
                mgconfig['Funcofw_3']  = block.get_double(modified_gravity, 'OmegaX_idx_3' , default=0.7)
                mgconfig['Funcofw_4']  = block.get_double(modified_gravity, 'OmegaX_idx_4' , default=0.7)
                mgconfig['Funcofw_5']  = block.get_double(modified_gravity, 'OmegaX_idx_5' , default=0.7)
                mgconfig['Funcofw_6']  = block.get_double(modified_gravity, 'OmegaX_idx_6' , default=0.7)
                mgconfig['Funcofw_7']  = block.get_double(modified_gravity, 'OmegaX_idx_7' , default=0.7)
                mgconfig['Funcofw_8']  = block.get_double(modified_gravity, 'OmegaX_idx_8' , default=0.7)
                mgconfig['Funcofw_9']  = block.get_double(modified_gravity, 'OmegaX_idx_9' , default=0.7)
                mgconfig['Funcofw_10'] = block.get_double(modified_gravity, 'OmegaX_idx_10', default=0.7)
                mgconfig['Funcofw_11'] = block.get_double(modified_gravity, 'OmegaX_idx_11', default=0.7)

            if mgconfig['pure_MG_flag'] == 0:
                print ( "Please set pure_MG_flag when setting MG_flag to 1")
                print ( "   pure_MG_flag = 1 : mu, gamma parametrization")
                print ( "   pure_MG_flag = 2 : mu, sigma parametrization")
                print ( "   pure_MG_flag = 3 : Q, R  parametrization")
                exit(1)
            # mugamma model 
            if mgconfig['pure_MG_flag'] == 1:
                if mgconfig['mugamma_par'] == 0:
                    print ( "Please set mugamma_par when setting pure_MG_flag to 1")
                    print ( "   mugamma_par = 1 : BZ parametrization ( introduced in arXiv:0809.3791 )")
                    print ( "   mugamma_par = 2 : Planck parametrization")
                    exit(1)
                if mgconfig['mugamma_par'] == 1:
                    mgconfig['B1'] = block.get_double(modified_gravity, 'b1')
                    mgconfig['B2'] = block.get_double(modified_gravity, 'b2')
                    mgconfig['lambda1_2'] = block.get_double(modified_gravity, 'lambda1_2')
                    mgconfig['lambda2_2'] = block.get_double(modified_gravity, 'lambda2_2')
                    mgconfig['ss'] = block.get_double(modified_gravity, 'ss')
                    
                elif mgconfig['mugamma_par'] == 2:
                    mgconfig['E11'] = block.get_double(modified_gravity, 'E11')
                    mgconfig['E22'] = block.get_double(modified_gravity, 'E22')
                    # read e11, e22 
            # musigma model
            elif mgconfig['pure_MG_flag'] == 2:
                if mgconfig['musigma_par'] == 0:
                    print ( "Please set musigma_par when setting pure_MG_flag to 2")
                    print ( "   musigma_par = 1 : Omega_L parametrization")
                    print ( "   musigma_par = 2 : a^s parametrization")
                    print ( "   musigma_par = 3 : (a,k) parametrization")
                    exit(1)
                elif mgconfig['musigma_par'] == 1:
                    mgconfig['mu0'] = block.get_double(modified_gravity, 'mu0')
                    mgconfig['sigma0'] = block.get_double(modified_gravity, 'sigma0')
                    #modified_gravity.set_params(mu0=mu0, sigma0=sigma0)
                elif mgconfig['musigma_par'] == 2:
                    mgconfig['mu0_s'] = block.get_double(modified_gravity, 'mu0_s')
                    mgconfig['sigma0_s'] = block.get_double(modified_gravity, 'sigma0_s')
                    mgconfig['mg_s'] = block.get_double(modified_gravity, 'mg_s')
                    #modified_gravity.set_params(mu0=mu0_s, sigma0=sigma0_s,mg_s=mg_s)
                elif mgconfig['musigma_par'] == 3:
                    mgconfig['mu0_s'] = block.get_double(modified_gravity, 'mu0_s')
                    mgconfig['sigma0_s'] = block.get_double(modified_gravity, 'sigma0_s')
                    mgconfig['m_Mu'] = block.get_double(modified_gravity, 'm_Mu')
                    mgconfig['m_Sigma'] = block.get_double(modified_gravity, 'm_Sigma')
                    #modified_gravity.set_params(mu0=mu0_s, sigma0=sigma0_s,m_Mu=m_Mu, m_Sigma=m_Sigma)
            # QR model 
            elif mgconfig['pure_MG_flag'] == 3:
                if mgconfig['QR_par'] == 0:
                    print ( "Please set qr_par when setting pure_MG_flag to 3")
                    print ( "   QR_par = 1 : (Q,R) 		( introduced in arXiv:1002.4197 )")
                    print ( "   QR_par = 2 : (Q0,R0,s)	( introduced in arXiv:1002.4197 )")
                    exit(1)
                elif mgconfig['QR_par'] == 1:
                    mgconfig['MGQfix'] = block.get_double(modified_gravity, 'MGQfix')
                    mgconfig['MGRfix'] = block.get_double(modified_gravity, 'MGRfix')
                    #modified_gravity.set_params(MGQfix=MGQfix, MGRfix=MGRfix) 
                elif mgconfig['QR_par'] == 2:
                    mgconfig['Qnot'] = block.get_double(modified_gravity, 'Qnot')
                    mgconfig['Rnot'] = block.get_double(modified_gravity, 'Rnot')
                    mgconfig['sss'] = block.get_double(modified_gravity, 'sss')
                    #modified_gravity.set_params(Qnot=Qnot, Rnot=Rnot,sss=sss) 

        elif mgconfig['MG_flag'] == 2:
            if mgconfig['alt_MG_flag'] == 0: 
                print ( "Please set alt_MG_flag when setting MG_flag to 2")
                print ( "   alt_MG_flag = 1 : Linder Gamma parametrization ( introduced in arXiv:0507263 )")
                exit(1)
            # Linder Gamma
            elif mgconfig['alt_MG_flag'] == 1:    
                mgconfig['Linder_gamma'] = block.get_double(modified_gravity, 'Linder_gamma')

        elif mgconfig['MG_flag'] == 3:
            if mgconfig['QSA_flag'] == 0: 
                print ("Please set qsa_flag when setting mg_flag to 3")
                print ("    QSA_flag = 1 : f(R)")
                print ("    QSA_flag = 2 : Symmetron")
                print ("    QSA_flag = 3 : Dilaton")
                print ("    QSA_flag = 4 : Hu-Sawicki f(R)")
                exit(1)
            elif mgconfig['QSA_flag'] == 1: 
                mgconfig['B1'] = 4./3.
                mgconfig['lambda1_2'] = block.get_double(modified_gravity, 'B0')
                mgconfig['B2'] = 0.5
                mgconfig['ss'] = 4.0

            elif mgconfig['QSA_flag'] == 2: 
                mgconfig['beta_star'] = block.get_double(modified_gravity, 'beta_star')
                mgconfig['a_star'] = block.get_double(modified_gravity, 'a_star')
                mgconfig['xi_star'] = block.get_double(modified_gravity, 'xi_star')
                mgconfig['GRtrans'] = mgconfig['a_star']
            elif mgconfig['QSA_flag'] == 3:
                mgconfig['beta0'] = block.get_double(modified_gravity, 'beta0')
                mgconfig['xi0'] = block.get_double(modified_gravity, 'xi0')
                mgconfig['DilS'] = block.get_double(modified_gravity, 'DilS')
                mgconfig['DilR'] = block.get_double(modified_gravity, 'DilR')
            elif mgconfig['QSA_flag'] == 4:
                mgconfig['F_R0'] = block.get_double(modified_gravity, 'F_R0')
                mgconfig['FRn'] = block.get_double(modified_gravity, 'FRn')
                mgconfig['beta0'] = 1/np.sqrt(6.0)

        elif mgconfig['MG_flag'] == 4:
            # CDM coupling model
            print ("MG_flag = 4 is not implemented yet")
            exit(1)

        elif mgconfig['MG_flag'] == 5:
            # direct mu-sigma parametrization
            print ("MG_flag = 5 is not implemented yet")
            exit(1)
            
        elif mgconfig['MG_flag'] == 6:
            # reconstruction
            print ('DE_model flag is set to 3')
            mgconfig['DE_model'] = 3
            #Mu bins
            mgconfig['MGCAMB_Mu_idx_1']  = block.get_double(modified_gravity, 'Mu_idx_1' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_2']  = block.get_double(modified_gravity, 'Mu_idx_2' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_3']  = block.get_double(modified_gravity, 'Mu_idx_3' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_4']  = block.get_double(modified_gravity, 'Mu_idx_4' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_5']  = block.get_double(modified_gravity, 'Mu_idx_5' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_6']  = block.get_double(modified_gravity, 'Mu_idx_6' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_7']  = block.get_double(modified_gravity, 'Mu_idx_7' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_8']  = block.get_double(modified_gravity, 'Mu_idx_8' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_9']  = block.get_double(modified_gravity, 'Mu_idx_9' , default=1.0)
            mgconfig['MGCAMB_Mu_idx_10'] = block.get_double(modified_gravity, 'Mu_idx_10', default=1.0)
            mgconfig['MGCAMB_Mu_idx_11'] = block.get_double(modified_gravity, 'Mu_idx_11', default=1.0)
            # Sigma bins
            mgconfig['MGCAMB_Sigma_idx_1']  = block.get_double(modified_gravity, 'Sigma_idx_1' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_2']  = block.get_double(modified_gravity, 'Sigma_idx_2' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_3']  = block.get_double(modified_gravity, 'Sigma_idx_3' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_4']  = block.get_double(modified_gravity, 'Sigma_idx_4' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_5']  = block.get_double(modified_gravity, 'Sigma_idx_5' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_6']  = block.get_double(modified_gravity, 'Sigma_idx_6' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_7']  = block.get_double(modified_gravity, 'Sigma_idx_7' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_8']  = block.get_double(modified_gravity, 'Sigma_idx_8' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_9']  = block.get_double(modified_gravity, 'Sigma_idx_9' , default=1.0)
            mgconfig['MGCAMB_Sigma_idx_10'] = block.get_double(modified_gravity, 'Sigma_idx_10', default=1.0)
            mgconfig['MGCAMB_Sigma_idx_11'] = block.get_double(modified_gravity, 'Sigma_idx_11', default=1.0)
            # Omega_x bins
            mgconfig['Funcofw_1']  = block.get_double(modified_gravity, 'OmegaX_idx_1' , default=0.7)
            mgconfig['Funcofw_2']  = block.get_double(modified_gravity, 'OmegaX_idx_2' , default=0.7)
            mgconfig['Funcofw_3']  = block.get_double(modified_gravity, 'OmegaX_idx_3' , default=0.7)
            mgconfig['Funcofw_4']  = block.get_double(modified_gravity, 'OmegaX_idx_4' , default=0.7)
            mgconfig['Funcofw_5']  = block.get_double(modified_gravity, 'OmegaX_idx_5' , default=0.7)
            mgconfig['Funcofw_6']  = block.get_double(modified_gravity, 'OmegaX_idx_6' , default=0.7)
            mgconfig['Funcofw_7']  = block.get_double(modified_gravity, 'OmegaX_idx_7' , default=0.7)
            mgconfig['Funcofw_8']  = block.get_double(modified_gravity, 'OmegaX_idx_8' , default=0.7)
            mgconfig['Funcofw_9']  = block.get_double(modified_gravity, 'OmegaX_idx_9' , default=0.7)
            mgconfig['Funcofw_10'] = block.get_double(modified_gravity, 'OmegaX_idx_10', default=0.7)
            mgconfig['Funcofw_11'] = block.get_double(modified_gravity, 'OmegaX_idx_11', default=0.7)


# We may reinstate these later depending on support in camb itself
# def extract_accuracy_params(block, config, more_config):
#     accuracy = camb.model.AccuracyParams(**more_config["accuracy_params"])
#     return accuracy

# def extract_transfer_params(block, config, more_config):
#     PK_num_redshifts = more_config['nz']
#     PK_redshifts = np.linspace(more_config['zmin'], more_config['zmax'], PK_num_redshifts)[::-1]
#     transfer = camb.model.TransferParams(
#         PK_num_redshifts=PK_num_redshifts,
#         PK_redshifts=PK_redshifts,
#         **more_config["transfer_params"]
#     )
#     return transfer


def extract_camb_params(block, config, more_config):
    want_perturbations = more_config['mode'] not in [MODE_BG, MODE_THERM]
    want_thermal = more_config['mode'] != MODE_BG

    # JMedit - check for input sigma8
    samplesig8 = block.has_value(cosmo, 'sigma_8_input')
    
    # if want_perturbations:
    if not samplesig8: #JMedit; if we're sampling sigma8, wait til we have A_s
        init_power = extract_initial_power_params(block, config, more_config)
    nonlinear = extract_nonlinear_params(block, config, more_config)
# else:
    #     init_power = None
    #     nonlinear = None

    # if want_thermal:
    recomb = extract_recombination_params(block, config, more_config)
    reion = extract_reionization_params(block, config, more_config)
    # else:
    #     recomb = None
    #     reion = None

    dark_energy = extract_dark_energy_params(block, config, more_config)

    # SJ edit
    # modified gravity params 
    extract_modified_gravity_params(block, config, more_config)

    # Currently the camb.model.*Params classes default to 0 for attributes (https://github.com/cmbant/CAMB/issues/50),
    # so we're not using them.
    #accuracy = extract_accuracy_params(block, config, more_config)
    #transfer = extract_transfer_params(block, config, more_config)

    # Get optional parameters from datablock.
    cosmology_params = get_optional_params(block, cosmo, 
        ["TCMB", "YHe", "mnu", "nnu", "standard_neutrino_neff", "num_massive_neutrinos",
         ("A_lens", "Alens")])

    if block.has_value(cosmo, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the effective number of relativistic species in the early Universe.")

    if (block.has_value(cosmo, "omega_nu") or block.has_value(cosmo, "omnuh2")) and not (block.has_value(cosmo, "mnu")):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")



    # Set h if provided, otherwise look for theta_mc
    if block.has_value(cosmo, "hubble"):
        cosmology_params["H0"] = block[cosmo, "hubble"]
    elif block.has_value(cosmo, "h0"):
        cosmology_params["H0"] = block[cosmo, "h0"]*100
    else:
        cosmology_params["cosmomc_theta"] = block[cosmo, "cosmomc_theta"]/100

    #JMedit
    if samplesig8:
        # compute linear matter power spec to figure out what A_s should be
        #  for desired input sigma8, add it to the datblock
        sigma8_to_As(block, config, more_config, cosmology_params, dark_energy, reion)
        init_power = extract_initial_power_params(block, config, more_config)
        
    p = mgcamb.CAMBparams(
        InitPower = init_power,
        Recomb = recomb,
        DarkEnergy = dark_energy,
        #Accuracy = accuracy,
        #Transfer = transfer,
        NonLinearModel=nonlinear,
        **config,
    )

    # Setting up neutrinos by hand is hard. We let CAMB deal with it instead.
    p.set_cosmology(ombh2 = block[cosmo, 'ombh2'],
                    omch2 = block[cosmo, 'omch2'],
                    omk = block[cosmo, 'omega_k'],
                    **more_config["cosmology_params"],
                    **cosmology_params)

    # Fix for CAMB version < 1.0.10
    if np.isclose(p.omnuh2, 0) and "nnu" in cosmology_params and not np.isclose(cosmology_params["nnu"], p.num_nu_massless): 
        p.num_nu_massless = cosmology_params["nnu"]

    # Setting reionization before setting the cosmology can give problems when
    # sampling in cosmomc_theta
    # if want_thermal:
    p.Reion = reion

    p.set_for_lmax(**more_config["lmax_params"])
    p.set_accuracy(**more_config["accuracy_params"])

    # SJ edit
    # setting MG parameters
    p.set_mgparams(**more_config['modified_gravity_params'])
    
    if want_perturbations:
        if "zmid" in more_config:
            z = np.concatenate((np.linspace(more_config['zmin'], 
                                            more_config['zmid'], 
                                            more_config['nz_mid'], 
                                            endpoint=False),
                                np.linspace(more_config['zmid'], 
                                            more_config['zmax'], 
                                            more_config['nz']-more_config['nz_mid'])))[::-1]
        else:
            z = np.linspace(more_config['zmin'], more_config['zmax'], more_config["nz"])[::-1]

        p.set_matter_power(redshifts=z, nonlinear=config["NonLinear"] in ["NonLinear_both", "NonLinear_pk"], **more_config["transfer_params"])

    return p




def save_derived_parameters(r, p, block):
    # Write the default derived parameters to distance section
    derived = r.get_derived_params()
    for k, v in derived.items():
        block[names.distances, k] = v

    block[names.distances, 'rs_zdrag'] = block[names.distances, 'rdrag']
    
    p.omegal = 1 - p.omegam - p.omk
    p.ommh2 = p.omegam * p.h**2

    for cosmosis_name, CAMB_name, scaling in [("h0"               , "h",               1),
                                              ("hubble"           , "h",             100),
                                              ("omnuh2"           , "omnuh2",          1),
                                              ("n_eff"            , "N_eff",           1),
                                              ("num_nu_massless"  , "num_nu_massless", 1),
                                              ("num_nu_massive"   , "num_nu_massive",  1),
                                              ("massive_nu"       , "num_nu_massive",  1),
                                              ("massless_nu"      , "num_nu_massless", 1),
                                              ("omega_b"          , "omegab",          1),
                                              ("omega_c"          , "omegac",          1),
                                              ("omega_nu"         , "omeganu",         1),
                                              ("omega_m"          , "omegam",          1),
                                              ("omega_lambda"     , "omegal",          1),
                                              ("ommh2"            , "ommh2",           1),]:

        CAMB_value = getattr(p, CAMB_name)*scaling

        if block.has_value(names.cosmological_parameters, cosmosis_name):
            input_value = block[names.cosmological_parameters, cosmosis_name]
            if not np.isclose(input_value, CAMB_value, rtol=0.002):
                warnings.warn(f"Parameter {cosmosis_name} inconsistent: input was {input_value} but value is now {CAMB_value}.")
        # Either way we save the results
        block[names.cosmological_parameters, cosmosis_name] = CAMB_value


def save_distances(r, p, block, more_config):

    # Evaluate z on a different grid than the spectra, so we can easily extend it further
    z_background = np.linspace(
        more_config["zmin_background"], more_config["zmax_background"], more_config["nz_background"])

    #If desired, append logarithmically distributed redshifts
    log_z = np.geomspace(more_config["zmax_background"], more_config['zmax_logz'], num = more_config['n_logz'])
    z_background = np.append(z_background, log_z[1:])

    # Write basic distances and related quantities to datablock
    block[names.distances, "nz"] = len(z_background)
    block[names.distances, "z"] = z_background
    block[names.distances, "a"] = 1/(z_background+1)
    block[names.distances, "D_A"] = r.angular_diameter_distance(z_background)
    block[names.distances, "D_M"] = r.comoving_radial_distance(z_background)
    d_L = r.luminosity_distance(z_background)
    block[names.distances, "D_L"] = d_L

    if more_config['want_chistar']:
        chistar = (r.conformal_time(0)- r.tau_maxvis)
        block[names.distances, "CHISTAR"] = chistar

    # Deal with mu(0), which is -np.inf
    mu = np.zeros_like(d_L)
    pos = d_L > 0
    mu[pos] = 5*np.log10(d_L[pos])+25
    mu[~pos] = -np.inf
    block[names.distances, "MU"] = mu
    block[names.distances, "H"] = r.h_of_z(z_background)

    if more_config['do_bao']:
        rs_DV, _, _, F_AP = r.get_BAO(z_background, p).T
        block[names.distances, "rs_DV"] = rs_DV
        block[names.distances, "F_AP"] = F_AP

def compute_growth_factor(r, block, P_tot, k, z, more_config):
    if P_tot is None:
        # If we don't have it already, get the default matter power interpolator,
        # which we use for the growth.
        P_tot = r.get_matter_power_interpolator(nonlinear=False, extrap_kmax=more_config['kmax_extrapolate'])

    # Evaluate it at the smallest k, for the 
    kmin = k.min()
    P_kmin = P_tot.P(z, kmin)

    D = np.sqrt(P_kmin / P_kmin[0]).squeeze()
    return D



def save_matter_power(r, p, block, more_config):

    # Grids in k, z on which to save matter power.
    # There are two kmax values - the max one calculated directly,
    # and the max one extrapolated out too.  We output to the larger
    # of these
    kmax_power = max(more_config['kmax'], more_config['kmax_extrapolate'])
    k = np.logspace(np.log10(more_config['kmin']), np.log10(kmax_power), more_config['nk'])
    z = np.linspace(more_config['zmin'], more_config['zmax'], more_config['nz'])

    P_tot = None

    for transfer_type in more_config['power_spectra']:
        # Deal with case consistency in Weyl option
        tt1 = transfer_type if transfer_type != 'weyl' else 'Weyl'
        if transfer_type == 'weyl_matter':
            tt1='delta_tot'; tt2='Weyl'
        else: tt2=tt1

        # Get an interpolator.  By default bicubic if size is large enough,
        # otherwise it drops down to linear.
        # First we do the linear version of the spectrum
        P, zcalc, kcalc = r.get_matter_power_interpolator(nonlinear=False, var1=tt1, var2=tt2, return_z_k=True,
                                        extrap_kmax=more_config['kmax_extrapolate'])
        assert P.islog
        # P.P evaluates at k instead of logk
        p_k = P.P(z, k, grid=True)

        # Save this for the growth rate later
        if transfer_type == 'delta_tot':
            P_tot = P

        # Save the linear
        section_name = matter_power_section_names[transfer_type] + "_lin"
        block.put_grid(section_name, "z", z, "k_h", k, "p_k", p_k)

        # Now if requested we also save the linear version
        if p.NonLinear is not mgcamb.model.NonLinear_none:
            # Exact same process as before
            P = r.get_matter_power_interpolator(nonlinear=True, var1=tt1, var2=tt2,
                                            extrap_kmax=more_config['kmax_extrapolate'])
            p_k = P.P(z, k, grid=True)
            section_name = matter_power_section_names[transfer_type] + "_nl"
            block.put_grid(section_name, "z", z, "k_h", k, "p_k", p_k)

    # Get growth rates and sigma_8
    sigma_8 = r.get_sigma8()[::-1]
    fsigma_8 = r.get_fsigma8()[::-1]
    rs_DV, H, DA, F_AP = r.get_BAO(z, p).T

    #D = compute_growth_factor(r, block, P_tot, k, z, more_config)
    
    # copied from old mgcamb interface. Linder-gamma cannot produce the same D(z) with GR
    # with the new D function 
    D = sigma_8 / sigma_8[0]
    f = fsigma_8 / sigma_8
    
    # Save growth rates and sigma_8
    block[names.growth_parameters, "z"] = z
    block[names.growth_parameters, "a"] = 1/(1+z)
    block[names.growth_parameters, "sigma_8"] = sigma_8
    block[names.growth_parameters, "fsigma_8"] = fsigma_8
    block[names.growth_parameters, "rs_DV"] = rs_DV
    block[names.growth_parameters, "H"] = H
    block[names.growth_parameters, "DA"] = DA
    block[names.growth_parameters, "F_AP"] = F_AP
    block[names.growth_parameters, "d_z"] = D
    block[names.growth_parameters, "f_z"] = f

    block[names.cosmological_parameters, "sigma_8"] = sigma_8[0]    

    # sigma12 and S_8 - other variants of sigma_8
    sigma12 = r.get_sigmaR(R=12.0, z_indices=-1, hubble_units=False)
    block[names.cosmological_parameters, "sigma_12"] = sigma12
    block[names.cosmological_parameters, "S_8"] = sigma_8[0]*np.sqrt(p.omegam/0.3)


def save_cls(r, p, block):
    # Get total (scalar + tensor) lensed CMB Cls
    cl = r.get_total_cls(raw_cl=False, CMB_unit="muK")
    ell = np.arange(2,cl.shape[0])
    block[names.cmb_cl, "ell"] = ell
    block[names.cmb_cl, "TT"] = cl[2:,0]
    block[names.cmb_cl, "EE"] = cl[2:,1]
    block[names.cmb_cl, "BB"] = cl[2:,2]
    block[names.cmb_cl, "TE"] = cl[2:,3]

    if p.DoLensing:
        # Get CMB lensing potential
        # The cosmosis-standard-library clik interface expects ell(ell+1)/2 pi Cl
        # for all angular power spectra, including the lensing potential.
        # For compatability reasons, we provide that scaling here as well.
        cl = r.get_lens_potential_cls(lmax=ell[-1], raw_cl=True, CMB_unit="muK")
        block[names.cmb_cl, "PP"] = cl[2:,0]*(ell*(ell+1))/(2*np.pi)
        block[names.cmb_cl, "PT"] = cl[2:,1]*(ell*(ell+1))/(2*np.pi)
        block[names.cmb_cl, "PE"] = cl[2:,2]*(ell*(ell+1))/(2*np.pi)


# JMedit: new function here
def sigma8_to_As(block, config, more_config, cosmology_params, dark_energy, reion):
    """
    If input parameters include sigma_8_input, convert that to A_s.

    This function will run CAMB once to compute the linear  matter power spectrum 

    This function is adapted from the sigma8toAs module in the
    KIDS KCAP repoistory written by by Tilman Troester.
    """
    sigma_8_input = block[cosmo,'sigma_8_input']
    temp_As = 2.1e-9
    block[cosmo,'A_s'] = temp_As
    init_power_temp = extract_initial_power_params(block, config, more_config)

    # do nothing except get linear power spectrum
    p_temp = mgcamb.CAMBparams(WantTransfer=True,
                             Want_CMB=False, Want_CMB_lensing=False, DoLensing=False,
                             NonLinear="NonLinear_none",
                             WantTensors=False, WantVectors=False, WantCls=False,
                             WantDerivedParameters=False,
                             want_zdrag=False, want_zstar=False,\
                             DarkEnergy=dark_energy,
                             InitPower = init_power_temp,\
                             )
    # making these choices match main setup
    p_temp.set_accuracy(**more_config["accuracy_params"])
    p_temp.set_cosmology(ombh2 = block[cosmo, 'ombh2'],
                         omch2 = block[cosmo, 'omch2'],
                         omk = block[cosmo, 'omega_k'],
                         **more_config["cosmology_params"],
                         **cosmology_params)
    p_temp.set_matter_power(redshifts=[0.], nonlinear=False, **more_config["transfer_params"])
    p_temp.Reion = reion
    r_temp = mgcamb.get_results(p_temp)
    temp_sig8 = r_temp.get_sigma8()[-1] #what sigma8 comes out from using temp_As?
    As = temp_As*(sigma_8_input/temp_sig8)**2
    block[cosmo,'A_s'] = As
    #print(">>>>> temp_As",temp_As,'As',As,'sigma_8_input',sigma_8_input,'temp_sig8',temp_sig8)
    

def execute(block, config):
    config, more_config = config
    p = extract_camb_params(block, config, more_config)
    

    try:
        if (not p.WantCls) and (not p.WantTransfer):
            # Background only mode
            r = mgcamb.get_background(p)
        else:
            # other modes
            r = mgcamb.get_results(p)
    except camb.CAMBError:
        if more_config["n_printed_errors"] <= more_config["max_printed_errors"]:
            print("CAMB error caught: for these parameters")
            print(p)
            print(traceback.format_exc())
            if more_config["n_printed_errors"] == more_config["max_printed_errors"]:
                print("\nFurther errors will not be printed.")
            more_config["n_printed_errors"] += 1
        return 1

    save_derived_parameters(r, p, block)
    save_distances(r, p, block, more_config)

    if p.WantTransfer:
        save_matter_power(r, p, block, more_config)

    if p.WantCls:
        save_cls(r, p, block)
    
    return 0

# Transfer – camb.model.TransferParams

# nu_mass_eigenstates – (integer) Number of non-degenerate mass eigenstates
# share_delta_neff – (boolean) Share the non-integer part of num_nu_massless between the eigenstates
# nu_mass_degeneracies – (float64 array) Degeneracy of each distinct eigenstate
# nu_mass_fractions – (float64 array) Mass fraction in each distinct eigenstate
# nu_mass_numbers – (integer array) Number of physical neutrinos per distinct eigenstate
# scalar_initial_condition – (integer/string, one of: initial_adiabatic, initial_iso_CDM, initial_iso_baryon, initial_iso_neutrino, initial_iso_neutrino_vel, initial_vector)

# MassiveNuMethod – (integer/string, one of: Nu_int, Nu_trunc, Nu_approx, Nu_best)
# DoLateRadTruncation – (boolean) If true, use smooth approx to radition perturbations after decoupling on small scales, saving evolution of irrelevant osciallatory multipole equations

# Evolve_baryon_cs – (boolean) Evolve a separate equation for the baryon sound speed rather than using background approximation
# Evolve_delta_xe – (boolean) Evolve ionization fraction perturbations
# Evolve_delta_Ts – (boolean) Evolve the splin temperature perturbation (for 21cm)

# Log_lvalues – (boolean) Use log spacing for sampling in L
# use_cl_spline_template – (boolean) When interpolating use a fiducial spectrum shape to define ratio to spline


def test(**kwargs):
    from cosmosis.datablock import DataBlock
    options = DataBlock.from_yaml('test_setup.yml')
    for k,v in kwargs.items():
        options[opt, k] = v
        print("set", k)
    config = setup(options)
    block = DataBlock.from_yaml('test_execute.yml')
    return execute(block, config)
    

if __name__ == '__main__':
    test()
