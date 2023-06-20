from cosmosis.datablock import names, option_section
import numpy as np


def setup(options):
    model = options.get_string(option_section, "model", default="").lower()

    models = ['power_law','linear_in_z','linear_in_a']
    if model not in models:
        raise ValueError('choice of z dependence not recognised: %s'%model)

    return model


##### z dependency options
def lin_z(z,A,z0,beta):
    return A + beta*(z-z0)

def lin_a(z,A,z0,beta):
    a = 1/(1+z)
    a0 = 1/(1+z0)
    return A - beta*(a-a0)

def power_law_z(z,A,z0,alpha):
    return A*((1+z)/(1+z0))**alpha


def get_z_factor(block,i,model,z,z0):
    # amplitude at the pivot redshift                                                                                                                                                      
    A = block['intrinsic_alignment_parameters','A%d'%i]

    # now choose a redshift dependence and get the corresponding scaling
    if (model=='power_law'):
        alpha = block['intrinsic_alignment_parameters','alpha%d'%i]
        Az = power_law_z(z,A,z0,alpha)

    elif (model=='linear_in_z'):
        beta = block['intrinsic_alignment_parameters','beta%d'%i]
        Az = lin_z(z,A,z0,beta)

    elif (model=='linear_in_a'):
        gamma = block['intrinsic_alignment_parameters','gamma%d'%i]
        Az = lin_a(z,A,z0,gamma)

    return Az


def execute(block, config):
    # Get the names of the sections to save to
    model = config

    # we just need this for the z array
    z, k, Pk = block.get_grid("matter_power_lin", "z", "k_h", "p_k")
    z0 = block['intrinsic_alignment_parameters','z_piv']

    # if A1 exists, rescale it by the relevant z dependent factor
    # and then resave it to the block
    if block.has_value('intrinsic_alignment_parameters','A1'):
        A1_z = get_z_factor(block, 1, model, z, z0)
        block['intrinsic_alignment_parameters','A1_z'] = A1_z

    # exactly the same thing for A2
    if block.has_value('intrinsic_alignment_parameters','A2'):
        A2_z = get_z_factor(block, 2, model, z, z0)
        block['intrinsic_alignment_parameters','A2_z'] = A2_z

    return 0


def cleanup(config):
    pass
