from cosmosis.datablock import option_section, names
import numpy as np
  
def setup(options):
    output_name = options.get_string(option_section, "output_name", default="cmbkappa_cl")
    return {'output_name':output_name}

def execute(block, config):
    ell = block[names.cmb_cl,'ell']
    Clkk = block[names.cmb_cl,'pp']
    block[config['output_name'],'ell'] = ell
    block[config['output_name'],'sep_name'] = 'ell'         
    block[config['output_name'],'bin_1_1'] = Clkk
    block[config['output_name'],'is_auto'] = True
    block[config['output_name'],'nbin'] = 1
    block[config['output_name'],'nbin_a'] = 1
    block[config['output_name'],'nbin_b'] = 1         
    block[config['output_name'],'save_name'] = ''
    block[config['output_name'],'sample_a'] = 'cmb'
    block[config['output_name'],'sample_b'] = 'cmb'
    return 0

def cleanup(config):
    pass
