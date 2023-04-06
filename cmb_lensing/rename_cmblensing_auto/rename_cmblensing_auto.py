from cosmosis.datablock import option_section, names
import numpy as np
  
def setup(options):
    output_name = options.get_string(option_section, "output_name", default="cmbkappa_cl")
    convert_to_convergence = options.get_bool(option_section, "convert_to_convergence", default=True)
    return {'output_name':output_name, 'convert_to_convergence':convert_to_convergence}

def execute(block, config):
    ell = block[names.cmb_cl,'ell']
    Cl_block = block[names.cmb_cl,'pp'] # this is Clpp*(ell*(ell+1))/2pi
    if config['convert_to_convergence']:
        Clpp = Cl_block*2.*np.pi/(ell*(ell+1.))
        Clkk = Clpp*(1./4.)*(ell*(ell+1.))**2.
        block[config['output_name'],'bin_1_1'] = Clkk
    else:
        block[config['output_name'],'bin_1_1'] = Cl_block
        
    block[config['output_name'],'ell'] = ell
    block[config['output_name'],'sep_name'] = 'ell'         
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
