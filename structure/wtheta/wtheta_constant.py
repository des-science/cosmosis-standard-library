from cosmosis.datablock import names, option_section
import numpy as np

def setup(options):
    A_section = options.get_string(option_section, "A_section", "A_section")
    return A_section
def execute(block, config):
    A_section = config
    nbins = 0
    for i in range(1,9999):
        A_label = "A%d"%i
        if not block.has_value(A_section, A_label):
            break
        A_in = block[A_section, A_label]
        name = 'bin_%d_%d' % (i, i)
        wt_in = block['galaxy_xi', name]
        wt_out = wt_in + np.power(10,A_in)
        block['galaxy_xi', name] = wt_out
        nbins += 1
    if nbins == 0:
        raise ValueError("we found zero bins, this seems wrong so let's raise an error")
    return 0
