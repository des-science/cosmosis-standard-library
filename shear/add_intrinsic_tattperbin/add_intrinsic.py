from __future__ import print_function
from builtins import range
from cosmosis.datablock import option_section, names

def setup(options):
    do_shear_shear = options.get_bool(option_section, "shear-shear", True)
    do_position_shear = options.get_bool(option_section, "position-shear", True)
    perbin = options.get_bool(option_section, "perbin", False)

    suffix = options.get_string(option_section, "suffix", "")

    print()
    print("The add_intrinsic module will try to combine IA terms into these spectra:")
    if do_shear_shear:
        print(" - shear-shear.")
    print()

    if suffix:
        suffix = "_" + suffix

    sec_names = {
        "shear_shear": "shear_cl" + suffix,
        "shear_shear_bb": "shear_cl_bb" + suffix,
        "shear_shear_gg": "shear_cl_gg" + suffix,
        "galaxy_shear": "galaxy_shear_cl" + suffix,
        "shear_intrinsic": "shear_cl_gi" + suffix,
        "galaxy_intrinsic": "galaxy_intrinsic_cl"  + suffix,
        "magnification_intrinsic" : "magnification_intrinsic_cl" + suffix,
        "intrinsic_intrinsic": "shear_cl_ii"  + suffix,
        "intrinsic_intrinsic_bb": "shear_cl_bb"  + suffix,
        "parameters": "intrinsic_alignment_parameters" + suffix,
    }   

    return do_shear_shear, do_position_shear, sec_names


def execute(block, config):
    do_shear_shear, do_position_shear, sec_names = config

    shear_shear = sec_names['shear_shear']
    shear_shear_gg = sec_names['shear_shear_gg']
    shear_intrinsic = sec_names['shear_intrinsic']
    intrinsic_intrinsic = sec_names['intrinsic_intrinsic']
    intrinsic_intrinsic_bb = sec_names['intrinsic_intrinsic_bb']
    shear_shear_bb = sec_names['shear_shear_bb']
    galaxy_shear = sec_names['galaxy_shear']
    galaxy_intrinsic = sec_names['galaxy_intrinsic']
    magnification_intrinsic = sec_names['magnification_intrinsic']
    nbin_shear = block[shear_shear, 'nbin']

    if do_position_shear:
        nbin_pos = block[galaxy_shear, 'nbin_a']


    # for shear-shear, we're replacing 'shear_cl' (the GG term) with GG+GI+II...
    # so in case useful, save the GG term to shear_cl_gg.
    # also check for a b-mode contribution from IAs
    block[shear_shear_gg, 'ell'] = block[shear_shear, 'ell']

    # Add metadata to the backup gg-only section
    for key in ["nbin_a", "nbin_b", "nbin", "sample_a", "sample_b", "is_auto", "auto_only", "sep_name"]:
        if block.has_value(shear_shear, key):
            block[shear_shear_gg, key] = block[shear_shear, key]


    for i in range(nbin_shear):
        for j in range(i + 1):
            bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
            bin_ji = 'bin_{1}_{0}'.format(i + 1, j + 1)

            A1_i = block["intrinsic_alignment_parameters","A1_%d"%(i+1)]
            A2_i = block["intrinsic_alignment_parameters","A2_%d"%(i+1)] 

            A1_j = block["intrinsic_alignment_parameters","A1_%d"%(j+1)]
            A2_j = block["intrinsic_alignment_parameters","A2_%d"%(j+1)] 

            block[shear_shear_gg, bin_ij] = block[shear_shear, bin_ij]

            # we need to reconstruct the GI and II C_ells now
            GI = A1_j * block[shear_intrinsic + "_xa1", bin_ij] + A2_j * block[shear_intrinsic + "_xa2", bin_ij] 
            IG = A1_i * block[shear_intrinsic + "_xa1", bin_ji] + A2_i * block[shear_intrinsic + "_xa2", bin_ji] 

            # factors of 0.5 below counteract some factors of 2 in the mixed terms in the TATT module
            II_EE = (A1_i * A1_j * block[intrinsic_intrinsic + "_xa1a1", bin_ij] 
                  + A2_i * A2_j * block[intrinsic_intrinsic + "_xa2a2", bin_ij] 
                  + 0.5 * A1_i * A2_j * block[intrinsic_intrinsic + "_xa1a2", bin_ij] 
                  + 0.5 * A1_j * A2_i * block[intrinsic_intrinsic + "_xa1a2", bin_ji])


            II_BB = (A1_i * A1_j * block[intrinsic_intrinsic_bb + "_xa1a1", bin_ij] 
                  + A2_i * A2_j * block[intrinsic_intrinsic_bb + "_xa2a2", bin_ij] 
                  + 0.5 * A1_i * A2_j * block[intrinsic_intrinsic_bb + "_xa1a2", bin_ij] 
                  + 0.5 * A1_j * A2_i * block[intrinsic_intrinsic_bb + "_xa1a2", bin_ji])

            # add everything to GG
            block[shear_shear, bin_ij] += II_EE + GI + IG
            if block.has_section(intrinsic_intrinsic_bb):
                    block[shear_shear_bb, bin_ij] = II_BB


    if do_position_shear:
        for i in range(nbin_pos):
            for j in range(nbin_shear):
                bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
                A1_j = block["intrinsic_alignment_parameters","A1_%d"%(j+1)]
                A2_j = block["intrinsic_alignment_parameters","A2_%d"%(j+1)] 

                gI = A1_j * block[galaxy_intrinsic+"_xa1", bin_ij] + A2_j * block[galaxy_intrinsic+"_xa2", bin_ij]
                mI = A1_j * block[magnification_intrinsic+"_xa1", bin_ij] + A2_j * block[magnification_intrinsic+"_xa2", bin_ij]

                block[galaxy_shear, bin_ij] += gI + mI

    return 0
