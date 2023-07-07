from cosmosis.datablock import names, option_section
#import matplotlib.pyplot as plt

def setup(options):
    return []


def execute(block, config):
    #need to implement binwise bias here
    #b = block['galaxy_bias', 'b']

    # Copy the non-linear matter power to both galaxy-power and
    # matter-galaxy cross power (the latter is used in lensing-position spectra)
    block._copy_section(names.matter_power_nl, names.galaxy_power)
    block._copy_section(names.matter_power_nl, names.matter_galaxy_power)
    block._copy_section(names.matter_power_lin, names.galaxy_power+"_lin")
    block._copy_section(names.matter_power_lin, names.matter_galaxy_power+"_lin")
    cosmo = names.cosmological_parameters
    active_norm = 93.14
    sterile_norm = 94.1
    lin = names.matter_power_lin
    f_cb = 1.0
    has_mnu = block.has_value(cosmo, "mnu")
    has_neff = block.has_value(cosmo, "deltaneff")
    has_meff = block.has_value(cosmo, "meff")
    #todo: check to make sure we can call lin_cb    
    if not block.has_section("cdm_baryon_power_lin"):
        print("ERROR: need to calculate cdm_baryon power spectrum in ini file\n")
    lin_cb = names.cdm_baryon_power_lin
    print(has_mnu, has_neff, has_meff)
    if (has_mnu==False) and ((has_neff and has_meff)==False):
        print("ERROR: no neutrinos for nisdb, implementing constant bias\n")
    

    if has_mnu:
        f_cb -= (block[cosmo, "mnu"]/active_norm/block[cosmo, "h0"]/block[cosmo, "h0"])/block[cosmo, "omega_m"]
    if has_neff and has_meff:
        #thermal case
        f_cb -= (block[cosmo, "meff"]/sterile_norm/block[cosmo, "h0"]/block[cosmo, "h0"]/(block[cosmo, "deltaNeff"]))/block[cosmo, "omega_m"]
    print(f_cb, block[cosmo, "omega_m"],block[cosmo, "h0"],block[cosmo, "mnu"])
    z_lin, k_lin, p_lin = block.get_grid(lin, "z", "k_h", "p_k")
    z_lin_cb, k_lin_cb, p_lin_cb = block.get_grid(lin_cb, "z", "k_h", "p_k")
    test_z,test_k,test_pk = block.get_grid(names.cdm_baryon_power_nl, "z", "k_h", "p_k")


    z_gg, k_gg, p_gg = block.get_grid(names.galaxy_power, "z", "k_h", "p_k")
    z_gm, k_gm, p_gm = block.get_grid(names.matter_galaxy_power, "z", "k_h", "p_k")
    z_gg_lin, k_gg_lin, p_gg_lin = block.get_grid(names.galaxy_power+"_lin", "z", "k_h", "p_k")
    z_gm_lin, k_gm_lin, p_gm_lin = block.get_grid(names.matter_galaxy_power+"_lin", "z", "k_h", "p_k")
    #plt.plot(test_k, test_pk[0], label="CDM+baryon NL")
    #plt.plot(test_k, p_lin_cb[0], label="CDM+baryon L")
    #plt.plot(test_k, p_gg[0], label="Total Matter NL")
    #plt.plot(test_k, p_lin[0], label="Total Matter L")
    #plt.plot(test_k, (f_cb**2 * test_pk[0] + (p_lin[0] - f_cb**2 * p_lin_cb[0]))/p_gg[0] - 1, label="NL Approx")
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.legend()
    #plt.show()

    if block.has_section("matter_intrinsic_power"):
        block._copy_section("matter_intrinsic_power", "galaxy_intrinsic_power")
        z_intr, k_intr, p_intr = block.get_grid(names.galaxy_intrinsic_power, "z", "k_h", "p_k")
        #block._copy_section("matter_intrinsic_power_lin", "galaxy_intrinsic_power_lin")
        #z_intr_lin, k_intr_lin, p_intr_lin = block.get_grid(names.galaxy_intrinsic_power+"_lin", "z", "k_h", "p_k")
    #assert(z_lin==z_lin_cb and z_lin==z_gg and z_lin==z_gm)
    #assert(k_lin==k_lin_cb and k_lin==k_gg and k_lin==k_gm)
    #if block.has_section("matter_intrinsic_power"): assert(z_lin==z_intr and k_lin==k_intr)
    print(block.has_section("matter_intrinsic_power"), names.galaxy_intrinsic_power)
    for i in range(len(z_lin)):
        for j in range(len(k_lin)):
            nisdb = (1.0 + f_cb *(p_lin_cb[i][j]/p_lin[i][j]))/(1.0+f_cb)
            p_gg[i][j]*= nisdb**2
            p_gm[i][j]*=nisdb
            p_gg_lin[i][j]*= nisdb**2
            p_gm_lin[i][j]*=nisdb
            if block.has_section("matter_intrinsic_power"): 
                p_intr[i][j]*=nisdb
                #p_intr_lin[i][j]*=nisdb
    
    # Now apply constant biases to the values we have just copied.
    block.put_grid(names.galaxy_power,"z", z_gg, "k_h", k_gg, "p_k", p_gg)
    block.put_grid(names.matter_galaxy_power,"z", z_gm, "k_h", k_gm, "p_k", p_gm)
    block.put_grid(names.galaxy_power+"_lin","z", z_gg_lin, "k_h", k_gg_lin, "p_k", p_gg_lin)
    block.put_grid(names.matter_galaxy_power+"_lin","z", z_gm_lin, "k_h", k_gm_lin, "p_k", p_gm_lin)    
    # We may have a matter intrinsic power aleady worked out.
    if block.has_section("matter_intrinsic_power"): 
        block.put_grid(names.galaxy_intrinsic_power,"z", z_intr, "k_h", k_intr, "p_k", p_intr)
        #block.put_grid(names.galaxy_intrinsic_power+"_lin","z", z_intr_lin, "k_h", k_intr_lin, "p_k", p_intr_lin)

    return 0
