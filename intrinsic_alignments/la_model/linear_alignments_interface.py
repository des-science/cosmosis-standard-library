from cosmosis.datablock import names, option_section
from linear_alignments import kirk_rassat_host_bridle_power
from linear_alignments import bridle_king
from linear_alignments import bridle_king_corrected
import numpy as np

def setup(options):
	method = options[option_section, "method"].lower()
	if method not in ["krhb", "bk", "bk_corrected"]:
		raise ValueError('The method in the linear alignment module must'
			'be either "KRHB" (for Kirk, Rassat, Host, Bridle) or BK for '
			'Bridle & King or "BK_corrected" for the corrected version of that')
	return method

def execute(block, config):
	# load z_lin, k_lin, P_lin, z_nl, k_nl, P_nl, C1, omega_m, H0
	lin = names.matter_power_lin
	nl = names.matter_power_nl
	ia = names.intrinsic_alignment_parameters
	cosmo = names.cosmological_parameters

	method = config

	#z_lin = block[lin, "z"]
	#k_lin = block[lin, "k_h"]
	#p_lin = block[lin, "p_k"]
	#z_nl = block[nl, "z"]
	#k_nl = block[nl, "k_h"]
	#p_nl = block[nl, "p_k"]

	#p_lin = p_lin.reshape((z_lin.size, k_lin.size)).T
	#p_nl = p_nl.reshape((z_nl.size, k_nl.size)).T
	z_lin,k_lin,p_lin=block.get_grid(lin,"z","k_h","p_k")
	z_nl,k_nl,p_nl=block.get_grid(nl,"z","k_h","p_k")

	omega_m = block[cosmo, "omega_m"]
	A = block[ia, "A"]
	#if abs(A)<1e-6:
	#	block.put_grid(ia,"z",z_lin,"k_h",k_lin,"P_GI",np.zeros_like(p_nl))
	#	block.put_grid(ia,"z",z_lin,"k_h",k_lin,"P_II",np.zeros_like(p_nl))

	#run computation
	if method=='krhb':
		P_II, P_GI = kirk_rassat_host_bridle_power(z_lin, k_lin, p_lin, z_nl, k_nl, p_nl, A, omega_m)
		block.put_grid(ia, "z", z_lin, "k_h", k_lin, "P_GI", P_GI)
		block.put_grid(ia, "z", z_lin, "k_h", k_lin, "P_II", P_II)
	elif method=='bk':
		P_II, P_GI = bridle_king(z_nl, k_nl, p_nl, A, omega_m)
		block.put_grid(ia, "z", z_nl, "k_h", k_nl, "P_GI", P_GI)
		block.put_grid(ia, "z", z_nl, "k_h", k_nl, "P_II", P_II)
	elif method=='bk_corrected':
		P_II, P_GI = bridle_king_corrected(z_nl, k_nl, p_nl, A, omega_m)
		block.put_grid(ia, "z", z_nl, "k_h", k_nl, "P_GI", P_GI)
		block.put_grid(ia, "z", z_nl, "k_h", k_nl, "P_II", P_II)



	return 0

def cleanup(config):
	pass