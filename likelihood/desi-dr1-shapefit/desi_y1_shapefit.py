import os
import numpy as np
from numpy import log, pi, interp
from scipy.interpolate import RectBivariateSpline

from cosmosis.datablock import option_section, names
from cosmosis.gaussian_likelihood import GaussianLikelihood

ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]
DAT_DEFAULT_FILENAME=os.path.join(ROOT_DIR, "desi_2024_gaussian_shapefit-bao_ALL_GCcomb_mean.txt")
COV_DEFAULT_FILENAME=os.path.join(ROOT_DIR, "desi_2024_gaussian_shapefit-bao_ALL_GCcomb_cov.txt")

# Table 11 from https://arxiv.org/pdf/2411.12021

#Rs_fid = 99.0792  #Mpc/h
RS_FID = 1.0 

class DESIY1ShapeFitLikelihood(GaussianLikelihood):

    like_name = "desi_y1_shapefit"

    def __init__(self, options):

        dat_file = options.get_string("data_filename", default=DAT_DEFAULT_FILENAME)
        cov_file = options.get_string("cov_filename", default=COV_DEFAULT_FILENAME)

        self.data = np.genfromtxt(dat_file, comments="#", dtype=[("z", float), ("value", float), ("quantity", "U32")])
        self.cov = np.loadtxt(cov_file)

        self.feedback = options.get_bool("feedback", default=False)
        super().__init__(options)

    def build_data(self):

        zeffs, flatdata, quantities = self.data['z'], self.data['value'], self.data['quantity']

        # Unique z in file order (no duplicates)
        _, first_idx = np.unique(zeffs, return_index=True)
        self.zeff = zeffs[np.sort(first_idx)]
        self.nzeff = self.zeff.size

        if self.feedback:
            print ('Data points >>')
            # Arrange values in a 2D array: rows = z, cols = quantity
            q_unique = np.unique(quantities)
            values = np.full((zeffs.size, q_unique.size), np.nan)

            for j, q in enumerate(q_unique):
                m = quantities == q
                values[np.unique(zeffs[m], return_inverse=True)[1], j] = flatdata[m]

            # Example: DM_over_rs column as 1D array aligned with z_unique
            dm_over_rs = values[:, np.where(q_unique == "DM_over_rs")[0][0]]
            dh_over_rs = values[:, np.where(q_unique == "DH_over_rs")[0][0]]
            f_sigmas8 = values[:, np.where(q_unique == "f_sigmas8")[0][0]]

            print('zeff   DM/rd   DH/rd   fs8')
            for z, dm, dh, fs in zip(self.zeff, dm_over_rs[:self.nzeff], dh_over_rs[:self.nzeff], f_sigmas8[:self.nzeff]):
                print(f'{z:.2f}   {dm:.2f}   {dh:.2f}    {fs:.3f}')

        return zeffs, flatdata

    def build_covariance(self):
        return self.cov

    def build_inverse_covariance(self):
        return np.linalg.inv(self.cov)

    def extract_theory_points(self, block):
        
        z = block[names.distances, 'z']

        # Sound horizon at the drag epoch
        rd = block[names.distances, "rs_zdrag"]
        if self.feedback:
            print(f'rs_zdrag = {rd}')

        # Comoving distance
        DM_z = block[names.distances, 'd_m']  # in Mpc

        # Hubble distance
        DH_z = 1/block[names.distances, 'H'] # in Mpc

        # Angle-averaged distance
        #DV_z = (z * DM_z**2 * DH_z)**(1/3) # in Mpc

        # z and distance maybe are loaded in chronological order
        # Reverse to start from low z
        if (z[1] < z[0]):
            z  = z[::-1]
            DM_z = DM_z[::-1]
            DH_z = DH_z[::-1]

        # Find theory DM and DH at effective redshift by interpolation
        DMrd = np.interp(self.zeff, z, DM_z)/rd
        DHrd = np.interp(self.zeff, z, DH_z)/rd
        #DVrd = (self.zeff * DMrd**2 * DHrd)**(1/3)

        # computed fsigmas8 prediction 
        z_gro = block[names.growth_parameters, 'z']
        if z_gro.max() < self.zeff.max():
            raise ValueError("You need to calculate growth parameters up to z={:.2f} to use the DESI Y1 ShapeFit likelihood".format(self.zeff.max()))
        #if block.has_value(names.growth_parameters, "fsigma_8"):
        #    fsig = interp(self.zeff, z_gro, block[names.growth_parameters, "fsigma_8"])

        # Compute fsigma_s8
        omegab = block[names.cosmological_parameters, 'omega_b']
        omegac = block[names.cosmological_parameters, 'omega_c']
        omegam = block[names.cosmological_parameters, 'omega_m']
        wb = omegab/omegam
        wc = omegac/omegam

        rs = rd / RS_FID
        s8 = 8 * rs/ 99.0792

        # should be same with R values from camb_interface.py
        R = block[names.growth_parameters, "R"]
        #R_vc, z_vc, sigma8_vc = r.get_sigmaR(R, var1='v_newtonian_cdm', var2='v_newtonian_cdm', return_R_z=True)
        sigma8_vc = block[names.growth_parameters, "sigma_8_vc"]
        sigma8_vb = block[names.growth_parameters, "sigma_8_vb"]
        sigma8_vcb = block[names.growth_parameters, "sigma_8_vcb"]

        fsigmas8_vc = RectBivariateSpline(z_gro, R, sigma8_vc, kx=3, ky=3)(z_gro, s8)
        fsigmas8_vb = RectBivariateSpline(z_gro, R, sigma8_vb, kx=3, ky=3)(z_gro, s8)
        fsigmas8_vcb = RectBivariateSpline(z_gro, R, sigma8_vcb, kx=3, ky=3)(z_gro, s8)

        fsigmas8 = wb **2 * fsigmas8_vb + wc ** 2 * fsigmas8_vc + 2 * wb * wc * fsigmas8_vcb
        block[names.growth_parameters, "fsigmas8"] = fsigmas8.ravel()
        fsigs8 = interp(self.zeff, z_gro, block[names.growth_parameters, "fsigmas8"])
        fsigs8[-1] = np.nan

        if self.feedback:
            print('Theory distances')
            print('zeff   DM/rd   DH/rd   fs8')
            for z, dm, dh, fs in zip(self.zeff, DMrd, DHrd, fsigs8):
                print(f'{z:.2f}   {dm:.2f}   {dh:.2f}    {fs:.3f}')

        flattheory = np.column_stack([DMrd, DHrd, fsigs8]).ravel() [:-1]
        return flattheory

setup, execute, cleanup = DESIY1ShapeFitLikelihood.build_module()

