import os
import numpy as np
from numpy import log, pi, interp

from cosmosis.datablock import option_section, names
from cosmosis.gaussian_likelihood import GaussianLikelihood

ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]
ISO_DEFAULT_FILENAME=os.path.join(ROOT_DIR, "desi_y1_shapefit_iso.txt")
ANI_DEFAULT_FILENAME=os.path.join(ROOT_DIR, "desi_y1_shapefit_ani.txt")

# Table 11 from https://arxiv.org/pdf/2411.12021

#Rd_fid = 99.0792  #Mpc/h
class DESIY1BAORSDLikelihood(GaussianLikelihood):

    like_name = "desi_y1_rsd"

    def __init__(self, options):

        iso_file = options.get_string("iso_data_filename", default=ISO_DEFAULT_FILENAME)
        ani_file = options.get_string("ani_data_filename", default=ANI_DEFAULT_FILENAME)

        iso_cols = ['z_eff', 'DVrd', 'DVrd_error', 'fs8', 'fs8_error']
        ani_cols = ['z_eff', 'DMrd', 'DMrd_error', 'DHrd', 'DHrd_error', 'corr', 'fs8', 'fs8_error']

        self.iso_data = {k: v for k,v in zip(iso_cols, np.loadtxt(iso_file, unpack=True))}
        self.ani_data = {k: v for k,v in zip(ani_cols, np.loadtxt(ani_file, unpack=True))}

        self.niso = len(self.iso_data['z_eff'])
        self.nani = len(self.ani_data['z_eff'])

        self.feedback = options.get_bool("feedback", default=False)
        # mode=False: BAO+RSD, mode=True: RSD only
        self.mode = options.get_bool("mode", default=False)
        super().__init__(options)

    def build_data(self):

        if self.mode:
            zeff = np.concatenate([self.iso_data['z_eff'], self.ani_data['z_eff']])
            dist = np.concatenate([self.iso_data['fs8'],self.ani_data['fs8']])

            if self.feedback:
                print('Data distances')
                print('zeff   fs8')
                for z, fs in zip(self.iso_data['z_eff'], self.iso_data['fs8']):
                    print(f'{z:.2f}   {fs:.3f}')
                for z, fs in zip(self.ani_data['z_eff'], self.ani_data['fs8']):
                    print(f'{z:.2f}   {fs:.3f}')
            #import ipdb; ipdb.set_trace()
            #return zeff, dist

        else: 
            zeff = np.concatenate([self.iso_data['z_eff'], self.ani_data['z_eff'], self.ani_data['z_eff'], 
            self.iso_data['z_eff'], self.ani_data['z_eff']])
            dist = np.concatenate([self.iso_data['DVrd'],  self.ani_data['DMrd'],  self.ani_data['DHrd'], 
            self.iso_data['fs8'],self.ani_data['fs8']])
            if self.feedback:
                print('Data distances')
                print('zeff   DV/rd   DM/rd   DH/rd   fs8')
                for z, v, fs in zip(self.iso_data['z_eff'], self.iso_data['DVrd'], self.iso_data['fs8']):
                    print(f'{z:.2f}   {v:.2f}                      {fs:.3f}')
                for z, m, h, fs in zip(self.ani_data['z_eff'], self.ani_data['DMrd'], self.ani_data['DHrd'], self.ani_data['fs8']):
                    print(f'{z:.2f}           {m:.2f}   {h:.2f}    {fs:.3f}')

            #import ipdb; ipdb.set_trace()
        return zeff[:-1], dist[:-1]

    def build_covariance(self):

        if self.mode:
            cov = np.diag(np.concatenate([self.iso_data['fs8_error'],  self.ani_data['fs8_error']]))**2 
        else: 
            cov = np.diag(np.concatenate([self.iso_data['DVrd_error'], self.ani_data['DMrd_error'], self.ani_data['DHrd_error'], 
            self.iso_data['fs8_error'],  self.ani_data['fs8_error']]))**2 

            zipped = zip(self.ani_data['DMrd_error'], self.ani_data['DHrd_error'], self.ani_data['corr'])

            for i, (DMrd_error, DHrd_error, corr) in enumerate(zipped, start=self.niso):
                cov[i,i+self.nani] = DMrd_error * DHrd_error * corr
                cov[i+self.nani,i] = cov[i,i+self.nani]

        #import ipdb; ipdb.set_trace()
        return cov[:-1, :-1]

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
        DV_z = (z * DM_z**2 * DH_z)**(1/3) # in Mpc

        # z and distance maybe are loaded in chronological order
        # Reverse to start from low z
        if (z[1] < z[0]):
            z  = z[::-1]
            DM_z = DM_z[::-1]
            DH_z = DH_z[::-1]

        # Find theory DM and DH at effective redshift by interpolation
        z_eff = self.data_x[:self.niso+self.nani]
        #import ipdb; ipdb.set_trace()

        DMrd = np.interp(z_eff, z, DM_z)/rd
        DHrd = np.interp(z_eff, z, DH_z)/rd
        DVrd = (z_eff * DMrd**2 * DHrd)**(1/3)


        # computed fs8 prediction 
        z_gro = block[names.growth_parameters, 'z']
        if block.has_value(names.growth_parameters, "fsigma_8"):
            fsig = interp(z_eff, z_gro, block[names.growth_parameters, "fsigma_8"])
            
        else:
            #growth parameters
            d_z = block[names.growth_parameters, 'd_z']
            f_z = block[names.growth_parameters, 'f_z']
            sig = block[names.cosmological_parameters, 'sigma_8']
            #omm = block[names.cosmological_parameters, 'omega_m'] 
            try:
                z0 = where(z_gro==0)[0][0]
            except IndexError:
                raise ValueError("You need to calculate f(z) and d(z) down to z=0 to use the BOSS f*sigma8 likelihood")

            # find fsigma8 at effective redshift by interpolation
            fsigma = (sig*(d_z/d_z[z0]))*f_z
            fsig = interp(z_eff, zgro, fsigma)

        #import ipdb; ipdb.set_trace()
        # find definition of fs8
        #fs8_over_fsig8fid = fsig/FS8_fid
        #import ipdb; ipdb.set_trace()
		#if feedback:
		#	print("Growth parameters: z = ",redshift, "fsigma_8  = ",fsig)

        if self.mode:
            if self.feedback:
                print('Theory fsig8')
                print('zeff   fs8')
                for i, (z, fs) in enumerate(zip(z_eff, fsig)):
                    if i < self.niso:
                        print(f'{z:.2f}   {fs:.2f}')
                    else:
                        print(f'{z:.2f}    {fs:.2f}')

            DV_theory = np.concatenate([fsig[:self.niso], fsig[self.niso:]])
            #import ipdb; ipdb.set_trace()

        else: 
            if self.feedback:
                print('Theory distances')
                print('zeff   DV/rd   DM/rd   DH/rd   fs8')
                for i, (z, m, h, v, fs) in enumerate(zip(z_eff, DMrd, DHrd, DVrd, fsig)):
                    if i < self.niso:
                        print(f'{z:.2f}   {v:.2f}                    {fs:.2f}')
                    else:
                        print(f'{z:.2f}           {m:.2f}   {h:.2f}    {fs:.2f}')

            DV_theory = np.concatenate([DVrd[:self.niso], DMrd[self.niso:], DHrd[self.niso:], fsig[:self.niso], fsig[self.niso:]])[:-1]
            #return np.concatenate([DVrd[:self.niso], DMrd[self.niso:], DHrd[self.niso:], fsig[:self.niso], fsig[self.niso:]])[:-1]
        #import ipdb; ipdb.set_trace()
        return DV_theory

setup, execute, cleanup = DESIY1BAORSDLikelihood.build_module()

