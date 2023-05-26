import sys
import os
from cosmosis.datablock import names, option_section
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.interpolate import interp1d
import numpy as np

dirname = os.path.split(__file__)[0]

# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'
    
    
    
from numpy import log, exp, log10 
import sys

#ripped from fastpt for convenience
class k_extend: 

    def __init__(self,k,low=None,high=None):
                
        self.DL=log(k[1])-log(k[0]) 
        
        if low is not None:
            if (low > log10(k[0])):
                low=log10(k[0])
                print('Warning, you selected a extrap_low that is greater than k_min. Therefore no extrapolation will be done.')
                #raise ValueError('Error in P_extend.py. You can not request an extension to low k that is greater than your input k_min.')
        
            low=10**low
            low=log(low)
            N=np.absolute(int((log(k[0])-low)/self.DL))
           
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[0]) -( np.arange(0,N)+1)*self.DL 
            s=s[::-1]
            self.k_min=k[0]
            self.k_low=exp(s) 
           
            self.k=np.append(self.k_low,k)
            self.id_extrap=np.where(self.k >=self.k_min)[0] 
            k=self.k
            

        if high is not None:
            if (high < log10(k[-1])):
                high=log10(k[-1])
                print('Warning, you selected a extrap_high that is less than k_max. Therefore no extrapolation will be done.')
                #raise ValueError('Error in P_extend.py. You can not request an extension to high k that is less than your input k_max.')
            
            high=10**high
            high=log(high)
            N=np.absolute(int((log(k[-1])-high)/self.DL))
            
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[-1]) + (np.arange(0,N)+1)*self.DL 
            self.k_max=k[-1]
            self.k_high=exp(s)
            self.k=np.append(k,self.k_high)
            self.id_extrap=np.where(self.k <= self.k_max)[0] 
            

        if (high is not None) & (low is not None):
            self.id_extrap=np.where((self.k <= self.k_max) & (self.k >=self.k_min))[0]
            
            
    def extrap_k(self):
        return self.k 
        
    def extrap_P_low(self,P):
      
        ns=(log(P[1])-log(P[0]))/self.DL
        Amp=P[0]/self.k_min**ns
        P_low=self.k_low**ns*Amp
        return np.append(P_low,P) 

    def extrap_P_high(self,P):
       
        ns=(log(P[-1])-log(P[-2]))/self.DL
        Amp=P[-1]/self.k_max**ns
        P_high=self.k_high**ns*Amp
        return np.append(P,P_high) 
    
    def PK_original(self,P): 
        return self.k[self.id_extrap], P[self.id_extrap]    



def setup(options):
    input_section = options.get_string(option_section, "input_section", default=names.matter_power_lin)
    output_section = options.get_string(option_section, "output_section", default=names.matter_power_nl)

    # Set up the path to let us import the emulator
    pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
#    install_dir = dirname + f"/ee_install/lib/python{pyversion}/site-packages/"
#    with open(f"{install_dir}/easy-install.pth") as f:
#        pth = f.read().strip()
#        install_dir = install_dir + pth
#    sys.path.insert(0, install_dir)

    # check everything imports
    from aemulus_heft.heft_emu import HEFTEmulator

    emulator = HEFTEmulator()
    #emulator = None
    # euclidemu2.PyEuclidEmulator()

    return [input_section, output_section, emulator]

def _cleft_pk(k, p_lin):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            Note that we require p_lin at the redshift that you wish
            to make predictions for, because in cosmologies with neutrinos
            a constant linear growth rescaling no longer works.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''

    
    cleftobj = RKECLEFT(k, p_lin)
    cleftobj.make_ptable(D=1, kmin=k[0], kmax=k[-1], nk=1000)
    cleftpk = cleftobj.pktable.T

    cleftpk[2, :] /= 2 #(1 d)
    cleftpk[6, :] /= 0.25 # (d2 d2)
    cleftpk[7, :] /= 2 #(1 s)
    cleftpk[8, :] /= 2 #(d s)

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

    return cleftspline, cleftobj

def lpt_spectra(k, pk_m_lin, pk_cb_lin):

    pk_cb_m_lin = np.sqrt(pk_m_lin * pk_cb_lin)
    cleft_m_spline, _ = _cleft_pk(k, pk_m_lin)
    cleft_cb_spline, _ = _cleft_pk(k, pk_cb_lin)
    cleft_cb_m_spline, _ = _cleft_pk(k, pk_cb_m_lin)
    
    pk_m_cleft = cleft_m_spline(k)[1:]
    pk_cb_cleft = cleft_cb_spline(k)[1:]
    pk_cb_m_cleft = cleft_cb_m_spline(k)[1:]
    s_m_map = {1:0, 3: 1, 6: 3, 10: 6}
    s_cb_map = {2: 0, 4: 1, 5: 2, 7: 3, 8: 4, 9: 5, 11: 6, 12: 7, 13: 8, 14: 9}
        
    pk_cleft = np.zeros((15,len(k)))
    for s in np.arange(15):
        if s==0:
            pk_cleft[s, :] = pk_m_cleft[0]
        elif s in [1, 3, 6, 10]:
            pk_cleft[s, :] = pk_cb_m_cleft[s_m_map[s]]
        else:
            pk_cleft[s, :] = pk_cb_cleft[s_cb_map[s]]

    return pk_cleft

def execute(block, config):
#    from aemulus_heft.heft_emu import HEFTEmulator

    # Recover config information
    input_section, output_section, emulator = config
    # (ombh2, omch2, w0, ns, 10^9 As, H0, mnu)
    
    # Get cosmo params from block
    pars = names.cosmological_parameters
    params = [ block[pars, "Omega_b"] * block[pars, "h0"]**2,
               block[pars, "Omega_c"] * block[pars, "h0"]**2,
               block[pars, "w"],
               block[pars, "n_s"],
               10**9*block[pars, "A_s"],
               block[pars, "h0"] * 100,
               block[pars, "mnu"]]
    try:
        bk0 = block.get('heft_matter_bk', 'bkmaL_0')
        bk_alpha = block.get('heft_matter_bk', 'bkmaL_alpha')
    except:
        bk0 = 0
        bk_alpha = 0
        
    r_cut = 2
    z0 = 0.3

    # Get z and k from the NL power section
    z, k, P_m_lin = block.get_grid(names.matter_power_lin, "z", "k_h", "P_k")
    z, k, P_cb_lin = block.get_grid(names.cdm_baryon_power_lin, "z", "k_h", "P_k")
    
    if np.min(np.log10(k))>-3:
        raise ValueError('kmin needs to be < 1e-3 h/Mpc for HEFT')
    
    kt = np.logspace(np.min(np.log10(k)), 1, 100)
    kidx = k.searchsorted(3.99)
    k_heft = k[:kidx]
    
    
    P_m_lin = 10**interp1d(np.log10(k), np.log10(P_m_lin), axis=1, kind='cubic')(np.log10(kt))
    P_cb_lin = 10**interp1d(np.log10(k), np.log10(P_cb_lin), axis=1, kind='cubic')(np.log10(kt))
    
    sigma8z = block[names.growth_parameters, "sigma_8"]
    z_growth = block[names.growth_parameters, "z"]
    
    assert(np.allclose(z,z_growth))
    
    params_heft = params.copy()
    params_heft.append(0)
    
    spec_heft_all_temp = np.zeros((len(z), len(k_heft), 15))
    spec_lpt_all_temp = np.zeros((len(z), len(kt), 15))
    
    spec_heft_all = np.zeros((len(z), len(k), 15))
    spec_lpt_all = np.zeros((len(z), len(k), 15))    

    for i in range(len(z)):
        spec_lpt = lpt_spectra(kt, P_m_lin[i,...], P_cb_lin[i,...])
        params_heft[-1] = sigma8z[i]
        
        try:
            spec_heft = emulator.predict(k_heft, np.array(params_heft), spec_lpt, k_lpt=kt)
            heft_fail = False
        
        #If HEFT fails (because outside emulator range) revert to LPT
        except ValueError as e:
            print(e)
            heft_fail = True
            spec_heft = spec_lpt
        
        spec_heft_all_temp[i,:,:] = spec_heft.T
        spec_lpt_all_temp[i,:,:] = spec_lpt.T
        
    for i in range(15):
        #extrapolate HEFT spectra
        Pij = spec_heft_all_temp[...,i]
        
        if i>2:
            kidx = k_heft < 1
            knl = k_heft[kidx]
            Pij = Pij[:,kidx]
        else:
            knl = np.copy(k_heft)
        
        nz = len(z)
        if (k[0] < knl[0]) or (k[-1] > knl[-1]):
            EK1 = k_extend(knl, np.log10(k[0]), np.log10(k[-1]))
            knl = EK1.extrap_k()
            Pij_temp = np.zeros((nz, len(knl)))
            
            for j in range(nz):
                
                Pij_zp = np.copy(Pij[j])
                Pij_zp = EK1.extrap_P_low(Pij_zp)
                Pij_zp = EK1.extrap_P_high(Pij_zp)
                
                Pij_zm = np.copy(Pij[j])
                Pij_zm = EK1.extrap_P_low(-Pij_zm)
                Pij_zm = EK1.extrap_P_high(Pij_zm)
                
                Pij_temp[j,:] = Pij_zp
                Pij_temp[j,:][Pij_temp[j,:]!=Pij_temp[j,:]] = -Pij_zm[Pij_temp[j,:]!=Pij_temp[j,:]]
        
        spec_heft_all[...,i] = interp1d(np.log(knl), Pij_temp, kind='cubic', axis=1, 
                                        fill_value='extrapolate', bounds_error=False)(np.log(k))  
        
    
    for i in range(15):
        #extrapolate LPT spectra
        Pij = spec_lpt_all_temp[...,i]
        
        knl = np.copy(kt)
        nz = len(z)
        
        if (k[0] < knl[0]) or (k[-1] > knl[-1]):
            EK1 = k_extend(knl, np.log10(k[0]), np.log10(k[-1]))
            knl = EK1.extrap_k()
            Pij_temp = np.zeros((nz, len(knl)))
            
            for j in range(nz):
                
                Pij_zp = np.copy(Pij[j])
                Pij_zp = EK1.extrap_P_low(Pij_zp)
                Pij_zp = EK1.extrap_P_high(Pij_zp)
                Pij_zm = np.copy(Pij[j])
                Pij_zm = EK1.extrap_P_low(-Pij_zm)
                Pij_zm = EK1.extrap_P_high(Pij_zm)
                
                Pij_temp[j,:] = Pij_zp
                Pij_temp[j,:][Pij_temp[j,:]!=Pij_temp[j,:]] = -Pij_zm[Pij_temp[j,:]!=Pij_temp[j,:]]
        
        spec_lpt_all[...,i] = interp1d(np.log(knl), Pij_temp, kind='cubic', axis=1)(np.log(k))           
        
    counterterm = k[np.newaxis,:,np.newaxis]**2 / (1 + (r_cut * k[np.newaxis,:,np.newaxis]**2)) * spec_heft_all[:,:,:3]
    spec_heft_all[:,:,:3] = spec_heft_all[:,:,:3] - bk0 * ((1+z0)/(1 + z[:,np.newaxis,np.newaxis]))**bk_alpha * counterterm
    
    counterterm = k[np.newaxis,:,np.newaxis]**2 / (1 + (r_cut * k[np.newaxis,:,np.newaxis]**2)) * spec_lpt_all[:,:,:3]
    spec_lpt_all[:,:,:3] = spec_lpt_all[:,:,:3] - bk0 * ((1+z0)/(1 + z[:,np.newaxis,np.newaxis]))**bk_alpha * counterterm
        
    if output_section == input_section:
        # save the original grid to a new section, with _dm added
        # and save the new grid back to the input
        block.put_grid(input_section + "_dm", "z", z, "k_h", k, "P_k", P)

        #0th index is nl total matter power spectrum
        if not heft_fail:
            block.replace_grid(input_section, "z", z, "k_h", k, "P_k", spec_heft_all[:,:,0])
        else:
            block.replace_grid(input_section, "z", z, "k_h", k, "P_k", spec_lpt_all[:,:,0])
            
    else:
        # just save the output
        if not heft_fail:
            block.put_grid(output_section, "z", z, "k_h", k, "P_k", spec_heft_all[:,:,0])
        else:
            block.put_grid(output_section, "z", z, "k_h", k, "P_k", spec_lpt_all[:,:,0])
            
    
    for i in range(spec_heft_all.shape[-1]):
        if not heft_fail:
            block.put_grid(f'aemulus_heft_P{i}_nl', "z", z, "k_h", k, "Pij_k", spec_heft_all[...,i])
        else:
            block.put_grid(f'aemulus_heft_P{i}_nl', "z", z, "k_h", k, "Pij_k", spec_lpt_all[...,i])
            
        block.put_grid(f'aemulus_heft_P{i}_1loop', "z", z, "k_h", k, "Pij_k", spec_lpt_all[...,i])

    return 0

if __name__ == '__main__':
    setup(None)
