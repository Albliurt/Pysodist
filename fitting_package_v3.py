import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,rfft,irfft
from math import pi,e,ceil,floor
import time
import csv
import options_parser

options=options_parser.OptionsParser().options
print(options)
RFFT=True if options['rfft']=='True' else False
HIGH_WEIGHT=float(options['peak_weight'])
#PLOT_PROGRESS=True if options['plot_progress']=='True' else False
PRINT_PROGRESS=True if options['print_progress']=='True' else False
UPDATE_FREQUENCY=int(options['update_frequency'])
MAX_ITERS=int(options['max_iters'])
FTOL=float(options['ftol'])
LOSS=options['loss']
MATCH_HIGH_POINTS=False#True if options['match_high_points']=='True' else False
#CARRY_OVER_PARAMS=True if options['auto_save']=='True' else False

j=1j #imaginary unit



class FittingProblem():
    #Overarching class that contains model generation and fitting
    def __init__(self, N, dm, AtomInfo, ResidueInfo, BatchInfo, i, params, m_hd, target):#, logfile):
        self.N = N
        self.dm = dm
        self.m_hd = m_hd

        self.AtomInfo = AtomInfo
        self.ResidueInfo = ResidueInfo
        self.BatchInfo = BatchInfo
        self.PeptideInfo = (self.BatchInfo.batch_mults[i], self.BatchInfo.batch_syms[i])
        self.pep_num = i
        self.params = params
        try:
            self.var_atom
        except:
            self.var_atoms = []
            for atom in self.params['var_atoms']:
                self.var_atoms.append(atom)
        try:
            self.var_res
        except:
            self.var_res = []
            for res in self.params['var_res']:
                self.var_res.append(res)
        self.target = target

        self.current_param = None
        self.ft_atom_models = dict()
        for atom in self.AtomInfo.atom_masses:
            self.ft_atom_models[atom] = None

        self.ft_residue_models = []
        for i in range(self.ResidueInfo.num_species):
            res_init = dict()
            for res in self.ResidueInfo.residue_info:
                res_init[res] = None
            self.ft_residue_models.append(res_init)


        self.ft_species_models = []
        self.ft_stick = None
        self.ft_gauss = None
        self.schedule = {'amps': 0, 'm_off': 0, 'gw': 0, 'var_atoms': 0, 'var_res': 0}

        if type(target[0]) is list:
            self.mode='unbinned'
            if MATCH_HIGH_POINTS:
                self.target_max=max(target[1])
                self.model_scale=1
            self.target_masses=np.array(target[0])
            self.target_intensities=np.array(target[1])
        else:
            if MATCH_HIGH_POINTS:
                self.target_max=max(target)
                self.model_scale=1
            self.mode='binned'
            #make target the Array type used:
            self.target_intensities=np.array(self.target)
        self.masses = None
        self.residual = 0

        #self.logfile = logfile
        #with open(self.logfile, 'a', newline = None) as log:
        #    log.write(self.BatchInfo.pep_names[self.pep_num] + "\n")

    def Ft_Shift(self, shift):
        #Creates the fourier transform of a delta function, which functions as a shifter
        array_size=self.N//2+1 if RFFT else self.N
        m_idx=shift/self.dm
        temp_array = np.arange(0,array_size)
        fourier_array = np.exp((-2*pi*j*m_idx/self.N)*temp_array)
        return fourier_array

    def Convolution(self, ft_spectra,mults,names=None):
        #Convolves fourier spectra with given multiplicities
        if names == None:
            names = list(range(len(mults)))
        conv = np.ones(len(ft_spectra[names[0]]), dtype=complex)
        for i in range(len(mults)):
            conv = np.multiply(conv, np.power(ft_spectra[names[i]],mults[i]))
        return conv

    def LinCombFt(self, spectra,amps):
        #Takes the linear combination of multiple fourier transformed spectra. Returns in the mass domain
        comb = np.zeros(len(irfft(spectra[0])))
        for i in range(len(spectra)):
            comb += amps[i]*irfft(spectra[i])
        return comb

    def AtomSpectrum(self):
        #Creates the atom spectra based on the data from the atom model
        atom_masses = self.AtomInfo.atom_masses
        atom_freqs = self.AtomInfo.atom_freqs
        ft_atom_models = dict()
        array_size=self.N//2+1 if RFFT else self.N
        for atom in atom_masses:
            if self.ft_atom_models[atom] is None or atom in self.var_atoms:
                fourier_array=np.zeros(array_size,dtype=complex)
                for i in range(len(atom_masses[atom])):
                    mass,freq=atom_masses[atom][i],atom_freqs[atom][i]
                    m_idx = mass/self.dm
                    temp_fourier_array = (-2*pi*j*m_idx/self.N)*np.arange(0,array_size)
                    fourier_array += freq*np.exp(temp_fourier_array)
                self.ft_atom_models[atom] = fourier_array

    def ResidueSpectrum(self):
        #Creates the residue spectra by convolving atom spectra together
        #Adding together based on the residue fraction for each labeling scheme
        ft_atom_models = self.ft_atom_models
        res = self.ResidueInfo
        ft_residue_models = []
        res_info = res.residue_info
        res_freqs = res.res_freqs

        for i in range(res.num_species):
            ft_residue_models.append(dict())
            for residue in res_info:
                if self.ft_residue_models[i][residue] is None or residue in self.var_res or self.schedule['var_atoms'] == 1:
                    mults = res_info[residue][i]
                    ft_residue_models[i][residue] = self.Convolution(ft_atom_models,mults,res.atom_names)
                    ft_residue_models[i][residue] = res_freqs[residue][i]*ft_residue_models[i][residue] + (1-res_freqs[residue][i])*ft_residue_models[0][residue]
                    self.ft_residue_models[i][residue] = ft_residue_models[i][residue]

    def Ft_StickSpectrum(self):
        #Creates the total stick spectrum of the peptide. Convolves residues within each labeling scheme and then linearly combines them
        res = self.ResidueInfo
        ft_residue_models = self.ft_residue_models
        res_info = res.residue_info
        ft_species_models = []
        for j in range(res.num_species):
            mults, syms = self.PeptideInfo
            ft_species_model = self.Convolution(ft_residue_models[j], mults, syms)
            ft_species_models.append(ft_species_model)
        unindexed = self.LinCombFt(ft_species_models, res.species_amps)
        self.ft_species_models = ft_species_models
        self.ft_stick = rfft(unindexed)

    def Ft_Gaussian(self):
        #makes a Gaussian mass array with a given gw
        N = self.N
        sd = self.params['gw']
        dm = self.dm

        temp_array = np.arange(0,N//2)
        temp_array2 = np.power(dm*temp_array/sd,2)
        half_intensity_array = (1/(sd*(2*pi)**0.5))*np.exp(-0.5*temp_array2)
        full_intensity_array = np.concatenate((half_intensity_array,half_intensity_array[::-1]))
        scale_factor = sum(full_intensity_array)
        full_intensity_array*=1/scale_factor
        self.ft_gauss = rfft(full_intensity_array)

    def MakeModel(self):
        #Calculates the new spectrum/model based on the parameters. Determines what parts need to be recomputed based on the fitting schedule
        if self.schedule['var_atoms'] == 1 or self.ft_stick is None:
            self.AtomSpectrum()

        if self.schedule['var_atoms'] == 1 or self.ft_stick is None or self.schedule['var_res'] == 1:
            self.ResidueSpectrum()
            self.Ft_StickSpectrum()
        elif self.schedule['amps'] == 1:
            self.Ft_StickSpectrum()

        if self.schedule['gw'] == 1 or self.ft_gauss is None:
            self.Ft_Gaussian()

        stick = self.ft_stick
        gauss = self.ft_gauss

        shift = self.Ft_Shift(-self.m_hd)
        m_off_shift = self.Ft_Shift(-self.params['m_off'])

        model = self.Convolution([stick,gauss,shift, m_off_shift],[1,1,1,1])
        return irfft(model)

    def plot(self):
        #Plots the fit
        mass_axis=np.arange(0,self.N*self.dm,self.dm)
        mass_axis=mass_axis[:self.N] #sometimes a bit too long, rounding issues?
        if self.m_hd is not None:
            mass_axis+=self.m_hd
        model_ys = self.masses
        if MATCH_HIGH_POINTS:
            model_max = max(model_ys)
            self.model_scale = self.target_max/model_max
            model_ys *= self.model_scale
        if self.mode=='binned':
            plt.scatter(mass_axis,self.target,color='red',s=5)
        elif self.mode=='unbinned':
            plt.scatter(self.target_masses,self.target_intensities,color='red',s=5)
        plt.plot(mass_axis,model_ys)
        plt.xlabel('mass')
        plt.ylabel('intensity')
        plt.show()

    def calc_mw(self):
        #Calculates the peptide's molecular weight
        atom_masses = self.AtomInfo.atom_masses
        res = self.ResidueInfo
        res_mults, res_syms = self.PeptideInfo
        mw = 0
        for i in range(len(res_syms)):
            atom_mult = res.residue_info[res_syms[i]][0]
            for j in range(len(atom_mult)):
                mw += atom_mult[j]*atom_masses[res.atom_names[j]][0]*res_mults[i]
        return mw


    def save_fit(self,params_file, model_tsv, vert_shift=0,charge=1):
        #Saves the calculated spectrum into a tsv
        #Also outputs the final parameters into a csv
        mass_axis=np.arange(0,self.N*self.dm,self.dm)
        mass_axis=mass_axis[:self.N]
        if self.m_hd is not None:
            mass_axis+=self.m_hd
        model_ys=self.masses
        if MATCH_HIGH_POINTS:
            model_max = max(model_ys)
            self.model_scale = self.target_max/model_max
            model_ys *= self.model_scale

        #Saves the spectrum in a tsv
        f=open(model_tsv,'w')
        for i in range(self.N):
            line=str(mass_axis[i]/charge)+', '+str(model_ys[i])+'\n'
            f.write(line)
        f.close()

        peptide_sequence = self.BatchInfo.pep_names[self.pep_num]
        pep_name = peptide_sequence[1:3]
        charge = self.BatchInfo.charges[self.pep_num]
        molecular_weight = self.calc_mw()
        mz = molecular_weight/charge

        #Saves the parameters in a csv
        with open(params_file, mode='a', newline=None) as f:
            param_writer = csv.writer(f, lineterminator = '\n')
            line = [model_tsv, pep_name, peptide_sequence, molecular_weight, charge, self.residual, mz,
                vert_shift,self.params['m_off'],self.params['gw']]

            line += self.params['amps']
            line += self.params['var_atoms'].values()
            line += self.params['var_res'].values()
            param_writer.writerow(line)
            #print("Saved")

    def loground(self, roundnumber, header = False):
        #with open(self.logfile, mode = 'a', newline = None) as log:
        if header:
            line = "ROUND " + str(roundnumber) + "\tCHI_SQUARE"
            if self.schedule['m_off'] == 1:
                line += "\tFREE M_OFF"
            else:
                line += "\tFIX M_OFF"

            if self.schedule['gw'] == 1:
                line += "\tFREE GW"
            else:
                line += "\tFIX GW"

            if self.schedule['amps'] == 1:
                for species_name in self.ResidueInfo.species_names:
                    line += "\tFREE AMP_"+species_name
            else:
                for species_name in self.ResidueInfo.species_names:
                    line += "\tFIX AMP_"+species_name

            if self.schedule['var_atoms'] == 1:
                for var_atom in self.params['var_atoms']:
                    line += "\tFREE FRC_"+var_atom
            else:
                for var_atom in self.params['var_atoms']:
                    line += "\tFIX FRC_"+var_atom

            if self.schedule['var_res'] == 1:
                for var_res in self.params['var_res']:
                    line += "\tFREE FRC_"+var_res
            else:
                for var_res in self.params['var_res']:
                    line += "\tFIX FRC_"+var_res
            line += "\n"
            print(line)

        line = "\t"+str("{:.4e}".format(self.residual)) +"\t" +"{:.4f}".format(self.params['m_off']) + "\t" + "{:.4f}".format(self.params['gw'])

        normalized_amps = self.params['amps']/sum(self.params['amps'])
        for amp in normalized_amps:
            line += "\t"+"{:.4f}".format(amp)
        for var_atom_value in self.params['var_atoms'].values():
            line += "\t"+"{:.4f}".format(var_atom_value)
        for var_res_value in self.params['var_res'].values():
            line += "\t" + "{:.4f}".format(var_res_value)
        line += "\n"
        print(line)
        #log.write(line)

    def estimate_intensity(self,mass):
        #Estimates the intensity of the experimental spectrum between two of the points, in order to compare the spectra
        #Linear interpolation
        masses = self.masses
        lower = floor(mass/self.dm)
        upper = ceil(mass/self.dm)
        if upper == len(masses):
            return masses[lower]
        if lower == upper:
            return masses[lower]
        else:
            return masses[lower] + (mass/self.dm-lower)*(masses[upper]-masses[lower])

    def get_params(self):
        #Returns a parameter vector containing the relevant parameters that are being fitted
        params = []
        if self.schedule['m_off'] == 1:
            params.append(self.params['m_off'])
        if self.schedule['gw'] == 1:
            params.append(self.params['gw'])
        if self.schedule['amps'] == 1:
            params = params + list(self.params['amps'])
        if self.schedule['var_atoms']==1:
            for atom in self.var_atoms:
                params.append(self.params['var_atoms'][atom])
        if self.schedule['var_res']==1:
            for res in self.var_res:
                params.append(self.params['var_res'][res])
        return params

    def testParams(self, params):
        #Testing method that constructs a spectrum given input parameters, and does not fit
        self.schedule['var_res'] = 1
        self.params['m_off'] = params['m_off']
        self.params['gw'] = params['gw']
        amp_length = len(self.ResidueInfo.species_amps)
        self.params['amps'] = params['amps']
        self.ResidueInfo.species_amps = params['amps']
        for i in range(len(self.var_atoms)):# atom in vector[3]:
            atom = self.var_atoms[i]
            self.AtomInfo.atom_freqs[atom][1] = params['var_atom'][atom]
            self.AtomInfo.atom_freqs[atom][0] = 1-params['var_atom'][atom]
            self.params['var_atoms'][atom] = params['var_atom'][atom]
        for i in range(len(self.var_res)):
            residue = self.var_res[i]
            self.ResidueInfo.res_freqs[residue][1:] = [params['var_res'][residue]]*(len(self.ResidueInfo.res_freqs[residue])-1)
            self.params['var_res'][residue] = params['var_res'][residue]
        self.masses = self.MakeModel()
        self.current_param = 'final'
        self.residual = sum(self.compute_residual()**2)


    def fitschedule(self):
        #Schedules the parameters to be fit in each round


        start = time.time()
        roundnumber = 1
        #print("Round " + str(roundnumber) + ": Amplitudes")
        # Fitting amplitudes preliminarily
        self.schedule['amps'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        #print("Round " + str(roundnumber) +": M_Off")
        # Fitting m_off
        self.schedule['amps'] = 0
        self.schedule['m_off'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        #print("Round " + str(roundnumber) + ": M_Off + Gaussian Width")
        # Fitting gaussian width with m_off
        self.schedule['gw'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        # Fitting variable atoms
        if len(self.params['var_atoms']) > 0:
            #print("Round " + str(roundnumber) + ": Variable Atoms")
            #Other parameters are not fit during this. Fit appears to work without them, though it is not what isodist does
            self.schedule['gw'] = 0
            self.schedule['m_off'] = 0
            self.schedule['amps'] = 0
            self.schedule['var_atoms'] = 1
            self.scipy_optimize_ls()
            self.loground(roundnumber, True)
            roundnumber += 1

        self.schedule['var_atoms'] = 0

        # Fitting variable residues (SILAC)
        if len(self.params['var_res']) > 0:
            #print("Round " + str(roundnumber) + ": Variable Residues")
            self.schedule['gw'] = 0
            self.schedule['m_off'] = 0
            self.schedule['amps'] = 0
            self.schedule['var_res'] = 1
            self.scipy_optimize_ls()
            self.loground(roundnumber, True)
            roundnumber += 1

        #print("Round " + str(roundnumber) + ": All Parameters")
        # Fitting everything together
        self.schedule['amps'] = 1
        self.schedule['gw'] = 1
        self.schedule['m_off'] = 1
        if len(self.params['var_atoms']) > 0:
            self.schedule['var_atoms'] = 1
        if len(self.params['var_res']) > 0:
            self.schedule['var_res'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        #print("Time to fit: " + str(time.time()-start))

        #Normalize the amplitudes
        amp_sum = sum(self.params['amps'])
        self.params['amps'] = [x/amp_sum  for x in self.params['amps']]

    def set_params(self,vector):
        #Sets the parameters given the parameter vector
        if self.schedule['m_off'] == 1:
            self.params['m_off'] = vector[0]
            vector = vector[1:]
        if self.schedule['gw'] == 1:
            self.params['gw'] = vector[0]
            vector = vector[1:]
        if self.schedule['amps'] == 1:
            amp_length = len(self.ResidueInfo.species_amps)
            self.params['amps'] = vector[0:amp_length]
            self.ResidueInfo.species_amps = vector[0:amp_length]
            if len(vector) > amp_length:
                vector = vector[amp_length:]
        if self.schedule['var_atoms'] == 1:
            for i in range(len(self.var_atoms)):# atom in vector[3]:
                atom = self.var_atoms[i]
                self.AtomInfo.atom_freqs[atom][1] = vector[i]
                self.AtomInfo.atom_freqs[atom][0] = 1-vector[i]
                self.params['var_atoms'][atom] = vector[i]

            if len(vector) > len(self.var_atoms):
                vector = vector[len(self.var_atoms):]

        if self.schedule['var_res'] == 1:
            for i in range(len(self.var_res)):
                residue = self.var_res[i]
                self.ResidueInfo.res_freqs[residue][1:] = [vector[i]]*(len(self.ResidueInfo.res_freqs[residue])-1)
                self.params['var_res'][residue] = vector[i]

            if len(vector) > len(self.var_res):
                vector = vector[len(self.var_res):]


    def get_bounds(self):#for scipy.optimize.least_squares
        #Compute the bounds vector for each parameter that is being fit
        bound_dict={'gw':[0,.1],'var_atoms':[0,1],'amps':[0,np.inf],'m_off':[-0.1,0.1],'var_res':[0,1]} #maps parameter dist type to bounds
        lowers=[]
        uppers=[]

        if self.schedule['m_off'] == 1:
            lowers.append(bound_dict['m_off'][0])
            uppers.append(bound_dict['m_off'][1])
        if self.schedule['gw'] == 1:
            lowers.append(bound_dict['gw'][0])
            uppers.append(bound_dict['gw'][1])
        if self.schedule['amps'] == 1:
            amp_length = len(self.ResidueInfo.species_amps)
            lowers = lowers + [bound_dict['amps'][0]]*amp_length
            uppers = uppers + [bound_dict['amps'][1]]*amp_length
        if self.schedule['var_atoms'] == 1:
            num_vars = len(self.params['var_atoms'])
            lowers = lowers + [bound_dict['var_atoms'][0]]*num_vars
            uppers = uppers + [bound_dict['var_atoms'][1]]*num_vars
        if self.schedule['var_res'] == 1:
            num_vars = len(self.params['var_res'])
            lowers = lowers + [bound_dict['var_res'][0]]*num_vars
            uppers = uppers + [bound_dict['var_res'][1]]*num_vars

        return (lowers,uppers)


    def compute_residual(self, param_vector=None):
        #Computes the chi square residual, as well as the residual vector
        if not self.current_param == 'final':
            self.set_params(param_vector)
        self.masses = self.MakeModel()

        if self.mode=='binned':
            model_masses = self.MakeModel()
        elif self.mode == 'unbinned':
            model_masses=list(map(lambda mass:self.estimate_intensity(mass-self.m_hd),self.target_masses))
            model_masses=np.array(model_masses)

        if MATCH_HIGH_POINTS:
            model_max = max(model_masses)
            self.model_scale = self.target_max/model_max
            model_masses *= self.model_scale
        out = (model_masses-self.target_intensities)
        self.residual = sum(out**2)
        #print('Square Error: '+str(self.residual))
        return out

    def scipy_optimize_ls(self):
        #Runs the least square optimization algorithm with trust region bounds
        params = self.get_params()
        bounds = self.get_bounds()
        #print(self.params)
        scaled_tolerance = FTOL*(1e6/max(self.target_intensities))*(2.5e4/self.N)
        scipy.optimize.least_squares(self.compute_residual,params,bounds=bounds,ftol=scaled_tolerance,max_nfev=MAX_ITERS)
