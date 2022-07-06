import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
#from numpy.fft import fft,ifft,rfft,irfft
from scipy.fft import rfft, irfft#, set_global_backend
from math import pi,e,ceil,floor
import time
import csv
import options_parser
from functools import reduce
#import pyfftw


options=options_parser.OptionsParser().options
print(options)
#RFFT=True if options['rfft']=='True' else False
#HIGH_WEIGHT=float(options['peak_weight'])
#PLOT_PROGRESS=True if options['plot_progress']=='True' else False
#PRINT_PROGRESS=True if options['print_progress']=='True' else False
#UPDATE_FREQUENCY=int(options['update_frequency'])
MAX_ITERS=int(options['max_iters'])
LOSS=options['loss']

FTOL=float(options['ftol'])
MATCH_HIGH_POINTS=True if options['match_high_points']=='True' else False
#CARRY_OVER_PARAMS=True if options['carry_over_params']=='True' else False
#set_global_backend(pyfftw.interfaces.scipy_fft)
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
            self.var_atoms
        except:
            self.var_atoms = [atom for atom in self.params['var_atoms']]

        if not len(self.var_atoms) == 0:
            self.var_atom_index = [self.ResidueInfo.atom_names.index(atom) for atom in self.var_atoms]
            #print(self.var_atom_index)


        try:
            self.var_res
        except:
            self.var_res = [res for res in self.params['var_res']]
        self.target = target

        self.current_param = None
        self.unmixed_ft_atom_models = dict()
        self.ft_atom_models = dict()
        for atom in self.AtomInfo.atom_masses:
            self.unmixed_ft_atom_models[atom] = [np.exp((-2*pi*j/self.N)*(AtomInfo.atom_masses[atom][i]/self.dm)*np.arange(self.N//2+1)) for i in range(len(AtomInfo.atom_masses[atom]))]
            self.ft_atom_models[atom] = None

        self.unmixed_ft_residue_models = []
        if not len(self.var_atoms) == 0:
            self.ft_nonvar_residue_models = []
        #if self.var_atom is
        self.ft_residue_models = []
        for i in range(self.ResidueInfo.num_species):
            res_init = dict()
            for res in self.ResidueInfo.residue_info:
                res_init[res] = None
            self.ft_residue_models.append(res_init)
            self.unmixed_ft_residue_models.append(res_init.copy())
            if not len(self.var_atoms) == 0:
                self.ft_nonvar_residue_models.append(res_init.copy())

        #print(self.unmixed_ft_residue_models)

        self.ft_species_models = []
        self.ft_stick = None
        self.ft_gauss = None
        self.schedule = {'amps': 0, 'm_off': 0, 'gw': 0, 'var_atoms': 0, 'var_res': 0}

        # if type(target[0]) is list:
        self.mode='unbinned'
        if MATCH_HIGH_POINTS:
            self.target_max=max(target[1])
            self.model_scale=1
        self.target_masses=np.array(target[0])
        self.shifted_masses = self.target_masses - self.m_hd
        self.target_intensities=np.array(target[1])
        # else:
        #     if MATCH_HIGH_POINTS:
        #         self.target_max=max(target)
        #         self.model_scale=1
        #     self.mode='binned'
        #     #make target the Array type used:
        #     self.target_intensities=np.array(self.target)
        self.masses = None
        self.residual = 0
        self.totaltime = 0
        self.asdf = 0
        self.asdfasdf = 0
        self.totalmodelgen = 0
        #self.logfile = logfile
        #with open(self.logfile, 'a', newline = None) as log:
        #    log.write(self.BatchInfo.pep_names[self.pep_num] + "\n")

    def Ft_Shift(self, shift):
        #Creates the fourier transform of a delta function, which functions as a shifter
        fourier_array = np.exp((-2*pi*j*shift/(self.N*self.dm))*np.arange(self.N//2+1))
        return fourier_array

    def Convolution(self, ft_spectra,mults,names=None):
        #Convolves fourier spectra with given multiplicities
        # starttime = time.time()
        if names == None:
            names = range(len(mults))

        length = self.N//2 + 1#len(ft_spectra[names[0]])

        conv = np.ones(length, dtype=complex)
        #hold = np.ones(len(ft_spectra[names[0]]), dtype = complex)
        for i in range(len(mults)):
            if mults[i] == 0:
                #self.asdf += time.time() - starttime
                continue
            # self.asdfasdf += 1
            '''blah = mults[i]*[ft_spectra[names[i]]]
            temp = reduce((lambda x,y: x*y), blah)'''#, hold)
            temp = np.ones(length, dtype = complex)
            spec = ft_spectra[names[i]]
            for j in range(mults[i]):
               temp *= spec#ft_spectra[names[i]]
            conv *= temp
            # conv *= ft_spectra[names[i]]**mults[i]'''
        # self.asdf += time.time() - starttime
        return conv

    def LinCombFt(self, spectra,amps):
        #Takes the linear combination of multiple fourier transformed spectra. Returns in the mass domain
        return np.dot(amps, spectra)

    def AtomSpectrum(self):
        #Creates the atom spectra based on the data from the atom model
        atom_masses = self.AtomInfo.atom_masses
        atom_freqs = self.AtomInfo.atom_freqs
        unmixed_ft_atom_models = self.unmixed_ft_atom_models
        ft_atom_models = self.ft_atom_models
        #ft_atom_models = dict()
        #array_size=self.N//2+1 # if RFFT else self.N
        for atom in atom_masses:
            if ft_atom_models[atom] is None or atom in self.var_atoms:
                #fourier_array=np.zeros(array_size,dtype=complex)
#                self.ft_atom_models[atom] = np.dot(atom_freqs[atom], [np.exp((-2*pi*j/self.N)*(atom_masses[atom][i]/self.dm)*np.arange(array_size)) for i in range(len(atom_masses[atom]))])

                # for i in range(len(atom_masses[atom])):
                #     mass,freq=atom_masses[atom][i],atom_freqs[atom][i]
                #     m_idx = mass/self.dm
                #     temp_fourier_array = (-2*pi*j*m_idx/self.N)*np.arange(0,array_size)
                #     fourier_array += freq*np.exp(temp_fourier_array)
                ft_atom_models[atom] = np.dot(atom_freqs[atom], unmixed_ft_atom_models[atom])#fourier_array
        #print(self.unmixed_ft_residue_models)
        self.ft_atom_models = ft_atom_models


    def ResidueSpectrum(self):
        #Creates the residue spectra by convolving atom spectra together
        #Adding together based on the residue fraction for each labeling scheme
        ft_atom_models = self.ft_atom_models
        res = self.ResidueInfo
        ft_residue_models = [{} for i in range(res.num_species)]
        res_info = res.residue_info
        res_freqs = res.res_freqs
        unmixed_ft_residue_models = self.unmixed_ft_residue_models
        ft_nonvar_residue_models = self.ft_nonvar_residue_models


        for residue in res_info:
            for i in range(res.num_species):
                #print(self.unmixed_ft_residue_models[i][residue])
                if ft_nonvar_residue_models[i][residue] is None:
                    temp_copy = res_info[residue][i].copy()
                    names = res.atom_names.copy()
                    for j in self.var_atom_index:
                        temp_copy.pop(j)
                        names.pop(j)
                    ft_nonvar_residue_models[i][residue] = self.Convolution(ft_atom_models,temp_copy, names)
                #print(self.unmixed_ft_residue_models[i][residue])


                if unmixed_ft_residue_models[i][residue] is None or self.schedule['var_atoms'] == 1:
                    # if unmixed_ft_residue_models[i][residue] is None:
                    #     #print("Hi")
                    ft_atom_models['Def'] = ft_nonvar_residue_models[i][residue]
                    names = ['Def']
                    mults = [1]
                    for j in self.var_atom_index:
                        names.append(res.atom_names[j])
                        mults.append(res_info[residue][i][j])
                    # print(names)
                    # print(mults)
                    unmixed_ft_residue_models[i][residue] = self.Convolution(ft_atom_models, mults, names)
                    #unmixed_ft_residue_models[i][residue] = self.Convolution(ft_atom_models, res_info[residue][i], res.atom_names)



                if self.ft_residue_models[i][residue] is None or (residue in self.var_res and self.schedule['var_res'] == 1) or self.schedule['var_atoms'] == 1:
                    ft_residue_models[i][residue] = res_freqs[residue][i]*unmixed_ft_residue_models[i][residue] + (1-res_freqs[residue][i])*unmixed_ft_residue_models[0][residue]
                    self.ft_residue_models[i][residue] = ft_residue_models[i][residue]
        self.unmixed_ft_residue_models = unmixed_ft_residue_models

        # for i in range(res.num_species):
        #     for residue in res_info:
        #         if self.ft_residue_models[i][residue] is None or residue in self.var_res or self.schedule['var_atoms'] == 1:
        #             mults = res_info[residue][i]
        #             ft_residue_models[i][residue] = self.Convolution(ft_atom_models,mults,res.atom_names)
        #             ft_residue_models[i][residue] = res_freqs[residue][i]*ft_residue_models[i][residue] + (1-res_freqs[residue][i])*ft_residue_models[0][residue]
        #             self.ft_residue_models[i][residue] = ft_residue_models[i][residue]

    def Ft_StickSpectrum(self):
        #Creates the total stick spectrum of the peptide. Convolves residues within each labeling scheme and then linearly combines them
        res = self.ResidueInfo
        ft_residue_models = self.ft_residue_models
        res_info = res.residue_info
        mults, syms = self.PeptideInfo
        # print("TEST")
        ft_species_models = [self.Convolution(ft_residue_models[j], mults, syms) for j in range(res.num_species)]
        # for j in range(res.num_species):
        #
        #     ft_species_model = self.Convolution(ft_residue_models[j], mults, syms)
        #     ft_species_models.append(ft_species_model)
        #unindexed = self.LinCombFt(ft_species_models, res.species_amps)
        self.ft_species_models = ft_species_models

        self.ft_stick = self.LinCombFt(ft_species_models, res.species_amps)

    def Ft_Gaussian(self):
        #makes a Gaussian mass array with a given gw
        N = self.N
        sd = self.params['gw']
        dm = self.dm

        temp_array = np.arange(0,N//2)
        half_intensity_array = (1/(sd*(2*pi)**0.5))*np.exp(-0.5*dm**2*temp_array**2/sd**2)
        full_intensity_array = np.concatenate((half_intensity_array,half_intensity_array[::-1]))
        # scale_factor = sum(full_intensity_array)
        # full_intensity_array*=1/scale_factor
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
        #starttime = time.time()
        # with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
            # returnspectrum = scipy.fft.irfft(model)
        #print(model)
        returnspectrum = irfft(model, n = self.N)
        self.totalmodelgen += 1
        #self.totaltime += time.time() - starttime
        return returnspectrum#irfft(model)

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
        #if self.mode=='binned':
        #    plt.scatter(mass_axis,self.target,color='red',s=5)
        #elif self.mode=='unbinned':
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
        # if MATCH_HIGH_POINTS:
        #     model_max = max(model_ys)
        #     self.model_scale = self.target_max/model_max
        #     model_ys *= self.model_scale

        #Saves the spectrum in a tsv
        f=open(model_tsv,'w')
        for i in range(self.N):
            line=str(mass_axis[i])+', '+str(model_ys[i])+'\n'
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
            line = [model_tsv[:-4], pep_name, peptide_sequence, molecular_weight, charge, self.residual, mz,
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

    # def estimate_intensity(self,mass):
    #     #Estimates the intensity of the experimental spectrum between two of the points, in order to compare the spectra
    #     #Linear interpolation
    #
    #     masses = self.masses
    #     intens_diffs = self.intens_diffs
    #     #starttime = time.time()
    #     lower = floor(mass/self.dm)
    #     #upper = ceil(mass/self.dm)
    #     #self.totaltime += time.time() - starttime
    #     # if upper == len(masses):
    #     #     return masses[lower]
    #     # else:
    #     #     return masses[lower] + (mass/self.dm-lower)*(masses[upper]-masses[lower])
    #
    #     return masses[lower] + (mass/self.dm-lower)*intens_diffs[lower]#(masses[upper]-masses[lower])

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


        #start = time.time()
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
        #self.intens_diffs = self.masses[1:] - self.masses[:-1]
        dm = self.dm
        masses = self.masses
        #print(len(masses))
        #if self.mode=='binned':
        #    model_masses = self.MakeModel()
        #elif self.mode == 'unbinned':
        shifted_masses = self.shifted_masses
        #print(shifted_masses)
        #starttime = time.time()
        model_masses = np.interp(shifted_masses, dm*np.arange(len(masses)), masses)
        #model_masses = [self.estimate_intensity(mass) for mass in shifted_masses]
        #self.totaltime += time.time() -starttime
        #self.totaltime += time.time() -starttime
        #model_masses=[masses[(floor(mass/dm))] + (mass/dm - floor(mass/dm))*(intens_diffs[floor(mass/dm)]) for mass in shifted_masses]
        #list(map(lambda mass:self.estimate_intensity(mass),shifted_masses))
        #starttime = time.time()
        model_masses=np.array(model_masses)


        if MATCH_HIGH_POINTS:
            model_max = max(model_masses)
            self.model_scale = self.target_max/model_max
            model_masses *= self.model_scale
            self.masses *= self.model_scale

        out = (model_masses-self.target_intensities)
        self.residual = np.sum(out**2)
        #print('Square Error: '+str(self.residual))
        #self.totaltime += time.time() - starttime
        return out

    def scipy_optimize_ls(self):
        #Runs the least square optimization algorithm with trust region bounds
        params = self.get_params()
        bounds = self.get_bounds()
        #print(self.params)
        scaled_tolerance = FTOL*(1e6/max(self.target_intensities))*(2.5e4/self.N)
        scipy.optimize.least_squares(self.compute_residual,params,bounds=bounds,ftol=scaled_tolerance,max_nfev=MAX_ITERS)#, diff_step = [2]*len(params))
