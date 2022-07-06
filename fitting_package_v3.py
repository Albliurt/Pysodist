import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
#from numpy.fft import fft,ifft,rfft,irfft
from numpy.fft import rfft, irfft#, set_global_backend
from math import pi,e,ceil,floor
import time
import csv
import options_parser
from functools import reduce


options=options_parser.OptionsParser().options
print(options)
MAX_ITERS=int(options['max_iters'])
LOSS=options['loss']

FTOL=float(options['ftol'])
MATCH_HIGH_POINTS=True if options['match_high_points']=='True' else False
j=1j #imaginary unit



class FittingProblem():
    def __init__(self, N, dm, AtomInfo, ResidueInfo, BatchInfo, i, params, m_hd, target):#, logfile):
        """
        Overarching class that contains model generation and fitting
        :param N: number of data points in the spectrum to be generated. Determined by pysodist - if auto_N, then it is the mass range divided by dm
        :param dm: distance between consecutive points. Resolution of the spectra
        :param AtomInfo: AtomInfo object that contains the information in the atom definitions
        :param ResidueInfo: ResidueInfo object that contains the model file for the particular labeling scheme, as well as the elemental composition of each residue
        :param BatchInfo: BatchInfo object that contains the peptide sequences, their spectral file, and charge
        :param i: index of peptide to be fit within BatchInfo
        :param m_hd: heterodyne shift
        :param target: target spectrum to be fit to
        """

        self.N = N
        self.dm = dm
        self.m_hd = m_hd

        self.AtomInfo = AtomInfo
        self.ResidueInfo = ResidueInfo
        self.BatchInfo = BatchInfo
        self.PeptideInfo = (self.BatchInfo.batch_mults[i], self.BatchInfo.batch_syms[i])
        self.pep_num = i
        self.params = params

        self.var_atoms = [atom for atom in self.params['var_atoms']]
        #Stores the indices of the variable atoms in the atom arrays, in order to reduce convolutions needed later
        if not len(self.var_atoms) == 0:
            self.var_atom_index = [self.ResidueInfo.atom_names.index(atom) for atom in self.var_atoms]
        self.var_res = [res for res in self.params['var_res']]

        self.target = target
        self.current_param = None

        #Defines separately the "unmixed" atom models which are the fourier transform of delta functions
        #Since only the atom frequencies change, these unmixed models are added at given frequencies
        self.unmixed_ft_atom_models = dict()
        self.ft_atom_models = dict()
        for atom in self.AtomInfo.atom_masses:
            self.unmixed_ft_atom_models[atom] = [np.exp((-2*pi*j/self.N)*(AtomInfo.atom_masses[atom][i]/self.dm)*np.arange(self.N//2+1)) for i in range(len(AtomInfo.atom_masses[atom]))]
            self.ft_atom_models[atom] = None

        #For SILAC experiments, variable residues can be computed by computing separate labeling schemes and adding them
        #unmixed_ft_residue_models describes these
        self.unmixed_ft_residue_models = []
        self.ft_residue_models = []

        #To reduce convolution time, ft_nonvar_residue_models contains residue spectra except for the variable atom, which is convolved later
        #These should remain constant after initialization
        #They are lists of dictionaries. n dictionaries for n species, with keys as residues
        if not len(self.var_atoms) == 0:
            self.ft_nonvar_residue_models = []
        for i in range(self.ResidueInfo.num_species):
            res_init = dict()
            for res in self.ResidueInfo.residue_info:
                res_init[res] = None
            self.ft_residue_models.append(res_init)
            self.unmixed_ft_residue_models.append(res_init.copy())
            if not len(self.var_atoms) == 0:
                self.ft_nonvar_residue_models.append(res_init.copy())


        self.ft_species_models = []
        self.ft_stick = None
        self.ft_gauss = None
        self.schedule = {'amps': 0, 'm_off': 0, 'gw': 0, 'var_atoms': 0, 'var_res': 0}

        self.mode='unbinned'
        #Optional scaling for some peptides. If the highest point in the target spectrum is the desired peak, then this scales the generated spectrum to the target
        if MATCH_HIGH_POINTS:
            self.target_max=max(target[1])
            self.model_scale=1
        self.target_masses=np.array(target[0])
        self.shifted_masses = self.target_masses - self.m_hd
        self.target_intensities=np.array(target[1])

        self.masses = None
        self.residuals = None

        self.timing = 0
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        self.time4 = 0
        self.time_m = 0
        self.time_f = 0
        self.a = 0
        self.count = 0

    def Ft_Shift(self, shift):
        '''Generates the Fourier transform of a delta functions, which shifts a spectrum by the shift amount
        :param shift: by how much mass to shift the spectrum

        returns: Fourier transformed delta function
        '''
        s = time.time()
        #mass_array = (-2*pi*j*shift/(self.N*self.dm))*np.arange(self.N//2+1)
        #self.time_m += time.time() - s

        fourier_array = np.exp((-2*pi*j*shift/(self.N*self.dm))*np.arange(self.N//2+1))
        # fourier_array = np.exp(1j*np.arange(self.N//2+1))
        # fourier_array = np.power(fourier_array, (-2*pi*shift/(self.N*self.dm)))
        self.time_f += time.time() - s
        return fourier_array

    def Convolution(self, ft_spectra,mults,names=None):
        '''Convolves Fourier spectra with given multiplicities. Raises each Fourier spectrum to corresponding multiplicity, and then multiplies them
        :param ft_spectra: list or dictionary of spectra
        :param mults: list of multiplicities for each spectrum. Indexed with names if ft_spectra is a dictionary
        :param names: optional parameter if ft_spectra is a dictionary. Provides the keys for the dictionary

        returns: convolved spectrum in the Fourier domain
        '''
        if names == None:
            names = range(len(mults))

        length = self.N//2 + 1
        conv = np.ones(length, dtype=complex)
        for i in range(len(mults)):
            if mults[i] == 0:
                continue
            elif mults[i] == 1:
                conv *= ft_spectra[names[i]]
            else:
                conv *= ft_spectra[names[i]]**mults[i]

            # temp = np.ones(length, dtype = complex)
            # spec = ft_spectra[names[i]]
            # for j in range(mults[i]):
            #    temp *= spec#ft_spectra[names[i]]
            # conv *= temp

        return conv

    def LinCombFt(self, spectra,amps):
        '''Takes the linear combination of multiple Fourier transformed spectra with given weights (amps). Returns in the Fourier domain
        :param spectra: array of spectra to be linearly combined
        :param amps: amplitudes of corresponding spectra

        returns: linearly combined spectrum in the Fourier domain, containing the sum of amps[i]*spectra[i]
        '''
        return np.dot(amps, spectra)

    def AtomSpectrum(self):
        '''Creates the atom spectra based on the data from the atom model file and atomic frequencies. Utilizes the unmixed model and saves the
        generated models in the instance variable ft_atom_models

        returns: None
        '''
        atom_masses = self.AtomInfo.atom_masses
        atom_freqs = self.AtomInfo.atom_freqs
        unmixed_ft_atom_models = self.unmixed_ft_atom_models
        ft_atom_models = self.ft_atom_models

        for atom in atom_masses:
            if ft_atom_models[atom] is None or atom in self.var_atoms:
                ft_atom_models[atom] = np.dot(atom_freqs[atom], unmixed_ft_atom_models[atom])#fourier_array
        self.ft_atom_models = ft_atom_models


    def ResidueSpectrum(self):
        '''Creates the residue spectra by convolving atomic spectra together. Differently labeled residues are then added together based on
        specified fractions based on the particular labeling scheme (SILAC). Saved in the instance variable ft_residue_models. Additionally,
        initializes/modified unmixed_ft_residue_models and ft_nonvar_residue_models

        returns: None'''
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
                    popped = 0
                    for k in self.var_atom_index:
                        temp_copy.pop(k-popped)
                        names.pop(k-popped)
                        popped += 1
                    ft_nonvar_residue_models[i][residue] = self.Convolution(ft_atom_models,temp_copy, names)

                if unmixed_ft_residue_models[i][residue] is None or self.schedule['var_atoms'] == 1:
                    ft_atom_models['Def'] = ft_nonvar_residue_models[i][residue]
                    names = ['Def']
                    mults = [1]
                    for k in self.var_atom_index:
                        names.append(res.atom_names[k])
                        mults.append(res_info[residue][i][k])
                    unmixed_ft_residue_models[i][residue] = self.Convolution(ft_atom_models, mults, names)

                if self.ft_residue_models[i][residue] is None or (residue in self.var_res and self.schedule['var_res'] == 1 and not i == 0) or self.schedule['var_atoms'] == 1:
                    starttime = time.time()
                    ft_residue_models[i][residue] = self.LinCombFt([unmixed_ft_residue_models[i][residue], unmixed_ft_residue_models[0][residue]],[res_freqs[residue][i], (1-res_freqs[residue][i])])
                    #ft_residue_models[i][residue] = res_freqs[residue][i]*unmixed_ft_residue_models[i][residue] + (1-res_freqs[residue][i])*unmixed_ft_residue_models[0][residue]
                    self.a += time.time()-starttime
                    self.ft_residue_models[i][residue] = ft_residue_models[i][residue]
                    # self.count += 1
                    # print(self.count)
                # self.a += time.time()-starttime
        self.unmixed_ft_residue_models = unmixed_ft_residue_models

    def Ft_StickSpectrum(self):
        '''Creates the total stick spectrum of the peptide. Convolves residues within each labeling scheme based on the peptide sequence
        and then linearly combines them based on the fitting parameters 'amps'

        returns: None
        '''
        res = self.ResidueInfo
        ft_residue_models = self.ft_residue_models
        res_info = res.residue_info
        mults, syms = self.PeptideInfo
        ft_species_models = [self.Convolution(ft_residue_models[k], mults, syms) for k in range(res.num_species)]
        self.ft_species_models = ft_species_models

        self.ft_stick = self.LinCombFt(ft_species_models, res.species_amps)

    def Ft_Gaussian(self):
        '''Makes a Gaussian mass array with a given gw, specified by the fitting parameter 'gw'

        returns: None'''
        N = self.N
        sd = self.params['gw']
        dm = self.dm
        temp_array = np.arange(0,N//2)
        half_intensity_array = (1/(sd*(2*pi)**0.5))*np.exp(-0.5*(dm/sd*temp_array)**2)
        full_intensity_array = np.concatenate((half_intensity_array,half_intensity_array[::-1]))
        self.ft_gauss = rfft(full_intensity_array)

    def MakeModel(self):
        '''Overarching method to construct the model. Determines what needs to be recomputed (e.g. atomic spectra do not have to be recomputed when gw is varied).
        Computes a stick spectrum, Gaussian, and necessary shifts, and then convolves them.

        returns: mass domain theoretical spectrum'''
        starttime = time.time()
        if self.schedule['var_atoms'] == 1 or self.ft_stick is None:
            self.AtomSpectrum()

        temptime = time.time()
        self.time1 += temptime - starttime
        if self.schedule['var_atoms'] == 1 or self.ft_stick is None or self.schedule['var_res'] == 1:
            self.ResidueSpectrum()
            self.time2 += time.time() - temptime
            temptime = time.time()

            self.Ft_StickSpectrum()
            self.time3 += time.time() - temptime
            temptime = time.time()
        elif self.schedule['amps'] == 1:
            self.Ft_StickSpectrum()
            self.time3 += time.time() - temptime
            temptime = time.time()
        if self.schedule['gw'] == 1 or self.ft_gauss is None:
            self.Ft_Gaussian()

        stick = self.ft_stick
        gauss = self.ft_gauss

        self.time4 += time.time()-temptime

        shift = self.Ft_Shift(-self.m_hd)
        m_off_shift = self.Ft_Shift(-self.params['m_off'])

        model = self.Convolution([stick,gauss,shift, m_off_shift],[1,1,1,1])
        returnspectrum = irfft(model, n = self.N)
        self.timing += time.time() - starttime
        # self.timing += time.time() - starttime

        return returnspectrum

    def plot(self):
        '''Plots the fit versus the target spectrum

        returns: None'''
        mass_axis=np.arange(0,self.N*self.dm,self.dm)
        mass_axis=mass_axis[:self.N] #sometimes a bit too long, rounding issues?
        if self.m_hd is not None:
            mass_axis+=self.m_hd
        model_ys = self.masses
        if MATCH_HIGH_POINTS:
            model_max = max(model_ys)
            self.model_scale = self.target_max/model_max
            model_ys *= self.model_scale
        plt.scatter(self.target_masses,self.target_intensities,color='red',s=5)
        plt.plot(mass_axis,model_ys)
        plt.xlabel('mass')
        plt.ylabel('intensity')
        plt.show()

    def calc_mw(self):
        '''Calculates the peptide's monoisotopic molecular weight. Used for saving fit into csv.

        returns: monoisotopic molecular_weight
        '''
        atom_masses = self.AtomInfo.atom_masses
        res = self.ResidueInfo
        res_mults, res_syms = self.PeptideInfo
        mw = 0
        for i in range(len(res_syms)):
            atom_mult = res.residue_info[res_syms[i]][0]
            for k in range(len(atom_mult)):
                mw += atom_mult[k]*atom_masses[res.atom_names[k]][0]*res_mults[i]
        return mw


    def save_fit(self,params_file, model_tsv, vert_shift=0,charge=1):
        '''Outputs the generated fit spectrum. Outputs final parameters into an output csv
        :param params_file: csv file to output fit parameters and peptide information into
        :param model_tsv: location to save the spectrum datapoints to
        :param vert_shift: optional baseline shift of the peptide
        :param charge: charge of the peptide

        returns: None'''
        mass_axis=np.arange(0,self.N*self.dm,self.dm)
        mass_axis=mass_axis[:self.N]
        if self.m_hd is not None:
            mass_axis+=self.m_hd
        model_ys=self.masses

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
            chi_sq = np.sum(self.residuals**2)
            line = [model_tsv[:-4], pep_name, peptide_sequence, molecular_weight, charge, chi_sq, mz,
                vert_shift,self.params['m_off'],self.params['gw']]

            line.extend(self.params['amps'])
            line.extend(self.params['var_atoms'].values())
            line.extend(self.params['var_res'].values())
            param_writer.writerow(line)

    def loground(self, roundnumber, header = False):
        '''Logs the fitting round's results into a designated logfile. Prints parameter values and chi square values.
        Logging is handled by run_isodist. Print statement is siphoned into the logfile.

        :param roundnumber: which round/step of fitting was just concluded.

        returns: None
        '''
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

        line = "\t"+str("{:.4e}".format(np.sum(self.residuals**2))) +"\t" +"{:.4f}".format(self.params['m_off']) + "\t" + "{:.4f}".format(self.params['gw'])

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

    def get_params(self):
        '''Generates a list of current parameter values based on the current fitting round.

        returns: array of parameter values'''
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

    # def testParams(self, params):
    #     #Testing method that constructs a spectrum given input parameters, and does not fit
    #     self.schedule['var_res'] = 1
    #     self.params['m_off'] = params['m_off']
    #     self.params['gw'] = params['gw']
    #     amp_length = len(self.ResidueInfo.species_amps)
    #     self.params['amps'] = params['amps']
    #     self.ResidueInfo.species_amps = params['amps']
    #     for i in range(len(self.var_atoms)):# atom in vector[3]:
    #         atom = self.var_atoms[i]
    #         self.AtomInfo.atom_freqs[atom][1] = params['var_atom'][atom]
    #         self.AtomInfo.atom_freqs[atom][0] = 1-params['var_atom'][atom]
    #         self.params['var_atoms'][atom] = params['var_atom'][atom]
    #     for i in range(len(self.var_res)):
    #         residue = self.var_res[i]
    #         self.ResidueInfo.res_freqs[residue][1:] = [params['var_res'][residue]]*(len(self.ResidueInfo.res_freqs[residue])-1)
    #         self.params['var_res'][residue] = params['var_res'][residue]
    #     self.masses = self.MakeModel()
    #     self.current_param = 'final'
    #     self.residual = sum(self.compute_residual()**2)


    def fitschedule(self):
        '''Overarching scheduler to determine which parameters are to be held constant and which to fit. Based on the
        original isodist schedule, with slight rearrangement of order. Amplitudes are first fit to be within the right
        order of magnitude of the problem. M_off is then fit before gw to avoid difficulty fitting from narrow Gaussian
        peaks missing each other. The gaussian width is then tuned, and variable atoms/residues tuned as needed based on
        the labeling scheme. Final round of fitting fits all parameters. Scheduling is done through instance dictionary
        "schedule"

        returns: None
        '''
        #Schedules the parameters to be fit in each round
        roundnumber = 1

        # Fitting amplitudes preliminarily
        self.schedule['amps'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        # Fitting m_off
        self.schedule['amps'] = 0
        self.schedule['m_off'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        # Fitting gaussian width with m_off
        self.schedule['gw'] = 1
        self.scipy_optimize_ls()
        self.loground(roundnumber, True)
        roundnumber += 1

        # Fitting variable atoms
        if len(self.params['var_atoms']) > 0:
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

        '''Normalize the amplitudes
        #amp_sum = sum(self.params['amps'])
        #self.params['amps'] = [x/amp_sum  for x in self.params['amps']]'''
        #Removed in order to fit amplitudes to Gaussian

    def set_params(self,vector):
        '''Sets the model parameters based on a vector. Determines which parameters are specified by checking
        the fitting schedule'''
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
        '''Computes bounds for each parameter based on the fitting schedule.

        returns: tuple of arrays, containing lower and upper bounds'''

        bound_dict={'gw':[0.001,.1],'var_atoms':[0,1],'amps':[0,np.inf],'m_off':[-0.1,0.1],'var_res':[0,1]} #maps parameter dist type to bounds
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
        '''Computes the residual of the generated model versus the target spectrum. Interpolates the datapoints at the recorded
        mass spectrum masses based on the generated model through linear interpolation.

        returns: array of residuals, out = model - target. Additionally, sets instance array residuals to the residuals vector for
        chi-square calculation as necessary later.
        '''
        #Computes the chi square residual, as well as the residual vector
        if not self.current_param == 'final':
            self.set_params(param_vector)
        self.masses = self.MakeModel()
        dm = self.dm
        masses = self.masses
        shifted_masses = self.shifted_masses
        model_masses = np.interp(shifted_masses, dm*np.arange(len(masses)), masses)
        model_masses=np.array(model_masses)


        if MATCH_HIGH_POINTS:
            model_max = max(model_masses)
            self.model_scale = self.target_max/model_max
            model_masses *= self.model_scale
            self.masses *= self.model_scale

        out = (model_masses-self.target_intensities)
        self.residuals = out#np.sum(out**2)
        return out

    def scipy_optimize_ls(self):
        '''Runs the least square optimization algorithm, using trust region bounds as default

        returns: None
        '''
        params = self.get_params()
        bounds = self.get_bounds()
        scaled_tolerance = FTOL*(1e6/max(self.target_intensities))*(2.5e4/self.N)
        try:
            scipy.optimize.least_squares(self.compute_residual,params,bounds=bounds,ftol=scaled_tolerance,max_nfev=MAX_ITERS)#, diff_step = [2]*len(params))
        except:
            print(self.compute_residual())
