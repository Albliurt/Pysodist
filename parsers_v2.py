import numpy as np
from numpy.fft import fft,ifft,rfft,irfft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import pi,e
import ast

class MainInfoParser(): #parses the .in file to hold relevant variables
    def __init__(self, in_file):
        self.file=in_file
        f=open(in_file,'r')
        self.opt = f.readline().split()[0]

        self.batchfile = f.readline().split()[0]
        self.atomfile = f.readline().split()[0]
        self.resfile = f.readline().split()[0]
        self.fit_iter = int(f.readline().split()[0])
        self.sig_global = float(f.readline().split()[0])

        baseline_line = f.readline()
        self.b_init = baseline_line.split()[0]
        self.b_tag = baseline_line.split()[1]

        self.m_off_init = float(f.readline().split()[0])
        self.gw_init = float(f.readline().split()[0])

        f.close()

class AtomInfoParser(): #parses the atom model file to hold relevant data - masses and frequencies
    def __init__(self, file='atom_defs.txt'):
        self.file = file
        self.atom_masses = dict()
        self.atom_freqs = dict()

        reading=True
        current_atom=None
        f = open(self.file,'r')
        while reading:
            new_line=f.readline()
            if new_line=='':
                reading=False
                break
            elif new_line.split()[1][0].isalpha(): #check if new atom
                current_atom=new_line.split()[1]
                self.atom_masses[current_atom]=[]
                self.atom_freqs[current_atom]=[]
            else: #otherwise add mass,freq to data
                split_line=new_line.split()
                self.atom_masses[current_atom].append(float(split_line[0]))
                self.atom_freqs[current_atom].append(float(split_line[1]))

class ResInfoParser(): #parses the residue model file. Gives the variable residues, the atomic composition of each residue, as well as amplitudes and variable atoms
    def __init__(self, res_file, atom_parser):

        f = open(res_file,'r')
        f.readline()#skip initial line
        ##################
        #collect all info
        ##################

        #Parse the species data - number of species and initial amplitudes
        num_species = int(f.readline().split()[0]) #number of species
        species_names=[] #U, L, F, etc. Name of the species (labeling condition)
        species_amps=[] #Amplitudes of each species
        for i in range(num_species):
            line=f.readline().split()

            species_names.append(line[0])
            species_amps.append(float(line[1]))

        #Parse the variable atoms and their initial values
        num_atoms = int(f.readline().split()[0]) #number of atom types in model: C, H, N, O, S, and variable atoms, etc.
        atom_names = [] #Elemental symbol
        atom_init_values = [] #Initial frequency of special atoms (variable or fixed)
        atom_modes = [] #How to treat the special atoms

        corr_atoms = []
        for i in range(num_atoms):
            line=f.readline().split()
            atom_names.append(line[0])
            atom_modes.append(line[1])
            try:
                atom_init_values.append(float(line[2]))
            except:
                atom_init_values.append(None)
            if('C' in line[0] and not line[0] == 'C'):
                corr_atoms.append(line[0])

        #residues multiplicity of atoms
        residue_composition=dict() #keys are symbols, values lists of atom multiplicities
        #one for each species
        reading=True
        residue_names = [] #One letter identifier for amino acids
        residue_modes = [] #Modes for each amino acid, for SILAC

        residue_corrs = dict()#[] #Correlated carbons for each amino acid

        residue_init_values = [] #Initial frequencies for special residues
        res_freqs = dict() #Dictionary for frequencies overall. Frequency is just 1
        #for most residues, but are set to be the values in residue_init_values in
        #pysodist

        while reading:
            line = f.readline()
            if line=='':
                reading=False
                break

            current_res = line.split()[0]
            current_mode = line.split()[1]
            if len(line.split()) > 2:
                current_corr = ast.literal_eval(line.split()[2])
            else:
                current_corr = []

            residue_names.append(current_res)
            residue_modes.append(current_mode)
            residue_corrs[current_res] = current_corr
            try:
                residue_init_values.append(float(line.split()[2]))
            except:
                residue_init_values.append(None)

            residue_composition[current_res] = []
            res_freqs[current_res] = []

            for i in range(num_species):
                a = f.readline()
                line=list(map(int,a.split()[:num_atoms]))
                residue_composition[current_res].append(line)
                res_freqs[current_res].append(1)

        totallength = len(atom_names)
        for i in range(num_species):
            for residue in residue_composition:
                carbon_correlation_sets = list(set(residue_corrs[residue]))
                carbon_correlation_mults = list(map(lambda n:residue_corrs[residue].count(n), carbon_correlation_sets))
                zero_padded_mults = []
                for k in range(2,6):
                    if k in carbon_correlation_sets:
                        zero_padded_mults.append(carbon_correlation_mults[carbon_correlation_sets.index(k)])
                    else:
                        zero_padded_mults.append(0)
                print(residue)
                print(residue_composition[residue][i])
                for k in range(totallength):
                    atom = atom_names[k]
                    if 'C' in atom and not 'C' == atom:
                        for num in range(2,6):
                            if not (atom+ str(num)) in atom_names:
                                atom_names.append(atom + str(num))
                                atom_modes.append("Correlated")
                                atom_init_values.append(None)
                        if residue_composition[residue][i][k] > 0:
                            residue_composition[residue][i][k] -= sum(residue_corrs[residue])
                            residue_composition[residue][i] += zero_padded_mults# carbon_correlation_sets[carbon_correlation_sets.index(num)]
                        else:
                            residue_composition[residue][i] += len(zero_padded_mults)*[0]
                print(residue_composition[residue][i])

        self.atom_names = atom_names
        self.atom_init_values = atom_init_values
        self.atom_modes = atom_modes

        self.residue_composition = residue_composition
        #print(self.residue_composition)
        self.residue_modes = residue_modes
        self.residue_names = residue_names
        self.residue_init_values = residue_init_values
        self.residue_corrs = residue_corrs
        self.corr_atoms = corr_atoms

        self.species_amps = species_amps
        self.original_species_amps = species_amps.copy()
        self.species_names = species_names
        self.num_species = num_species
        self.res_freqs = res_freqs


class BatchInfoParser(): #Parses the batch file in the input file. Gives the sequence, charges, and spectrum file
    def __init__(self, batch_file):
        f=open(batch_file,'r')
        #f.readline() #skip header line
        batch_syms = [] #List of symbols for each peptide. Used as index for multiplicities
        batch_mults = [] #Multiplicities for number of each residue in a peptide
        charges=[] #Charges for each peptide
        data_files=[] #File path for each peptide's spectrum file
        pep_names=[] #Sequence of the peptide on paper, without added protons/water
        reading=True
        while reading:
            line = f.readline()
            if line=='':
                break
            line=line.split()
            pepseq=line[0]

            charge=int(line[1])
            charges.append(charge)
            data_files.append(line[2]) #skip over retentiontime
            #build batch_model
            pep_names.append(pepseq)
            pepseq+='Z'+charge*'X'#adds on a water and protons

            res_syms = list(set(pepseq))
            batch_syms.append(res_syms)
            res_mults = list(map(lambda sym:str.count(pepseq,sym),res_syms))
            batch_mults.append(res_mults)

        ########################
        #Externally useful data
        ########################

        self.num_batches=len(batch_mults)
        self.charges=charges
        self.data_files=data_files
        self.pep_names=pep_names
        self.batch_syms = batch_syms
        self.batch_mults = batch_mults


class ExpSpectrumParser(): #parses the experimental spectrum data file
    def __init__(self,data_file,charge):
        lines=open(data_file,'r').read().split('\n')[:-1] #cut off empty line at end
        self.masses=[]
        self.raw_intensities=[]
        try:
            for line in lines:
                splitline=line.split(' ')
                self.masses.append(float(splitline[0])*charge)
                self.raw_intensities.append(float(splitline[1]))
            self.m_hd=self.masses[0]
            self.largest_mass=self.masses[-1]
        except:
            self.m_hd=None
    def get_unbinned_target_array(self):#returns a tuple of target masses, values
        #subtract off baseline offset
        min_intensity=min(self.raw_intensities)
        baselined_intensities=list(map(lambda x:x-min_intensity,self.raw_intensities))
        self.vert_shift=min_intensity
        return self.masses,baselined_intensities

    def get_target_array_v1(self,N,dm,exp_box_size,m_hd):
        #subtract off baseline offset
        min_intensity=min(self.raw_intensities)
        baselined_intensities=list(map(lambda x:x-min_intensity,self.raw_intensities))
        self.vert_shift=min_intensity
        #bin into N,dm array
        target_array=np.zeros(N,'float')
        for i,mass in enumerate(self.masses):
            index=round(exp_box_size/dm*round((mass-m_hd)/exp_box_size))%N
            #index=round((mass-m_hd)/dm)%N
            target_array[index]+=baselined_intensities[i]#normalized_intensities[i]
        return self.cubic_interpolate_zeros(target_array)
    def cubic_interpolate_zeros(self,array):
        non_zero_indices=[]
        non_zero_values=[]
        for i,val in enumerate(array):
            if val>0:
                non_zero_indices.append(i)
                non_zero_values.append(val)
        f_interp=interp1d(non_zero_indices,non_zero_values,kind='cubic')
        out=array.copy()
        for i in range(len(out)):
            if out[i]==0:
                try:
                    out[i]=max(f_interp(i),0)
                except: #i out of interpolation range, just leave
                    pass
        return out
