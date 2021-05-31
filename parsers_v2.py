#import fitting_package_v3 as pysofit
import numpy as np
from numpy.fft import fft,ifft,rfft,irfft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import pi,e

from pathlib import Path

class MainInfoParser(): #parses the .in file to hold relevant variables
    def __init__(self,infile):
        self.file=infile
        f=open(infile,'r')
        self.opt = f.readline().split()[0]
        print(self.opt)
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

class AtomInfoParser():
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

class ResInfoParser():
    def __init__(self, res_file, atom_parser):

        f = open(res_file,'r')
        print(res_file)
        f.readline()#skip initial line
            ##################
            #collect all info
            ##################
        num_species = int(f.readline().split()[0])
        species_names=[]
        species_amps=[]
        for i in range(num_species):
            line=f.readline().split()
            species_names.append(line[0])
            print(species_names)
            species_amps.append(float(line[1]))
            print(species_amps)
        print(len(species_amps))

        n_var_atoms = 0 #number of variable atoms
        num_atoms = int(f.readline().split()[0]) #number of atom types: "elements"
        atom_names = []
        atom_init_values = [] #Labeled atoms. Need to allow this to override the atoms
        atom_modes = []

        for i in range(num_atoms):
            line=f.readline().split()
            print(line)
            atom_names.append(line[0])
            atom_modes.append(line[1])
            try:
                atom_init_values.append(float(line[2]))
            except:
                atom_init_values.append(None)

        #residues multiplicity of atoms
        residue_info=dict() #keys are symbols, values lists of atom multiplicities
        #one for each species
        reading=True
        residue_names = []
        n_var_res = 0
        residue_modes = []
        residue_init_values = []
        res_freqs = dict()
        while reading:
            line = f.readline()
            print(line)
            if line=='':
                reading=False
                break
            current_res = line.split()[0]
            current_mode = line.split()[1]

            residue_names.append(current_res)
            residue_modes.append(current_mode)
            try:
                residue_init_values.append(float(line.split()[2]))
            except:
                residue_init_values.append(None)
            residue_info[current_res] = []
            res_freqs[current_res] = []
            for i in range(num_species):
                a = f.readline()
                print(a)
                line=list(map(int,a.split()[:num_atoms]))
                residue_info[current_res].append(line)
                res_freqs[current_res].append(1)

        self.atom_names = atom_names
        self.atom_init_values = atom_init_values
        self.atom_modes = atom_modes
        self.residue_info = residue_info
        self.residue_modes = residue_modes
        self.residue_names = residue_names
        self.residue_init_values = residue_init_values
        self.species_amps = species_amps
        self.species_names = species_names
        self.num_species = num_species
        self.res_freqs = res_freqs

class BatchInfoParser():
    def __init__(self, batch_file):
        f=open(batch_file,'r')
        #f.readline() #skip header line
        batch_models=[]
        batch_syms = []
        batch_mults = []
        charges=[]
        data_files=[]
        pep_names=[]
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


class ExpSpectrumParser():
    def __init__(self,data_file,charge):
        lines=open(data_file,'r').read().split('\n')[:-1] #cut off empty line at end
        self.masses=[]
        self.raw_intensities=[]
        for line in lines:
            splitline=line.split(' ')
            self.masses.append(float(splitline[0])*charge)
            self.raw_intensities.append(float(splitline[1]))
        self.m_hd=self.masses[0]
        self.largest_mass=self.masses[-1]
    def get_unbinned_target_array(self):#returns a tuple of target masses, values
        #subtract off baseline offset
        min_intensity=min(self.raw_intensities)
        #min_intensity = 0.5*(sorted(self.raw_intensities)[len(self.raw_intensities)//4] + sorted(self.raw_intensities)[len(self.raw_intensities)//2])
        baselined_intensities=list(map(lambda x:x-min_intensity,self.raw_intensities))
        self.vert_shift=min_intensity
        #normalize
        #intensity_sum=sum(baselined_intensities)
        #self.scaledown=intensity_sum
        #normalized_intensities=list(map(lambda x:x/intensity_sum,baselined_intensities))
        return self.masses,baselined_intensities#normalized_intensities
    def get_target_array_v1(self,N,dm,exp_box_size,m_hd):
        #subtract off baseline offset
        min_intensity=min(self.raw_intensities)
        #min_intensity = 0.5*(sorted(self.raw_intensities)[len(self.raw_intensities)//4] + sorted(self.raw_intensities)[len(self.raw_intensities)//2])
        baselined_intensities=list(map(lambda x:x-min_intensity,self.raw_intensities))
        self.vert_shift=min_intensity
        #normalize
        #intensity_sum=sum(baselined_intensities)
        #self.scaledown=intensity_sum
        #normalized_intensities=list(map(lambda x:x/intensity_sum,baselined_intensities))
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
