from pathlib import Path
import os
#from numba import jit

filepath = Path(__file__).parent
workingpath = Path.cwd()
os.chdir(filepath)
#print(Path(__file__).parent)
#print(Path.cwd())

import sys
import parsers_v2 as parsers
import fitting_package_v3 as pysofit
import matplotlib.pyplot as plt
import numpy as np
import time
#import options_parser
#@jit(nopython=True)

sep_line="++++++++++++++++++++++++++++++++"
np.set_printoptions(threshold=sys.maxsize)


os.chdir(workingpath)

def main(input_file, N = 65536, AUTO_N = True, PLOT_PROGRESS=True, DM = 0.001, USE_RAW_DATA = True,
	EXP_BOX_SIZE = 0.05, AUTO_SAVE=True, DOUBLE_CHECK_FINAL_FIT=True, CARRY_OVER_PARAMS=True):
	########################
	#Parsing of Input Files
	########################
	MainInfo = parsers.MainInfoParser(input_file)
	AtomInfo = parsers.AtomInfoParser(MainInfo.atomfile)
	ResidueInfo = parsers.ResInfoParser(MainInfo.resfile, AtomInfo)
	BatchInfo = parsers.BatchInfoParser(MainInfo.batchfile)

	if CARRY_OVER_PARAMS:
		saved_params=None



	for i in range(BatchInfo.num_batches):
		params = dict()
		var_atoms = dict()
		var_res = dict()
		atom_freqs = AtomInfo.atom_freqs
		res_freqs = ResidueInfo.res_freqs
		for j in range(len(ResidueInfo.atom_modes)):
			if ResidueInfo.atom_init_values[j] is not None:
				if ResidueInfo.atom_modes[j] == "fixed" or ResidueInfo.atom_modes[j] == "variable":
					atom_freqs[ResidueInfo.atom_names[j]][1] = ResidueInfo.atom_init_values[j]
					atom_freqs[ResidueInfo.atom_names[j]][0] = 1 - ResidueInfo.atom_init_values[j]
			if ResidueInfo.atom_modes[j] == "variable":
				var_atoms[ResidueInfo.atom_names[j]] = ResidueInfo.atom_init_values[j]

		for j in range(len(ResidueInfo.residue_modes)):
			if ResidueInfo.residue_modes[j] == "variable" or ResidueInfo.residue_modes[j] == "fixed":
				res_freqs[ResidueInfo.residue_names[j]][1:] = [ResidueInfo.residue_init_values[j]]*(len(res_freqs[ResidueInfo.residue_names[j]]) - 1)
			else:
				res_freqs[ResidueInfo.residue_names[j]] = [1]*(len(res_freqs[ResidueInfo.residue_names[j]]))
			if ResidueInfo.residue_modes[j] == "variable":
				var_res[ResidueInfo.residue_names[j]] = ResidueInfo.residue_init_values[j]

		params['var_atoms'] = var_atoms
		params['var_res'] = var_res

		print(sep_line)
		print('\nFitting Peak:',BatchInfo.pep_names[i])
		print('Data at:',BatchInfo.data_files[i])
		###########################
		#Build experimental target
		###########################
		#scales to be in mass (not mz), subtracts a baseline offset, normalizes intensity sum to one
		exp_data=parsers.ExpSpectrumParser(BatchInfo.data_files[i],BatchInfo.charges[i])
		m_hd=exp_data.m_hd #the 'heterodyne shift'
		if AUTO_N:
			largest_exp_mass=exp_data.largest_mass
			N=(largest_exp_mass-m_hd)//DM+1
			if N%2==1:
				N+=1
			N=int(N)
		if not USE_RAW_DATA:
			target=exp_data.get_target_array_v1(N,DM,EXP_BOX_SIZE,m_hd) #does cubic interpolation
		else:
			target=exp_data.get_unbinned_target_array()
		params['m_off'] = MainInfo.m_off_init
		params['gw'] = MainInfo.gw_init
		params['amps'] = ResidueInfo.species_amps
		##################
		#Gradient descent
		##################
		if CARRY_OVER_PARAMS:
			if saved_params is not None:
				params = saved_params.copy()

		PeptideInfo = (BatchInfo.batch_mults[i], BatchInfo.batch_syms[i])
		fit = pysofit.FittingProblem(N,DM,AtomInfo,ResidueInfo, PeptideInfo, params, m_hd, target)
		fit.fitschedule()

		#errors = []
		#for i in range(100):
		#	a = 0.4 + 0.001*i
		#	fit.testParams({'var_atoms': {}, 'var_res': {'K': a}, 'm_off': -3.5146834530166074e-05, 'gw': 0.011181667080367816, 'amps': [2.97827813e+04, 7.23443062e+08]})
		#	errors.append(fit.residual)
		#print(errors)
		#plt.clf()
		#plt.plot(range(100),errors)
		#plt.show()
		#fit.testParams({'var_atoms': {}, 'var_res': {'K': 0.5000000149011612}, 'm_off': -3.5146834530166074e-05, 'gw': 0.011181667080367816, 'amps': [2.97827813e+04, 7.23443062e+08]})

		print(BatchInfo.pep_names[i])
		fit.plot()
		if CARRY_OVER_PARAMS:
			saved_params = fit.params.copy()
		################
		#Saving the Fit
		################
		if not AUTO_SAVE:
			print('\nSaving fit data to:',results_tsv)
			fit.save_fit(results_tsv,exp_data.vert_shift,BatchInfo.charges[i]) #extra data here is because have to undo normalizations before saving
		else:
			save_loc=BatchInfo.data_files[i][:-4]+'_FIT.tsv'
			print('\nSaving fit data to:',save_loc)
			fit.save_fit(save_loc,exp_data.vert_shift,BatchInfo.charges[i])

input_file = sys.argv[1]
main(input_file)
