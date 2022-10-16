from pathlib import Path
import os
#Changing the path to provide the packages needed
filepath = Path(__file__).parent
workingpath = Path.cwd()
os.chdir(filepath)

import sys
import parsers_v2 as parsers
import fitting_package_v3 as pysofit
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import csv
import options_parser


options=options_parser.OptionsParser().options
DM = float(options['dm'])
carry_over_params = True if options['carry_over_params'] == 'True' else False
plot_progress = True if options['plot_progress'] == 'True' else False
sep_line="++++++++++++++++++++++++++++++++"

os.chdir(workingpath)

#def main(input_file, N = 65536, AUTO_N = True, PLOT_FIT=False, DM = 0.001, USE_RAW_DATA = True,
#	EXP_BOX_SIZE = 0.05, AUTO_SAVE=True, CARRY_OVER_PARAMS=False):

def main(input_file):
	###################################
	#Overarching method to run pysodist
	###################################
	#Input_file is the .in file generated by run_isodist. Other parameters are all options that can be passed in
	#Parsing of input files
	MainInfo = parsers.MainInfoParser(input_file)
	#Parsing of atom model file
	AtomInfo = parsers.AtomInfoParser(MainInfo.atomfile)
	#Parsing of residue model file
	ResidueInfo = parsers.ResInfoParser(MainInfo.resfile, AtomInfo)
	#Parsing of the batch file referenced in the .in file
	BatchInfo = parsers.BatchInfoParser(MainInfo.batchfile)

	#Initializing a saved params variable if they are to be carried across peptide fits
	if carry_over_params:#CARRY_OVER_PARAMS:
		saved_params=None

	#Creates the csv file that the saved parameters will be output into
	#Prints out the first row, which are the just the parameter names
	params_csv = MainInfo.batchfile+".csv"
	with open(params_csv, 'w', newline='') as f:
		line = ["file","protein","pep","mw","z_charge","chisq","mz","B","off","gw"]
		for i in ResidueInfo.species_names:
			line.append("AMP_" + str(i))
		for i in range(len(ResidueInfo.atom_modes)):
			if ResidueInfo.atom_modes[i] == "variable":
				line.append("FRC_"+str(ResidueInfo.atom_names[i]))
		for i in range(len(ResidueInfo.residue_modes)):
			if ResidueInfo.residue_modes[i] == "variable":
				line.append("FRC_"+str(ResidueInfo.residue_names[i]))
		f_writer = csv.writer(f)
		f_writer.writerow(line)

	for i in range(BatchInfo.num_batches):
		#Looping through each peptide in the batch file
		params = dict() #Parameters vector initialization
		var_atoms = dict() #Variable atoms: keys are atom symbols, and values are frequencies
		var_res = dict() #Variable residues: keys are residue symbols, and values are frequencies

		#The atom and residue data from the atom and residue model files
		atom_freqs = AtomInfo.atom_freqs
		res_freqs = ResidueInfo.res_freqs

		#Setting the variable atoms to have the desired frequency
		'''Does this work? Carry over params'''
		for j in range(len(ResidueInfo.atom_modes)):
			if ResidueInfo.atom_init_values[j] is not None:
				#Manually set if either fixed or variable
				if ResidueInfo.atom_modes[j] == "fixed" or ResidueInfo.atom_modes[j] == "variable":
					atom_freqs[ResidueInfo.atom_names[j]][1] = ResidueInfo.atom_init_values[j]
					atom_freqs[ResidueInfo.atom_names[j]][0] = 1 - ResidueInfo.atom_init_values[j]
				if 'C' in ResidueInfo.atom_names[j] and not 'C' == ResidueInfo.atom_names[j]:
					for k in range(2,6):
						atom_freqs[ResidueInfo.atom_names[j] + str(k)][1] = ResidueInfo.atom_init_values[j]
						atom_freqs[ResidueInfo.atom_names[j] + str(k)][0] = 1 - ResidueInfo.atom_init_values[j]

			if ResidueInfo.atom_modes[j] == "variable":
				#Add the atom to var_atoms if its mode is variable
				var_atoms[ResidueInfo.atom_names[j]] = ResidueInfo.atom_init_values[j]

		#Setting the variable residues to have the desired frequency
		for j in range(len(ResidueInfo.residue_modes)):
			if ResidueInfo.residue_modes[j] == "variable" or ResidueInfo.residue_modes[j] == "fixed":
				#Manually set the frequencies if fixed or variable
				res_freqs[ResidueInfo.residue_names[j]][1:] = [ResidueInfo.residue_init_values[j]]*(len(res_freqs[ResidueInfo.residue_names[j]]) - 1)
			else:
				#Otherwise let it equal 1
				res_freqs[ResidueInfo.residue_names[j]] = [1]*(len(res_freqs[ResidueInfo.residue_names[j]]))
			if ResidueInfo.residue_modes[j] == "variable":
				#Add the variable to var_res if its mode is variable
				var_res[ResidueInfo.residue_names[j]] = ResidueInfo.residue_init_values[j]

		params['var_atoms'] = var_atoms
		params['var_res'] = var_res

		print(sep_line)
		print('\nFitting Peak:',BatchInfo.pep_names[i])
		print('Data at:',BatchInfo.data_files[i])

		###########################
		#Build experimental target
		###########################

		#Gather the experimental data from the spectra file
		#scales to be in mass (not mz), subtracts a baseline offset
		exp_data=parsers.ExpSpectrumParser(BatchInfo.data_files[i],BatchInfo.charges[i])
		if exp_data.m_hd is None:
			print("An issue was encountered in " + BatchInfo.pep_names[i])
			print("Skipping this peptide")
			continue
		m_hd=exp_data.m_hd #the 'heterodyne shift'

		#Determine the number of points if AUTO_N.
		#if AUTO_N:
		largest_exp_mass=exp_data.largest_mass
		N=(largest_exp_mass-m_hd)//DM+2
		if N%2==1:
			N+=1
		N=int(N)

		#if not USE_RAW_DATA:
		#	target=exp_data.get_target_array_v1(N,DM,EXP_BOX_SIZE,m_hd) #does cubic interpolation
		#else:
		target=exp_data.get_unbinned_target_array()


		##################
		#Gradient descent
		##################

		#Loading out parameters from previous fit if carrying over params
		if carry_over_params:#CARRY_OVER_PARAMS:
			if saved_params is not None:
				params = saved_params.copy()

		params['m_off'] = MainInfo.m_off_init
		params['gw'] = MainInfo.gw_init

		if not carry_over_params:
			ResidueInfo.species_amps = ResidueInfo.original_species_amps.copy()
		else:
			ResidueInfo.species_amps = [x/(sum(ResidueInfo.species_amps)) for x in ResidueInfo.species_amps]
		# 	print(ResidueInfo.species_amps)
		# 	print("Bleh")
		params['amps'] = ResidueInfo.species_amps
		#print(params['amps'])
		#Array of peptide information to be passed into the fitting package
		#logfile = input_file[:-3] + ".log"
		fit = pysofit.FittingProblem(N,DM,AtomInfo,ResidueInfo, BatchInfo, i, params, m_hd, target, match_high_points)#, logfile)
		fit.fitschedule()

		print("Fitting peptide: "+ str(BatchInfo.pep_names[i]))
		if plot_progress:#PLOT_PROGRESS:
			fit.plot()
		if carry_over_params:#CARRY_OVER_PARAMS:
			saved_params = fit.params.copy()
		################
		#Saving the Fit
		################
		model_spectrum_tsv = BatchInfo.data_files[i][:-4]+'.fit'
		fit.save_fit(params_csv, model_spectrum_tsv, exp_data.vert_shift, BatchInfo.charges[i])

#Parsing command line arguments to be passed in. Not necessary when putting everything together
parser = argparse.ArgumentParser(description = '')
parser.add_argument('inputfile', help = 'Path to the .in file generated by run_isodist')
parser.add_argument('--plot_progress', action='store_true', default = False, help = 'Plot final fits using pyplot')
parser.add_argument('--cubic_interp', action='store_true', default = False, help = 'Use cubic interpolation, rather than the raw spectral data')
parser.add_argument('--force_new_params', action='store_true', default = False, help = 'Carry over parameters from one peptide to the next')
parser.add_argument('--match_high_points', action='store_true', default = False, help = 'Makes sure the maximum intensity of the experimental data and model match. Fails if spectrum has contaminating peaks. Default false.')


args = parser.parse_args()
input_file = args.inputfile.replace('\\','/')
plot_progress = args.plot_progress
cubic_interp = args.cubic_interp
carry_over_params = not args.force_new_params
carry_over_params = False
match_high_points = args.match_high_points
main(input_file)#, USE_RAW_DATA = not cubic_interp, CARRY_OVER_PARAMS = carry_over_params)
