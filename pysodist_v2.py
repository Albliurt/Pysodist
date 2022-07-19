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
#N = int(options['N'])
#AUTO_N = True if options['auto_N'] == 'True' else False
#EXP_BOX_SIZE = float(options['exp_box_size'])
#USE_RAW_DATA = True if options['use_raw_data'] == 'True' else False

DM = float(options['dm'])
CARRY_OVER_PARAMS = True if options['carry_over_params'] == 'True' else False
PLOT_PROGRESS = True if options['plot_progress'] == 'True' else False
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
	if CARRY_OVER_PARAMS:
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
	#asdf = 0

	for i in range(BatchInfo.num_batches):
		#Looping through each peptide in the batch file
		starttime = time.time()
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
		if CARRY_OVER_PARAMS:
			if saved_params is not None:
				params = saved_params.copy()

		params['m_off'] = MainInfo.m_off_init
		params['gw'] = MainInfo.gw_init
		params['amps'] = ResidueInfo.species_amps
		#Array of peptide information to be passed into the fitting package
		#logfile = input_file[:-3] + ".log"
		fit = pysofit.FittingProblem(N,DM,AtomInfo,ResidueInfo, BatchInfo, i, params, m_hd, target)#, logfile)
		fit.fitschedule()

		print("Fitting peptide: "+ str(BatchInfo.pep_names[i]))
		if PLOT_PROGRESS:
			fit.plot()
		if CARRY_OVER_PARAMS:
			saved_params = fit.params.copy()
		################
		#Saving the Fit
		################
		model_spectrum_tsv = BatchInfo.data_files[i][:-4]+'.fit'
		fit.save_fit(params_csv, model_spectrum_tsv, exp_data.vert_shift, BatchInfo.charges[i])
		print("Atom: " + str(fit.time1/fit.timing))
		print("Residue: " + str(fit.time2/fit.timing))
		print("Stick: " + str(fit.time3/fit.timing))
		print("Rest: " + str(fit.time4/fit.timing))
		print("Total: " + str(fit.timing))
		print(str(fit.time_m/fit.timing))
		print(str(fit.time_f/fit.timing))
		print(str(time.time()-starttime))
		print(str(fit.a/fit.timing))



		#if not AUTO_SAVE:
		#	print('\nSaving fit data to:',results_tsv)
		#	fit.save_fit(results_tsv,exp_data.vert_shift,BatchInfo.charges[i]) #extra data here is because have to undo normalizations before saving
		#else:
		#	save_loc=BatchInfo.data_files[i][:-4]+'_FIT.tsv'
		#	print('\nSaving fit data to:',save_loc)
		#	fit.save_fit(save_loc,exp_data.vert_shift,BatchInfo.charges[i])
	# endtime = time.time()
	# print("Total time: "+ str(endtime-starttime))
	# print("x time: " + str(totaltime))
	# print(fit.asdf)


#Parsing command line arguments to be passed in. Not necessary when putting everything together
parser = argparse.ArgumentParser(description = '')
parser.add_argument('inputfile', help = 'Path to the .in file generated by run_isodist')
parser.add_argument('--plotting', action='store_true', default = False, help = 'Plot final fits using pyplot')
parser.add_argument('--cubic_interp', action='store_true', default = False, help = 'Use cubic interpolation, rather than the raw spectral data')
parser.add_argument('--carry_over_params', action='store_true', default = False, help = 'Carry over parameters from one peptide to the next')



args = parser.parse_args()
input_file = args.inputfile.replace('\\','/')
plotting = args.plotting
cubic_interp = args.cubic_interp
carry_over_params = args.carry_over_params

main(input_file)#, USE_RAW_DATA = not cubic_interp, CARRY_OVER_PARAMS = carry_over_params)
