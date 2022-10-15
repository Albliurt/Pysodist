from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from math import pi,e,ceil,floor
import options_parser
import parsers_v2 as parsers
from gooey import Gooey, GooeyParser
import argparse

j = 1j
options=options_parser.OptionsParser().options

@Gooey(program_name = "Spectrum Plot Setup")
def InputArgs():
    parser = GooeyParser(description = "Parameters for initial desired spectrum")
    model_group = parser.add_argument_group("Model Files", "Choose an atom model file and residue model file")
    model_group.add_argument('Atom file', widget="FileChooser", help = "Atom model file path")
    model_group.add_argument('Residue file', widget="FileChooser", help = "Residue model file path")
    peptide_group = parser.add_argument_group("Peptide Info", "Specify peptide sequence, charge, labeling, etc.")
    peptide_group.add_argument('Sequence', help = "Sequence of the peptide of interest", default = "LEDGVVIPADGR")
    peptide_group.add_argument('Charge', help = "Peptide charge", default = 2)
    peptide_group.add_argument('Labeling', help = "Labeling scheme: e.g. C13, N15", default = "C13")
    display_group = parser.add_argument_group("Display Options", "Parameters for displaying the spectrum")
    display_group.add_argument('Resolution (dm)', help = "Spacing between points", default = 0.001)
    display_group.add_argument('Left Mass Pad', help = "Mass points to the left of lowest computed mass", default = 2.5)
    display_group.add_argument('Right Mass Pad', help = "Mass points to the right of highest computed mass", default = 2.5)
    display_group.add_argument('Gaussian width (gw)', help = "Gaussian width of each peak", default = 0.006)
    args = vars(parser.parse_args())
    return args

args = InputArgs()


spectrumfile = "C:\\Users\\alber\\Desktop\\PysodistDev\\CorrCarbon\\TestData\\13C_titration6\\spectra\\VGSEFQDVVLETV_2_711.362_243-255_86.1.tsv"
#Model files used to generate the spectrum
#AtomInfo contains isotopic frequencies and masses
#ResidueInfo contains atomic compositions of residues as well as amplitudes (initial)
atomfile = args["Atom file"].replace("\\","\\\\")
resfile = args["Residue file"].replace("\\","\\\\")

exp_data = parsers.ExpSpectrumParser(spectrumfile, 2)
target = exp_data.get_unbinned_target_array()

#Initial sequence, charge, labeling
#Mass pads are for visuals
sequence = args["Sequence"]
charge = int(args["Charge"])
labeling = args["Labeling"]
left_mass_pad = float(args["Left Mass Pad"])
right_mass_pad = float(args["Right Mass Pad"])
dm = float(args["Resolution (dm)"])
gw = float(args["Gaussian width (gw)"])
#print(args.'Atom file')

AtomInfo = parsers.AtomInfoParser(atomfile)
ResidueInfo = parsers.ResInfoParser(resfile, AtomInfo)

#Modifying sequence, computing multiplicities of residues
#Then, computing N
res_syms = list(set(sequence + 'Z' + charge*'X'))
res_mults = list(map(lambda sym:str.count(sequence + 'Z' + charge*'X',sym),res_syms))

def correlateCarbons():
    totallength = len(ResidueInfo.atom_names)
    for i in range(ResidueInfo.num_species):
        for residue in ResidueInfo.residue_composition:
            carbon_correlation_sets = list(set(ResidueInfo.residue_corrs[residue]))
            carbon_correlation_mults = list(map(lambda n:ResidueInfo.residue_corrs[residue].count(n), carbon_correlation_sets))
            zero_padded_mults = []
            for k in range(2,6):
                if k in carbon_correlation_sets:
                    zero_padded_mults.append(carbon_correlation_mults[carbon_correlation_sets.index(k)])
                else:
                    zero_padded_mults.append(0)



    return totallength




def computeNmw(sequence, charge):
    global res_syms
    global res_mults
    global mw
    global N

    res_syms = list(set(sequence + 'Z' + charge*'X'))
    res_mults = list(map(lambda sym:str.count(sequence + 'Z' + charge*'X',sym),res_syms))
    atoms = np.zeros(len(ResidueInfo.residue_composition[res_syms[0]][0]))

    mw = 0
    labeled_mass_diff = 0
    mass_pad = left_mass_pad + right_mass_pad
    for i in range(len(res_syms)):
        atom_mult = ResidueInfo.residue_composition[res_syms[i]][0]
        for k in range(len(atom_mult)):
            atoms[k] += res_mults[i]*atom_mult[k]
            mw += atom_mult[k]*AtomInfo.atom_masses[ResidueInfo.atom_names[k]][0]*res_mults[i]
            if 'C13' in labeling and 'C' in ResidueInfo.atom_names[k]:
                labeled_mass_diff += 1.003355*res_mults[i]*atom_mult[k]
            if 'N15' in labeling and 'N' in ResidueInfo.atom_names[k]:
                labeled_mass_diff += 0.997035*res_mults[i]*atom_mult[k]
    print('Molecular weight: ' + str(mw))
    print('Mass window: ' + str(mw - left_mass_pad) + ' - ' + str(right_mass_pad + mw + labeled_mass_diff))
    N = int((labeled_mass_diff+mass_pad)//dm + 1)

#Overarching spectrum generating function, which is called during the start
#and also called during any "update" event - when the sliders are changed
def signal(sequence, charge, vars, mw, amps = None, gw = 0.003, labeling = 'C13', m_off = 0, base = 0, dm = float(options['dm']), left_mass_pad = 2.5, right_mass_pad = 2.5):
    mass_pad = left_mass_pad + right_mass_pad
    sequence += 'Z' + charge*'X'

    res_syms = list(set(sequence))
    res_mults = list(map(lambda sym:str.count(sequence,sym),res_syms))
    def Convolution(ft_spectra,mults,names=None):
        '''Convolves Fourier spectra with given multiplicities. Raises each Fourier spectrum to corresponding multiplicity, and then multiplies them
        :param ft_spectra: list or dictionary of spectra
        :param mults: list of multiplicities for each spectrum. Indexed with names if ft_spectra is a dictionary
        :param names: optional parameter if ft_spectra is a dictionary. Provides the keys for the dictionary

        returns: convolved spectrum in the Fourier domain
        '''
        if names == None:
            names = range(len(mults))
        length = N//2 + 1
        conv = np.ones(length, dtype=complex)
        for i in range(len(mults)):
            if mults[i] == 0:
                continue
            elif mults[i] == 1:
                try:
                    conv *= ft_spectra[names[i]]
                except:
                    print(ft_spectra[names[i]])
            else:
                conv *= ft_spectra[names[i]]**mults[i]
        return conv

    def LinCombFt(spectra,amps):
        '''Takes the ar combination of multiple Fourier transformed spectra with given weights (amps). Returns in the Fourier domain
        :param spectra: array of spectra to be arly combined
        :param amps: amplitudes of corresponding spectra

        returns: arly combined spectrum in the Fourier domain, containing the sum of amps[i]*spectra[i]
        '''
        return np.dot(amps, spectra)

    def Ft_Shift(shift):
        '''Generates the Fourier transform of a delta functions, which shifts a spectrum by the shift amount
        :param shift: by how much mass to shift the spectrum

        returns: Fourier transformed delta function
        '''
        print(str(-2*pi*j*shift/(N*dm)))
        fourier_array = np.exp((-2*pi*j*shift/(N*dm))*np.arange(N//2+1))
        return fourier_array

    def Ft_Gaussian():
        '''Makes a Gaussian mass array with a given gw, specified by the fitting parameter 'gw'

        returns: None'''
        temp_array = np.arange(0,N//2)
        half_intensity_array = (1/(gw*(2*pi)**0.5))*np.exp(-0.5*(dm/gw*temp_array)**2)
        full_intensity_array = np.concatenate((half_intensity_array,half_intensity_array[::-1]))
        return rfft(full_intensity_array)

    '''Change the frequencies that you want to specify'''






    #Defines separately the "unmixed" atom models which are the fourier transform of delta functions
    #Since only the atom frequencies change, these unmixed models are added at given frequencies
    unmixed_ft_atom_models = dict()
    ft_atom_models = dict()
    for i in range(len(ResidueInfo.atom_modes)):
        if ResidueInfo.atom_init_values[i] is not None:
            #Manually set if either fixed or variable
            if ResidueInfo.atom_modes[i] == "fixed" or ResidueInfo.atom_modes[i] == "variable":
                AtomInfo.atom_freqs[ResidueInfo.atom_names[i]][1] = ResidueInfo.atom_init_values[i]
                AtomInfo.atom_freqs[ResidueInfo.atom_names[i]][0] = 1 - ResidueInfo.atom_init_values[i]

                if 'C' in ResidueInfo.atom_names[i] and not 'C' == ResidueInfo.atom_names[i]:
                    for k in range(2, 6):
                        AtomInfo.atom_freqs[ResidueInfo.atom_names[i] + str(k)][1] = ResidueInfo.atom_init_values[i]
                        AtomInfo.atom_freqs[ResidueInfo.atom_names[i] + str(k)][0] = 1-ResidueInfo.atom_init_values[i]



    for var in vars:
        AtomInfo.atom_freqs[var[0]][1] = var[1]
        AtomInfo.atom_freqs[var[0]][0] = 1-var[1]
        if 'C' in var[0] and not 'C' == var[0]:
            for k in range(2, 6):
                print(var[0])
                AtomInfo.atom_freqs[var[0] + str(k)][1] = var[1]
                AtomInfo.atom_freqs[var[0] + str(k)][0] = 1-var[1]

    print(AtomInfo.atom_freqs)
    print(ResidueInfo.atom_names)

    if amps is not None:
        ResidueInfo.species_amps = amps


    for atom in AtomInfo.atom_masses:
        unmixed_ft_atom_models[atom] = [np.exp((-2*pi*j/N)*(AtomInfo.atom_masses[atom][i]/dm)*np.arange(N//2+1)) for i in range(len(AtomInfo.atom_masses[atom]))]
        ft_atom_models[atom] = np.dot(AtomInfo.atom_freqs[atom], unmixed_ft_atom_models[atom])#fourier_array

    #For SILAC experiments, variable residues can be computed by computing separate labeling schemes and adding them
    #unmixed_ft_residue_models describes these
    unmixed_ft_residue_models = []
    ft_residue_models = []
    for i in range(ResidueInfo.num_species):
        res_init = dict()
        for res in ResidueInfo.residue_composition:
            res_init[res] = None
        ft_residue_models.append(res_init)
        unmixed_ft_residue_models.append(res_init.copy())

    for residue in ResidueInfo.residue_composition:
        for i in range(ResidueInfo.num_species):
            names = ResidueInfo.atom_names
            unmixed_ft_residue_models[i][residue] = Convolution(ft_atom_models, ResidueInfo.residue_composition[residue][i], names)
            ft_residue_models[i][residue] = ResidueInfo.res_freqs[residue][i]*unmixed_ft_residue_models[i][residue]+unmixed_ft_residue_models[0][residue]*(1-ResidueInfo.res_freqs[residue][i])

    ft_species_models = [Convolution(ft_residue_models[k], res_mults, res_syms) for k in range(ResidueInfo.num_species)]
    ft_stick = LinCombFt(ft_species_models, ResidueInfo.species_amps)
    gaussian_shifted_stick = Convolution([ft_stick, Ft_Gaussian(), Ft_Shift(left_mass_pad-mw)],[1,1,1])

    mz_stick = irfft(gaussian_shifted_stick, n=N)
    return mz_stick



''''''

mw = 0
N = 0

print("TEST")
computeNmw(sequence, charge)

fig = plt.figure(figsize = (12,7))
ax = fig.add_subplot(111)

fig.subplots_adjust(bottom=0.4, left = 0.3)

#x axis
mass_axis=np.arange(0,N*dm,dm) + mw - left_mass_pad
mass_axis=mass_axis[:N]

# Draw the initial plot
# The '' variable is used for modifying the  later
originallength = correlateCarbons()
spectrum = signal(sequence, charge, [], gw = gw, mw = mw)
'''#[] = '''
ax.scatter(target[0], np.array(target[1])/10000, color = 'blue', s = 5)
ax.plot(mass_axis, spectrum, color='red')
# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it

var_atom_sliders = []
var_atom_slider_axs = []
var_atom_names = []

for i in range(originallength):
    print(ResidueInfo.atom_modes)
    if ResidueInfo.atom_modes[i] == "variable" or ResidueInfo.atom_modes[i] == "fixed":
        var_atom_names.append(ResidueInfo.atom_names[i])
        var_atom_slider_axs.append(fig.add_axes([0.2, 0.3 - len(var_atom_sliders)*0.05, 0.25, 0.015]))#, facecolor=axis_color)
        var_atom_sliders.append(Slider(var_atom_slider_axs[-1], 'Label Freq: ' + ResidueInfo.atom_names[i], 0.0, 1.0, valinit= AtomInfo.atom_freqs[ResidueInfo.atom_names[i]][1]))

amp_sliders = []
amp_slider_axs = []
for i in range(ResidueInfo.num_species):
    amp_slider_axs.append(fig.add_axes([0.6, 0.3 - (len(amp_sliders))*0.05, 0.25, 0.015]))
    amp_sliders.append(Slider(amp_slider_axs[-1], 'AMP_'+ ResidueInfo.species_names[i], 0.1, 5.0, valinit = ResidueInfo.species_amps[i]))

gw_slider_ax = fig.add_axes([0.2, 0.3-len(var_atom_sliders)*0.05, 0.25, 0.015])
gw_slider = Slider(gw_slider_ax, 'Gaussian width', 0.001, 0.1, valinit = gw)

# Draw another slider
# freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])#, facecolor=axis_color)
# freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0)

# Define an action for modifying the  when any slider's value changes
def sliders_on_changed(val):
    var_vals = [slide.val for slide in var_atom_sliders]
    amp_vals = [slide.val for slide in amp_sliders]
    mass_axis=np.arange(0,N*dm,dm) + mw - left_mass_pad
    mass_axis=mass_axis[:N]
    ax.clear()

    ax.scatter(target[0], np.array(target[1])/10000, color = 'blue', s = 5)
    ax.plot(mass_axis, signal(sequence, charge, zip(var_atom_names, var_vals), amps = amp_vals, gw = gw, mw = mw), color = 'r')
    #.set_ydata(signal(sequence, charge, zip(var_atom_names, var_vals), amps = amp_vals, gw = gw))#amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()

def sequence_on_changed(text):
    global sequence
    global charge
    sequence = text
    computeNmw(sequence, charge)
    mass_axis=np.arange(0,N*dm,dm) + mw - left_mass_pad
    mass_axis=mass_axis[:N]
    ax.clear()
    ax.plot(mass_axis, signal(sequence, charge, [], gw = gw, mw = mw), color = 'r')
    ax.scatter(target[0], np.array(target[1])/10000, color = 'blue', s = 5)

    fig.canvas.draw_idle()

def charge_on_changed(chargeN):
    global charge
    global sequence
    charge = int(chargeN)
    computeNmw(sequence, charge)
    mass_axis=np.arange(0,N*dm,dm) + mw - left_mass_pad
    mass_axis=mass_axis[:N]
    ax.clear()
    ax.plot(mass_axis, signal(sequence, charge, [], gw = gw, mw = mw), color = 'r')
    ax.scatter(target[0], np.array(target[1])/10000, color = 'blue', s = 5)

def gw_on_changed(width):
    global gw
    gw = width
    mass_axis=np.arange(0,N*dm,dm) + mw - left_mass_pad
    mass_axis=mass_axis[:N]
    ax.clear()
    ax.plot(mass_axis, signal(sequence, charge, [], gw = gw, mw = mw), color = 'r')
    ax.scatter(target[0], np.array(target[1])/10000, color = 'blue', s = 5)

axbox = plt.axes([0.1, 0.75, 0.15, 0.05])
sequence_box = TextBox(axbox, 'Sequence: ', initial = sequence)
sequence_box.on_submit(sequence_on_changed)

axbox2 = plt.axes([0.1, 0.65, 0.15, 0.05])
charge_box = TextBox(axbox2, 'Charge: ', initial = charge)
charge_box.on_submit(charge_on_changed)

for slide in var_atom_sliders:
    slide.on_changed(sliders_on_changed)
for slide in amp_sliders:
    slide.on_changed(sliders_on_changed)

gw_slider.on_changed(gw_on_changed)

plt.suptitle(sequence)
plt.show()
