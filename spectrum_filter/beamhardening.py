'''Code to correct for beam hardening effects in tomography experiments.
The main application of this code is for synchrotron experiments with 
a bending magnet beam.  This beam is both polychromatic and has a spectrum
which varies with the vertical angle from the ring plane.  In principle,
this could be used for other polychromatic x-ray sources.

The mathematical approach is to filter the incident spectrum by a 
series of filters.  This filtered spectrum passes through a series of
thicknesses of the sample material.  For each thickness, the transmitted
spectrum illuminates the scintillator material.  The absorbed power in 
the scintillator material is computed as a function of the
sample thickness.  A univariate spline fit is then calculated
between the calculated transmission and the sample thickness for the centerline
of the BM fan.  This is then used as a lookup table to convert sample 
transmission to sample thickness, as an alternative to Beer's law.
To correct for the dependence of the spectrum on vertical angle,
at a reference transmission (0.1 by default, which works well with the APS BM
beam), the ratio between sample thickness computed with the centerline spline
fit and the actual sample thickness is computed as a correction factor. 
A second spline fit between vertical angle and correction factor is calculated,
and this is used to correct the images for the dependence of the spectrum
on the vertical angle in the fan.  

This code uses a set of text data files to both define the spectral
properties of the beam and to define the absorption and attenuation
properties of various materials.  

The spectra are in text files with 
two columns.  The first column gives energy in eV, the second the spectral
power of the beam.  A series of files are used, in the form 
Psi_##urad.dat, with the ## corresponding to the vertical angle from the ring
plane in microradians.  These files were created in the BM spectrum
tool of XOP.

The spectral properties of the various filter, sample, and scintillator 
materials were computed in XOP with the xCrossSec tool.  To add a new
material, compute the spectral properties with xCrossSec and add the 
file to the beam_hardening_data folder.

This code also uses a setup.cfg file, located in beam_hardening_data.
This mainly gives the options for materials, their densities, and 
the reference transmission for the angular correction factor.

Usage:
* Run fread_config_file to load in configuration information.
* Run fcompute_calibrations to compute the polynomial fits for correcting
    the beam hardening
* Run either fcorrect_as_pathlength or fcorrect_as_transmission, as desired,
    to correct an image.

'''
from copy import deepcopy
from pathlib import Path, PurePath

import numpy as np
import scipy.interpolate
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline

from spectrum_filter import log
from spectrum_filter import config

#Global variables we need for computing LUT
filters = []
scintillator = {}
sample_material = None
possible_materials = {}

# Add a trailing slash if missing
data_path = Path(__file__).parent.joinpath('beam_hardening_data')
default_source = data_path.joinpath('Psi_00urad.dat')

#Global variables for when we convert images
input_spectrum = None
centerline_spline = None


class Spectrum:
    '''Class to hold the spectrum: energies and spectral power.
    '''
    def __init__(self, energies, spectral_power):
        if len(energies) != len(spectral_power):
            raise ValueError
        self.energies = energies
        self.spectral_power = spectral_power

    def fintegrated_power(self):
        return scipy.integrate.simps(self.spectral_power, self.energies)

    def fmean_energy(self):
        return scipy.integrate.simps(self.spectral_power * self.energies, self.energies) / self.fintegrated_power()
    
    def __len__(self):
        return len(energies)
 
#Copy part of the Material class from Scintillator_Optimization code
class Material:
    '''Class that defines the absorption and attenuation properties of a material.
    Data based off of the xCrossSec database in XOP 2.4.
    '''
    def __init__(self,name,density):
        self.name = name
        self.density = density  #in g/cc
        self.fread_absorption_data()
        self.absorption_interpolation_function = self.interp_function(self.energy_array,self.absorption_array)
        self.attenuation_interpolation_function = self.interp_function(self.energy_array,self.attenuation_array)
    
    def __repr__(self):
        return "Material({0:s}, {1:f})".format(self.name, self.density)
    
    def fread_absorption_data(self):

        raw_data = np.genfromtxt(data_path.joinpath(self.name + '_properties_xCrossSec.dat'))
        self.energy_array = raw_data[:,0] / 1000.0      #in keV
        self.absorption_array = raw_data[:,3]   #in cm^2/g, from XCOM in XOP
        self.attenuation_array = raw_data[:,7]  #in cm^2/g, from XCOM in XOP, ignoring coherent scattering
    
    def interp_function(self,energies,absorptions):
        '''Return a function to interpolate logs of energies into logs of absorptions.
        '''
        return scipy.interpolate.interp1d(np.log(energies),np.log(absorptions),bounds_error=False)
    
    def finterpolate_absorption(self,input_energies):
        '''Interpolates absorption on log-log scale and scales back
        '''
        return np.exp(self.absorption_interpolation_function(np.log(input_energies)))
    
    def finterpolate_attenuation(self,input_energies):
        '''Interpolates attenuation on log-log scale and scales back
        '''
        return np.exp(self.attenuation_interpolation_function(np.log(input_energies)))
    
    def fcompute_proj_density(self,thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        return thickness /1e4 * self.density
    
    def fcompute_transmitted_spectrum(self,thickness,input_spectrum):
        '''Computes the transmitted spectral power through a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident spectrum
        Output:
        Spectrum object for transmitted intensity
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        output_spectrum.spectral_power *= (
                    np.exp(-self.finterpolate_attenuation(output_spectrum.energies) * filter_proj_density))
        return output_spectrum
    

    def fcompute_absorbed_spectrum(self,thickness,input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident beam
        Output:
        Spectrum object for absorbed spectrum
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral absorption using Beer-Lambert law
        ext_lengths = self.finterpolate_absorption(input_spectrum.energies) * filter_proj_density 
        absorption = 1.0 - np.exp(-ext_lengths)
        output_spectrum.spectral_power = input_spectrum.spectral_power * absorption 
        return output_spectrum
    
    def fcompute_absorbed_power(self,thickness,input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: Spectrum object for incident beam 
        Output:
        absorbed power
        '''
        return self.fcompute_absorbed_spectrum(thickness,input_spectrum).fintegrated_power()


def fread_config_file(config_filename=None):
    '''Read in parameters for beam hardening corrections from file.
    Default file is in same directory as this source code.
    Users can input an alternative config file as needed.
    '''
    if config_filename:
        config_path = Path(config_filename)
        if not config_path.exists():
            raise IOError('Config file does not exist: ' + str(config_path))
    else:
        config_path = Path.joinpath(Path(__file__).parent, 'beam_hardening_data', 'setup.cfg')
    with open(config_path, 'r') as config_file:
        while True:
            line = config_file.readline()
            if line == '':
                break
            if line.startswith('#'):
                continue
            elif line.startswith('symbol'):
                symbol = line.split(',')[0].split('=')[1].strip()
                density = float(line.split(',')[1].split('=')[1])
                possible_materials[symbol] = Material(symbol, density)


def fread_source_data(source_file = None):
    '''Reads the spectral power data from files.
    Data file comes from the BM spectrum module in XOP.
    Return:
    Dictionary of spectra at the various psi angles from the ring plane.
    '''
    if not source_file:
        source_file = default_source
    else:
        source_file = Path(source_file)
    log.info('  *** *** source file {:s} located'.format(str(source_file)))
    spectral_data = np.genfromtxt(source_file, comments='!')
    spectral_energies = spectral_data[:,0] / 1000.
    spectral_power = spectral_data[:,1]
    global input_spectrum
    input_spectrum = Spectrum(spectral_energies, spectral_power)
    return input_spectrum


def check_material(material_str):
    '''Checks whether a material is in the list of possible materials.
    Input: string representing the symbol for a material.
    Output: Material object with the same name as the input symbol.
    '''
    for mat in possible_materials.values():
        if mat.name  == material_str:
            return mat
    else:
        raise ValueError('No such material in possible_materials: {0:s}'.format(material_str))


def add_filter(symbol, thickness, filters):
    '''Add a filter of a given symbol and thickness.
    '''
    if symbol != 'none' and thickness > 0:
        log.info('  *** *** material = {:s}'.format(symbol))
        log.info('  *** *** thickness = {:f} \u03bcm'.format(thickness))
        filters.append((check_material(symbol), thickness))
    return filters


def parse_params(params):
    """
    Parse the input parameters to fill in filters, sample material,
    and scintillator material and thickness.
    """
    log.info('  Reading config parameters')
    global scintillator
    scintillator['material'] = check_material(params.scint_material)
    scintillator['thickness'] = params.scint_thickness
    log.info('  *** scintillator')
    log.info('  *** *** material = {:s}'.format(scintillator['material'].name))
    log.info('  *** *** thickness = {:f} \u03bcm'.format(scintillator['thickness']))
    global filters
    log.info('  *** filters')
    filters = parse_filters(params)
    for i, (m, t) in enumerate(filters):
        log.info('  *** *** Filter {:d}'.format(i))
        log.info('  *** *** *** Material: {:s}'.format(m.name))
        log.info('  *** *** *** Thickness: {:.0f} \u03bcm'.format(t))
    global sample_material
    sample_material = check_material(params.sample_material)
    global max_sample_thickness
    max_sample_thickness = params.sample_max_thickness
    log.info('  *** sample')
    log.info('  *** *** material = {:s}'.format(sample_material.name))
    log.info('  *** *** max thickness = {:f}'.format(max_sample_thickness))
    log.info('  *** config done')


def parse_filters(params):
    '''Parses the filter material and filter thickness lists.
    Input: 
    params: parameters from config file or CLI arguments
    Output:
    list of tuples with form (filter material, thickness)
    '''
    if len(params.filter_matl_list) != len(params.filter_thick_list):
        log.error('  Mismatch in lengths of filter material and thickness inputs')
        return
    return [(check_material(m.strip()),t) for m, t in zip(params.filter_matl_list, params.filter_thick_list)]


def fapply_filters(filters, input_spectrum):
    '''Computes the spectrum after all filters.
        Inputs:
        filters: dictionary giving filter materials as keys and thicknesses in microns as values.
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        spectral power transmitted through the filter set.
        '''
    log.info('  Applying filters')
    temp_spectrum = deepcopy(input_spectrum)
    for i, (m,t) in enumerate(filters):
        log.info('  *** applying filter {:d}: {:s}, thickness {:.0f} \u03bcm'.format(
                    i, m.name, t))
        temp_spectrum = m.fcompute_transmitted_spectrum(t, temp_spectrum)
    return temp_spectrum


def find_detected_spectrum(input_spectrum):
    return scintillator['material'].fcompute_absorbed_spectrum(
                            scintillator['thickness'], input_spectrum)


def find_sample_trans(input_spectrum, sample_thickness):
    '''Find the sample transmission, mean energy transmitted through the sample,
    and the mean detected energy for a given thickness of sample.
    Inputs:
    input_spectrum: input Spectrum object with incident spectrum on sample
    sample_thickness: thickness of sample material in microns
    Outputs:
    transmission of sample, mean energy (in keV) of beam transmitted through
        the sample, and mean energy (in keV) of detected beam.
    '''
    clear_air_signal = scintillator['material'].fcompute_absorbed_power(
                            scintillator['thickness'], input_spectrum)
    sample_filt_spectrum = sample_material.fcompute_transmitted_spectrum(
                            sample_thickness, input_spectrum)
    detected_spectrum = find_detected_spectrum(sample_filt_spectrum)
    sample_trans = sample_filt_spectrum.fintegrated_power() / input_spectrum.fintegrated_power()    
    detected_trans = detected_spectrum.fintegrated_power() / clear_air_signal
    return sample_trans, detected_trans, sample_filt_spectrum.fmean_energy(), detected_spectrum.fmean_energy()   


def find_sample_calibration(input_spectrum):
    '''Makes a spline function to relate transmission to sample thickness.
    Also makes a spline function to relate sample thickness to effective energies.
    Input: Spectrum object
    Returns:
    InterpolatedUnivariateSpline object which takes transmission as input, 
        returning sample thickness in microns.
    '''
    #Make an array of sample thicknesses
    sample_thicknesses = np.linspace(0, max_sample_thickness, 101)
    #For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(sample_thicknesses)
    sample_eff_energy = np.zeros_like(sample_thicknesses)
    detected_eff_energy = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                              input_spectrum)
        sample_eff_energy[i] = sample_filtered_power.fmean_energy()
        detected_spectrum = scintillator['material'].fcompute_absorbed_spectrum(
                                                    scintillator['thickness'], sample_filtered_power)
        detected_eff_energy[i] = detected_spectrum.fmean_energy()
        detected_power[i] = detected_spectrum.fintegrated_power()

    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / detected_power[0]
    #Return splines, but make sure things are sorted in ascending order
    trans_spline = InterpolatedUnivariateSpline(sample_effective_trans, sample_thicknesses)
    return trans_spline

def fcorrect_as_pathlength_centerline(input_trans):
    """Corrects for the beam hardening, assuming we are in the ring plane.
    Input: transmission
    Output: sample pathlength in microns.
    """
    data_dtype = input_trans.dtype
    return_data = mproc.distribute_jobs(input_trans,centerline_spline,args=(),axis=1)
    return return_data
