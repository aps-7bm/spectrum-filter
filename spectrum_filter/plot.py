from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import tabulate
from spectrum_filter import beamhardening as bh
from spectrum_filter import log
from beamhardening import beamhardening as bh1
#Initialize the beamhardening code
bh.fread_config_file()


def list_materials(params):
    '''List the material options for filters and scintillators.
    '''
    log.info('  List of material options for filters')
    for v in bh.possible_materials.values():
        log.info('  *** {:s}: density = {:.2f} g/cc'.format(
                    v.name, v.density))
    log.info('  *** materials list done')


def initialize_source(params):
    '''Initializes the spectral calculations.
    This sets up a beamhardening.BeamCorrector object, including reading
    the source spectrum.
    '''
    bc = bh1.BeamCorrector(calculate_source = params.calculate_source,
                            e_storage_ring = params.e_storage_ring,
                            b_storage_ring = params.b_storage_ring,
                            minimum_E = params.minimum_E,
                            maximum_E = params.maximum_E,
                            step_E = params.step_E,
                            )
    return bc


def add_filters(beamcorrector, params):
    '''Parses the filters as given in the input
    and traslates them to the form the beamhardening
    code wants.
    '''
    input_materials = params.filter_matl_list
    input_thicks = params.filter_thick_list
    for mat, thick in zip(input_materials, input_thicks):
        beamcorrector.add_filter(mat, thick, None)


def add_scintillator(beamcorrector, params):
    '''Parses the scintillator parameters in the input
    and translates them for the beamhardening code.
    '''
    if params.scint_density > 0.0:
        beamcorrector.add_scintillator(params.scint_material, params.scint_thickness,
                                        params.scint_density)
    else:
        beamcorrector.add_scintillator(params.scint_material, params.scint_thickness)


def add_sample(beamcorrector, params):
    '''Parses the sample parameters in the input
    and translates them for the beamhardening code.
    '''
    beamcorrector.add_sample(params.sample_material)


def plot_sample_hardening(params):
    '''Plots the beam hardening caused by the sample.
    Plots the transmission, transmitted effective energy,
    and detected effective energy vs. sample thickness.
    All filters are applied first.
    '''
    bc = initialize_source(params)
    add_filters(bc, params)
    add_scintillator(bc, params)
    add_sample(bc, params)
    source_spectrum = bc.centerline_spectrum()
    log.info('  Plot the transmission and effective energy vs. sample thickness')
    log.info('  *** sample material = {:s}'.format(bc.sample_material.name))
    log.info('  *** maximum thickness = {:0f}'.format(params.sample_max_thickness))
    sample_thicknesses = np.linspace(0, params.sample_max_thickness, 101)
    filtered_spectrum = bc.apply_filters(source_spectrum)
    sample_trans = np.zeros_like(sample_thicknesses)
    sample_eff_trans = np.zeros_like(sample_trans)
    sample_eff_energy = np.zeros_like(sample_thicknesses)
    det_eff_energy = np.zeros_like(sample_thicknesses)
    
    #Calculate the attenuation by various thicknesses of sample
    for i in range(sample_thicknesses.shape[0]):
        temp = bc.sample_material.compute_transmitted_spectrum(sample_thicknesses[i], filtered_spectrum)
        sample_trans[i] = temp.integrated_power()
        sample_eff_energy[i] = temp.mean_energy()
        temp_scint = bc.scintillator_material.compute_absorbed_spectrum(bc.scintillator_thickness, temp)
        sample_eff_trans[i] = temp_scint.integrated_power()
        det_eff_energy[i] = temp_scint.mean_energy()
    sample_trans /= sample_trans[0]
    sample_eff_trans /= sample_eff_trans[0]
    sample_eff_energy /= 1e3
    det_eff_energy /= 1e3 

    #Plot the results
    fig1, ax1 = plt.subplots()
    ax1.set_position([0.20, 0.25, 0.7, 0.65])
    ax1.plot(sample_thicknesses, sample_trans, 'r.-', label = 'From sample')
    ax1.plot(sample_thicknesses, sample_eff_trans, 'g.-', label='Detected')
    ax1.set_xlabel(r'Sample Thickness, $\mu$m')
    ax1.set_ylabel('Transmission')
    ax1.set_xlim(0, params.sample_max_thickness)
    ax1.legend(loc = 'upper right', fontsize=6)
    ax1.set_title('Transmission of {:s}'.format(params.sample_material))
    fig2, ax2 = plt.subplots()
    ax2.set_position([0.15, 0.25, 0.75, 0.65])
    ax2.plot(sample_thicknesses, sample_eff_energy, 'r.-', label = 'From sample')
    ax2.plot(sample_thicknesses, det_eff_energy, 'g.-', label = 'Detected')
    ax2.set_xlabel(r'Sample Thickness, $\mu$m')
    ax2.set_ylabel('Effective Energy, keV')
    ax2.legend(loc = 'lower right', fontsize = 6)
    ax2.set_xlim(0, params.sample_max_thickness)
    ax2.set_ylim(0, np.max(sample_eff_energy) * 1.1)
    ax2.set_title('Eff. Energy through {:s}'.format(params.sample_material))
    plt.show()


def plot_filter_effect(params):
    '''Plots the cumulative effect of filtering.
    '''
    log.info('  Plot the cumulative effect of filters')
    bc = initialize_source(params)
    add_filters(bc, params)
    add_scintillator(bc, params)
    input_spectrum = bc.centerline_spectrum()

    #Plot the input spectrum
    plt.figure(1, figsize=[6.4, 4.8])
    plt.plot(input_spectrum.energies / 1e3, input_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(2, figsize=[6.4, 4.8])
    det_spectrum = bc.find_detected_spectrum(input_spectrum)
    plt.plot(det_spectrum.energies / 1e3, det_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(3, figsize=[6.4, 4.8])
    plt.plot(det_spectrum.energies / 1e3, 
                det_spectrum.spectral_power / np.max(det_spectrum.spectral_power), 
                '--', label = 'Incident')
    plt.figure(4, figsize=[6.4, 4.8])
    filt_spectrum = deepcopy(input_spectrum)
    trans_spectrum = deepcopy(filt_spectrum)
    trans_spectrum.spectral_power = 1.0
    table_labels = ['Filter', 'Thickness(um)', 'Mean E (keV)', 'Eff. E (keV)', 'True trans', 'Eff. trans']
    filt_power_in = filt_spectrum.integrated_power()
    det_power_in = det_spectrum.integrated_power()
    table_row = ['Incident',
                0, 
                filt_spectrum.mean_energy() / 1e3,
                det_spectrum.mean_energy() / 1e3,
                1.0,
                1.0,]
    table_input = [table_row]
    for (m,t) in bc.filters.items():
        if t == 0:
            continue
        this_filter = [(m, t)]
        filt_spectrum = m.compute_transmitted_spectrum(t, filt_spectrum)
        det_spectrum = bc.find_detected_spectrum(filt_spectrum)
        table_row = [m.name, 
                t,
                filt_spectrum.mean_energy() / 1e3,
                det_spectrum.mean_energy() / 1e3,
                filt_spectrum.integrated_power() / filt_power_in,
                det_spectrum.integrated_power() / det_power_in,]
        plt.figure(1)
        plot_label = '{0:s}: {1:.0f} \u03bcm'.format(m.name, t)
        plt.plot(filt_spectrum.energies / 1e3, filt_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(2)
        plt.plot(det_spectrum.energies / 1e3, det_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(3)
        plt.plot(det_spectrum.energies / 1e3, 
                    det_spectrum.spectral_power / np.max(det_spectrum.spectral_power),
                    '-', label = plot_label)
        plt.figure(4)
        trans_spectrum = m.compute_transmitted_spectrum(t, trans_spectrum)
        plt.plot(trans_spectrum.energies / 1e3, trans_spectrum.spectral_power, label = 'After ' + plot_label)
        table_input.append(table_row)
    log.info(tabulate.tabulate(table_input, table_labels, 
                                floatfmt=("s", ".0f", ".2f", ".2f", ".4f", ".4f"),
                                tablefmt='grid'))
    plt.figure(1)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Spectrum through filter', fontsize = 12)
    plt.yticks(fontsize=10)
    plt.figure(2)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Spectrum detected by {:.0f} \u03bcm {:s}'.format(
                params.scint_thickness, params.scint_material), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(3)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Norm. spectrum detected by {:.0f} \u03bcm {:s}'.format(
                params.scint_thickness, params.scint_material), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(4)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'lower right', fontsize = 8)
    plt.title('Cumulative filter transmission', fontsize = 12)
    plt.yticks(fontsize=10)
    plt.show()


def plot_filter_set(params):
    '''Plots the individual effect of a set of filters.
    '''
    bc = initialize_source(params)
    add_filters(bc, params)
    add_scintillator(bc, params)
    input_spectrum = bc.centerline_spectrum()
    log.info('  Plot the individual effect of filters in a set.')
    #Plot the input spectrum
    plt.figure(1, figsize=[6.4, 4.8])
    plt.plot(input_spectrum.energies / 1e3, input_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(2, figsize=[6.4, 4.8])
    det_spectrum = bc.find_detected_spectrum(input_spectrum)
    plt.plot(det_spectrum.energies / 1e3, det_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(3, figsize=[6.4, 4.8])
    plt.plot(det_spectrum.energies / 1e3, 
                det_spectrum.spectral_power / np.max(det_spectrum.spectral_power), 
                '--', label = 'Incident')
    plt.figure(4, figsize=[6.4, 4.8])
    filt_spectrum = deepcopy(input_spectrum)
    trans_spectrum = deepcopy(filt_spectrum)
    trans_spectrum.spectral_power = 1.0
    table_labels = ['Filter', 'Thickness(um)', 'Mean E (keV)', 'Eff. E (keV)', 'True trans', 'Eff. trans']
    filt_power_in = filt_spectrum.integrated_power()
    det_power_in = det_spectrum.integrated_power()
    table_row = ['Incident',
                0, 
                filt_spectrum.mean_energy() / 1e3,
                det_spectrum.mean_energy() / 1e3,
                1.0,
                1.0,]
    table_input = [table_row]
    for (m,t) in bc.filters.items():
        if t == 0:
            continue
        this_filter = [(m, t)]
        filt_spectrum = m.compute_transmitted_spectrum(t, input_spectrum)
        det_spectrum = bc.find_detected_spectrum(filt_spectrum)
        table_row = [m.name, 
                t,
                filt_spectrum.mean_energy() / 1e3,
                det_spectrum.mean_energy() / 1e3,
                filt_spectrum.integrated_power() / filt_power_in,
                det_spectrum.integrated_power() / det_power_in,]
        plt.figure(1)
        plot_label = '{0:s}: {1:.0f} \u03bcm'.format(m.name, t)
        plt.plot(filt_spectrum.energies / 1e3, filt_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(2)
        plt.plot(det_spectrum.energies / 1e3, det_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(3)
        plt.plot(det_spectrum.energies / 1e3, 
                    det_spectrum.spectral_power / np.max(det_spectrum.spectral_power),
                    '-', label = plot_label)
        plt.figure(4)
        temp_spectrum = m.compute_transmitted_spectrum(t, trans_spectrum)
        plt.plot(trans_spectrum.energies / 1e3, temp_spectrum.spectral_power, label = plot_label)
        table_input.append(table_row)
    log.info(tabulate.tabulate(table_input, table_labels, 
                                floatfmt=("s", ".0f", ".2f", ".2f", ".4f", ".4f"),
                                tablefmt='grid'))
    plt.figure(1)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Spectrum through filter', fontsize = 12)
    plt.yticks(fontsize=10)
    plt.figure(2)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Spectrum detected by {:.0f} \u03bcm {:s}'.format(
                bc.scintillator_thickness, bc.scintillator_material.name), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(3)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Norm spectrum detected by {:.0f} \u03bcm {:s}'.format(
                bc.scintillator_thickness, bc.scintillator_material.name), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(4)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'lower right', fontsize = 8)
    plt.title('Individual filter transmission', fontsize = 12)
    plt.yticks(fontsize=10)
    plt.show()


def plot_scintillator_thickness_effect(params):
    '''Plot the effect of varying scintillator thickness.
    Plots:
    1. Total and marginal light output vs. depth into scintillator
    2. Effective total detected energy vs. depth into scintillator
    3. Marginal light output vs. depth into scintillator
    4. Marginal detected energy vs. depth
    ''' 
    bc = initialize_source(params)
    add_filters(bc, params)
    add_scintillator(bc, params)
    input_spectrum = bc.centerline_spectrum()
    filtered_spectrum = bc.apply_filters(input_spectrum)
    input_power = filtered_spectrum.integrated_power()
    log.info('  Plot the transmission and effective energy vs. scintillator thickness')
    log.info('  *** scintillator material = {:s}'.format(bc.scintillator_material.name))
    log.info('  *** maximum thickness = {:0f}'.format(bc.scintillator_thickness))
    #Set up plots
    fig1, ax1 = plt.subplots()
    ax1.set_position([0.20, 0.25, 0.55, 0.65])
    plt.title('Light output for {:s}'.format(bc.scintillator_material.name))
    ax1r = ax1.twinx()
    ax1r.set_position([0.20, 0.25, 0.55, 0.65])
    fig2, ax2 = plt.subplots()
    ax2.set_position([0.15, 0.25, 0.75, 0.65])
    plt.title('Mean Detected Energy')
    scint_thicknesses = np.linspace(0, bc.scintillator_thickness, 101)
    marginal_det_spectra = []
    cum_det_spectra = []
    trans_spectra = [filtered_spectrum]
    marginal_det_energy = []
    cum_det_energy = []
    marginal_power = []
    cum_power = []
    delta_scint = scint_thicknesses[1] - scint_thicknesses[0]
    for i, t in enumerate(scint_thicknesses):
        marginal_det_spectra.append(bc.scintillator_material.compute_absorbed_spectrum(delta_scint, trans_spectra[-1]))
        marginal_det_energy.append(marginal_det_spectra[-1].mean_energy())
        marginal_power.append(marginal_det_spectra[-1].integrated_power())
        trans_spectra.append(bc.scintillator_material.compute_transmitted_spectrum(delta_scint, trans_spectra[-1]))
        if i == 0:
            cum_det_spectra.append(marginal_det_spectra[-1])
        else:
            new_cum_det_spectrum = deepcopy(cum_det_spectra[-1])
            new_cum_det_spectrum.spectral_power += marginal_det_spectra[-1].spectral_power
            cum_det_spectra.append(new_cum_det_spectrum)
        cum_det_energy.append(cum_det_spectra[-1].mean_energy())
        cum_power.append(cum_det_spectra[-1].integrated_power())
    ax1.plot(scint_thicknesses, cum_power / input_power, 'r.-', label = 'Cumulative')
    ax1r.plot(scint_thicknesses, marginal_power / input_power, 'g.-', label = 'Marginal')
    ax1.set_xlabel('Scintillator thickness, \u03bcm')
    ax1.set_ylabel('Cumulative Absorption', color='r')
    ax1r.set_ylabel('Absorption per {:.1f} \u03bcm'.format(delta_scint), color='g')
    ax1r.grid(False)
    ax1.set_xlim(0, np.max(scint_thicknesses) * 1.1)
    ax1.set_ylim(0, np.max(cum_power / input_power)* 1.1)
    ax1r.set_ylim(0, np.max(marginal_power / input_power) * 1.1)
    ax2.plot(scint_thicknesses, np.array(cum_det_energy) / 1e3, 'r.-', label = 'Cumulative')
    ax2.plot(scint_thicknesses, np.array(marginal_det_energy) / 1e3, 'g.-', label = 'Marginal')
    ax2.set_xlabel('Scintillator thickness, \u03bcm')
    ax2.set_xlim(0, np.max(scint_thicknesses) * 1.1)
    ax2.set_ylim(0, np.max(marginal_det_energy) * 1.1 / 1e3)
    ax2.set_ylabel('Mean Detected Energy, keV')
    ax2.legend(loc = 'lower right')
    plt.show()
