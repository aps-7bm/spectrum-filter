from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import tabulate
from spectrum_filter import beamhardening as bh
from spectrum_filter import log

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


def plot_sample_hardening(params):
    '''Plots the beam hardening caused by the sample.
    Plots the transmission, transmitted effective energy,
    and detected effective energy vs. sample thickness.
    All filters are applied first.
    '''
    input_spectrum = bh.fread_source_data(params.source_spectrum)
    bh.parse_params(params)
    log.info('  Plot the transmission and effective energy vs. sample thickness')
    log.info('  *** sample material = {:s}'.format(bh.sample_material.name))
    log.info('  *** maximum thickness = {:0f}'.format(bh.max_sample_thickness))
    filtered_spectrum = bh.fapply_filters(bh.filters, input_spectrum)
    sample_thicknesses = np.linspace(0, params.sample_max_thickness, 101)
    sample_trans = np.zeros_like(sample_thicknesses)
    sample_eff_trans = np.zeros_like(sample_trans)
    sample_eff_energy = np.zeros_like(sample_thicknesses)
    det_eff_energy = np.zeros_like(sample_thicknesses)
    for i, t in enumerate(sample_thicknesses):
        sample_trans[i], sample_eff_trans[i], sample_eff_energy[i], det_eff_energy[i] = bh.find_sample_trans(filtered_spectrum, t)
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
    input_spectrum = bh.fread_source_data(params.source_spectrum)
    bh.parse_params(params)
    log.info('  Plot the cumulative effect of filters')
    #Plot the input spectrum
    plt.figure(1, figsize=[6.4, 4.8])
    plt.plot(input_spectrum.energies, input_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(2, figsize=[6.4, 4.8])
    det_spectrum = bh.find_detected_spectrum(input_spectrum)
    plt.plot(det_spectrum.energies, det_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(3, figsize=[6.4, 4.8])
    plt.plot(det_spectrum.energies, 
                det_spectrum.spectral_power / np.max(det_spectrum.spectral_power), 
                '--', label = 'Incident')
    plt.figure(4, figsize=[6.4, 4.8])
    filt_spectrum = deepcopy(input_spectrum)
    trans_spectrum = deepcopy(filt_spectrum)
    trans_spectrum.spectral_power = 1.0
    table_labels = ['Filter', 'Thickness(um)', 'Mean E (keV)', 'Eff. E (keV)', 'True trans', 'Eff. trans']
    filt_power_in = filt_spectrum.fintegrated_power()
    det_power_in = det_spectrum.fintegrated_power()
    table_row = ['Incident',
                0, 
                filt_spectrum.fmean_energy(),
                det_spectrum.fmean_energy(),
                1.0,
                1.0,]
    table_input = [table_row]
    for (m,t) in bh.filters:
        if t == 0:
            continue
        this_filter = [(m, t)]
        filt_spectrum = bh.fapply_filters(this_filter, filt_spectrum)
        det_spectrum = bh.find_detected_spectrum(filt_spectrum)
        table_row = [m.name, 
                t,
                filt_spectrum.fmean_energy(),
                det_spectrum.fmean_energy(),
                filt_spectrum.fintegrated_power() / filt_power_in,
                det_spectrum.fintegrated_power() / det_power_in,]
        plt.figure(1)
        plot_label = '{0:s}: {1:.0f} \u03bcm'.format(m.name, t)
        plt.plot(filt_spectrum.energies, filt_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(2)
        plt.plot(det_spectrum.energies, det_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(3)
        plt.plot(det_spectrum.energies, 
                    det_spectrum.spectral_power / np.max(det_spectrum.spectral_power),
                    '-', label = plot_label)
        plt.figure(4)
        trans_spectrum = m.fcompute_transmitted_spectrum(t, trans_spectrum)
        plt.plot(trans_spectrum.energies, trans_spectrum.spectral_power, label = 'After ' + plot_label)
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
                bh.scintillator['thickness'], bh.scintillator['material'].name), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(3)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Norm. spectrum detected by {:.0f} \u03bcm {:s}'.format(
                bh.scintillator['thickness'], bh.scintillator['material'].name), fontsize = 10)
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
    input_spectrum = bh.fread_source_data(params.source_spectrum)
    bh.parse_params(params)
    log.info('  Plot the individual effect of filters in a set.')
    #Plot the input spectrum
    plt.figure(1, figsize=[6.4, 4.8])
    plt.plot(input_spectrum.energies, input_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(2, figsize=[6.4, 4.8])
    det_spectrum = bh.find_detected_spectrum(input_spectrum)
    plt.plot(det_spectrum.energies, det_spectrum.spectral_power, '--', label = 'Incident')
    plt.figure(3, figsize=[6.4, 4.8])
    plt.plot(det_spectrum.energies, 
                det_spectrum.spectral_power / np.max(det_spectrum.spectral_power), 
                '--', label = 'Incident')
    plt.figure(4, figsize=[6.4, 4.8])
    filt_spectrum = deepcopy(input_spectrum)
    trans_spectrum = deepcopy(filt_spectrum)
    trans_spectrum.spectral_power = 1.0
    table_labels = ['Filter', 'Thickness(um)', 'Mean E (keV)', 'Eff. E (keV)', 'True trans', 'Eff. trans']
    filt_power_in = filt_spectrum.fintegrated_power()
    det_power_in = det_spectrum.fintegrated_power()
    table_row = ['Incident',
                0, 
                filt_spectrum.fmean_energy(),
                det_spectrum.fmean_energy(),
                1.0,
                1.0,]
    table_input = [table_row]
    for (m,t) in bh.filters:
        if t == 0:
            continue
        this_filter = [(m, t)]
        filt_spectrum = bh.fapply_filters(this_filter, input_spectrum)
        det_spectrum = bh.find_detected_spectrum(filt_spectrum)
        table_row = [m.name, 
                t,
                filt_spectrum.fmean_energy(),
                det_spectrum.fmean_energy(),
                filt_spectrum.fintegrated_power() / filt_power_in,
                det_spectrum.fintegrated_power() / det_power_in,]
        plt.figure(1)
        plot_label = '{0:s}: {1:.0f} \u03bcm'.format(m.name, t)
        plt.plot(filt_spectrum.energies, filt_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(2)
        plt.plot(det_spectrum.energies, det_spectrum.spectral_power, '-', label = plot_label)
        plt.figure(3)
        plt.plot(det_spectrum.energies, 
                    det_spectrum.spectral_power / np.max(det_spectrum.spectral_power),
                    '-', label = plot_label)
        plt.figure(4)
        temp_spectrum = m.fcompute_transmitted_spectrum(t, trans_spectrum)
        plt.plot(trans_spectrum.energies, temp_spectrum.spectral_power, label = plot_label)
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
                bh.scintillator['thickness'], bh.scintillator['material'].name), fontsize = 10)
    plt.yticks(fontsize=10)
    plt.figure(3)
    plt.xlabel('Energy, keV')
    plt.legend(loc = 'upper right', fontsize = 8)
    plt.title('Norm spectrum detected by {:.0f} \u03bcm {:s}'.format(
                bh.scintillator['thickness'], bh.scintillator['material'].name), fontsize = 10)
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
    input_spectrum = bh.fread_source_data(params.source_spectrum)
    bh.parse_params(params)
    print(bh.filters)
    filtered_spectrum = bh.fapply_filters(bh.filters, input_spectrum)
    input_power = filtered_spectrum.fintegrated_power()
    log.info('  Plot the transmission and effective energy vs. scintillator thickness')
    log.info('  *** scintillator material = {:s}'.format(bh.scintillator['material'].name))
    log.info('  *** maximum thickness = {:0f}'.format(bh.scintillator['thickness']))
    #Set up plots
    fig1, ax1 = plt.subplots()
    ax1.set_position([0.15, 0.25, 0.65, 0.65])
    plt.title('Light output for {:s}'.format(bh.scintillator['material'].name))
    ax1r = ax1.twinx()
    ax1r.set_position([0.15, 0.25, 0.65, 0.65])
    fig2, ax2 = plt.subplots()
    ax2.set_position([0.15, 0.25, 0.75, 0.65])
    plt.title('Mean Detected Energy')
    scint_thicknesses = np.linspace(0, bh.scintillator['thickness'], 101)
    marginal_det_spectra = []
    cum_det_spectra = []
    trans_spectra = [filtered_spectrum]
    marginal_det_energy = []
    cum_det_energy = []
    marginal_power = []
    cum_power = []
    delta_scint = scint_thicknesses[1] - scint_thicknesses[0]
    for i, t in enumerate(scint_thicknesses):
        marginal_det_spectra.append(bh.scintillator['material'].fcompute_absorbed_spectrum(delta_scint, trans_spectra[-1]))
        marginal_det_energy.append(marginal_det_spectra[-1].fmean_energy())
        marginal_power.append(marginal_det_spectra[-1].fintegrated_power())
        trans_spectra.append(bh.scintillator['material'].fcompute_transmitted_spectrum(delta_scint, trans_spectra[-1]))
        if i == 0:
            cum_det_spectra.append(marginal_det_spectra[-1])
        else:
            new_cum_det_spectrum = deepcopy(cum_det_spectra[-1])
            new_cum_det_spectrum.spectral_power += marginal_det_spectra[-1].spectral_power
            cum_det_spectra.append(new_cum_det_spectrum)
        cum_det_energy.append(cum_det_spectra[-1].fmean_energy())
        cum_power.append(cum_det_spectra[-1].fintegrated_power())
    ax1.plot(scint_thicknesses, cum_power / input_power, 'r.-', label = 'Cumulative')
    ax1r.plot(scint_thicknesses, marginal_power / input_power, 'g.-', label = 'Marginal')
    ax1.set_xlabel('Scintillator thickness, \u03bcm')
    ax1.set_ylabel('Cumulative Absorption', color='r')
    ax1r.set_ylabel('Absorption per {:.1f} \u03bcm'.format(delta_scint), color='g')
    ax1r.grid(False)
    ax1.set_xlim(0, np.max(scint_thicknesses) * 1.1)
    ax1.set_ylim(0, np.max(cum_power / input_power)* 1.1)
    ax1r.set_ylim(0, np.max(marginal_power / input_power) * 1.1)
    ax2.plot(scint_thicknesses, cum_det_energy, 'r.-', label = 'Cumulative')
    ax2.plot(scint_thicknesses, marginal_det_energy, 'g.-', label = 'Marginal')
    ax2.set_xlabel('Scintillator thickness, \u03bcm')
    ax2.set_xlim(0, np.max(scint_thicknesses) * 1.1)
    ax2.set_ylim(0, np.max(marginal_det_energy) * 1.1)
    ax2.set_ylabel('Mean Detected Energy, keV')
    ax2.legend(loc = 'lower right')
    plt.show()
