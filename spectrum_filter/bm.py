import numpy as np
import scipy.special
import scipy.constants as constants

def _total_radiation(self, E, psi, critical_energy, gamma):
    '''Gives the emission for sigma + pi polarization as a function
    of energy E and vertical angle from the ring plane psi.
    '''
    return (_sigma_radiation(E, psi, critical_energy, gamma)
            + _pi_radiation(E, psi, critical_energy, gamma))

def _sigma_radiation(self, E, psi, critical_energy, gamma):
    '''Gives the emission for sigma + pi polarization as a function
    of energy E and vertical angle from the ring plane psi.
    '''
    E_ratio_term = (3. * E / (4.0 * critical_energy)).** (2. / 3.)
    airy = scipy.special.airy(E_ratio_term * (gamma ** 2 * psi ** 2 + 1))
    return 9. / (2. * np.pi) * E_ratio_term * airy[1] ** 2


def _pi_radiation(self, E, psi, critical_energy, gamma):
    '''Gives the emission for sigma + pi polarization as a function
    of energy E and vertical angle from the ring plane psi.
    '''
    E_ratio_term = (3. * E / (4.0 * critical_energy))** (2. / 3.)
    airy = scipy.special.airy(E_ratio_term * (gamma**2 * psi**2 + 1))
    return 9. / (2. * np.pi) * (E_ratio_term * gamma * psi) ** 2 * airy[0]**2


def calculate_source_data(self, kwargs):
    """Calculates BM radiation for various psi angles
    using first-principles for BM radiation.
    Equations taken from "Elements of Modern X-Ray Physics" 2nd
    Edition Chapter 2 and 
    A Hoffman "The Physics of Synchrotron Radiation" section 5.3.
    """
    self.spectra_dict = {}
    E_sr = float(kwargs['E_storage_ring'])
    B = float(kwargs['B_storage_ring'])
    gamma = E_sr * 1e9 * constants.elementary_charge / (constants.m_e * constants.c**2)
    radius = (gamma * constants.m_e * constants.c) / (constants.elementary_charge * B)
    critical_energy = (3 * gamma**3 * constants.c 
                        / (2 * radius) * constants.hbar / constants.elementary_charge)
    energies = np.arange(
                            kwargs['minimum_E'],
                            kwargs['maximum_E'] + kwargs['step_E'],
                            kwargs['step_E'],
                            )
    psi_values = np.linspace(
                            kwargs['minimum_psi_urad'],
                            kwargs['maximum_psi_urad'],
                            5,
                            )
    for i in psi_values:
        spectral_power = self._total_radiation(energies, i * 1e-6, critical_energy, gamma)
        self.spectra_dict[i] = Spectrum(energies, spectral_power)
