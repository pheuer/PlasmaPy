import astropy.units as u
import astropy.constants as const
import numpy as np
import pytest

import plasmapy.formulary.radiation as radiation
from plasmapy.utils.exceptions import PhysicsError


def test_thermal_bremsstrahlung():
    # Test correct spectrum created
    frequencies = (10 ** np.arange(15, 16, 0.01)) / u.s
    ne, Te = 1e22 * u.cm ** -3, 1e2 * u.eV
    ion_species = "H+"

    spectrum = radiation.thermal_bremsstrahlung(frequencies, ne, Te,
                                                ion_species=ion_species)

    assert np.isclose(np.max(spectrum).value, 128.4, 1), (
        f"Spectrum maximum is {np.max(spectrum).value} "
        "instead of expected value 128.4"
    )

    # Test violates w > wpe limit
    small_frequencies = (10 ** np.arange(12, 16, 0.01)) / u.s
    with pytest.raises(PhysicsError):
        spectrum = radiation.thermal_bremsstrahlung(
            small_frequencies, ne, Te, ion_species=ion_species
        )

    # Test violates Rayleigh-Jeans limit
    small_Te = 1 * u.eV
    with pytest.raises(PhysicsError):
        spectrum = radiation.thermal_bremsstrahlung(
            frequencies, ne, small_Te, ion_species=ion_species
        )


def test_electron_ion_bremsstrahlung():
    frequencies = (10 ** np.arange(15, 16, 0.01)) / u.s
    v_e = 7e6*u.m/u.s
    n_i = 1e22*u.cm**-3
    ion_species = 'H+'


    test_gff = np.sqrt(3)/np.pi*np.log(const.m_e.cgs*v_e**3/
                                             (np.pi*1*const.e.gauss**2*frequencies))

    spectrum = radiation.electron_ion_bremsstrahlung(frequencies, v_e, n_i,
                                                     ion_species,
                                                     Gaunt_factor='classical')

# Note that it is not really possible to compare the thermal_bremsstrahlung
# and electron_ion_bremsstrahlung functions against one another, because
# the latter only works for fast particles, while a thermal distribution
# inherently has many slow particles.

def test_Bethe_Heitler_formula():

    energies = np.arange(10, 1000)*u.eV

    v_e = 7e6*u.m/u.s
    n_i = 1e22*u.cm**-3
    ion_species = 'H+'

    spectrum = radiation.Bethe_Heitler_formula(energies, v_e, n_i,
                                               ion_species=ion_species)
test_Bethe_Heitler_formula()