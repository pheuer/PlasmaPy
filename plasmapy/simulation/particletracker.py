"""
Class representing a group of particles.
"""
__all__ = ["ParticleTracker"]

import astropy.units as u
import numpy as np
import scipy.interpolate as interp

import astropy.constants as const

from plasmapy.particles import atomic
from plasmapy.utils.decorators import validate_quantities


class ParticleSpecies:
    """
    Object representing a number of macroparticles of the same species
    to enter into ParticleTracker

    Parameters
    ----------

    particle_type : `Particle` instance or str
        The type of particle that makes up this species

    position : `u.Quantity` array of shape (N,3)
        Initial positions of each particle

    velocity : `u.Quantity` array of shape (N,3)
        Initial velocity of each particle

    """

    def __init__(
        self,
        particle_type,
        position,
        velocity,
        ):

        if not isinstance(particle, Particle):
            self.particle = Particle(particle_type)
        else:
            self.particle = particle_type


        if position.shape != velocity.shape:
            raise ValueError("Position and Velocity arrays must have the "
                             "same shape, but entered array shapes are "
                             f" {position.shape} and {velocity.shape}")

        self.x = position
        self.v = velocity

        # Number of particles
        self.N = self.x.shape[0]
        # Particle charge
        self.q = self.particle.charge
        # Particle mass
        self.m = self.particle.mass



# TODO: Have the option

class ParticleTracker:
    """
    Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular initial velocity distribution.

    Parameters
    ----------
    plasma : 'Plasma3D' object or list of same
        A list of 'Plasma3D' objects, each of which contains a grid and
        one or more quantities on the grid.

        This allows different quantities (eg. E and B) to be defined on
        different grids.

    species : 'Species' object or list of same
        A list of `Species` objects, each of which describes a number of
        particles to be tracked.

    dt : `astropy.units.Quantity`
        length of timestep

    end_condition : Integer or a function that takes a ParticleTracker as input
        Either an integer number of iterations or a function that takes
        the ParticleTracker as input and returns a boolean. If provided,
        this function is evalulated every iteration and the tracker stops
        once the function returns True.

    recorded_variables  : List of strings
        A list of strings representing properties either of the particles
        or defined on the grid. If this keyword is set, at each timestep
        the value of each of these properties will be recorded for each
        particle (grid values are interpolated to particle positons).

        If record_file is set, the data is saved there. Otherwise it will be
        stored in memory.

    record_file : path string
        A path for an HDF5 file where particle histories will be saved. If
        None, no histories will be saved.


    Examples
    ----------
    See `Particle Stepper Notebook`_.

    .. _`Particle Stepper Notebook`: ../notebooks/particle_stepper.ipynb
    """

    @validate_quantities(dt=u.s)
    def __init__(
        self,
        plasma,
        species,
        dt=None,
        end_condition = 100,
        record_file = None,
        ):


        # Create a dictionary that links quantities on each grid to the
        # grid they come from.
        self.quantities = {}
        for p in plasma:

            # TODO: Somehow deal with the possibility of multiple grids
            # having the same quantity (which to choose?)

            for quantity_key in p.quantities.keys():
                self.quantities[quantity_key] = p.interpolator

        self.species = species


        self.end_condition = end_condition

        # Counter for the number of interations
        self.iterations = 0





    def _field_interpolators(self):
        """
        Interpolates E and B fields required for any charged particle pusher
        """

        # fetch references to the E and B interpolators within the Plasma3D
        # object(s)

        return b_interpolator, e_interpolator



    def boris_push(self, init=False):
        r"""
        Implements the Boris algorithm for moving particles and updating their
        velocities.

        Arguments
        ----------
        init : bool (optional)
            If `True`, does not change the particle positions and sets dt
            to -dt/2.

        Notes
        ----------
        The Boris algorithm is the standard energy conserving algorithm for
        particle movement in plasma physics. See [1]_ for more details.

        Conceptually, the algorithm has three phases:

        1. Add half the impulse from electric field.
        2. Rotate the particle velocity about the direction of the magnetic
           field.
        3. Add the second half of the impulse from the electric field.

        This ends up causing the magnetic field action to be properly
        "centered" in time, and the algorithm conserves energy.

        References
        ----------
        .. [1] C. K. Birdsall, A. B. Langdon, "Plasma Physics via Computer
               Simulation", 2004, p. 58-63
        """

        b_interpolator, e_interpolator = self._field_interpolators()

        # TODO: Replace this with a function that calculates an adaptive dt
        dt = -self.dt / 2 if init else self.dt

        # Apply pusher to each species
        for s in species:

            # Interpolate the E and B fields at each particle position
            e = e_interpolator(s.x)
            b = b_interpolator(s.x)


            # add first half of electric impulse
            vminus = s.v + s.q / s.m * dt * 0.5

            # rotate to add magnetic field
            t = -b * s.q / s.m * dt * 0.5
            u = 2 * t / (1 + (t * t).sum(axis=1, keepdims=True))
            vprime = vminus + np.cross(vminus.si.value, t)
            vplus = vminus + np.cross(vprime.si.value, u)

            # add second half of electric impulse
            v_new = vplus + s.q * e / s.m * dt * 0.5

            s.x += s.v * dt



    # TODO: Add relativistically correct Boris push algorithm here


    def run(self):
        r"""
        Runs a simulation instance.
        """

        while not _end(self):
            # Push particles
            self.boris_push()

            # Record variables when required
            if len(self.recorded_variables) != 0:
                self._record_variables()

            # Increment
            self.iterations += 1


    def _end(self):

        if isinstance(self.end_condition, int):
            if self.iterations >= self.end_condition:
                return True
            else:
                return False

        # If not an integer, end_condition is a function
        return self.end_condition(self)



    def _record_variables(self):
        """
        Records the current particle positions and velocities in memory or
        and HDF5 file.
        """







# ****************************************************************************
# Functions that accept a ParticleTracker and make basic plots
# ****************************************************************************



def plot_trajectories(particle_tracker):  # coverage: ignore
    r"""Draws trajectory history."""
    import matplotlib.pyplot as plt

    from astropy.visualization import quantity_support
    from mpl_toolkits.mplot3d import Axes3D

    quantity_support()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for p_index in range(self.N):
        r = self.position_history[:, p_index]
        x, y, z = r.T
        ax.plot(x, y, z)
    ax.set_title(self.name)
    ax.set_xlabel("$x$ position")
    ax.set_ylabel("$y$ position")
    ax.set_zlabel("$z$ position")
    plt.show()

def plot_time_trajectories(particle_tracker, plot="xyz"):  # coverage: ignore
    r"""
    Draws position history versus time.

    Parameters
    ----------
    plot : str (optional)
        Enable plotting of position component x, y, z for each of these
        letters included in `plot`.
    """
    import matplotlib.pyplot as plt

    from astropy.visualization import quantity_support
    from mpl_toolkits.mplot3d import Axes3D

    quantity_support()
    fig, ax = plt.subplots()
    for p_index in range(self.N):
        r = self.position_history[:, p_index]
        x, y, z = r.T
        if "x" in plot:
            ax.plot(self.t, x, label=f"x_{p_index}")
        if "y" in plot:
            ax.plot(self.t, y, label=f"y_{p_index}")
        if "z" in plot:
            ax.plot(self.t, z, label=f"z_{p_index}")
    ax.set_title(self.name)
    ax.legend(loc="best")
    ax.grid()
    plt.show()


def kinetic_energy_history(particle_tracker):
    r"""
    Calculates the kinetic energy history for each particle.

    Returns
    --------
    ~astropy.units.Quantity
        Array of kinetic energies, shape (nt, n).
    """
    return (self.velocity_history ** 2).sum(axis=-1) * self.eff_m / 2
