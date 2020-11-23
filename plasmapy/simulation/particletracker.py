"""
Class representing a group of particles.
"""
__all__ = ["ParticleSpecies", "ParticleTracker"]

import astropy.units as u
import numpy as np
import scipy.interpolate as interp

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
        length of timestep. If not specified, an adaptive timestep
        will be used.

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



    def run(self):
        r"""
        Runs a simulation instance.
        """

        while not _end(self):
            # Push particles
            # TODO: add options for different pushers
            self._push()

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
        an external file.
        """