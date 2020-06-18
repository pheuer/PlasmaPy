import numpy as np
import astropy.units as u
from astropy import constants as const

from plasmapy.particles import Particle
from plasmapy.formulary import parameters

from scipy.optimize import fsolve


def _to_unit(x, unit):
    """
        Converts any astropy Quantity to unit Unit or if given a float value
        adds the unit on.
        """

    if isinstance(x, u.Quantity):
        return x.to(unit)
    else:
        return x * unit


def _dimless_dict(d, units):
    """
    Helper function to convert a dictionary to SI units then return
    just the values (for use in numerical calculation)

    """
    x = {}
    for k in d.keys():
        x[k] = _to_unit(d[k], units[k]).value
    return x


class MhdShock:
    def __init__(self, knowns, unknowns, gamma=5 / 3, ion1="p+", ion2="p+"):
        """
        Initialize shock with dictionaries of known and unknown quantities.
        The values associated with known quantites are fixed, while
        the values given for unknown quantities are used as initial
        guesses.
        """
        # Define the variables and their order
        # TODO: replace with ordered dictionary to combine this with
        # self.units?
        self.variables = [
            "vx1",
            "vy1",
            "bx1",
            "by1",
            "p1",
            "n1",
            "vx2",
            "vy2",
            "bx2",
            "by2",
            "p2",
            "n2",
        ]

        # Define the units expected for each variable
        self.units = {
            "vx1": u.m / u.s,
            "vy1": u.m / u.s,
            "bx1": u.T,
            "by1": u.T,
            "p1": u.Pa,
            "n1": u.m ** -3,
            "vx2": u.m / u.s,
            "vy2": u.m / u.s,
            "bx2": u.T,
            "by2": u.T,
            "p2": u.Pa,
            "n2": u.m ** -3,
            "gamma": u.dimensionless_unscaled,
        }

        self.ion1 = Particle(ion1)
        self.ion2 = Particle(ion2)
        self.m1 = self.ion1.mass.to(u.kg)
        self.m2 = self.ion1.mass.to(u.kg)

        # Condition and set variables
        self.knowns = knowns
        self.unknowns = unknowns
        self.gamma = _to_unit(gamma, u.dimensionless_unscaled)
        self.solution = {}  # To be filled by solve()

        # Create dimensionless versions of the variables in SI units
        self.knowns_si = _dimless_dict(self.knowns, self.units)
        self.unknowns_si = _dimless_dict(self.unknowns, self.units)

    def _jump_conditions(self, x):
        """
        Calculates the Rankine-Hugoniot Jump Conditions for a magnetized
        fluid shock based o
        """

        # x being a list allows values to be popped off in order as needed
        x = list(x)

        # Define some constant values
        mu0 = const.mu0.value
        gamma = self.gamma.value
        m1 = self.m1.value
        m2 = self.m2.value

        # For each value in the variables dictionary, load either the known
        # value (if it is a known quantity) or pop the value from the x list
        # notice that this method relies on the standard ordering of the
        # variables dictionary
        v = {}
        for var in self.variables:
            if var in self.knowns_si.keys():
                v[var] = self.knowns_si[var]

            elif var in self.unknowns_si.keys():
                v[var] = x.pop(0)

            else:
                raise KeyError("ERROR missing required key %s" % var)

        # Create the output array
        out = np.zeros(6)

        # Jump conditions for MHD from
        # http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node79.html

        # These equations need to be re-factored to be normalized to be
        # dimensionless (rather than just stripping away units) so the
        # quantities aren't so small...

        out[0] = v["bx2"] - v["bx1"]

        out[1] = (
            v["vx2"] * v["by2"]
            - v["vy2"] * v["bx2"]
            - v["vx1"] * v["by1"]
            + v["vy1"] * v["bx1"]
        )

        out[2] = m2 * v["n2"] * v["vx2"] - m1 * v["n1"] * v["vx1"]

        out[3] = (
            m2 * v["n2"] * v["vx2"] ** 2
            + v["p2"]
            + v["by2"] ** 2 / 2 / mu0
            - m1 * v["n1"] * v["vx1"] ** 2
            + v["p1"]
            + v["by1"] ** 2 / 2 / mu0
        )

        out[4] = (
            m1 * v["n2"] * v["vx2"] * v["vy2"]
            - v["bx2"] * v["by2"] / mu0
            - m1 * v["n1"] * v["vx1"] * v["vy1"]
            + v["bx1"] * v["by1"] / mu0
        )

        gr = gamma / (gamma - 1)
        vsq1 = v["vx1"] ** 2 + v["vy1"] ** 2
        vsq2 = v["vx2"] ** 2 + v["vy2"] ** 2

        out[5] = (
            0.5 * m2 * v["n2"] * vsq2 * v["vx2"]
            + gr * v["p2"] * v["vx2"]
            + v["by2"] * (v["vx2"] * v["by2"] - v["vy2"] * v["bx2"]) / mu0
            - 0.5 * m1 * v["n1"] * vsq1 * v["vx1"]
            - gr * v["p1"] * v["vx1"]
            - v["by1"] * (v["vx1"] * v["by1"] - v["vy1"] * v["bx1"]) / mu0
        )

        print(out)
        # return LHS of each equation (should be = 0 at solution)
        return out

    def solve(self):
        """
        Solves the RH jump condtions numerically.

        """
        # Create a lambda function for fitting
        fcn = lambda x: self._jump_conditions(x)

        x0 = []
        # Create an array of the guess values, x0
        for var in self.variables:
            if var in self.unknowns_si.keys():
                x0.append(self.unknowns_si[var])
        x0 = np.array(x0)

        # Solve the system numerically
        x = fsolve(fcn, x0)

        print(self._jump_conditions(x))

        # Turn the solutions into a list so values can be popped off
        x = list(x)

        # Create the solution array
        for var in self.variables:
            if var in self.unknowns.keys():
                self.solution[var] = _to_unit(x.pop(0), self.units[var])
            elif var in self.knowns.keys():
                self.solution[var] = _to_unit(self.knowns[var], self.units[var])
        return self.solution

    def mach_c(self):
        """
        Calculate the upstream sound wave Mach number (assuming the relevant
        quantities are included in the known variables)
        """
        if not all(e in self.knowns.keys() for e in ["vx1", "p1", "n1"]):
            raise KeyError(
                "Required information to calculate upstream Mach "
                "number not in knowns."
            )

        vs1 = np.sqrt(self.gamma * self.knowns["p1"] / self.m1 / self.knowns["n1"])

        mach_c = self.knowns["vx1"] / vs1

        return mach_c.to(u.dimensionless_unscaled)

    def mach_a(self):
        """
        Calculate the upstream Alfven Mach number (assuming the relevant
        quantities are included in the known variables)
        """
        if not all(e in self.knowns.keys() for e in ["vx1", "n1", "bx1"]):
            raise KeyError(
                "Required information to calculate upstream Mach "
                "number not in knowns."
            )

        # TODO: should va be calculated using |B| or the normal component bx1?
        if self.knowns["bx1"].value == 0:
            return np.infty

        va = parameters.Alfven_speed(self.knowns["bx1"], self.knowns["n1"], self.ion1)

        mach_a = self.knowns["vx1"] / va

        return mach_a.to(u.dimensionless_unscaled)


class HydroShock(MhdShock):
    def __init__(self, knowns, unknowns, gamma=5 / 3):

        # Fluid shock is just an MHD shock with zero magnetic field
        knowns["bx1"], knowns["by1"] = 0 * u.T, 0 * u.T
        unknowns["bx2"], unknowns["by2"] = 0 * u.T, 0 * u.T

        MhdShock.__init__(self, knowns, unknowns, gamma=gamma)

    def compression_ratio(self):
        """
        Calculate the shock compression ratio: this formula is only valid for
        a fluid (unmagnetized) shock.
        """
        mach_c = self.mach_c()

        return (self.gamma + 1) * mach_c ** 2 / (2 + (self.gamma - 1) * mach_c ** 2)


if __name__ == "__main__":
    """
    Ideas for tests

    - Test compression ratio in fluid shock is predicted by compression ratio
    formula
    """

    knowns = {
        "vx1": 400 * u.km / u.s,
        "vy1": 0 * u.km / u.s,
        "bx1": 0 * u.T,
        "by1": 0 * u.T,
        "p1": 1e-9 * u.Pa,
        "n1": 1e3 * u.cm ** -3,
    }

    unknowns = {
        "vx2": 100 * u.km / u.s,
        "vy2": 0 * u.km / u.s,
        "bx2": 0 * u.T,
        "by2": 0 * u.T,
        "p2": 5e-9 * u.Pa,
        "n2": 5e3 * u.cm ** -3,
    }

    shock = HydroShock(knowns=knowns, unknowns=unknowns)

    print("M_s: {:.2f}".format(shock.mach_c()))
    print("M_A: {:.2f}".format(shock.mach_a()))

    print("Density Comp. Ratio: {:.2f}".format(shock.compression_ratio()))

    sol = shock.solve()

    print(sol["n2"] / sol["n1"])
