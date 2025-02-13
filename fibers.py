"""
####################################################################################################

Fiber Simulation Library

This library provides a set of classes for the simulation of Si-Ge binary glass
optical fibers.

Patrick Banner
RbRy Lab, University of Maryland-College Park
December 27, 2024

####################################################################################################
"""

import numpy as np
from scipy import optimize as opt

# Constants
# -------------------------------------------------------------------------------------------------------------------------------------
pi = np.pi
C_c = 299792458

# Material properties for silica and germania
# -------------------------------------------------------------------------------------------------------------------------------------
# Sellmeier coefficients
# Intended for use in formula Bi*w0^2/(w0^2 - Ci^2)
# Overall structure for the list is [Bi, Ci], with Bi in in 1/um^2, Ci is in um
# For silica, each list is a list of five coefficients representing the T^n
# coefficients, for temperature variation. For germania, they are single coeffs
# measured at 24°C, and the calculating method will add on the thermo-optic
# coefficients for the change with temperature. For fluorine, they are the linear
# and quadratic coefficients in the molar fraction of fluorine, to be added
# to the pure silica number.
_SellmeierCoeffs = {'SiO2': np.array(
                            [[[1.10127, -4.94251e-5, 5.27414e-7, -1.59700e-9, 1.75949e-12],
                              [-8.906e-2, 9.0873e-6, -6.53638e-8, 7.77072e-11, 6.84605e-14]],
                             [[1.78752e-5, 4.76391e-5, -4.49019e-7, 1.44546e-9, -1.57223e-12],
                              [2.97562e-1, -8.59578e-4, 6.59069e-6, -1.09482e-8, 7.85145e-13]],
                             [[7.93552e-1, -1.27815e-3, 1.84595e-5, -9.20275e-8, 1.48829e-10],
                              [9.34454, -70.9788e-3, 1.01968e-4, -5.07660e-7, 8.21348e-10]]]
                            ),
                    'GeO2': np.array(
                            [[0.80686642, 0.068972606],
                             [0.71815848, 0.15396605],
                             [0.85416831, 11.841931]]
                            ),
                    'F': np.array(
                         [[[-61.25, 0.2565],
                           [-23.0,0.101]],
                          [[73.9,-1.836],
                           [10.7,-0.005]],
                          [[233.5,-5.82],
                           [1090.5,-24.695]]]
                         )
                   }
# Coefficients of thermal expansion (/K or /°C)
_CTE = {'SiO2': 5.4e-7, 'GeO2': 10e-6}
# Softening temperatures (°C)
_SofteningTemperature = {'SiO2': 1100, 'GeO2': 300}
# Poisson's ratios
_PoissonRatio = {'SiO2': 0.170, 'GeO2': 0.212}
# Photo-elastic constants [p11, p12]
_PhotoelasticConstants = {'SiO2': [0.121, 0.270], 'GeO2': [0.130, 0.288]}
# Young's moduli in Pascals
_YoungModulus = {'SiO2': 74e9, 'GeO2': 45.5e9}

# Validation utility methods
# These methods just check if the input matches various conditions,
# namely (1) is it a number and (2) is it positive, nonnegative, or 
# between 0 and 1, respectively
# -------------------------------------------------------------------------------------------------------------------------------------
def _validatePositive(val):
    if not isinstance(val, int | float | np.int32 | np.float64 | np.intc):
        raise TypeError("Number expected; this is a" + str(type(val)))
    if not (val > 0):
        raise ValueError("Value should be greater than zero.")
    return val
def _validateNonnegative(val):
    if not isinstance(val, int | float | np.int32 | np.float64 | np.intc):
        raise TypeError("Number expected.")
    if not (val >= 0):
        raise ValueError("Value should be greater than zero.")
    return val
def _validateFractions(frac):
    if not isinstance(frac, float | np.float64 | int | np.int32):
        raise TypeError("Number expected.")
    if not ((0 <= frac) and (frac <= 1)):
        raise ValueError("Fraction should be between 0 and 1. This fraction is {:.3f}.".format(frac))
    return frac

# Methods for calculating material properties of silica-germania binary glasses
# -------------------------------------------------------------------------------------------------------------------------------------
# Many formulas use an eccentricity but I use epsilon, and the conversion
# unfortunately depends on the value of epsilon. Use signFlag to add a negative
# sign as appropriate: if +1, this will always return a positive value
# whose square root is the real eccentricity, but if signFlag=-1, then
# epsilon < 1 will return -e^2, which can be useful for compact calculations
# of birefringence.
def epsilonToEccSq(epsilon, signFlag=1):
    if (epsilon >= 1):
        return 1-(1/epsilon**2)
    else:
        return signFlag*(1-epsilon**2)

def _calcN_Ge(w0, T0):
    """ Get the refractive index of germania at a given temperature and wavelength. """
    # We have Sellmeier coefficients and a formula for the thermo-optic coefficient
    # So we'll just add the two together
    wc = w0*1e6; Tc = T0 + 273.15 # Unit conversions
    n0 = np.sqrt(1 + np.sum(np.array([_SellmeierCoeffs['GeO2'][i][0]*wc**2/(wc**2 - _SellmeierCoeffs['GeO2'][i][1]**2) for i in range(len(_SellmeierCoeffs['GeO2']))])))
    Delta_n0 = 6.2153e-13/4*(Tc**4 - (24+273.15)**4) - 5.3387e-10/3*(Tc**3 - (24+273.15)**3) + 1.6654e-7/2*(Tc**2 - (24+273.15)**2)
    return n0 + Delta_n0
def _calcNs(w0, T0, m0, m1):
    """
    Calculates the refractive indices for the core and cladding silica-based material at
    wavelength w0 (m) and temperature T0 (°C). The molar fractions are m0 for the core and
    m1 for the cladding. If m is negative, dope the silica with fluorine; if positive,
    dope it with germania.
    Returns: n0 (refractive index of core), n1 (refractive index of cladding)
    """
    wc = w0*1e6; Tc = T0 + 273.15 # Unit conversions
    Tpows = np.array([Tc**i for i in [0,1,2,3,4]])
    nGe = _calcN_Ge(w0, T0)
    sc0 = np.zeros((3,2))
    sc1 = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            sc0[i][j] = np.dot(_SellmeierCoeffs['SiO2'][i][j], Tpows)
            sc1[i][j] = sc0[i][j] + (m1 < 0)*(_SellmeierCoeffs['F'][i][j][0]*np.abs(m1)**2 + _SellmeierCoeffs['F'][i][j][1]*np.abs(m1))
            sc0[i][j] = sc0[i][j] + (m0 < 0)*(_SellmeierCoeffs['F'][i][j][0]*np.abs(m0)**2 + _SellmeierCoeffs['F'][i][j][1]*np.abs(m0))
    n0 = np.sqrt(1 + np.sum(np.array([sc0[i][0]*wc**2/(wc**2 - sc0[i][1]**2) for i in range(len(sc0))])))
    n1 = np.sqrt(1 + np.sum(np.array([sc1[i][0]*wc**2/(wc**2 - sc1[i][1]**2) for i in range(len(sc1))])))
    if (m0 > 0):
        n0 = (1-m0)*n0 + m0*nGe
    if (m1 > 0):
        n1 = (1-m1)*n1 + m1*nGe
    return n0, n1
def _fromDiffN(mProps, r0):
    """
    Calculates the molar fraction germania of a glass given certain 
    properties of the fiber. mProps must be a dictionary and keys
    must include one of n0, n1, m0, m1, neff specifying the refractive
    index of the core/cleadding, molar conentration of core/cladding,
    or effective refractive index of the mode; keys must also include ALL
    of dn, T, and w0, the fractional difference in refractive indices
    (n0-n1)/n1 between core and cladding, and the temperature (°C) and
    wavelength (m) at which n0/n1/m0/m1/neff and dn are specified.
    Parameter r0 is the radius of the core, needed for some of the choices
    of specification.

    Returns m_co, m_cl (molar fractions germania of the core and cladding).
    """

    w0 = mProps['w0']; T0 = mProps['T']

    if ('n0' in mProps.keys()):
        # Core is specified
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([mProps['n0'],mProps['n0']/(1 + mProps['dn'])]), np.array([0,0]))
        return soln[0], soln[1]
    elif ('n1' in mProps.keys()):
        # Cladding is specified
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([mProps['n1']*(1 + mProps['dn']),mProps['n1']]), np.array([0,0]))
        return soln[0], soln[1]
    elif ('m0' in mProps.keys()):
        n0 = _calcNs(w0, T0, mProps['m0'], 0)[0]
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, mProps['m0'], m)[1] - n0/(1 + mProps['dn']), 0)[0]
        return mProps['m0'], soln
    elif ('m1' in mProps.keys()):
        n1 = _calcNs(w0, T0, 0, mProps['m1'])[1]
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m, mProps['m1'])[0] - n1*(1 + mProps['dn']), 0)[0]
        return soln, mProps['m1']
    elif ('neff' in mProps.keys()):
        # Effective mode index is specified...
        # We use n_eff to find n0 and then use that and dn to calculate m0, m1
        def func(na):
            return na**2*((2*pi/w0)**2) - (1/r0**2)*(((1+np.sqrt(2))*r0*(2*pi/w0)*np.sqrt(na**2 - na**2/(1 + mProps['dn'])**2))/(1 + (4 + (r0*(2*pi/w0)*np.sqrt(na**2 - na**2/(1 + mProps['dn'])**2))**4)**(1/4)))**2 - (mProps['neff']*(2*pi/w0))**2
        nco = opt.fsolve(func, 1.45)[0] # 1.45 is the initial guess
        soln = opt.fsolve(lambda m: _calcNs(w0, T0, m[0], m[1]) - np.array([nco, nco/(1 + mProps['dn'])]), np.array([0,0]))
        return soln[0], soln[1]
    else:
        raise Exception("mProps missing an index specifier; keys should include one of n0, n1, m0, m1, or neff.")

# Calculate V-parameter and propagation constant beta
def _calcV(r0, w0, n0, n1):
    return r0*(2*pi/w0)*np.sqrt(n0**2 - n1**2)
def _calcBeta(n0, w0, r0, v):
    return np.sqrt((n0**2)*((2*pi/w0)**2) - (1/r0**2)*(((1+np.sqrt(2))*v)/(1+(4+(v**4))**(1/4)))**2)

# Calculate thermally adjusted length
def _calcLt(L0, alpha0, T0, Tref):
    """
    Get the length adjusted for thermal expansion.
    Arguments: L0 (length measured at Tref, m), alpha0 (coeff of thermal expansion, 1/°C),
               T0 (actual temperature, °C), Tref (temp at which L0 is measured, °C)
    """
    return L0*(1+alpha0*(T0 - Tref))

# Calculate doped coefficient of thermal expansion, softening temperature, 
# Poisson ratio, and photoelastic constants
def _calcCTE(m):
    if (m <= 0):
        return _CTE['SiO2']
    return (1-m)*_CTE['SiO2'] + m*_CTE['GeO2']
def _calcTS(m):
    if (m <= 0):
        return _SofteningTemperature['SiO2']
    return (1-m)*_SofteningTemperature['SiO2'] + m*_SofteningTemperature['GeO2']
def _calcPoissonRatio(m):
    if (m <= 0):
        return _PoissonRatio['SiO2']
    return (1-m)*_PoissonRatio['SiO2'] + m*_PoissonRatio['GeO2']
def _calcPhotoelasticConstants(m):
    if (m <= 0):
        return _PhotoelasticConstants['SiO2']
    return [(1-m)*_PhotoelasticConstants['SiO2'][0] + m*_PhotoelasticConstants['GeO2'][0],
            (1-m)*_PhotoelasticConstants['SiO2'][1] + m*_PhotoelasticConstants['GeO2'][1]]
def _calcYoungModulus(m):
    if (m <= 0):
        return _YoungModulus['SiO2']
    return (1-m)*_YoungModulus['SiO2'] + m*_YoungModulus['GeO2']

# Calculate birefringences due to core noncircularity, asymmetric thermal stress,
# bending, and twisting
def _calc_B_CNC(epsilon, n0, n1, r0, v):
    return (epsilonToEccSq(epsilon, signFlag=-1)*(1 - n1**2/n0**2)**(3/2))/(r0) * (4/v**3) * (np.log(v))**3 / (1 + np.log(v))
def _calc_B_ATS(w0, r0, n0, beta, v, p11, p12, alpha0, alpha1, T0, TS, nu_p, epsilon):
    return (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(p11 - p12)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*((epsilon - 1)/(epsilon + 1)))
def _calc_B_BND(w0, n0, p11, p12, nu_p, r0, rc, E, tf = 0):
    if (rc == 0):
        return 0
    return (2*pi/w0)*((n0**3)/2)*(p11-p12)*(1+nu_p)*(0.5*(r0**2/rc**2) + ((2-3*nu_p)/(1-nu_p))*(r0/rc)*(tf/(pi*(r0**2)*E)))
def _calc_B_TWS(n0, p11, p12, tr):
    return (1+((n0**2)/2)*(p11-p12))*(tr)

def _calc_deltaB_CNC(epsilon, n0, n1, r0, v):
    dbx = -((1 - n1**2/n0**2)**(3/2))/(r0) * (2*(np.log(v))**2/v**3) * (1 - (np.log(v)/(1+np.log(v)))*epsilonToEccSq(epsilon, signFlag=-1))
    dby = -((1 - n1**2/n0**2)**(3/2))/(r0) * (2*(np.log(v))**2/v**3) * (1 + (np.log(v)/(1+np.log(v)))*epsilonToEccSq(epsilon, signFlag=-1))
    return np.array([(dbx+dby)/2, dbx, dby])
def _calc_deltaB_ATS(w0, r0, n0, beta, v, p11, p12, alpha0, alpha1, T0, TS, nu_p, epsilon):
    dbx = (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*(1/(r0*(np.sqrt(epsilon) + 1/np.sqrt(epsilon)))))*(p11*r0*np.sqrt(epsilon) + p12*r0/np.sqrt(epsilon))
    dby = (2*pi/w0)*(1-((r0**2)*((n0**2)*(2*pi/w0)**2 - beta**2))/(v**2))*(0.5*(n0**3)*(alpha1 - alpha0)*np.abs(TS - T0)/(1 - nu_p**2)*(1/(r0*(np.sqrt(epsilon) + 1/np.sqrt(epsilon)))))*(p12*r0*np.sqrt(epsilon) + p11*r0/np.sqrt(epsilon))
    return np.array([(dbx+dby)/2, dbx, dby])
def _calc_deltaB_BND(w0, n0, p11, p12, nu_p, r0, rc, E, tf = 0):
    if (rc == 0):
        return np.array([0, 0, 0])
    dbx = (2*pi/w0)*(n0**3/4)*p11*(1+nu_p)*(r0**2/rc**2)
    dby = (2*pi/w0)*(n0**3/4)*p12*(1+nu_p)*(r0**2/rc**2)
    return np.array([(dbx+dby)/2, dbx, dby])

# Calculate Jones matrix given birefringences
def _calc_J0(beta, B_CNC, B_ATS, B_BND, B_TWS, Lt):
    """
    Calculates a Jones matrix given birefringences.
    For now, if a twist birefringence is given, the other birefringnces
        are ignored and the returned matrix ONLY contains the twist birefringence.
    Arguments: 
        beta (propagation constant, 1/m)
        B_CNC (birefringence due to core noncircularity, rad/m)
        B_ATS (birefringence due to asymmetric thermal stress, rad/m)
        B_BND (birefringence due to bending, rad/m)
        B_TWS (birefringence due to twisting, rad/m)
        Lt (thermally adjusted length, m)
    Return: J0 (the Jones matrix), a 2×2 NumPy array
    """
    Jbase = np.array([
                    [   np.exp(1.0j*((0 + (B_CNC + B_ATS + B_BND)/2)*Lt)), 0],
                    [0, np.exp(1.0j*((0 - (B_CNC + B_ATS + B_BND)/2)*Lt))]
                    ])
    if (B_TWS != 0):
        Jbase = np.matmul(np.array([[np.cos(B_TWS*Lt/2), -np.sin(B_TWS*Lt/2)], [np.sin(B_TWS*Lt/2), np.cos(B_TWS*Lt/2)]]), Jbase)
    return Jbase

# Making random arbitrary rotators following the Czegledi algorithm
def makeRotators(n0):
    '''
    Makes n0 arbitrary polarization rotators, returns an array of n0
    Rotator() instances with random orientations.
    '''
    rotators = np.array([], dtype=object)
    alphaData = np.random.normal(loc=0.0, scale=1.0, size=(n0,4))
    for i in range(n0):
        rotators = np.append(rotators, Rotator(alphaData[i]))
    return rotators

# Methods and arrays for random optical fibers
# Each dictionary entry can be either
#     (1) A single number, in which case all instances of the property
#         will be set to that number, or
#     (2) A dictionary containing keys 'mean', 'scale', and 'dist' to
#         match _getRandom() below
#         Note: 'L0' can not have 'mean' as the mean is usually determined
#         by a user input
_randomDistDefaults = {'T0': {'mean': 25, 'scale': 2, 'dist': 'normal'},
                       'Tref': 20,
                       'm0': 0.0036,
                       'm1': 0,
                       'mProps': {},
                       'epsilon': {'mean': 1, 'scale': 0.007, 'dist': 'uniform'},
                       'r0': 4.1e-6, 'r1': 125e-6/2,
                       'rc': {'mean': 10, 'scale': 10, 'dist': 'uniform'},
                       'tr': 0,
                       'tf': 0,
                       'nPaddles': {'mean': 3, 'scale': 1, 'dist': 'uniform_int'},
                       'Ns': {'mean': 3, 'scale': 2, 'dist': 'uniform_int'},
                       'gapLs': {'mean': 0.02, 'scale': 0.005, 'dist': 'uniform'},
                       'angles': {'mean': 0, 'scale': 100*pi/180, 'dist': 'uniform'},
                       'tfs': 0,
                       'rps': {'mean': 0.05, 'scale': 0.02, 'dist': 'uniform'},
                       'L0': {'scale': 10, 'dist': 'normal'},
                       'alpha': {'mean': 0.0, 'scale': 1.0, 'dist': 'normal'}
                      }

def _getRandom(n0, mean, scale, dist):
    """
    This is a utility method for assisting with the creation of random
    fiber configurations. 
    Parameters:
        n0: The size of the needed random numbers. Can be a single number or 2-tuple.
        mean: The mean of the distribution
        scale: A scaling parameter. For the uniform distributions, specify
               the half-width; for Gaussian distributions, specify the standard deviation
        dist: A string determining the distribution; pick from
              uniform: A uniform distribution (mean-scale to mean+scale)
              uniform_int: A uniform distribution of integers only
              normal or Gaussian: A Gaussian distribution
              normal_pos or Gaussian_pos: A Gaussian distribution 
                  cut off at zero, so as to be only the positive part
    """
    if (dist == 'uniform'):
        return (np.random.random(size=n0) - 0.5)*(scale*2) + mean
    elif (dist == 'normal' or dist == 'Gaussian'):
        return np.random.normal(loc = mean, scale = scale, size = n0)
    elif (dist == 'normal_pos' or dist == 'Gaussian_pos'):
        arr = np.random.normal(loc = mean, scale = scale, size = n0)
        q = np.where([arr <= 0])[1]
        while (len(q) != 0):
            arr[q] = np.random.normal(loc=mean, scale=scale, size=len(q))
            q = np.where([arr <= 0])[1]
        return arr
    elif (dist == 'uniform_int'):
        return np.random.randint(int(mean - scale), high=int(mean + scale), size=n0)

# Class FiberLength() definition
# -------------------------------------------------------------------------------------------------------------------------------------
class FiberLength():
    """
    A single length of fiber.
    
    This class allows the simulation of Si-Ge binary glasses. It
    assumes a pure silica cladding and a core made of silica doped
    with germania, GeO2.

    Properties (user sets these):
        w0: Wavelength of light in the fiber (in m)
        T0: Temperature of the fiber (in °C)
        L0: Length of the fiber (in m)
        Tref: Reference temperature at which L0 was measured (in °C)
        r0: Radius of core (in m)
        r1: Outer radius of cladding (in m)
        m0: Molar fraction in core
        m1: Molar fraction in cladding
            ***For both m0 and m1, if negative it's a fraction of fluorine;
            if positive, it's fraction of germania. It's not recommended to
            try specifying fractions of fluorine above 3-4%.
        epsilon: Core noncircularity, defined as a/b, where a, b are
            semimajor and semiminor axes
            Sometimes an eccentricity e is defined as (r_y/r_x)^2 for
            r_y < r_x. epsilon is related as e = sqrt(1 - 1/epsilon^2).
            Then r_x = r0/(1-e^2)^(1/4) and r_y = r0*(1-e^2)^(1/4).
        rc: Bend radius of curvature (m)
        tf: Axial tension force on bends (N)
        tr: Twist rate (rad/m)

    Attributes (derived quantities):
        n0: Core index of refraction
        n1: Cladding index of refraction
        v: Normalized frequency
        beta: Propagation constant (1/m)
        alpha0: Coefficient of thermal expansion of the core (1/°C)
        alpha1: Coefficient of thermal expansion of the cladding (1/°C)
        Lt: Thermally adjusted length (m)
        nu_p: Poisson's ratio
        p11, p12: Photoelastic constants of the core
        TS: Softening temperature of the core (°C)
        E: Young's modulus for the core (Pa)
        B_CNC: Birefringence due to core noncircularity (rad/m)
        B_ATS: Birefringence due to asymmetric thermal stress (rad/m)
        B_BND: Birefringence due to bending (rad/m)
        B_TWS: Birefringence due to twisting (rad/m)
        J0: Total Jones matrix

    Methods:
        calcDGD(dw0): Calculate the DGD of the fiber using a small
            wavelength change dw0 (default 0.1e-9 m or 0.1 nm).
        calcBeatLength(): Calculates the polarization beat length
            due to the birefringences of core nincircularity, asymmetric
            thermal stress, and bending (ignores twisting for now).
        calcD_CD(dw0): Calculate the chromatic dispersion coefficient
            D_CD of the fiber using a small wavelength change dw0
            (default 0.1e-9 m or 0.1 nm).
        calcNGEff(dw0): Calculate the effective group index of the
            fiber using a small wavelength change dw0 (default 0.1e-9
            m or 0.1 nm).
    """

    # Getters for properties
    # User-set properties
    @property
    def m0(self): 
        """
        m0: Molar fraction in core
        """
        return self._m0
    @property
    def m1(self): return self._m1
    @property
    def w0(self): return self._w0
    @property
    def r0(self): return self._r0
    @property
    def r1(self): return self._r1
    @property
    def epsilon(self): return self._epsilon
    @property
    def T0(self): return self._T0
    @property
    def Tref(self): return self._Tref
    @property
    def L0(self): return self._L0
    @property
    def rc(self): return self._rc
    @property
    def tf(self): return self._tf
    @property
    def tr(self): return self._tr
    
    # Derived quantities
    @property
    def n0(self):
        """Core index of refraction"""
        return _calcNs(self.w0, self.T0, self.m0, self.m1)[0]
    @property
    def n1(self):
        """Cladding index of refraction"""
        return _calcNs(self.w0, self.T0, self.m0, self.m1)[1]
    @property
    def v(self):
        """Normalized frequency"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        return _calcV(self.r0, self.w0, n0, n1)
    @property
    def beta(self):
        """Propagation constant (1/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        return _calcBeta(n0, self.w0, self.r0, v)
    @property
    def alpha0(self):
        """Coefficient of thermal expansion of the core (1/°C)"""
        return _calcCTE(self.m0)
    @property
    def alpha1(self):
        """Coefficient of thermal expansion of the cladding (1/°C)"""
        return _calcCTE(self.m1)
    @property
    def Lt(self):
        """Thermally adjusted length (m)"""
        alpha0 = _calcCTE(self.m0)
        return _calcLt(self.L0, alpha0, self.T0, self.Tref)
    @property
    def nu_p(self):
        """Poisson's ratio"""
        return _calcPoissonRatio(self.m0)
    @property
    def p11(self):
        """p11, p12: Photoelastic constants of the core"""
        return _calcPhotoelasticConstants(self.m0)[0]
    @property
    def p12(self):
        """p11, p12: Photoelastic constants of the core"""
        return _calcPhotoelasticConstants(self.m0)[1]
    @property
    def TS(self):
        """Softening temperature of the core (°C)"""
        return _calcTS(self.m0)
    @property
    def E(self):
        """Young's modulus for the core (Pa)"""
        return _calcYoungModulus(self.m0)
    @property
    def B_CNC(self):
        """Birefringence due to core noncircularity (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        return _calc_B_CNC(self.epsilon, n0, n1, self.r0, v)
    @property
    def B_ATS(self):
        """Birefringence due to asymmetric thermal stress (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0); alpha1 = _calcCTE(self.m1)
        TS = _calcTS(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        return _calc_B_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)
    @property
    def B_BND(self):
        """Birefringence due to bending (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        E = _calcYoungModulus(self.m0)
        return _calc_B_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf = self.tf)
    @property
    def B_TWS(self):
        """Birefringence due to twisting (rad/m)"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        return _calc_B_TWS(n0, p11, p12, self.tr)
    @property
    def J0(self):
        """Total Jones matrix"""
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0); alpha1 = _calcCTE(self.m1)
        Lt = _calcLt(self.L0, alpha0, self.T0, self.Tref)
        TS = _calcTS(self.m0)
        E = _calcYoungModulus(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        B_CNC = _calc_B_CNC(self.epsilon, n0, n1, self.r0, v)
        B_ATS = _calc_B_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)
        B_BND = _calc_B_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf = self.tf)
        B_TWS = _calc_B_TWS(n0, p11, p12, self.tr)
        return _calc_J0(beta, B_CNC, B_ATS, B_BND, B_TWS, Lt)

    # Setters for properties
    @m0.setter
    def m0(self, value):
        self._m0 = value
    @m1.setter
    def m1(self, value):
        self._m1 = value
    @w0.setter
    def w0(self, value):
        self._w0 = _validatePositive(value)
    @r0.setter
    def r0(self, value):
        self._r0 = _validatePositive(value)
    @r1.setter
    def r1(self, value):
        self._r1 = _validatePositive(value)
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = _validateNonnegative(value)
    @T0.setter
    def T0(self, value):
        self._T0 = value
    @L0.setter
    def L0(self, value):
        self._L0 = _validatePositive(value)
    @Tref.setter
    def Tref(self, value):
        self._Tref = value
    @rc.setter
    def rc(self, value):
        self._rc = _validateNonnegative(value)
    @tf.setter
    def tf(self, value):
        self._tf = _validateNonnegative(value)
    @tr.setter
    def tr(self, value):
        self._tr = value

    # Initialization
    # Units: w0 (m), T0 (°C), L0 (m), r0,r1 (m), epsilon (unitless), m0,m1 (unitless), Tref (°C), rc (m), tf (N), tr(rad/m)
    # Note on rc: if set to zero, treated as infinity
    def __init__(self, w0, T0, L0, r0, r1, epsilon, m0, m1, Tref, rc, tf, tr, mProps = {}):
        """
        Initialize a new fiber length. 
        
        Parameters:
            w0: wavelength (m)
            T0: temperature (°C)
            L0: length measured at Tref (m)
            r0: radius of core (m)
            r1: outer radius of cladding (m)
            epsilon: core noncircularity, ratio of larger axis to smaller one
                (see class documentation for more details)
            m0, m1: molar fractions of fluorine (if negative) or germania (if positive)
                of the core, cladding (may be overriden by mProps, see below)
            Tref: temperature for length reference (°C)
            rc: bending radius of curvature (m)
            tf: axial tension force on bends (N)
            tr: twist rate (rad/m)
            mProps (optional): a dictionary with alternate specifications of the
                doping concentrations of the fiber; if {}, then m0,m1 will be used;
                if not {}, this will override m0, m1, and keys must include one of
                n0, n1, m0, m1, neff specifying the refractive index of the core
                or cladding, the molar concentration of the core or cladding,
                or effective refractive index of the mode; keys must also include
                ALL of dn, T, and w0, the fractional difference in refractive
                indices (n0-n1)/n1 between core and cladding, and the temperature
                (°C) and wavelength (m) at which n0/n1/m0/m1/neff and dn are
                specified. Default {}.
        """
        self.w0 = w0
        self.T0 = T0; self.L0 = L0
        self.r0 = r0; self.r1 = r1; self.epsilon = epsilon; 
        if (mProps == {}):
            if (m0 <= m1):
                raise Exception("m0 needs to be larger than m1 for a physical fiber. You have m0 = {:.6f} and m1 = {:.6f}.".format(m0, m1))
            else:
                self.m0 = m0; self.m1 = m1
        else:
            if not (np.all([k in mProps.keys() for k in ['dn', 'w0', 'T']]) and np.any([k in mProps.keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']])):
                raise Exception("Unable to initialize FiberLength because mProps doesn't contain the right keys. See FiberLength.__init__ documentation for more details.")
            else:
                a, b = _fromDiffN(mProps, r0)
                if (a <= b):
                    raise Exception("m0 needs to be larger than m1 for a physical fiber. Review your mProps dictionary.")
                else:
                    self.m0 = a; self.m1 = b
        self.Tref = Tref
        self.rc = rc; self.tf = tf; self.tr = tr
    
    def calcDGD(self, dw0 = 0.1e-9):
        """
        Calculates the DGD of the fiber length by varying the wavelength
        in both directions and calculating the Jones matrix.

        Parameters:
            dw0 (optional): the small wavelength step to take (m), default 0.1e-9

        Outputs: dgd, the DGD in s
        """
        # Uses the Jones matrix to get the DGD of the fiber
        # dw0 is the small amount by which to change the wavelength, in m

        # Store current variables
        wb = self.w0
        Jb = self.J0

        # Get Jones matrices for ± dw0
        self.w0 = self._w0 - dw0
        Ja = self.J0
        self.w0 = self._w0 + 2*dw0
        Jc = self.J0

        # Reset values for this object
        self.w0 = wb

        # Get DGD estimates for ± dw0
        matM = np.matmul(Ja, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matM)
        dgdM = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        matP = np.matmul(Jc, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matP)
        dgdP = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        # Average and return
        dgd = (dgdM + dgdP)/2
        return dgd

    def calcBeatLength(self):
        """
        Returns the polarization beat length of the fiber (m).
        Ignores twisting; only accounts for core nincircularity, asymmetric
        thermal stress, and bending birefringences.
        """
        return np.abs(2*pi/(self.B_CNC + self.B_ATS + self.B_BND + self.B_TWS))

    def calcD_CD(self, dw0 = 0.1e-9):
        """
        Method for calculating the chromatic dispersion of the fiber length.
        Returns the effective D_CD from the propagation constant of the fiber
        mode.
        Parameters:
            dw0 (optional): the small wavelength step to take (m), default 0.1e-9

        Outputs: the D_CD in ps/(nm.km)
        """
        # Store current variables
        wb = self.w0
        nB = self.beta/(2*pi/self.w0)

        # Get relevant parameters for ± dw0
        self.w0 = self._w0 - dw0
        nA = self.beta/(2*pi/self.w0)
        self.w0 = self._w0 + 2*dw0
        nC = self.beta/(2*pi/self.w0)

        # Reset values for this object
        self.w0 = wb

        dcd = -self.w0/C_c*(nC - 2*nB + nA)/dw0**2*1e12*1e-9*1e3
        return dcd

    def calcNGEff(self, dw0 = 0.1e-9):
        """
        Method for calculating the effective group index of the fiber,
        c/v_g = n(lambda) - lambda*dn/d-lambda.
        
        Parameters:
            dw0 (optional): the small wavelength step to take (m), default 0.1e-9

        Outputs: the effective group index
        """
        # Store current variables
        wb = self.w0
        nb = self.beta/(2*pi/self.w0)
        # Get relevant parameters for ± dw0
        self.w0 = wb - dw0
        na = self.beta/(2*pi/self.w0)
        self.w0 = wb + dw0
        nc = self.beta/(2*pi/self.w0)
        # Reset values for this object
        self.w0 = wb
        # Calculate
        dndwP = (nc-nb)/dw0
        dndwM = (nb-na)/dw0
        dndw = (dndwP + dndwM)/2
        return nb - wb*(dndw)

    def calcPhaseDelay(self):
        """
        Calculates the time (s) for the light to propagate through the fiber.
        """
        n0, n1 = _calcNs(self.w0, self.T0, self.m0, self.m1)
        v = _calcV(self.r0, self.w0, n0, n1)
        beta = _calcBeta(n0, self.w0, self.r0, v)
        p11, p12 = _calcPhotoelasticConstants(self.m0)
        alpha0 = _calcCTE(self.m0); alpha1 = _calcCTE(self.m1)
        TS = _calcTS(self.m0)
        nu_p = _calcPoissonRatio(self.m0)
        E = _calcYoungModulus(self.m0)
        bcnc = _calc_deltaB_CNC(self.epsilon, n0, n1, self.r0, v)
        bats = _calc_deltaB_ATS(self.w0, self.r0, n0, beta, v, p11, p12, alpha0, alpha1, self.T0, TS, nu_p, self.epsilon)
        bbnd = _calc_deltaB_BND(self.w0, n0, p11, p12, nu_p, self.r1, self.rc, E, tf = self.tf)
        return self.Lt/((2*pi*C_c/self.w0)/(np.ones(3)*self.beta + bcnc + bats + bbnd))

    def __str__(self):
        info_array = ["Fiber length, properties:",
                      "Length: {:.4f} m".format(self.L0),
                      "Operating wavelength: {:.4f} nm".format(self.w0*1e9),
                      "Operating temperature: {:.4f} °C".format(self.T0),
                      "Reference temperature: {:.4f} °C".format(self.Tref),
                      "Base radius of the core: {:.4f} um".format(self.r0*1e6),
                      "Radius of the cladding: {:.4f} um".format(self.r1*1e6),
                      "Noncircularity a/b: {:.4f}".format(self.epsilon),
                      "Molar fraction of dopant in the core: {:.4f}".format(self.m0),
                      "Molar fraction of dopant in the cladding: {:.4f}".format(self.m1),
                      "Core index of refraction: {:.5f}".format(self.n0),
                      "Cladding index of refraction: {:.5f}".format(self.n1),
                      "Mode effective index of refraction: {:.5f}".format(self.beta/(2*pi/self.w0)),
                      "Effective group index: {:.5f}".format(self.calcNGEff()),
                      "Normalized frequency V: {:.4f}".format(self.v),
                     ]
        if (self.rc == 0):
            info_array = info_array + ["Bend radius of curvature: N/A", "Axial tension on bend: N/A"]
        else:
            info_array = info_array + ["Bend radius of curvature: {:.4f} m".format(self.rc), "Axial tension on bend: {:.4f} N".format(self.tf)]
        if (self.tr == 0):
            info_array = info_array + ["Twist rate: N/A"]
        else:
            info_array = info_array + ["Twist rate: {:.4f} rad/m".format(self.tr)]
        info_array = info_array + ["Differential group delay: {:.4f} ps".format(self.calcDGD()*1e12),
                                   "Polarization beat length: {:.4f} m".format(self.calcBeatLength()),
                                   "Chromatic dispersion coefficient D_CD: {:.4f} ps/(nm^2 km)".format(self.calcD_CD())]
        sep = '\n'
        return sep.join(info_array)

# Class FiberPaddleSet() definition
# -------------------------------------------------------------------------------------------------------------------------------------
class FiberPaddleSet():
    """
    This class implements a set of fiber paddles via an array of
    FiberLength() objects. 

    From the code standpoint, we alternate FiberLengths with nonzero
    twist rates and FiberLengths with nonzero bend radii representing
    the paddles. However, physically we suppose it's all one fiber, so
    all the FiberLengths will share some common properties.

    Properties:
        w0: The wavelength (m)
        T0: Temperature (°C)
        Tref: Reference temperature for the lengths (°C)
        r0: The radius of the core (m)
        r1: The outer radius of the cladding (m)
        epsilon: Core noncircularity
        m0, m1: Doping concentrations in core, cladding (see FiberLength
            class documentation for more details)
        nPaddles: Number of paddles
        finalTwistBool: A Boolean to indicate whether there is a final
            section of twisted fiber after the last paddle or not.
        
        rps: Radii of curvature for each paddle (m)
        angles: The angle of each paddle (rad)
        tfs: Tension forces on each paddle (N)
        Ns: The number of turns of fiber on each paddle
        gapLs: The lengths of each of the straight sections of fiber
            between the paddles, including one before the first paddle
            and, if finalTwistBool is True, one after the last paddle
            (lengths in m)

    Attributes:
        fibers: The array of fibers in the paddle set
        J0: The total Jones matrix of the entire paddle set
        L0: The total (non-thermally-adjusted) length of the fiber
            forming the paddle set
    """

    # Properties of all components
    @property
    def m0(self): return self._m0
    @property
    def m1(self): return self._m1
    @property
    def w0(self): return self._w0
    @property
    def r0(self): return self._r0
    @property
    def r1(self): return self._r1
    @property
    def epsilon(self): return self._epsilon
    @property
    def T0(self): return self._T0
    @property
    def Tref(self): return self._Tref
    @property
    def nPaddles(self): return self._nPaddles
    @property
    def finalTwistBool(self): return self._ftb

    @property
    def fibers(self):
        # Build the FiberLengths array
        fa = np.array([], dtype=object)
        angs = np.concatenate(([0], self.angles))
        for i in range(self.nPaddles):
            # Twist
            fa = np.append(fa, FiberLength(self.w0, self.T0, self.gapLs[i], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, 0, 0, (angs[i+1] - angs[i])/self.gapLs[i]))
            # Bend
            fa = np.append(fa, FiberLength(self.w0, self.T0, 2*pi*self.rps[i]*self.Ns[i], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, self.rps[i], self.tfs[i], 0))
        # Final twist
        if (self._ftb):
            fa = np.append(fa, FiberLength(self.w0, self.T0, self.gapLs[-1], self.r0, self.r1, self.epsilon, self.m0, self.m1, self.Tref, 0, 0, (0-angs[-1])/self.gapLs[-1]))

        return fa
    @property
    def J0(self):
        fa = self.fibers
        Jtot = np.array([[1,0],[0,1]])
        for i in range(len(fa)):
            Jtot = np.matmul(fa[i].J0, Jtot)
        return Jtot
    @property
    def L0(self):
        # This is the non-thermally-adjusted length
        return np.sum(self.gapLs) + 2*pi*np.dot(self.rps, self.Ns)

    # Setters for properties
    @m0.setter
    def m0(self, value):
        self._m0 = value
    @m1.setter
    def m1(self, value):
        self._m1 = value
    @w0.setter
    def w0(self, value):
        self._w0 = _validatePositive(value)
    @r0.setter
    def r0(self, value):
        self._r0 = _validatePositive(value)
    @r1.setter
    def r1(self, value):
        self._r1 = _validatePositive(value)
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = _validateNonnegative(value)
    @T0.setter
    def T0(self, value):
        self._T0 = value
    @Tref.setter
    def Tref(self, value):
        self._Tref = value
    @finalTwistBool.setter
    def finalTwistBool(self, value):
        self._ftb = value

    # We're going to let the entire set have the same fiber properties
    def __init__(self, w0, T0, r0, r1, epsilon, m0, m1, Tref, nPaddles, rps, angles, tfs, Ns, gapLs, mProps = {}, finalTwistBool = False):
        """
        Initialize a FiberPaddleSet.

        Parameters:
            w0: The wavelength (m)
            T0: Temperature (°C)
            r0: The radius of the core (m)
            r1: The outer radius of the cladding (m)
            epsilon: Core noncircularity
            m0, m1: Doping concentrations in core, cladding (see FiberLength
                class documentation for more details)
            Tref: Reference temperature for the lengths (°C)
            nPaddles: Number of paddles
            rps: Radii of curvature for each paddle (m) (array of length nPaddles)
            angles: The angle of each paddle (rad) (array of length nPaddles)
            tfs: The axial tension force on each paddle (N) (array of length nPaddles)
            Ns: The number of turns of fiber on each paddle (array of length nPaddles)
            gapLs: The lengths of each of the straight sections of fiber
                between the paddles, including one before the first paddle
                and, if finalTwistBool is True, one after the last paddle
                (lengths in m) (array of length nPaddles + int(finalTwistBool))
            mProps (optional): Same as the FiberLength class (default {})
            finalTwistBool (optional): A Boolean to indicate whether there is a final
                section of twisted fiber after the last paddle or not (default False)
        """
        
        # Do some error checking
        _validatePositive(nPaddles)
        if (len(rps) != nPaddles):
            raise Exception("Length of rp array doesn't match nPaddles!")
        [_validatePositive(i) for i in rps]
        if (len(angles) != nPaddles):
            raise Exception("Length of angles array doesn't match nPaddles!")
        if (len(Ns) != nPaddles):
            raise Exception("Length of Ns array doesn't match nPaddles!")
        [_validatePositive(i) for i in Ns]
        if (len(gapLs) != (nPaddles+int(finalTwistBool))):
            raise Exception("Length of gapLs array doesn't match nPaddles + {:.0f}!".format(int(finalTwistBool)))
        [_validatePositive(i) for i in gapLs]

        self.w0 = w0
        self.T0 = T0;
        self.r0 = r0; self.r1 = r1; self.epsilon = epsilon; 
        if (mProps == {}):
            if (m0 <= m1):
                raise Exception("m0 needs to be larger than m1 for a physical fiber. You have m0 = {:.6f} and m1 = {:.6f}.".format(m0, m1))
            else:
                self.m0 = m0; self.m1 = m1
        else:
            if not (np.all([k in mProps.keys() for k in ['dn', 'w0', 'T']]) and np.any([k in mProps.keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']])):
                raise Exception("Unable to initialize FiberLength because mProps doesn't contain the right keys. See FiberLength.__init__ documentation for more details.")
            else:
                a, b = _fromDiffN(mProps, r0)
                if (a <= b):
                    raise Exception("m0 needs to be larger than m1 for a physical fiber. Review your mProps dictionary.")
                else:
                    self.m0 = a; self.m1 = b
        self.Tref = Tref
        self._nPaddles = nPaddles
        self.rps = rps; self.angles = angles; self.tfs = tfs; self.Ns = Ns
        self.gapLs = gapLs
        self.finalTwistBool = finalTwistBool

    def __str__(self):
        info_array = ["Fiber paddle set, properties:",
                      "Operating wavelength: {:.4f} nm".format(self.w0*1e9),
                      "Operating temperature: {:.4f} °C".format(self.T0),
                      "Reference temperature: {:.4f} °C".format(self.Tref),
                      "Base radius of the core: {:.4f} um".format(self.r0*1e6),
                      "Radius of the cladding: {:.4f} um".format(self.r1*1e6),
                      "Noncircularity a/b: {:.4f}".format(self.epsilon),
                      "Molar fraction of dopant in the core: {:.4f}".format(self.m0),
                      "Molar fraction of dopant in the cladding: {:.4f}".format(self.m1),
                      " ",
                      "Number of paddles: {:.0f}".format(self.nPaddles),
                      "Radii of curvature of paddles (m): " + str(self.rps),
                      "Number of fiber turns on each paddle: " + str(self.Ns),
                      "Angles of paddles (m): " + str(self.angles),
                      "Tension forces on wrappings of each paddle (N): " + str(self.tfs),
                      "Lengths of fiber between each paddle (m): " + str(self.gapLs),
                      "Total fiber length: {:.4f} m".format(self.L0),
                      " ",
                      "Use object.fibers to get more details about each fiber component."
                     ]
        sep = '\n'
        return sep.join(info_array)

    def calcPhaseDelay(self):
        """
        Method for calculating the time for the light to propagate through the fiber.
        """
        fa = self.fibers
        t0 = np.array([0,0,0])
        for i in range(len(fa)):
            t0 = t0 + fa[i].calcPhaseDelay()
        return t0

# Class Rotator() definition
# -------------------------------------------------------------------------------------------------------------------------------------
class Rotator():
    """
    Implements an arbitrary rotator following the formalism of Czegledi et al.

    Attributes:
        alpha: The 4-vector that specifies the rotator
        theta: The angle of rotation (the axis is alpha[1:])
        L0: = 0 (for compatibility purposes)
        J0: The Jones matrix of the rotator
    """
    # Following the formalism of Czegledi, these rotators are
    # specified by a 4-vector alpha

    @property
    def theta(self): return np.arccos(self.alpha[0])
    # This one is for compatibility purposes
    @property
    def L0(self): return 0
    @property
    def J0(self):
        aVec = self.alpha[1:]/np.sin(self.theta)
        if (self.theta == 0):
            aVec = np.array([0,0,0])
        J0 = np.cos(self.theta)*np.array([[1,0],[0,1]]) - 1j*np.sin(self.theta)*(aVec[0]*np.array([[1,0],[0,-1]]) + aVec[1]*np.array([[0,1],[1,0]]) + aVec[2]*np.array([[0,-1j],[1j,0]]))
        return J0

    def __init__(self, alpha):
        """
        Initialize a Rotator.
        
        Parameters:
            alpha: a 4-vector defining the rotation
        """
        self.alpha = alpha/np.linalg.norm(alpha)

    def calcPhaseDelay(self):
        """
        This is for compatibility only.
        """
        return np.array([0,0,0])

    def __str__(self):
        return "Arbitrary rotator over angle {:.4f}° about an axis.".format(self.theta*180/pi)
        

# Class Fiber() definition
# -------------------------------------------------------------------------------------------------------------------------------------
class Fiber():
    """
    Implements a full optical fiber with alternating segments and hinges
    according to the "hinge model" of optical fibers.

    Properties:
        w0: Wavelength of light (m)
        N0: Number of long segments
        arbRotStart: Boolean, whether to start the fiber with an arbitrary
            rotation (sometimes useful in simulatory applications)
        addRotators: None, single number, or dictionary about
            arbitrary rotators along the fiber lengths (see initialization
            or Fiber.random() documentation for more details)

    Attributes:
        hingeType: if 0, hinges are FiberPaddles; if 1, hinges are Rotators
        hingeStart: Boolean, whether there's a hinge before the first fiber segment;
            if arbRotStart is True, the arbitrary rotation precedes this first hinge
        hingeEnd: Boolean, whether there's a hinge after the last fiber segment
        startRotator: The Rotator at the start of the fiber (always a
            Rotator object, but its Jones matrix is the identity if arbRotStart
            is False)
        addedRotators: A dictionary of rotator parameters and fiber lengths for the
            arbitrary rotators along the long fiber segments, when addRotators
            is not None
        fibers: The array of FiberLength and FiberPaddleSet objects
            constituting the fiber
        J0: The total Jones matrix of the fiber
        L0: The total length of the fiber

        **NOTE:** The two dictionaries below describe the entire fiber from its
            constituent components. Generally, the value can be either sufficient to
            describe one component, in which case it's used for all components, or it
            can be an array sufficient to describe all components individually.
            One notable exception to this is mProps: if specified, it needs to be a
            dictionary of single numbers only.
        segmentDict: A dictionary containing the properties of the long
            segments of the fiber. Should contain keys 'T0', 'L0', 'r0', 'r1',
            'epsilon', 'm0' and 'm1' or 'mProps', 'Tref', 'rc', 'tf', 'tr' which are each
            either single numbers or arrays of length N0 (if single number, all segments
            will have that number as their property).
        hingeDict: A dictionary containing the properties of the hinges of the fiber.
            If hingeType = 1, this dictionary only contains 'alpha', a length-4 array
            or a (4×N0h)-length array.
            If hingeType = 0, this dictionary will needs keys 'T0', 'r0', 'r1', 'epsilon',
            'm0' and 'm1' or 'mProps', 'Tref', 'nPaddles', 'finalTwistBool' (which are
            single numbers or 1×N0h arrays) and 'rps', 'angles', 'tfs', 'Ns', 'gapLs' (which
            are 1×nPaddles arrays or N0h×nPaddles arrays).
            Here N0h = N0-1 + hingeStart + hingeEnd.

    Methods:
        calcDGD: Calculate the total DGD of the fiber.
        getHingeLocations: Returns the indices in self.fibers that are hinges.

    Class Methods:
        random: Generate a random optical fiber. See the random()
            documentation for more details. Call as Fiber.random().
    """

    segmentDictKeys = np.array(['L0', 'T0', 'Tref', 'epsilon', 'm0', 'm1', 'mProps', 'r0', 'r1', 'rc', 'tf', 'tr'])
    hingeDictKeys = np.array(['Ns', 'T0', 'Tref', 'angles', 'epsilon', 'finalTwistBool', 'gapLs', 'm0', 'm1', 'mProps', 'nPaddles', 'r0', 'r1', 'rps', 'tfs'])
    
    @property
    def w0(self): return self._w0
    @w0.setter
    def w0(self, value):
        self._w0 = _validatePositive(value)
        
    @property
    def N0(self): return self._N0
    @N0.setter
    def N0(self, value):
        self._N0 = _validatePositive(value)

    @property
    def arbRotStart(self): return self._arbRotStart
    @arbRotStart.setter
    def arbRotStart(self, newVal):
        self.toggleStartRotator(newVal)
    @property
    def startRotator(self): return self._startRotator

    @property
    def addRotators(self): return self._addRotators
    @addRotators.setter
    def addRotators(self, newVal):
        self.toggleAddedRotators(newVal)
    @property
    def addedRotators(self): return self._addedRotators

    @property
    def fibers(self):
        # The actual number of hinges
        N0h = self.N0-1 + int(self.hingeStart) + int(self.hingeEnd)
        # Take some compliance measures on the dictionaries
        # Look for optional parameters and set appropriately if not found
        ref = [[self.segmentDict, self.N0]]
        if (self.hingeType == 0):
            ref = [[self.segmentDict, self.N0], [self.hingeDict, N0h]]
        for d,n in ref:
            if ('mProps' in d.keys() and d['mProps'] != {}):
                d['m0'] = np.zeros(n)
                d['m1'] = np.zeros(n)
            else:
                d['mProps'] = {}
        if (self.hingeType == 0) and ('finalTwistBool' not in self.hingeDict.keys()):
            self.hingeDict['finalTwistBool'] = np.zeros(N0h)
        if (self.hingeType == 1) and ('alpha' not in self.hingeDict.keys()):
            self.hingeDict['alpha'] = []
        
        # Now both dicts should have a specific set of keys... check that that's true
        d1 = np.sort(list(self.segmentDict.keys()))
        if (len(d1) != len(self.segmentDictKeys)):
            raise Exception("Your fiber segment dictionary does not have the expected number of keys." +
                           "\nHere are the necessary keys: " + str(self.segmentDictKeys) + 
                           "\nHere is your dictionary:     " + str(d1)
                          )
        elif (not all(d1 == self.segmentDictKeys)):
            raise Exception("Something in your fiber segment dictionary is specified incorrectly." +
                            "\nHere are the necessary keys: " + str(self.segmentDictKeys) + 
                            "\nHere is your dictionary:     " + str(d1)
                           )
        d2 = np.sort(list(self.hingeDict.keys()))
        if (len(d2) != len(self.hingeDictKeys)) and (len(d2) != 1):
            raise Exception("Your hinge dictionary does not have the expected number of keys." +
                           "\nHere are the necessary keys: " + str(self.hingeDictKeys) + ' or [\'alpha\']' +
                           "\nHere is your dictionary:     " + str(d2)
                          )
        elif ((len(d2) == len(self.hingeDictKeys)) and (not all(d2 == self.hingeDictKeys))) or ((len(d2) == 1) and (not all(d2 == np.array(['alpha'])))):
            raise Exception("Something in your hinge dictionary is specified incorrectly." +
                            "\nHere are the necessary keys: " + str(self.hingeDictKeys) + ' or [\'alpha\']' + 
                            "\nHere is your dictionary:     " + str(d2)
                           )

        # Now we're going to check each property to see if it's the correct-sized
        # array. If it's a single number or a 1×whatever array, do the conversion
        # here. Throw errors if I really can't make it fit.

        # Pre-process r0 and mProps together (because we need r0 to process mProps)
        # Do the following for both segmentDict and hingeDict
        ref = [[self.segmentDict, self.N0]]
        if (self.hingeType == 0):
            ref = [[self.segmentDict, self.N0], [self.hingeDict, N0h]]
        for d,n in ref:
            # Check if mProps is a dict and, if so, if it has the right keys
            if not isinstance(d['mProps'], dict):
                raise Exception("mProps needs to be a dictionary but is not.")
            elif not ((d['mProps'] == {}) or (np.all([k in d['mProps'].keys() for k in ['dn', 'w0', 'T']]) and np.any([k in d['mProps'].keys() for k in ['n0', 'n1', 'm0', 'm1', 'neff']]))):
                raise Exception("mProps dictionary is invalid.")
            else:
                # mProps is a valid dictionary
                if (isinstance(d['r0'], int | float | np.int32 | np.float64)) or (isinstance(d['r0'], np.ndarray) and (len(d['r0']) == 1)):
                    # r0 is a single number...
                    if (d['mProps'] == {}):
                        # mProps isn't being used, so m0 and m1 are
                        # So just standardize r0
                        d['r0'] = np.array([d['r0']]*n).flatten()
                    else:
                        # Everything is a single number and we can compute it for all fiber segments and be done
                        a, b = _fromDiffN(d['mProps'], d['r0'])
                        d['m0'] = np.array([a]*n); d['m1'] = np.array([b]*n)
                        d['r0'] = np.array([d['r0']]*n).flatten()
                elif (len(d['r0']) != n):
                    raise Exception("The r0 array of the dictionary is the wrong length.")
                else:
                    # If mProps is an empty dictionary don't need to do anything else
                    if (d['mProps'] != {}):
                        # r0 is an array of the right length
                        d['m0'] = np.zeros(n); d['m1'] = np.zeros(n)
                        for i in range(n):
                            a, b = _fromDiffN(d['mProps'], d['r0'][i])
                            d['m0'][i] = a; d['m1'][i] = b

        # Conversions for segments
        self.segmentDict.pop('mProps', None)
        for p in self.segmentDict.keys():
            if (isinstance(self.segmentDict[p], int | float | np.int32 | np.float64)) or (isinstance(self.segmentDict[p], np.ndarray) and (len(self.segmentDict[p]) == 1)):
                self.segmentDict[p] = np.array([self.segmentDict[p]]*self.N0).flatten()
            elif (len(self.segmentDict[p]) != self.N0):
                raise Exception("Array in segment dictionary with key " + str(p) + " has the wrong shape, should be 1×{:.0f} but is ".format(self.N0) + str(np.shape(self.segmentDict[p])))
                
        # Conversions for hinges
        if (N0h != 0):
            self.hingeDict.pop('mProps', None)
            if (self.hingeType == 0):
                if (isinstance(self.hingeDict['nPaddles'], int | float | np.int32 | np.float64)) or (isinstance(self.hingeDict['nPaddles'], np.ndarray) and (len(self.hingeDict['nPaddles']) == 1)):
                    self.hingeDict['nPaddles'] = np.array([self.hingeDict['nPaddles']]*N0h).flatten()
                elif (len(self.hingeDict['nPaddles']) != N0h):
                        raise Exception("Array in hinge dictionary with key nPaddles has the wrong shape, should be 1×{:.0f} but is ".format(N0h) + str(np.shape(self.hingeDict['nPaddles'])))
                for p in self.hingeDict.keys():
                    if (p not in ['rps', 'angles', 'tfs', 'Ns', 'gapLs']):
                        if (isinstance(self.hingeDict[p], int | float | np.int32 | np.float64)) or (isinstance(self.hingeDict[p], np.ndarray) and (len(self.hingeDict[p]) == 1)):
                            self.hingeDict[p] = np.array([self.hingeDict[p]]*N0h).flatten()
                        elif (len(self.hingeDict[p]) != N0h):
                            raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape, should be 1×{:.0f} but is ".format(N0h) + str(np.shape(self.hingeDict[p])))
                    else:
                        if (len(np.shape(self.hingeDict[p])) == 1):
                            if (len(self.hingeDict[p]) >= np.max(self.hingeDict['nPaddles'])):
                                self.hingeDict[p] = np.array([list(self.hingeDict[p])]*N0h)
                            else:
                                raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
                        elif (len(np.shape(self.hingeDict[p])) == 2) and (np.shape(self.hingeDict[p])[0] == 1):
                            if (len(self.hingeDict[p][0]) >= np.max(self.hingeDict['nPaddles'])):
                                self.hingeDict[p] = np.array([list(self.hingeDict[p][0])]*N0h)
                            else:
                                raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
                        elif ((np.shape(self.hingeDict[p])[0] != N0h) or (np.shape(self.hingeDict[p])[1] < np.max(self.hingeDict['nPaddles']))):
                            raise Exception("Array in hinge dictionary with key " + str(p) + " has the wrong shape. It should be 1×(at least {:.0f}) or {:.0f}×(at least {:.0f}) but is ".format(np.max(self.hingeDict['nPaddles']), N0h, np.max(self.hingeDict['nPaddles'])) + str(np.shape(self.hingeDict[p])))
            elif (self.hingeType == 1):
                if (len(np.shape(self.hingeDict['alpha'])) == 1):
                    if (len(self.hingeDict['alpha']) == 4):
                        self.hingeDict['alpha'] = np.array([list(self.hingeDict['alpha'])]*N0h)
                    else:
                        raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))
                elif (len(np.shape(self.hingeDict['alpha'])) == 2) and (np.shape(self.hingeDict['alpha'])[0] == 1):
                    if (len(self.hingeDict['alpha'][0]) == 4):
                        self.hingeDict['alpha'] = np.array([list(self.hingeDict['alpha'][0])]*N0h)
                    else:
                        raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))
                elif (np.shape(self.hingeDict['alpha']) != (N0h, 4)):
                    raise Exception("Alpha specs for rotator hinges should be 1×4 or {:.0f}×4, but your spec is".format(N0h) + str(np.shape(self.hingeDict['alpha'])))
        
        # Now make the fiber array.
        # If we aren't putting rotators in the segments, then just interleave the segments
        # and hinges properly and call it a day. If we ARE adding rotators, then make the
        # rotators and the proper-length segments and interleave those, putting a hinge
        # where appropriate.
        # Pre-make the hinge objects, whatever they are:
        hinges = np.array([], dtype=object)
        if (self.hingeType == 0):
            for i in range(N0h):
                npad = self.hingeDict['nPaddles'][i]
                ftb = self.hingeDict['finalTwistBool'][i]
                hinges = np.append(hinges, FiberPaddleSet(self.w0, self.hingeDict['T0'][i], self.hingeDict['r0'][i], self.hingeDict['r1'][i], self.hingeDict['epsilon'][i], self.hingeDict['m0'][i], self.hingeDict['m1'][i], self.hingeDict['Tref'][i], npad, self.hingeDict['rps'][i][:npad], self.hingeDict['angles'][i][:npad], self.hingeDict['tfs'][i][:npad], self.hingeDict['Ns'][i][:npad], self.hingeDict['gapLs'][i][:int(npad+int(ftb))], mProps = {}, finalTwistBool = ftb))
        elif (self.hingeType == 1):
            for i in range(N0h):
                hinges = np.append(hinges, Rotator(self.hingeDict['alpha'][i]))
        
        # Now put it all together.
        hingeInds = np.array([], dtype=int)
        fa = np.array([], dtype=object)
        if (self.arbRotStart):
            fa = np.append(fa, self._startRotator)
        if (self.hingeStart):
            fa = np.append(fa, hinges[0])
            hingeInds = np.append(hingeInds, 0)
        for i in range(self.N0):
            # Add the segment
            if (self.addRotators == None):
                fa = np.append(fa, FiberLength(self.w0, self.segmentDict['T0'][i], self.segmentDict['L0'][i], self.segmentDict['r0'][i], self.segmentDict['r1'][i], self.segmentDict['epsilon'][i], self.segmentDict['m0'][i], self.segmentDict['m1'][i], self.segmentDict['Tref'][i], self.segmentDict['rc'][i], self.segmentDict['tf'][i], self.segmentDict['tr'][i], mProps = {}))
            else:
                Ns = 0
                thisSegmentDict = {'T0': self.segmentDict['T0'][i], 'L0': self._addedRotators['L'][i], 'r0': self.segmentDict['r0'][i], 'r1': self.segmentDict['r1'][i], 'epsilon': self.segmentDict['epsilon'][i], 'm0': self.segmentDict['m0'][i], 'm1': self.segmentDict['m1'][i], 'Tref': self.segmentDict['Tref'][i], 'rc': self.segmentDict['rc'][i], 'tf': self.segmentDict['tf'][i], 'tr': self.segmentDict['tr'][i], 'mProps': {}}
                if isinstance(self.addRotators, int | float | np.int32 | np.float64):
                    Ns, _ = np.divmod(self.segmentDict['L0'][i], self.addRotators)
                elif isinstance(self.addRotators, dict):
                    Ns, _ = np.divmod(self.segmentDict['L0'][i], self.addRotators['mean'])
                fseg = Fiber(self.w0, thisSegmentDict, {'alpha': self._addedRotators['alpha'][i]}, int(Ns)+1, hingeType = 1, hingeStart = False, hingeEnd = False, arbRotStart = False, addRotators = None)
                fa = np.append(fa, fseg.fibers.copy()).flatten()
                del fseg
            # Add the next hinge
            if ((i != self.N0-1) or self.hingeEnd):
                fa = np.append(fa, hinges[i + int(self.hingeStart)])
                hingeInds = np.append(hingeInds, len(fa)-1)
            
        if (self._printingBool):
            return fa, hingeInds
        return fa
        
    @property
    def J0(self):
        fa = self.fibers
        Jtot = np.array([[1,0],[0,1]])
        for i in range(len(fa)):
            Jtot = np.matmul(fa[i].J0, Jtot)
        return Jtot
    @property
    def L0(self):
        fa = 0; hingeInds = 0
        if (self._printingBool):
            fa, hingeInds = self.fibers
        else:
            fa = self.fibers
        L0 = 0
        for i in range(len(fa)):
            L0 = L0 + fa[i].L0
        if (self._printingBool):
            return fa, hingeInds, L0
        return L0
        
    # N0: number of long segments
    # hingeType = 0 (fiber paddle sets), 1 (arbitrary rotators)
    def __init__(self, w0, segmentDict, hingeDict, N0, hingeType = 0, hingeStart = True, hingeEnd = True, arbRotStart = False, addRotators = None):
        """
        Initializes a fiber.

        Parameters:
            w0: Wavelength of light (m)
            segmentDict, hingeDict: Dictionaries containing the properties of the constituent
                parts; see class documentation for more details.
            N0: The number of long segments of fiber.
            hingeType (optional): 0 means hinges are FiberPaddleSets, 1 means hinges are
                arbitrary Rotators (default 0)
            hingeStart (optional): Boolean, whether there's a hinge before the first fiber segment;
                if arbRotStart is True, the arbitrary rotation precedes this first hinge (default True)
            hingeEnd (optional): Boolean, whether there's a hinge after the last fiber segment
                (default True)
            arbRotStart (optional): Boolean, whether to start the fiber with an arbitrary rotator
                (default False)
            addRotators (optional): A parameter to add arbitrary rotators along the fiber segments,
                separate from the hinges. Can be either None (no rotators added) or one of the following:
                (1) a single number, the exact distance between each rotator (in meters) (it will be rounded
                    automatically to the nearest multiple of the fiber length);
                (2) a dictionary with 'mean', 'scale', and 'dist' to get random distances between rotators
                    (see _getRandom documentation for more details).
        """
        self.w0 = w0
        self.N0 = N0
        self.segmentDict = segmentDict
        self.hingeDict = hingeDict
        self.hingeType = hingeType
        self.hingeStart = hingeStart; self.hingeEnd = hingeEnd
        self.arbRotStart = arbRotStart
        self.addRotators = addRotators
        
        # This is a flag for internal use only
        self._printingBool = False

    def toggleStartRotator(self, newVal):
        if (newVal):
            self._arbRotStart = True
            self._startRotator = makeRotators(1)[0]
        else:
            self._arbRotStart = False
            self._startRotator = Rotator([1,0,0,0])

    def toggleAddedRotators(self, newVal):
        if (newVal == None):
            self._addedRotators = None
            self._addRotators = None
        else:
            # Gotta get the lengths right...
            Ls = 0
            if (isinstance(self.segmentDict['L0'], int | float | np.int32 | np.float64)) or (isinstance(self.segmentDict['L0'], np.ndarray) and (len(self.segmentDict['L0']) == 1)):
                Ls = np.array([self.segmentDict['L0']]*self.N0).flatten()
            elif (len(self.segmentDict['L0']) != self.N0):
                raise Exception("Array in segment dictionary with key " + str(p) + " has the wrong shape, should be 1×{:.0f} but is ".format(self.N0) + str(np.shape(self.segmentDict['L0'])))
            else:
                Ls = self.segmentDict['L0']
            # Now make the rotators
            self._addedRotators = {'alpha': {}, 'L': {}}
            if isinstance(newVal, int | float | np.int32 | np.float64):
                for i in range(self.N0):
                    Ns, rem = np.divmod(Ls[i], newVal)
                    if (Ns == 0):
                        raise Exception("The given length is larger than the available length, so no rotators are fitting in this segment.")
                    Ns = int(Ns)
                    self._addRotators = newVal
                    self._addedRotators['alpha'][i] = _getRandom((Ns, 4), **_randomDistDefaults['alpha'])
                    self._addedRotators['L'][i] = newVal + rem/Ns
            elif isinstance(newVal, dict):
                for i in range(self.N0):
                    Ns, _ = np.divmod(Ls[i], newVal['mean'])
                    if (Ns == 0):
                        raise Exception("The given length is larger than the available length, so no rotators are fitting in this segment.")
                    Ns = int(Ns)
                    self._addRotators = newVal
                    self._addedRotators['alpha'][i] = _getRandom((Ns, 4), **_randomDistDefaults['alpha'])
                    self._addedRotators['L'][i] = _getRandom(int(Ns+1), **newVal)

    def calcDGD(self, dw0 = 0.1e-9):
        """
        Calculates the DGD of the fiber length by varying the wavelength
        in both directions and calculating the Jones matrix.

        Parameters:
            dw0 (optional): the small wavelength step to take (m), default 0.1e-9

        Outputs: dgd, the DGD in s
        """

        # Store current variables
        wb = self.w0
        Jb = self.J0

        # Get Jones matrices for ± dw0
        self.w0 = self._w0 - dw0
        Ja = self.J0
        self.w0 = self._w0 + 2*dw0
        Jc = self.J0

        # Reset values for this object
        self.w0 = wb

        # Get DGD estimates for ± dw0
        matM = np.matmul(Ja, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matM)
        dgdM = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        matP = np.matmul(Jc, np.linalg.inv(Jb))
        vals, vecs = np.linalg.eig(matP)
        dgdP = np.abs(np.angle(vals[0]/vals[1])/((2*pi*C_c)/self.w0**2*dw0))

        # Average and return
        dgd = (dgdM + dgdP)/2
        return dgd

    def getHingeLocations(self):
        """
        Returns an array of the indices in self.fibers that are hinges.
        """
        self._printingBool = True
        _, hingeInds = self.fibers
        self._printingBool = False
        return hingeInds

    @classmethod
    def random(cls, w0, Ltot, N0, segmentDict, hingeDict, hingeType = 0, hingeStart = True, hingeEnd = True, arbRotStart = False, addRotators = None):
        """
        Generates a random optical fiber following input specs.
        
        The general idea for specifying fibers here is that the dictionaries
        will contain either numbers or arrays that will be used directly, 
        specs for the distribution to randomly draw from, or nothing at all,
        in which case numbers will be drawn from random defaults. For example,
        for the segment core noncircularities, if segmentDict contains an entry
        'epsilon': 1.005, then all the segments will have epsilon = 1.005; it could
        also be a N0-length array specifying the epsilon for each segment. (Properties
        specified directly must follow the rules for doing so, see e.g. FiberPaddleSet
        documentation for more details.) Or one can specify the properties of the
        distribution to be randomly drawn from; for example, 
        'epsilon': {'mean': 1.005, 'scale': 0.002, 'dist': 'uniform'} draws the
        epsilons for all the segments from a uniform distribution between 1.003 and
        1.007 (see also the _getRandom() documentation). If segmentDict has NO entry
        'epsilon', then epsilons will be drawn randomly using the information in
        _randomDistDefaults.

        If the lengths are left to randomness, the method corrects the fiber segments
        to ensure that the total length of the fiber is the specified Ltot.

        Parameters:
            w0: Wavelength of light (m)
            Ltot: Total length of the fiber (m)
            N0: Number of long birefringent segments
            segmentDict, hingeDict: The dictionaries containing the info about the
                segments and hinges, to follow the above description.
                If mProps is specified, it must be single numbers, i.e. it cannot be randomized.
            hingeType (optional): 0 means hinges are FiberPaddleSets, 1 means hinges are
                arbitrary Rotators (default 0)
            hingeStart (optional): Boolean, whether there's a hinge before the first fiber segment;
                if arbRotStart is True, the arbitrary rotation precedes this first hinge (default True)
            hingeEnd (optional): Boolean, whether there's a hinge after the last fiber segment
                (default True)
            arbRotStart (optional): Boolean, whether to start the fiber with an arbitrary rotator
                (default False)
            addRotators (optional): A parameter to add arbitrary rotators along the fiber segments,
                separate from the hinges. Can be either None (no rotators added) or one of the following:
                (1) a single number, the exact distance between each rotator (in meters);
                (2) a dictionary with 'mean', 'scale', and 'dist' to get random distances between rotators
                    (see _getRandom documentation for more details).

        Outputs: A random Fiber following the given specifications.
        
        """
    
        N0h = N0 - 1 + hingeStart + hingeEnd
    
        newSegmentDict = {}
        newHingeDict = {}
        hingeLength = 0
    
        # Start with properties that can reuse code for both segments and hinges
        ref = [[segmentDict, newSegmentDict, N0]]
        if (hingeType == 0):
            ref = [[segmentDict, newSegmentDict, N0], [hingeDict, newHingeDict, N0h]]
        for d1,d2,n in ref:
            # Have to handle the doping fraction which can be specified one of two ways
            # If neither 'm0','m1' nor 'mProps' is in there, let the random defaults handle it
            # If they're both in there, pass both along; Fiber() class logic handles it
            if (('m0' not in d1.keys()) or ('m1' not in d1.keys())) and ('mProps' in d1.keys()):
                d1['m0'] = 0; d1['m1'] = 0
            elif (('m0' in d1.keys()) and ('m1' in d1.keys())) and ('mProps' not in d1.keys()):
                d1['mProps'] = {}
            # Now loop over each property and pass or randomize as necessary
            for prop in ['T0', 'Tref', 'epsilon', 'r0', 'r1', 'rc', 'tf', 'tr', 'm0', 'm1', 'mProps']:
                da = d1
                if (prop not in d1.keys()):
                    da = _randomDistDefaults
                
                if isinstance(da[prop], int | float | np.int32 | np.float64 | np.ndarray):
                    d2[prop] = da[prop]
                elif (isinstance(da[prop], dict) and all([s in da[prop].keys() for s in ['mean', 'scale', 'dist']])):
                    d2[prop] = _getRandom(n, **da[prop])
                elif (prop == 'mProps'):
                    # Just gonna assume that what's specified is fine
                    d2[prop] = da[prop]
                else:
                    raise Exception("On property " + str(prop) + ", something is incorrectly specified.")
            # Save a few lines of code by just doing this
            newHingeDict.pop('rc', None); newHingeDict.pop('tr', None); newHingeDict.pop('tf', None)
        
        if (hingeType == 0):
            # nPaddles is important for remaining hinge properties
            if ('nPaddles' in hingeDict.keys()):
                if isinstance(hingeDict['nPaddles'], int | float | np.int32 | np.float64 | np.ndarray):
                    newHingeDict['nPaddles'] = hingeDict['nPaddles']
                elif (isinstance(hingeDict['nPaddles'], dict) and all([s in hingeDict['nPaddles'].keys() for s in ['mean', 'scale']])):
                    hingeDict['nPaddles']['dist'] = 'uniform_int'
                    newHingeDict['nPaddles'] = _getRandom(N0h, **hingeDict['nPaddles'])
            else:
                newHingeDict['nPaddles'] = _getRandom(N0h, **_randomDistDefaults['nPaddles'])
            # Handle finalTwistBool
            if ('finalTwistBool' in hingeDict.keys()):
                newHingeDict['finalTwistBool'] = hingeDict['finalTwistBool']
            else:
                newHingeDict['finalTwistBool'] = False
            # Now do the arrayed hinge properties
            nPadMax = np.max(newHingeDict['nPaddles'])
            for prop in ['Ns', 'angles', 'tfs', 'rps', 'gapLs']:
                if (prop in hingeDict.keys()):
                    if isinstance(hingeDict[prop], int | float | np.int32 | np.float64 | np.ndarray):
                        newHingeDict[prop] = hingeDict[prop]
                    elif (isinstance(hingeDict[prop], dict) and all([s in hingeDict[prop].keys() for s in ['mean', 'scale', 'dist']])):
                        # I add 1 for gapLs in case there's a finalTwistBool = True... the later methods will just discard it if not needed
                        newHingeDict[prop] = _getRandom((N0h, nPadMax + 1*(prop == 'gapLs')), **hingeDict[prop])
                else:
                    if isinstance(_randomDistDefaults[prop], dict):
                        newHingeDict[prop] = _getRandom((N0h, nPadMax + 1*(prop == 'gapLs')), **_randomDistDefaults[prop])
                    else:
                        newHingeDict[prop] = np.ones((N0h, nPadMax + 1*(prop == 'gapLs')))*_randomDistDefaults[prop]
            # Finally, do the lengths, ensuring they add up to Ltot including the hinge lengths
            # Have to calculate the hinge lengths first...
            hingeLengthCalcs = {}
            # The arrays have to be max(nPaddles) for uniformity, but not all those
            # numbers will be used, so let's gather that info first
            if isinstance(newHingeDict['nPaddles'], int | float | np.int32 | np.float64):
                hingeLengthCalcs['nPaddles'] = np.array([newHingeDict['nPaddles']]*N0h)
            else:
                hingeLengthCalcs['nPaddles'] = newHingeDict['nPaddles']
            if isinstance(newHingeDict['finalTwistBool'], bool | int | float | np.int32 | np.float64):
                hingeLengthCalcs['finalTwistBool'] = np.array([newHingeDict['finalTwistBool']]*N0h)
            else:
                hingeLengthCalcs['finalTwistBool'] = newHingeDict['finalTwistBool']
            # Now for the actual lengths...
            for prop in ['gapLs', 'rps', 'Ns']:
                if (len(np.shape(newHingeDict[prop])) == 1):
                    hingeLengthCalcs[prop] = np.array([list(newHingeDict[prop])]*N0h)
                elif (len(np.shape(newHingeDict[prop])) == 2) and (np.shape(newHingeDict[prop])[0] == 1):
                    hingeLengthCalcs[prop] = np.array([list(newHingeDict[prop][0])]*N0h)
                else:
                    hingeLengthCalcs[prop] = newHingeDict[prop]  
            # Now do the appropriate summation
            hingeLength = 2*pi*np.sum(np.array([np.sum(hingeLengthCalcs['rps'][i][:hingeLengthCalcs['nPaddles'][i]] * hingeLengthCalcs['Ns'][i][:hingeLengthCalcs['nPaddles'][i]]) for i in range(N0h)])) + np.sum(np.array([np.sum(hingeLengthCalcs['gapLs'][i][:int(hingeLengthCalcs['nPaddles'][i] + int(hingeLengthCalcs['finalTwistBool'][i]))]) for i in range(N0h)]))
        
        elif (hingeType == 1):
            if ('alpha' in hingeDict.keys()):
                # Must be 1×4 or N0h×4 array or dict for random generation
                if isinstance(hingeDict['alpha'], np.ndarray):
                    newHingeDict['alpha'] = hingeDict['alpha']
                elif isinstance(hingeDict['alpha'], dict):
                    newHingeDict['alpha'] = _getRandom((N0h,4), **hingeDict['alpha'])
            else:
                newHingeDict['alpha'] = _getRandom((N0h,4), **_randomDistDefaults['alpha'])
            hingeLength = 0
        
        # Now generate segment lengths...
        if ('L0' in segmentDict.keys()):
            if isinstance(segmentDict['L0'], int | float | np.int32 | np.float64):
                newSegmentDict['L0'] = np.array([segmentDict['L0']]*N0)
            elif isinstance(segmentDict['L0'], np.ndarray):
                newSegmentDict['L0'] = segmentDict['L0']
            elif (isinstance(segmentDict['L0'], dict) and all([s in segmentDict['L0'].keys() for s in ['scale', 'dist']])):
                newSegmentDict['L0'] = _getRandom(N0, (Ltot - hingeLength)/N0, **segmentDict['L0'])
        else:
            newSegmentDict['L0'] = _getRandom(N0, (Ltot - hingeLength)/N0, **_randomDistDefaults['L0'])
        segmentLength = np.sum(newSegmentDict['L0'])
        # Ensure total length compliance with Ltot
        lengthDiff = (segmentLength + hingeLength) - Ltot
        newSegmentDict['L0'] = newSegmentDict['L0'] - (lengthDiff/N0)
    
        return cls(w0, newSegmentDict, newHingeDict, N0, hingeType = hingeType, hingeStart = hingeStart, hingeEnd = hingeEnd, arbRotStart = arbRotStart, addRotators = addRotators)

    def calcPhaseDelay(self):
        """
        Method for calculating the time for the light to propagate through the fiber.
        """
        fa = self.fibers
        t0 = np.array([0,0,0])
        for i in range(len(fa)):
            t0 = t0 + fa[i].calcPhaseDelay()
        return t0

    def __str__(self):
        self._printingBool = True
        fa, hingeInds, L0 = self.L0
        hingeTypeStr = "sets of fiber paddles"
        if (self.hingeType == 1):
            hingeTypeStr = "arbitrary rotators"
        info_array = ["Optical fiber of total length {:.4f} m, containing {:.0f} separate fiber objects.".format(L0, len(fa))]
        if isinstance(self.addRotators, int | float | np.int32 | np.float64):
            info_array = info_array + ["There are arbitrary rotators, added every {:.4f} m.".format(self.addRotators)]
        elif isinstance(self.addRotators, dict):
            info_array = info_array + ["There are arbitrary rotators added at random distances with mean {:.4f} and scale {:.4f}.".format(self.addRotators['mean'], self.addRotators['scale'])]
        info_array = info_array + ["The hinges, which are " + hingeTypeStr + ", are at the following indices of the object.fibers array.", str(hingeInds)]
        self._printingBool = False
        sep = "\n"
        return sep.join(info_array)
