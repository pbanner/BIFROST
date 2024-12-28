### 12/28/24

*  Discovered and corrected an error in the calculation of the coefficients of thermal expansion, in which the core and cladding CTEs were flipped.
*  Discovered and corrected an error in the calculation of the geometrical birefringence from core noncircularity, namely replaced epsilon with 1/epsilon.
*  Added ``calcNGEff()`` method to ``FiberLength`` class for calculating the effective group index of refraction for the fiber.
*  Implemented ``m0`` and ``m1`` in the ``FiberLength`` class and propagated the changes through the other classes. This allows both the core and cladding to be doped for better matching of the refractive indices to various commercial fibers.
*  Allowed ``m0`` and ``m1`` to be negative; now a positive fraction corresponds to germanium doping and a negative one corresponds to fluorine doping. Added the necessary fluorine properties and changes to calculations.
*  Changed the optional specification of the fiber refractive index properties to the ``mProps`` dictionary; see ``FiberLength`` class documentation or examples notebook for details.
*  Added the possibility of tension during wrapping to the ``FiberLength`` class, which also required implementing Young's modulus calculation (``FiberLength.E``).
*  Cleaned up the chromatic dispersion D_CD calculation.
*  Corrected the calculation of birefringence in the case of twists.
*  Added overrides to the str methods for ``FiberLength``, ``FiberPaddleSet``, ``Rotator``, and ``Fiber`` classes; execute ``print(f)``.
*  Updated documentation for all of the above changes and updated/added to the examples notebook to reflect the changes.

### 12/18/24

*  Changed the syntax for setting individual components of the ``FiberPaddleSet`` class; I've exposed ``FiberPaddleSet.angles``, ``.rps``, ``.gapLs``, and ``.Ns`` with no error checking so that individual elements can be set with a syntax like ``fps.angles[1] = pi``. In the future I may return to this and put appropriate error-checking around it. I also updated the Examples notebook to reflect this.
*  Removed the degrees-to-radians unit conversion in the calculation of twist birefringence, and updated all relevant calls so that the user now specifies ``FiberPaddleClass`` paddle angles as well as all twist rates in radians. I also updated the Examples notebook to reflect this.
*  Added the ability to calculate the group velocity dispersion $D_{CD}$ to the ``FiberLength`` class, and added examples of its use to the Examples notebook.

### 12/10/24

*  Added ``addRotators`` option to ``Fiber`` class initialization and ``Fiber.random()`` method, which allows the user to add arbitrary rotators along the long birefringent segments as a crude way of simulating low-PMD optical fibers.

### 12/15/24

*  Added link to Overleaf notes to readme doc
