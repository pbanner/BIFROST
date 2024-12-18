### 12/18/24

*  Changed the syntax for setting individual components of the ``FiberPaddleSet`` class; I've exposed ``FiberPaddleSet.angles``, ``.rps``, ``.gapLs``, and ``.Ns`` with no error checking so that individual elements can be set with a syntax like ``fps.angles[1] = pi``. In the future I may return to this and put appropriate error-checking around it. I also updated the Examples notebook to reflect this.
*  Removed the degrees-to-radians unit conversion in the calculation of twist birefringence, and updated all relevant calls so that the user now specifies ``FiberPaddleClass`` paddle angles as well as all twist rates in radians. I also updated the Examples notebook to reflect this.
*  Added the ability to calculate the group velocity dispersion $D_{CD}$ to the ``FiberLength`` class, and added examples of its use to the Examples notebook.

### 12/10/24

*  Added ``addRotators`` option to ``Fiber`` class initialization and ``Fiber.random()`` method, which allows the user to add arbitrary rotators along the long birefringent segments as a crude way of simulating low-PMD optical fibers.

### 12/15/24

*  Added link to Overleaf notes to readme doc
