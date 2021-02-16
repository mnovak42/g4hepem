# Testing the Restricted Macroscopic Cross Sections for e-/e+ `ionisation` and `bremsstrahlung`.

The test constructs a *"fake"* ``Geant4`` setup using its NIST pre-defined materials to create material-cuts couples (>300). After initialising ``G4HepEm``, **material-cuts** and **primary particle kinetic energy pairs** are selected uniformly random **as test cases**. The **restricted macroscopic cross sections** for ionisation and bremsstrahlung **are evaluated both on the host and device** side based on the data structures and functionalities provided by ``G4HepEm``. The HOST and DEVICE side results are compared and success is reported only if all values are identical. The first case failed (i.e. different macroscopic cross section values on the HOST and DEVICE) is reported otherwise.

Of course, in case of ``G4HepEm``, that had been built without ``CUDA`` support, i.e. without providing the ``-DG4HepEm_CUDA_BUILD=ON`` ``CMake`` configuration option, the test evaluates the macroscopic cross section values only on the host.


## Host

The restricted macroscopic cross sections are evaluated by using the functionalities provided by the ``G4HepEmElectronManager``. These are **exactly the same** functionalities that are **used at run-time by ``G4HepEm``**.


## Device

Special **``CUDA`` kernels are provided** as part of the test **for evaluating the macroscopic cross section** values on the device. These interpolations are based on the device side representation of the corresponding data, provided by ``G4HepEm``.
