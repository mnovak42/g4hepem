
# Testing the Target Element Selector Data for e-/e+ interaction models

The test constructs a *"fake"* ``Geant4`` setup using its NIST pre-defined materials to create material-cuts couples (>300). After initialising ``G4HepEm``, **material-cuts** and **primary particle kinetic energy pairs** are selected uniformly random **as test cases**.
for the *Seltzer-Berger*(SB) and *Relativistic*(RB) models for bremsstrahlung (target atom selection is not utilised in the *Moller-Bhabha*(MB) model).  The **index of the target element**, on which the interaction takes place is **sampled both on the host and device** side based on the data structures and functionalities provided by ``G4HepEm``. A consistency check on the HOST side results is performed (e.g check if the selected element index is valid) while the HOST and DEVICE side results are compared and success is reported only if all values are consistent and identical. The first case failed (i.e. non-consistent index on the HOST or different target element indices obtained on the HOST and DEVICE) is reported otherwise.

Of course, in case of ``G4HepEm``, that had been built without ``CUDA`` support, i.e. without providing the ``-DG4HepEm_CUDA_BUILD=ON`` ``CMake`` configuration option, the test samples target element indices only on the host with the corresponding consistency check.


## Host 

The target element index for the interaction is sampled by using the functionalities defined in and provided by the ``G4HepEmElectronInteractionBremSB``. The corresponding ``SelectTargetAtomBrem`` function can be used both for the **Seltzer-Berger** and for the **Relativistic** models for **bremsstrahlung**. This is  **exactly the same** functionality and data that are **used at run-time by ``G4HepEm``**.


## Device

Special **``CUDA`` kernels are provided** as part of the test **for target atom selection** in case of the **bremsstrahlung models** on the device. These kernels utilise the device side representation of the corresponding data, provided by ``G4HepEm``.


## Note

The provided ``CUDA`` kernels have been developed for:

 - **testing** the consistency of the HOST and DEVICE side data and 
 - providing **example kernels** to show how the corresponding device side data structures can be used to obtain the interpolated values

In case of a device side simulations, these kernels **should be used only as base lines** for the developments. This is due to the fact, that the final data access pattern, that strongly determines the final form and performance of the kernels are not known in advance. In other words, how much one can profit from *coalesced memory access*, utilisation of the available *shared memory* or reduction of *bank conflicts*, etc. when developing kernels depend on the final form of utilisation and access of these data.


## Build

Having ``G4HepEm`` and ``Geant4`` installed under the ``G4HepEm_INSTALL`` and 
``G4_INSTALL`` (i.e. such that corresponding ``CMake`` configuration files are 
``G4HepEm_INSTALL/lib/cmake/G4HepEm/G4HepEmConfig.cmake`` and ``G4_INSTALL/lib/Geant4-VERSION/``), one can build the test as:

    cd ElectronTargetElementSelector
    mkdir build
    cd build
    cmake ../ -DGeant4_DIR=/G4_INSTALL/lib/Geant4-VERSION/ -DG4HepEm_DIR=/G4HepEm_INSTALL/lib/cmake/G4HepEm/

and execute as

    ./TestElemSelectorData


 
