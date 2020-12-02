The ``G4HepEmRun`` library documentation
---------------------------------------------

Documentation of the **run-time** functionalities stored in the ``G4HepEmRun`` library.

.. _ref-Particle-managers:

Particle managers 
..................

A separate manager object is designed for each particle type that is used to provide 
all the information and perform all the actions related to physics. These top level 
managers (:cpp:class:`G4HepEmElectronManager` for e-/e+ and :cpp:class:`G4HepEmGammaManager` 
for :math:`\gamma` particles) can be used to obtain the physics step limits and to perform 
all physics interactions in a particle transport simulation.

The two functions, through the manager provides its functionalities are the  

  - :math:`\texttt{HowFar}`  : provides the information regarding how far the particle can go (along its original direction), till its next stop due to physics interaction(s)
  - :math:`\texttt{Perform}` : performs the corresponding physics interaction(s) (including all continuous, discrete and at-rest)
  
.. note:: While each of the worker :cpp:class:`G4HepEmRunManager` objects has their 
   own instance of these top level managers (per particle type), a single instance 
   form each could also be shared by all workers since **these particle manager objects 
   do not have any state variables**. 


Code documentation
^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: G4HepEmElectronManager
   :project: The G4HepEm R&D project
   :members:
   :private-members:
   
.. doxygenstruct:: G4HepEmGammaManager
   :project: The G4HepEm R&D project
   :members:
   :private-members:

.. doxygenclass:: G4HepEmTLData
   :project: The G4HepEm R&D project
   :members:
   :private-members:
