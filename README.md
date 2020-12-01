![GitHub Workflow Status](https://img.shields.io/github/workflow/status/mnovak42/g4hepem/cpu-build?label=Tests%20%28CI%29&logo=github&logoColor=white&style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/g4hepem?label=%20Building%20docs&logo=read%20the%20docs&logoColor=white&style=plastic)

<p align="center">  
  <a href="https://g4hepem.readthedocs.io/en/latest/">
    <img src="./docs/source/logo_HepEM3.png"></a> 
</p>



# The G4HepEm R&D Project

The ``G4HepEm`` R&D project was initiated by the `Electromagnetic Physics Working Group` of the ``Geant4`` collaboration as part of looking for solutions to reduce the computing performance bottleneck experienced by the `High Energy Physics` (HEP) detector simulation applications. 
 


## Description

The project (initially) targets the most performance critical part of the HEP detector simulation applications, i.e. the EM shower generation covering <img src="https://render.githubusercontent.com/render/math?math=e^{-}/e^{%2B}"> and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> particle transport. By

  - starting from the EM physics modelling part, the project **isolates and extracts** **all the data and functionalities** required for EM shower simulation in HEP detectors **in a clear, compact and well documented form**
  
  - investigates the possible computing performance benefits of replacing the **current general** particle transport simulation **stepping-loop** of ``Geant4`` **by alternatives, highly specialised for particle types** and **tailored for HEP** detector simulation applications

  - by providing a clear, compact and well documented environment for EM shower generation (already connected to ``Geant4`` applications), the project also **provides an excellent domain for further** related **R&D activities**. 
    
  - especially, the **projects facilitates** R&D activities targeting **EM shower simulation on GPU** by providing:
    
    - functionalities of making **all required physics data** **available on** the main **device memory**, together **with example kernels** for their run-time usage.
      
    - isolated, **self-contained** (i.e. `"single function"`) **implementations of both the** physics related parts of the **stepping loop** and **all required physics interactions**

All details can be found in the **[documentation](https://g4hepem.readthedocs.io/en/latest/)**. 


## Requirements

Only Geant4... 


## Quick start

How to build and install ...
How to use one of the example applications...
