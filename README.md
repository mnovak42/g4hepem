![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mnovak42/g4hepem/cpu_build.yml?branch=master&label=Tests%20%28CI%29&logo=github&logoColor=white&style=plastic)
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

``G4HepEm`` is one of the ongoing developments, carried out within the EM physics working group of the ``Geant4`` collaboration. It **has been made available in order to facilitate and catalyse correlated R&D activities by providing and sharing the related expertise and specific knowledge**. Therefore, ``G4HepEm`` is tightly connected to and depends on the ``Geant4`` simulation toolkit.

The only requirement of ``G4HepEm`` is a recent ``Geant4`` version to be available on the system. The recommended version, ``Geant4-10.7.p01`` is available at the corresponding **[Geant4 Download](https://geant4.web.cern.ch/support/download)** area. All information regarding the **[Prerequisites of Geant4](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/gettingstarted.html)** as well as the detailed **[Geant4 Installation Instructions](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/installguide.html)**  can be found in the related sections of the **[Geant4 Installation Guide](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/index.html)**. 


## Quick start

More detailed instructions can be found in the **[Build and Install](https://g4hepem.readthedocs.io/en/latest/IntroAndInstall/install.html)** section of the documentation.

It is assumed in the followings, that the required version of the ``Geant4`` toolkit is installed on the system and the corresponding ``Geant4Config.cmake`` ``CMake`` configuration file is located under the ``G4_INSTALL`` directory. Then building and installing ``G4HepEm`` (under the ``G4HepEm_INSTALL`` directory) can be done by simple:

    $ git clone https://github.com/mnovak42/g4hepem.git    
    $ cd g4hepem/
    $ mkdir build
    $ cd build/
    $ cmake ../ -DGeant4_DIR=G4_INSTALL -DCMAKE_INSTALL_PREFIX=G4HepEm_INSTALL
    $ make install

<details>
 <summary> <b>Example: build and execute an example application</b> </summary>

After building and installing G4HepEm under the `G4HepEm_INSTALL` directory, the `g4hepem/apps/examples/TestEm3` (general) simplified sampling calorimeter example application can be built and executed as:

    $ cd g4hepem/apps/examples/TestEm3/
    $ mkdir build
    $ cd build/
    $ cmake ../ -DGeant4_DIR=G4_INSTALL -DG4HepEm_DIR=G4HepEm_INSTALL/lib/cmake/G4HepEm/  
    $ make
    $ ./TestEm3 -m ../ATLASbar.mac

Execute the application as `./TestEm3 --help` for more information and see the `g4hepem/apps/examples/TestEm3/ATLASbar.mac` example input macro file for more details.

</details>
