.. _install_doc:

Build and Install
==================

``G4HepEm`` is one of the ongoing developments, carried out within the EM physics 
working group of the ``Geant4`` collaboration, that has been made available in order 
to facilitate and catalyse correlated R&D activities by providing and sharing 
the related expertise and specific knowledge. Therefore, ``G4HepEm`` is tightly 
connected to and depends on the ``Geant4`` simulation toolkit. 


Requirements
--------------

As mentioned above, the only requirement of ``G4HepEm`` is a recent ``Geant4`` 
version to be available on the system. The recommended version, ``Geant4-10.6.p03``
is available at the corresponding ``Geant4`` `Download <https://geant4.web.cern.ch/node/1837>`_ area. All information regarding the ``Geant4``
`Prerequisites <https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/gettingstarted.html>`_
as well as the installation instruction (`Building and Installing <https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/installguide.html>`_) 
can be found in the related sections of the `Geant4 Installation Guide <https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/index.html>`_. 



Build and install
--------------------

It is assumed in the followings, that the required version of the ``Geant4`` toolkit is installed on the system and the corresponding 
``Geant4Config.cmake`` ``CMake`` configuration file is located under the ``G4_INSTALL`` directory. Then 
building and installing ``G4HepEm`` can be done by simple:

  1. Cloning the ``G4HepEm`` repository: :: 

      $ git clone https://github.com/mnovak42/g4hepem.git
      Cloning into 'g4hepem'...
      remote: Enumerating objects: 488, done.
      remote: Counting objects: 100% (488/488), done.
      remote: Compressing objects: 100% (291/291), done.
      remote: Total 488 (delta 221), reused 412 (delta 167), pack-reused 0
      Receiving objects: 100% (488/488), 1.30 MiB | 2.63 MiB/s, done.
      Resolving deltas: 100% (221/221), done.

  2. Configuration: ::
    
      $ cd g4hepem/
      $ mkdir build
      $ cd build/
      $ cmake ../ -DGeant4_DIR=G4_INSTALL -DCMAKE_INSTALL_PREFIX=G4HepEm_INSTALL
      -- The C compiler identification is GNU 8.3.1
      -- The CXX compiler identification is GNU 8.3.1
      -- Check for working C compiler: /usr/bin/cc
      -- Check for working C compiler: /usr/bin/cc -- works
      -- Detecting C compiler ABI info
      -- Detecting C compiler ABI info - done
      -- Detecting C compile features
      -- Detecting C compile features - done
      -- Check for working CXX compiler: /usr/bin/c++
      -- Check for working CXX compiler: /usr/bin/c++ -- works
      -- Detecting CXX compiler ABI info
      -- Detecting CXX compiler ABI info - done
      -- Detecting CXX compile features
      -- Detecting CXX compile features - done
      -- Found EXPAT: /usr/lib64/libexpat.so (found suitable version "2.2.5", minimum required is "2.2.5") 
      -- Found XercesC: /usr/lib64/libxerces-c.so (found suitable version "3.2.2", minimum required is "3.2.2") 
      -- Configuring done
      -- Generating done
  
  Note, the the ``-DCMAKE_INSTALL_PREFIX=G4HepEm_INSTALL`` ``CMake`` configuration variable specifies the ``G4HepEm_INSTALL`` directory as 
  the location where ``G4HepEm`` is required to be installed. The following ``CMake`` configuration options are also available at this point:   
  
    - ``-DG4HepEm_CUDA_BUILD=ON/OFF`` : activates/deactivates(default) GPU support (see more at the :ref:`GPU Support Section <ref-GPU-support>`). 
      This requires a CUDA capable GPU device to be available with the appropriate driver and CUDA libraries to be installed.
    - ``-DBUILD_TESTS=ON/OFF`` : activates/deactivates(default) building the test applications (that are located under the ``apps/tests`` and ``apps/examples`` directories)
 
  3. Build and install ::
  
      $ make install 
      Scanning dependencies of target g4HepEmData
      [  3%] Building CXX object G4HepEm/G4HepEmData/CMakeFiles/g4HepEmData.dir/src/G4HepEmData.cc.o
      [  7%] Building CXX object G4HepEm/G4HepEmData/CMakeFiles/g4HepEmData.dir/src/G4HepEmElectronData.cc.o
      [ 10%] Building CXX object G4HepEm/G4HepEmData/CMakeFiles/g4HepEmData.dir/src/G4HepEmElementData.cc.o
      ...
      ...
      ...
      [ 92%] Building CXX object G4HepEm/CMakeFiles/g4HepEm.dir/src/G4HepEmProcess.cc.o
      [ 96%] Building CXX object G4HepEm/CMakeFiles/g4HepEm.dir/src/G4HepEmRunManager.cc.o
      [100%] Linking CXX shared library ../lib/libg4HepEm.so
      [100%] Built target g4HepEm
      Install the project...
      ...
      ...
      ...

    
  When building the test applications was required by setting the ``-DBUILD_TESTS=ON`` ``CMake`` option during the configuration above, 
  the test applications can be executed as  ::
  
      $ make test 
      Running tests...
      Test project g4hepem/build
          Start 1: TestMaterialAndRelated
      1/5 Test #1: TestMaterialAndRelated ...........   Passed    1.80 sec
          Start 2: TestEnergyLossData
      2/5 Test #2: TestEnergyLossData ...............   Passed    1.78 sec
          Start 3: TestElemSelectorData
      3/5 Test #3: TestElemSelectorData .............   Passed    1.91 sec
          Start 4: TestXSectionData
      4/5 Test #4: TestXSectionData .................   Passed    1.79 sec
          Start 5: TestEm3
      5/5 Test #5: TestEm3 ..........................   Passed    0.08 sec

      100% tests passed, 0 tests failed out of 5

      Total Test time (real) =   7.36 sec


Example 
---------

After building and installing ``G4HepEm`` under the ``G4HepEm_INSTALL`` directory, 
any of the test and/or example applications provided under the ``apps/tests`` and 
``apps/examples`` directories can be built and used. For example, building and 
executing the ``TestEm3`` (general) simplified sampling calorimeter example 
application can be done by the following configuration, build and execute steps:

  1. Configuration (note, that ``G4HepEm_INSTALL/lib/cmake/G4HepEm/`` directory contains the ``G4HepEmConfig.cmake`` ``CMake`` configuration file) ::

      $ cd g4hepem/apps/examples/TestEm3/
      $ mkdir build
      $ cd build/
      $ cmake ../ -DGeant4_DIR=G4_INSTALL -DG4HepEm_DIR=G4HepEm_INSTALL/lib/cmake/G4HepEm/
      -- The C compiler identification is GNU 8.3.1
      -- The CXX compiler identification is GNU 8.3.1
      -- Check for working C compiler: /usr/bin/cc
      ...
      ...
      ...
    
  2. Build: ::
    
      $ make  
      Scanning dependencies of target TestEm3
      [  4%] Building CXX object CMakeFiles/TestEm3.dir/TestEm3.cc.o
      [  8%] Building CXX object CMakeFiles/TestEm3.dir/src/ActionInitialization.cc.o
      [ 13%] Building CXX object CMakeFiles/TestEm3.dir/src/DetectorConstruction.cc.o
      ...
      ...
      ...
    
  3. Execute (run the application as ``./TestEm3 --help`` for more information and see the ``g4hepem/apps/examples/TestEm3/ATLASbar.mac`` example input macro file for more details): ::  
  
      $ ./TestEm3 -m ../ATLASbar.mac 

      **************************************************************
       Geant4 version Name: geant4-10-06-patch-03 [MT]   (6-November-2020)
        << in Multi-threaded mode >> 
      ...
      ...
      ...
  
  

