#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.8...3.19)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
project(TestEm3)

#----------------------------------------------------------------------------
# Find Geant4 package
#
find_package(Geant4 REQUIRED)

#----------------------------------------------------------------------------
# Find G4HepEm
#
find_package(G4HepEm REQUIRED)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include)
set(sources
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EmAcceptance.cc
  src/EventAction.cc
  src/G4HepEmProcess.cc
  src/HistoManager.cc
  src/PhysicsList.cc
  src/PhysicsListMessenger.cc
  src/PhysListEmStandard.cc
  src/PhysListG4Em.cc
  src/PhysListHepEm.cc
  src/PrimaryGeneratorAction.cc
  src/PrimaryGeneratorMessenger.cc
  src/RunAction.cc
  src/RunActionMessenger.cc
  src/Run.cc
  src/StepMax.cc
  src/StepMaxMessenger.cc
  src/SteppingAction.cc
  src/SteppingVerbose.cc
  src/TrackingAction.cc
)

#----------------------------------------------------------------------------
# Also consider the case of an installed G4HepEm (with tracking manager support)
# beyond the cxx code based check in the main CMake file.
#
if (G4HepEm_trackingmanager_FOUND)
  set(G4HepEm_HAS_G4VTRACKINGMANAGER TRUE)
endif()

if(G4HepEm_HAS_G4VTRACKINGMANAGER)
  set(sources ${sources}
    src/PhysListG4EmTracking.cc
    src/PhysListHepEmTracking.cc
  )
endif()

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(TestEm3 ${PROJECT_SOURCE_DIR}/TestEm3.cc ${sources})
target_compile_definitions(TestEm3 PRIVATE $<$<BOOL:${G4HepEm_HAS_G4VTRACKINGMANAGER}>:G4HepEm_HAS_G4VTRACKINGMANAGER>)
target_link_libraries(TestEm3 ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES})
