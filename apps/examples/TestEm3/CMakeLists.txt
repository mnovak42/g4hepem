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
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(TestEm3 ${PROJECT_SOURCE_DIR}/TestEm3.cc ${sources} ${headers})
target_link_libraries(TestEm3 ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES})
