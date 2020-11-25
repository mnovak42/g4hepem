#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(TestEm3)

#----------------------------------------------------------------------------
# Find Geant4 package
#
find_package(Geant4 REQUIRED)


#----------------------------------------------------------------------------
# Find G4HepEm
#
find_package(G4HepEm REQUIRED)
message (STATUS "====> ${G4HepEm_LIBRARIES}")


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${Geant4_INCLUDE_DIR}
                    ${G4HepEm_INCLUDE_DIR}
                    )
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/inc/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(TestEm3 ${CMAKE_SOURCE_DIR}/TestEm3.cc ${sources} ${headers})
target_link_libraries(TestEm3 ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES})

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS TestEm3 DESTINATION bin)
install(FILES ATLASbar.mac DESTINATION bin)
