#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(TestEm3)

#----------------------------------------------------------------------------
# Find Geant4 and set G4 libraries, include directories, etc... 
#
find_package(Geant4 REQUIRED)
message (STATUS "Geant4_INCLUDE_DIRS = ${Geant4_INCLUDE_DIRS}")
message (STATUS "Geant4_LIBRARIES    = ${Geant4_LIBRARIES}")
message (STATUS "Geant4_CXX_FLAGS    = ${Geant4_CXX_FLAGS}")
message (STATUS "Geant4_BUILD_TYPE   = ${Geant4_BUILD_TYPE}")
message (STATUS "Geant4_static_FOUND = ${Geant4_static_FOUND}")

# set the same configuration that was used to build G4 
set (CMAKE_CXX_FLAGS "${Geant4_CXX_FLAGS}")
#set (CMAKE_CXX_FLAGS "-g -O0 -fno-inline -std=c++11")
set (CMAKE_BUILD_TYPE "${Geant4_BUILD_TYPE}")

message(STATUS "${CMAKE_CXX_FLAGS}")

set (BUILD_SHARED_LIBS ON)
set (BUILD_STATIC_LIBS OFF)
if (Geant4_static_FOUND)
  set (BUILD_SHARED_LIBS OFF)
  set (BUILD_STATIC_LIBS ON)
endif ()


# set the directory where the libraries will be placed 
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)


#----------------------------------------------------------------------------
# CUDA
option (G4HepEm_CUDA_BUILD "BUILD with CUDA support" OFF)
if (G4HepEm_CUDA_BUILD)
  find_package (CUDA 6.0 REQUIRED)
  if (CUDA_FOUND)
    add_definitions (-DG4HepEm_CUDA_BUILD)
    message (STATUS "Found CUDA at ${CUDA_TOOLKIT_ROOT_DIR} (${CUDA_VERSION_STRING})")
    include_directories (AFTER SYSTEM ${CUDA_INCLUDE_DIRS})
#    message (STATUS "CUDA INCLUDE DIRECTORY = ${CUDA_INCLUDE_DIRS}")
#    message (STATUS "CUDA LIBRARIES = ${CUDA_LIBRARIES}")
#    message (STATUS "CUDA CUBLAS LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}")
#    message (STATUS "CUDA cudart static LIBRARY = ${CUDA_cudart_static_LIBRARY}")
#    message (STATUS "CUDA cusolver LIBRARY = ${CUDA_cusolver_LIBRARY}")
#    message (STATUS " ")
######
     enable_language(CUDA)
#     set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} 
#       -Xcompiler -fPIC -Xcudafe 
#       --diag_suppress=boolean_controlling_expr_is_constant
#       --disable-warnings
#       -lineinfo"
#     )
  else (CUDA_FOUND)
    message(FATAL_ERROR "REQUIRED CUDA LIBRARIES WERE NOT FOUND")
  endif (CUDA_FOUND)  
endif (G4HepEm_CUDA_BUILD)


#----------------------------------------------------------------------------
# Build G4HepEm libraries
add_subdirectory (G4HEPEM)



#----------------------------------------------------------------------------
# Build the example applications

