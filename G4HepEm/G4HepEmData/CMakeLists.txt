set(G4HEPEMDATA_headers
  include/G4HepEmCuUtils.hh
  include/G4HepEmData.hh
  include/G4HepEmElectronData.hh
  include/G4HepEmElementData.hh
  include/G4HepEmGammaData.hh
  include/G4HepEmMatCutData.hh
  include/G4HepEmMaterialData.hh
  include/G4HepEmParameters.hh
  include/G4HepEmSBTableData.hh
  include/G4HepEmState.hh
)
set(G4HEPEMDATA_CXX_sources
  src/G4HepEmData.cc
  src/G4HepEmElectronData.cc
  src/G4HepEmElementData.cc
  src/G4HepEmGammaData.cc
  src/G4HepEmMatCutData.cc
  src/G4HepEmMaterialData.cc
  src/G4HepEmParameters.cc
  src/G4HepEmSBTableData.cc
)

g4hepem_add_library(g4HepEmData SOURCES ${G4HEPEMDATA_CXX_sources} HEADERS ${G4HEPEMDATA_headers})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(g4HepEmData PUBLIC $<$<BOOL:${G4HepEm_CUDA_BUILD}>:G4HepEm_CUDA_BUILD>)
  target_link_libraries(g4HepEmData PRIVATE $<$<BOOL:${G4HepEm_CUDA_BUILD}>:CUDA::cudart>)
endif()

if(BUILD_STATIC_LIBS)
  target_compile_definitions(g4HepEmData-static PUBLIC $<$<BOOL:${G4HepEm_CUDA_BUILD}>:G4HepEm_CUDA_BUILD>)
  # TODO: Determine when/if to use CUDA::cudart_static
  target_link_libraries(g4HepEmData-static PRIVATE $<$<BOOL:${G4HepEm_CUDA_BUILD}>:CUDA::cudart>)
endif()
