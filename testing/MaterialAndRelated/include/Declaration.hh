
#ifndef Declaration_HH
#define Declaration_HH


// checks G4HepEmElemData (host) by comparing to those in Geant4
bool TestElementData   ( const struct G4HepEmData* hepEmData );
bool TestMaterialData  ( const struct G4HepEmData* hepEmData );
bool TestMatCutData    ( const struct G4HepEmData* hepEmData );


#ifdef G4HepEm_CUDA_BUILD

  bool TestElementDataOnDevice  ( const struct G4HepEmData* hepEmData );

  bool TestMaterialDataOnDevice ( const struct G4HepEmData* hepEmData );

  bool TestMatCutDataOnDevice   ( const struct G4HepEmData* hepEmData );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
