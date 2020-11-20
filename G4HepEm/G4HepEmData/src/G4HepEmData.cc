
#include "G4HepEmData.hh"
#include <iostream>

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"

#include "G4HepEmElectronData.hh"
#include "G4HepEmSBTableData.hh"

void InitG4HepEmData (struct G4HepEmData* theHepEmData) {
  // free all previous (if any)
  FreeG4HepEmData (theHepEmData);
  //
  theHepEmData->fTheMatCutData       = nullptr;
  theHepEmData->fTheMaterialData     = nullptr;  
  theHepEmData->fTheElementData      = nullptr;
  
  theHepEmData->fTheElectronData     = nullptr;
  theHepEmData->fThePositronData     = nullptr;
  theHepEmData->fTheSBTableData      = nullptr;
  
#ifdef G4HepEm_CUDA_BUILD
  theHepEmData->fTheMatCutData_gpu   = nullptr;
  theHepEmData->fTheMaterialData_gpu = nullptr;
  theHepEmData->fTheElementData_gpu  = nullptr;

  theHepEmData->fTheElectronData_gpu = nullptr;
  theHepEmData->fThePositronData_gpu = nullptr;
#endif // G4HepEm_CUDA_BUILD



}


void FreeG4HepEmData (struct G4HepEmData* theHepEmData) {
  FreeMatCutData   ( &(theHepEmData->fTheMatCutData)   );
  FreeMaterialData ( &(theHepEmData->fTheMaterialData) );
  FreeElementData  ( &(theHepEmData->fTheElementData)  );
  
  FreeElectronData ( &(theHepEmData->fTheElectronData) );
  FreeElectronData ( &(theHepEmData->fThePositronData) );
  FreeSBTableData  ( &(theHepEmData->fTheSBTableData)  );

#ifdef G4HepEm_CUDA_BUILD
  FreeMatCutDataOnGPU      ( &(theHepEmData->fTheMatCutData_gpu)   );
  FreeMaterialDataOnGPU    ( &(theHepEmData->fTheMaterialData_gpu) );
  FreeElementDataOnGPU     ( &(theHepEmData->fTheElementData_gpu)  );

  FreeElectronDataOnDevice ( &(theHepEmData->fTheElectronData_gpu) );
  FreeElectronDataOnDevice ( &(theHepEmData->fThePositronData_gpu) );
  // 
  // no SB data on device yet...
#endif // G4HepEm_CUDA_BUILD  
 
}



