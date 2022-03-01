
#include "G4HepEmData.hh"
#include <iostream>

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"

#include "G4HepEmElectronData.hh"
#include "G4HepEmSBTableData.hh"

#include "G4HepEmGammaData.hh"

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

  theHepEmData->fTheGammaData        = nullptr;

#ifdef G4HepEm_CUDA_BUILD
  theHepEmData->fTheMatCutData_gpu   = nullptr;
  theHepEmData->fTheMaterialData_gpu = nullptr;
  theHepEmData->fTheElementData_gpu  = nullptr;

  theHepEmData->fTheElectronData_gpu = nullptr;
  theHepEmData->fThePositronData_gpu = nullptr;

  theHepEmData->fTheSBTableData_gpu  = nullptr;

  theHepEmData->fTheGammaData_gpu    = nullptr;
#endif // G4HepEm_CUDA_BUILD
}


void FreeG4HepEmData (struct G4HepEmData* theHepEmData) {
  if(theHepEmData == nullptr) {
    return;
  }

  FreeMatCutData   ( &(theHepEmData->fTheMatCutData)   );
  FreeMaterialData ( &(theHepEmData->fTheMaterialData) );
  FreeElementData  ( &(theHepEmData->fTheElementData)  );

  FreeElectronData ( &(theHepEmData->fTheElectronData) );
  FreeElectronData ( &(theHepEmData->fThePositronData) );

  FreeSBTableData  ( &(theHepEmData->fTheSBTableData)  );

  FreeGammaData    ( &(theHepEmData->fTheGammaData)  );

#ifdef G4HepEm_CUDA_BUILD
  FreeMatCutDataOnGPU      ( &(theHepEmData->fTheMatCutData_gpu)   );
  FreeMaterialDataOnGPU    ( &(theHepEmData->fTheMaterialData_gpu) );
  FreeElementDataOnGPU     ( &(theHepEmData->fTheElementData_gpu)  );

  FreeElectronDataOnDevice ( &(theHepEmData->fTheElectronData_gpu) );
  FreeElectronDataOnDevice ( &(theHepEmData->fThePositronData_gpu) );

  FreeSBTableDataOnDevice  ( &(theHepEmData->fTheSBTableData_gpu)  );

  FreeGammaDataOnDevice    ( &(theHepEmData->fTheGammaData_gpu)  );
#endif // G4HepEm_CUDA_BUILD
}

#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

void CopyG4HepEmDataToGPU (struct G4HepEmData* onCPU) {
  // Deep copy each members represented by their pointers.

  // 1. Copy the G4HepEmMatCutData member and set the device ptr.
  CopyMatCutDataToGPU ( onCPU->fTheMatCutData,      &(onCPU->fTheMatCutData_gpu) );

  // 2. Copy the G4HepEmMaterialData member and set the device ptr.
  CopyMaterialDataToGPU ( onCPU->fTheMaterialData,  &(onCPU->fTheMaterialData_gpu) );

  // 3. Copy the G4HepEmElementData member and set the device ptr.
  CopyElementDataToGPU ( onCPU->fTheElementData,    &(onCPU->fTheElementData_gpu) );

  // 4. Copy electron data to the GPU
  CopyElectronDataToDevice( onCPU->fTheElectronData, &(onCPU->fTheElectronData_gpu));

  // 5. Copy positron data to the GPU
  CopyElectronDataToDevice( onCPU->fThePositronData, &(onCPU->fThePositronData_gpu));

  // 6. Copy SB-brem sampling tables to the GPU
  CopySBTableDataToDevice( onCPU->fTheSBTableData,   &(onCPU->fTheSBTableData_gpu));

  // 7. Copy gamma related data to the GPU
  CopyGammaDataToDevice( onCPU->fTheGammaData,     &(onCPU->fTheGammaData_gpu));

}
#endif
